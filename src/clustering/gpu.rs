/// GPU-accelerated upper-triangular similarity matrix (SYRK) via wgpu.
///
/// Output is a packed upper-triangular array of size n*(n+1)/2.
/// Element at row r, col c (r ≤ c): index = r*(2*n - r - 1)/2 + c.
use wgpu;

/// Triangular dispatch shader.
///
/// Instead of dispatching a G×G grid and discarding ~50% of workgroups,
/// we dispatch only G*(G+1)/2 workgroups (one per upper-triangle tile block)
/// and decode the flat workgroup index back to (group_row, group_col) using
/// the anti-diagonal enumeration:
///   flat → (0,0), (0,1),(1,1), (0,2),(1,2),(2,2), …
///
/// Formula: d = floor((√(8·flat + 1) − 1) / 2)
///          group_col = d
///          group_row = flat − d*(d+1)/2
///
/// Barrier uniformity is preserved because every thread in a workgroup
/// participates in all tile loads regardless of whether it will write output.
const SHADER_UPPER_SYRK: &str = r#"
struct Params { m: u32, k: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    // --- Decode flat upper-triangle workgroup index ---
    // Anti-diagonal order: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),...
    let flat = wid.x;
    var d = u32((sqrt(f32(8u * flat + 1u)) - 1.0) * 0.5);
    // Correct rounding errors from f32 sqrt
    d += u32((d + 1u) * (d + 2u) / 2u <= flat);
    let group_col = d;
    let group_row = flat - d * (d + 1u) / 2u;

    let row = group_row * 16u + lid.y;
    let col = group_col * 16u + lid.x;

    var sum = 0.0f;
    let num_tiles = (params.k + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t++) {
        let ka = t * 16u + lid.x;
        tileA[lid.y][lid.x] = select(0.0, A[row * params.k + ka], row < params.m && ka < params.k);
        let kb = t * 16u + lid.y;
        tileB[lid.y][lid.x] = select(0.0, A[col * params.k + kb], col < params.m && kb < params.k);
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) { sum += tileA[lid.y][i] * tileB[i][lid.x]; }
        workgroupBarrier();
    }

    // Write result.
    // Off-diagonal blocks (group_row < group_col): all threads are upper-triangle.
    // Diagonal blocks (group_row == group_col): only threads with row <= col.
    let in_bounds = row < params.m && col < params.m;
    let is_upper  = (group_row < group_col) || (row <= col);
    if (in_bounds && is_upper) {
        let packed_idx = row * (2u * params.m - row - 1u) / 2u + col;
        C[packed_idx] = sum;
    }
}
"#;

pub struct GpuSimilarity {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffer_a: wgpu::Buffer,
    buffer_c: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    pub max_m: usize,
    k: usize,
}

impl GpuSimilarity {
    /// Try to initialise the GPU context.  Returns `None` when no suitable
    /// hardware GPU is available (soft-rasterizer / CPU adapters are skipped).
    ///
    /// `max_m` — maximum number of spectra per batch this instance will handle.
    /// `k`     — feature-vector (hash) length.
    pub fn try_new(max_m: usize, k: usize) -> Option<Self> {
        pollster::block_on(Self::init_async(max_m, k))
    }

    async fn init_async(max_m: usize, k: usize) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;

        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return None;
        }

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GpuSimilarity"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()?;

        let packed_max = max_m * (max_m + 1) / 2;

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_spectral"),
            size: (max_m * k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sim_packed"),
            size: (packed_max * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_staging"),
            size: (packed_max * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upper_syrk_triag"),
            source: wgpu::ShaderSource::Wgsl(SHADER_UPPER_SYRK.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("upper_syrk_triag"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upper_syrk_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffer_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
            ],
        });

        Some(Self {
            device, queue, pipeline, bind_group,
            buffer_a, buffer_c, staging_buffer, uniform_buffer,
            max_m, k,
        })
    }

    /// Compute the packed upper-triangular similarity matrix for `n` spectra.
    ///
    /// `spectral_data` — flat row-major [n × k] matrix of L2-normalised hashes.
    ///
    /// Returns a `Vec<f32>` of length `n*(n+1)/2`.  Element (r, c) with r ≤ c
    /// is at index `r*(2*n - r - 1)/2 + c`.
    pub fn compute(&self, spectral_data: &[f32], n: usize) -> Vec<f32> {
        debug_assert!(n <= self.max_m, "n={n} exceeds GpuSimilarity::max_m={}", self.max_m);
        debug_assert_eq!(spectral_data.len(), n * self.k);

        let packed_size = n * (n + 1) / 2;
        let uniforms = [n as u32, self.k as u32, 0u32, 0u32];

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniforms));
        self.queue.write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(spectral_data));

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("upper_syrk_triag"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            // Triangular dispatch: G*(G+1)/2 workgroups where G = ceil(n/16).
            let g = (n as u32 + 15) / 16;
            cpass.dispatch_workgroups(g * (g + 1) / 2, 1, 1);
        }

        let copy_bytes = (packed_size * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.buffer_c, 0, &self.staging_buffer, 0, copy_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(0..copy_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let mut result = vec![0.0f32; packed_size];
        if let Ok(Ok(())) = rx.recv() {
            let mapped = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&mapped));
            drop(mapped);
            self.staging_buffer.unmap();
        }
        result
    }
}
