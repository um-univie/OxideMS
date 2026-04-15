use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, RngExt};
use rayon::prelude::*;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    pub fn cblas_ssyrk(
        layout: i32,   // Order: RowMajor or ColMajor
        uplo: i32,     // Upper or Lower triangle of C
        trans: i32,    // Transpose flag for A
        n: i32,        // Order of matrix C (N x N)
        k: i32,        // Number of columns in A (if NoTrans)
        alpha: f32,    // Scalar multiplier for A * A^T
        a: *const f32, // Matrix A
        lda: i32,      // Leading dimension of A
        beta: f32,     // Scalar multiplier for C
        c: *mut f32,   // Matrix C (Result)
        ldc: i32,      // Leading dimension of C
    );
}

const SHADER_IMPLICIT: &str = r#"
struct Params { m: u32, m_vec4: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    var sum = 0.0;
    let num_tiles = (params.k + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_idx_a = t * 16u + local_id.x;
        if (row < params.m && k_idx_a < params.k) { tileA[local_id.y][local_id.x] = A[row * params.k + k_idx_a]; } else { tileA[local_id.y][local_id.x] = 0.0; }

        let k_idx_b = t * 16u + local_id.y;
        if (col < params.m && k_idx_b < params.k) { tileB[local_id.y][local_id.x] = A[col * params.k + k_idx_b]; } else { tileB[local_id.y][local_id.x] = 0.0; }

        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) { sum += tileA[local_id.y][i] * tileB[i][local_id.x]; }
        workgroupBarrier();
    }
    if (row < params.m && col < params.m) { C[row * params.m + col] = sum; }
}
"#;

const SHADER_EXPLICIT: &str = r#"
struct Params { m: u32, m_vec4: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f32, 9>, 64>;
var<workgroup> tileB: array<array<vec4<f32>, 16>, 8>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let base_row = group_id.y * 64u + local_id.y * 4u;
    let col_vec4 = group_id.x * 16u + local_id.x;
    var acc = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let num_tiles = (params.k + 7u) / 8u;
    let local_idx = local_id.y * 16u + local_id.x;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        for (var i = 0u; i < 2u; i = i + 1u) {
            let load_idx = local_idx * 2u + i;
            let a_row = load_idx / 8u;
            let a_col = load_idx % 8u;
            let global_a_row = group_id.y * 64u + a_row;
            let global_a_col = t * 8u + a_col;

            if (global_a_row < params.m && global_a_col < params.k) {
                tileA[a_row][a_col] = A[global_a_row * params.k + global_a_col];
            } else {
                tileA[a_row][a_col] = 0.0;
            }
        }

        if (local_idx < 128u) {
            let b_row = local_idx / 16u;
            let b_col = local_idx % 16u;
            let global_b_row = t * 8u + b_row;
            let global_b_col = group_id.x * 16u + b_col;

            if (global_b_col < params.m_vec4 && global_b_row < params.k) {
                tileB[b_row][b_col] = B[global_b_row * params.m_vec4 + global_b_col];
            } else {
                tileB[b_row][b_col] = vec4<f32>(0.0);
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < 8u; k = k + 1u) {
            let b_val = tileB[k][local_id.x];
            acc[0] += vec4<f32>(tileA[local_id.y * 4u + 0u][k]) * b_val;
            acc[1] += vec4<f32>(tileA[local_id.y * 4u + 1u][k]) * b_val;
            acc[2] += vec4<f32>(tileA[local_id.y * 4u + 2u][k]) * b_val;
            acc[3] += vec4<f32>(tileA[local_id.y * 4u + 3u][k]) * b_val;
        }
        workgroupBarrier();
    }

    if (col_vec4 < params.m_vec4) {
        if (base_row + 0u < params.m) { C[(base_row + 0u) * params.m_vec4 + col_vec4] = acc[0]; }
        if (base_row + 1u < params.m) { C[(base_row + 1u) * params.m_vec4 + col_vec4] = acc[1]; }
        if (base_row + 2u < params.m) { C[(base_row + 2u) * params.m_vec4 + col_vec4] = acc[2]; }
        if (base_row + 3u < params.m) { C[(base_row + 3u) * params.m_vec4 + col_vec4] = acc[3]; }
    }
}
"#;
const SHADER_TRANSPOSE: &str = r#"
struct Params { m: u32, m_vec4: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// 17 columns instead of 16 prevents shared memory bank conflicts!
var<workgroup> tile: array<array<f32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let x = global_id.x; // Col in A (K dimension)
    let y = global_id.y; // Row in A (M dimension)

    // 1. Coalesced Read from Global VRAM into Shared Memory
    if (x < params.k && y < params.m) {
        tile[local_id.y][local_id.x] = src[y * params.k + x];
    }
    workgroupBarrier();

    // 2. Transposed Coords
    let out_x = group_id.y * 16u + local_id.x; // Col in B (M dimension)
    let out_y = group_id.x * 16u + local_id.y; // Row in B (K dimension)

    // 3. Coalesced Write from Shared Memory to Global VRAM
    if (out_x < params.m && out_y < params.k) {
        dst[out_y * params.m + out_x] = tile[local_id.x][local_id.y];
    }
}
"#;

// --- SHADER 3: DBSCAN Pipeline ---
const SHADER_DBSCAN: &str = r#"
struct DbscanParams { m: u32, min_pts: u32, eps: f32, pad: u32 };

@group(0) @binding(0) var<storage, read> sim_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> core_points: array<u32>;
@group(0) @binding(2) var<storage, read_write> cluster_labels: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: DbscanParams;

@compute @workgroup_size(64)
fn init_labels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < params.m) {
        atomicStore(&cluster_labels[id], id);
    }
}

@compute @workgroup_size(64)
fn find_core_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= params.m) { return; }

    var count = 0u;
    let row_start = id * params.m;
    for (var j = 0u; j < params.m; j = j + 1u) {
        if (sim_matrix[row_start + j] >= params.eps) {
            count = count + 1u;
        }
    }
    if (count >= params.min_pts) {
        core_points[id] = 1u;
    } else {
        core_points[id] = 0u;
    }
}

@compute @workgroup_size(64)
fn merge_clusters(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= params.m) { return; }

    if (core_points[id] == 1u) {
        let row_start = id * params.m;
        for (var j = 0u; j < params.m; j = j + 1u) {
            if (sim_matrix[row_start + j] >= params.eps && core_points[j] == 1u) {
                atomicMin(&cluster_labels[id], atomicLoad(&cluster_labels[j]));
                atomicMin(&cluster_labels[j], atomicLoad(&cluster_labels[id]));
            }
        }
    }
}
"#;

// --- WGPU Setup ---

struct WgpuRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline_implicit: wgpu::ComputePipeline,
    pipeline_explicit: wgpu::ComputePipeline,
    bind_group_implicit: wgpu::BindGroup,
    bind_group_explicit: wgpu::BindGroup,

    pipeline_dbscan_init: wgpu::ComputePipeline,
    pipeline_dbscan_core: wgpu::ComputePipeline,
    pipeline_dbscan_merge: wgpu::ComputePipeline,
    pipeline_transpose: wgpu::ComputePipeline,
    bind_group_transpose: wgpu::BindGroup,
    bind_group_dbscan: wgpu::BindGroup,

    buffer_a: wgpu::Buffer,
    buffer_b_transposed: wgpu::Buffer,
    buffer_c: wgpu::Buffer,
    staging_buffer_c: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    // New DBSCAN Buffers
    #[allow(dead_code)]
    buffer_core_points: wgpu::Buffer,
    buffer_labels: wgpu::Buffer,
    staging_buffer_labels: wgpu::Buffer,
    uniform_dbscan: wgpu::Buffer,
}

impl WgpuRunner {
    async fn new(max_m: usize, k: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find adapter");

        let hardware_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("M1 Max Device"),
                required_features: wgpu::Features::empty(),
                required_limits: hardware_limits,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix A"),
            size: (max_m * k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_b_transposed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix B (Explicit AT)"),
            size: (k * max_m * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C (Output)"),
            size: (max_m * max_m * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C (Staging)"),
            size: (max_m * max_m * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Meta Uniforms"),
            size: 16, // m, k, pad1, pad2
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // DBSCAN Buffers (Only M elements, 4 bytes each!)
        let buffer_core_points = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBSCAN Core Points"),
            size: (max_m * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buffer_labels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBSCAN Cluster Labels"),
            size: (max_m * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer_labels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBSCAN Labels (Staging)"),
            size: (max_m * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_dbscan = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBSCAN Meta Uniforms"),
            size: 16, // m, min_pts, eps (f32), pad
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Modules ---
        let module_implicit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Implicit"),
            source: wgpu::ShaderSource::Wgsl(SHADER_IMPLICIT.into()),
        });
        let module_explicit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Explicit"),
            source: wgpu::ShaderSource::Wgsl(SHADER_EXPLICIT.into()),
        });
        let module_dbscan = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DBSCAN"),
            source: wgpu::ShaderSource::Wgsl(SHADER_DBSCAN.into()),
        });
        let module_transpose = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transpose"),
            source: wgpu::ShaderSource::Wgsl(SHADER_TRANSPOSE.into()),
        });

        let pipeline_transpose = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Transpose Pipeline"),
            layout: None,
            module: &module_transpose,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let bind_group_transpose = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transpose Bind Group"),
            layout: &pipeline_transpose.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b_transposed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipelines (SGEMM) ---
        let pipeline_implicit = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Implicit Pipeline"),
            layout: None,
            module: &module_implicit,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        let pipeline_explicit = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Explicit Pipeline"),
            layout: None,
            module: &module_explicit,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        // --- Pipelines (DBSCAN) ---
        let dbscan_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DBSCAN Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let dbscan_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DBSCAN Pipeline Layout"),
                bind_group_layouts: &[Some(&dbscan_bind_group_layout)],
                ..Default::default()
            });

        // --- Pipelines (DBSCAN) ---
        // Now using `layout: Some(&dbscan_pipeline_layout)` instead of `None`
        let pipeline_dbscan_init =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DBSCAN Init"),
                layout: Some(&dbscan_pipeline_layout),
                module: &module_dbscan,
                entry_point: Some("init_labels"),
                cache: None,
                compilation_options: Default::default(),
            });
        let pipeline_dbscan_core =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DBSCAN Core"),
                layout: Some(&dbscan_pipeline_layout),
                module: &module_dbscan,
                entry_point: Some("find_core_points"),
                cache: None,
                compilation_options: Default::default(),
            });
        let pipeline_dbscan_merge =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DBSCAN Merge"),
                layout: Some(&dbscan_pipeline_layout),
                module: &module_dbscan,
                entry_point: Some("merge_clusters"),
                cache: None,
                compilation_options: Default::default(),
            });

        // --- Bind Groups ---
        let bind_group_implicit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Implicit Bind Group"),
            layout: &pipeline_implicit.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_explicit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Explicit Bind Group"),
            layout: &pipeline_explicit.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b_transposed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Use the explicit layout we generated above instead of deriving it from the init pipeline
        let bind_group_dbscan = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DBSCAN Bind Group"),
            layout: &dbscan_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_core_points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_labels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_dbscan.as_entire_binding(),
                },
            ],
        });
        Self {
            device,
            queue,
            pipeline_explicit,
            pipeline_implicit,
            bind_group_explicit,
            bind_group_implicit,
            pipeline_dbscan_init,
            pipeline_dbscan_core,
            pipeline_dbscan_merge,
            bind_group_dbscan,
            bind_group_transpose,
            pipeline_transpose,
            buffer_a,
            buffer_b_transposed,
            buffer_c,
            staging_buffer_c,
            uniform_buffer,
            buffer_core_points,
            buffer_labels,
            staging_buffer_labels,
            uniform_dbscan,
        }
    }

    fn execute_and_wait(&self, m: u32, k: u32, use_explicit: bool) {
        assert!(m % 4 == 0, "M must be divisible by 4");
        let m_vec4 = m / 4;
        let meta = [m, m_vec4, k, 0u32];

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&meta));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            if use_explicit {
                cpass.set_pipeline(&self.pipeline_explicit);
                cpass.set_bind_group(0, &self.bind_group_explicit, &[]);
                cpass.dispatch_workgroups((m_vec4 + 15) / 16, (m + 63) / 64, 1);
            } else {
                cpass.set_pipeline(&self.pipeline_implicit);
                cpass.set_bind_group(0, &self.bind_group_implicit, &[]);
                cpass.dispatch_workgroups((m + 15) / 16, (m + 15) / 16, 1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    fn execute_and_wait_with_transfer(
        &self,
        m: u32,
        k: u32,
        use_explicit: bool,
        a_data: &[f32],
        b_data: Option<&[f32]>,
        c_out: &mut [f32],
    ) {
        assert!(m % 4 == 0, "M must be divisible by 4");
        let m_vec4 = m / 4;
        let meta = [m, m_vec4, k, 0u32];

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&meta));
        self.queue
            .write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(a_data));

        if use_explicit {
            if let Some(b) = b_data {
                self.queue
                    .write_buffer(&self.buffer_b_transposed, 0, bytemuck::cast_slice(b));
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            if use_explicit {
                cpass.set_pipeline(&self.pipeline_explicit);
                cpass.set_bind_group(0, &self.bind_group_explicit, &[]);
                cpass.dispatch_workgroups((m_vec4 + 15) / 16, (m + 63) / 64, 1);
            } else {
                cpass.set_pipeline(&self.pipeline_implicit);
                cpass.set_bind_group(0, &self.bind_group_implicit, &[]);
                cpass.dispatch_workgroups((m + 15) / 16, (m + 15) / 16, 1);
            }
        }

        let c_byte_size = (m * m * 4) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(&self.buffer_c, 0, &self.staging_buffer_c, 0, c_byte_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer_c.slice(0..c_byte_size);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            c_out[..result.len()].copy_from_slice(result);
            drop(data);
            self.staging_buffer_c.unmap();
        } else {
            panic!("Failed to map staging buffer from GPU!");
        }
    }

    fn execute_full_dbscan(
        &self,
        m: u32,
        k: u32,
        eps: f32,
        min_pts: u32,
        use_explicit: bool,
        a_data: &[f32],
        labels_out: Option<&mut [u32]>,
    ) {
        assert!(m % 4 == 0, "M must be divisible by 4");
        let m_vec4 = m / 4;

        let sgemm_meta = [m, m_vec4, k, 0u32];
        let dbscan_meta: [u32; 4] = [m, min_pts, eps.to_bits(), 0u32];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&sgemm_meta));
        self.queue
            .write_buffer(&self.uniform_dbscan, 0, bytemuck::cast_slice(&dbscan_meta));

        // Only upload A! The GPU handles B completely.
        self.queue
            .write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(a_data));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // --- NEW: PASS 0: GPU TRANSPOSE ---
        if use_explicit {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Transpose Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_transpose);
            cpass.set_bind_group(0, &self.bind_group_transpose, &[]);
            // Grid dimensions: (K / 16) x (M / 16)
            cpass.dispatch_workgroups((k + 15) / 16, (m + 15) / 16, 1);
        }

        // PASS 1: SGEMM
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SGEMM Pass"),
                timestamp_writes: None,
            });
            if use_explicit {
                cpass.set_pipeline(&self.pipeline_explicit);
                cpass.set_bind_group(0, &self.bind_group_explicit, &[]);
                cpass.dispatch_workgroups((m_vec4 + 15) / 16, (m + 63) / 64, 1);
            } else {
                cpass.set_pipeline(&self.pipeline_implicit);
                cpass.set_bind_group(0, &self.bind_group_implicit, &[]);
                cpass.dispatch_workgroups((m + 15) / 16, (m + 15) / 16, 1);
            }
        }
        let dbscan_groups = (m + 63) / 64;

        // PASS 2: DBSCAN
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DBSCAN Pass"),
                timestamp_writes: None,
            });

            // Init Labels
            cpass.set_pipeline(&self.pipeline_dbscan_init);
            cpass.set_bind_group(0, &self.bind_group_dbscan, &[]);
            cpass.dispatch_workgroups(dbscan_groups, 1, 1);

            // Find Core Points
            cpass.set_pipeline(&self.pipeline_dbscan_core);
            cpass.dispatch_workgroups(dbscan_groups, 1, 1);

            // Merge Clusters (Running a few iterations for convergence)
            cpass.set_pipeline(&self.pipeline_dbscan_merge);
            for _ in 0..3 {
                cpass.dispatch_workgroups(dbscan_groups, 1, 1);
            }
        }

        // If labels requested, copy to staging buffer (Only M * 4 bytes)
        let labels_byte_size = (m * 4) as wgpu::BufferAddress;
        if labels_out.is_some() {
            encoder.copy_buffer_to_buffer(
                &self.buffer_labels,
                0,
                &self.staging_buffer_labels,
                0,
                labels_byte_size,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        // 3. Optional Readback
        if let Some(out_slice) = labels_out {
            let buffer_slice = self.staging_buffer_labels.slice(0..labels_byte_size);
            let (sender, receiver) = std::sync::mpsc::channel();

            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();

            if let Ok(Ok(())) = receiver.recv() {
                let data = buffer_slice.get_mapped_range();
                let result: &[u32] = bytemuck::cast_slice(&data);
                out_slice[..result.len()].copy_from_slice(result);
                drop(data);
                self.staging_buffer_labels.unmap();
            }
        } else {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
    }
}

// --- Benchmark Runner ---

fn bench_sgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGEMM Benchmark (K=800)");
    group.sample_size(10);

    let k = 800;
    let m_sizes = [256, 1024, 5000, 10000, 20000, 30000];

    let dbscan_eps = 0.90; // Similarity threshold
    let dbscan_minpts = 10;

    let max_m = *m_sizes.last().unwrap();
    let wgpu_runner = pollster::block_on(WgpuRunner::new(max_m, k));

    for m in m_sizes {
        let mut rng = rand::rng();

        let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>()).collect();

        let mut c_out = vec![0.0f32; m * m];
        let mut labels_out = vec![0u32; m];

        // 1. CPU Benchmarks
        group.bench_with_input(
            BenchmarkId::new("CPU (matrixmultiply) implicit transpose", m),
            &m,
            |b_iter, &m_val| {
                b_iter.iter(|| unsafe {
                    matrixmultiply::sgemm(
                        m_val,
                        k,
                        m_val,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        a.as_ptr(),
                        1,
                        k as isize,
                        0.0,
                        c_out.as_mut_ptr(),
                        m_val as isize,
                        1,
                    );
                });
            },
        );
        // 1. CPU Benchmarks accelerate
        group.bench_with_input(
            BenchmarkId::new("CPU (cblas_sgemm) implicit transpose", m),
            &m,
            |b_iter, &m_val| {
                b_iter.iter(|| unsafe {
                    cblas_sgemm(
                        101,                // CblasRowMajor
                        111,                // transA: CblasNoTrans
                        112,                // transB: CblasTrans
                        m_val as i32,       // M: rows of A/C
                        m_val as i32,       // N: cols of B/C
                        k as i32,           // K: cols of A / rows of B
                        1.0,                // alpha
                        a.as_ptr(),         // A
                        k as i32,           // lda: row stride of A (k)
                        a.as_ptr(),         // B: using the same pointer
                        k as i32,           // ldb: row stride of the memory at B (k)
                        0.0,                // beta
                        c_out.as_mut_ptr(), // C
                        m_val as i32,       // ldc: row stride of C (m_val)
                    );
                });
            },
        );

        // 1. CPU Benchmarks accelerate
        group.bench_with_input(
            BenchmarkId::new("CPU (cblas_ssyrk) implicit transpose", m),
            &m,
            |b_iter, &m_val| {
                b_iter.iter(|| unsafe {
                    cblas_ssyrk(
                        101,                // CblasRowMajor
                        121,                // CblasUpper (only calculate the top-right triangle)
                        111,                // CblasNoTrans (A is Row-Major)
                        m_val as i32,       // N: dimension of the result (20,000)
                        k as i32,           // K: shared dimension (800)
                        1.0,                // alpha
                        a.as_ptr(),         // A
                        k as i32,           // lda: row stride (800)
                        0.0,                // beta
                        c_out.as_mut_ptr(), // C
                        m_val as i32,       // ldc: row stride (20,000)
                    );
                });
            },
        );

        //     // 2. GPU Baseline (Just SGEMM, no transfer)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU (wgpu) SGEMM no transfer (implicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_and_wait(m_val as u32, k as u32, false);
        //             });
        //         },
        //     );
        //     // 2. GPU Baseline (Just SGEMM, no transfer)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU (wgpu) SGEMM no transfer (explicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_and_wait(m_val as u32, k as u32, true);
        //             });
        //         },
        //     );
        //     // 3. GPU Data Return (The heavy 1.6 GB copy)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU (wgpu) + Data Transfer (implicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_and_wait_with_transfer(
        //                     m_val as u32,
        //                     k as u32,
        //                     false,
        //                     &a,
        //                     None,
        //                     &mut c_out,
        //                 );
        //             });
        //         },
        //     );
        //     // 3. GPU Data Return (The heavy 1.6 GB copy)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU (wgpu) + Data Transfer (explicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_and_wait_with_transfer(
        //                     m_val as u32,
        //                     k as u32,
        //                     true,
        //                     &a,
        //                     Some(&b),
        //                     &mut c_out,
        //                 );
        //             });
        //         },
        //     );

        //     // 5. End-to-End: GPU SGEMM + DBSCAN + Labels Transfer (Fast 80 KB Copy)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU End-to-End DBSCAN + Label Transfer, (implicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_full_dbscan(
        //                     m_val as u32,
        //                     k as u32,
        //                     dbscan_eps,
        //                     dbscan_minpts,
        //                     false,
        //                     &a,
        //                     Some(&mut labels_out),
        //                 );
        //             });
        //         },
        //     );

        //     // 5. End-to-End: GPU SGEMM + DBSCAN + Labels Transfer (Fast 80 KB Copy)
        //     group.bench_with_input(
        //         BenchmarkId::new("GPU End-to-End DBSCAN + Label Transfer (explicit)", m),
        //         &m,
        //         |b_iter, &m_val| {
        //             b_iter.iter(|| {
        //                 wgpu_runner.execute_full_dbscan(
        //                     m_val as u32,
        //                     k as u32,
        //                     dbscan_eps,
        //                     dbscan_minpts,
        //                     true,
        //                     &a,
        //                     Some(&mut labels_out),
        //                 );
        //             });
        //         },
        //     );
    }

    group.finish();
}

criterion_group!(benches, bench_sgemm);
criterion_main!(benches);
