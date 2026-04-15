CONST CORE_POINTS = "
@group(0) @binding(0) var<storage, read> sim_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> core_points: array<u32>;
// ... uniforms ...

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_id = global_id.x;
    var neighbor_count = 0u;

    // Read the row for this point directly from the similarity matrix
    let row_start = point_id * 20000u;
    for (var j = 0u; j < 20000u; j++) {
        // Assuming higher is more similar (e.g., Cosine Similarity)
        if (sim_matrix[row_start + j] >= epsilon) {
            neighbor_count++;
        }
    }

    if (neighbor_count >= min_pts) {
        core_points[point_id] = 1u;
    } else {
        core_points[point_id] = 0u;
    }
}
";


CONST UNION_FIND = "@group(0) @binding(0) var<storage, read> sim_matrix: array<f32>;
@group(0) @binding(1) var<storage, read> core_points: array<u32>;
@group(0) @binding(2) var<storage, read_write> cluster_labels: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_id = global_id.x;

    // Only core points propagate their labels
    if (core_points[point_id] == 1u) {
        let row_start = point_id * 20000u;

        for (var j = 0u; j < 20000u; j++) {
            if (sim_matrix[row_start + j] >= epsilon && core_points[j] == 1u) {
                // Merge labels
                atomicMin(&cluster_labels[point_id], atomicLoad(&cluster_labels[j]));
                atomicMin(&cluster_labels[j], atomicLoad(&cluster_labels[point_id]));
            }
        }
    }
}";
