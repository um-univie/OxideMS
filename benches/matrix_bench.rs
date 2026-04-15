use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use matrixmultiply::sgemm;
use rand::{Rng, RngExt};

/// Generates both Dense and Sparse representations of the same randomized matrix
fn generate_data(rows: usize, cols: usize, density: f64) -> (Vec<f32>, Vec<Vec<(u16, f32)>>) {
    let mut rng = rand::rng();
    let mut dense = vec![0.0; rows * cols];
    let mut sparse = vec![Vec::new(); rows];

    for i in 0..rows {
        for j in 0..cols {
            if rng.random::<f64>() < density {
                let val: f32 = rng.random_range(0.1..1.0);
                // Populate Dense
                dense[i * cols + j] = val;
                // Populate Sparse
                sparse[i].push((j as u16, val));
            }
        }
    }
    (dense, sparse)
}

/// Dense implementation utilizing SGEMM with stride tricks for A * A^T
fn run_dense_sgemm(rows: usize, cols: usize, flat_data: &[f32], out: &mut [f32]) {
    unsafe {
        sgemm(
            rows,
            cols,
            rows,
            1.0,
            flat_data.as_ptr(),
            cols as isize,
            1, // Matrix A
            flat_data.as_ptr(),
            1,
            cols as isize, // Matrix A treated as transposed
            0.0,
            out.as_mut_ptr(),
            rows as isize,
            1, // Matrix C (Output)
        );
    }
}

/// The core two-pointer sparse dot product
#[inline(always)]
fn sparse_dot(a: &[(u16, f32)], b: &[(u16, f32)]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            score += a[i].1 * b[j].1;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    score
}

/// Sparse implementation computing the full matrix to keep the comparison 1:1 with SGEMM
fn run_sparse_full(rows: usize, sparse_data: &[Vec<(u16, f32)>], out: &mut [f32]) {
    for i in 0..rows {
        for j in 0..rows {
            out[i * rows + j] = sparse_dot(&sparse_data[i], &sparse_data[j]);
        }
    }
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let cols = 800;
    let row_sizes = [500, 1000, 2000];
    let densities = [0.01, 0.05, 0.10]; // 1%, 5% (your 50/800 spec), and 10%

    let mut group = c.benchmark_group("Matrix_Symmetry_A_x_At");

    for &rows in &row_sizes {
        for &density in &densities {
            let (dense_data, sparse_data) = generate_data(rows, cols, density);
            let mut out = vec![0.0; rows * rows];

            // Benchmark Dense SGEMM
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Dense_SGEMM_{}x{}", rows, cols),
                    format!("Density_{:.2}", density),
                ),
                &rows,
                |b, &_| b.iter(|| run_dense_sgemm(rows, cols, &dense_data, black_box(&mut out))),
            );

            // Benchmark Sparse
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Sparse_Custom_{}x{}", rows, cols),
                    format!("Density_{:.2}", density),
                ),
                &rows,
                |b, &_| b.iter(|| run_sparse_full(rows, &sparse_data, black_box(&mut out))),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_matrix_multiplication);
criterion_main!(benches);
