use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, RngExt, rngs::ThreadRng};

/// 1. The Standard Branching Approach (For baseline comparison)
pub fn branching_cosine_sim(a_bins: &[u32], a_ints: &[f32], b_bins: &[u32], b_ints: &[f32]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    let len_a = a_bins.len();
    let len_b = b_bins.len();

    while i < len_a && j < len_b {
        let a_bin = a_bins[i];
        let b_bin = b_bins[j];

        if a_bin == b_bin {
            score += a_ints[i] * b_ints[j];
            i += 1;
            j += 1;
        } else if a_bin < b_bin {
            i += 1;
        } else {
            j += 1;
        }
    }
    score
}
pub fn fast_branching_cosine_sim(
    a_bins: &[u32],
    a_ints: &[f32],
    b_bins: &[u32],
    b_ints: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    let len_a = a_bins.len();
    let len_b = b_bins.len();

    // SAFETY: The while loop strictly bounds `i` and `j` to the lengths of the arrays.
    while i < len_a && j < len_b {
        unsafe {
            let a_bin = *a_bins.get_unchecked(i);
            let b_bin = *b_bins.get_unchecked(j);

            if a_bin == b_bin {
                // We only pay the cost of float loads and multiplication on an actual match
                score += *a_ints.get_unchecked(i) * *b_ints.get_unchecked(j);
                i += 1;
                j += 1;
            } else if a_bin < b_bin {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    score
}

/// 2. The Branchless SoA Approach
pub fn branchless_cosine_sim(
    a_bins: &[u32],
    a_ints: &[f32],
    b_bins: &[u32],
    b_ints: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    let len_a = a_bins.len();
    let len_b = b_bins.len();

    while i < len_a && j < len_b {
        let a_bin = unsafe { *a_bins.get_unchecked(i) };
        let b_bin = unsafe { *b_bins.get_unchecked(j) };

        let advance_i = (a_bin <= b_bin) as usize;
        let advance_j = (a_bin >= b_bin) as usize;
        let is_match = (a_bin == b_bin) as u32;

        let a_int = unsafe { *a_ints.get_unchecked(i) };
        let b_int = unsafe { *b_ints.get_unchecked(j) };

        score += a_int * b_int * (is_match as f32);

        i += advance_i;
        j += advance_j;
    }
    score
}

/// Helper to generate realistic sorted & deduplicated sparse arrays
fn generate_sorted_soa(size: usize, max_bin: u32) -> (Vec<u32>, Vec<f32>) {
    let mut rng = ThreadRng::default();

    // Generate random pairs
    let mut pairs: Vec<(u32, f32)> = (0..size)
        .map(|_| (rng.random_range(0..max_bin), rng.random::<f32>()))
        .collect();

    // Sort by mz_bin
    pairs.sort_unstable_by_key(|k| k.0);

    // Deduplicate exact bin collisions (simulating real workflow)
    pairs.dedup_by_key(|k| k.0);

    // Unzip directly into SoA arrays
    pairs.into_iter().unzip()
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Cosine Similarity");

    // Test typical Top-N sizes: 50 (strict), 150 (windowed), 500 (raw)
    let sizes = [50, 150, 500];

    // Assuming 20ppm over 200-1500 mz yields roughly 30,000 bins
    let max_bin = 30_000;

    for size in sizes {
        // Generate static arrays for the benchmark loop
        let (a_bins, a_ints) = generate_sorted_soa(size, max_bin);
        let (b_bins, b_ints) = generate_sorted_soa(size, max_bin);

        // Benchmark the Branching baseline
        group.bench_with_input(BenchmarkId::new("Fast Branching", size), &size, |b, _| {
            b.iter(|| {
                branching_cosine_sim(
                    black_box(&a_bins),
                    black_box(&a_ints),
                    black_box(&b_bins),
                    black_box(&b_ints),
                )
            });
        });
        // Benchmark the Branching baseline
        group.bench_with_input(BenchmarkId::new("Branching", size), &size, |b, _| {
            b.iter(|| {
                branching_cosine_sim(
                    black_box(&a_bins),
                    black_box(&a_ints),
                    black_box(&b_bins),
                    black_box(&b_ints),
                )
            });
        });

        // Benchmark the Branchless SoA
        group.bench_with_input(BenchmarkId::new("Branchless", size), &size, |b, _| {
            b.iter(|| {
                branchless_cosine_sim(
                    black_box(&a_bins),
                    black_box(&a_ints),
                    black_box(&b_bins),
                    black_box(&b_ints),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);
