//! Compares the two ways `BucketIndex` can train its coarse quantizer --
//! k-means++ plus 10 Lloyd iterations, and e-means -- on synthetic
//! spectrum-like vectors: build time, how tightly vectors sit on their
//! centroid, and how many of the true top-k neighbours a block of queries
//! recovers. A brute-force matmul of the same query block against the whole
//! window is timed alongside, since that is what the index has to beat.
//!
//! `cargo run --release --example ivf_training_bench -- [n_vectors ...]`

use faer::linalg::matmul::matmul;
use oxidems_lib::clustering::k_cluster_ivf::{
    BucketIndex, IvfTraining, WINDOWED_EDGES_BLOCK_SIZE, windowed_mass_edges_with,
};
use oxidems_lib::workflow::MassTolerance;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const DIM: usize = 800;
const QUERIES: usize = 4096;
const TOP_K: usize = 64;
const MIN_SIMILARITY: f32 = 0.7;

/// `n` unit vectors drawn from `n / 20` prototypes of ~40 non-zero bins,
/// each member a prototype with its intensities jittered and a few noise
/// peaks added. Prototype popularity is skewed, as replicate counts are.
fn synthetic(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_proto = (n / 20).max(1);
    let protos: Vec<Vec<(usize, f32)>> = (0..n_proto)
        .map(|_| {
            (0..40)
                .map(|_| (rng.random_range(0..DIM), rng.random_range(0.05f32..1.0).powi(3)))
                .collect()
        })
        .collect();
    let mut data = vec![0.0f32; n * DIM];
    for row in data.chunks_exact_mut(DIM) {
        let u: f32 = rng.random_range(0.0f32..1.0);
        let proto = &protos[((u * u) * n_proto as f32) as usize % n_proto];
        for &(bin, intensity) in proto {
            row[bin] += intensity * rng.random_range(0.6f32..1.4);
        }
        for _ in 0..6 {
            row[rng.random_range(0..DIM)] += rng.random_range(0.0f32..0.15);
        }
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x /= norm);
    }
    data
}

/// Exact top-`TOP_K` neighbours at or above the threshold for the first
/// `QUERIES` rows. Returns the edge set and the matmul's own wall time.
fn brute_force(data: &[f32], n: usize) -> (HashSet<(usize, usize)>, f64) {
    let q = QUERIES.min(n);
    let mut scores = vec![0.0f32; q * n];
    let started = Instant::now();
    matmul(
        faer::MatMut::from_row_major_slice_mut(&mut scores, q, n),
        faer::Accum::Replace,
        faer::MatRef::from_row_major_slice(&data[..q * DIM], q, DIM),
        faer::MatRef::from_row_major_slice(data, n, DIM).transpose(),
        1.0,
        faer::Par::rayon(0),
    );
    let elapsed = started.elapsed().as_secs_f64();
    let edges = scores
        .par_chunks_exact(n)
        .enumerate()
        .flat_map_iter(|(i, row)| {
            let mut hits: Vec<(usize, f32)> = row
                .iter()
                .copied()
                .enumerate()
                .filter(|&(j, s)| j != i && s >= MIN_SIMILARITY)
                .collect();
            hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            hits.truncate(TOP_K);
            hits.into_iter().map(move |(j, _)| (i, j))
        })
        .collect();
    (edges, elapsed)
}

fn report(name: &str, data: &[f32], n: usize, training: IvfTraining, truth: &HashSet<(usize, usize)>) {
    let started = Instant::now();
    let index = BucketIndex::build_with(data, DIM, faer::Par::rayon(0), training);
    let build = started.elapsed().as_secs_f64();

    let mut assigned = vec![0usize; n];
    for list in 0..index.n_list {
        for &id in &index.flat_indices[index.list_pointers[list]..index.list_pointers[list + 1]] {
            assigned[id] = list;
        }
    }
    let fit: f64 = assigned
        .par_iter()
        .enumerate()
        .map(|(i, &c)| {
            let v = &data[i * DIM..(i + 1) * DIM];
            let centroid = &index.centroids[c * DIM..(c + 1) * DIM];
            v.iter().zip(centroid).map(|(a, b)| a * b).sum::<f32>() as f64
        })
        .sum::<f64>()
        / n as f64;
    let sizes: Vec<usize> = (0..index.n_list)
        .map(|c| index.list_pointers[c + 1] - index.list_pointers[c])
        .collect();
    let empty = sizes.iter().filter(|&&s| s == 0).count();
    let largest = sizes.iter().max().copied().unwrap_or(0);

    let started = Instant::now();
    let edges =
        index.extract_edges_for_queries(0, QUERIES.min(n), TOP_K, 1.0 - MIN_SIMILARITY, faer::Par::rayon(0));
    let search = started.elapsed().as_secs_f64();
    let found = edges.iter().filter(|e| truth.contains(&(e.source_id, e.target_id))).count();

    println!(
        "  {name:<14} build {build:>7.2}s  search {search:>5.2}s  mean cos {fit:.4}  recall {:.4}  lists {} (empty {empty}, largest {largest})",
        found as f64 / truth.len().max(1) as f64,
        index.n_list,
    );
}

/// The path the engine takes for one dense neighbourhood: every row within
/// mass tolerance of every other, so each block's window runs to the end.
/// `threshold` decides whether blocks go through the IVF index or straight
/// to an exact matmul.
fn end_to_end(name: &str, data: &[f32], n: usize, threshold: usize, training: IvfTraining) {
    let masses: Vec<f64> = (0..n).map(|i| 500.0 + i as f64 * 1e-9).collect();
    let started = Instant::now();
    let edges = windowed_mass_edges_with(
        data,
        DIM,
        &masses,
        MassTolerance::Da(0.01),
        1.0 - MIN_SIMILARITY,
        WINDOWED_EDGES_BLOCK_SIZE,
        threshold,
        training,
    );
    println!("  {name:<22} {:>8.2}s  {} edges", started.elapsed().as_secs_f64(), edges.len());
}

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let sizes = if sizes.is_empty() { vec![40_000, 200_000] } else { sizes };
    for n in sizes {
        let data = synthetic(n, 7);
        let (truth, brute) = brute_force(&data, n);
        println!(
            "n = {n}, dim = {DIM}: {} true edges from {} queries; brute-force block matmul {brute:.2}s",
            truth.len(),
            QUERIES.min(n)
        );
        report("lloyd x10", &data, n, IvfTraining::Lloyd, &truth);
        for epochs in [1.0, 2.0, 3.0, 5.0] {
            report(&format!("e-means T={epochs}"), &data, n, IvfTraining::EMeans { epochs }, &truth);
        }
        if n <= 60_000 {
            println!("  whole neighbourhood through windowed_mass_edges (IVF above 30000 rows):");
            end_to_end("exact matmul", &data, n, usize::MAX, IvfTraining::Lloyd);
            end_to_end("ivf, lloyd", &data, n, 30_000, IvfTraining::Lloyd);
            end_to_end("ivf, e-means T=2", &data, n, 30_000, IvfTraining::EMeans { epochs: 2.0 });
        }
    }
}
