use crate::clustering::cluster::{ClusterLabel, fast_dot_product};
use crate::workflow::MassTolerance;
use faer::linalg::matmul;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// A helper struct for our Max-Heap

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub score: f32,
}

// 1. Tell Rust we are treating this struct as strictly equatable.
impl Eq for Hit {}

// 2. Implement the Partial Comparison
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // REVERSED COMPARISON: `other` compared to `self`.
        // This tricks the BinaryHeap into acting as a Min-Heap.
        other.score.partial_cmp(&self.score)
    }
}

// 3. Implement the Total Ordering required by BinaryHeap
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Safe fallback: If a cosine score somehow results in NaN (e.g., zero-vector),
        // we default to Ordering::Equal so the thread doesn't panic and crash.
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<'a> BucketIndex<'a> {
    /// Initializes centroids using the K-Means++ algorithm for better convergence.
    fn kmeans_plusplus_init(
        bucket_data: &[f32],
        dim: usize,
        n_list: usize,
        n_vectors: usize,
    ) -> Vec<f32> {
        let mut centroids = vec![0.0; n_list * dim];
        // Deterministic seed derived from the bucket's shape and contents so the ANN
        // (k-means++) path is reproducible run-to-run, as claimed in the manuscript.
        let seed = {
            let mut s = 0x9E3779B97F4A7C15u64
                ^ (n_vectors as u64).wrapping_mul(0x2545F491_4F6CDD1D)
                ^ (n_list as u64).wrapping_mul(0x9E3779B9);
            for &v in bucket_data.iter().take(8) {
                s = s.rotate_left(5) ^ (v.to_bits() as u64);
            }
            s
        };
        let mut rng = StdRng::seed_from_u64(seed);

        // Pick the very first centroid uniformly at random
        let first_idx = rng.random_range(0..n_vectors);
        let first_offset = first_idx * dim;
        centroids[..dim].copy_from_slice(&bucket_data[first_offset..first_offset + dim]);

        // Track the shortest squared distance from each point to its nearest centroid.
        // Initialize with f32::MAX so the first pass always overwrites it.
        let mut min_sq_distances = vec![f32::MAX; n_vectors];

        for k in 1..n_list {
            // The centroid we just added in the previous loop iteration
            let latest_centroid = &centroids[(k - 1) * dim..k * dim];

            // Parallel distance update
            // We iterate over the tracking array in parallel, mapping each index
            // back to its corresponding vector in the flat `bucket_data` slice.
            min_sq_distances
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, min_dist)| {
                    let vec_offset = i * dim;
                    let vector = &bucket_data[vec_offset..vec_offset + dim];

                    // Calculate squared L2 distance to the *newest* centroid only
                    let mut sq_dist = 0.0;
                    for d in 0..dim {
                        let diff = vector[d] - latest_centroid[d];
                        sq_dist += diff * diff;
                    }

                    // If this new centroid is closer than any previous one, update the record
                    if sq_dist < *min_dist {
                        *min_dist = sq_dist;
                    }
                });

            // Sum the distances to establish the total probability pool, and the
            // weighted random selection below, are both an O(n_vectors) scan --
            // with n_list picks (up to 65536 for large buckets) that's
            // O(n_vectors * n_list) sequential work if done as one big linear
            // pass. Instead: compute per-chunk sums in parallel, find which
            // chunk holds the random target (a scan over only `num_chunks`
            // values, cheap), then do the same cumulative-weight walk as
            // before but confined to that one chunk -- same selection rule,
            // same result, just without scanning the other chunks.
            let num_chunks = rayon::current_num_threads().max(1);
            let chunk_size = n_vectors.div_ceil(num_chunks).max(1);
            let chunk_sums: Vec<f32> = min_sq_distances
                .par_chunks(chunk_size)
                .map(|chunk| chunk.iter().sum())
                .collect();
            let total_weight: f32 = chunk_sums.iter().sum();

            // Weighted random selection (Roulette Wheel Selection)
            // We pick a random float between 0 and the total weight.
            let target = rng.random_range(0.0..1.0) * total_weight;

            let mut cumulative_weight = 0.0;
            let mut target_chunk = num_chunks - 1;
            for (ci, &chunk_sum) in chunk_sums.iter().enumerate() {
                if cumulative_weight + chunk_sum >= target {
                    target_chunk = ci;
                    break;
                }
                cumulative_weight += chunk_sum;
            }

            let chunk_start = target_chunk * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n_vectors);
            let mut next_centroid_idx = n_vectors - 1; // Fallback to the last item just in case

            // Walk through this chunk's weights until we hit our random target
            for (local_i, &weight) in min_sq_distances[chunk_start..chunk_end].iter().enumerate() {
                cumulative_weight += weight;
                if cumulative_weight >= target {
                    next_centroid_idx = chunk_start + local_i;
                    break;
                }
            }

            // Copy the chosen vector into our centroids array
            let next_offset = next_centroid_idx * dim;
            let dest_offset = k * dim;
            centroids[dest_offset..dest_offset + dim]
                .copy_from_slice(&bucket_data[next_offset..next_offset + dim]);
        }

        centroids
    }
    pub fn build(bucket_data: &'a [f32], dim: usize, max_k: usize) -> Self {
        let n_vectors = bucket_data.len() / dim;

        let n_list = if n_vectors < 100 {
            1
        } else if n_vectors < 1_000_000 {
            2_usize.pow((n_vectors as f64 / 39.0).log2().floor() as u32)
        } else {
            65536
        };

        let n_probe = if n_list == 1 {
            1
        } else {
            std::cmp::min((n_list as f64 / 8.0).ceil() as usize, 32)
        };

        // Initialize centroids: copy the first `n_list` vectors
        let mut centroids = Self::kmeans_plusplus_init(bucket_data, dim, n_list, n_vectors);

        let mut assignments = vec![0; n_vectors];
        let mut scores_buffer = vec![0.0; n_vectors * n_list];
        let rows_per_chunk = 64;

        // Per-chunk scratch buffers for the centroid recompute step below,
        // allocated once and reused (zeroed via `fill`, not reallocated)
        // across all 10 iterations. These used to be allocated fresh inside
        // the loop (once per chunk per iteration -- up to 52MB each for a
        // large bucket) via `.map(|_| vec![...]).reduce(...)`; for the
        // largest partitions that's hundreds of multi-ten-MB allocations
        // over the course of `build()`, pure overhead next to the actual
        // arithmetic.
        let num_chunks = rayon::current_num_threads().max(1);
        let chunk_size = n_vectors.div_ceil(num_chunks).max(1);
        let mut local_centroid_buffers: Vec<Vec<f32>> =
            (0..num_chunks).map(|_| vec![0.0f32; n_list * dim]).collect();
        let mut local_count_buffers: Vec<Vec<f32>> =
            (0..num_chunks).map(|_| vec![0.0f32; n_list]).collect();
        let mut counts = vec![0.0f32; n_list];

        for _iter in 0..10 {

// 1. Calculate the number of rows in bucket_data (m)
let m = bucket_data.len() / dim;

// 2. Create safe row-major matrix views
let a_mat = faer::MatRef::from_row_major_slice(&bucket_data, m, dim);
let b_mat = faer::MatRef::from_row_major_slice(&centroids, n_list, dim);
let c_mat = faer::MatMut::from_row_major_slice_mut(&mut scores_buffer, m, n_list);

// 3. Perform the matrix multiplication (Scores = BucketData * Centroids^T)
faer::linalg::matmul::matmul(
    c_mat,
    faer::Accum::Replace,  // Overwrite C entirely (equivalent to beta = 0.0)
    a_mat,                 // A (lhs: bucket data)
    b_mat.transpose(),     // B^T (rhs: centroids, transposed)
    1.0,                   // alpha = 1.0
    faer::Par::rayon(0),   // Let faer handle chunking and threading via Rayon
);
            // 2. Argmax: Find best centroid for each vector.
            // Each row is independent (reads its own slice of scores_buffer,
            // writes only its own assignments[i]), so this parallelizes with
            // no shared mutable state -- a plain sequential loop here left
            // every large partition's assignment step running on a single
            // thread while the rest of the pool sat idle.
            assignments
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, assignment)| {
                    let row_start = i * n_list;
                    let row_scores = &scores_buffer[row_start..row_start + n_list];

                    *assignment = row_scores
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                        .unwrap()
                        .0;
                });
            // 3. Recalculate Centroids.
            // Unlike argmax, this is a scatter-add: many vectors can share a
            // target centroid, so naively parallelizing per-vector would race
            // on `centroids[cent_offset..]`. Instead each chunk of vectors
            // accumulates into its own thread-local (centroid_sums, counts)
            // buffer, and those are summed together at the end -- the only
            // cross-thread step is the final reduce, not every addition.
            local_centroid_buffers
                .par_iter_mut()
                .zip(local_count_buffers.par_iter_mut())
                .enumerate()
                .for_each(|(chunk_idx, (local_centroids, local_counts))| {
                    local_centroids.fill(0.0);
                    local_counts.fill(0.0);
                    let start = (chunk_idx * chunk_size).min(n_vectors);
                    let end = (start + chunk_size).min(n_vectors);
                    for i in start..end {
                        let c_id = assignments[i];
                        let vec_offset = i * dim;
                        let cent_offset = c_id * dim;
                        for d in 0..dim {
                            local_centroids[cent_offset + d] += bucket_data[vec_offset + d];
                        }
                        local_counts[c_id] += 1.0;
                    }
                });
            // Combine the per-chunk partial sums into `centroids`/`counts`.
            // Parallelized over centroid index (not over chunks), so total
            // wall time is ~O(n_list * dim) same as the old tree-reduce --
            // just reading from the persistent per-chunk buffers above and
            // writing straight into `centroids`/`counts` in place, no
            // allocation.
            centroids
                .par_chunks_mut(dim)
                .zip(counts.par_iter_mut())
                .enumerate()
                .for_each(|(c_id, (centroid_slice, count))| {
                    for d in 0..dim {
                        let pos = c_id * dim + d;
                        centroid_slice[d] = local_centroid_buffers.iter().map(|buf| buf[pos]).sum();
                    }
                    *count = local_count_buffers.iter().map(|buf| buf[c_id]).sum();
                });
            // 4. L2 Normalize Centroids (Spherical K-Means)
            for c in 0..n_list {
                let cent_offset = c * dim;
                let centroid_slice = &mut centroids[cent_offset..cent_offset + dim];

                let norm: f32 = centroid_slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    centroid_slice.iter_mut().for_each(|x| *x /= norm);
                }
            }
        }
        let mut radii = vec![1.0; n_list]; // Start at max similarity
        for i in 0..n_vectors {
            let c_id = assignments[i];
            let score = fast_dot_product(
                &bucket_data[i * dim..(i + 1) * dim],
                &centroids[c_id * dim..(c_id + 1) * dim],
            );
            if score < radii[c_id] {
                radii[c_id] = score;
            }
        }
        // 2. CONSTRUCT FLAT INVERTED LIST (CSR Layout)
        // Pass 1: Count sizes for each bucket
        let mut bucket_counts = vec![0; n_list];
        for &c_id in &assignments {
            bucket_counts[c_id] += 1;
        }

        // Pass 2: Create list_pointers using prefix sums
        let mut list_pointers = Vec::with_capacity(n_list + 1);
        let mut current_offset = 0;
        for &count in &bucket_counts {
            list_pointers.push(current_offset);
            current_offset += count;
        }
        list_pointers.push(current_offset); // Final boundary

        // Pass 3: Fill flat_indices
        let mut flat_indices = vec![0; n_vectors];
        let mut write_offsets = list_pointers.clone(); // Use a copy to track current write position
        for (vec_id, &c_id) in assignments.iter().enumerate() {
            let target_pos = write_offsets[c_id];
            flat_indices[target_pos] = vec_id;
            write_offsets[c_id] += 1;
        }
        Self {
            dim,
            n_list,
            n_probe,
            n_vectors,
            centroids,
            bucket_data,
            centroid_radii: radii,
            flat_indices,

            list_pointers, // Pointers into flat_indices
        }
    }
    pub fn extract_edges(&self, k: usize, epsilon: f32) -> Vec<Edge> {
        let min_similarity = 1.0 - epsilon;
        let total_routing_elements = self.n_vectors * self.n_list;

        // 1. AVOID ZEROING MEMORY: sgemm (beta = 0.0) overwrites everything.
        let mut routing_scores = Vec::with_capacity(total_routing_elements);


            // Safety: We immediately pass this to SGEMM which initializes all elements.
        unsafe {
            routing_scores.set_len(total_routing_elements);
        }

// 1. Create safe row-major matrix views
let a_mat = faer::MatRef::from_row_major_slice(&self.bucket_data, self.n_vectors, self.dim);
let b_mat = faer::MatRef::from_row_major_slice(&self.centroids, self.n_list, self.dim);
let c_mat = faer::MatMut::from_row_major_slice_mut(&mut routing_scores, self.n_vectors, self.n_list);

// 2. Perform the matrix multiplication (Scores = BucketData * Centroids^T)
faer::linalg::matmul::matmul(
    c_mat,
    faer::Accum::Replace,  // beta = 0.0
    a_mat,                 // A
    b_mat.transpose(),     // B^T
    1.0,                   // alpha = 1.0
    faer::Par::rayon(0),   // Multithreading via Rayon
);

        // 2. PARALLELIZE WITH RAYON (while preserving allocation hoisting)
        // We chunk the queries so we only allocate the hot-loop buffers a few times per CPU core.
        let chunk_size = 128; // Tune this: 64, 128, or 256 usually work best

        let all_edges: Vec<Edge> = (0..self.n_vectors)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map_iter(|chunk| {
                // --- HOISTED ALLOCATIONS (PER CHUNK) ---
                let mut centroid_scores: Vec<(usize, f32)> = Vec::with_capacity(self.n_list);
                let mut top_k_heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(k);
                let mut local_edges = Vec::with_capacity(chunk.len() * k);

                for query_id in chunk {
                    centroid_scores.clear();
                    top_k_heap.clear();

                    let q_offset = query_id * self.dim;
                    let query_vec = &self.bucket_data[q_offset..q_offset + self.dim];

                    // FAST COPY: use extend instead of looping push
                    let r_offset = query_id * self.n_list;
                    let scores = &routing_scores[r_offset..r_offset + self.n_list];
                    // OPTIMIZATION 1: Early Discard before sorting
                    for (c_id, &c_score) in scores.iter().enumerate() {
                        // Maximum possible similarity this bucket could contain
                        let max_potential = c_score + (1.0 - self.centroid_radii[c_id]);
                        if max_potential >= min_similarity {
                            centroid_scores.push((c_id, c_score));
                        }
                    }
                    // centroid_scores.extend(
                    //     routing_scores[r_offset..r_offset + self.n_list]
                    //         .iter()
                    //         .copied()
                    //         .enumerate(),
                    // );

                    if self.n_probe < centroid_scores.len() {
                        centroid_scores.select_nth_unstable_by(self.n_probe, |a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                        });
                    }

                    let mut current_min_score = f32::NEG_INFINITY;

                    for (centroid_id, _) in centroid_scores.iter().take(self.n_probe) {
                        // OPTIMIZATION 2: Flattened List Access (CSR)
                        let start = self.list_pointers[*centroid_id];
                        let end = self.list_pointers[*centroid_id + 1];

                        for &target_id in &self.flat_indices[start..end] {
                            if query_id == target_id {
                                continue;
                            }

                            let t_offset = target_id * self.dim;
                            let target_vec = &self.bucket_data[t_offset..t_offset + self.dim];

                            // FAST DOT PRODUCT
                            assert_eq!(query_vec.len(), target_vec.len());
                            let score: f32 = fast_dot_product(query_vec, target_vec);

                            // 3. FAST HEAP MUTATION
                            // `min_similarity` is the real threshold: without it, whenever a
                            // query has fewer than `k` candidates that actually clear it (e.g.
                            // a small bucket, or k > n_vectors), the heap fills up with
                            // whatever ranks highest regardless of absolute score, silently
                            // emitting edges below the caller's requested similarity.
                            if score >= min_similarity && (score > current_min_score || top_k_heap.len() < k) {
                                if top_k_heap.len() < k {
                                    top_k_heap.push(Hit {
                                        id: target_id,
                                        score,
                                    });
                                    if top_k_heap.len() == k {
                                        current_min_score = top_k_heap.peek().unwrap().score;
                                    }
                                } else
                                // peek_mut prevents a push + pop reallocation
                                {
                                    let mut top = top_k_heap.peek_mut().unwrap();
                                    top.id = target_id;
                                    top.score = score;
                                    drop(top);
                                    current_min_score = top_k_heap.peek().unwrap().score;
                                } // The heap automatically sifts down when `top` is dropped here.
                            }
                        }
                    }

                    for hit in top_k_heap.drain() {
                        local_edges.push(Edge {
                            source_id: query_id,
                            target_id: hit.id,
                            score: hit.score,
                        });
                    }
                }

                local_edges
            })
            .collect();

        all_edges
    }
}

pub fn graph_dbscan(
    n_vectors: usize,
    edges: &[Edge],
    eps: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let min_similarity = 1.0 - eps;
    let mut adj_list = vec![Vec::new(); n_vectors];

    // Build undirected adjacency list for edges passing the distance threshold
    for edge in edges {
        if edge.score >= min_similarity {
            adj_list[edge.source_id].push(edge.target_id);
            adj_list[edge.target_id].push(edge.source_id);
        }
    }

    // Initialize all spectra as Unclassified
    let mut labels = vec![ClusterLabel::Unclassified; n_vectors];
    let mut current_cluster_id = 0;

    for i in 0..n_vectors {
        if labels[i] != ClusterLabel::Unclassified {
            continue;
        }

        // Core Point Check
        if adj_list[i].len() + 1 < min_pts {
            labels[i] = ClusterLabel::Noise;
            continue;
        }

        // Start a new cluster
        labels[i] = ClusterLabel::Cluster(current_cluster_id);

        let mut queue = Vec::new();
        queue.extend_from_slice(&adj_list[i]);

        let mut head = 0;
        while head < queue.len() {
            let neighbor = queue[head];
            head += 1;

            match labels[neighbor] {
                ClusterLabel::Noise => {
                    // Border point: reachable from a core point, but not a core point itself
                    labels[neighbor] = ClusterLabel::Cluster(current_cluster_id);
                }
                ClusterLabel::Unclassified => {
                    // Unvisited node: assign and check if it can expand the cluster
                    labels[neighbor] = ClusterLabel::Cluster(current_cluster_id);

                    if adj_list[neighbor].len() + 1 >= min_pts {
                        queue.extend_from_slice(&adj_list[neighbor]);
                    }
                }
                ClusterLabel::Cluster(_) => {
                    // Already belongs to a cluster, do nothing.
                }
            }
        }

        current_cluster_id += 1;
    }

    labels
}

/// Default block size for [`windowed_mass_edges`]'s batched matmul. Large
/// enough to keep GEMM efficient (a 4096 x ~4096 f32 matmul is a solid-sized
/// unit of work for faer/rayon), small enough that the "lookahead" superset
/// each block computes past its own rows doesn't balloon.
pub const WINDOWED_EDGES_BLOCK_SIZE: usize = 4096;

/// Finds cosine-similarity edges within `mz_tolerance` of each other, given
/// `bucket_data` (n_vectors x dim, L2-normalized hashed spectra) and
/// `neutral_masses` (len n_vectors) **already sorted ascending** by mass —
/// which every caller of this function gets for free, since spectra are
/// sorted by `(charge, precursor_mz)` well before reaching this point (see
/// [`crate::clustering::cluster::mz_partition::mz_charge_boundaries`] and
/// `ClusterEngine::split_on_mz_and_charge`).
///
/// This replaces the IVF/spherical-k-means approximate search
/// ([`BucketIndex`]) for the "large batch" clustering path. That path exists
/// because `mz_charge_boundaries`'s partitioning rule -- group while the gap
/// between *consecutive sorted* spectra stays under tolerance -- is a
/// single-linkage rule, and single-linkage chains: given enough density, a
/// partition's first and last spectra can end up hundreds of Da apart even
/// though `mz_tolerance` at these masses is a few thousandths of a Da,
/// because a chain of tiny sub-tolerance gaps links the whole span
/// transitively. IVF/k-means treated the whole chained partition as one
/// generic high-dimensional ANN problem, spending a full `n_vectors *
/// n_list` matmul (plus 10 k-means training iterations at the same cost)
/// computing similarity between spectra that could *never* validly cluster
/// together regardless of cosine similarity, since their real mass gap
/// already disqualifies them -- `collect_and_refine`'s post-hoc per-cluster
/// m/z split was the only thing standing between that and a wrong answer,
/// and it ran after all that wasted work, not instead of it.
///
/// Mass is an *exact*, already-sorted range constraint here, not a
/// high-dimensional nearest-neighbor problem needing an approximate index:
/// once sorted, "every neighbor within tolerance of spectrum i" is just "the
/// next few spectra in the list, until the gap exceeds tolerance" -- no
/// training, no approximation, and the O(chain length) cost of a giant
/// partition collapses to O(n_vectors * local mass density) instead of O(n *
/// n_list). To keep the GEMM efficiency IVF had, spectra are processed in
/// contiguous blocks: for each block, one matmul is computed between that
/// block's rows and every row within reach of the block's own last (i.e.
/// heaviest, since sorted ascending) member — a superset of what any single
/// row in the block actually needs — then each row's real neighbors are
/// picked out of its slice of that block's scores with an exact per-pair
/// mass check, breaking out of the inner scan the moment the gap exceeds
/// tolerance (masses are sorted, so nothing further in that row can match
/// either). Only forward (i < j) edges are computed — `graph_dbscan` builds
/// its adjacency list symmetrically from each edge it's given, so a reverse
/// edge would just be redundant work for the same result.
///
/// This assumes a block's candidate window stays small enough that a direct
/// `query_len x window_len` matmul is cheap. That's true when chaining is
/// the cause of a big partition — the span between first and last spectrum
/// is huge, but *local* density (how many spectra actually fall within one
/// row's real tolerance) stays modest, which is exactly what let the
/// 969,687-spectrum worst case in this corpus resolve in ~10s. It would stop
/// being true for a dataset with genuinely dense local clustering — e.g.
/// hundreds of thousands of near-duplicate injections of the same standard,
/// all within real tolerance of each other, not just chained through a wide
/// span. That's a fundamentally different problem (an actual large nearest-
/// neighbor search, not a range join), and it's what `BucketIndex`'s IVF
/// approach is for — so when a block's window exceeds
/// `dense_window_threshold`, this falls back to it for that block only,
/// still filtering the resulting edges by the exact mass tolerance
/// afterward (IVF has no notion of mass at all, so this is required for
/// correctness, not just an optimization).
pub fn windowed_mass_edges(
    bucket_data: &[f32],
    dim: usize,
    neutral_masses: &[f64],
    tolerance: MassTolerance,
    epsilon: f32,
    block_size: usize,
    dense_window_threshold: usize,
) -> Vec<Edge> {
    let n_vectors = neutral_masses.len();
    if n_vectors < 2 {
        return Vec::new();
    }
    let min_similarity = 1.0 - epsilon;
    let max_gap_at = |mass: f64| -> f64 {
        match tolerance {
            MassTolerance::Da(tol) => tol,
            MassTolerance::Ppm(tol) => mass * (tol / 1_000_000.0),
        }
    };

    let num_blocks = n_vectors.div_ceil(block_size);

    // For each block, how far forward the candidate window must extend to
    // cover every possible tolerance-neighbor of any row in that block --
    // driven by the block's last (heaviest) row, since that's the furthest
    // any row in the block could reach. `lookahead` only ever moves forward
    // across the whole loop, so this is one O(n_vectors) pass total, not one
    // per block.
    let mut block_window_end = vec![0usize; num_blocks];
    let mut lookahead = 0usize;
    for (b, window_end) in block_window_end.iter_mut().enumerate() {
        let block_end = ((b + 1) * block_size).min(n_vectors);
        let last_mass = neutral_masses[block_end - 1];
        let max_gap = max_gap_at(last_mass);
        if lookahead < block_end {
            lookahead = block_end;
        }
        while lookahead < n_vectors && neutral_masses[lookahead] - last_mass <= max_gap {
            lookahead += 1;
        }
        *window_end = lookahead;
    }

    (0..num_blocks)
        .into_par_iter()
        .flat_map_iter(|b| {
            let block_start = b * block_size;
            let block_end = ((b + 1) * block_size).min(n_vectors);
            let window_end = block_window_end[b];
            let query_len = block_end - block_start;
            let window_len = window_end - block_start;

            if window_len > dense_window_threshold {
                return dense_window_edges_via_ivf(
                    bucket_data,
                    dim,
                    neutral_masses,
                    block_start,
                    block_end,
                    window_end,
                    max_gap_at,
                    min_similarity,
                );
            }

            let query_slice = &bucket_data[block_start * dim..block_end * dim];
            let candidate_slice = &bucket_data[block_start * dim..window_end * dim];

            let mut scores = vec![0.0f32; query_len * window_len];
            let a_mat = faer::MatRef::from_row_major_slice(query_slice, query_len, dim);
            let b_mat = faer::MatRef::from_row_major_slice(candidate_slice, window_len, dim);
            let c_mat = faer::MatMut::from_row_major_slice_mut(&mut scores, query_len, window_len);
            faer::linalg::matmul::matmul(
                c_mat,
                faer::Accum::Replace,
                a_mat,
                b_mat.transpose(),
                1.0,
                faer::Par::rayon(0),
            );

            let mut local_edges = Vec::new();
            for qi in 0..query_len {
                let global_i = block_start + qi;
                for ci in (qi + 1)..window_len {
                    let global_j = block_start + ci;
                    let gap = neutral_masses[global_j] - neutral_masses[global_i];
                    if gap > max_gap_at(neutral_masses[global_j]) {
                        break;
                    }
                    let score = scores[qi * window_len + ci];
                    if score >= min_similarity {
                        local_edges.push(Edge {
                            source_id: global_i,
                            target_id: global_j,
                            score,
                        });
                    }
                }
            }
            local_edges
        })
        .collect()
}

/// Fallback for a single [`windowed_mass_edges`] block whose candidate
/// window is too dense for a direct matmul to be cheap: builds an IVF index
/// ([`BucketIndex`]) over the block's `[block_start, window_end)` candidate
/// range and searches it for each of the block's own rows (`[block_start,
/// block_end)`), rather than every row in the range — otherwise a later
/// block would redo the same comparisons for rows that are candidates here
/// but queries there. IVF has no notion of mass, so every returned edge
/// still gets the same exact per-pair tolerance check `windowed_mass_edges`
/// would have applied, before being kept.
fn dense_window_edges_via_ivf(
    bucket_data: &[f32],
    dim: usize,
    neutral_masses: &[f64],
    block_start: usize,
    block_end: usize,
    window_end: usize,
    max_gap_at: impl Fn(f64) -> f64,
    min_similarity: f32,
) -> Vec<Edge> {
    let candidate_slice = &bucket_data[block_start * dim..window_end * dim];
    let index = BucketIndex::build(candidate_slice, dim, 64);
    let local_edges = index.extract_edges(64, 1.0 - min_similarity);
    let query_len = block_end - block_start;

    // `extract_edges` searches from every row in the window, not just this
    // block's own rows, and a mutual pair can be found from either side
    // (i's search lands on j, j's search lands on i) -- both filtered down
    // to the same canonical (smaller, larger) id pair below, so dedup on
    // that pair rather than assuming the direction alone is unique.
    let mut seen = std::collections::HashSet::new();
    local_edges
        .into_iter()
        .filter_map(|edge| {
            // Only keep edges whose query side is this block's own row --
            // the candidate side may reach into later blocks' rows, which
            // will (re-)search their own neighborhoods when their turn comes.
            if edge.source_id >= query_len {
                return None;
            }
            let global_i = block_start + edge.source_id;
            let global_j = block_start + edge.target_id;
            let (global_i, global_j) = if global_i < global_j {
                (global_i, global_j)
            } else {
                (global_j, global_i)
            };
            if global_i == global_j || !seen.insert((global_i, global_j)) {
                return None;
            }
            let gap = neutral_masses[global_j] - neutral_masses[global_i];
            if gap > max_gap_at(neutral_masses[global_j]) {
                return None;
            }
            Some(Edge {
                source_id: global_i,
                target_id: global_j,
                score: edge.score,
            })
        })
        .collect()
}

pub struct BucketIndex<'a> {
    pub dim: usize,
    pub n_list: usize,
    pub n_probe: usize,
    pub n_vectors: usize,
    /// Flat array of centroids. Shape: (n_list * dim)
    pub centroids: Vec<f32>,
    /// Borrowed flat array of vector data. Shape: (n_vectors * dim)
    pub bucket_data: &'a [f32],
    pub centroid_radii: Vec<f32>, // Store the minimum cosine similarity for each bucket
    pub flat_indices: Vec<usize>, // Flattened IVList
    pub list_pointers: Vec<usize>, // Pointers into flat_indices
}

#[derive(Debug)]
pub struct Edge {
    pub source_id: usize,
    pub target_id: usize,
    pub score: f32,
}

#[cfg(test)]
mod windowed_mass_edges_tests {
    use super::*;

    /// Builds `n_a + n_b` L2-normalized synthetic vectors: `n_a` near
    /// `[1,0,0,0]`, `n_b` near `[0,1,0,0]`, each with a tiny per-vector
    /// perturbation so they're not bit-identical but stay far above any
    /// reasonable similarity threshold within a group and far below it
    /// across groups. Masses are packed into a tight band so every point is
    /// mutually within tolerance of every other -- the whole set is one
    /// "block window" regardless of `block_size`, which is exactly the
    /// scenario `dense_window_edges_via_ivf` exists for.
    fn synthetic_two_clusters(n_a: usize, n_b: usize) -> (Vec<f32>, Vec<f64>) {
        let dim = 4;
        let n = n_a + n_b;
        let mut bucket_data = vec![0.0f32; n * dim];
        let mut neutral_masses = vec![0.0f64; n];
        for i in 0..n {
            let (base, group_idx) = if i < n_a { (0, i) } else { (1, i - n_a) };
            let jitter = 0.01 * (group_idx as f32 % 7.0 - 3.0);
            let mut v = [0.0f32; 4];
            v[base] = 1.0;
            v[(base + 1) % dim] = jitter;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim {
                bucket_data[i * dim + d] = v[d] / norm;
            }
            // Tight mass band: all points within Da(1.0) of each other.
            neutral_masses[i] = 1000.0 + (i as f64) * 0.001;
        }
        (bucket_data, neutral_masses)
    }

    fn edge_set(edges: &[Edge]) -> std::collections::HashSet<(usize, usize)> {
        edges
            .iter()
            .map(|e| {
                if e.source_id < e.target_id {
                    (e.source_id, e.target_id)
                } else {
                    (e.target_id, e.source_id)
                }
            })
            .collect()
    }

    #[test]
    fn ivf_fallback_matches_direct_matmul() {
        let (bucket_data, neutral_masses) = synthetic_two_clusters(20, 20);
        let dim = 4;
        let tolerance = MassTolerance::Da(1.0);
        let epsilon = 0.3; // min_similarity = 0.7, well below in-group (~1.0) and above cross-group (~0.0)

        // Never triggers the fallback: every block resolved by direct matmul.
        let direct = windowed_mass_edges(
            &bucket_data,
            dim,
            &neutral_masses,
            tolerance,
            epsilon,
            8,
            usize::MAX,
        );
        // Forces the fallback for every block (window spans all 40 points,
        // which exceeds a threshold of 1).
        let via_ivf = windowed_mass_edges(
            &bucket_data,
            dim,
            &neutral_masses,
            tolerance,
            epsilon,
            8,
            1,
        );

        assert!(
            !direct.is_empty(),
            "sanity check: the two tight in-group clusters should produce edges"
        );
        assert_eq!(
            edge_set(&direct),
            edge_set(&via_ivf),
            "IVF fallback must find the same edges as the direct windowed matmul"
        );

        // No cross-group edges either way -- the two clusters are ~orthogonal.
        for edge in &via_ivf {
            let same_group = (edge.source_id < 20) == (edge.target_id < 20);
            assert!(
                same_group,
                "unexpected cross-cluster edge: {edge:?}"
            );
        }
    }
}
