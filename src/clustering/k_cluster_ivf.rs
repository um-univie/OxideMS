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
        let seed = bucket_seed(bucket_data, n_list, n_vectors);
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
    /// `par` controls the k-means routing matmul's own threading: pass
    /// [`faer::Par::Seq`] when the caller is already inside a rayon task, so
    /// the matmul does not compete with its sibling tasks for the pool.
    pub fn build(bucket_data: &'a [f32], dim: usize, _max_k: usize, par: faer::Par) -> Self {
        Self::build_with(bucket_data, dim, par, IvfTraining::default())
    }

    /// [`build`](Self::build) with an explicit choice of how the coarse
    /// quantizer is trained.
    pub fn build_with(
        bucket_data: &'a [f32],
        dim: usize,
        par: faer::Par,
        training: IvfTraining,
    ) -> Self {
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

        if let IvfTraining::EMeans { epochs } = training
            && n_list > 1
        {
            let (centroids, assignments) =
                train_emeans(bucket_data, dim, n_list, n_vectors, epochs, par);
            return Self::from_assignments(bucket_data, dim, n_list, n_probe, centroids, assignments);
        }

        // Lloyd: k-means++ seeding, then 10 full-batch iterations.
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
    par,
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
        Self::from_assignments(bucket_data, dim, n_list, n_probe, centroids, assignments)
    }

    /// Builds the inverted lists and per-list radii from a trained quantizer.
    /// `assignments` need not be each vector's *nearest* centroid: radii are
    /// measured against the centroid a vector was actually filed under.
    fn from_assignments(
        bucket_data: &'a [f32],
        dim: usize,
        n_list: usize,
        n_probe: usize,
        centroids: Vec<f32>,
        assignments: Vec<usize>,
    ) -> Self {
        let n_vectors = assignments.len();
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
        self.extract_edges_for_queries(0, self.n_vectors, k, epsilon, faer::Par::rayon(0))
    }

    /// [`extract_edges`](Self::extract_edges) restricted to queries in
    /// `query_start..query_end`. Candidates are still every indexed vector;
    /// only the set of rows searched *from* is narrowed, so a caller holding
    /// an index over a superset of its own rows pays for its own searches
    /// alone. Returned ids are indices into the indexed data, as always.
    pub fn extract_edges_for_queries(
        &self,
        query_start: usize,
        query_end: usize,
        k: usize,
        epsilon: f32,
        par: faer::Par,
    ) -> Vec<Edge> {
        let query_end = query_end.min(self.n_vectors);
        if query_start >= query_end {
            return Vec::new();
        }
        let n_queries = query_end - query_start;
        let min_similarity = 1.0 - epsilon;
        let total_routing_elements = n_queries * self.n_list;

        // 1. AVOID ZEROING MEMORY: sgemm (beta = 0.0) overwrites everything.
        let mut routing_scores = Vec::with_capacity(total_routing_elements);


            // Safety: We immediately pass this to SGEMM which initializes all elements.
        unsafe {
            routing_scores.set_len(total_routing_elements);
        }

// 1. Create safe row-major matrix views. Only the query rows are routed --
// the rest of the index is candidates, never a search origin here.
let query_slice = &self.bucket_data[query_start * self.dim..query_end * self.dim];
let a_mat = faer::MatRef::from_row_major_slice(query_slice, n_queries, self.dim);
let b_mat = faer::MatRef::from_row_major_slice(&self.centroids, self.n_list, self.dim);
let c_mat = faer::MatMut::from_row_major_slice_mut(&mut routing_scores, n_queries, self.n_list);

// 2. Perform the matrix multiplication (Scores = BucketData * Centroids^T)
faer::linalg::matmul::matmul(
    c_mat,
    faer::Accum::Replace,  // beta = 0.0
    a_mat,                 // A
    b_mat.transpose(),     // B^T
    1.0,                   // alpha = 1.0
    par,
);

        // 2. PARALLELIZE WITH RAYON (while preserving allocation hoisting)
        // We chunk the queries so we only allocate the hot-loop buffers a few times per CPU core.
        let chunk_size = 128; // Tune this: 64, 128, or 256 usually work best

        let all_edges: Vec<Edge> = (query_start..query_end)
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
                    let r_offset = (query_id - query_start) * self.n_list;
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

/// How [`BucketIndex`] trains its coarse quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IvfTraining {
    /// k-means++ seeding followed by 10 full-batch spherical Lloyd iterations.
    #[default]
    Lloyd,
    /// Spherical e-means (Bessa et al. 2026, arXiv:2611.09501): mini-batch
    /// updates of exponentially decayed per-cluster sums and counts, from a
    /// random-sample start, for `epochs` passes over the data. `epochs` is
    /// held to 1..=64; a fractional part shorter than one mini-batch is
    /// dropped.
    EMeans { epochs: f32 },
}

/// Deterministic seed derived from the bucket's shape and contents so the ANN
/// path is reproducible run-to-run, as claimed in the manuscript.
fn bucket_seed(bucket_data: &[f32], n_list: usize, n_vectors: usize) -> u64 {
    let mut s = 0x9E3779B97F4A7C15u64
        ^ (n_vectors as u64).wrapping_mul(0x2545F491_4F6CDD1D)
        ^ (n_list as u64).wrapping_mul(0x9E3779B9);
    for &v in bucket_data.iter().take(8) {
        s = s.rotate_left(5) ^ (v.to_bits() as u64);
    }
    s
}

/// Batch size and initial step from the e-means scaling laws (section 5 of
/// the paper), so neither needs a sweep. The laws were fitted for `n` in
/// 1e5..1e8 and `k` in 256..65536; below that they are an extrapolation.
fn emeans_batch_and_step(n: usize, k: usize, epochs: f32) -> (usize, f32) {
    let (nf, kf, t) = (n as f64, k as f64, epochs as f64);
    let mut b = 0.035134 * kf.powf(0.134749) * nf.powf(0.876473) * t.powf(0.813344);
    if n > 2_000_000 {
        b /= 8.0;
    }
    let batch = 2f64.powf(b.log2().round()) as usize;
    let batch = batch.clamp(k.min(n), (n / 2).max(1));
    let alpha0 = 0.8244
        * kf.powf(-0.4617)
        * (batch as f64).powf(0.9003)
        * nf.powf(-0.4498)
        * t.powf(-0.1057);
    (batch, alpha0.clamp(1e-4, 1.0) as f32)
}

/// Spherical e-means. Returns `(centroids, assignments)`: unit-norm centroids
/// (`k * dim`) and, for every vector, the centroid it was assigned to during
/// the final pass.
///
/// Each cluster keeps a decayed sum `S_j` and count `N_j`; a mini-batch with
/// per-cluster sum `s_j` and count `n_j` updates them as
/// `S_j <- (1-a) S_j + a s_j`, `N_j <- (1-a) N_j + a n_j`, with `a` annealed
/// by a cosine schedule from `a0` to `0.1 a0`. The paper's method is
/// Euclidean (centroid `S_j / N_j`); the vectors here are L2-normalized and
/// compared by cosine, so the centroid is `S_j / |S_j|` instead and
/// assignment is by largest dot product -- `N_j` is kept only to detect dead
/// clusters. Every fifth batch, clusters whose count has decayed below a
/// tenth of the mean are revived by splitting a cluster drawn in proportion
/// to its count. The last pass also accumulates exact per-cluster sums, and
/// the returned centroids are those means rather than the EMA readout.
fn train_emeans(
    data: &[f32],
    dim: usize,
    k: usize,
    n: usize,
    epochs: f32,
    par: faer::Par,
) -> (Vec<f32>, Vec<usize>) {
    use rand::seq::SliceRandom;

    const REVIVE_EVERY: usize = 5;
    const REVIVE_THRESHOLD: f32 = 0.1;
    const SPLIT_DELTA: f32 = 1.0 / 128.0;
    const SCHEDULE_FLOOR: f32 = 0.1;

    let epochs = if epochs.is_finite() { epochs.clamp(1.0, 64.0) } else { 1.0 };
    let (batch, alpha0) = emeans_batch_and_step(n, k, epochs);
    let mut rng = StdRng::seed_from_u64(bucket_seed(data, k, n));

    // Random-sample start: k-means++ costs about one Lloyd iteration in
    // sequential picks and the paper finds it buys nothing here.
    let mut centroids = vec![0.0f32; k * dim];
    for (slot, idx) in rand::seq::index::sample(&mut rng, n, k).into_iter().enumerate() {
        centroids[slot * dim..(slot + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
    }
    let mut counts = vec![1e-10f32; k];
    let mut sums: Vec<f32> = centroids.iter().map(|&c| c * 1e-10).collect();

    // A leading partial pass carries any fractional part of `epochs`; the
    // final pass is always whole so every vector ends up with an assignment.
    let full_passes = epochs.floor() as usize;
    let partial_len = (((epochs - full_passes as f32) * n as f32) as usize).min(n);
    let mut pass_lens = Vec::with_capacity(full_passes + 1);
    if partial_len >= batch {
        pass_lens.push(partial_len);
    }
    pass_lens.extend(std::iter::repeat_n(n, full_passes));
    let total_steps: usize = pass_lens.iter().map(|len| len.div_ceil(batch)).sum();

    let mut order: Vec<usize> = (0..n).collect();
    let mut assignments = vec![0usize; n];
    let mut final_sums = vec![0.0f32; k * dim];
    let mut final_counts = vec![0u32; k];

    let mut batch_data = vec![0.0f32; batch * dim];
    let mut scores = vec![0.0f32; batch * k];
    let mut batch_assign = vec![0usize; batch];
    // Batch rows grouped by assigned centroid (counting sort), so the update
    // can run one task per centroid with no shared accumulator.
    let mut group_start = vec![0usize; k + 1];
    let mut grouped = vec![0usize; batch];

    let mut step = 0usize;
    let last_pass = pass_lens.len() - 1;
    for (pass, &pass_len) in pass_lens.iter().enumerate() {
        order.shuffle(&mut rng);
        let is_last = pass == last_pass;
        for ids in order[..pass_len].chunks(batch) {
            let m = ids.len();
            let progress = step as f32 / total_steps as f32;
            let alpha = alpha0
                * (SCHEDULE_FLOOR
                    + (1.0 - SCHEDULE_FLOOR) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));
            step += 1;

            batch_data[..m * dim]
                .par_chunks_exact_mut(dim)
                .zip(ids.par_iter())
                .for_each(|(row, &id)| row.copy_from_slice(&data[id * dim..(id + 1) * dim]));

            matmul::matmul(
                faer::MatMut::from_row_major_slice_mut(&mut scores[..m * k], m, k),
                faer::Accum::Replace,
                faer::MatRef::from_row_major_slice(&batch_data[..m * dim], m, dim),
                faer::MatRef::from_row_major_slice(&centroids, k, dim).transpose(),
                1.0,
                par,
            );
            batch_assign[..m]
                .par_iter_mut()
                .zip(scores[..m * k].par_chunks_exact(k))
                .for_each(|(assignment, row)| {
                    *assignment = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                        .unwrap()
                        .0;
                });

            group_start.fill(0);
            for &c in &batch_assign[..m] {
                group_start[c + 1] += 1;
            }
            for c in 0..k {
                group_start[c + 1] += group_start[c];
            }
            let mut cursor = group_start.clone();
            for (row, &c) in batch_assign[..m].iter().enumerate() {
                grouped[cursor[c]] = row;
                cursor[c] += 1;
            }
            if is_last {
                for (&id, &c) in ids.iter().zip(&batch_assign[..m]) {
                    assignments[id] = c;
                }
            }

            let batch_rows = &batch_data[..m * dim];
            let (group_start, grouped) = (&group_start, &grouped);
            sums.par_chunks_exact_mut(dim)
                .zip(centroids.par_chunks_exact_mut(dim))
                .zip(counts.par_iter_mut())
                .zip(final_sums.par_chunks_exact_mut(dim).zip(final_counts.par_iter_mut()))
                .enumerate()
                .for_each_init(
                    || vec![0.0f32; dim],
                    |batch_sum, (c, (((sum, centroid), count), (final_sum, final_count)))| {
                        let members = &grouped[group_start[c]..group_start[c + 1]];
                        batch_sum.fill(0.0);
                        for &row in members {
                            let v = &batch_rows[row * dim..(row + 1) * dim];
                            for d in 0..dim {
                                batch_sum[d] += v[d];
                            }
                        }
                        if is_last {
                            for d in 0..dim {
                                final_sum[d] += batch_sum[d];
                            }
                            *final_count += members.len() as u32;
                        }
                        *count = (1.0 - alpha) * *count + alpha * members.len() as f32;
                        let mut norm_sq = 0.0f32;
                        for d in 0..dim {
                            sum[d] = (1.0 - alpha) * sum[d] + alpha * batch_sum[d];
                            norm_sq += sum[d] * sum[d];
                        }
                        // A sum decayed to nothing keeps its last direction
                        // until revival replaces it.
                        if norm_sq > 0.0 {
                            let inv = norm_sq.sqrt().recip();
                            for d in 0..dim {
                                centroid[d] = sum[d] * inv;
                            }
                        }
                    },
                );

            // Not during the last pass: its members are being filed under
            // centroid ids for the final means, and a revived id would mix
            // members of two unrelated regions.
            if step % REVIVE_EVERY == 0 && !is_last {
                revive_dead_clusters(
                    &mut sums,
                    &mut counts,
                    &mut centroids,
                    dim,
                    REVIVE_THRESHOLD,
                    SPLIT_DELTA,
                    &mut rng,
                );
            }
        }
    }

    // Finalize on the exact mean of the last pass's members. A cluster that
    // received none keeps its EMA centroid and an empty list.
    centroids
        .par_chunks_exact_mut(dim)
        .zip(final_sums.par_chunks_exact(dim))
        .zip(final_counts.par_iter())
        .for_each(|((centroid, final_sum), &final_count)| {
            let norm_sq: f32 = final_sum.iter().map(|x| x * x).sum();
            if final_count > 0 && norm_sq > 0.0 {
                let inv = norm_sq.sqrt().recip();
                for d in 0..dim {
                    centroid[d] = final_sum[d] * inv;
                }
            }
        });

    (centroids, assignments)
}

/// Replaces every cluster whose decayed count is below `threshold` times the
/// mean with half of a live cluster drawn in proportion to its count: the
/// donor's sum is split into two copies perturbed in opposite directions
/// (alternating `1 +/- delta` over its non-zero coordinates) and its count
/// is halved.
fn revive_dead_clusters(
    sums: &mut [f32],
    counts: &mut [f32],
    centroids: &mut [f32],
    dim: usize,
    threshold: f32,
    delta: f32,
    rng: &mut StdRng,
) {
    let k = counts.len();
    let mean = counts.iter().sum::<f32>() / k as f32;
    let cutoff = threshold * mean;
    let dead: Vec<usize> = (0..k).filter(|&c| counts[c] < cutoff).collect();
    if dead.is_empty() || dead.len() == k {
        return;
    }
    let live: Vec<usize> = (0..k).filter(|&c| counts[c] >= cutoff).collect();
    let mut cumulative = Vec::with_capacity(live.len());
    let mut total = 0.0f64;
    for &c in &live {
        total += counts[c] as f64;
        cumulative.push(total);
    }
    let mut used = vec![false; live.len()];
    let mut n_used = 0usize;

    for dead_id in dead {
        if n_used == live.len() {
            break;
        }
        // Without replacement: redraw on a donor already split this round,
        // falling back to the next unused one if the draws keep colliding.
        let mut draw = |rng: &mut StdRng| {
            let target = rng.random_range(0.0..total);
            cumulative.partition_point(|&w| w < target)
        };
        let mut pick = draw(rng);
        for _ in 0..8 {
            if !used[pick.min(live.len() - 1)] {
                break;
            }
            pick = draw(rng);
        }
        let mut pick = pick.min(live.len() - 1);
        while used[pick] {
            pick = (pick + 1) % live.len();
        }
        used[pick] = true;
        n_used += 1;
        let donor_id = live[pick];

        let half = counts[donor_id] * 0.5;
        counts[donor_id] = half;
        counts[dead_id] = half;
        let (mut donor_norm, mut dead_norm) = (0.0f32, 0.0f32);
        // The sign alternates over the non-zero coordinates, not over all of
        // them: sparse vectors can have every non-zero on one parity, and
        // scaling all of them alike would leave both halves pointing the
        // same way.
        let mut sign = delta;
        for d in 0..dim {
            let base = sums[donor_id * dim + d] * 0.5;
            if base != 0.0 {
                sign = -sign;
            }
            let (a, b) = (base * (1.0 - sign), base * (1.0 + sign));
            sums[donor_id * dim + d] = a;
            sums[dead_id * dim + d] = b;
            donor_norm += a * a;
            dead_norm += b * b;
        }
        if donor_norm > 0.0 && dead_norm > 0.0 {
            let (donor_inv, dead_inv) = (donor_norm.sqrt().recip(), dead_norm.sqrt().recip());
            for d in 0..dim {
                centroids[donor_id * dim + d] = sums[donor_id * dim + d] * donor_inv;
                centroids[dead_id * dim + d] = sums[dead_id * dim + d] * dead_inv;
            }
        }
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

/// Upper bound on the rows in one [`windowed_mass_edges`] block. Blocks are
/// normally cut earlier than this, by mass span (see [`mass_span_blocks`]);
/// this caps the pathological case where a huge run of spectra sits inside a
/// single tolerance width, which would otherwise put a `rows x rows` score
/// matrix on the heap.
pub const WINDOWED_EDGES_BLOCK_SIZE: usize = 4096;

/// How many tolerance widths of mass a [`windowed_mass_edges`] block may span
/// before it is cut. A block costs `rows x window_len` dot products but only
/// the pairs within one tolerance of each other can become edges, so the
/// wasted fraction of that matmul is roughly this multiple. Keeping it small
/// keeps the waste small; keeping it above 1 means the lookahead past the
/// block's own rows stays a minority of the window.
const BLOCK_SPAN_TOLERANCES: f64 = 8.0;

/// Floor on block rows, as a divisor of the maximum: a block never falls
/// below `WINDOWED_EDGES_BLOCK_SIZE / 16` rows even if its mass span is
/// already over budget. Sparse mass regions would otherwise cut blocks down
/// to a handful of rows, where the matmul is too small to be worth
/// dispatching.
const BLOCK_MIN_ROWS_DIVISOR: usize = 16;

/// Cuts `neutral_masses` (sorted ascending) into blocks for
/// [`windowed_mass_edges`], returning boundary indices that start with `0`
/// and end with `neutral_masses.len()`, so consecutive pairs
/// `boundaries[i]..boundaries[i + 1]` are the blocks.
///
/// A block ends once it has covered [`BLOCK_SPAN_TOLERANCES`] tolerance
/// widths of mass, clamped to at least `max_rows / BLOCK_MIN_ROWS_DIVISOR`
/// and at most `max_rows` rows. Sizing by mass span rather than by a fixed
/// row count is what keeps the per-block matmul honest: the block computes
/// every pair among its rows, but only pairs within one tolerance of each
/// other can survive the mass check, so a block spanning a thousand
/// tolerance widths throws away all but a thousandth of its own work. Total
/// dot products across all blocks are `n_vectors x (block rows + lookahead)`,
/// i.e. linear in the block size, so cutting blocks at the point where they
/// stop being locally dense is a direct saving in both time and in the size
/// of the score buffer held per in-flight block.
///
/// Runs in O(n) overall: each row is examined by exactly one block.
fn mass_span_blocks(
    neutral_masses: &[f64],
    max_rows: usize,
    max_gap_at: impl Fn(f64) -> f64,
) -> Vec<usize> {
    let n = neutral_masses.len();
    let mut boundaries = vec![0usize];
    let max_rows = max_rows.max(1);
    let min_rows = (max_rows / BLOCK_MIN_ROWS_DIVISOR).max(1);
    let mut start = 0usize;
    while start < n {
        let hard_end = (start + max_rows).min(n);
        // Always take `min_rows` first, then keep taking rows while the block
        // still fits in its mass budget. `end > start` always holds, so this
        // terminates.
        let mut end = (start + min_rows).min(n);
        let span_limit = BLOCK_SPAN_TOLERANCES * max_gap_at(neutral_masses[start]);
        while end < hard_end && neutral_masses[end] - neutral_masses[start] <= span_limit {
            end += 1;
        }
        boundaries.push(end);
        start = end;
    }
    boundaries
}

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
/// contiguous blocks, sized by mass span rather than by a fixed row count
/// (see [`mass_span_blocks`] -- `block_size` here is the upper bound on a
/// block's rows, not its usual size): for each block, one matmul is computed
/// between that block's rows and every row within reach of its own last (i.e.
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
    windowed_mass_edges_with(
        bucket_data,
        dim,
        neutral_masses,
        tolerance,
        epsilon,
        block_size,
        dense_window_threshold,
        IvfTraining::default(),
    )
}

/// [`windowed_mass_edges`] with an explicit choice of how the IVF fallback
/// trains its quantizer. Irrelevant to blocks resolved by direct matmul.
#[allow(clippy::too_many_arguments)]
pub fn windowed_mass_edges_with(
    bucket_data: &[f32],
    dim: usize,
    neutral_masses: &[f64],
    tolerance: MassTolerance,
    epsilon: f32,
    block_size: usize,
    dense_window_threshold: usize,
    ivf_training: IvfTraining,
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

    let block_boundaries = mass_span_blocks(neutral_masses, block_size, max_gap_at);
    let num_blocks = block_boundaries.len() - 1;

    // For each block, how far forward the candidate window must extend to
    // cover every possible tolerance-neighbor of any row in that block --
    // driven by the block's last (heaviest) row, since that's the furthest
    // any row in the block could reach. The bound has to match the per-pair
    // test below, which evaluates the tolerance at the *heavier* partner's
    // mass: under ppm that is strictly wider than the tolerance at the
    // block's own last row, and pairs in the difference would otherwise be
    // scored by nobody, since the next block only looks forward.
    // `lookahead` only ever moves forward across the whole loop, so this is
    // one O(n_vectors) pass total, not one per block.
    let mut block_window_end = vec![0usize; num_blocks];
    let mut lookahead = 0usize;
    for (b, window_end) in block_window_end.iter_mut().enumerate() {
        let block_end = block_boundaries[b + 1];
        let last_mass = neutral_masses[block_end - 1];
        if lookahead < block_end {
            lookahead = block_end;
        }
        while lookahead < n_vectors
            && neutral_masses[lookahead] - last_mass <= max_gap_at(neutral_masses[lookahead])
        {
            lookahead += 1;
        }
        *window_end = lookahead;
    }

    // Blocks already run one per rayon task, so a matmul that also spreads
    // itself across the pool just fights its siblings for the same threads.
    // The exception is having too few blocks to fill the pool -- one large
    // dense partition can be a handful of blocks on a many-core machine, and
    // there a sequential matmul would leave most cores idle.
    let inner_par = if num_blocks >= rayon::current_num_threads() {
        faer::Par::Seq
    } else {
        faer::Par::rayon(0)
    };

    (0..num_blocks)
        .into_par_iter()
        .flat_map_iter(|b| {
            let block_start = block_boundaries[b];
            let block_end = block_boundaries[b + 1];
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
                    inner_par,
                    ivf_training,
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
                inner_par,
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
    par: faer::Par,
    training: IvfTraining,
) -> Vec<Edge> {
    let candidate_slice = &bucket_data[block_start * dim..window_end * dim];
    let index = BucketIndex::build_with(candidate_slice, dim, par, training);
    let query_len = block_end - block_start;
    // Search only from this block's own rows. The rest of the window is
    // candidates here and queries in their own block, so searching from them
    // too would compute the same neighbourhoods twice -- for a window several
    // times the block's size, most of the search was thrown away by the
    // `source_id >= query_len` test that used to sit below.
    let local_edges = index.extract_edges_for_queries(0, query_len, 64, 1.0 - min_similarity, par);

    // A mutual pair can still be found from either side within this block
    // (i's search lands on j, j's search lands on i) -- both filtered down to
    // the same canonical (smaller, larger) id pair below, so dedup on that
    // pair rather than assuming the direction alone is unique.
    let mut seen = std::collections::HashSet::new();
    local_edges
        .into_iter()
        .filter_map(|edge| {
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

    /// Random L2-normalized vectors in one of two near-orthogonal
    /// directions, with randomly spaced masses. The two directions are far
    /// enough apart that no similarity lands near the threshold, so the only
    /// thing that decides an edge is the mass window -- which is what these
    /// tests are about, and it keeps the comparison against the brute-force
    /// reference free of float-boundary flakiness.
    fn random_dataset(n: usize, seed: u64, max_step: f64) -> (Vec<f32>, Vec<f64>) {
        let dim = 4;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bucket_data = vec![0.0f32; n * dim];
        let mut neutral_masses = vec![0.0f64; n];
        let mut mass = 1000.0f64;
        for i in 0..n {
            let base = rng.random_range(0..2usize);
            let jitter = rng.random_range(-0.02f32..0.02f32);
            let mut v = [0.0f32; 4];
            v[base] = 1.0;
            v[(base + 2) % dim] = jitter;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim {
                bucket_data[i * dim + d] = v[d] / norm;
            }
            neutral_masses[i] = mass;
            mass += rng.random_range(0.0..max_step);
        }
        (bucket_data, neutral_masses)
    }

    /// O(n^2) reference: every forward pair whose mass gap is within
    /// tolerance *evaluated at the heavier partner* (the convention
    /// [`windowed_mass_edges`] and `mz_charge_boundaries` both use) and whose
    /// cosine similarity clears the threshold.
    fn brute_force_edge_set(
        bucket_data: &[f32],
        dim: usize,
        neutral_masses: &[f64],
        tolerance: MassTolerance,
        epsilon: f32,
    ) -> std::collections::HashSet<(usize, usize)> {
        let min_similarity = 1.0 - epsilon;
        let n = neutral_masses.len();
        let mut out = std::collections::HashSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let gap = neutral_masses[j] - neutral_masses[i];
                let max_gap = match tolerance {
                    MassTolerance::Da(tol) => tol,
                    MassTolerance::Ppm(tol) => neutral_masses[j] * (tol / 1_000_000.0),
                };
                if gap > max_gap {
                    continue;
                }
                let score = fast_dot_product(
                    &bucket_data[i * dim..(i + 1) * dim],
                    &bucket_data[j * dim..(j + 1) * dim],
                );
                if score >= min_similarity {
                    out.insert((i, j));
                }
            }
        }
        out
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

    /// e-means has to hand back a usable quantizer: unit centroids, every
    /// vector filed exactly once, and edge recall on par with the Lloyd-trained index.
    #[test]
    fn emeans_index_recall_is_close_to_lloyd() {
        let (dim, n, n_groups) = (16, 2000, 8);
        let mut rng = StdRng::seed_from_u64(11);
        let mut data = vec![0.0f32; n * dim];
        for (i, row) in data.chunks_exact_mut(dim).enumerate() {
            row[i % n_groups] = 1.0;
            for x in row.iter_mut() {
                *x += rng.random_range(0.0f32..0.05);
            }
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            row.iter_mut().for_each(|x| *x /= norm);
        }

        let emeans = BucketIndex::build_with(
            &data,
            dim,
            faer::Par::Seq,
            IvfTraining::EMeans { epochs: 3.0 },
        );
        assert!(emeans.n_list > 1, "test must exercise real training");
        for centroid in emeans.centroids.chunks_exact(dim) {
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "centroid norm {norm}");
        }
        let mut filed = emeans.flat_indices.clone();
        filed.sort_unstable();
        assert_eq!(filed, (0..n).collect::<Vec<_>>());

        // Neither index is exact (only `n_probe` lists are scanned), so
        // both are held against the true edge set rather than each other.
        let (k, epsilon) = (n, 0.2f32);
        let mut truth = std::collections::HashSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let score =
                    fast_dot_product(&data[i * dim..(i + 1) * dim], &data[j * dim..(j + 1) * dim]);
                if score >= 1.0 - epsilon {
                    truth.insert((i, j));
                }
            }
        }
        let recall = |index: &BucketIndex| {
            let found = edge_set(&index.extract_edges(k, epsilon));
            assert!(found.is_subset(&truth), "index returned an edge below the threshold");
            found.len() as f64 / truth.len() as f64
        };
        let lloyd = BucketIndex::build_with(&data, dim, faer::Par::Seq, IvfTraining::Lloyd);
        let (emeans_recall, lloyd_recall) = (recall(&emeans), recall(&lloyd));
        eprintln!("recall: e-means {emeans_recall:.4}, lloyd {lloyd_recall:.4}");
        assert!(
            emeans_recall >= lloyd_recall - 0.05,
            "e-means recall {emeans_recall:.4} well below lloyd {lloyd_recall:.4}"
        );
    }

    #[test]
    fn emeans_is_deterministic() {
        let (data, _) = random_dataset(3000, 5, 0.01);
        let build = || {
            BucketIndex::build_with(&data, 4, faer::Par::Seq, IvfTraining::EMeans { epochs: 2.0 })
        };
        let (a, b) = (build(), build());
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.flat_indices, b.flat_indices);
    }

    #[test]
    fn matches_brute_force_with_da_tolerance() {
        let dim = 4;
        let epsilon = 0.3;
        let tolerance = MassTolerance::Da(0.05);
        for seed in 0..4u64 {
            let (bucket_data, neutral_masses) = random_dataset(600, seed, 0.02);
            let expected =
                brute_force_edge_set(&bucket_data, dim, &neutral_masses, tolerance, epsilon);
            assert!(!expected.is_empty(), "seed {seed}: reference found no edges");
            for block_size in [1, 2, 7, 64, 512, 4096] {
                let actual = edge_set(&windowed_mass_edges(
                    &bucket_data,
                    dim,
                    &neutral_masses,
                    tolerance,
                    epsilon,
                    block_size,
                    usize::MAX,
                ));
                assert_eq!(
                    actual, expected,
                    "seed {seed}, block_size {block_size}: windowed edges differ from brute force"
                );
            }
        }
    }

    #[test]
    fn matches_brute_force_with_ppm_tolerance() {
        let dim = 4;
        let epsilon = 0.3;
        let tolerance = MassTolerance::Ppm(5_000.0);
        for seed in 10..14u64 {
            let (bucket_data, neutral_masses) = random_dataset(600, seed, 2.0);
            let expected =
                brute_force_edge_set(&bucket_data, dim, &neutral_masses, tolerance, epsilon);
            assert!(!expected.is_empty(), "seed {seed}: reference found no edges");
            for block_size in [1, 2, 7, 64, 512, 4096] {
                let actual = edge_set(&windowed_mass_edges(
                    &bucket_data,
                    dim,
                    &neutral_masses,
                    tolerance,
                    epsilon,
                    block_size,
                    usize::MAX,
                ));
                assert_eq!(
                    actual, expected,
                    "seed {seed}, block_size {block_size}: windowed edges differ from brute force"
                );
            }
        }
    }

    /// Blocking is an implementation detail of how the work is batched, so
    /// the edge set must not depend on it -- including on how the mass-span
    /// rule happens to cut blocks for a given dataset.
    #[test]
    fn block_size_does_not_change_the_edge_set() {
        let dim = 4;
        let epsilon = 0.3;
        let (bucket_data, neutral_masses) = random_dataset(800, 42, 0.01);
        let tolerance = MassTolerance::Da(0.05);
        let reference = edge_set(&windowed_mass_edges(
            &bucket_data,
            dim,
            &neutral_masses,
            tolerance,
            epsilon,
            1,
            usize::MAX,
        ));
        for block_size in [2, 3, 7, 64, 333, 1024, 8192] {
            let actual = edge_set(&windowed_mass_edges(
                &bucket_data,
                dim,
                &neutral_masses,
                tolerance,
                epsilon,
                block_size,
                usize::MAX,
            ));
            assert_eq!(actual, reference, "block_size {block_size} changed the edges");
        }
    }

    /// Every edge must be forward (`source_id < target_id`) and appear once:
    /// `graph_dbscan` builds its adjacency list symmetrically from each edge,
    /// so a reverse or duplicate edge inflates neighbour counts.
    #[test]
    fn edges_are_forward_and_unique() {
        let dim = 4;
        let (bucket_data, neutral_masses) = random_dataset(500, 7, 0.01);
        let edges = windowed_mass_edges(
            &bucket_data,
            dim,
            &neutral_masses,
            MassTolerance::Da(0.05),
            0.3,
            64,
            usize::MAX,
        );
        assert!(!edges.is_empty());
        let mut seen = std::collections::HashSet::new();
        for edge in &edges {
            assert!(
                edge.source_id < edge.target_id,
                "non-forward edge: {edge:?}"
            );
            assert!(seen.insert((edge.source_id, edge.target_id)), "duplicate edge: {edge:?}");
        }
    }

    fn block_sizes(boundaries: &[usize]) -> Vec<usize> {
        boundaries.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Masses packed inside a single tolerance width: nothing to cut on, so
    /// blocks run to the row cap.
    #[test]
    fn mass_span_blocks_cap_dense_regions_at_max_rows() {
        let masses: Vec<f64> = (0..1000).map(|i| 1000.0 + i as f64 * 1e-6).collect();
        let boundaries = mass_span_blocks(&masses, 256, |_| 0.01);
        assert_eq!(block_sizes(&boundaries), vec![256, 256, 256, 232]);
    }

    /// Masses spaced well beyond the span budget: every block bottoms out at
    /// the row floor rather than shrinking to single rows.
    #[test]
    fn mass_span_blocks_floor_sparse_regions_at_min_rows() {
        let masses: Vec<f64> = (0..100).map(|i| 1000.0 + i as f64 * 10.0).collect();
        let boundaries = mass_span_blocks(&masses, 64, |_| 0.01);
        // 64 / BLOCK_MIN_ROWS_DIVISOR = 4 rows per block.
        assert_eq!(block_sizes(&boundaries), vec![4; 25]);
    }

    /// Whatever the mass distribution, the boundaries must be a partition of
    /// `0..n`: strictly increasing, non-empty blocks, no row skipped or
    /// visited twice.
    #[test]
    fn mass_span_blocks_partition_every_row() {
        for (n, seed, max_step) in [(0, 1, 0.0), (1, 2, 0.1), (7, 3, 0.1), (997, 4, 0.5)] {
            let (_, masses) = random_dataset(n, seed, max_step);
            let boundaries = mass_span_blocks(&masses, 64, |m| m * 1e-5);
            assert_eq!(boundaries.first().copied(), Some(0));
            assert_eq!(boundaries.last().copied(), Some(n), "n = {n}");
            for w in boundaries.windows(2) {
                assert!(w[1] > w[0], "empty or backwards block in {boundaries:?}");
                assert!(w[1] - w[0] <= 64, "block over the row cap in {boundaries:?}");
            }
        }
    }

    /// Regression test for the lookahead bound.
    ///
    /// A block's candidate window used to be extended while
    /// `mass - last_mass <= max_gap_at(last_mass)`, but the per-pair test
    /// that decides an edge uses `max_gap_at(mass_of_heavier)`, which is
    /// strictly larger under ppm tolerance. Pairs in that sliver were never
    /// scored: the earlier block's window stopped short of them, and the
    /// later block only looks forward. The ppm here is exaggerated so the
    /// sliver is wide enough to write down; at instrument tolerances it is
    /// `mass * ppm^2` wide.
    #[test]
    fn lookahead_reaches_neighbours_of_the_blocks_last_row() {
        let dim = 4;
        // Identical vectors: similarity is 1.0 for every pair, so the mass
        // window alone decides the edges.
        let bucket_data = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        // 10% tolerance. Row 1 (the last row of a 2-row block) reaches row 2:
        // gap 105.0 <= max_gap_at(1105.0) = 110.5. Bounding the window by
        // max_gap_at(1000.0) = 100.0 instead stops one row short.
        let neutral_masses = vec![1000.0, 1000.0, 1105.0];
        let tolerance = MassTolerance::Ppm(100_000.0);
        let edges = edge_set(&windowed_mass_edges(
            &bucket_data,
            dim,
            &neutral_masses,
            tolerance,
            0.3,
            2,
            usize::MAX,
        ));
        assert_eq!(
            edges,
            brute_force_edge_set(&bucket_data, dim, &neutral_masses, tolerance, 0.3),
            "window must cover every tolerance-neighbour of the block's last row"
        );
    }
}
