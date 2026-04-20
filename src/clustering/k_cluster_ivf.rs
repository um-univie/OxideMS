use crate::clustering::cluster::{ClusterLabel, fast_dot_product};
use matrixmultiply::sgemm;
use rand::RngExt;
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
        let mut rng = rand::rng();

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

            // Sum the distances to establish the total probability pool
            let total_weight: f32 = min_sq_distances.iter().sum();

            // Weighted random selection (Roulette Wheel Selection)
            // We pick a random float between 0 and the total weight.
            let target = rng.random_range(0.0..1.0) * total_weight;
            let mut cumulative_weight = 0.0;
            let mut next_centroid_idx = n_vectors - 1; // Fallback to the last item just in case

            // Walk through the weights until we hit our random target
            for (i, &weight) in min_sq_distances.iter().enumerate() {
                cumulative_weight += weight;
                if cumulative_weight >= target {
                    next_centroid_idx = i;
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

        for _iter in 0..10 {
            // 1. SGEMM: Compute scores (Assignments)
            scores_buffer
                .par_chunks_mut(rows_per_chunk * n_list)
                .enumerate()
                .for_each(|(chunk_idx, c_chunk)| {
                    let current_m = c_chunk.len() / n_list;
                    let start_row = chunk_idx * rows_per_chunk;

                    unsafe {
                        let a_lhs_ptr = bucket_data.as_ptr().add(start_row * dim);
                        let a_rhs_ptr = centroids.as_ptr(); // Centroids matrix

                        sgemm(
                            current_m,
                            dim,
                            n_list,
                            1.0,
                            a_lhs_ptr,
                            dim as isize,
                            1, // A (batch data)
                            a_rhs_ptr,
                            1,
                            dim as isize, // B (centroids, transposed via strides)
                            0.0,
                            c_chunk.as_mut_ptr(),
                            n_list as isize,
                            1, // C (scores)
                        );
                    }
                });

            // 2. Argmax: Find best centroid for each vector
            for i in 0..n_vectors {
                let row_start = i * n_list;
                let row_scores = &scores_buffer[row_start..row_start + n_list];

                let best_centroid = row_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                    .unwrap()
                    .0;
                assignments[i] = best_centroid;
            }
            // 3. Recalculate Centroids
            centroids.fill(0.0);
            let mut counts = vec![0.0; n_list];

            for i in 0..n_vectors {
                let c_id = assignments[i];
                let vec_offset = i * dim;
                let cent_offset = c_id * dim;

                // Add vector to its assigned centroid
                for d in 0..dim {
                    centroids[cent_offset + d] += bucket_data[vec_offset + d];
                }
                counts[c_id] += 1.0;
            }
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

        unsafe {
            // Safety: We immediately pass this to SGEMM which initializes all elements.
            routing_scores.set_len(total_routing_elements);

            sgemm(
                self.n_vectors,
                self.dim,
                self.n_list,
                1.0,
                self.bucket_data.as_ptr(),
                self.dim as isize,
                1,
                self.centroids.as_ptr(),
                1,
                self.dim as isize,
                0.0,
                routing_scores.as_mut_ptr(),
                self.n_list as isize,
                1,
            );
        }

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
                            if score > current_min_score || top_k_heap.len() < k {
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
