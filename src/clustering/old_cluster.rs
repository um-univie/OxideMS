use crate::{
    HashedSpectrumConfig, MSNSpectrum, SpectrumHasher,
    clustering::{
        bitset::{BitSetOpsMSB0, WORDSIZE},
        k_cluster_ivf::{BucketIndex, graph_dbscan},
    },
    io::mgf::{MGF_SPECTRA_SIZE, MgfParser, MgfParserConfig},
    workflow::*,
};

use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{MatMut, MatRef};
use matrixmultiply::sgemm;
use memmap2::Mmap;
use nohash_hasher::IntMap;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize, rancor::Error};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, ErrorKind, Write};
use std::path::Path;

const MIN_NEIGHBORS: usize = 2;
const LARGE_BATCH_THRESHOLD: usize = 30_000;
const FAST_BATCH_NUMBER: usize = 10000;

#[cfg(target_vendor = "apple")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_ssyrk(
        layout: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// ---------------------------------------------------------
// Helper for the Min-Heap: Floats don't implement Ord in Rust
// so we wrap them to allow sorting in the Priority Queue.
// ---------------------------------------------------------
#[derive(PartialEq, PartialOrd)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Fallback to Equal if NaN is encountered
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------
// Internal Node representing a cluster during the linkage
// ---------------------------------------------------------
struct Node {
    min: f64,
    max: f64,
    left: Option<usize>,
    right: Option<usize>,
    active: bool,
    // Instead of allocating a Vec<usize> for every merge,
    // we track the contiguous bounds in the sorted slice!
    start_idx: usize,
    end_idx: usize,
}

pub struct FlatClusters {
    pub data: Vec<usize>,    // All IDs packed together
    pub offsets: Vec<usize>, // Starting positions for each cluster
}

impl FlatClusters {
    // Helper to get the slice of IDs belonging to a specific cluster
    fn get_cluster(&self, cluster_id: usize) -> &[usize] {
        if cluster_id >= self.offsets.len() - 1 {
            return &[];
        }
        let start = self.offsets[cluster_id];
        let end = self.offsets[cluster_id + 1];
        &self.data[start..end]
    }

    /// Get the size of a specific cluster
    pub fn cluster_size(&self, cluster_id: usize) -> usize {
        if cluster_id >= self.offsets.len() - 1 {
            return 0;
        }
        self.offsets[cluster_id + 1] - self.offsets[cluster_id]
    }

    // --- Statistics ---

    /// Total number of clusters (including empty ones)
    pub fn num_clusters(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of clusters that actually contain members
    pub fn active_clusters_count(&self) -> usize {
        self.offsets.windows(2).filter(|w| w[1] > w[0]).count()
    }

    /// Total number of elements assigned to clusters
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// The size of the largest cluster
    pub fn max_cluster_size(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }
    /// The size of the largest cluster
    pub fn min_cluster_size(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .min()
            .unwrap_or(0)
    }

    /// Average size of active clusters
    pub fn avg_cluster_size(&self) -> f64 {
        let active = self.active_clusters_count();
        if active == 0 {
            return 0.0;
        }
        self.total_elements() as f64 / active as f64
    }

    /// Returns an iterator over the clusters
    pub fn iter(&self) -> ClusterIter<'_> {
        ClusterIter {
            clusters: self,
            current_id: 0,
        }
    }

    pub fn get_cluster_spans(&self, mz_values: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );
        let mut spans = vec![];

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_mzs = &mz_values[start..end];
            spans.push(cluster_mzs.last().unwrap() - cluster_mzs.get(0).unwrap())
        }
        spans
    }
    pub fn get_cluster_spans_ppm(&self, mz_values: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );
        let mut spans = vec![];

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_mzs = &mz_values[start..end];
            let max_mass = cluster_mzs.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            let min_mass = cluster_mzs.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
            spans.push(1_000_000.0 * ((max_mass - min_mass) / max_mass))
        }
        spans
    }
    pub fn refine_mz_clusters(&self, mz_values: &[f64], tolerance: MassTolerance) -> FlatClusters {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );

        let mut new_data = Vec::with_capacity(self.data.len());
        // Here we could surely make a heuristic based on the span of the values as this will force clusters?
        let mut new_offsets = Vec::with_capacity(self.offsets.len());
        new_offsets.push(0);

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_ids = &self.data[start..end];
            let cluster_mzs = &mz_values[start..end];

            if cluster_ids.len() <= 1 {
                continue;
            }

            Self::refine_single_cluster(
                cluster_ids,
                cluster_mzs,
                tolerance,
                &mut new_data,
                &mut new_offsets,
            );
        }

        FlatClusters {
            data: new_data,
            offsets: new_offsets,
        }
    }

    fn refine_single_cluster(
        cluster_ids: &[usize],
        cluster_mzs: &[f64],
        tolerance: MassTolerance,
        new_data: &mut Vec<usize>,
        new_offsets: &mut Vec<usize>,
    ) {
        let n = cluster_ids.len();

        let mut local_indices: Vec<usize> = (0..n).collect();
        local_indices.sort_unstable_by(|&a, &b| {
            cluster_mzs[a]
                .partial_cmp(&cluster_mzs[b])
                .unwrap_or(Ordering::Equal)
        });

        let mut nodes: Vec<Node> = Vec::with_capacity(n * 2);
        for (i, &local_idx) in local_indices.iter().enumerate() {
            let val = cluster_mzs[local_idx];
            nodes.push(Node {
                min: val,
                max: val,
                left: if i > 0 { Some(i - 1) } else { None },
                right: if i < n - 1 { Some(i + 1) } else { None },
                active: true,
                start_idx: i,
                end_idx: i + 1,
            });
        }

        let mut heap = BinaryHeap::new();
        for i in 0..n - 1 {
            let dist = nodes[i + 1].max - nodes[i].min;
            heap.push(Reverse((OrderedF64(dist), i, i + 1)));
        }

        while let Some(Reverse((OrderedF64(dist), left_id, right_id))) = heap.pop() {
            if !nodes[left_id].active || !nodes[right_id].active {
                continue;
            }

            // PPM is dynamic. We check if the distance is valid for the current cluster range.
            // We use the max m/z in the merge to determine the allowed window.
            if dist > tolerance.tolerance_at(nodes[right_id].max) {
                continue;
            }

            nodes[left_id].active = false;
            nodes[right_id].active = false;

            let next_id = nodes.len();
            let new_node = Node {
                min: nodes[left_id].min,
                max: nodes[right_id].max,
                left: nodes[left_id].left,
                right: nodes[right_id].right,
                active: true,
                start_idx: nodes[left_id].start_idx,
                end_idx: nodes[right_id].end_idx,
            };

            if let Some(left_neighbor_id) = new_node.left {
                nodes[left_neighbor_id].right = Some(next_id);
                let new_dist_left = new_node.max - nodes[left_neighbor_id].min;
                heap.push(Reverse((
                    OrderedF64(new_dist_left),
                    left_neighbor_id,
                    next_id,
                )));
            }

            if let Some(right_neighbor_id) = new_node.right {
                nodes[right_neighbor_id].left = Some(next_id);
                let new_dist_right = nodes[right_neighbor_id].max - new_node.min;
                heap.push(Reverse((
                    OrderedF64(new_dist_right),
                    next_id,
                    right_neighbor_id,
                )));
            }

            nodes.push(new_node);
        }
        // 5. Extract results from active nodes
        for node in nodes.iter().filter(|n| n.active) {
            // CHANGE 2: Check the size of the resulting sub-cluster.
            // If it is 1 (or 0), skip it entirely.
            if node.end_idx - node.start_idx <= 1 {
                continue;
            }

            for i in node.start_idx..node.end_idx {
                let original_idx = local_indices[i];
                new_data.push(cluster_ids[original_idx]);
            }
            new_offsets.push(new_data.len());
        }
    }
}
/// Custom Iterator to allow: for (id, members) in flat_clusters.iter()
pub struct ClusterIter<'a> {
    clusters: &'a FlatClusters,
    current_id: usize,
}

impl<'a> Iterator for ClusterIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_id < self.clusters.num_clusters() {
            let members = self.clusters.get_cluster(self.current_id);
            self.current_id += 1;

            if !members.is_empty() {
                return Some(members);
            }
        }
        None
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterLabel {
    Unclassified,
    Noise,
    Cluster(usize),
}

/// In-place L2 normalization of a flattened vector of spectra.
pub fn normalize_spectra_in_place(flat_data: &mut [f32], stride: usize) {
    // chunks_mut yields non-overlapping mutable slices of length `stride`
    for spectrum in flat_data.chunks_mut(stride) {
        // 1. Calculate the L2 norm (Euclidean length) of the spectrum
        let norm_sq: f32 = spectrum.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();

        // 2. Divide each
        if norm > std::f32::EPSILON {
            for x in spectrum.iter_mut() {
                *x /= norm;
            }
        }
    }
}
/// Computes the 1D index for an upper triangular matrix (excluding the diagonal).
/// `i` and `j` are the row and column indices. Requires `i < j`.
#[inline]
fn upper_triangular_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j, "i must be strictly less than j");
    // This formula calculates the offset by summing the lengths of all rows before `i`,
    // then adding the offset within row `i`.
    // Algebraically simplified to avoid loop operations:
    (i * (2 * n - i - 1)) / 2 + (j - i - 1)
}

/// Retrieves the similarity score between two points.
fn get_similarity(matrix: &[f32], n: usize, p: usize, q: usize) -> f32 {
    if p == q {
        return 1.0; // A point is perfectly similar to itself
    }

    // Ensure we always query the upper triangle (i < j)
    let (i, j) = if p < q { (p, q) } else { (q, p) };
    let idx = upper_triangular_index(n, i, j);

    matrix[idx]
}

/// Runs DBSCAN on a precomputed upper-triangular similarity matrix.
///
/// * `similarity_matrix` - The flat 1D vector generated by your parallel iterator.
/// * `num_points` - The total number of spectra (`number_of_spectra`).
/// * `min_similarity` - The threshold for two spectra to be considered neighbors (e.g., 0.85).
/// * `min_pts` - The minimum number of points required to form a dense region (cluster).
pub fn dbscan(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;

    // Helper closure to find all neighbors for a given point
    let get_neighbors = |p: usize| -> Vec<usize> {
        (0..num_points)
            .filter(|&q| get_similarity(similarity_matrix, num_points, p, q) >= min_similarity)
            .collect()
    };

    for p in 0..num_points {
        if labels[p] != ClusterLabel::Unclassified {
            continue;
        }

        let neighbors = get_neighbors(p);

        if neighbors.len() < min_pts {
            labels[p] = ClusterLabel::Noise;
        } else {
            current_cluster_id += 1;
            labels[p] = ClusterLabel::Cluster(current_cluster_id);

            let mut seed_set = Vec::new();

            // Eagerly assign labels to neighbors to prevent duplicate processing later
            for &n in &neighbors {
                if n == p {
                    continue;
                }

                if labels[n] == ClusterLabel::Noise {
                    // Border point: belongs to cluster, but doesn't expand it
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                } else if labels[n] == ClusterLabel::Unclassified {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    seed_set.push(n);
                }
            }

            // Expand the cluster using a while loop since `seed_set` grows dynamically
            let mut i = 0;
            while i < seed_set.len() {
                let q = seed_set[i];
                let q_neighbors = get_neighbors(q);

                if q_neighbors.len() >= min_pts {
                    for &n in &q_neighbors {
                        if labels[n] == ClusterLabel::Noise {
                            labels[n] = ClusterLabel::Cluster(current_cluster_id);
                        } else if labels[n] == ClusterLabel::Unclassified {
                            labels[n] = ClusterLabel::Cluster(current_cluster_id);
                            seed_set.push(n);
                        }
                    }
                }
                i += 1;
            }
        }
    }

    labels
}

/// Runs DBSCAN on a precomputed similarity matrix.
///
/// * `similarity_matrix` - he flat 1D vector similarity matrix
/// * `num_points` - The total number of spectra (`number_of_spectra`).
/// * `min_similarity` - The threshold for two spectra to be considered neighbors (e.g., 0.85).
/// * `min_pts` - The minimum number of points required to form a dense region (cluster).
pub fn dbscan_full(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;

    // Helper closure to find all neighbors for a given point this includes the point itself
    let get_neighbors = |p: usize| -> Vec<usize> {
        (0..num_points)
            .filter(|&q| {
                get_similarity_square(similarity_matrix, num_points, p, q) >= min_similarity
            })
            .collect()
    };

    let mut seed_set = Vec::new();
    for p in 0..num_points {
        if labels[p] != ClusterLabel::Unclassified {
            continue;
        }

        let neighbors = get_neighbors(p);

        if neighbors.len() < min_pts {
            labels[p] = ClusterLabel::Noise;
        } else {
            labels[p] = ClusterLabel::Cluster(current_cluster_id);

            seed_set.clear();
            // Eagerly assign labels to neighbors to prevent duplicate processing later
            for &n in &neighbors {
                if n == p {
                    continue;
                }

                if labels[n] == ClusterLabel::Noise {
                    // Border point: belongs to cluster, but doesn't expand it
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                } else if labels[n] == ClusterLabel::Unclassified {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    seed_set.push(n);
                }
            }

            // Expand the cluster using a while loop since `seed_set` grows dynamically
            let mut i = 0;
            while i < seed_set.len() {
                let q = seed_set[i];
                let q_neighbors = get_neighbors(q);

                if q_neighbors.len() >= min_pts {
                    for &n in &q_neighbors {
                        if labels[n] == ClusterLabel::Noise {
                            labels[n] = ClusterLabel::Cluster(current_cluster_id);
                        } else if labels[n] == ClusterLabel::Unclassified {
                            labels[n] = ClusterLabel::Cluster(current_cluster_id);
                            seed_set.push(n);
                        }
                    }
                }
                i += 1;
            }
            current_cluster_id += 1;
        }
    }

    labels
}

/// Retrieves the similarity score from a full N x N flat matrix.
fn get_similarity_square(
    matrix: &[f32],
    matrix_size: usize,
    row_index: usize,
    column_index: usize,
) -> f32 {
    // Standard 2D-to-1D mapping for a square matrix
    let idx = row_index * matrix_size + column_index;
    matrix[idx]
}
/// A compact Union-Find implementation using path compression and union-by-rank.
struct DSU {
    parent: Vec<i32>,
}

impl DSU {
    fn new(size: usize) -> Self {
        // Initialize with -1. Negative values indicate a root,
        // and the absolute value is the rank (tree height).
        Self {
            parent: vec![-1; size],
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] < 0 {
            i
        } else {
            // Path compression: point node directly to the root
            let root = self.find(self.parent[i] as usize);
            self.parent[i] = root as i32;
            root
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            // Union by rank: attach the shorter tree to the taller one
            if self.parent[root_i] < self.parent[root_j] {
                self.parent[root_j] = root_i as i32;
            } else {
                if self.parent[root_i] == self.parent[root_j] {
                    self.parent[root_j] -= 1; // Increase rank
                }
                self.parent[root_i] = root_j as i32;
            }
        }
    }
}

pub fn dbscan_single_linkage_upper_triangular(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
) -> Vec<ClusterLabel> {
    // 1. Initialize Union-Find / Disjoint-Set structures
    let mut parent: Vec<usize> = (0..num_points).collect();
    let mut size = vec![1; num_points];

    // Helper closure for Path Compression Find
    // Uses path halving for maximum speed without recursion overhead
    let mut find = |mut i: usize, parent: &mut [usize]| -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    };

    // Helper closure for Union by Size
    let mut link = |mut i: usize, mut j: usize, parent: &mut [usize], size: &mut [usize]| {
        i = find(i, parent);
        j = find(j, parent);
        if i != j {
            // Attach smaller tree to the root of larger tree
            if size[i] < size[j] {
                std::mem::swap(&mut i, &mut j);
            }
            parent[j] = i;
            size[i] += size[j];
        }
    };

    // 2. Single Pass: Traverse the upper-triangular matrix
    // We skip j=i (self-similarity) and the lower triangle
    for i in 0..num_points {
        let row_offset = i * num_points;
        for j in (i + 1)..num_points {
            // If they are similar enough, link them
            if similarity_matrix[row_offset + j] >= min_similarity {
                link(i, j, &mut parent, &mut size);
            }
        }
    }

    // 3. Resolve final cluster labels
    let mut labels = vec![ClusterLabel::Noise; num_points];
    let mut next_cluster_id = 0;
    let mut root_to_id = vec![None; num_points];

    for i in 0..num_points {
        let root = find(i, &mut parent);

        // If the component size is >= 2, it's a valid cluster
        if size[root] >= 2 {
            let id = match root_to_id[root] {
                Some(id) => id,
                None => {
                    let new_id = next_cluster_id;
                    next_cluster_id += 1;
                    root_to_id[root] = Some(new_id);
                    new_id
                }
            };
            labels[i] = ClusterLabel::Cluster(id);
        } else {
            // Size 1 means it has no edges, making it Noise
            labels[i] = ClusterLabel::Noise;
        }
    }

    labels
}

pub fn dbscan_upper_triangular(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let words_per_vertex = (num_points + WORDSIZE - 1) / WORDSIZE;

    // 1. Pre-allocate the Bit-Matrix (N x words_per_vertex)
    // Using a single flat Vec for cache locality
    let mut bit_matrix = vec![0usize; num_points * words_per_vertex];

    // 2. Build Bit-Matrix in Parallel
    // We process rows in parallel. Each row becomes a full BitSet.
    bit_matrix
        .par_chunks_mut(words_per_vertex)
        .enumerate()
        .for_each(|(i, row_bits)| {
            for j in 0..num_points {
                // Access similarity (Symmetric logic)
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                if similarity_matrix[r * num_points + c] >= min_similarity {
                    row_bits.set_bit(j);
                }
            }
        });
    dbscan_bitmatrix(&bit_matrix, num_points, min_pts, words_per_vertex)
}

fn dbscan_bitmatrix(
    bit_matrix: &[usize],
    num_points: usize,
    min_pts: usize,
    words_per_vertex: usize,
) -> Vec<ClusterLabel> {
    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;
    let mut seed_set = Vec::with_capacity(num_points);

    for p in 0..num_points {
        if labels[p] != ClusterLabel::Unclassified {
            continue;
        }

        let p_row = &bit_matrix[p * words_per_vertex..(p + 1) * words_per_vertex];

        // Fast neighbor count using POPCNT
        if p_row.population_count() < min_pts as u32 {
            labels[p] = ClusterLabel::Noise;
            continue;
        }

        // Found a core point
        labels[p] = ClusterLabel::Cluster(current_cluster_id);
        seed_set.clear();

        // Initial expansion from p
        let mut idx = p_row.next_set_bit(0);
        while let Some(q) = idx {
            if q != p {
                if labels[q] == ClusterLabel::Unclassified || labels[q] == ClusterLabel::Noise {
                    labels[q] = ClusterLabel::Cluster(current_cluster_id);
                    seed_set.push(q);
                }
            }
            idx = p_row.next_set_bit(q + 1);
        }

        // 3. Cluster Expansion (The "Greedy" Loop)
        let mut i = 0;
        while i < seed_set.len() {
            let q = seed_set[i];
            let q_row = &bit_matrix[q * words_per_vertex..(q + 1) * words_per_vertex];

            if q_row.population_count() >= min_pts as u32 {
                let mut neighbor_idx = q_row.next_set_bit(0);
                while let Some(n) = neighbor_idx {
                    if labels[n] == ClusterLabel::Unclassified {
                        labels[n] = ClusterLabel::Cluster(current_cluster_id);
                        seed_set.push(n);
                    } else if labels[n] == ClusterLabel::Noise {
                        labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    }
                    neighbor_idx = q_row.next_set_bit(n + 1);
                }
            }
            i += 1;
        }
        current_cluster_id += 1;
    }

    labels
}

/// Retrieves the similarity score from an N x N matrix where only the
/// upper triangle is guaranteed to be valid.
fn get_similarity_upper(matrix: &[f32], num_points: usize, row: usize, col: usize) -> f32 {
    // If we are in the lower triangle, swap row and column to access the upper triangle
    let (r, c) = if row <= col { (row, col) } else { (col, row) };

    // Standard row-major mapping: index = row * width + col
    let idx = r * num_points + c;
    matrix[idx]
}

/// Packed upper-triangular index: for row r ≤ col c in an n×n matrix.
/// Layout matches the GPU shader: `r*(2*n - r - 1)/2 + c`.
#[inline]
pub fn packed_upper_idx(r: usize, c: usize, n: usize) -> usize {
    r * (2 * n - r - 1) / 2 + c
}

/// DBSCAN on a **packed** upper-triangular similarity matrix produced by the GPU.
///
/// `similarity_matrix` has length `n*(n+1)/2`.  Element (r, c) with r ≤ c is
/// at `packed_upper_idx(r, c, n)`.
pub fn dbscan_upper_triangular_packed(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let words_per_vertex = (num_points + WORDSIZE - 1) / WORDSIZE;

    let mut bit_matrix = vec![0usize; num_points * words_per_vertex];

    bit_matrix
        .par_chunks_mut(words_per_vertex)
        .enumerate()
        .for_each(|(i, row_bits)| {
            for j in 0..num_points {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                if similarity_matrix[packed_upper_idx(r, c, num_points)] >= min_similarity {
                    row_bits.set_bit(j);
                }
            }
        });

    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;
    let mut seed_set = Vec::with_capacity(num_points);

    for p in 0..num_points {
        if labels[p] != ClusterLabel::Unclassified {
            continue;
        }

        let p_row = &bit_matrix[p * words_per_vertex..(p + 1) * words_per_vertex];

        if p_row.population_count() < min_pts as u32 {
            labels[p] = ClusterLabel::Noise;
            continue;
        }

        labels[p] = ClusterLabel::Cluster(current_cluster_id);
        seed_set.clear();

        let mut idx = p_row.next_set_bit(0);
        while let Some(q) = idx {
            if q != p
                && (labels[q] == ClusterLabel::Unclassified || labels[q] == ClusterLabel::Noise)
            {
                labels[q] = ClusterLabel::Cluster(current_cluster_id);
                seed_set.push(q);
            }
            idx = p_row.next_set_bit(q + 1);
        }

        let mut i = 0;
        while i < seed_set.len() {
            let q = seed_set[i];
            let q_row = &bit_matrix[q * words_per_vertex..(q + 1) * words_per_vertex];

            if q_row.population_count() >= min_pts as u32 {
                let mut neighbor_idx = q_row.next_set_bit(0);
                while let Some(n) = neighbor_idx {
                    if labels[n] == ClusterLabel::Unclassified {
                        labels[n] = ClusterLabel::Cluster(current_cluster_id);
                        seed_set.push(n);
                    } else if labels[n] == ClusterLabel::Noise {
                        labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    }
                    neighbor_idx = q_row.next_set_bit(n + 1);
                }
            }
            i += 1;
        }
        current_cluster_id += 1;
    }

    labels
}

/// Represents a mass spectrum as a collection of peaks
#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq))]
pub struct MassSpectrumBatchCollection {
    batch: Vec<MassSpectrumBatch>,
}

impl MassSpectrumBatchCollection {
    pub fn dump(&self, file_path: &str) -> Result<(), std::io::Error> {
        let file = std::fs::File::create(file_path).unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(rkyv::to_bytes::<Error>(self).unwrap().as_slice())
    }
}
/// Represents a mass spectrum as a collection of peaks
#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq))]
pub struct MassSpectrumBatch {
    batch: Vec<MSNSpectrum>,
}

impl MassSpectrumBatch {
    pub fn dump(&self, file_path: &str) -> Result<(), std::io::Error> {
        let file = std::fs::File::create(file_path).unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(rkyv::to_bytes::<Error>(self).unwrap().as_slice())
    }
}

const TEMP_FOLDER_NAME: &str = "temp_spectra";
const TEST_FILE_NAME: &str = "test_files/test2.mgf";

#[derive(Default)]
pub struct SpectralExperiment {
    pub file_names: Vec<String>,
    /// Maps a global Spectrum ID to its lightweight metadata
    pub metadata: IntMap<usize, SpectrumMetadata>,
    pub workflow: WorkFlowManager,
    pub parser: MgfParser,
    // You can keep the engine here, or instantiate it when needed
    pub engine: ClusterEngine,
    pub spectra: Vec<MSNSpectrum>,
    pub total_spectra_ingested: usize,
}

impl SpectralExperiment {
    pub fn default() -> Self {
        let mut workflow = WorkFlowManager::default();
        workflow.add_operation(FilterMZ::new(101.0, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        Self {
            workflow,
            engine: ClusterEngine::default(),
            parser: MgfParser::default(),
            spectra: Vec::new(),
            metadata: IntMap::default(),
            file_names: Vec::new(),
            total_spectra_ingested: 0,
        }
    }
    pub fn proteomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        let min_mz = 200.0;
        // The higher lower limit here helps to remove individual amino acids which would inflate scoring
        workflow.add_operation(FilterMZ::new(min_mz, 1500.0));

        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(SquareRootTransformIntensity::new());
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        let mut engine = ClusterEngine::default();
        engine.hasher.min_mz = min_mz;

        Self {
            workflow,
            engine: ClusterEngine::default(),
            parser: MgfParser::default(),
            spectra: Vec::new(),
            metadata: IntMap::default(),
            file_names: Vec::new(),
            total_spectra_ingested: 0,
        }
    }

    pub fn metabolomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        workflow.add_operation(FilterMZ::new(50.0, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(SquareRootTransformIntensity::new());
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        Self {
            workflow,
            engine: ClusterEngine::default(),
            parser: MgfParser::default(),
            spectra: Vec::new(),
            metadata: IntMap::default(),
            file_names: Vec::new(),
            total_spectra_ingested: 0,
        }
    }

    pub fn calculate_n10_value(&self, clusters: &FlatClusters) -> usize {
        let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();
        cluster_sizes.sort_by(|a, b| b.cmp(a));
        let mut cumulator: usize = 0;
        for cluster_size in cluster_sizes {
            cumulator += cluster_size;
            if cumulator >= (self.total_spectra_ingested * 10) / 100 {
                return cluster_size;
            }
        }
        0
    }

    pub fn calculate_cluster_purities(
        &self,
        clusters: &FlatClusters,
        mz_tol: f64,
        rt_tol: f32,
    ) -> Vec<f64> {
        clusters
            .iter()
            .map(|cluster| self.calculate_cluster_purity(cluster, mz_tol, rt_tol))
            .collect()
    }

    pub fn average_weighted_cluster_purity(
        &self,
        clusters: &FlatClusters,
        mz_tol: f64,
        rt_tol: f32,
    ) -> f64 {
        let purities = self.calculate_cluster_purities(&clusters, mz_tol, rt_tol);

        let (product, sum) = clusters
            .iter()
            .zip(purities)
            .fold((0.0, 0), |acc, (cluster, purity)| {
                (acc.0 + cluster.len() as f64 * purity, acc.1 + cluster.len())
            });
        product / sum as f64
    }
    pub fn average_cluster_purity(&self, clusters: &FlatClusters, mz_tol: f64, rt_tol: f32) -> f64 {
        let purities = self.calculate_cluster_purities(&clusters, mz_tol, rt_tol);

        let sum = purities.iter().sum::<f64>();
        sum / purities.len() as f64
    }

    pub fn calculate_cluster_purity(&self, cluster_ids: &[usize], mz_tol: f64, rt_tol: f32) -> f64 {
        let n = cluster_ids.len();
        if n <= 1 {
            return 1.0;
        }

        // 1. DSU (Union-Find) structures to track connected components
        let mut parent: Vec<usize> = (0..n).collect();
        let mut component_size: Vec<usize> = vec![1; n];

        fn find(i: usize, parent: &mut Vec<usize>) -> usize {
            if parent[i] == i {
                i
            } else {
                parent[i] = find(parent[i], parent);
                parent[i]
            }
        }

        fn unite(i: usize, j: usize, parent: &mut Vec<usize>, size: &mut Vec<usize>) {
            let root_i = find(i, parent);
            let root_j = find(j, parent);
            if root_i != root_j {
                parent[root_i] = root_j;
                size[root_j] += size[root_i];
            }
        }

        // 2. Identify which file each spectrum belongs to
        // We store the file index for each spectrum in the cluster
        let mut file_assignments = Vec::with_capacity(n);
        for &id in cluster_ids {
            let f_idx = self.metadata.get(&id).unwrap().file_index;
            file_assignments.push(f_idx);
        }

        // 3. Build the network: Pairwise comparison within the cluster
        for i in 0..n {
            for j in i + 1..n {
                // Only connect if they belong to the same file (Internal Control)
                if file_assignments[i] == file_assignments[j] {
                    let s1 = self.metadata.get(&cluster_ids[i]).unwrap();
                    let s2 = self.metadata.get(&cluster_ids[j]).unwrap();

                    let mz_diff = (s1.precursor_mz - s2.precursor_mz).abs();
                    let rt_diff =
                        (s1.retention_time_in_seconds - s2.retention_time_in_seconds).abs();

                    if mz_diff <= mz_tol && rt_diff <= rt_tol {
                        unite(i, j, &mut parent, &mut component_size);
                    }
                }
            }
        }

        // 4. Calculate Max Component Size per file
        // file_max_components[file_index] = size of largest connected group in that file
        let mut file_max_components = vec![0usize; self.file_names.len()];

        // We only care about the size of the root nodes in our DSU
        for i in 0..n {
            let root = find(i, &mut parent);
            let f_idx = file_assignments[i];
            let size = component_size[root];

            if size > file_max_components[f_idx] {
                file_max_components[f_idx] = size;
            }
        }

        // 5. Calculate Final Purity
        // Sum of (Max Component per file) / Total spectra in cluster
        let sum_max_components: usize = file_max_components.iter().sum();

        sum_max_components as f64 / n as f64
    }
    pub fn ingest_files_parallel(&mut self, folder_path: &str) {
        self.workflow.use_parallel = false;
        let paths: Vec<_> = std::fs::read_dir(folder_path)
            .unwrap()
            .filter_map(|e| e.ok()?.path().to_str().map(|s| s.to_string()))
            .collect();

        // --- Optimization: Pre-allocate based on total file sizes ---
        let total_disk_size: u64 = paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();

        let estimated_total_spectra = (total_disk_size / MGF_SPECTRA_SIZE as u64) as usize;

        self.spectra.reserve(estimated_total_spectra);
        self.metadata.reserve(estimated_total_spectra);

        // 1. Prepare global file names (needed for the file_index)
        let start_index = self.file_names.len();

        self.file_names.extend(paths.clone());

        // 2. Process in parallel
        // We use par_iter on the paths with their index
        let results: Vec<(Vec<MSNSpectrum>, Vec<(usize, SpectrumMetadata)>)> = paths
            .into_par_iter()
            .enumerate()
            .map(|(i, path)| {
                let current_file_index = start_index + i;
                self.process_single_file(&path, current_file_index)
            })
            .collect();

        // 3. Sequential Merge
        for (file_spectra, file_metadata) in results {
            self.total_spectra_ingested += file_spectra.len();
            self.metadata.extend(file_metadata);
            self.spectra.extend(file_spectra);
        }
    }

    /// Pure function/method that doesn't mutate 'self' globally,
    /// making it thread-safe for Rayon.
    fn process_single_file(
        &self,
        path: &str,
        file_index: usize,
    ) -> (Vec<MSNSpectrum>, Vec<(usize, SpectrumMetadata)>) {
        // Create a local parser or use a stateless one
        let mut parser = MgfParser::default();
        parser.config.use_parallel_parsing = false;
        let mut spectra = parser.parse_file(path).unwrap();

        self.workflow.execute(&mut spectra);

        let mut local_metadata = Vec::new();
        let processed_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| !s.peaks().is_empty() && s.get_charge() != 0)
            .enumerate() // Use the local loop index
            .map(|(local_idx, s)| {
                let mut scan_number = s
                    .get_metadata("SCANS")
                    .unwrap_or("0")
                    .parse::<usize>()
                    .unwrap_or(0);
                let mut msn = s.to_msn();

                // Generate a Global ID: (file_index << 32) | local_idx
                let global_id = (file_index << 32) | local_idx;
                msn.id = global_id as u64;

                local_metadata.push((
                    global_id,
                    SpectrumMetadata {
                        precursor_mz: msn.precursor_mz,
                        retention_time_in_seconds: msn.retention_time_in_seconds,
                        charge: msn.charge(),
                        scan_number,
                        file_index,
                    },
                ));
                msn
            })
            .collect();

        (processed_spectra, local_metadata)
    }
    /// Ingests a file, runs filters, stores metadata, and returns the processed spectra
    /// ready to be dumped to disk.
    pub fn ingest_file(&mut self, file_path: &str) {
        let file_index = self.file_names.len();
        // TODO make this robust
        self.file_names.push(
            file_path
                .split("/")
                .last()
                .unwrap()
                .split(".mgf")
                .next()
                .unwrap()
                .to_string(),
        );

        let mut spectra = self.parser.parse_file(file_path).unwrap();
        self.total_spectra_ingested += spectra.len();
        self.workflow.execute(&mut spectra);

        let msn_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| !(s.peaks().len() == 0) && s.get_charge() != 0)
            .map(|s| {
                (
                    s.get_metadata("SCANS")
                        .unwrap_or("0")
                        .parse::<usize>()
                        .unwrap_or(0),
                    s.to_msn(),
                )
            })
            .inspect(|(scan_number, msn)| {
                // Store the metadata globally before the spectrum goes to disk
                self.metadata.insert(
                    msn.id as usize,
                    SpectrumMetadata {
                        precursor_mz: msn.precursor_mz,
                        retention_time_in_seconds: msn.retention_time_in_seconds, // Assuming this exists
                        charge: msn.charge(),
                        scan_number: *scan_number,
                        file_index,
                    },
                );
            })
            .map(|(_, msn_spectrum)| msn_spectrum)
            .collect();

        self.spectra.extend(msn_spectra)
    }
    /// Ingests a file, runs filters, stores metadata, and returns the processed spectra
    /// ready to be dumped to disk.
    pub fn ingest_files(&mut self, folder_path: &str) {
        for file_entry in std::fs::read_dir(folder_path).unwrap() {
            let file_path = file_entry.unwrap().path();
            if let Some(path_str) = file_path.to_str() {
                if !path_str.ends_with(".mgf") {
                    continue;
                }
                self.ingest_file(path_str);
            }
        }
    }

    pub fn prepare_clustering(&mut self) {
        self.engine
            .split_on_mz_and_charge(&mut self.spectra)
            .unwrap()
        // self.engine.split_and_refine_clusters(&mut self.spectra);
    }

    pub fn cluster(&mut self) -> FlatClusters {
        let clusters = self.engine.cluster().unwrap();

        // 4. Extract MZ values using the map
        let mz_values: Vec<f64> = clusters
            .data
            .iter()
            .map(|id| self.metadata.get(id).unwrap().precursor_mz)
            .collect();
        // 5. Refine and return
        let clusters = clusters.refine_mz_clusters(&mz_values, self.engine.mz_tolerance);

        // let mz_values: Vec<f64> = clusters
        //     .data
        //     .iter()
        //     .map(|id| self.metadata.get(id).unwrap().precursor_mz)
        //     .collect();
        // let spans = clusters.get_cluster_spans_ppm(&mz_values);
        // let sum = spans.iter().sum::<f64>();
        // let max = spans.iter().max_by(|a, b| a.total_cmp(&b)).unwrap();

        // println!(
        //     "Average span ppm: {}, Max ppm: {}",
        //     sum / clusters.num_clusters() as f64,
        //     max
        // );
        clusters
    }

    /// Executes the clustering engine on the parsed spectra and returns the refined clusters.
    /// Takes a mutable reference to `msn_spectra` since `split_on_mz_and_charge` mutates it.
    pub fn cluster_spectra(&mut self) -> FlatClusters {
        self.prepare_clustering();
        self.cluster()
    }

    pub fn write_results(&self, clusters: &FlatClusters, output_file_name: &str) {
        if clusters.total_elements() == 0 {
            return;
        }
        let mut file = std::fs::File::create(output_file_name).unwrap();
        let mut writer = std::io::BufWriter::new(&mut file);
        writer
            .write(format!("# oxidems\n# eps = {}\n# hash_len = {}\nidentifier,precursor_charge,precursor_mz,retention_time,cluster\n",  self.engine.epsilon, self.engine.hash_size()).as_bytes())
            .unwrap();
        let mut counter = 0;
        for cluster in clusters.iter() {
            counter += 1;
            for spectrum in cluster {
                let metadata = self.metadata.get(spectrum).unwrap();
                let precursor_charge = metadata.charge;
                let precursor_mz = metadata.precursor_mz;
                let retention_time = metadata.retention_time_in_seconds;
                writer
                    .write(
                        format!(
                            "mzspec:USI000000:{}:scan:{},{},{},{},{}\n",
                            &self.file_names[metadata.file_index],
                            metadata.scan_number,
                            precursor_charge,
                            precursor_mz,
                            retention_time,
                            counter,
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            }
        }
    }
}

pub struct SpectrumMetadata {
    pub precursor_mz: f64,
    pub retention_time_in_seconds: f32,
    pub file_index: usize,
    pub scan_number: usize,
    pub charge: i8,
}

pub struct ClusterEngine {
    spectral_buffer: Vec<f32>,
    similarity_matrix_buffer: Vec<f32>,
    pub mz_tolerance: MassTolerance,
    pub epsilon: f32,
    batch_size_threshold: usize,
    hasher: SpectrumHasher,
    pub exact: bool,
    pub use_gpu: bool,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::clustering::gpu::GpuSimilarity>,
}

impl Default for ClusterEngine {
    fn default() -> Self {
        let number_of_spectra = 128;
        let hash_size = 800;
        let eps = 0.1;
        let matrix_size = number_of_spectra * number_of_spectra;
        ClusterEngine {
            spectral_buffer: vec![0.0; number_of_spectra * hash_size],
            similarity_matrix_buffer: vec![0.0; matrix_size],
            epsilon: eps,
            mz_tolerance: MassTolerance::Ppm(20.0),
            batch_size_threshold: FAST_BATCH_NUMBER,
            hasher: SpectrumHasher::new(100.0, 1500.0, MassTolerance::Da(0.05), hash_size),
            exact: false,
            use_gpu: false,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }
}

impl ClusterEngine {
    pub fn new(hash_size: usize, eps: f32) -> Self {
        let number_of_spectra = 128;
        let matrix_size = number_of_spectra * (number_of_spectra + 1) / 2;
        ClusterEngine {
            spectral_buffer: vec![0.0; number_of_spectra * hash_size],
            similarity_matrix_buffer: vec![0.0; matrix_size],
            epsilon: eps,
            mz_tolerance: MassTolerance::Ppm(20.0),
            batch_size_threshold: FAST_BATCH_NUMBER,
            hasher: SpectrumHasher::new(100.0, 1500.0, MassTolerance::Da(0.05), hash_size),
            exact: false,
            use_gpu: false,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Attempt to initialise GPU acceleration.
    ///
    /// If a suitable GPU is found, `use_gpu` is set to `true` and subsequent
    /// slow-batch similarity computations will run on the GPU.  If no GPU is
    /// available the call is a no-op and CPU fallback remains active.
    ///
    /// `max_batch` — largest slow-batch size (in spectra) the GPU will handle.
    ///              Batches larger than this fall back to CPU automatically.
    ///              Defaults to `LARGE_BATCH_THRESHOLD` when 0 is passed.
    #[cfg(feature = "gpu")]
    pub fn enable_gpu(&mut self, max_batch: usize) {
        let max_m = if max_batch == 0 {
            LARGE_BATCH_THRESHOLD
        } else {
            max_batch
        };
        if let Some(ctx) = crate::clustering::gpu::GpuSimilarity::try_new(max_m, self.hash_size()) {
            self.gpu = Some(ctx);
            self.use_gpu = true;
        }
    }
    pub fn hash_size(&self) -> usize {
        self.hasher.hash_len
    }
    pub fn set_hash_size(&mut self, hash_len: usize) {
        self.hasher.hash_len = hash_len;
        self.hasher.recalculate_lookup_table();
    }

    pub fn split_and_refine_clusters(
        &mut self,
        spectra: &mut [MSNSpectrum],
    ) -> Result<(), std::io::Error> {
        // 1. Parallel Sort by Charge then M/Z
        spectra.par_sort_unstable_by(|a, b| {
            (a.charge(), a.precursor_mz)
                .partial_cmp(&(b.charge(), b.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fast_batches = Vec::new();
        let mut slow_batches = Vec::new();

        // 2. Coarse Grouping (Partitioning)
        // We group by Charge and a base tolerance to create manageable work units
        for coarse_group in spectra.chunk_by_mut(|a, b| {
            if a.charge() != b.charge() {
                return false;
            }

            let m_a = a.get_neutral_mass();
            let m_b = b.get_neutral_mass();
            let gap = (m_b - m_a).abs();

            match self.mz_tolerance {
                MassTolerance::Da(tol) => gap <= tol,
                MassTolerance::Ppm(tol) => gap <= m_b * (tol / 1_000_000.0),
            }
        }) {
            if coarse_group.len() <= 1 {
                continue;
            }

            // 3. Fine-Grained Refinement
            // We extract the m/z values for the refinement logic
            let mzs: Vec<f64> = coarse_group.iter().map(|s| s.get_neutral_mass()).collect();
            let mut refined_data = Vec::with_capacity(coarse_group.len());
            let mut refined_offsets = vec![0];

            // We use a dummy ID set (0..N) because we are operating on the slice directly
            let local_ids: Vec<usize> = (0..coarse_group.len()).collect();

            FlatClusters::refine_single_cluster(
                &local_ids,
                &mzs,
                self.mz_tolerance,
                &mut refined_data,
                &mut refined_offsets,
            );

            // 4. Map refined results back to MSNSpectrum batches
            for window in refined_offsets.windows(2) {
                let start = window[0];
                let end = window[1];
                let sub_cluster_indices = &refined_data[start..end];

                if sub_cluster_indices.len() > 1 {
                    // Collect the actual spectra based on the refined indices
                    let batch_spectra: Vec<MSNSpectrum> = sub_cluster_indices
                        .iter()
                        .map(|&idx| coarse_group[idx].clone())
                        .collect();

                    let batch = MassSpectrumBatch {
                        batch: batch_spectra,
                    };

                    if sub_cluster_indices.len() < self.batch_size_threshold {
                        fast_batches.push(batch);
                    } else {
                        slow_batches.push(batch);
                    }
                }
            }
        }

        // Create folder to store spectra
        ensure_empty_dir(TEMP_FOLDER_NAME)?;

        // println!(
        //     "Archiving {} fast batches and {} slow batches.",
        //     fast_batches.len(),
        //     slow_batches.len()
        // );

        // Wrap them in their respective collections and write to separate files
        if !fast_batches.is_empty() {
            let fast_collection = MassSpectrumBatchCollection {
                batch: fast_batches,
            };
            fast_collection.dump(&format!("{TEMP_FOLDER_NAME}/fast_batches.bin"))?;
        }

        if !slow_batches.is_empty() {
            let slow_collection = MassSpectrumBatchCollection {
                batch: slow_batches,
            };
            slow_collection.dump(&format!("{TEMP_FOLDER_NAME}/slow_batches.bin"))?;
        }
        Ok(())
    }

    pub fn split_on_mz_and_charge(
        &mut self,
        spectra: &mut [MSNSpectrum],
    ) -> Result<(), std::io::Error> {
        // 1. Sort spectra
        spectra.par_sort_unstable_by(|spectrum1, spectrum2| {
            (spectrum1.charge(), spectrum1.precursor_mz)
                .partial_cmp(&(spectrum2.charge(), spectrum2.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fast_batches = Vec::new();
        let mut slow_batches = Vec::new();

        // 2. Use chunk_by to group spectra by tolerance/charge logic
        // This replaces the manual 'splits' logic and the index loop.
        for group in spectra.chunk_by(|a, b| {
            if a.charge() != b.charge() {
                return false;
            }

            let m_a = a.get_neutral_mass();
            let m_b = b.get_neutral_mass();
            let gap = (m_b - m_a).abs();

            match self.mz_tolerance {
                MassTolerance::Da(tol) => gap <= tol,
                MassTolerance::Ppm(tol) => {
                    let allowed_gap = m_b * (tol / 1_000_000.0);
                    gap <= allowed_gap
                }
            }
        }) {
            // Skip singletons as per your original logic
            if group.len() > 1 {
                let batch = MassSpectrumBatch {
                    batch: group.to_vec(), // Still an allocation, but cleaner
                };

                if group.len() < self.batch_size_threshold {
                    fast_batches.push(batch);
                } else {
                    slow_batches.push(batch);
                }
            }
        }

        // Create folder to store spectra
        ensure_empty_dir(TEMP_FOLDER_NAME)?;

        // println!(
        //     "Archiving {} fast batches and {} slow batches.",
        //     fast_batches.len(),
        //     slow_batches.len()
        // );

        // Wrap them in their respective collections and write to separate files
        if !fast_batches.is_empty() {
            let fast_collection = MassSpectrumBatchCollection {
                batch: fast_batches,
            };
            fast_collection.dump(&format!("{TEMP_FOLDER_NAME}/fast_batches.bin"))?;
        }

        if !slow_batches.is_empty() {
            let slow_collection = MassSpectrumBatchCollection {
                batch: slow_batches,
            };
            slow_collection.dump(&format!("{TEMP_FOLDER_NAME}/slow_batches.bin"))?;
        }

        Ok(())
    }

    pub fn cluster(&mut self) -> Result<FlatClusters, std::io::Error> {
        let mut cluster_assignments = Vec::new();
        let mut current_clusters_size: usize = 0;

        let hash_size = self.hash_size();
        let epsilon = self.epsilon;
        let fast_file_path = format!("{TEMP_FOLDER_NAME}/fast_batches.bin");
        let start_time = std::time::Instant::now();
        println!("Starting fast batch");

        if let Ok(file) = File::open(&fast_file_path) {
            // Map the file into virtual memory
            let mmap = unsafe { Mmap::map(&file)? };

            // `mmap` implements Deref<Target = [u8]>, so you can pass it straight to rkyv.
            // Because OS memory maps are page-aligned (usually 4KB or 16KB boundaries),
            // this safely satisfies rkyv's strict memory alignment requirements!
            let collection =
                rkyv::access::<ArchivedMassSpectrumBatchCollection, Error>(&mmap).unwrap();

            let start_time = std::time::Instant::now();
            use faer::sparse::{SparseRowMat, Triplet};
            // Adjust import path based on faer 0.24.0 prelude/docs
            use faer::sparse::linalg::matmul::sparse_sparse_matmul;

            let threshold = 1.0 - self.epsilon;
            let batch_results: Vec<_> = if self.exact {
                // 2. Parallel Compute Phase
                collection
                    .batch
                    .par_iter()
                    .map_init(
                        || {
                            // 2. faer Triplet buffer for sparse ingestion
                            let triplets_buffer: Vec<Triplet<usize, usize, f32>> = Vec::new();

                            // 3. The dense N*N result buffer for DBSCAN
                            let bit_matrix_buffer: Vec<usize> = Vec::new();

                            (triplets_buffer, bit_matrix_buffer)
                        },
                        |(triplets_buffer, bit_matrix_buffer), batch| {
                            let archived_spectra = batch.batch.as_slice();

                            let n_spectra = archived_spectra.len();
                            if n_spectra == 2 && MIN_NEIGHBORS == 2 {
                                let mut spectrum1 =
                                    archived_spectra[0].spectrum.to_soa(self.mz_tolerance);
                                spectrum1.l2_normalize();
                                let mut spectrum2 =
                                    archived_spectra[1].spectrum.to_soa(self.mz_tolerance);
                                spectrum2.l2_normalize();

                                let similarity = super::similarities::branching_cosine_sim(
                                    &spectrum1.mz_bins,
                                    &spectrum1.intensities,
                                    &spectrum2.mz_bins,
                                    &spectrum2.intensities,
                                );
                                let id1 = archived_spectra[0].id;
                                let id2 = archived_spectra[1].id;
                                if similarity >= threshold {
                                    return (vec![(0_usize, id1), (0_usize, id2)], 1);
                                } else {
                                    return (vec![], 0);
                                }
                            }

                            let words_per_vertex = (n_spectra + WORDSIZE - 1) / WORDSIZE;
                            // bit_matrix_buffer.clear();
                            // bit_matrix_buffer.resize(n_spectra * words_per_vertex, 0);
                            // let soa_spectra: Vec<crate::MassSpectrumSoA> = archived_spectra
                            //     .iter()
                            //     .map(|spectrum| {
                            //         let mut soa = spectrum.spectrum.to_soa(self.mz_tolerance);
                            //         soa.l2_normalize();
                            //         soa
                            //     })
                            //     .collect();

                            // for i in 0..n_spectra {
                            //     // Self-similarity is always 1.0 after L2 norm
                            //     bit_matrix_buffer[i * words_per_vertex..(i + 1) * words_per_vertex]
                            //         .set_bit(i);

                            //     let mut spectrum1 = &soa_spectra[i];

                            //     for j in (i + 1)..n_spectra {
                            //         let mut spectrum2 = &soa_spectra[j];
                            //         let similarity = super::similarities::branching_cosine_sim(
                            //             &spectrum1.mz_bins,
                            //             &spectrum1.intensities,
                            //             &spectrum2.mz_bins,
                            //             &spectrum2.intensities,
                            //         );

                            //         if similarity >= threshold {
                            //             bit_matrix_buffer
                            //                 [i * words_per_vertex..(i + 1) * words_per_vertex]
                            //                 .set_bit(j);
                            //             bit_matrix_buffer
                            //                 [j * words_per_vertex..(j + 1) * words_per_vertex]
                            //                 .set_bit(i);
                            //         }
                            //     }
                            // }

                            // 1. SETUP & CLEAR BUFFERS
                            triplets_buffer.clear();
                            triplets_buffer.reserve(n_spectra * 50); // tune to your average spectrum density
                            bit_matrix_buffer.clear();
                            let mut max_mz = 0;

                            let words_per_vertex = (n_spectra + WORDSIZE - 1) / WORDSIZE;
                            // We STILL allocate the dense O(N^2) buffer for downstream DBSCAN
                            bit_matrix_buffer.resize(n_spectra * words_per_vertex, 0);

                            for (row_idx, spectrum) in archived_spectra.iter().enumerate() {
                                let mut soa = spectrum.spectrum.to_soa(self.mz_tolerance);
                                soa.l2_normalize();

                                // Compute max_mz in a separate pass — avoids branch in the push loop
                                if let Some(&last) = soa.mz_bins.last() {
                                    if last > max_mz {
                                        max_mz = last;
                                    }
                                }

                                triplets_buffer.extend(
                                    soa.mz_bins
                                        .iter()
                                        .zip(&soa.intensities)
                                        .filter(|(_, v)| v.abs() > 1e-6)
                                        .map(|(&col, &val)| {
                                            Triplet::new(row_idx, col as usize, val)
                                        }),
                                );
                            }

                            // 3. COMPRESS TO FAER SPARSE MATRIX
                            // This groups, sorts, and compresses the triplets into a CSR matrix format.
                            let a_mat = SparseRowMat::try_new_from_triplets(
                                n_spectra,
                                max_mz as usize + 1,
                                triplets_buffer.as_slice(),
                            )
                            .unwrap();

                            let a_col = a_mat.to_col_major().unwrap();
                            let a_ref = a_col.as_ref();
                            let a_transposed = a_mat.transpose();

                            let sparse_similarity =
                                sparse_sparse_matmul(a_ref, a_transposed, 1.0, faer::Par::Seq)
                                    .unwrap();
                            let symbolic = sparse_similarity.symbolic();

                            for col in 0..n_spectra {
                                let col_values = sparse_similarity.val_of_col(col);
                                let row_indices = symbolic.row_idx_of_col(col);
                                for (idx, row) in row_indices.enumerate() {
                                    let val = col_values[idx];
                                    if val >= threshold {
                                        bit_matrix_buffer
                                            [row * words_per_vertex..(row + 1) * words_per_vertex]
                                            .set_bit(col);
                                        bit_matrix_buffer
                                            [col * words_per_vertex..(col + 1) * words_per_vertex]
                                            .set_bit(row);
                                    }
                                }
                            }

                            let mut dbscan_labels = dbscan_bitmatrix(
                                &bit_matrix_buffer,
                                n_spectra,
                                MIN_NEIGHBORS,
                                words_per_vertex,
                            );

                            let mut local_cluster_assignments = Vec::new();
                            let mut max_value = 0;

                            for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate()
                            {
                                match cluster_assignment {
                                    ClusterLabel::Cluster(n) => {
                                        max_value = max_value.max(n + 1);
                                        local_cluster_assignments
                                            .push((n, archived_spectra[index].id));
                                    }
                                    _ => (),
                                };
                            }

                            (local_cluster_assignments, max_value)
                        },
                    )
                    .collect()
            } else {
                collection
                    .batch
                    .par_iter()
                    .map_init(
                        || {
                            // This init closure runs ONCE per Rayon task.
                            // These buffers are reused for every batch this task processes.
                            let spectral_buffer: Vec<f32> = Vec::new();
                            let similarity_matrix_buffer: Vec<f32> = Vec::new(); // Adjust type if needed
                            (spectral_buffer, similarity_matrix_buffer)
                        },
                        |(spectral_buffer, similarity_matrix_buffer), batch| {
                            let archived_spectra = batch.batch.as_slice();
                            let n_spectra = batch.batch.len();

                            spectral_buffer.resize(archived_spectra.len() * hash_size, 0.0);
                            spectral_buffer.fill(0.0);

                            spectral_buffer
                                .chunks_mut(hash_size)
                                .zip(archived_spectra.iter())
                                .for_each(|(buffer_slice, spectrum)| {
                                    self.hasher.into_vector(
                                        spectrum.spectrum.peaks.as_slice(),
                                        true,
                                        buffer_slice,
                                    );
                                });
                            let mut dbscan_labels;
                            #[cfg(all(target_vendor = "apple"))]
                            {
                                unsafe {
                                    similarity_matrix_buffer.resize(n_spectra * n_spectra, 0.0);
                                    cblas_ssyrk(
                                        101,                                   // CblasRowMajor
                                        121, // CblasUpper (only calculate the top-right triangle)
                                        111, // CblasNoTrans (A is Row-Major)
                                        n_spectra as i32, // N: dimension of the result (20,000)
                                        hash_size as i32, // K: shared dimension (800)
                                        1.0, // alpha
                                        spectral_buffer.as_ptr(), // A
                                        hash_size as i32, // lda: row stride (800)
                                        0.0, // beta
                                        similarity_matrix_buffer.as_mut_ptr(), // C
                                        n_spectra as i32, // ldc: row stride (20,000)
                                    );
                                }

                                dbscan_labels = dbscan_upper_triangular(
                                    similarity_matrix_buffer,
                                    n_spectra,
                                    1.0 - self.epsilon,
                                    MIN_NEIGHBORS,
                                );
                            }

                            #[cfg(not(all(target_vendor = "apple")))]
                            {
                                // Compute using single threaded_sgemm
                                compute_similarity_matrix(
                                    similarity_matrix_buffer,
                                    spectral_buffer,
                                    hash_size,
                                );

                                dbscan_labels = dbscan_upper_triangular(
                                    similarity_matrix_buffer,
                                    n_spectra,
                                    1.0 - epsilon,
                                    MIN_NEIGHBORS,
                                );
                            }

                            let mut local_cluster_assignments = Vec::new();
                            let mut max_value = 0;

                            for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate()
                            {
                                match cluster_assignment {
                                    ClusterLabel::Cluster(n) => {
                                        // This is because clusters are zero indexed. This could cause them merge
                                        max_value = max_value.max(n + 1);
                                        // Keep local zero-indexed for now

                                        local_cluster_assignments
                                            .push((n, archived_spectra[index].id));
                                    }
                                    _ => (),
                                };
                            }

                            // Return the local results for this batch
                            (local_cluster_assignments, max_value)
                        },
                    )
                    .collect()
            };

            for (local_cluster_assignments, local_max_value) in batch_results {
                for mut assignment in local_cluster_assignments {
                    assignment.0 += current_clusters_size;
                    cluster_assignments.push(assignment);
                }

                current_clusters_size += local_max_value;
            }
            drop(mmap);
            drop(file);
        }

        // Starting slow batch
        // #[cfg(debug_assertions)]
        println!(
            "Done with fast batch after: {}",
            start_time.elapsed().as_secs_f32()
        );

        // #[cfg(debug_assertions)]
        println!("Starting slow batch");
        let start_time = std::time::Instant::now();
        // After finishing the fast batches fix-up phase:
        let slow_file_path = format!("{TEMP_FOLDER_NAME}/slow_batches.bin");
        if let Ok(file) = File::open(&slow_file_path) {
            // Map the file into virtual memory
            let mmap = unsafe { Mmap::map(&file)? };

            let collection =
                rkyv::access::<ArchivedMassSpectrumBatchCollection, Error>(&mmap).unwrap();

            for batch in collection.batch.iter() {
                // Because batch.batch is an ArchivedVec, we use `.as_slice()` to easily get
                // the length and parallel iterators without moving out of the archive.
                let archived_spectra = batch.batch.as_slice();
                let n_spectra = batch.batch.len();

                let start_time = std::time::Instant::now();
                self.spectral_buffer
                    .resize(archived_spectra.len() * self.hash_size(), 0.0);
                self.spectral_buffer.fill(0.0);

                // 2. Zip the mutable chunks with the batch items in parallel
                self.spectral_buffer
                    .par_chunks_exact_mut(hash_size)
                    .zip(archived_spectra.par_iter())
                    .for_each(|(buffer_slice, spectrum)| {
                        self.hasher.into_vector(
                            spectrum.spectrum.peaks.as_slice(),
                            true,
                            buffer_slice,
                        );
                    });

                let dbscan_labels: Vec<ClusterLabel>;
                if n_spectra > LARGE_BATCH_THRESHOLD {
                    let index = BucketIndex::build(&self.spectral_buffer, hash_size, 64);
                    let edges = index.extract_edges(64, self.epsilon);
                    dbscan_labels = graph_dbscan(n_spectra, &edges, self.epsilon, MIN_NEIGHBORS);
                } else {
                    // Try GPU (packed upper-triangle SYRK) first; fall back to CPU.
                    dbscan_labels = self.compute_dense_dbscan(n_spectra, hash_size);

                }

                let mut max_value = 0;
                for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate() {
                    match cluster_assignment {
                        ClusterLabel::Noise => (),
                        ClusterLabel::Unclassified => (),
                        ClusterLabel::Cluster(n) => {
                            max_value = max_value.max(n + 1);

                            cluster_assignments
                                .push((current_clusters_size + n, archived_spectra[index].id));
                        }
                    };
                }
                current_clusters_size += max_value;
            }
        };

        // Starting slow batch
        // #[cfg(debug_assertions)]
        println!(
            "Done with slow batch after: {}",
            start_time.elapsed().as_secs_f32()
        );

        // let assignments: Vec<(usize, usize)> = cluster_assignments
        //     .into_iter()
        //     .zip(ids.into_iter())
        //     .map(|(assignment, index)| {
        //         (
        //             index.to_native() as usize,
        //             match assignment {
        //                 ClusterLabel::Cluster(n) => n,
        //                 _ => 0,
        //             },
        //         )
        //     })
        //     .collect();
        // 2. Sort by cluster ID so they are contiguous
        // This is the most important step for flattening
        cluster_assignments.par_sort_unstable_by_key(|p| p.0);
        let mut data = Vec::with_capacity(cluster_assignments.len());
        let mut offsets = Vec::with_capacity(cluster_assignments.len() / 5);

        if !cluster_assignments.is_empty() {
            let mut current_cluster = cluster_assignments[0].0;
            offsets.push(0); // Start of the first cluster

            for (cluster_id, spectrum_id) in cluster_assignments {
                if cluster_id != current_cluster {
                    // We've hit a new cluster; record the offset
                    offsets.push(data.len());
                    current_cluster = cluster_id;
                }
                data.push(spectrum_id.to_native() as usize);
            }
        }
        // let mut inverse_map: Vec<Vec<usize>> = vec![Vec::new(); current_clusters_size];
        // for (index, cluster) in assignments {
        //     inverse_map[cluster].push(index);
        // }

        Ok(FlatClusters { data, offsets })
    }

    /// Dense DBSCAN for a slow batch: tries GPU first (when feature enabled),
    /// falls back to platform-optimised CPU (Apple Accelerate or matrixmultiply).
    fn compute_dense_dbscan(&mut self, n_spectra: usize, hash_size: usize) -> Vec<ClusterLabel> {
        let min_sim = 1.0 - self.epsilon;

        // --- GPU path -----------------------------------------------------------
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            if let Some(ref gpu_ctx) = self.gpu {
                if n_spectra <= gpu_ctx.max_m {
                    let packed =
                        gpu_ctx.compute(&self.spectral_buffer[..n_spectra * hash_size], n_spectra);
                    return dbscan_upper_triangular_packed(
                        &packed,
                        n_spectra,
                        min_sim,
                        MIN_NEIGHBORS,
                    );
                }
            }
        }

        // --- CPU fallback -------------------------------------------------------
        #[cfg(target_vendor = "apple")]
        {
            unsafe {
                self.similarity_matrix_buffer
                    .resize(n_spectra * n_spectra, 0.0);
                cblas_ssyrk(
                    101,
                    121,
                    111,
                    n_spectra as i32,
                    hash_size as i32,
                    1.0,
                    self.spectral_buffer.as_ptr(),
                    hash_size as i32,
                    0.0,
                    self.similarity_matrix_buffer.as_mut_ptr(),
                    n_spectra as i32,
                );
            }
            return dbscan_upper_triangular(
                &self.similarity_matrix_buffer,
                n_spectra,
                min_sim,
                MIN_NEIGHBORS,
            );
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            compute_similarity_matrix_parallel(&mut self.similarity_matrix_buffer, &self.spectral_buffer, hash_size);

            dbscan_upper_triangular(

                &self.similarity_matrix_buffer,
                n_spectra,
                min_sim,
                MIN_NEIGHBORS,
            )
        }
    }

    fn compute_similarity_matrix(&mut self) {
        let number_of_spectra = self.spectral_buffer.len() / self.hash_size();
        if number_of_spectra < 2 {
            return; // Nothing to compare
        }

        let hash_size = self.hash_size();
        let buffer_slice = &self.spectral_buffer;

        // 1. Calculate the exact size of the 1D upper triangular matrix: N * (N - 1) / 2
        let total_pairs = (number_of_spectra * (number_of_spectra - 1)) / 2;

        self.similarity_matrix_buffer.resize(total_pairs, 0.0);
        self.similarity_matrix_buffer.fill(0.0);

        // 3. Sequentially chunk the mutable slice into the exact lengths needed for each row.
        // Row 0 needs (N - 1) elements, Row 1 needs (N - 2), etc.
        let mut row_slices: Vec<&mut [f32]> = Vec::with_capacity(number_of_spectra);
        let mut remaining = self.similarity_matrix_buffer.as_mut_slice();

        for i in 0..number_of_spectra {
            let row_len = number_of_spectra - i - 1;
            if row_len == 0 {
                break; // The last row has no elements to its right
            }
            let (row, rest) = remaining.split_at_mut(row_len);
            row_slices.push(row);
            remaining = rest;
        }

        // 4. Iterate over the mutable row slices in parallel, writing results directly
        // into the pre-allocated memory without any intermediate allocations.
        row_slices
            .into_par_iter()
            .enumerate()
            .for_each(|(i, row_slice)| {
                let start_i = i * hash_size;
                let end_i = start_i + hash_size;
                let vec_i = &buffer_slice[start_i..end_i];

                // Inner loop remains sequential, mapping directly to `row_slice[offset]`
                for (offset, j) in (i + 1..number_of_spectra).enumerate() {
                    let start_j = j * hash_size;
                    let end_j = start_j + hash_size;
                    let vec_j = &buffer_slice[start_j..end_j];

                    row_slice[offset] = fast_dot_product(vec_i, vec_j)
                }
            });
    }
    pub fn compute_similarity_matrix_parallel(&mut self) {
        let k = self.hash_size(); // Number of features (columns in A)
        let m = self.spectral_buffer.len() / k; // Number of spectra (rows in A)
        let n = m; // Output matrix C is M x M

        if m < 2 {
            return;
        }

        self.similarity_matrix_buffer.resize(m * n, 0.0);
        let a_slice = &self.spectral_buffer;

        // let mat_a = Mat::from(self.spectral_buffer);

        // 2. Define how many rows of the output matrix each thread should compute.
        // 64 or 128 is usually a good sweet spot to amortize thread overhead
        // while keeping CPU cache usage efficient.
        let rows_per_chunk = 64;

        // 3. Process horizontal bands of the output matrix in parallel
        self.similarity_matrix_buffer
            .par_chunks_mut(rows_per_chunk * n)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                // How many rows are we actually computing?
                // (The last chunk might be smaller than rows_per_chunk)
                let current_m = c_chunk.len() / n;

                // Which row of matrix A does this chunk start on?
                let start_row = chunk_idx * rows_per_chunk;

                unsafe {
                    // LHS pointer: Jump to the specific rows in A that this thread is responsible for.
                    let a_lhs_ptr = a_slice.as_ptr().add(start_row * k);

                    // RHS pointer: We need the *entirety* of A to act as our transposed right-hand side.
                    let a_rhs_ptr = a_slice.as_ptr();

                    sgemm(
                        current_m,
                        k,
                        n,   // Dimensions for this specific block
                        1.0, // alpha
                        // LHS Matrix (Sub-matrix of A)
                        a_lhs_ptr,
                        k as isize, // Row stride
                        1,          // Col stride
                        // RHS Matrix (Entirety of A, transposed on the fly)
                        a_rhs_ptr,
                        1,          // Row stride (swapped to transpose)
                        k as isize, // Col stride (swapped to transpose)
                        0.0,        // beta
                        // Output Matrix (Our safe mutable chunk of C)
                        c_chunk.as_mut_ptr(),
                        n as isize, // Row stride for C is the full width 'n'
                        1,          // Col stride
                    );
                }
            });

        // Optional: Pack the full M x M matrix down into your 1D upper-triangular buffer
        // self.pack_upper_triangle(&c_matrix, m);
    }
}

fn ensure_empty_dir<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    let path = path.as_ref();

    // Step 1: Attempt to delete the directory and all its contents.
    match std::fs::remove_dir_all(path) {
        Ok(_) => {
            #[cfg(debug_assertions)]
            println!("Existing folder deleted.")
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            #[cfg(debug_assertions)]
            println!("Folder does not exit yet. Creating folder.")
        }
        Err(e) => return Err(e), // Bubble up any other errors (like permission denied)
    }

    // Step 2: Create a fresh, empty directory
    std::fs::create_dir_all(path)?;
    Ok(())
}

pub fn fast_dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Standard length check
    assert_eq!(a.len(), b.len());

    // Wrap the raw slices in zero-copy faer Matrix views (N rows, 1 column)
    let a_mat = MatRef::from_column_major_slice(a, a.len(), 1);
    let b_mat = MatRef::from_column_major_slice(b, b.len(), 1);

    // Compute A^T * B.
    // Mathematically, a (1 x N) matrix times an (N x 1) matrix is a dot product.
    // faer intercepts this operation and routes it directly to its SIMD micro-kernels.
    let c = a_mat.transpose() * b_mat;

    // The result is a 1x1 matrix. Extract the scalar float.
    c[(0, 0)]
}

pub fn calculate_cluster_recall(exact_labels: &[ClusterLabel], lsh_labels: &[ClusterLabel]) -> f32 {
    assert_eq!(exact_labels.len(), lsh_labels.len());
    let n = exact_labels.len();

    let mut true_positives = 0;
    let mut false_negatives = 0;

    // Note: This is an O(N^2) evaluation, so it's meant for your test suite,
    // not your production pipeline!
    for i in 0..n {
        for j in (i + 1)..n {
            // Check if they are in the exact same valid cluster (ignore noise for this metric)
            let exact_same = match (exact_labels[i], exact_labels[j]) {
                (ClusterLabel::Cluster(id1), ClusterLabel::Cluster(id2)) => id1 == id2,
                _ => false,
            };

            if exact_same {
                let lsh_same = match (lsh_labels[i], lsh_labels[j]) {
                    (ClusterLabel::Cluster(id1), ClusterLabel::Cluster(id2)) => id1 == id2,
                    _ => false,
                };

                if lsh_same {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }
    }

    let total_true_pairs = true_positives + false_negatives;
    if total_true_pairs == 0 {
        return 1.0;
    }

    true_positives as f32 / total_true_pairs as f32
}

// #[test]
// fn test_clustering_read() {
//     use crate::clustering::ms_rt::*;
//     let start_time = std::time::Instant::now();
//     let config = MgfParserConfig::default();
//     let mut parser = MgfParser::new(config);
//     let mut msn_spectra = Vec::new();

//     let mut spectral_experiment = SpectralExperiment::default();
//     // spectral_experiment.workflow.add_operation(op);
//     let operations: Vec<Box<dyn Operation>> = vec![FilterMZ::new(101.0, 1500.0)];

//     let mut manager = WorkFlowManager::new();
//     manager.add_operation(FilterMZ::new(101.0, 1500.0));
//     manager.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
//     manager.add_operation(FilterIntensityRelative::new(0.01));
//     manager.add_operation(KeepTopNPeaks::new(50));
//     manager.add_operation(MinNumberOfPeaks::new(5));
//     manager.add_operation(MinMZRangeFilter::new(250.0));

//     let mut total_spectra = 0;
//     let mut file_boundaries = Vec::new();
//     let folder_name = "pxd_tests/PXD023047";
//     // let folder_name = "/Users/manu/phd_thesis/projects/ms_tools/PXD_converted";
//     for file in std::fs::read_dir(folder_name).unwrap() {
//         let file_name = file.unwrap().path();
//         let mut temp_spectra = parser.parse_file(file_name.to_str().unwrap()).unwrap();
//         // let file_name = "test_files/ALL_GNPS.mgf";
//         // let mut temp_spectra = parser.parse_file("test_files/ALL_GNPS.mgf").unwrap();
//         // println!("Number of spectra parsed: {}", temp_spectra.len());
//         total_spectra += temp_spectra.len();

//         file_boundaries.push((file_name.to_str().unwrap().to_owned(), total_spectra));
//         // file_boundaries.push((file_name.to_owned(), total_spectra));
//         manager.execute(&mut temp_spectra);
//         let temp_msn_spectra: Vec<MSNSpectrum> = temp_spectra
//             .iter()
//             .filter(|spectrum| !spectrum.peaks().is_empty())
//             .map(|spectrum| spectrum.to_msn())
//             .collect();

//         msn_spectra.extend_from_slice(&temp_msn_spectra);
//     }
//     println!(
//         "Done parsing and processing spectra after {}s",
//         start_time.elapsed().as_secs_f32()
//     );

//     let start_time = std::time::Instant::now();

//     let mz_tolerance = MassTolerance::Ppm(20.0);
//     let mut manager = ClusterEngine::default(); //, Some(msn_spectra.len()));
//     manager.epsilon = 0.1;

//     manager.split_on_mz_and_charge(&mut msn_spectra);

//     println!(
//         "Done with splitting spectra and writing temp files: {}s",
//         start_time.elapsed().as_secs_f32()
//     );

//     let start_time = std::time::Instant::now();
//     let clusters = manager.cluster().unwrap();

//     println!(
//         "Done with clustering spectra and writing temp files: {}s",
//         start_time.elapsed().as_secs_f32()
//     );

//     let mut count = 0;
//     let mut clustered_spectra = 0;
//     for clusterlabel in clusters.iter() {
//         if clusterlabel.len() < 2 {
//             println!("Singleton cluster");
//             continue;
//         }
//         count += 1;
//         clustered_spectra += clusterlabel.len();
//     }

//     let average_cluster_size = clusters.avg_cluster_size();
//     let min_cluster_size = clusters.min_cluster_size();
//     let max_cluster_size = clusters.max_cluster_size();

//     println!("Number of clusters found: {}", count);
//     println!("Smallest cluster size:  {}", min_cluster_size);
//     println!("Average cluster size:  {}", average_cluster_size);
//     println!("Largest cluster size: {}", max_cluster_size);
//     println!("Number of spectra read: {}", total_spectra);
//     println!(
//         "Number of spectra processed: {} ({:.0}%)",
//         msn_spectra.len(),
//         msn_spectra.len() * 100 / total_spectra
//     );
//     println!(
//         "Number of spectra clustered: {} ({:.0}%)",
//         clustered_spectra,
//         clustered_spectra * 100 / total_spectra
//     );

//     let spectral_map: IntMap<usize, &MSNSpectrum> = msn_spectra
//         .iter()
//         .map(|spectrum| (spectrum.id as usize, spectrum))
//         .collect();
//     let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();

//     plot_cluster_distribution(
//         &mut cluster_sizes,
//         total_spectra,
//         "cluster_distributions.svg",
//     );

//     plot_cluster_size_distribution(&mut cluster_sizes, "cluster_size_distributions.svg");
//     let mut cluster_purities: Vec<f64> = clusters
//         .iter()
//         .map(|cluster| {
//             calculate_cluster_purity(cluster, &spectral_map, &file_boundaries, 0.01, 65.0)
//         })
//         .collect();

//     let results = &cluster_sizes
//         .into_iter()
//         .zip(cluster_purities.into_iter())
//         .collect::<Vec<(usize, f64)>>();

//     let stats = bin_results(&results);
//     println!(
//         "Average weighted purity: {}",
//         calculate_average_purity(&results)
//     );
//     plot_msrt_benchmarks(&stats, "purity_plot.png");

//     let mz_values: Vec<f64> = clusters
//         .data
//         .iter()
//         .map(|id| spectral_map.get(id).unwrap().precursor_mz)
//         .collect();

//     let clusters = clusters.refine_mz_clusters(&mz_values, mz_tolerance);

//     let mut count = 0;
//     let mut clustered_spectra = 0;
//     for clusterlabel in clusters.iter() {
//         if clusterlabel.len() < 2 {
//             println!("Singleton cluster");
//             continue;
//         }
//         count += 1;
//         clustered_spectra += clusterlabel.len();
//     }

//     let average_cluster_size = clusters.avg_cluster_size();
//     let min_cluster_size = clusters.min_cluster_size();
//     let max_cluster_size = clusters.max_cluster_size();
//     println!("Number of clusters found: {}", count);
//     println!("Smallest cluster size:  {}", min_cluster_size);
//     println!("Average cluster size:  {}", average_cluster_size);
//     println!("Largest cluster size: {}", max_cluster_size);
//     println!("Number of spectra read: {}", total_spectra);
//     println!(
//         "Number of spectra processed: {} ({:.0}%)",
//         msn_spectra.len(),
//         msn_spectra.len() * 100 / total_spectra
//     );
//     println!(
//         "Number of spectra clustered: {} ({:.0}%)",
//         clustered_spectra,
//         clustered_spectra * 100 / total_spectra
//     );

//     let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();

//     plot_cluster_distribution(
//         &mut cluster_sizes,
//         total_spectra,
//         "cluster_distributions_after_refinement.svg",
//     );

//     plot_cluster_size_distribution(
//         &mut cluster_sizes,
//         "cluster_size_distributions_after_refinement.svg",
//     );
//     let mut cluster_purities: Vec<f64> = clusters
//         .iter()
//         .map(|cluster| {
//             calculate_cluster_purity(cluster, &spectral_map, &file_boundaries, 0.01, 65.0)
//         })
//         .collect();

//     let results = &cluster_sizes
//         .into_iter()
//         .zip(cluster_purities.into_iter())
//         .collect::<Vec<(usize, f64)>>();

//     let stats = bin_results(&results);
//     println!(
//         "Average weighted purity after refinement: {}",
//         calculate_average_purity(&results)
//     );
//     plot_msrt_benchmarks(&stats, "purity_plot_after_refinement.png");
// }

fn compute_similarity_matrix(
    similarity_matrix: &mut Vec<f32>,
    spectral_buffer: &[f32],
    hash_size: usize,
) {
    let k = hash_size; // Number of features (columns in A)
    let m = spectral_buffer.len() / k; // Number of spectra (rows in A)
    let n = m; // Output matrix C is M x M

    if m < 2 {
        return;
    }

    similarity_matrix.clear();
    similarity_matrix.reserve(m * n);

    unsafe {
        similarity_matrix.set_len(m * n);
    }

    let a = MatRef::from_row_major_slice(spectral_buffer, m, k);
    let mut c = MatMut::from_row_major_slice_mut(similarity_matrix, m, n);
    // Create lightweight, zero-copy views

    // Compute only the upper triangle of C = A * A^T
    triangular_matmul(
        c,
        BlockStructure::TriangularUpper, // Match CblasUpper
        faer::Accum::Replace,            // beta = 0.0
        a,                               // A
        BlockStructure::Rectangular,
        a.transpose(), // A^T
        BlockStructure::Rectangular,
        1.0,            // alpha
        faer::Par::Seq, // Multithreading
    );
}

fn compute_similarity_matrix_parallel(
    similarity_matrix: &mut Vec<f32>,
    spectral_buffer: &[f32],
    hash_size: usize,
) {
    let k = hash_size; // Number of features (columns in A)
    let m = spectral_buffer.len() / k; // Number of spectra (rows in A)
    let n = m; // Output matrix C is M x M

    if m < 2 {
        return;
    }

    similarity_matrix.clear();
    similarity_matrix.reserve(m * n);

    unsafe {
        similarity_matrix.set_len(m * n);
    }

    let a = MatRef::from_row_major_slice(spectral_buffer, m, k);
    let mut c = MatMut::from_row_major_slice_mut(similarity_matrix, m, n);
    // Create lightweight, zero-copy views

    // Compute only the upper triangle of C = A * A^T
    triangular_matmul(
        c,
        BlockStructure::TriangularUpper, // Match CblasUpper
        faer::Accum::Replace,            // beta = 0.0
        a,                               // A
        BlockStructure::Rectangular,
        a.transpose(), // A^T
        BlockStructure::Rectangular,
        1.0,                 // alpha
        faer::Par::rayon(0), // Multithreading
    );
}
