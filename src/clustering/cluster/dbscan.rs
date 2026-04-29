use crate::clustering::bitset::{BitSetOpsMSB0, WORDSIZE};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterLabel {
    Unclassified,
    Noise,
    Cluster(usize),
}

#[inline]
fn upper_triangular_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j, "i must be strictly less than j");
    (i * (2 * n - i - 1)) / 2 + (j - i - 1)
}

fn get_similarity(matrix: &[f32], n: usize, p: usize, q: usize) -> f32 {
    if p == q {
        return 1.0;
    }
    let (i, j) = if p < q { (p, q) } else { (q, p) };
    let idx = upper_triangular_index(n, i, j);
    matrix[idx]
}

fn get_similarity_square(
    matrix: &[f32],
    matrix_size: usize,
    row_index: usize,
    column_index: usize,
) -> f32 {
    let idx = row_index * matrix_size + column_index;
    matrix[idx]
}

fn get_similarity_upper(matrix: &[f32], num_points: usize, row: usize, col: usize) -> f32 {
    let (r, c) = if row <= col { (row, col) } else { (col, row) };
    let idx = r * num_points + c;
    matrix[idx]
}

/// Packed upper-triangular index: for row r ≤ col c in an n×n matrix.
#[inline]
pub fn packed_upper_idx(r: usize, c: usize, n: usize) -> usize {
    r * (2 * n - r - 1) / 2 + c
}

struct DSU {
    parent: Vec<i32>,
}

impl DSU {
    fn new(size: usize) -> Self {
        Self {
            parent: vec![-1; size],
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] < 0 {
            i
        } else {
            let root = self.find(self.parent[i] as usize);
            self.parent[i] = root as i32;
            root
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            if self.parent[root_i] < self.parent[root_j] {
                self.parent[root_j] = root_i as i32;
            } else {
                if self.parent[root_i] == self.parent[root_j] {
                    self.parent[root_j] -= 1;
                }
                self.parent[root_i] = root_j as i32;
            }
        }
    }
}

/// Standard DBSCAN clustering on a full (dense) similarity matrix.
///
/// Points with fewer than `min_pts` neighbors (similarity ≥ `min_similarity`) are
/// labelled `Noise`. Remaining unvisited points are seeded as new clusters and expanded
/// via BFS-like seed-set growth.
///
/// Complexity: O(n²) — suitable for small spectra sets. For large datasets use
/// [`dbscan_bitmatrix`] (bitset version) or [`graph_dbscan`] (index-based version).
pub fn dbscan(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;

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

            for &n in &neighbors {
                if n == p {
                    continue;
                }

                if labels[n] == ClusterLabel::Noise {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                } else if labels[n] == ClusterLabel::Unclassified {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    seed_set.push(n);
                }
            }

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

/// DBSCAN on a full (square) similarity matrix.
///
/// Like [`dbscan`] but reads similarities from a row-major square matrix
/// (`similarity_matrix[p * num_points + q]`). Cluster IDs start at 0.
///
/// Use when the matrix was computed as a full `n×n` layout rather than
/// packed upper-triangular.
pub fn dbscan_full(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
    min_pts: usize,
) -> Vec<ClusterLabel> {
    let mut labels = vec![ClusterLabel::Unclassified; num_points];
    let mut current_cluster_id = 0;

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
            for &n in &neighbors {
                if n == p {
                    continue;
                }

                if labels[n] == ClusterLabel::Noise {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                } else if labels[n] == ClusterLabel::Unclassified {
                    labels[n] = ClusterLabel::Cluster(current_cluster_id);
                    seed_set.push(n);
                }
            }

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

/// Single-linkage clustering on an upper-triangular similarity matrix.
///
/// No `min_pts` threshold — any pair with similarity ≥ `min_similarity` is merged into
/// the same cluster via union-find. Singletons (no qualifying neighbor) are labelled
/// [`ClusterLabel::Noise`].
///
/// Reads from a row-major square matrix where only the upper triangle is populated
/// (`similarity_matrix[i * num_points + j]` for `j ≥ i`).
pub fn dbscan_single_linkage_upper_triangular(
    similarity_matrix: &[f32],
    num_points: usize,
    min_similarity: f32,
) -> Vec<ClusterLabel> {
    let mut parent: Vec<usize> = (0..num_points).collect();
    let mut size = vec![1; num_points];

    let mut find = |mut i: usize, parent: &mut [usize]| -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    };

    let mut link = |mut i: usize, mut j: usize, parent: &mut [usize], size: &mut [usize]| {
        i = find(i, parent);
        j = find(j, parent);
        if i != j {
            if size[i] < size[j] {
                std::mem::swap(&mut i, &mut j);
            }
            parent[j] = i;
            size[i] += size[j];
        }
    };

    for i in 0..num_points {
        let row_offset = i * num_points;
        for j in (i + 1)..num_points {
            if similarity_matrix[row_offset + j] >= min_similarity {
                link(i, j, &mut parent, &mut size);
            }
        }
    }

    let mut labels = vec![ClusterLabel::Noise; num_points];
    let mut next_cluster_id = 0;
    let mut root_to_id = vec![None; num_points];

    for i in 0..num_points {
        let root = find(i, &mut parent);

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
            labels[i] = ClusterLabel::Noise;
        }
    }

    labels
}

/// DBSCAN on an upper-triangular similarity matrix via bitset acceleration.
///
/// Converts the float matrix to a parallel bitmatrix (each row stores neighbors as
/// set bits) then delegates to [`dbscan_bitmatrix`]. Faster than [`dbscan_full`] for
/// larger point sets because neighbor lookup becomes a bitset iteration.
///
/// Reads from a row-major square matrix where only the upper triangle is populated.
pub fn dbscan_upper_triangular(
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
                if similarity_matrix[r * num_points + c] >= min_similarity {
                    row_bits.set_bit(j);
                }
            }
        });
    dbscan_bitmatrix(&bit_matrix, num_points, min_pts, words_per_vertex)
}

/// Core DBSCAN on a precomputed bitmatrix.
///
/// `bit_matrix` is a flat array of `num_points * words_per_vertex` words where row `p`
/// occupies `[p*words_per_vertex .. (p+1)*words_per_vertex]` and bit `q` is set iff
/// the similarity between `p` and `q` met the threshold.
///
/// Neighbor counts use `population_count`; neighbor iteration uses `next_set_bit`.
/// Called by [`dbscan_upper_triangular`] and [`dbscan_upper_triangular_packed`] after
/// they build the bitmatrix.
pub(crate) fn dbscan_bitmatrix(
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

        if p_row.population_count() < min_pts as u32 {
            labels[p] = ClusterLabel::Noise;
            continue;
        }

        labels[p] = ClusterLabel::Cluster(current_cluster_id);
        seed_set.clear();

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

/// DBSCAN on a packed upper-triangular similarity matrix produced by the GPU.
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
