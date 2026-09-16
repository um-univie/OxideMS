use super::dbscan::{
    ClusterLabel, dbscan_bitmatrix, dbscan_upper_triangular, dbscan_upper_triangular_packed,
};
use super::flat_clusters::FlatClusters;
use super::mz_partition::{mass_window_ends, mz_charge_boundaries};
use crate::{
    MSNSpectrum, SpectrumHasher,
    clustering::{
        bitset::{BitSetOpsMSB0, WORDSIZE},
        k_cluster_ivf::{Edge, WINDOWED_EDGES_BLOCK_SIZE, graph_dbscan, windowed_mass_edges},
    },
    workflow::MassTolerance,
};
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::sparse::linalg::matmul::sparse_sparse_matmul;
use faer::sparse::{SparseRowMat, Triplet};
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const MIN_NEIGHBORS: usize = 2;
/// Empirical tail of the cosine similarity between *unrelated* spectra:
/// 3M pairs drawn from different precursor-mass windows of PXD023047 under
/// the proteomics preset. Pairs from different windows can never legitimately
/// cluster, so they are a true null.
///
/// A same-mass, different-identity null was measured too (2M annotated pairs
/// from PXD000561) and is ~40x heavier above 0.9 similarity. It is the more
/// literal null, but calibrating on it produced *worse* clustering at matched
/// completeness (0.392% vs 0.368% error at completeness 0.50, depth 8.7).
/// The reason is that it mixes two populations a threshold cannot treat
/// alike: avoidable false positives, and isobaric sequence variants whose
/// spectra are genuinely near-identical (median cosine 0.952). No threshold
/// separates the latter from true replicates, so including them tells the
/// model to tighten for a problem tightening cannot fix, distorting how tau
/// scales with candidate count. The cross-window null is "too easy" in
/// absolute terms but captures that scaling better.
///
/// Stored as a table, interpolated in log space, so no functional form is
/// assumed.
const NULL_TAIL: [(f32, f32); 9] = [
    (0.500, -3.4522),
    (0.600, -3.7799),
    (0.700, -4.0571),
    (0.800, -4.4089),
    (0.850, -4.6446),
    (0.900, -4.9458),
    (0.925, -5.2730),
    (0.950, -5.6321),
    (0.970, -6.1761),
];

/// Inverts [`NULL_TAIL`]: the similarity at which the null's survival
/// probability falls to `10^target_log10_p`, by linear interpolation in
/// (log10 p, tau). Beyond the sampled range the last segment's slope is
/// extrapolated, which is why callers clamp the result.
fn null_threshold_for(table: &[(f32, f32)], target_log10_p: f32) -> f32 {
    let n = table.len();
    if target_log10_p >= table[0].1 {
        return table[0].0;
    }
    for w in table.windows(2) {
        let ((t0, p0), (t1, p1)) = (w[0], w[1]);
        if target_log10_p <= p0 && target_log10_p >= p1 {
            let f = (p0 - target_log10_p) / (p0 - p1);
            return t0 + f * (t1 - t0);
        }
    }
    // Past the measured tail: continue the final segment.
    let ((t0, p0), (t1, p1)) = (table[n - 2], table[n - 1]);
    let slope = (t1 - t0) / (p1 - p0);
    t1 + (target_log10_p - p1) * slope
}
const LARGE_BATCH_THRESHOLD: usize = 30_000;
const FAST_BATCH_NUMBER: usize = 10000;
/// Below this many spectra the dense path always uses the triangular
/// matmul: per-partition setup dominates when a partition holds a handful of
/// spectra (the median on PXD021518 is 9).
const WINDOWED_DENSE_MIN_SPECTRA: usize = 8192;
/// A partition must span at least this many precursor tolerance widths
/// before block-windowed edges beat the triangular matmul. The triangle
/// computes `n^2 / 2` products; the block scheme computes whole
/// `block x window` rectangles, so it only wins once the mass span is wide
/// enough to cut blocks well below `n / 2`. Astral's largest partitions span
/// 0.12 tolerance widths -- every pair is admissible and there is nothing to
/// skip -- while PXD000561 and PXD021518 partitions span 8 to 13.
const DENSE_SPAN_TOLERANCES: f64 = 4.0;
/// Materialising the `n * n` f32 similarity matrix is capped here; past it
/// the windowed path is used whatever the span, because the allocation
/// itself is the problem (27,832 spectra would need 3.1 GB).
const DENSE_MATRIX_BYTE_BUDGET: usize = 2 << 30;

/// Whether the dense path should build its adjacency from
/// [`windowed_mass_edges`] rather than a full similarity matrix. Both admit
/// exactly the same pairs -- this is purely a cost decision.
fn should_use_windowed_dense(spectra: &[MSNSpectrum], mz_tolerance: MassTolerance) -> bool {
    let n = spectra.len();
    if n < WINDOWED_DENSE_MIN_SPECTRA {
        return false;
    }
    if n.saturating_mul(n).saturating_mul(4) > DENSE_MATRIX_BYTE_BUDGET {
        return true;
    }
    // `spectra` is mass-sorted here, so the span is first to last.
    let lo = spectra[0].get_neutral_mass();
    let hi = spectra[n - 1].get_neutral_mass();
    let tol = mz_tolerance.tolerance_at(hi);
    tol > 0.0 && (hi - lo) / tol >= DENSE_SPAN_TOLERANCES
}

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

/// Groups DBSCAN labels by cluster id, refines each group by precursor m/z, and
/// returns a flat assignment list together with the number of refined clusters produced.
/// Counters for the connectivity re-check in [`collect_and_refine`]:
/// (mass intervals examined, intervals that were not internally connected,
/// spectra dropped because they ended up alone in their component).
static REFINE_INTERVALS: AtomicUsize = AtomicUsize::new(0);
static REFINE_INTERVALS_SPLIT: AtomicUsize = AtomicUsize::new(0);
static REFINE_SPECTRA_DROPPED: AtomicUsize = AtomicUsize::new(0);
/// Diagnostics for *why* a spectrum ends up alone: whether it still has a
/// neighbour elsewhere in its original DBSCAN cluster, and whether any
/// component holding such a neighbour could absorb it without breaking the
/// precursor-mass diameter rule.
static ORPHANS_WITH_NEIGHBOR: AtomicUsize = AtomicUsize::new(0);
static ORPHANS_PLACEABLE: AtomicUsize = AtomicUsize::new(0);
static ORPHANS_ALL_NEIGHBORS_OUT_OF_TOLERANCE: AtomicUsize = AtomicUsize::new(0);

/// Reads and resets the connectivity re-check counters. Diagnostic only --
/// tells you how often the precursor-mass split was leaving intervals whose
/// members are not actually linked to each other.
pub fn take_connectivity_stats() -> (usize, usize, usize) {
    (
        REFINE_INTERVALS.swap(0, Ordering::Relaxed),
        REFINE_INTERVALS_SPLIT.swap(0, Ordering::Relaxed),
        REFINE_SPECTRA_DROPPED.swap(0, Ordering::Relaxed),
    )
}

/// Reads and resets the orphan diagnostics: (orphans that still have a
/// neighbour in their original cluster, orphans some other component could
/// absorb within the mass diameter, orphans whose every neighbour is further
/// away in mass than the tolerance allows).
pub fn take_orphan_stats() -> (usize, usize, usize) {
    (
        ORPHANS_WITH_NEIGHBOR.swap(0, Ordering::Relaxed),
        ORPHANS_PLACEABLE.swap(0, Ordering::Relaxed),
        ORPHANS_ALL_NEIGHBORS_OUT_OF_TOLERANCE.swap(0, Ordering::Relaxed),
    )
}

/// A batch's pairwise links, as they exist at the moment DBSCAN's labels are
/// turned into clusters.
///
/// [`collect_and_refine`] splits each cluster by precursor mass alone, which
/// can leave an interval whose members are not linked to one another -- they
/// shared a DBSCAN cluster only through a spectrum the split moved into a
/// different interval. Checking that requires the adjacency the clustering
/// path already computed, so each path hands over its own representation
/// rather than rebuilding one.
pub(crate) enum BatchLinks<'a> {
    /// Row-major `n * n` similarity matrix with only the upper triangle
    /// populated, as produced by [`compute_similarity_matrix`].
    UpperTriangular {
        similarity: &'a [f32],
        n: usize,
        min_similarity: f32,
    },
    /// Bit matrix; row `p` occupies `[p * words_per_vertex..]` and bit `q` is
    /// set iff `p` and `q` cleared the similarity threshold.
    Bits {
        bits: &'a [usize],
        words_per_vertex: usize,
    },
    /// Sorted, deduplicated adjacency lists built from an edge list.
    Adjacency(Vec<Vec<usize>>),
    /// The path cannot hand over an adjacency in a form this understands
    /// (the GPU packed layout); the connectivity re-check is skipped and
    /// behaviour matches the mass-only split.
    Unavailable,
}

impl BatchLinks<'_> {
    pub(crate) fn from_edges(n: usize, edges: &[Edge]) -> Self {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in edges {
            adj[edge.source_id].push(edge.target_id);
            adj[edge.target_id].push(edge.source_id);
        }
        for list in adj.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        BatchLinks::Adjacency(adj)
    }

    fn linked(&self, i: usize, j: usize) -> bool {
        match self {
            Self::UpperTriangular {
                similarity,
                n,
                min_similarity,
            } => {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                similarity[r * n + c] >= *min_similarity
            }
            Self::Bits {
                bits,
                words_per_vertex,
            } => bits[i * words_per_vertex..(i + 1) * words_per_vertex].is_bit_set(j),
            Self::Adjacency(adj) => adj[i].binary_search(&j).is_ok(),
            Self::Unavailable => false,
        }
    }
}

/// Disjoint-set over `0..n`, used to find the connected components of one
/// mass interval.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// True when `spectra` is in ascending neutral-mass order, which
/// [`windowed_mass_edges`] requires. Batches normally arrive that way from
/// `split_on_mz_and_charge`, but when `rt_tolerance` is set that function
/// re-sorts each m/z group by retention time before splitting it, so the
/// property has to be checked rather than assumed.
fn is_mass_sorted(spectra: &[MSNSpectrum]) -> bool {
    spectra
        .windows(2)
        .all(|w| w[0].get_neutral_mass() <= w[1].get_neutral_mass())
}

/// Dense DBSCAN restricted to pairs whose precursor masses are within
/// tolerance of each other.
///
/// The unrestricted dense path materialises the whole `n * n` similarity
/// matrix and thresholds on cosine similarity alone, so it produces edges
/// between spectra whose precursor masses already disqualify them from
/// sharing a cluster. Those edges are not merely wasted work: at
/// `MIN_NEIGHBORS = 2` a single one is enough to make a core point, letting
/// a cluster expand through a spectrum it should never have reached, and the
/// mass split afterwards can only cut the result up by mass -- it cannot
/// undo the expansion. Measured on PXD021518, 78% of the spectra that ended
/// up unlinked inside their own cluster got there through such an edge.
///
/// [`windowed_mass_edges`] with no dense-window fallback is exactly the same
/// computation restricted to the admissible pairs -- an exact range join,
/// not an approximation -- so this both removes those edges and skips the
/// matmul work they came from. It also never allocates the `n * n` f32
/// matrix; a 27,832-spectrum partition needed 3.1 GB for that alone.
///
/// Returns the labels and the bit matrix's `words_per_vertex`, leaving the
/// adjacency in `bit_matrix` so the caller can hand it to
/// [`collect_and_refine`].
fn windowed_dense_labels(
    spectral_buffer: &[f32],
    hash_size: usize,
    spectra: &[MSNSpectrum],
    mz_tolerance: MassTolerance,
    epsilon: f32,
    bit_matrix: &mut Vec<usize>,
) -> (Vec<ClusterLabel>, usize) {
    let n_spectra = spectra.len();
    let words_per_vertex = n_spectra.div_ceil(WORDSIZE);
    bit_matrix.clear();
    bit_matrix.resize(n_spectra * words_per_vertex, 0);

    let neutral_masses: Vec<f64> = spectra.iter().map(|s| s.get_neutral_mass()).collect();
    let edges = windowed_mass_edges(
        spectral_buffer,
        hash_size,
        &neutral_masses,
        mz_tolerance,
        epsilon,
        WINDOWED_EDGES_BLOCK_SIZE,
        // No IVF fallback: this path must stay exact.
        usize::MAX,
    );

    for edge in &edges {
        bit_matrix[edge.source_id * words_per_vertex..(edge.source_id + 1) * words_per_vertex]
            .set_bit(edge.target_id);
        bit_matrix[edge.target_id * words_per_vertex..(edge.target_id + 1) * words_per_vertex]
            .set_bit(edge.source_id);
    }
    // The full-matrix path sets the diagonal too (a normalized vector's
    // self-similarity is 1.0), and `dbscan_bitmatrix` counts it towards
    // `min_pts`, so set it here for the same neighbour counts.
    for i in 0..n_spectra {
        bit_matrix[i * words_per_vertex..(i + 1) * words_per_vertex].set_bit(i);
    }

    (
        dbscan_bitmatrix(bit_matrix, n_spectra, MIN_NEIGHBORS, words_per_vertex),
        words_per_vertex,
    )
}

/// Builds the DBSCAN adjacency from a full similarity matrix, admitting only
/// pairs within the precursor-mass window.
///
/// The mass filter is what [`windowed_dense_labels`] achieves by not
/// computing the inadmissible pairs at all; for a partition small enough
/// that the triangular matmul is the cheaper way to get the similarities,
/// this applies the identical rule to its output instead. Restricting the
/// scan to `i + 1..window_ends[i]` also makes the bit-setting pass
/// proportional to local density rather than to `n^2`.
fn dense_labels_mass_filtered(
    similarity_matrix: &[f32],
    n_spectra: usize,
    min_similarity: f32,
    window_ends: &[usize],
    bit_matrix: &mut Vec<usize>,
) -> (Vec<ClusterLabel>, usize) {
    let words_per_vertex = n_spectra.div_ceil(WORDSIZE);
    bit_matrix.clear();
    bit_matrix.resize(n_spectra * words_per_vertex, 0);

    for i in 0..n_spectra {
        bit_matrix[i * words_per_vertex..(i + 1) * words_per_vertex].set_bit(i);
        for j in (i + 1)..window_ends[i] {
            if similarity_matrix[i * n_spectra + j] >= min_similarity {
                bit_matrix[i * words_per_vertex..(i + 1) * words_per_vertex].set_bit(j);
                bit_matrix[j * words_per_vertex..(j + 1) * words_per_vertex].set_bit(i);
            }
        }
    }

    (
        dbscan_bitmatrix(bit_matrix, n_spectra, MIN_NEIGHBORS, words_per_vertex),
        words_per_vertex,
    )
}

fn collect_and_refine(
    dbscan_labels: Vec<ClusterLabel>,
    spectra: &[MSNSpectrum],
    mz_tolerance: MassTolerance,
    links: &BatchLinks,
) -> (Vec<(usize, u64)>, usize) {
    let max_label = dbscan_labels
        .iter()
        .filter_map(|l| {
            if let ClusterLabel::Cluster(n) = l {
                Some(n + 1)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0);

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_label];
    for (idx, label) in dbscan_labels.into_iter().enumerate() {
        if let ClusterLabel::Cluster(n) = label {
            groups[n].push(idx);
        }
    }

    let mut assignments: Vec<(usize, u64)> = Vec::new();
    let mut cluster_count = 0usize;
    let mut sub_data: Vec<usize> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    let mut cluster_mzs: Vec<f64> = Vec::new();
    let mut component_of: Vec<usize> = Vec::new();
    let mut group_components: Vec<Vec<usize>> = Vec::new();
    // Batch index -> position within the interval currently being checked;
    // `u32::MAX` means "not in it". Only the entries an interval touches are
    // written and cleared, so this stays O(interval) per interval.
    let mut position_in_interval: Vec<u32> = vec![u32::MAX; spectra.len()];

    for group in groups {
        if group.len() <= 1 {
            continue;
        }
        sub_data.clear();
        sub_offsets.clear();
        sub_offsets.push(0);

        cluster_mzs.clear();
        cluster_mzs.extend(group.iter().map(|&i| spectra[i].precursor_mz));

        FlatClusters::refine_single_cluster(
            &group,
            &cluster_mzs,
            mz_tolerance,
            &mut sub_data,
            &mut sub_offsets,
        );

        for window in sub_offsets.windows(2) {
            let members = &sub_data[window[0]..window[1]];
            REFINE_INTERVALS.fetch_add(1, Ordering::Relaxed);

            if matches!(links, BatchLinks::Unavailable) {
                for &batch_idx in members {
                    assignments.push((cluster_count, spectra[batch_idx].id));
                }
                cluster_count += 1;
                continue;
            }

            // The mass split cut this cluster into intervals without looking
            // at similarity at all, so an interval can hold members that
            // were only ever connected through a spectrum now in a different
            // interval. Emit the interval's connected components instead of
            // the interval itself.
            let k = members.len();
            let mut uf = UnionFind::new(k);
            match links {
                BatchLinks::Adjacency(adj) => {
                    for (p, &member) in members.iter().enumerate() {
                        position_in_interval[member] = p as u32;
                    }
                    for (p, &member) in members.iter().enumerate() {
                        for &neighbor in &adj[member] {
                            let q = position_in_interval[neighbor];
                            if q != u32::MAX {
                                uf.union(p, q as usize);
                            }
                        }
                    }
                    for &member in members {
                        position_in_interval[member] = u32::MAX;
                    }
                }
                _ => {
                    for p in 0..k {
                        for q in (p + 1)..k {
                            if links.linked(members[p], members[q]) {
                                uf.union(p, q);
                            }
                        }
                    }
                }
            }

            component_of.clear();
            component_of.resize(k, usize::MAX);
            let mut components: Vec<Vec<usize>> = Vec::new();
            for p in 0..k {
                let root = uf.find(p);
                let slot = &mut component_of[root];
                if *slot == usize::MAX {
                    *slot = components.len();
                    components.push(Vec::new());
                }
                components[*slot].push(members[p]);
            }

            if components.len() > 1 {
                REFINE_INTERVALS_SPLIT.fetch_add(1, Ordering::Relaxed);
            }
            group_components.append(&mut components);
        }

        // An orphan is a spectrum left alone once its interval was cut into
        // components. It was in this DBSCAN cluster because it had at least
        // one edge (MIN_NEIGHBORS = 2 makes one edge enough to be a core
        // point), so the interesting question is whether some other
        // component of the same cluster -- one holding a spectrum it is
        // actually linked to -- could take it without breaking the
        // precursor-mass diameter rule.
        for (ci, component) in group_components.iter().enumerate() {
            if component.len() != 1 {
                continue;
            }
            let orphan = component[0];
            let orphan_mz = spectra[orphan].precursor_mz;
            let mut has_neighbor = false;
            let mut placeable = false;
            let mut any_neighbor_in_tolerance = false;

            for (cj, other) in group_components.iter().enumerate() {
                if cj == ci {
                    continue;
                }
                let linked_here = other.iter().any(|&m| links.linked(orphan, m));
                if !linked_here {
                    continue;
                }
                has_neighbor = true;
                if other
                    .iter()
                    .any(|&m| {
                        let d = (spectra[m].precursor_mz - orphan_mz).abs();
                        d <= mz_tolerance.tolerance_at(spectra[m].precursor_mz.max(orphan_mz))
                    })
                {
                    any_neighbor_in_tolerance = true;
                }
                let lo = other
                    .iter()
                    .map(|&m| spectra[m].precursor_mz)
                    .fold(orphan_mz, f64::min);
                let hi = other
                    .iter()
                    .map(|&m| spectra[m].precursor_mz)
                    .fold(orphan_mz, f64::max);
                if hi - lo <= mz_tolerance.tolerance_at(hi) {
                    placeable = true;
                }
            }

            if has_neighbor {
                ORPHANS_WITH_NEIGHBOR.fetch_add(1, Ordering::Relaxed);
            }
            if placeable {
                ORPHANS_PLACEABLE.fetch_add(1, Ordering::Relaxed);
            }
            if has_neighbor && !any_neighbor_in_tolerance {
                ORPHANS_ALL_NEIGHBORS_OUT_OF_TOLERANCE.fetch_add(1, Ordering::Relaxed);
            }
        }

        for component in group_components.drain(..) {
            // A singleton is a spectrum the mass split had grouped with
            // spectra it has no link to; it is noise, not a cluster.
            if component.len() <= 1 {
                REFINE_SPECTRA_DROPPED.fetch_add(component.len(), Ordering::Relaxed);
                continue;
            }
            for batch_idx in component {
                assignments.push((cluster_count, spectra[batch_idx].id));
            }
            cluster_count += 1;
        }
    }

    (assignments, cluster_count)
}

/// Performs in-place L2 normalization on a flattened buffer of spectral vectors.
///
/// Each spectrum is assumed to have a fixed length defined by `stride`.
/// Normalization is computed as $v_i = \frac{v_i}{\|v\|_2}$. If the norm is
/// less than [`f32::EPSILON`], the vector is left unchanged to avoid
/// division by zero.
///
/// This is typically a prerequisite for computing cosine similarity via
/// dot product.
pub fn normalize_spectra_in_place(flat_data: &mut [f32], stride: usize) {
    for spectrum in flat_data.chunks_mut(stride) {
        let norm_sq: f32 = spectrum.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();

        if norm > std::f32::EPSILON {
            for x in spectrum.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Computes the dot product of two vectors using optimized linear algebra routines.
///
/// This function leverages the `faer` library to perform the calculation,
/// which may utilize SIMD instructions and other hardware-specific
/// optimizations.
///
/// # Panics
///
/// Panics if the lengths of `a` and `b` are not equal.
pub fn fast_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let a_mat = MatRef::from_column_major_slice(a, a.len(), 1);
    let b_mat = MatRef::from_column_major_slice(b, b.len(), 1);

    let c = a_mat.transpose() * b_mat;
    c[(0, 0)]
}

/// Computes the upper triangular similarity matrix for a set of spectra.
///
/// This calculates $C = A \times A^T$ where $A$ is the row-major `spectral_buffer`.
/// The computation is performed sequentially. Only the upper triangle of the
/// output matrix is populated to save computation time and memory.
///
/// The `spectral_buffer` should contain normalized vectors to ensure the
/// resulting matrix represents cosine similarities.
pub(crate) fn compute_similarity_matrix(
    similarity_matrix: &mut Vec<f32>,
    spectral_buffer: &[f32],
    hash_size: usize,
) {
    let k = hash_size;
    let m = spectral_buffer.len() / k;
    let n = m;

    if m < 2 {
        return;
    }

    similarity_matrix.clear();
    similarity_matrix.reserve(m * n);

    unsafe {
        similarity_matrix.set_len(m * n);
    }

    let a = MatRef::from_row_major_slice(spectral_buffer, m, k);
    let c = MatMut::from_row_major_slice_mut(similarity_matrix, m, n);

    triangular_matmul(
        c,
        BlockStructure::TriangularUpper,
        faer::Accum::Replace,
        a,
        BlockStructure::Rectangular,
        a.transpose(),
        BlockStructure::Rectangular,
        1.0,
        faer::Par::Seq,
    );
}

/// Computes the upper triangular similarity matrix for a set of spectra in parallel.
///
/// Similar to [`compute_similarity_matrix`], but utilizes Rayon-backed parallelism
/// within the `faer` library. This is optimized for large matrices where the
/// overhead of thread synchronization is outweighed by the throughput gains.
pub(crate) fn compute_similarity_matrix_parallel(
    similarity_matrix: &mut Vec<f32>,
    spectral_buffer: &[f32],
    hash_size: usize,
) {
    let k = hash_size;
    let m = spectral_buffer.len() / k;
    let n = m;

    if m < 2 {
        return;
    }

    similarity_matrix.clear();
    similarity_matrix.reserve(m * n);

    unsafe {
        similarity_matrix.set_len(m * n);
    }

    let a = MatRef::from_row_major_slice(spectral_buffer, m, k);
    let c = MatMut::from_row_major_slice_mut(similarity_matrix, m, n);

    triangular_matmul(
        c,
        BlockStructure::TriangularUpper,
        faer::Accum::Replace,
        a,
        BlockStructure::Rectangular,
        a.transpose(),
        BlockStructure::Rectangular,
        1.0,
        faer::Par::rayon(0),
    );
}

/// A container for a batch of mass spectra that are processed together.
///
/// Batches are typically created based on precursor properties to localize
/// similarity searches and improve cache locality.
pub struct MassSpectrumBatch {
    pub(super) batch: Vec<MSNSpectrum>,
}

/// Physical and experimental metadata associated with a mass spectrum.
pub struct SpectrumMetadata {
    /// The precursor mass-to-charge ratio.
    pub precursor_mz: f64,
    /// The time at which the spectrum was acquired.
    pub retention_time_in_seconds: f32,
    /// Index of the source file.
    pub file_index: usize,
    /// The scan identifier within the source file.
    pub scan_number: usize,
    /// The charge state of the precursor ion.
    pub charge: i8,
    /// Byte offset in the source file where this spectrum's block starts.
    /// Lets [`crate::clustering::cluster::experiment::SpectralExperiment::build_consensus_spectrum`]
    /// reparse just this spectrum's peaks on demand, without keeping every
    /// spectrum's peaks resident in RAM for the life of the run.
    pub source_byte_start: u64,
    /// Byte offset in the source file where this spectrum's block ends.
    pub source_byte_end: u64,
}

/// The main engine for orchestrating large-scale spectral clustering.
///
/// `ClusterEngine` manages memory buffers and configurations for high-performance
/// clustering using DBSCAN. It supports different execution paths based on batch
/// size and hardware availability, including:
///
/// - Dense similarity matrices for medium batches.
/// - GPU acceleration via Metal/OpenCL (if enabled).
/// - Exact sparse similarity for high-resolution refinement.
/// - Approximate graph-based clustering for very large datasets ($> 30,000$ spectra).
pub struct ClusterEngine {
    /// Pre-allocated buffer for dense spectral vectors.
    pub spectral_buffer: Vec<f32>,
    /// Pre-allocated buffer for the similarity matrix.
    pub similarity_matrix_buffer: Vec<f32>,
    /// Tolerance for precursor m/z matching.
    pub mz_tolerance: MassTolerance,
    /// The DBSCAN epsilon parameter ($1 - \text{similarity threshold}$).
    pub rt_tolerance: Option<f32>,
    /// Scratch adjacency for the mass-windowed dense path, kept across
    /// batches so each one does not reallocate it.
    dense_bit_matrix: Vec<usize>,
    /// EXPERIMENTAL. When set, each batch derives its own similarity
    /// threshold instead of using `epsilon`, so that the expected number of
    /// spurious links *per spectrum* stays at this value however many
    /// candidates the batch offers.
    ///
    /// A fixed threshold does not survive growing data: with replication
    /// depth `d`, every identity in a precursor window is present in `d`
    /// copies, so a spectrum is compared against `O(d x identities)`
    /// candidates and the expected number of false links grows with the
    /// corpus while the threshold stays put. Measured on PXD000561, the
    /// disagreement rate rises from 0.28% to 0.51% as depth goes 3 -> 16
    /// while corpus growth at *constant* depth leaves it flat.
    ///
    /// The correction is the clustering analogue of a spectral E-value:
    /// holding `candidates * p0(tau) = alpha` gives
    /// `tau = (log10(alpha / candidates) - NULL_LOG10_INTERCEPT) /
    /// NULL_LOG10_SLOPE`.
    pub adaptive_alpha: Option<f32>,
    /// Null similarity tail to invert, as `(tau, log10 p0)` ascending in tau.
    /// Defaults to [`NULL_TAIL`], which was measured under a plain cosine on
    /// hashed intensities. Any change to the similarity -- a rank or binary
    /// intensity transform, a different fragment tolerance -- moves that
    /// distribution, so the table has to be re-measured alongside it or the
    /// derived threshold is meaningless.
    pub null_tail: Option<Vec<(f32, f32)>>,
    pub epsilon: f32,
    /// Maximum size of a batch to be considered "fast".
    pub batch_size_threshold: usize,
    /// Threshold for switching to approximate graph-based clustering.
    pub approximate_threshold: usize,
    /// Component used to vectorize sparse peaks.
    pub hasher: SpectrumHasher,
    /// Whether to use exact peak alignment instead of binning.
    pub exact: bool,
    /// Minimum neighbors for a DBSCAN core point.
    pub min_neighbors: usize,
    /// Enable/disable GPU offloading.
    pub use_gpu: bool,
    /// Print timing and progress information to stdout.
    pub verbose: bool,
    pub(super) fast_batches: Vec<MassSpectrumBatch>,
    pub(super) slow_batches: Vec<MassSpectrumBatch>,
    #[cfg(feature = "gpu")]
    pub(super) gpu: Option<crate::clustering::gpu::GpuSimilarity>,
}

impl Default for ClusterEngine {
    fn default() -> Self {
        let hash_size = 800;
        let eps = 0.1;
        ClusterEngine {
            spectral_buffer: Vec::new(),
            similarity_matrix_buffer: Vec::new(),
            epsilon: eps,
            mz_tolerance: MassTolerance::Ppm(20.0),
            rt_tolerance: None,
            dense_bit_matrix: Vec::new(),
            adaptive_alpha: None,
            null_tail: None,
            batch_size_threshold: FAST_BATCH_NUMBER,
            approximate_threshold: LARGE_BATCH_THRESHOLD,
            hasher: SpectrumHasher::new(101.0, 1500.0, MassTolerance::Da(0.05), hash_size),
            exact: false,
            min_neighbors: MIN_NEIGHBORS,
            use_gpu: false,
            verbose: false,
            fast_batches: Vec::new(),
            slow_batches: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }
}

impl ClusterEngine {
    /// Configures the engine to use GPU acceleration if a compatible device is found.
    ///
    /// # Arguments
    ///
    /// * `max_batch` - The maximum batch size the GPU buffer should allocate for.
    ///   Passing `0` will use [`LARGE_BATCH_THRESHOLD`].
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

    /// Returns the length of the vectorized hash produced by the internal hasher.
    pub fn hash_size(&self) -> usize {
        self.hasher.hash_len
    }

    /// Updates the hash size and re-initializes the internal lookup tables.
    pub fn set_hash_size(&mut self, hash_len: usize) {
        self.hasher.hash_len = hash_len;
        self.hasher.recalculate_lookup_table();
    }

    /// Updates the hasher m/z range and re-initializes the internal lookup tables.
    ///
    /// Must match the `FilterMZ` bounds in the workflow — peaks outside this range
    /// are clamped to edge bins and will produce incorrect similarity scores.
    pub fn set_mz_range(&mut self, min_mz: f64, max_mz: f64) {
        self.hasher.min_mz = min_mz;
        self.hasher.max_mz = max_mz;
        self.hasher.recalculate_lookup_table();
    }

    /// Partitions and refines a set of spectra into manageable batches for clustering.
    ///
    /// This method performs two main steps:
    /// 1. Coarse grouping by charge and precursor m/z within `mz_tolerance`.
    /// 2. Refinement of groups into chains where every spectrum is within tolerance
    ///     of at least one other.
    ///
    /// The resulting batches are sorted into `fast_batches` or `slow_batches`
    /// based on their size.
    pub fn split_and_refine_clusters(
        &mut self,
        spectra: &mut [MSNSpectrum],
    ) -> Result<(), std::io::Error> {
        spectra.par_sort_unstable_by(|a, b| {
            (a.charge(), a.precursor_mz)
                .partial_cmp(&(b.charge(), b.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fast_batches = Vec::new();
        let mut slow_batches = Vec::new();

        // Same grouping rule as `split_on_mz_and_charge` and the low-memory
        // ingest path, from the one place it is defined -- a second inline
        // copy here would silently drift from them.
        let boundaries = mz_charge_boundaries(
            spectra,
            |s: &MSNSpectrum| s.charge(),
            |s: &MSNSpectrum| s.get_neutral_mass(),
            self.mz_tolerance,
        );

        for window in boundaries.windows(2) {
            let coarse_group = &mut spectra[window[0]..window[1]];
            if coarse_group.len() <= 1 {
                continue;
            }

            let mzs: Vec<f64> = coarse_group.iter().map(|s| s.get_neutral_mass()).collect();
            let mut refined_data = Vec::with_capacity(coarse_group.len());
            let mut refined_offsets = vec![0];

            let local_ids: Vec<usize> = (0..coarse_group.len()).collect();

            FlatClusters::refine_single_cluster(
                &local_ids,
                &mzs,
                self.mz_tolerance,
                &mut refined_data,
                &mut refined_offsets,
            );

            for window in refined_offsets.windows(2) {
                let start = window[0];
                let end = window[1];
                let sub_cluster_indices = &refined_data[start..end];

                if sub_cluster_indices.len() > 1 {
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

        self.fast_batches = fast_batches;
        self.slow_batches = slow_batches;
        Ok(())
    }

    /// Partitions spectra into batches based strictly on charge and precursor m/z tolerance.
    ///
    /// This is a simpler alternative to [`split_and_refine_clusters`] that does
    /// not perform transitive refinement.
    pub fn split_on_mz_and_charge(
        &mut self,
        mut spectra: Vec<MSNSpectrum>,
    ) -> Result<(), std::io::Error> {
        spectra.par_sort_unstable_by(|spectrum1, spectrum2| {
            (spectrum1.charge(), spectrum1.precursor_mz)
                .partial_cmp(&(spectrum2.charge(), spectrum2.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fast_batches = Vec::new();
        let mut slow_batches = Vec::new();

        // Read-only pass: record where each mz/charge group starts.
        let boundaries = mz_charge_boundaries(
            &spectra,
            |s: &MSNSpectrum| s.charge(),
            |s: &MSNSpectrum| s.get_neutral_mass(),
            self.mz_tolerance,
        );

        let rt_tolerance = self.rt_tolerance;
        let batch_size_threshold = self.batch_size_threshold;
        let mut rt_boundaries: Vec<usize> = Vec::new();

        // Drain mz-groups back-to-front so earlier indices remain valid.
        for window in boundaries.windows(2).rev() {
            let (start, end) = (window[0], window[1]);
            if end - start <= 1 {
                continue;
            }

            let mut mz_group: Vec<MSNSpectrum> = spectra.drain(start..end).collect();

            if let Some(rt_tol) = rt_tolerance {
                // Sub-split by RT: sort the mz-group by RT then do a second
                // consecutive-gap scan. Total extra work across all groups is
                // O(k log k) per group, O(n log n) overall.
                mz_group.sort_unstable_by(|a, b| {
                    a.retention_time_in_seconds
                        .partial_cmp(&b.retention_time_in_seconds)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let m = mz_group.len();
                rt_boundaries.clear();
                rt_boundaries.push(0);
                for i in 1..m {
                    let gap = (mz_group[i].retention_time_in_seconds
                        - mz_group[i - 1].retention_time_in_seconds)
                        .abs();
                    if gap > rt_tol {
                        rt_boundaries.push(i);
                    }
                }
                rt_boundaries.push(m);

                for rt_window in rt_boundaries.windows(2).rev() {
                    let (rs, re) = (rt_window[0], rt_window[1]);
                    if re - rs > 1 {
                        let mut sub: Vec<MSNSpectrum> = mz_group.drain(rs..re).collect();
                        // The RT scan above left this sub-group in retention
                        // time order. A partition has to satisfy both
                        // constraints, and every consumer downstream -- the
                        // mass window in the dense and exact paths,
                        // `windowed_mass_edges` -- needs ascending neutral
                        // mass, so restore the batch's primary ordering.
                        sub.sort_unstable_by(|a, b| {
                            a.precursor_mz
                                .partial_cmp(&b.precursor_mz)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let len = sub.len();
                        let batch = MassSpectrumBatch { batch: sub };
                        if len < batch_size_threshold {
                            fast_batches.push(batch);
                        } else {
                            slow_batches.push(batch);
                        }
                    }
                }
            } else {
                let len = mz_group.len();
                let batch = MassSpectrumBatch { batch: mz_group };
                if len < batch_size_threshold {
                    fast_batches.push(batch);
                } else {
                    slow_batches.push(batch);
                }
            }
        }

        // Restore sorted order — we built lists back-to-front.
        fast_batches.reverse();
        slow_batches.reverse();

        self.fast_batches = fast_batches;
        self.slow_batches = slow_batches;
        Ok(())
    }

    /// Executes the full clustering workflow on the prepared batches.
    ///
    /// This method processes all queued `fast_batches` in parallel and then
    /// processes `slow_batches` sequentially, using the best available
    /// algorithm for each (Dense DBSCAN, Graph DBSCAN, or GPU-accelerated).
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if any internal processing fails.
    pub fn cluster(&mut self) -> Result<FlatClusters, std::io::Error> {
        let mut cluster_assignments: Vec<(usize, u64)> = Vec::new();
        let mut cluster_offset = 0usize;

        let start = std::time::Instant::now();
        if self.verbose {
            println!("Starting fast batch");
        }

        let fast_batches = std::mem::take(&mut self.fast_batches);
        let (fast_assignments, fast_count) = self.process_fast_batches(fast_batches);
        for mut a in fast_assignments {
            a.0 += cluster_offset;
            cluster_assignments.push(a);
        }
        cluster_offset += fast_count;

        if self.verbose {
            println!(
                "Done with fast batch after: {}",
                start.elapsed().as_secs_f32()
            );
            println!("Starting slow batch");
        }
        let start = std::time::Instant::now();

        for batch in std::mem::take(&mut self.slow_batches) {
            let spectra = batch.batch;
            let (local_assignments, local_max) = self.cluster_slow_batch(&spectra);
            for mut a in local_assignments {
                a.0 += cluster_offset;
                cluster_assignments.push(a);
            }
            cluster_offset += local_max;
        }

        if self.verbose {
            println!(
                "Done with slow batch after: {}",
                start.elapsed().as_secs_f32()
            );
        }

        Ok(Self::build_flat_clusters(cluster_assignments))
    }

    /// The similarity threshold this batch should use.
    ///
    /// With [`Self::adaptive_alpha`] unset this is just `1 - epsilon`.
    /// When set, the batch's candidate count -- the median number of
    /// spectra inside one precursor-mass window, which is what a spectrum
    /// is actually compared against -- sets the threshold so the expected
    /// spurious links per spectrum stay at `alpha`.
    fn batch_threshold(&self, spectra: &[MSNSpectrum]) -> f32 {
        let Some(alpha) = self.adaptive_alpha else {
            return 1.0 - self.epsilon;
        };
        if spectra.len() < 2 {
            return 1.0 - self.epsilon;
        }
        let ends = mass_window_ends(
            spectra,
            |s: &MSNSpectrum| s.get_neutral_mass(),
            self.mz_tolerance,
        );
        let mut widths: Vec<usize> = ends
            .iter()
            .enumerate()
            .map(|(i, &e)| e.saturating_sub(i))
            .collect();
        widths.sort_unstable();
        let candidates = widths[widths.len() / 2].max(2) as f32;
        let table = self.null_tail.as_deref().unwrap_or(&NULL_TAIL);
        let tau = null_threshold_for(table, (alpha / candidates).log10());
        // Below 0.5 the threshold stops discriminating at all; above 0.999
        // nothing but bit-identical spectra survive.
        tau.clamp(0.5, 0.999)
    }

    /// Processes all fast batches in parallel, returning aggregated assignments and cluster count.
    fn process_fast_batches(&self, batches: Vec<MassSpectrumBatch>) -> (Vec<(usize, u64)>, usize) {
        if batches.is_empty() {
            return (Vec::new(), 0);
        }

        let hash_size = self.hash_size();
        let precursor_mz_tolerance = self.mz_tolerance;
        let min_neighbors = self.min_neighbors;
        let hasher = &self.hasher;

        let batch_results: Vec<_> = if self.exact {
            batches
                .par_iter()
                .map_init(
                    || {
                        (
                            Vec::<Triplet<usize, usize, f32>>::new(),
                            Vec::<usize>::new(),
                        )
                    },
                    |(triplets_buffer, bit_matrix_buffer), batch| {
                        let spectra = batch.batch.as_slice();
                        let n_spectra = spectra.len();
                        let threshold = self.batch_threshold(spectra);

                        if n_spectra == 2 && min_neighbors == 2 {
                            let mut s0 = spectra[0].spectrum.to_soa(hasher.tolerance);
                            s0.l2_normalize();
                            let mut s1 = spectra[1].spectrum.to_soa(hasher.tolerance);
                            s1.l2_normalize();
                            let sim = crate::clustering::similarities::branching_cosine_sim(
                                &s0.mz_bins,
                                &s0.intensities,
                                &s1.mz_bins,
                                &s1.intensities,
                            );
                            return if sim >= threshold {
                                (vec![(0_usize, spectra[0].id), (0_usize, spectra[1].id)], 1)
                            } else {
                                (vec![], 0)
                            };
                        }

                        let words_per_vertex = (n_spectra + WORDSIZE - 1) / WORDSIZE;
                        triplets_buffer.clear();
                        triplets_buffer.reserve(n_spectra * 50);
                        bit_matrix_buffer.clear();
                        bit_matrix_buffer.resize(n_spectra * words_per_vertex, 0);

                        let mut max_mz = 0u32;
                        for (row_idx, spectrum) in spectra.iter().enumerate() {
                            let mut soa = spectrum.spectrum.to_soa(hasher.tolerance);
                            soa.l2_normalize();
                            if let Some(&last) = soa.mz_bins.last() {
                                max_mz = max_mz.max(last);
                            }
                            triplets_buffer.extend(
                                soa.mz_bins
                                    .iter()
                                    .zip(&soa.intensities)
                                    .filter(|(_, v)| v.abs() > 1e-6)
                                    .map(|(&col, &val)| Triplet::new(row_idx, col as usize, val)),
                            );
                        }

                        let a_mat = SparseRowMat::try_new_from_triplets(
                            n_spectra,
                            max_mz as usize + 1,
                            triplets_buffer.as_slice(),
                        )
                        .unwrap();
                        let a_col = a_mat.to_col_major().unwrap();
                        let sparse_sim = sparse_sparse_matmul(
                            a_col.as_ref(),
                            a_mat.transpose(),
                            1.0,
                            faer::Par::Seq,
                        )
                        .unwrap();
                        let symbolic = sparse_sim.symbolic();

                        // Same precursor-mass window the slow exact path and
                        // the approximate path apply: a pair whose masses are
                        // further apart than `mz_tolerance` is not a
                        // neighbour however similar its peaks are, so it must
                        // not become a DBSCAN edge and get chained through.
                        let window_ends = mass_window_ends(
                            spectra,
                            |s: &MSNSpectrum| s.get_neutral_mass(),
                            precursor_mz_tolerance,
                        );
                        for col in 0..n_spectra {
                            for (idx, row) in symbolic.row_idx_of_col(col).enumerate() {
                                let (lo, hi) = if row < col { (row, col) } else { (col, row) };
                                if hi >= window_ends[lo] {
                                    continue;
                                }
                                if sparse_sim.val_of_col(col)[idx] >= threshold {
                                    bit_matrix_buffer
                                        [row * words_per_vertex..(row + 1) * words_per_vertex]
                                        .set_bit(col);
                                    bit_matrix_buffer
                                        [col * words_per_vertex..(col + 1) * words_per_vertex]
                                        .set_bit(row);
                                }
                            }
                        }

                        let labels = dbscan_bitmatrix(
                            bit_matrix_buffer,
                            n_spectra,
                            MIN_NEIGHBORS,
                            words_per_vertex,
                        );
                        let links = BatchLinks::Bits {
                            bits: bit_matrix_buffer,
                            words_per_vertex,
                        };
                        collect_and_refine(labels, spectra, precursor_mz_tolerance, &links)
                    },
                )
                .collect()
        } else {
            batches
                .par_iter()
                .map_init(
                    || (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<usize>::new()),
                    |(spectral_buffer, similarity_matrix_buffer, bit_matrix_buffer), batch| {
                        let spectra = batch.batch.as_slice();
                        let n_spectra = spectra.len();
                        let threshold = self.batch_threshold(spectra);

                        spectral_buffer.resize(n_spectra * hash_size, 0.0);
                        spectral_buffer.fill(0.0);
                        spectral_buffer
                            .chunks_mut(hash_size)
                            .zip(spectra.iter())
                            .for_each(|(buf, spectrum)| {
                                hasher.into_vector_native(
                                    spectrum.spectrum.peaks.as_slice(),
                                    true,
                                    buf,
                                );
                            });

                        // Mass-windowed edges when the batch is in mass
                        // order (the normal case); otherwise the full
                        // matrix, which does not care about ordering.
                        if is_mass_sorted(spectra) {
                            let (labels, words_per_vertex) =
                                if should_use_windowed_dense(spectra, precursor_mz_tolerance) {
                                    windowed_dense_labels(
                                        spectral_buffer,
                                        hash_size,
                                        spectra,
                                        precursor_mz_tolerance,
                                        1.0 - threshold,
                                        bit_matrix_buffer,
                                    )
                                } else {
                                    compute_similarity_matrix(
                                        similarity_matrix_buffer,
                                        spectral_buffer,
                                        hash_size,
                                    );
                                    let window_ends = mass_window_ends(
                                        spectra,
                                        |s: &MSNSpectrum| s.get_neutral_mass(),
                                        precursor_mz_tolerance,
                                    );
                                    dense_labels_mass_filtered(
                                        similarity_matrix_buffer,
                                        n_spectra,
                                        threshold,
                                        &window_ends,
                                        bit_matrix_buffer,
                                    )
                                };
                            let links = BatchLinks::Bits {
                                bits: bit_matrix_buffer,
                                words_per_vertex,
                            };
                            return collect_and_refine(
                                labels,
                                spectra,
                                precursor_mz_tolerance,
                                &links,
                            );
                        }

                        #[cfg(target_vendor = "apple")]
                        let labels = {
                            unsafe {
                                similarity_matrix_buffer.resize(n_spectra * n_spectra, 0.0);
                                cblas_ssyrk(
                                    101,
                                    121,
                                    111,
                                    n_spectra as i32,
                                    hash_size as i32,
                                    1.0,
                                    spectral_buffer.as_ptr(),
                                    hash_size as i32,
                                    0.0,
                                    similarity_matrix_buffer.as_mut_ptr(),
                                    n_spectra as i32,
                                );
                            }
                            dbscan_upper_triangular(
                                similarity_matrix_buffer,
                                n_spectra,
                                threshold,
                                MIN_NEIGHBORS,
                            )
                        };

                        #[cfg(not(target_vendor = "apple"))]
                        let labels = {
                            compute_similarity_matrix(
                                similarity_matrix_buffer,
                                spectral_buffer,
                                hash_size,
                            );
                            dbscan_upper_triangular(
                                similarity_matrix_buffer,
                                n_spectra,
                                threshold,
                                MIN_NEIGHBORS,
                            )
                        };

                        let links = BatchLinks::UpperTriangular {
                            similarity: similarity_matrix_buffer,
                            n: n_spectra,
                            min_similarity: threshold,
                        };
                        collect_and_refine(labels, spectra, precursor_mz_tolerance, &links)
                    },
                )
                .collect()
        };

        let mut all_assignments = Vec::new();
        let mut total_count = 0usize;
        for (local_assignments, local_max) in batch_results {
            for mut a in local_assignments {
                a.0 += total_count;
                all_assignments.push(a);
            }
            total_count += local_max;
        }
        (all_assignments, total_count)
    }

    /// Dispatches a single slow batch to the appropriate algorithm and returns
    /// cluster assignments with the number of clusters produced.
    fn cluster_slow_batch(&mut self, spectra: &[MSNSpectrum]) -> (Vec<(usize, u64)>, usize) {
        let n_spectra = spectra.len();
        let hash_size = self.hash_size();
        let epsilon = 1.0 - self.batch_threshold(spectra);
        let precursor_mz_tolerance = self.mz_tolerance;

        // Paths that build their own edge list hand it over; the dense path
        // leaves its similarity matrix in `self.similarity_matrix_buffer`,
        // which is still valid here unless the GPU packed layout was used.
        let mut owned_links: Option<BatchLinks> = None;
        let mut dense_matrix_valid = false;

        let dbscan_labels = if n_spectra > self.approximate_threshold {
            let (labels, links) = self.cluster_approx_graph(spectra, hash_size, epsilon);
            owned_links = Some(links);
            labels
        } else if self.exact {
            let (labels, links) = self.cluster_exact_sparse(spectra);
            owned_links = Some(links);
            labels
        } else {
            let hasher = &self.hasher;
            self.spectral_buffer.resize(n_spectra * hash_size, 0.0);
            self.spectral_buffer.fill(0.0);
            self.spectral_buffer
                .par_chunks_exact_mut(hash_size)
                .zip(spectra.par_iter())
                .for_each(|(buf, spectrum)| {
                    hasher.into_vector_native(spectrum.spectrum.peaks.as_slice(), true, buf);
                });
            if is_mass_sorted(spectra) {
                let mut bit_matrix = std::mem::take(&mut self.dense_bit_matrix);
                let (labels, words_per_vertex) = if should_use_windowed_dense(spectra, self.mz_tolerance)
                {
                    windowed_dense_labels(
                        &self.spectral_buffer,
                        hash_size,
                        spectra,
                        self.mz_tolerance,
                        epsilon,
                        &mut bit_matrix,
                    )
                } else {
                    compute_similarity_matrix_parallel(
                        &mut self.similarity_matrix_buffer,
                        &self.spectral_buffer,
                        hash_size,
                    );
                    let window_ends = mass_window_ends(
                        spectra,
                        |s: &MSNSpectrum| s.get_neutral_mass(),
                        self.mz_tolerance,
                    );
                    dense_labels_mass_filtered(
                        &self.similarity_matrix_buffer,
                        n_spectra,
                        1.0 - epsilon,
                        &window_ends,
                        &mut bit_matrix,
                    )
                };
                let links = BatchLinks::Bits {
                    bits: &bit_matrix,
                    words_per_vertex,
                };
                let result =
                    collect_and_refine(labels, spectra, precursor_mz_tolerance, &links);
                self.dense_bit_matrix = bit_matrix;
                return result;
            }
            let (labels, matrix_valid) = self.compute_dense_dbscan(n_spectra, hash_size);
            dense_matrix_valid = matrix_valid;
            labels
        };

        let links = match owned_links {
            Some(links) => links,
            None if dense_matrix_valid => BatchLinks::UpperTriangular {
                similarity: &self.similarity_matrix_buffer,
                n: n_spectra,
                min_similarity: 1.0 - epsilon,
            },
            None => BatchLinks::Unavailable,
        };
        collect_and_refine(dbscan_labels, spectra, precursor_mz_tolerance, &links)
    }

    /// Clusters a large batch using windowed mass-bounded graph DBSCAN.
    ///
    /// `spectra` arrives already sorted by `(charge, precursor_mz)` — same
    /// charge throughout a batch (batches are split on charge first), and
    /// `precursor_mz` order matches `neutral_mass` order for a fixed charge —
    /// so [`windowed_mass_edges`] can exploit that sortedness directly
    /// instead of routing through an approximate ANN index. See that
    /// function's docs for why: this "large batch" path exists specifically
    /// because chaining in `split_on_mz_and_charge`'s partitioning can make
    /// a batch span a much wider mass range than `mz_tolerance` would ever
    /// actually link, and mass being an exact, pre-sorted range constraint
    /// (not a generic high-dimensional nearest-neighbor problem) means a
    /// sliding window over that sort order is both cheaper and exact, where
    /// IVF/k-means was neither.
    fn cluster_approx_graph(
        &mut self,
        spectra: &[MSNSpectrum],
        hash_size: usize,
        epsilon: f32,
    ) -> (Vec<ClusterLabel>, BatchLinks<'static>) {
        let n_spectra = spectra.len();
        let hasher = &self.hasher;
        self.spectral_buffer.resize(n_spectra * hash_size, 0.0);
        self.spectral_buffer.fill(0.0);
        self.spectral_buffer
            .par_chunks_exact_mut(hash_size)
            .zip(spectra.par_iter())
            .for_each(|(buf, spectrum)| {
                hasher.into_vector_native(spectrum.spectrum.peaks.as_slice(), true, buf);
            });
        let neutral_masses: Vec<f64> = spectra.iter().map(|s| s.get_neutral_mass()).collect();
        let edges = windowed_mass_edges(
            &self.spectral_buffer,
            hash_size,
            &neutral_masses,
            self.mz_tolerance,
            epsilon,
            WINDOWED_EDGES_BLOCK_SIZE,
            // Reuses the same "how big before we need approximate search"
            // threshold that decides dense-vs-this-path in the first place:
            // a block whose candidate window is itself this large is a
            // genuinely dense local neighborhood (not chaining), which is
            // exactly the ANN problem IVF exists for.
            self.approximate_threshold,
        );
        let links = BatchLinks::from_edges(n_spectra, &edges);
        (
            graph_dbscan(n_spectra, &edges, epsilon, MIN_NEIGHBORS),
            links,
        )
    }

    /// Clusters a batch using exact sparse cosine similarity via an inverted
    /// index, bounded by the same precursor-mass window the approximate path
    /// uses.
    ///
    /// `spectra` arrives sorted by `(charge, precursor_mz)` with one charge
    /// per batch, so neutral masses are ascending and each row's admissible
    /// partners are a contiguous run starting just after it (see
    /// [`mass_window_ends`]). Without that bound this accumulated a score for
    /// every later row sharing any m/z bin across the whole batch -- work
    /// spent on pairs whose precursor masses already disqualify them, which
    /// `collect_and_refine` then had to split apart afterwards. Chaining in
    /// the partitioning rule means a batch can span far more mass than
    /// `mz_tolerance` ever links, so that was the bulk of the work on exactly
    /// the batches that were slowest.
    fn cluster_exact_sparse(
        &self,
        spectra: &[MSNSpectrum],
    ) -> (Vec<ClusterLabel>, BatchLinks<'static>) {
        let n_spectra = spectra.len();
        let mz_tolerance = self.hasher.tolerance;
        // Same per-batch threshold the caller applies, so the exact path is
        // not left on the global epsilon when the adaptive rule is active.
        let threshold = self.batch_threshold(spectra);
        let epsilon = 1.0 - threshold;

        let window_ends = mass_window_ends(
            spectra,
            |s: &MSNSpectrum| s.get_neutral_mass(),
            self.mz_tolerance,
        );

        let rows_data: Vec<(usize, Vec<u32>, Vec<f32>)> = spectra
            .par_iter()
            .enumerate()
            .map(|(row_idx, spectrum)| {
                let mut soa = spectrum.spectrum.to_soa(mz_tolerance);
                soa.l2_normalize();
                (row_idx, soa.mz_bins, soa.intensities)
            })
            .collect();

        let max_mz = rows_data
            .iter()
            .flat_map(|(_, mz_bins, _)| mz_bins.iter().copied())
            .max()
            .unwrap_or(0) as usize;

        let mut inverted_index: Vec<Vec<(usize, f32)>> = vec![Vec::new(); max_mz + 1];
        for (row_idx, mz_bins, intensities) in &rows_data {
            for (&mz, &int) in mz_bins.iter().zip(intensities.iter()) {
                if int.abs() > 1e-6 {
                    inverted_index[mz as usize].push((*row_idx, int));
                }
            }
        }

        let edges: Vec<Edge> = rows_data
            .par_iter()
            .map_init(
                || (vec![0.0f32; n_spectra], Vec::with_capacity(1024)),
                |(scores, touched_indices), (row_i, mz_i, int_i)| {
                    let window_end = window_ends[*row_i];
                    for (&mz, &int) in mz_i.iter().zip(int_i.iter()) {
                        if int.abs() > 1e-6 {
                            if let Some(matches) = inverted_index.get(mz as usize) {
                                for &(row_j, int_j) in matches {
                                    if row_j > *row_i && row_j < window_end {
                                        if scores[row_j] == 0.0 {
                                            touched_indices.push(row_j);
                                        }
                                        scores[row_j] += int * int_j;
                                    }
                                }
                            }
                        }
                    }

                    let local_edges: Vec<Edge> = touched_indices
                        .iter()
                        .filter_map(|&row_j| {
                            let score = scores[row_j];
                            scores[row_j] = 0.0;
                            (score >= threshold).then_some(Edge {
                                source_id: *row_i,
                                target_id: row_j,
                                score,
                            })
                        })
                        .collect();
                    touched_indices.clear();
                    local_edges
                },
            )
            .flatten()
            .collect();

        let links = BatchLinks::from_edges(n_spectra, &edges);
        (
            graph_dbscan(n_spectra, &edges, epsilon, MIN_NEIGHBORS),
            links,
        )
    }

    /// Sorts cluster assignments and packs them into a [`FlatClusters`].
    fn build_flat_clusters(mut cluster_assignments: Vec<(usize, u64)>) -> FlatClusters {
        cluster_assignments.par_sort_unstable_by_key(|p| p.0);
        let mut data = Vec::with_capacity(cluster_assignments.len());
        let mut offsets = Vec::with_capacity(cluster_assignments.len() / 5);

        if !cluster_assignments.is_empty() {
            let mut current_cluster = cluster_assignments[0].0;
            offsets.push(0);
            for (cluster_id, spectrum_id) in cluster_assignments {
                if cluster_id != current_cluster {
                    offsets.push(data.len());
                    current_cluster = cluster_id;
                }
                data.push(spectrum_id as usize);
            }
            offsets.push(data.len());
        }

        FlatClusters { data, offsets }
    }

    /// Internal method to compute DBSCAN labels using a dense similarity matrix.
    ///
    /// This method abstracts over different hardware-accelerated backends
    /// (Apple BLAS, Generic Parallel BLAS, or GPU).
    /// Returns the labels and whether `self.similarity_matrix_buffer` holds
    /// the matrix they came from -- false on the GPU path, whose packed
    /// layout this buffer does not carry.
    fn compute_dense_dbscan(
        &mut self,
        n_spectra: usize,
        hash_size: usize,
    ) -> (Vec<ClusterLabel>, bool) {
        let min_sim = 1.0 - self.epsilon;

        #[cfg(feature = "gpu")]
        if self.use_gpu {
            if let Some(ref gpu_ctx) = self.gpu {
                if n_spectra <= gpu_ctx.max_m {
                    let packed =
                        gpu_ctx.compute(&self.spectral_buffer[..n_spectra * hash_size], n_spectra);
                    return (
                        dbscan_upper_triangular_packed(
                            &packed,
                            n_spectra,
                            min_sim,
                            MIN_NEIGHBORS,
                        ),
                        false,
                    );
                }
            }
        }

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
            return (
                dbscan_upper_triangular(
                    &self.similarity_matrix_buffer,
                    n_spectra,
                    min_sim,
                    MIN_NEIGHBORS,
                ),
                true,
            );
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            compute_similarity_matrix_parallel(
                &mut self.similarity_matrix_buffer,
                &self.spectral_buffer,
                hash_size,
            );

            (
                dbscan_upper_triangular(
                    &self.similarity_matrix_buffer,
                    n_spectra,
                    min_sim,
                    MIN_NEIGHBORS,
                ),
                true,
            )
        }
    }

    /// Computes the similarity matrix for the data currently in the internal `spectral_buffer`.
    fn compute_similarity_matrix_self(&mut self) {
        let hash_size = self.hash_size();
        compute_similarity_matrix(
            &mut self.similarity_matrix_buffer,
            &self.spectral_buffer,
            hash_size,
        );
    }

    /// Computes the similarity matrix for the data in the internal `spectral_buffer` in parallel.
    pub fn compute_similarity_matrix_parallel(&mut self) {
        let hash_size = self.hash_size();
        compute_similarity_matrix_parallel(
            &mut self.similarity_matrix_buffer,
            &self.spectral_buffer,
            hash_size,
        );
    }
}

#[cfg(test)]
mod connectivity_tests {
    use super::*;
    use crate::spectrum::MassSpectrum;

    /// The counters are process-global, so these tests must not interleave.
    static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn spectrum(id: u64, precursor_mz: f64, charge: i8) -> MSNSpectrum {
        MSNSpectrum {
            precursor_mz,
            retention_time_in_seconds: 0.0,
            id,
            spectrum: MassSpectrum {
                peaks: Vec::new(),
                charge,
            },
        }
    }

    /// Four spectra in one DBSCAN cluster, linked only as a path
    /// `0 - 1 - 2 - 3` in mass order. The mass split cuts them into two
    /// intervals; both halves stay connected, so nothing extra happens.
    #[test]
    fn connected_intervals_are_emitted_unchanged() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let spectra = vec![
            spectrum(10, 1000.000, 1),
            spectrum(11, 1000.001, 1),
            spectrum(12, 1000.050, 1),
            spectrum(13, 1000.051, 1),
        ];
        let labels = vec![ClusterLabel::Cluster(0); 4];
        let links = BatchLinks::from_edges(
            4,
            &[
                Edge { source_id: 0, target_id: 1, score: 1.0 },
                Edge { source_id: 1, target_id: 2, score: 1.0 },
                Edge { source_id: 2, target_id: 3, score: 1.0 },
            ],
        );
        let _ = take_connectivity_stats();
        let (assignments, n_clusters) =
            collect_and_refine(labels, &spectra, MassTolerance::Da(0.01), &links);
        assert_eq!(n_clusters, 2, "the mass split should produce two intervals");
        assert_eq!(assignments.len(), 4, "every spectrum stays clustered");
        let (_, split, dropped) = take_connectivity_stats();
        assert_eq!((split, dropped), (0, 0));
    }

    /// The failure the re-check exists for: spectra 0 and 2 are close in
    /// mass but have no link between them -- they shared a DBSCAN cluster
    /// only through spectrum 1, which the mass split puts in the other
    /// interval. Without the re-check they would be emitted as one cluster.
    #[test]
    fn mass_interval_that_is_not_connected_is_split() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let spectra = vec![
            spectrum(10, 1000.000, 1),
            spectrum(11, 1000.050, 1),
            spectrum(12, 1000.001, 1),
        ];
        // 0 - 1 - 2 as a path: 0 and 2 are never directly linked.
        let links = BatchLinks::from_edges(
            3,
            &[
                Edge { source_id: 0, target_id: 1, score: 1.0 },
                Edge { source_id: 1, target_id: 2, score: 1.0 },
            ],
        );
        let labels = vec![ClusterLabel::Cluster(0); 3];
        let _ = take_connectivity_stats();
        let (assignments, n_clusters) =
            collect_and_refine(labels, &spectra, MassTolerance::Da(0.01), &links);
        let (_, split, dropped) = take_connectivity_stats();
        assert_eq!(split, 1, "the interval holding 0 and 2 is not connected");
        assert_eq!(
            dropped, 2,
            "0 and 2 each end up alone in their component, so neither is a cluster"
        );
        assert_eq!(n_clusters, 0, "nothing survives as a cluster here");
        assert!(assignments.is_empty());
    }

    /// A pair whose precursor masses are further apart than the tolerance
    /// must not become an edge, however identical their spectra are. The
    /// unrestricted dense path linked them and left the mass split to clean
    /// up afterwards, which it cannot fully do.
    #[test]
    fn windowed_dense_does_not_link_across_the_mass_tolerance() {
        let hash_size = 2;
        // Three identical unit vectors: similarity is 1.0 for every pair.
        let spectral_buffer = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let spectra = vec![
            spectrum(10, 1000.000, 1),
            spectrum(11, 1000.001, 1),
            spectrum(12, 1000.050, 1),
        ];
        let mut bits = Vec::new();
        let (labels, _) = windowed_dense_labels(
            &spectral_buffer,
            hash_size,
            &spectra,
            MassTolerance::Da(0.01),
            0.1,
            &mut bits,
        );
        assert_eq!(labels[0], labels[1], "the close pair still clusters");
        assert!(matches!(labels[0], ClusterLabel::Cluster(_)));
        assert_eq!(
            labels[2],
            ClusterLabel::Noise,
            "the distant spectrum has no admissible partner and is noise"
        );
    }

    /// When every pair is inside the tolerance the window removes nothing,
    /// so the result is the plain dense clustering.
    #[test]
    fn windowed_dense_keeps_everything_when_all_pairs_are_admissible() {
        let spectral_buffer = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let spectra = vec![
            spectrum(10, 1000.000, 1),
            spectrum(11, 1000.001, 1),
            spectrum(12, 1000.002, 1),
        ];
        let mut bits = Vec::new();
        let (labels, _) = windowed_dense_labels(
            &spectral_buffer,
            2,
            &spectra,
            MassTolerance::Da(0.01),
            0.1,
            &mut bits,
        );
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert!(matches!(labels[0], ClusterLabel::Cluster(_)));
    }

    /// A partition has to satisfy the retention-time constraint *and* stay
    /// in mass order, because every windowed path downstream assumes
    /// ascending neutral mass. The RT sub-split sorts by retention time to
    /// find its gaps, so each sub-group has to be put back in mass order
    /// before it is handed on.
    #[test]
    fn rt_split_batches_are_still_mass_sorted() {
        let mut engine = ClusterEngine::default();
        engine.rt_tolerance = Some(10.0);
        engine.mz_tolerance = MassTolerance::Da(0.05);

        // One m/z group (all within 0.05 Da), two RT clusters far apart.
        // Mass order and RT order deliberately disagree.
        let mut spectra = Vec::new();
        for (i, (mz, rt)) in [
            (1000.00, 100.0),
            (1000.01, 900.0),
            (1000.02, 105.0),
            (1000.03, 905.0),
            (1000.04, 102.0),
            (1000.05, 902.0),
        ]
        .iter()
        .enumerate()
        {
            let mut s = spectrum(i as u64, *mz, 1);
            s.retention_time_in_seconds = *rt;
            spectra.push(s);
        }
        engine.split_on_mz_and_charge(spectra).unwrap();

        let batches: Vec<&MassSpectrumBatch> = engine
            .fast_batches
            .iter()
            .chain(engine.slow_batches.iter())
            .collect();
        assert_eq!(batches.len(), 2, "the RT gap should split this in two");
        for batch in batches {
            assert!(
                is_mass_sorted(&batch.batch),
                "batch left in RT order: {:?}",
                batch.batch.iter().map(|s| s.precursor_mz).collect::<Vec<_>>()
            );
        }
    }

    /// Both dense strategies must admit exactly the same pairs -- only their
    /// cost differs -- so they must agree label for label.
    #[test]
    fn both_dense_strategies_agree() {
        let hash_size = 2;
        let n = 40;
        let mut spectral_buffer = vec![0.0f32; n * hash_size];
        let mut spectra = Vec::new();
        for i in 0..n {
            // Two alternating directions, so only every other pair is similar.
            let base = i % 2;
            spectral_buffer[i * hash_size + base] = 1.0;
            spectra.push(spectrum(i as u64, 1000.0 + i as f64 * 0.002, 1));
        }
        let tolerance = MassTolerance::Da(0.005);
        let epsilon = 0.1f32;

        let mut bits_a = Vec::new();
        let (windowed, _) = windowed_dense_labels(
            &spectral_buffer,
            hash_size,
            &spectra,
            tolerance,
            epsilon,
            &mut bits_a,
        );

        let mut similarity = Vec::new();
        compute_similarity_matrix(&mut similarity, &spectral_buffer, hash_size);
        let window_ends = mass_window_ends(
            &spectra,
            |s: &MSNSpectrum| s.get_neutral_mass(),
            tolerance,
        );
        let mut bits_b = Vec::new();
        let (filtered, _) = dense_labels_mass_filtered(
            &similarity,
            n,
            1.0 - epsilon,
            &window_ends,
            &mut bits_b,
        );

        assert_eq!(bits_a, bits_b, "the two strategies built different adjacency");
        assert_eq!(windowed, filtered, "the two strategies produced different labels");
        assert!(
            windowed.iter().any(|l| matches!(l, ClusterLabel::Cluster(_))),
            "sanity check: something should cluster"
        );
    }

    /// `windowed_mass_edges` needs ascending neutral mass; `rt_tolerance`
    /// makes `split_on_mz_and_charge` hand out RT-ordered batches instead,
    /// which is why the caller checks before using the windowed path.
    #[test]
    fn mass_sortedness_is_detected() {
        let ascending = vec![
            spectrum(1, 1000.0, 1),
            spectrum(2, 1000.5, 1),
            spectrum(3, 1001.0, 1),
        ];
        let shuffled = vec![
            spectrum(1, 1000.0, 1),
            spectrum(2, 1001.0, 1),
            spectrum(3, 1000.5, 1),
        ];
        assert!(is_mass_sorted(&ascending));
        assert!(!is_mass_sorted(&shuffled));
    }

    /// With no adjacency available the behaviour must be exactly the old
    /// mass-only split.
    #[test]
    fn unavailable_links_fall_back_to_the_mass_only_split() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let spectra = vec![
            spectrum(10, 1000.000, 1),
            spectrum(11, 1000.050, 1),
            spectrum(12, 1000.001, 1),
        ];
        let labels = vec![ClusterLabel::Cluster(0); 3];
        let _ = take_connectivity_stats();
        let (assignments, n_clusters) = collect_and_refine(
            labels,
            &spectra,
            MassTolerance::Da(0.01),
            &BatchLinks::Unavailable,
        );
        let (_, split, dropped) = take_connectivity_stats();
        assert_eq!((split, dropped), (0, 0), "no re-check runs without links");
        // `refine_single_cluster` drops intervals of one, so the lone
        // spectrum at 1000.050 is not emitted; 1000.000 and 1000.001 are,
        // even though nothing links them.
        assert_eq!(n_clusters, 1);
        assert_eq!(assignments.len(), 2);
    }
}
