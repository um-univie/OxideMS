use super::dbscan::{
    ClusterLabel, dbscan_bitmatrix, dbscan_upper_triangular, dbscan_upper_triangular_packed,
};
use super::flat_clusters::FlatClusters;
use crate::{
    MSNSpectrum, SpectrumHasher,
    clustering::{
        bitset::{BitSetOpsMSB0, WORDSIZE},
        k_cluster_ivf::{BucketIndex, Edge, graph_dbscan},
    },
    workflow::MassTolerance,
};
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::sparse::linalg::matmul::sparse_sparse_matmul;
use faer::sparse::{SparseRowMat, Triplet};
use faer::{MatMut, MatRef};
use rayon::prelude::*;

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

/// Groups DBSCAN labels by cluster id, refines each group by precursor m/z, and
/// returns a flat assignment list together with the number of refined clusters produced.
fn collect_and_refine(
    dbscan_labels: Vec<ClusterLabel>,
    spectra: &[MSNSpectrum],
    mz_tolerance: MassTolerance,
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
            for &batch_idx in &sub_data[window[0]..window[1]] {
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
            batch_size_threshold: FAST_BATCH_NUMBER,
            approximate_threshold: LARGE_BATCH_THRESHOLD,
            hasher: SpectrumHasher::new(101.0, 1500.0, MassTolerance::Da(0.05), hash_size),
            exact: false,
            min_neighbors: MIN_NEIGHBORS,
            use_gpu: false,
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
        let n = spectra.len();
        let mut boundaries: Vec<usize> = vec![0];
        for i in 1..n {
            let a = &spectra[i - 1];
            let b = &spectra[i];
            let same_group = if a.charge() != b.charge() {
                false
            } else {
                let m_a = a.get_neutral_mass();
                let m_b = b.get_neutral_mass();
                let gap = (m_b - m_a).abs();
                match self.mz_tolerance {
                    MassTolerance::Da(tol) => gap <= tol,
                    MassTolerance::Ppm(tol) => gap <= m_b * (tol / 1_000_000.0),
                }
            };
            if !same_group {
                boundaries.push(i);
            }
        }
        if n > 0 {
            boundaries.push(n);
        }

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
                        let sub: Vec<MSNSpectrum> = mz_group.drain(rs..re).collect();
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
        let mut current_clusters_size: usize = 0;

        let hash_size = self.hash_size();
        let epsilon = self.epsilon;
        let mz_tolerance = self.hasher.tolerance;
        let precursor_mz_tolerance = self.mz_tolerance;
        let threshold = 1.0 - epsilon;

        let mut fast_batches = std::mem::take(&mut self.fast_batches);
        let start_time = std::time::Instant::now();
        println!("Starting fast batch");

        if !fast_batches.is_empty() {
            let hasher = &self.hasher;
            let batch_results: Vec<_> = if self.exact {
                fast_batches
                    .par_iter()
                    .map_init(
                        || {
                            let triplets_buffer: Vec<Triplet<usize, usize, f32>> = Vec::new();
                            let bit_matrix_buffer: Vec<usize> = Vec::new();
                            (triplets_buffer, bit_matrix_buffer)
                        },
                        |(triplets_buffer, bit_matrix_buffer), batch| {
                            let spectra = batch.batch.as_slice();
                            let n_spectra = spectra.len();

                            if n_spectra == 2 && self.min_neighbors == 2 {
                                let mut spectrum1 =
                                    spectra[0].spectrum.to_soa(self.hasher.tolerance);
                                spectrum1.l2_normalize();
                                let mut spectrum2 =
                                    spectra[1].spectrum.to_soa(self.hasher.tolerance);
                                spectrum2.l2_normalize();

                                let similarity =
                                    crate::clustering::similarities::branching_cosine_sim(
                                        &spectrum1.mz_bins,
                                        &spectrum1.intensities,
                                        &spectrum2.mz_bins,
                                        &spectrum2.intensities,
                                    );
                                if similarity >= threshold {
                                    return (
                                        vec![(0_usize, spectra[0].id), (0_usize, spectra[1].id)],
                                        1,
                                    );
                                } else {
                                    return (vec![], 0);
                                }
                            }

                            let words_per_vertex = (n_spectra + WORDSIZE - 1) / WORDSIZE;

                            triplets_buffer.clear();
                            triplets_buffer.reserve(n_spectra * 50);
                            bit_matrix_buffer.clear();
                            let mut max_mz = 0;

                            bit_matrix_buffer.resize(n_spectra * words_per_vertex, 0);

                            for (row_idx, spectrum) in spectra.iter().enumerate() {
                                let mut soa = spectrum.spectrum.to_soa(self.hasher.tolerance);
                                soa.l2_normalize();

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

                            let dbscan_labels = dbscan_bitmatrix(
                                &bit_matrix_buffer,
                                n_spectra,
                                MIN_NEIGHBORS,
                                words_per_vertex,
                            );

                            collect_and_refine(dbscan_labels, spectra, precursor_mz_tolerance)
                        },
                    )
                    .collect()
            } else {
                fast_batches
                    .par_iter()
                    .map_init(
                        || {
                            let spectral_buffer: Vec<f32> = Vec::new();
                            let similarity_matrix_buffer: Vec<f32> = Vec::new();
                            (spectral_buffer, similarity_matrix_buffer)
                        },
                        |(spectral_buffer, similarity_matrix_buffer), batch| {
                            let spectra = batch.batch.as_slice();
                            let n_spectra = spectra.len();

                            spectral_buffer.resize(n_spectra * hash_size, 0.0);
                            // This is necessary as we will not override all values
                            spectral_buffer.fill(0.0);

                            spectral_buffer
                                .chunks_mut(hash_size)
                                .zip(spectra.iter())
                                .for_each(|(buffer_slice, spectrum)| {
                                    hasher.into_vector_native(
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

                                dbscan_labels = dbscan_upper_triangular(
                                    similarity_matrix_buffer,
                                    n_spectra,
                                    threshold,
                                    MIN_NEIGHBORS,
                                );
                            }

                            #[cfg(not(all(target_vendor = "apple")))]
                            {
                                compute_similarity_matrix(
                                    similarity_matrix_buffer,
                                    spectral_buffer,
                                    hash_size,
                                );

                                dbscan_labels = dbscan_upper_triangular(
                                    similarity_matrix_buffer,
                                    n_spectra,
                                    threshold,
                                    MIN_NEIGHBORS,
                                );
                            }

                            collect_and_refine(dbscan_labels, spectra, precursor_mz_tolerance)
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
        }

        println!(
            "Done with fast batch after: {}",
            start_time.elapsed().as_secs_f32()
        );

        println!("Starting slow batch");
        let start_time = std::time::Instant::now();

        let slow_batches = std::mem::take(&mut self.slow_batches);
        for batch in slow_batches.into_iter() {
            let spectra = batch.batch.as_slice();
            let n_spectra = spectra.len();


            let hasher = &self.hasher;
            let dbscan_labels: Vec<ClusterLabel>;
            if n_spectra > self.approximate_threshold {
                self.spectral_buffer.resize(n_spectra * hash_size, 0.0);
                self.spectral_buffer.fill(0.0);
                self.spectral_buffer
                    .par_chunks_exact_mut(hash_size)
                    .zip(spectra.par_iter())
                    .for_each(|(buffer_slice, spectrum)| {
                        hasher.into_vector_native(
                            spectrum.spectrum.peaks.as_slice(),
                            true,
                            buffer_slice,
                        );
                    });
                let index = BucketIndex::build(&self.spectral_buffer, hash_size, 64);
                let edges = index.extract_edges(64, epsilon);
                dbscan_labels = graph_dbscan(n_spectra, &edges, epsilon, MIN_NEIGHBORS);
            } else if self.exact {
                let start_time = std::time::Instant::now();

                let rows_data: Vec<(usize, Vec<u32>, Vec<f32>)> = spectra
                    .par_iter()
                    .enumerate()
                    .map(|(row_idx, spectrum)| {
                        let mut soa = spectrum.spectrum.to_soa(mz_tolerance);
                        soa.l2_normalize();
                        (row_idx, soa.mz_bins, soa.intensities)
                    })
                    .collect();
// 1. Find max_mz to size our inverted index
let max_mz = rows_data
    .iter()
    .flat_map(|(_, mz_bins, _)| mz_bins.iter().copied())
    .max()
    .unwrap_or(0) as usize;

// 2. Build an inverted index: mz_bin -> Vec<(spectrum_idx, intensity)>
// This is very fast to do sequentially and acts as our "transposed matrix"
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
    // map_init gives each Rayon thread its own reusable workspace
    .map_init(
        || {
            // Workspace per thread: 
            // 1. A dense array to accumulate scores (size = n_spectra)
            // 2. A list to track which indices we touched so we can reset them quickly
            (vec![0.0f32; n_spectra], Vec::with_capacity(1024))
        },
        |(scores, touched_indices), (row_i, mz_i, int_i)| {
            
            // 1. Scatter & Accumulate
            for (&mz, &int) in mz_i.iter().zip(int_i.iter()) {
                if int.abs() > 1e-6 {
                    if let Some(matches) = inverted_index.get(mz as usize) {
                        for &(row_j, int_j) in matches {
                            if row_j > *row_i {
                                if scores[row_j] == 0.0 {
                                    touched_indices.push(row_j);
                                }
                                scores[row_j] += int * int_j;
                            }
                        }
                    }
                }
            }

            let mut local_edges = Vec::new();

            // 2. Gather & Reset
            // We only iterate over the exact indices we modified, 
            // extracting the results and resetting the array to 0.0 for the next row.
            for &row_j in touched_indices.iter() {
                let score = scores[row_j];
                
                if score >= threshold {
                    local_edges.push(Edge {
                        source_id: *row_i,
                        target_id: row_j,
                        score,
                    });
                }
                
                // Crucial: Reset the dense array back to zero for the next iteration
                scores[row_j] = 0.0; 
            }
            touched_indices.clear(); // Reset the tracker

            local_edges
        },
    )
    .flatten() // Flattens the Vec<Vec<Edge>> from all threads into a single Vec<Edge>
    .collect();

                dbscan_labels = graph_dbscan(n_spectra, &edges, epsilon, MIN_NEIGHBORS);
            } else {
                self.spectral_buffer.resize(n_spectra * hash_size, 0.0);
                self.spectral_buffer.fill(0.0);
                self.spectral_buffer
                    .par_chunks_exact_mut(hash_size)
                    .zip(spectra.par_iter())
                    .for_each(|(buffer_slice, spectrum)| {
                        hasher.into_vector_native(
                            spectrum.spectrum.peaks.as_slice(),
                            true,
                            buffer_slice,
                        );
                    });
                dbscan_labels = self.compute_dense_dbscan(n_spectra, hash_size);
            }

            let (local_assignments, local_max) =
                collect_and_refine(dbscan_labels, spectra, precursor_mz_tolerance);
            for mut assignment in local_assignments {
                assignment.0 += current_clusters_size;
                cluster_assignments.push(assignment);
            }
            current_clusters_size += local_max;
        }

        println!(
            "Done with slow batch after: {}",
            start_time.elapsed().as_secs_f32()
        );

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
        }

        Ok(FlatClusters { data, offsets })
    }

    /// Internal method to compute DBSCAN labels using a dense similarity matrix.
    ///
    /// This method abstracts over different hardware-accelerated backends
    /// (Apple BLAS, Generic Parallel BLAS, or GPU).
    fn compute_dense_dbscan(&mut self, n_spectra: usize, hash_size: usize) -> Vec<ClusterLabel> {
        let min_sim = 1.0 - self.epsilon;

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
            compute_similarity_matrix_parallel(
                &mut self.similarity_matrix_buffer,
                &self.spectral_buffer,
                hash_size,
            );

            dbscan_upper_triangular(
                &self.similarity_matrix_buffer,
                n_spectra,
                min_sim,
                MIN_NEIGHBORS,
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
