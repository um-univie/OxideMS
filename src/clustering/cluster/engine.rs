use super::dbscan::{
    ClusterLabel, dbscan_bitmatrix, dbscan_upper_triangular, dbscan_upper_triangular_packed,
};
use super::flat_clusters::FlatClusters;
use crate::{
    MSNSpectrum, SpectrumHasher,
    clustering::{
        bitset::{BitSetOpsMSB0, WORDSIZE},
        k_cluster_ivf::{BucketIndex, graph_dbscan},
    },
    workflow::MassTolerance,
};
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
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

/// In-place L2 normalization of a flattened vector of spectra.
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

pub fn fast_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let a_mat = MatRef::from_column_major_slice(a, a.len(), 1);
    let b_mat = MatRef::from_column_major_slice(b, b.len(), 1);

    let c = a_mat.transpose() * b_mat;
    c[(0, 0)]
}

/// Computes only the upper triangle of C = A * A^T (single-threaded faer).
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

/// Computes only the upper triangle of C = A * A^T (parallel faer).
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

pub struct MassSpectrumBatch {
    pub(super) batch: Vec<MSNSpectrum>,
}

pub struct SpectrumMetadata {
    pub precursor_mz: f64,
    pub retention_time_in_seconds: f32,
    pub file_index: usize,
    pub scan_number: usize,
    pub charge: i8,
}

pub struct ClusterEngine {
    pub(super) spectral_buffer: Vec<f32>,
    pub(super) similarity_matrix_buffer: Vec<f32>,
    pub mz_tolerance: MassTolerance,
    pub epsilon: f32,
    pub(super) batch_size_threshold: usize,
    pub(super) hasher: SpectrumHasher,
    pub exact: bool,
    pub use_gpu: bool,
    pub(super) fast_batches: Vec<MassSpectrumBatch>,
    pub(super) slow_batches: Vec<MassSpectrumBatch>,
    #[cfg(feature = "gpu")]
    pub(super) gpu: Option<crate::clustering::gpu::GpuSimilarity>,
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
            fast_batches: Vec::new(),
            slow_batches: Vec::new(),
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
            fast_batches: Vec::new(),
            slow_batches: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

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

    pub fn split_on_mz_and_charge(
        &mut self,
        spectra: &mut [MSNSpectrum],
    ) -> Result<(), std::io::Error> {
        spectra.par_sort_unstable_by(|spectrum1, spectrum2| {
            (spectrum1.charge(), spectrum1.precursor_mz)
                .partial_cmp(&(spectrum2.charge(), spectrum2.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fast_batches = Vec::new();
        let mut slow_batches = Vec::new();

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
            if group.len() > 1 {
                let batch = MassSpectrumBatch {
                    batch: group.to_vec(),
                };

                if group.len() < self.batch_size_threshold {
                    fast_batches.push(batch);
                } else {
                    slow_batches.push(batch);
                }
            }
        }

        self.fast_batches = fast_batches;
        self.slow_batches = slow_batches;
        Ok(())
    }

    pub fn cluster(&mut self) -> Result<FlatClusters, std::io::Error> {
        let mut cluster_assignments: Vec<(usize, u64)> = Vec::new();
        let mut current_clusters_size: usize = 0;

        let hash_size = self.hash_size();
        let epsilon = self.epsilon;
        let mz_tolerance = self.mz_tolerance;
        let threshold = 1.0 - epsilon;

        let fast_batches = std::mem::take(&mut self.fast_batches);
        let start_time = std::time::Instant::now();
        println!("Starting fast batch");

        if !fast_batches.is_empty() {
            use faer::sparse::{SparseRowMat, Triplet};
            use faer::sparse::linalg::matmul::sparse_sparse_matmul;

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

                            if n_spectra == 2 && MIN_NEIGHBORS == 2 {
                                let mut spectrum1 = spectra[0].spectrum.to_soa(mz_tolerance);
                                spectrum1.l2_normalize();
                                let mut spectrum2 = spectra[1].spectrum.to_soa(mz_tolerance);
                                spectrum2.l2_normalize();

                                let similarity = crate::clustering::similarities::branching_cosine_sim(
                                    &spectrum1.mz_bins,
                                    &spectrum1.intensities,
                                    &spectrum2.mz_bins,
                                    &spectrum2.intensities,
                                );
                                if similarity >= threshold {
                                    return (vec![(0_usize, spectra[0].id), (0_usize, spectra[1].id)], 1);
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
                                let mut soa = spectrum.spectrum.to_soa(mz_tolerance);
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

                            let mut local_cluster_assignments = Vec::new();
                            let mut max_value = 0;

                            for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate()
                            {
                                if let ClusterLabel::Cluster(n) = cluster_assignment {
                                    max_value = max_value.max(n + 1);
                                    local_cluster_assignments.push((n, spectra[index].id));
                                }
                            }

                            (local_cluster_assignments, max_value)
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

                            let mut local_cluster_assignments = Vec::new();
                            let mut max_value = 0;

                            for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate()
                            {
                                if let ClusterLabel::Cluster(n) = cluster_assignment {
                                    max_value = max_value.max(n + 1);
                                    local_cluster_assignments.push((n, spectra[index].id));
                                }
                            }

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
        }

        println!(
            "Done with fast batch after: {}",
            start_time.elapsed().as_secs_f32()
        );

        println!("Starting slow batch");
        let start_time = std::time::Instant::now();

        let slow_batches = std::mem::take(&mut self.slow_batches);
        for batch in &slow_batches {
            let spectra = batch.batch.as_slice();
            let n_spectra = spectra.len();

            self.spectral_buffer.resize(n_spectra * hash_size, 0.0);
            self.spectral_buffer.fill(0.0);

            let hasher = &self.hasher;
            self.spectral_buffer
                .par_chunks_exact_mut(hash_size)
                .zip(spectra.par_iter())
                .for_each(|(buffer_slice, spectrum)| {
                    hasher.into_vector_native(spectrum.spectrum.peaks.as_slice(), true, buffer_slice);
                });

            let dbscan_labels: Vec<ClusterLabel>;
            if n_spectra > LARGE_BATCH_THRESHOLD {
                let index = BucketIndex::build(&self.spectral_buffer, hash_size, 64);
                let edges = index.extract_edges(64, epsilon);
                dbscan_labels = graph_dbscan(n_spectra, &edges, epsilon, MIN_NEIGHBORS);
            } else {
                dbscan_labels = self.compute_dense_dbscan(n_spectra, hash_size);
            }

            let mut max_value = 0;
            for (index, cluster_assignment) in dbscan_labels.into_iter().enumerate() {
                if let ClusterLabel::Cluster(n) = cluster_assignment {
                    max_value = max_value.max(n + 1);
                    cluster_assignments.push((current_clusters_size + n, spectra[index].id));
                }
            }
            current_clusters_size += max_value;
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

    fn compute_similarity_matrix_self(&mut self) {
        let hash_size = self.hash_size();
        compute_similarity_matrix(
            &mut self.similarity_matrix_buffer,
            &self.spectral_buffer,
            hash_size,
        );
    }

    pub fn compute_similarity_matrix_parallel(&mut self) {
        let hash_size = self.hash_size();
        compute_similarity_matrix_parallel(
            &mut self.similarity_matrix_buffer,
            &self.spectral_buffer,
            hash_size,
        );
    }
}
