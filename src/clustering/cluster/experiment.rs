use super::dbscan::ClusterLabel;
use super::engine::{ClusterEngine, SpectrumMetadata, fast_dot_product};
use super::flat_clusters::FlatClusters;
use super::mz_partition::mz_charge_boundaries;
use crate::{
    MSNSpectrum, MassSpectralRecord, Peak, SpectrumHasher,
    io::mgf::{LowMemRecord, MGF_SPECTRA_SIZE, MgfParser, MgfParserConfig, SpillKey},
    io::mzml::{MzmlParser, MzmlParserConfig},
    workflow::*,
};
use memmap2::Mmap;
use nohash_hasher::IntMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

/// Internal-validation quality metrics for a single cluster, computed against
/// a consensus (mean, L2-renormalized) vector in the same hashed vector space
/// used during clustering. See [`SpectralExperiment::calculate_cluster_quality_metrics`].
#[derive(Debug, Clone, Copy)]
pub struct ClusterQualityMetrics {
    pub size: usize,
    /// Average cosine similarity of member spectra to the cluster consensus (cohesion).
    pub avg_cohesion: f64,
    /// Minimum cosine similarity of any member spectrum to the consensus (worst-case cohesion).
    pub min_cohesion: f64,
}

#[derive(Default)]
pub struct SpectralExperiment {
    pub file_names: Vec<String>,
    /// Full path each `file_names` entry was ingested from, parallel to it —
    /// kept around so [`Self::build_consensus_spectrum`] can reopen the
    /// original file and reparse a member's peaks from its recorded byte
    /// offsets. `file_names` alone isn't enough for that: it's stripped down
    /// to a bare basename for display in `write_results`'s USI strings.
    pub file_paths: Vec<PathBuf>,
    pub metadata: Vec<SpectrumMetadata>,
    pub workflow: WorkFlowManager,
    pub parser: MgfParser,
    pub engine: ClusterEngine,
    pub spectra: Vec<MSNSpectrum>,
    pub filter_unknown_charge: bool,
    pub total_spectra_ingested: usize,
    pub filter_report: FilterReport,
    /// Keys accumulated by [`ingest_file_low_mem`](Self::ingest_file_low_mem)
    /// across one or more files, pending [`finalize_low_mem_partitions`](Self::finalize_low_mem_partitions).
    pub low_mem_keys: Vec<SpillKey>,
    /// Set by the first [`ingest_file_low_mem`](Self::ingest_file_low_mem) call;
    /// where shard and partition files live.
    pub low_mem_spill_dir: Option<PathBuf>,
}

/// Prints a partition-size distribution to stderr right after
/// [`SpectralExperiment::finalize_low_mem_partitions`] computes partition
/// boundaries — before any of the (potentially very slow) clustering work
/// starts. Merging multiple datasets into one low-mem run means spectra that
/// used to fall in separate, dataset-local partitions can now land in the
/// same `(charge, neutral_mass)` window, and clustering cost per partition
/// scales worse than linearly with its size — so knowing up front whether a
/// handful of oversized partitions are about to dominate the run is the
/// difference between "waiting" and "waiting and knowing what for".
fn log_partition_size_summary(
    sorted: &[SpillKey],
    valid_windows: &[(usize, usize)],
    approximate_threshold: usize,
) {
    if valid_windows.is_empty() {
        eprintln!("partitions: none");
        return;
    }
    let mut sizes: Vec<usize> = valid_windows.iter().map(|&(s, e)| e - s).collect();
    sizes.sort_unstable();

    let n = sizes.len();
    let total: usize = sizes.iter().sum();
    let median = sizes[n / 2];
    let p95 = sizes[(n * 95 / 100).min(n - 1)];
    let p99 = sizes[(n * 99 / 100).min(n - 1)];
    let max = sizes[n - 1];
    let over_threshold = sizes.iter().filter(|&&s| s > approximate_threshold).count();

    eprintln!(
        "partitions: {n} total, {total} spectra, sizes[min={} median={median} p95={p95} p99={p99} max={max}], {over_threshold} over approximate_threshold={approximate_threshold} (dense DBSCAN below, approximate graph DBSCAN above)",
        sizes[0],
    );
    if over_threshold > 0 {
        // Report (size, charge, neutral-mass range) for the 10 largest
        // windows so a partition-size anomaly can be traced back to *what*
        // spectra are in it -- a specific charge state and a narrow mass
        // band, or something wider, tells a very different story about
        // whether an oversized partition is expected (biological/technical
        // replicate density) or a partitioning bug.
        let mut by_size: Vec<&(usize, usize)> = valid_windows.iter().collect();
        by_size.sort_unstable_by_key(|&&(s, e)| std::cmp::Reverse(e - s));
        eprintln!("partitions: 10 largest (size, charge, neutral_mass range):");
        for &&(start, end) in by_size.iter().take(10) {
            let window = &sorted[start..end];
            let charge = window[0].charge;
            let min_mass = window
                .iter()
                .map(|k| k.neutral_mass)
                .fold(f64::INFINITY, f64::min);
            let max_mass = window
                .iter()
                .map(|k| k.neutral_mass)
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  size={}, charge={charge}, neutral_mass=[{min_mass:.2}, {max_mass:.2}]",
                end - start
            );
        }
    }
}

/// (n_spectra, elapsed_ms, used_approximate) collected for one partition by
/// [`SpectralExperiment::cluster_and_write_low_mem_with_quality`].
type PartitionStats = (usize, u128, bool);

/// Prints the post-hoc partition timing report: total wall time actually
/// spent inside per-partition clustering (summed across parallel workers, so
/// this can exceed real elapsed time — that gap is a rough measure of how
/// well the per-partition parallelism kept cores fed), a dense-vs-approximate
/// time split, and the slowest partitions by elapsed time — since a load
/// imbalance from one or two oversized partitions dominating wall-clock time
/// looks identical to "still running" from the outside otherwise.
fn log_partition_timing_summary(stats: &[PartitionStats], overall_elapsed: std::time::Duration) {
    if stats.is_empty() {
        return;
    }
    let total_cpu_ms: u128 = stats.iter().map(|&(_, ms, _)| ms).sum();
    let (approx_count, approx_ms): (usize, u128) = stats
        .iter()
        .filter(|&&(_, _, approx)| approx)
        .fold((0, 0), |(c, ms), &(_, e, _)| (c + 1, ms + e));
    let dense_ms = total_cpu_ms - approx_ms;

    eprintln!(
        "partition timing: {} partitions, {:.1}s wall, {:.1}s summed per-partition CPU time (dense={:.1}s, approximate={:.1}s over {approx_count} partitions)",
        stats.len(),
        overall_elapsed.as_secs_f64(),
        total_cpu_ms as f64 / 1000.0,
        dense_ms as f64 / 1000.0,
        approx_ms as f64 / 1000.0,
    );

    let mut by_elapsed: Vec<(usize, &PartitionStats)> = stats.iter().enumerate().collect();
    by_elapsed.sort_unstable_by(|a, b| b.1.1.cmp(&a.1.1));
    eprintln!("partition timing: slowest partitions (idx, n_spectra, elapsed_s, path):");
    for (idx, stats) in by_elapsed.iter().take(20) {
        let (n_spectra, ms, approx) = **stats;
        eprintln!(
            "  #{idx}: n_spectra={n_spectra}, elapsed={:.2}s, path={}",
            ms as f64 / 1000.0,
            if approx { "approximate" } else { "dense" }
        );
    }
}

/// Known spectral file extensions, in the order checked. Both parsers return
/// the same `Vec<MassSpectralRecord>`, so callers can treat any mix of
/// these formats identically once parsed.
const SPECTRAL_FILE_EXTENSIONS: [&str; 3] = ["mgf", "mzml", "mzML"];

fn has_known_spectral_extension(path: &str) -> bool {
    SPECTRAL_FILE_EXTENSIONS
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

/// Strips whichever known spectral extension a filename ends with, for
/// deriving a display/lookup name from a file path (e.g. "run1.mzML" ->
/// "run1"). Falls back to the original name unchanged if none match.
fn strip_spectral_extension(file_stem: &str) -> &str {
    for ext in SPECTRAL_FILE_EXTENSIONS {
        let suffix = format!(".{ext}");
        if let Some(stripped) = file_stem.strip_suffix(&suffix) {
            return stripped;
        }
    }
    file_stem
}

/// Parses a single spectral file with the parser matching its extension
/// (case-insensitive on `.mzml`/`.mzML`; `.mgf` otherwise). `parallel`
/// controls each parser's own internal per-file parallelism -- pass `false`
/// when the caller already parallelizes across files (e.g. rayon over
/// multiple files at once), to avoid oversubscribing threads.
fn parse_spectral_file(path: &str, parallel: bool) -> io::Result<Vec<MassSpectralRecord>> {
    if path.to_lowercase().ends_with(".mzml") {
        let mut config = MzmlParserConfig::default();
        config.use_parallel_parsing = parallel;
        MzmlParser::new(config).parse_file(path)
    } else {
        let mut parser = MgfParser::default();
        parser.config.use_parallel_parsing = parallel;
        parser.parse_file(path)
    }
}

impl SpectralExperiment {
    pub fn default() -> Self {
        let mut workflow = WorkFlowManager::default();
        workflow.add_operation(FilterMZ::new(101.0, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        // workflow.add_operation(RemoveCommonContaminants::new(MassTolerance::Ppm(20.0)));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        Self {
            workflow,
            filter_unknown_charge: true,
            ..Default::default()
        }
    }

    /// Proteomics preset, tuned by a 2-level factorial screen + a follow-up
    /// fine-grained epsilon sweep on three label-free human proteomics datasets
    /// (PXD000561, PXD000865, PXD004452) against MS-GF+ database-search ground
    /// truth (see `ms-cluster-benchmark/doe/README.md` and
    /// `oxidems_lib/publication/main.tex`, "Tuning and Validation of the
    /// Proteomics Preset"):
    ///   - `epsilon` 0.05: an earlier pass scored the epsilon sweep with a plain
    ///     `purity * completeness` product and concluded 0.3 was best. That metric
    ///     is badly misleading near the purity ceiling this domain operates in --
    ///     e.g. on PXD000561, purity 0.997->0.941 "looks like" a 5.6-point dip, but
    ///     is actually an ~19x increase in the clustering error rate (1-purity)
    ///     for only ~1.6x more completeness. Re-scored by completeness/(1-purity)
    ///     (completeness earned per unit of error, respecting that error rate is
    ///     the thing with real multiplicative cost) across a fine 9-point sweep
    ///     (0.01 to 0.3), 0.05 is a clean interior optimum on *all three* real
    ///     datasets simultaneously -- lower values (0.01-0.03) score worse too, so
    ///     this isn't a boundary artifact either direction.
    ///   - no intensity transform, `top_n_peaks` 50, `min_peaks`/`min_mz_range`
    ///     confirmed to have zero measurable effect on purity or completeness for
    ///     this domain (identical outputs across every tested level) -- kept at the
    ///     shared baseline's values, unaffected by the epsilon correction above.
    ///   - `min_mz` 200.0: Falcon's own default low-mass cutoff (101.0) was
    ///     tested as a controlled change on PXD021518 (matching both the
    ///     workflow peak filter AND the hasher's own mz range -- the CLI's
    ///     `--min-mz` override only touches the former, which gave a
    ///     misleadingly different, invalid result during investigation) and
    ///     did not improve RedshiftMS's purity-vs-N10 AUC relative to Falcon
    ///     for either hashed or exact mode; kept at 200.0.
    pub fn proteomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        let min_mz = 200.0;
        let max_mz = 1500.0;
        workflow.add_operation(FilterMZ::new(min_mz, max_mz));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(RemoveCommonContaminants::new(MassTolerance::Ppm(20.0)));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        let mut engine = ClusterEngine::default();
        engine.set_mz_range(min_mz, max_mz);
        engine.epsilon = 0.05;

        Self {
            workflow,
            engine,
            parser: MgfParser::default(),
            filter_unknown_charge: true,
            ..Default::default()
        }
    }

    /// Metabolomics preset, tuned by a full-factorial DOE (256 runs, mirroring the
    /// proteomics preset's methodology) against the real, charge-filtered Astral
    /// dataset scored by the internal MS-RT proxy (`average_weighted_cluster_purity`/
    /// `calculate_n10_value`), cross-checked against the ALL_GNPS community spectral
    /// library (InChIKey ground truth). See `oxidems_lib/publication/main.tex`
    /// ("Tuning and Validation of the Metabolomics Preset") for the full derivation.
    ///   - `epsilon` 0.5: scored by completeness/(1-purity) (completeness earned per
    ///     unit of clustering error -- see the proteomics preset's docs for why the
    ///     naive purity*completeness product is misleading), this score rises
    ///     monotonically from 0.02 through 0.5 (0.91->2.04) then falls back through
    ///     1.0 (1.97 at eps=0.7, 1.97 at eps=1.0), making 0.5 the real interior
    ///     optimum -- notably not the same value as the proteomics preset (0.05),
    ///     confirming these two domains genuinely need different operating points
    ///     rather than sharing one.
    ///   - no intensity transform: on the same real MS-RT ground truth, `none` clearly
    ///     beats `sqrt` on completeness (0.147 vs. 0.079) at equal purity (~0.91) --
    ///     the opposite of what the now-invalidated InChIKey-library comparison found,
    ///     which is exactly why this preset was re-derived against real acquisition
    ///     data instead.
    ///   - `min_mz_range` 50 Da: still the dominant completeness lever (100 Da discards
    ///     far more small-molecule spectra), confirmed again on this new ground truth.
    ///   - relaxed `min_peaks` 3, no relative-intensity floor, `top_n_peaks` 50: all three
    ///     confirmed to have negligible-to-zero measurable effect on this domain.
    ///   - 20 ppm precursor tolerance, 0.05 Da fragment tolerance: both confirmed inert
    ///     within the tested range, kept for consistency with the proteomics preset.
    pub fn metabolomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        let min_mz = 50.0;
        let max_mz = 1500.0;
        workflow.add_operation(FilterMZ::new(min_mz, max_mz));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(3));
        workflow.add_operation(MinMZRangeFilter::new(50.0));

        let mut engine = ClusterEngine::default();
        engine.set_mz_range(min_mz, max_mz);
        engine.mz_tolerance = MassTolerance::Ppm(20.0);
        engine.epsilon = 0.5;

        Self {
            workflow,
            engine,
            filter_unknown_charge: true,
            ..Default::default()
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
        let purities = self.calculate_cluster_purities(clusters, mz_tol, rt_tol);

        let (product, sum) = clusters
            .iter()
            .zip(purities)
            .fold((0.0, 0), |acc, (cluster, purity)| {
                (acc.0 + cluster.len() as f64 * purity, acc.1 + cluster.len())
            });
        product / sum as f64
    }

    pub fn average_cluster_purity(&self, clusters: &FlatClusters, mz_tol: f64, rt_tol: f32) -> f64 {
        let purities = self.calculate_cluster_purities(clusters, mz_tol, rt_tol);

        let sum = purities.iter().sum::<f64>();
        sum / purities.len() as f64
    }

    pub fn calculate_cluster_purity(&self, cluster_ids: &[usize], mz_tol: f64, rt_tol: f32) -> f64 {
        let n = cluster_ids.len();
        if n <= 1 {
            return 1.0;
        }

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

        let mut file_assignments = Vec::with_capacity(n);
        for &id in cluster_ids {
            let f_idx = self.metadata.get(id).unwrap().file_index;
            file_assignments.push(f_idx);
        }

        for i in 0..n {
            for j in i + 1..n {
                if file_assignments[i] == file_assignments[j] {
                    let s1 = self.metadata.get(cluster_ids[i]).unwrap();
                    let s2 = self.metadata.get(cluster_ids[j]).unwrap();

                    let mz_diff = (s1.precursor_mz - s2.precursor_mz).abs();
                    let rt_diff =
                        (s1.retention_time_in_seconds - s2.retention_time_in_seconds).abs();

                    if mz_diff <= mz_tol && rt_diff <= rt_tol {
                        unite(i, j, &mut parent, &mut component_size);
                    }
                }
            }
        }

        let mut file_max_components = vec![0usize; self.file_names.len()];

        for i in 0..n {
            let root = find(i, &mut parent);
            let f_idx = file_assignments[i];
            let size = component_size[root];

            if size > file_max_components[f_idx] {
                file_max_components[f_idx] = size;
            }
        }

        let sum_max_components: usize = file_max_components.iter().sum();

        sum_max_components as f64 / n as f64
    }

    /// Computes optional internal-validation quality metrics per cluster: average and
    /// minimum cosine similarity of member spectra to their cluster's consensus vector.
    ///
    /// The consensus is the mean of the member hashed vectors, L2-renormalized, in the
    /// same hashed vector space `ClusterEngine` uses for clustering. Cost is
    /// `O(N * hash_len)` per cluster — no pairwise (`O(N^2)`) comparisons — so this is
    /// cheap enough to run even on very large clusters. It is not called by
    /// [`ClusterEngine::cluster`](super::engine::ClusterEngine::cluster); call it
    /// separately on the resulting [`FlatClusters`] only when quality reporting is wanted.
    ///
    /// `spectra` must be indexable by the ids stored in `clusters` (i.e. `spectra[id]`
    /// is the spectrum with that id) — note [`Self::prepare_clustering`] drains
    /// `self.spectra`, so callers who want quality metrics must keep their own copy of
    /// the spectra (e.g. clone before clustering) to pass in here.
    ///
    /// This identity (`id` == position in `spectra`) only holds for the normal in-RAM
    /// path, where [`ingest_files_parallel`](Self::ingest_files_parallel) reassigns ids
    /// to dense positions at the end. It does **not** hold for ids coming out of the
    /// low-mem partitioned path (see
    /// [`cluster_and_write_low_mem_with_quality`](Self::cluster_and_write_low_mem_with_quality)),
    /// where ids are a global counter assigned at parse time — use
    /// [`Self::cluster_quality_metrics_indexed`] there instead.
    pub fn calculate_cluster_quality_metrics(
        &self,
        clusters: &FlatClusters,
        spectra: &[MSNSpectrum],
    ) -> Vec<ClusterQualityMetrics> {
        Self::cluster_quality_metrics_indexed(
            &self.engine.hasher,
            self.engine.hash_size(),
            clusters,
            spectra,
            |id| id,
        )
    }

    /// Core of [`calculate_cluster_quality_metrics`](Self::calculate_cluster_quality_metrics),
    /// generalized over how a cluster member id maps to a position in `spectra` via
    /// `resolve`. The normal path uses the identity (`id` is already a position); the
    /// low-mem path resolves through a `HashMap<u64, usize>` since ids there are a
    /// global parse-time counter, not a position within any one partition's `spectra`.
    fn cluster_quality_metrics_indexed(
        hasher: &SpectrumHasher,
        hash_size: usize,
        clusters: &FlatClusters,
        spectra: &[MSNSpectrum],
        resolve: impl Fn(usize) -> usize + Sync,
    ) -> Vec<ClusterQualityMetrics> {
        clusters
            .offsets
            .par_windows(2)
            .filter_map(|w| {
                let (start, end) = (w[0], w[1]);
                if end <= start {
                    return None;
                }
                let ids = &clusters.data[start..end];
                let n = ids.len();

                if n <= 1 {
                    return Some(ClusterQualityMetrics {
                        size: n,
                        avg_cohesion: 1.0,
                        min_cohesion: 1.0,
                    });
                }

                let mut vectors = vec![0.0f32; n * hash_size];
                for (chunk, &id) in vectors.chunks_mut(hash_size).zip(ids) {
                    let spectrum = &spectra[resolve(id)];
                    hasher.into_vector_native(spectrum.spectrum.peaks.as_slice(), true, chunk);
                }

                let mut consensus = vec![0.0f32; hash_size];
                for chunk in vectors.chunks(hash_size) {
                    for (c, &v) in consensus.iter_mut().zip(chunk) {
                        *c += v;
                    }
                }
                let norm_sq: f32 = consensus.iter().map(|&x| x * x).sum();
                if norm_sq > 0.0 {
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    for c in consensus.iter_mut() {
                        *c *= inv_norm;
                    }
                }

                let mut avg_cohesion = 0.0f64;
                let mut min_cohesion = f32::INFINITY;
                for chunk in vectors.chunks(hash_size) {
                    let sim = fast_dot_product(chunk, &consensus);
                    avg_cohesion += sim as f64;
                    min_cohesion = min_cohesion.min(sim);
                }
                avg_cohesion /= n as f64;

                Some(ClusterQualityMetrics {
                    size: n,
                    avg_cohesion,
                    min_cohesion: min_cohesion as f64,
                })
            })
            .collect()
    }

    pub fn ingest_files_parallel(&mut self, folder_path: &str) {
        self.workflow.use_parallel = false;
        let paths: Vec<_> = std::fs::read_dir(folder_path)
            .unwrap()
            .filter_map(|e| {
                let path = e.ok()?.path();
                let s = path.to_str()?.to_string();
                if has_known_spectral_extension(&s) { Some(s) } else { None }
            })
            .collect();

        let total_disk_size: u64 = paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();

        let estimated_total_spectra = (total_disk_size / MGF_SPECTRA_SIZE as u64) as usize;

        self.spectra.reserve(estimated_total_spectra);
        self.metadata.reserve(estimated_total_spectra);

        let start_index = self.file_names.len();

        self.file_names.extend(paths.iter().map(|p| {
            strip_spectral_extension(p.split('/').last().unwrap()).to_string()
        }));
        self.file_paths.extend(paths.iter().map(PathBuf::from));

        let start_time = std::time::Instant::now();
        let results: Vec<(Vec<MSNSpectrum>, Vec<SpectrumMetadata>, usize, FilterReport)> = paths
            .into_par_iter()
            .enumerate()
            .map(|(i, path)| {
                let current_file_index = start_index + i;
                self.process_single_file(&path, current_file_index)
            })
            .collect();

        for (mut file_spectra, mut file_metadata, raw_count, report) in results {
            self.total_spectra_ingested += raw_count;
            self.metadata.append(&mut file_metadata);
            self.spectra.append(&mut file_spectra);
            self.filter_report.merge(report);
        }
        self.spectra
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, spectrum)| spectrum.id = index as u64);
    }

    fn process_single_file(
        &self,
        path: &str,
        file_index: usize,
    ) -> (Vec<MSNSpectrum>, Vec<SpectrumMetadata>, usize, FilterReport) {
        let mut spectra = parse_spectral_file(path, false).unwrap();

        let raw_count = spectra.len();
        let mut report = self.workflow.execute_with_report(&mut spectra);

        let unknown_charge_dropped = if self.filter_unknown_charge {
            spectra
                .iter()
                .filter(|s| !s.peaks().is_empty() && s.get_charge() == 0)
                .count()
        } else {
            0
        };
        report.record("unknown_charge", unknown_charge_dropped);

        let mut local_metadata = Vec::new();
        let processed_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| {
                !s.peaks().is_empty()
                    && if self.filter_unknown_charge {
                        s.get_charge() != 0
                    } else {
                        true
                    }
            })
            .enumerate()
            .map(|(local_idx, s)| {
                // SCANS is sometimes a range/list (MSn-to-pseudo-MS2 spectra
                // merged from several underlying scans), which doesn't parse
                // as a plain integer -- fall back to a per-record unique id
                // rather than a constant, or every such spectrum in the file
                // would collapse onto the same scan_number (see also
                // mgf.rs::parse_file_low_mem).
                let scan_number = s
                    .get_metadata("SCANS")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(local_idx);
                let source_byte_start = s.source_byte_start;
                let source_byte_end = s.source_byte_end;
                let mut msn = s.to_msn();

                msn.id = local_idx as u64;

                local_metadata.push(SpectrumMetadata {
                    precursor_mz: msn.precursor_mz,
                    retention_time_in_seconds: msn.retention_time_in_seconds,
                    charge: msn.charge(),
                    scan_number,
                    file_index,
                    source_byte_start,
                    source_byte_end,
                });

                msn
            })
            .collect();

        (processed_spectra, local_metadata, raw_count, report)
    }

    pub fn ingest_file(&mut self, file_path: &str) {
        let file_index = self.file_names.len();
        self.file_names.push(
            strip_spectral_extension(file_path.split("/").last().unwrap()).to_string(),
        );
        self.file_paths.push(PathBuf::from(file_path));

        let mut spectra = parse_spectral_file(file_path, true).unwrap();
        self.total_spectra_ingested += spectra.len();
        let mut report = self.workflow.execute_with_report(&mut spectra);

        let unknown_charge_dropped = if self.filter_unknown_charge {
            spectra
                .iter()
                .filter(|s| !s.peaks().is_empty() && s.get_charge() == 0)
                .count()
        } else {
            0
        };
        report.record("unknown_charge", unknown_charge_dropped);
        self.filter_report.merge(report);

        let current_index = self.metadata.len();

        let msn_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| {
                !(s.peaks().len() == 0)
                    && if self.filter_unknown_charge {
                        s.get_charge() != 0
                    } else {
                        true
                    }
            })
            .enumerate()
            .map(|(index, mut s)| {
                s.id = current_index + index;
                (
                    // See the fallback note on the analogous SCANS parse above.
                    s.get_metadata("SCANS")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(current_index + index),
                    s.source_byte_start,
                    s.source_byte_end,
                    s.to_msn(),
                )
            })
            .inspect(|(scan_number, source_byte_start, source_byte_end, msn)| {
                self.metadata.push(SpectrumMetadata {
                    precursor_mz: msn.precursor_mz,
                    retention_time_in_seconds: msn.retention_time_in_seconds,
                    charge: msn.charge(),
                    scan_number: *scan_number,
                    file_index,
                    source_byte_start: *source_byte_start,
                    source_byte_end: *source_byte_end,
                });
            })
            .map(|(_, _, _, msn_spectrum)| msn_spectrum)
            .collect();

        self.spectra.extend(msn_spectra)
    }

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

    /// Low-memory alternative to [`ingest_file`](Self::ingest_file): parses
    /// and filters spectra the same way, but instead of accumulating them in
    /// `self.spectra`, spills each kept spectrum to disk under `spill_dir` as
    /// it's produced and keeps only a small [`SpillKey`] per spectrum
    /// resident. Peak memory during ingestion no longer scales with dataset
    /// size — see [`MgfParser::parse_file_low_mem`] for how.
    ///
    /// Can be called multiple times (once per input file) before calling
    /// [`finalize_low_mem_partitions`](Self::finalize_low_mem_partitions),
    /// the same way multiple [`ingest_file`](Self::ingest_file) calls
    /// accumulate into one `self.spectra`.
    ///
    /// This is an opt-in alternate path: `self.spectra` is untouched, so
    /// mixing this with [`ingest_file`](Self::ingest_file) on the same
    /// experiment works, but [`prepare_clustering`](Self::prepare_clustering)
    /// and [`cluster`](Self::cluster) only ever see what `self.spectra`
    /// holds — they know nothing about spilled partitions. Feed partitions
    /// into a `ClusterEngine` yourself after loading them with
    /// [`load_partition`](Self::load_partition).
    ///
    /// Unlike [`ingest_file`](Self::ingest_file), spilled `MSNSpectrum::id`s
    /// are parse-order based, not densely renumbered over kept spectra, and
    /// no corresponding `self.metadata` entries are populated (there is
    /// nothing yet for a spilled spectrum's id to index into).
    pub fn ingest_file_low_mem(&mut self, file_path: &str, spill_dir: &Path) -> io::Result<()> {
        let file_index = self.file_names.len();
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

        let (keys, total_parsed) = self.parser.parse_file_low_mem(
            file_path,
            file_index,
            spill_dir,
            &self.workflow,
            self.filter_unknown_charge,
            self.engine.hasher.min_mz,
            self.engine.hasher.max_mz,
        )?;
        self.total_spectra_ingested += total_parsed;
        self.low_mem_spill_dir = Some(spill_dir.to_path_buf());
        self.low_mem_keys.extend(keys);
        Ok(())
    }

    /// Computes m/z/charge partitions over every key accumulated by
    /// [`ingest_file_low_mem`](Self::ingest_file_low_mem) so far (using
    /// `self.engine.mz_tolerance`, the same tolerance
    /// [`prepare_clustering`](Self::prepare_clustering) would use), then
    /// redistributes the spilled records — copying already-serialized bytes,
    /// no re-parsing or deserializing — into one file per partition under
    /// `spill_dir/partitions/`. Returns each partition's file path (loadable
    /// with [`load_partition`](Self::load_partition)) alongside its spectrum
    /// count — known for free from the window bounds, and needed by callers
    /// that want to treat oversized partitions differently (see
    /// [`cluster_and_write_low_mem_with_quality`](Self::cluster_and_write_low_mem_with_quality))
    /// without a separate pass to load and count each partition first.
    ///
    /// Consumes `self.low_mem_keys`; call this once after all
    /// `ingest_file_low_mem` calls for the current batch are done.
    pub fn finalize_low_mem_partitions(&mut self) -> io::Result<Vec<(PathBuf, usize)>> {
        let spill_dir = self.low_mem_spill_dir.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "finalize_low_mem_partitions: no ingest_file_low_mem call has run yet",
            )
        })?;
        let partitions_dir = spill_dir.join("partitions");
        let _ = fs::remove_dir_all(&partitions_dir);
        fs::create_dir_all(&partitions_dir)?;

        let mut sorted = std::mem::take(&mut self.low_mem_keys);
        sorted.sort_unstable_by(|a, b| {
            (a.charge, a.precursor_mz)
                .partial_cmp(&(b.charge, b.precursor_mz))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let boundaries = mz_charge_boundaries(
            &sorted,
            |k: &SpillKey| k.charge,
            |k: &SpillKey| k.neutral_mass,
            self.engine.mz_tolerance,
        );

        // Matches ClusterEngine::split_on_mz_and_charge: a group of one
        // spectrum has no possible cluster partner, so it's dropped rather
        // than given its own batch/partition. Filtered up front (sequential,
        // but just integer comparisons — no I/O) so parallel tasks below can
        // derive stable, gap-free partition filenames from their index.
        let valid_windows: Vec<(usize, usize)> = boundaries
            .windows(2)
            .filter_map(|w| {
                let (start, end) = (w[0], w[1]);
                (end - start > 1).then_some((start, end))
            })
            .collect();

        log_partition_size_summary(&sorted, &valid_windows, self.engine.approximate_threshold);

        // Each partition's redistribution is independent I/O (its own
        // output file, random-access reads from shard files); parallelizing
        // this across partitions is what actually keeps cores busy here
        // instead of the file I/O serializing everything.
        //
        // Shard files are opened exactly once, up front, and shared
        // read-only across every partition task via positional reads
        // (`read_exact_at`/`pread`, which needs no seek and no lock since
        // each call is independent of any shared cursor). This used to open
        // a fresh `File` per (task, shard) pair instead — fine when shards
        // were scoped one-per-input-file (tens of thousands of them, each
        // touched by relatively few partitions), but with
        // [`MgfParser::open_shard_pool`]'s small fixed shard pool (a few
        // dozen shards for the whole corpus) nearly every partition touches
        // nearly every shard, so per-task reopening turned into repeatedly
        // reopening the same handful of (now much larger) files from
        // thousands of partition tasks.
        let num_shards = sorted
            .iter()
            .map(|k| k.shard)
            .max()
            .map_or(0, |max_shard| max_shard + 1);
        let shard_files: Vec<File> = (0..num_shards)
            .map(|shard_idx| File::open(spill_dir.join(format!("shard_{shard_idx:06}.bin"))))
            .collect::<io::Result<Vec<File>>>()?;

        let partition_paths: Vec<(PathBuf, usize)> = valid_windows
            .par_iter()
            .enumerate()
            .map(|(partition_id, &(start, end))| -> io::Result<(PathBuf, usize)> {
                let partition_path = partitions_dir.join(format!("part_{partition_id:06}.bin"));
                let mut writer = BufWriter::new(File::create(&partition_path)?);
                let mut buf = Vec::new();

                for key in &sorted[start..end] {
                    let file = &shard_files[key.shard as usize];
                    buf.resize(key.len as usize, 0);
                    // +4 to skip this record's own length prefix in the shard file.
                    file.read_exact_at(&mut buf, key.offset + 4)?;

                    writer.write_all(&(buf.len() as u32).to_le_bytes())?;
                    writer.write_all(&buf)?;
                }
                writer.flush()?;
                Ok((partition_path, end - start))
            })
            .collect::<io::Result<Vec<(PathBuf, usize)>>>()?;

        Ok(partition_paths)
    }

    /// Reads back a partition file written by
    /// [`finalize_low_mem_partitions`](Self::finalize_low_mem_partitions)
    /// into an in-memory `Vec<LowMemRecord>` — each holding an
    /// [`MSNSpectrum`] plus the scan number / file index normal ingestion
    /// would have tracked in `self.metadata`.
    pub fn load_partition(partition_path: &Path) -> io::Result<Vec<LowMemRecord>> {
        let mut reader = BufReader::new(File::open(partition_path)?);
        let mut records = Vec::new();
        let mut len_buf = [0u8; 4];
        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            let record: LowMemRecord = rkyv::from_bytes::<LowMemRecord, rkyv::rancor::Error>(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            records.push(record);
        }
        Ok(records)
    }

    /// Builds a fresh, independent [`ClusterEngine`] with the same settings
    /// as `self.engine` (tolerance, epsilon, hasher, exact, RT tolerance,
    /// etc.) but empty scratch buffers — cheap to construct since those
    /// buffers grow on demand per call, not copied from whatever size
    /// `self.engine`'s happen to be. Lets partitions be clustered
    /// concurrently without contending on `self.engine`'s shared buffers.
    fn engine_like_self(&self) -> ClusterEngine {
        let mut engine = ClusterEngine::default();
        engine.mz_tolerance = self.engine.mz_tolerance;
        engine.rt_tolerance = self.engine.rt_tolerance;
        engine.epsilon = self.engine.epsilon;
        engine.batch_size_threshold = self.engine.batch_size_threshold;
        engine.approximate_threshold = self.engine.approximate_threshold;
        engine.exact = self.engine.exact;
        engine.min_neighbors = self.engine.min_neighbors;
        engine.hasher = SpectrumHasher::new(
            self.engine.hasher.min_mz,
            self.engine.hasher.max_mz,
            self.engine.hasher.tolerance,
            self.engine.hasher.hash_len,
        );
        engine
    }

    /// Full low-memory pipeline: [`finalize_low_mem_partitions`](Self::finalize_low_mem_partitions),
    /// then clusters every partition **in parallel** — each on its own
    /// [`ClusterEngine`] (see [`engine_like_self`](Self::engine_like_self)),
    /// since partitions are independent by construction and a single
    /// sequential loop reusing `self.engine` left 32 cores mostly idle while
    /// paying per-call setup cost thousands of times over. Results are
    /// written in partition order in one final sequential pass — cheap
    /// relative to clustering, and keeps the global cluster counter simple.
    ///
    /// Output format matches [`write_results`](Self::write_results).
    /// Returns `(total spectra written across all clusters, total cluster
    /// count)`, mirroring what a caller would compute from
    /// [`cluster`](Self::cluster)'s returned `FlatClusters` in the normal
    /// path.
    pub fn cluster_and_write_low_mem(
        &mut self,
        output_file_name: &str,
    ) -> io::Result<(usize, usize)> {
        let partition_paths = self.finalize_low_mem_partitions()?;

        type ClusterRow = (usize, usize, i8, f64, f32); // file_index, scan_number, charge, precursor_mz, rt

        let process_partition = |partition_path: &Path| -> io::Result<Vec<Vec<ClusterRow>>> {
            let records = Self::load_partition(partition_path)?;
            let mut lookup: HashMap<u64, ClusterRow> = HashMap::with_capacity(records.len());
            let mut msn_spectra = Vec::with_capacity(records.len());
            for r in records {
                lookup.insert(
                    r.msn.id,
                    (
                        r.file_index,
                        r.scan_number,
                        r.msn.charge(),
                        r.msn.precursor_mz,
                        r.msn.retention_time_in_seconds,
                    ),
                );
                msn_spectra.push(r.msn);
            }

            let mut engine = self.engine_like_self();
            engine.split_on_mz_and_charge(msn_spectra).unwrap();
            let clusters = engine.cluster().unwrap();

            Ok(clusters
                .iter()
                .map(|cluster| {
                    cluster
                        .iter()
                        .map(|id| lookup[&(*id as u64)])
                        .collect::<Vec<ClusterRow>>()
                })
                .collect())
        };

        // Mirrors ClusterEngine::cluster()'s own fast_batches/slow_batches
        // split: a partition at or above batch_size_threshold pays a
        // per-partition O(n^2) dense-matrix cost for hashed mode
        // (compute_dense_dbscan) -- processing many such partitions
        // concurrently (previously: one flat par_iter over ALL partitions,
        // any size) let their peak memory sum instead of being bounded by
        // "at most one large partition resident at a time", the property
        // normal (non-low-mem) clustering already enforces by processing
        // slow_batches in a plain sequential loop. Small partitions stay
        // fully parallel; this only serializes the ones expensive enough
        // to matter.
        let batch_size_threshold = self.engine.batch_size_threshold;
        let (small_partitions, large_partitions): (Vec<_>, Vec<_>) = partition_paths
            .into_iter()
            .partition(|&(_, n_spectra)| n_spectra < batch_size_threshold);

        let mut per_partition: Vec<io::Result<Vec<Vec<ClusterRow>>>> = small_partitions
            .par_iter()
            .map(|(partition_path, _n_spectra)| process_partition(partition_path))
            .collect();
        for (partition_path, _n_spectra) in &large_partitions {
            per_partition.push(process_partition(partition_path));
        }

        let file = File::create(output_file_name)?;
        let mut writer = BufWriter::new(file);
        write!(
            writer,
            "# RedshiftMS (OxideMS core v{}) {}\n# eps = {}\n# hash_len = {}\n# total_spectra_parsed = {}\nidentifier,precursor_charge,precursor_mz,retention_time,cluster\n",
            env!("CARGO_PKG_VERSION"),
            if self.engine.exact { "(exact)" } else { "" },
            self.engine.epsilon,
            self.engine.hash_size(),
            self.total_spectra_ingested
        )?;

        let mut counter = 0usize;
        let mut total_clustered = 0usize;
        for partition_clusters in per_partition {
            for cluster in partition_clusters? {
                counter += 1;
                for (file_index, scan_number, charge, precursor_mz, rt) in cluster {
                    total_clustered += 1;
                    writeln!(
                        writer,
                        "mzspec:USI000000:{}:scan:{},{},{},{},{}",
                        &self.file_names[file_index],
                        scan_number,
                        charge,
                        precursor_mz,
                        rt,
                        counter,
                    )?;
                }
            }
        }
        writer.flush()?;
        Ok((total_clustered, counter))
    }

    /// Like [`cluster_and_write_low_mem`](Self::cluster_and_write_low_mem), but also
    /// computes [`ClusterQualityMetrics`] per cluster.
    ///
    /// Partitions are loaded, clustered, and dropped one at a time (in parallel across
    /// partitions, never the whole corpus at once), so the extra clone this needs for
    /// the quality pass is bounded by one partition's size, not a second full-corpus
    /// copy — this stays safe to run on datasets much larger than RAM. Cluster member
    /// ids from the low-mem path are a global parse-time counter, not a position within
    /// any one partition's spectra, so quality metrics are computed through an explicit
    /// id-to-position map via [`Self::cluster_quality_metrics_indexed`] rather than
    /// [`calculate_cluster_quality_metrics`](Self::calculate_cluster_quality_metrics)'s
    /// identity assumption.
    pub fn cluster_and_write_low_mem_with_quality(
        &mut self,
        output_file_name: &str,
    ) -> io::Result<(usize, usize, Vec<ClusterQualityMetrics>)> {
        let partition_paths = self.finalize_low_mem_partitions()?;

        let approximate_threshold = self.engine.approximate_threshold;
        let total_partitions = partition_paths.len();
        let completed = AtomicUsize::new(0);
        let overall_start = Instant::now();

        type ClusterRow = (usize, usize, i8, f64, f32); // file_index, scan_number, charge, precursor_mz, rt
        type PartitionOutput = (Vec<Vec<ClusterRow>>, Vec<ClusterQualityMetrics>, PartitionStats);

        // A partition over `approximate_threshold` triggers the approximate
        // graph-DBSCAN path, which internally parallelizes its own matmul /
        // k-means / edge-extraction steps across every rayon thread. Mixing
        // those into the same per-partition `par_iter` as the many small,
        // cheap dense partitions meant several huge partitions (or a huge
        // one plus a swarm of tiny ones) would compete for the same 56
        // threads at once -- observed directly as small dense partitions
        // that should finish in ~1s instead taking 2000s+ while stuck behind
        // a giant partition's nested parallelism, and as peak memory being
        // the *sum* of however many giant partitions' score matrices
        // (`n_vectors * n_list` floats each) happened to be in flight
        // simultaneously. Splitting into two phases fixes both: small
        // partitions run together (as before, they're cheap and don't
        // starve each other), and large partitions run one at a time so
        // each gets the full core count uncontended and only one giant
        // score matrix is ever resident at once.
        let process_one = |partition_idx: usize,
                            partition_path: &Path,
                            approximate: bool|
         -> io::Result<PartitionOutput> {
            let records = Self::load_partition(partition_path)?;
            let mut lookup: HashMap<u64, ClusterRow> = HashMap::with_capacity(records.len());
            let mut id_to_pos: HashMap<u64, usize> = HashMap::with_capacity(records.len());
            let mut msn_spectra = Vec::with_capacity(records.len());
            for r in records {
                lookup.insert(
                    r.msn.id,
                    (
                        r.file_index,
                        r.scan_number,
                        r.msn.charge(),
                        r.msn.precursor_mz,
                        r.msn.retention_time_in_seconds,
                    ),
                );
                id_to_pos.insert(r.msn.id, msn_spectra.len());
                msn_spectra.push(r.msn);
            }

            let n_spectra = msn_spectra.len();
            if approximate {
                eprintln!(
                    "partition {partition_idx}/{total_partitions}: starting, n_spectra={n_spectra} (approximate graph path, +{:.1}s since start)",
                    overall_start.elapsed().as_secs_f64()
                );
            }
            let partition_start = Instant::now();

            // Needed because `split_on_mz_and_charge` below consumes `msn_spectra`,
            // and the quality pass needs the peaks after clustering has run.
            let spectra_for_quality = msn_spectra.clone();

            let mut engine = self.engine_like_self();
            engine.split_on_mz_and_charge(msn_spectra).unwrap();
            let clusters = engine.cluster().unwrap();

            let elapsed_ms = partition_start.elapsed().as_millis();
            let done = completed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if approximate || elapsed_ms > 1000 {
                eprintln!(
                    "partition {partition_idx}/{total_partitions}: done ({done}/{total_partitions} complete), n_spectra={n_spectra}, elapsed={:.2}s, path={}",
                    elapsed_ms as f64 / 1000.0,
                    if approximate { "approximate" } else { "dense" }
                );
            }

            let quality = Self::cluster_quality_metrics_indexed(
                &self.engine.hasher,
                self.engine.hash_size(),
                &clusters,
                &spectra_for_quality,
                |id| id_to_pos[&(id as u64)],
            );

            let rows = clusters
                .iter()
                .map(|cluster| {
                    cluster
                        .iter()
                        .map(|id| lookup[&(*id as u64)])
                        .collect::<Vec<ClusterRow>>()
                })
                .collect();

            Ok((rows, quality, (n_spectra, elapsed_ms, approximate)))
        };

        let (small, large): (Vec<(usize, &(PathBuf, usize))>, Vec<(usize, &(PathBuf, usize))>) =
            partition_paths
                .iter()
                .enumerate()
                .partition(|(_, (_, n_spectra))| *n_spectra <= approximate_threshold);

        // Phase 1: small (dense-path) partitions, parallel across partitions.
        let mut results: Vec<Option<io::Result<PartitionOutput>>> = (0..total_partitions).map(|_| None).collect();
        let small_results: Vec<(usize, io::Result<PartitionOutput>)> = small
            .par_iter()
            .map(|&(idx, (path, _))| (idx, process_one(idx, path, false)))
            .collect();
        for (idx, result) in small_results {
            results[idx] = Some(result);
        }

        // Phase 2: large (approximate-path) partitions, one at a time -- each
        // still uses every core internally, just never shares them with a
        // sibling large partition or the small-partition swarm.
        for &(idx, (path, _)) in &large {
            results[idx] = Some(process_one(idx, path, true));
        }

        let per_partition: Vec<io::Result<PartitionOutput>> =
            results.into_iter().map(|r| r.expect("every partition index is covered by exactly one of the small/large phases")).collect();

        let file = File::create(output_file_name)?;
        let mut writer = BufWriter::new(file);
        write!(
            writer,
            "# RedshiftMS (OxideMS core v{}) {}\n# eps = {}\n# hash_len = {}\n# total_spectra_parsed = {}\nidentifier,precursor_charge,precursor_mz,retention_time,cluster\n",
            env!("CARGO_PKG_VERSION"),
            if self.engine.exact { "(exact)" } else { "" },
            self.engine.epsilon,
            self.engine.hash_size(),
            self.total_spectra_ingested
        )?;

        let mut counter = 0usize;
        let mut total_clustered = 0usize;
        let mut all_quality = Vec::new();
        let mut partition_stats: Vec<PartitionStats> = Vec::with_capacity(per_partition.len());
        for partition_result in per_partition {
            let (partition_clusters, quality, stats) = partition_result?;
            partition_stats.push(stats);
            all_quality.extend(quality);
            for cluster in partition_clusters {
                counter += 1;
                for (file_index, scan_number, charge, precursor_mz, rt) in cluster {
                    total_clustered += 1;
                    writeln!(
                        writer,
                        "mzspec:USI000000:{}:scan:{},{},{},{},{}",
                        &self.file_names[file_index],
                        scan_number,
                        charge,
                        precursor_mz,
                        rt,
                        counter,
                    )?;
                }
            }
        }
        writer.flush()?;
        log_partition_timing_summary(&partition_stats, overall_start.elapsed());
        Ok((total_clustered, counter, all_quality))
    }

    pub fn prepare_clustering(&mut self) {
        let spectra = std::mem::take(&mut self.spectra);
        self.engine.split_on_mz_and_charge(spectra).unwrap();
    }

    pub fn cluster(&mut self) -> FlatClusters {
        let clusters = self.engine.cluster().unwrap();

        clusters
        // let mz_values: Vec<f64> = clusters
        //     .data
        //     .iter()
        //     .map(|id| self.metadata.get(*id).unwrap().precursor_mz)
        //     .collect();
        // clusters.refine_mz_clusters(&mz_values, self.engine.mz_tolerance)
    }

    pub fn cluster_spectra(&mut self) -> FlatClusters {
        self.prepare_clustering();
        println!(
            "Largest clusters: {:?}",
            self.engine
                .fast_batches
                .iter()
                .map(|batch| batch.batch.len())
                .max()
                .unwrap_or(0)
        );
        println!(
            "Largest clusters: {:?}",
            self.engine
                .slow_batches
                .iter()
                .map(|batch| batch.batch.len())
                .max()
                .unwrap_or(0)
        );
        self.cluster()
    }

    /// Always writes the file, even when nothing clustered: a header-only CSV
    /// is a result ("no cluster met the threshold"), whereas a missing file is
    /// indistinguishable from a crash for anything downstream. The two low-mem
    /// paths have always behaved this way; this one used to return early.
    pub fn write_results(&self, clusters: &FlatClusters, output_file_name: &str) {
        let mut file = std::fs::File::create(output_file_name).unwrap();
        let mut writer = std::io::BufWriter::new(&mut file);
        writer.write(format!("# RedshiftMS (OxideMS core v{}) {}\n# eps = {}\n# hash_len = {}\n# total_spectra_parsed = {}\nidentifier,precursor_charge,precursor_mz,retention_time,cluster\n", env!("CARGO_PKG_VERSION"), if self.engine.exact {"(exact)"} else {""},  self.engine.epsilon, self.engine.hash_size(), self.total_spectra_ingested).as_bytes())
            .unwrap();
        let mut counter = 0;
        for cluster in clusters.iter() {
            counter += 1;
            for spectrum in cluster {
                let metadata = self.metadata.get(*spectrum).unwrap();
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

    /// Builds a real, peak-level consensus spectrum for one cluster's members.
    ///
    /// Reparses each member's original bytes via the byte offsets recorded in
    /// `self.metadata` (rather than the hashed feature vectors clustering
    /// used, or peaks kept resident from ingestion) and replays this run's
    /// workflow on each one, so the peaks going into the consensus match what
    /// was actually clustered. This is opt-in: nothing on the
    /// `cluster()`/`write_results` path calls it, so leaving it unused costs
    /// nothing beyond the two extra `u64` fields on `SpectrumMetadata`.
    ///
    /// `bin_width` sets how close two members' peaks must be in m/z to be
    /// merged into one consensus peak; pass the same fragment tolerance used
    /// for clustering (`self.engine.hasher.tolerance`) for a consistent
    /// notion of "the same fragment". Returns `None` if no member's bytes
    /// could be recovered (e.g. the source file has since moved).
    pub fn build_consensus_spectrum(
        &self,
        member_ids: &[usize],
        bin_width: f64,
    ) -> Option<MassSpectralRecord> {
        if member_ids.is_empty() || bin_width <= 0.0 {
            return None;
        }

        let mut mmap_cache: HashMap<usize, Mmap> = HashMap::new();
        let mut precursor_sum = 0.0f64;
        let mut rt_sum = 0.0f32;
        let mut charge_votes: HashMap<i8, usize> = HashMap::new();
        let mut raw_peaks: Vec<(f64, f32)> = Vec::new();
        let mut n_reconstructed = 0usize;

        for &member_id in member_ids {
            let Some(metadata) = self.metadata.get(member_id) else {
                continue;
            };
            let mmap = match mmap_cache.entry(metadata.file_index) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let Some(path) = self.file_paths.get(metadata.file_index) else {
                        continue;
                    };
                    let Ok(file) = File::open(path) else {
                        continue;
                    };
                    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
                        continue;
                    };
                    e.insert(mmap)
                }
            };

            let start = metadata.source_byte_start as usize;
            let end = metadata.source_byte_end as usize;
            if start >= end || end > mmap.len() {
                continue;
            }
            let Some(mut record) =
                MgfParser::parse_block_slice(&mmap[start..end], member_id as u64, false)
            else {
                continue;
            };
            self.workflow.execute(std::slice::from_mut(&mut record));
            if record.peaks().is_empty() {
                continue;
            }

            precursor_sum += metadata.precursor_mz;
            rt_sum += metadata.retention_time_in_seconds;
            *charge_votes.entry(metadata.charge).or_insert(0) += 1;
            n_reconstructed += 1;

            raw_peaks.extend(record.peaks().iter().map(|p| (p.mz, p.intensity)));
        }

        if n_reconstructed == 0 {
            return None;
        }

        let charge = charge_votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(c, _)| c)
            .unwrap_or(0);

        // Greedily merge sorted peaks into groups whose diameter (max mz -
        // min mz) stays within `bin_width` of the group's first (lowest) mz —
        // the same diameter-bounded merge `FlatClusters::refine_single_cluster`
        // uses for m/z refinement. Anchoring to the group's own first peak
        // rather than a fixed absolute grid avoids splitting two peaks a
        // fraction of `bin_width` apart into different groups just because
        // they happen to straddle a grid line.
        raw_peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut peaks: Vec<Peak> = Vec::new();
        let mut group_start_mz = raw_peaks[0].0;
        let mut group_mz_weighted_sum = 0.0f64;
        let mut group_intensity_sum = 0.0f32;
        for (mz, intensity) in raw_peaks {
            if mz - group_start_mz > bin_width {
                if group_intensity_sum > 0.0 {
                    peaks.push(Peak::new(
                        group_mz_weighted_sum / group_intensity_sum as f64,
                        group_intensity_sum / n_reconstructed as f32,
                    ));
                }
                group_start_mz = mz;
                group_mz_weighted_sum = 0.0;
                group_intensity_sum = 0.0;
            }
            group_mz_weighted_sum += mz * intensity as f64;
            group_intensity_sum += intensity;
        }
        if group_intensity_sum > 0.0 {
            peaks.push(Peak::new(
                group_mz_weighted_sum / group_intensity_sum as f64,
                group_intensity_sum / n_reconstructed as f32,
            ));
        }

        let mut consensus = MassSpectralRecord::new(peaks, charge, member_ids[0]);
        consensus.insert_metadata(
            "PEPMASS",
            &(precursor_sum / n_reconstructed as f64).to_string(),
        );
        consensus.insert_metadata("CHARGE", &format!("{}+", charge));
        consensus.insert_metadata(
            "RTINSECONDS",
            &(rt_sum / n_reconstructed as f32).to_string(),
        );
        consensus.insert_metadata(
            "TITLE",
            &format!("Consensus of {} spectra", n_reconstructed),
        );
        Some(consensus)
    }

    /// Writes one consensus spectrum per cluster (size >= 2) in `clusters` to
    /// `output_file_name` as an MGF file. See [`Self::build_consensus_spectrum`]
    /// for what `bin_width` controls and why this is safe to skip when unused.
    pub fn write_consensus_mgf(
        &self,
        clusters: &FlatClusters,
        output_file_name: &str,
        bin_width: f64,
    ) -> io::Result<()> {
        let consensus_spectra: Vec<MassSpectralRecord> = clusters
            .iter()
            .filter_map(|cluster| self.build_consensus_spectrum(cluster, bin_width))
            .collect();
        crate::io::mgf::MgfWriter::default()
            .write_parallel_to_file(output_file_name, &consensus_spectra)
    }
}

#[cfg(test)]
mod low_mem_tests {
    use super::*;

    /// The low-mem path assigns different (non-dense) ids than the normal
    /// path, so groups can't be compared by id. Both id schemes are derived
    /// from the same (charge, precursor_mz) content though, so a group's
    /// *content signature* — its spectra's (charge, neutral_mass) pairs,
    /// sorted — is what a correct partitioning must reproduce exactly,
    /// regardless of numbering scheme.
    fn group_signature(spectra: &[MSNSpectrum]) -> Vec<(i8, u64)> {
        let mut sig: Vec<(i8, u64)> = spectra
            .iter()
            .map(|s| (s.charge(), s.get_neutral_mass().to_bits()))
            .collect();
        sig.sort_unstable();
        sig
    }

    fn normalize_signatures(mut sigs: Vec<Vec<(i8, u64)>>) -> Vec<Vec<(i8, u64)>> {
        sigs.sort_unstable();
        sigs
    }

    #[test]
    fn low_mem_partitioning_matches_engine_split_on_mz_and_charge() {
        let file = "test_files/test2.mgf";
        let tmp_dir = std::env::temp_dir().join("oxidems_low_mem_test");
        let _ = fs::remove_dir_all(&tmp_dir);

        // Ground truth: the real, unmodified in-memory path.
        let mut baseline = SpectralExperiment::metabolomics();
        baseline.ingest_file(file);
        baseline.prepare_clustering();
        let mut ground_truth: Vec<Vec<(i8, u64)>> = baseline
            .engine
            .fast_batches
            .iter()
            .chain(baseline.engine.slow_batches.iter())
            .map(|b| group_signature(&b.batch))
            .collect();
        ground_truth = normalize_signatures(ground_truth);

        // Candidate: the low-mem streaming spill + partition path.
        let mut low_mem = SpectralExperiment::metabolomics();
        low_mem.ingest_file_low_mem(file, &tmp_dir).unwrap();
        let partition_paths = low_mem.finalize_low_mem_partitions().unwrap();
        let mut candidate: Vec<Vec<(i8, u64)>> = partition_paths
            .iter()
            .map(|(p, _)| {
                let msn_spectra: Vec<MSNSpectrum> = SpectralExperiment::load_partition(p)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.msn)
                    .collect();
                group_signature(&msn_spectra)
            })
            .collect();
        candidate = normalize_signatures(candidate);

        assert_eq!(
            baseline.total_spectra_ingested, low_mem.total_spectra_ingested,
            "total parsed count (pre-filter) should match"
        );
        let kept_baseline: usize = ground_truth.iter().map(|g| g.len()).sum();
        let kept_candidate: usize = candidate.iter().map(|g| g.len()).sum();
        assert_eq!(kept_baseline, kept_candidate, "kept spectra count mismatch");
        assert_eq!(
            ground_truth, candidate,
            "low-mem partitions must reproduce split_on_mz_and_charge's grouping exactly"
        );

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    /// Regression test for a real bug: shard files were named only by
    /// `shard_idx` (0..num_shards, restarting at 0 for every
    /// `ingest_file_low_mem` call), so a second file ingested into the same
    /// spill_dir would `File::create` — silently truncating and
    /// overwriting — the first file's already-spilled shard data, corrupting
    /// the first ingest's already-recorded `SpillKey` byte offsets. Fixed by
    /// making shard ids monotonic across calls (`MgfParser::next_shard_id`).
    #[test]
    fn low_mem_multi_file_ingest_does_not_corrupt_shards() {
        let file = "test_files/test1.mgf";
        let tmp_dir = std::env::temp_dir().join("oxidems_low_mem_multifile_test");
        let _ = fs::remove_dir_all(&tmp_dir);

        let mut exp = SpectralExperiment::metabolomics();
        // Same physical file ingested twice, simulating a two-file
        // directory ingest — exercises the collision directly.
        exp.ingest_file_low_mem(file, &tmp_dir).unwrap();
        let keys_after_first = exp.low_mem_keys.len();
        exp.ingest_file_low_mem(file, &tmp_dir).unwrap();
        let keys_after_second = exp.low_mem_keys.len();
        assert_eq!(
            keys_after_second,
            keys_after_first * 2,
            "second ingest should add as many kept spectra as the first"
        );

        let partition_paths = exp.finalize_low_mem_partitions().unwrap();
        let mut seen_ids = std::collections::HashSet::new();
        let mut total_loaded = 0usize;
        for (path, _) in &partition_paths {
            for record in SpectralExperiment::load_partition(path).unwrap() {
                total_loaded += 1;
                // A corrupted shard read deserializes into garbage (or
                // fails outright) rather than valid, in-range values.
                assert!(
                    record.msn.precursor_mz > 0.0 && record.msn.precursor_mz.is_finite(),
                    "corrupted precursor_mz: {}",
                    record.msn.precursor_mz
                );
                assert!(
                    !record.msn.spectrum.peaks.is_empty(),
                    "corrupted record: no peaks"
                );
                assert!(
                    seen_ids.insert(record.msn.id),
                    "duplicate id {} across partitions — shard data likely corrupted",
                    record.msn.id
                );
            }
        }
        assert_eq!(total_loaded, keys_after_second);

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}

pub fn calculate_cluster_recall(exact_labels: &[ClusterLabel], lsh_labels: &[ClusterLabel]) -> f32 {
    assert_eq!(exact_labels.len(), lsh_labels.len());
    let n = exact_labels.len();

    let mut true_positives = 0;
    let mut false_negatives = 0;

    for i in 0..n {
        for j in (i + 1)..n {
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
