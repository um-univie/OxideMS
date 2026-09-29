//! Full 8-factor 2-level factorial DOE screen (256 runs) against the real
//! Astral dataset (tests/Astral; unknown charges are kept and clustered only
//! with each other, as in every preset), scored via the
//! MS-RT proxy (calculate_n10_value / average_weighted_cluster_purity --
//! the same internal ground truth mechanism astral_figures.rs already uses
//! for the eps-only sweep, just extended to all 8 factors).
//!
//! Mirrors ms-cluster-benchmark/doe/factors/metabolomics_screen.json's grid
//! exactly (same factors, same two levels each) so results are directly
//! comparable to the GNPS/proteomics screens. Runs entirely in-process
//! (parses Astral once per combination via ingest_files_parallel -- no CLI
//! subprocess spawning), since the whole point is a from-scratch, from-real-
//! data screen using this tool's own internal MS-RT proxy, independent of
//! GNPS's InChIKey ground truth and its own known spectral-variability
//! ceiling.

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::workflow::{
    FilterIntensityRelative, KeepTopNPeaks, LogTransformIntensity, MassTolerance,
    MinMZRangeFilter, MinNumberOfPeaks, SquareRootTransformIntensity,
};

const MZ_TOL: f64 = 0.01;
const RT_TOL: f32 = 30.0;
const PATH: &str = "tests/Astral";

#[derive(Clone, Copy, Debug)]
struct Row {
    epsilon: f32,
    fragment_tolerance: f64,
    precursor_tolerance: f64,
    transform: Transform,
    top_n_peaks: usize,
    min_peaks: usize,
    min_mz_range: f64,
    min_relative_intensity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transform {
    None,
    Sqrt,
}

fn all_rows() -> Vec<Row> {
    let mut rows = Vec::with_capacity(256);
    for &epsilon in &[0.05f32, 0.3] {
        for &fragment_tolerance in &[0.02f64, 0.05] {
            for &precursor_tolerance in &[10f64, 20.0] {
                for &transform in &[Transform::None, Transform::Sqrt] {
                    for &top_n_peaks in &[50usize, 150] {
                        for &min_peaks in &[3usize, 5] {
                            for &min_mz_range in &[50f64, 100.0] {
                                for &min_relative_intensity in &[0.0f32, 0.01] {
                                    rows.push(Row {
                                        epsilon,
                                        fragment_tolerance,
                                        precursor_tolerance,
                                        transform,
                                        top_n_peaks,
                                        min_peaks,
                                        min_mz_range,
                                        min_relative_intensity,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    rows
}

fn run_row(row: &Row) -> (f64, usize, usize, f64, usize) {
    let mut experiment = SpectralExperiment::metabolomics();
    experiment.engine.epsilon = row.epsilon;
    experiment.engine.hasher.tolerance = MassTolerance::Da(row.fragment_tolerance);
    experiment.engine.mz_tolerance = MassTolerance::Ppm(row.precursor_tolerance);

    experiment.workflow.remove_operation::<SquareRootTransformIntensity>();
    experiment.workflow.remove_operation::<LogTransformIntensity>();
    if row.transform == Transform::Sqrt {
        experiment
            .workflow
            .add_operation(SquareRootTransformIntensity::new());
    }
    experiment
        .workflow
        .replace_operation(KeepTopNPeaks::new(row.top_n_peaks));
    experiment
        .workflow
        .replace_operation(MinNumberOfPeaks::new(row.min_peaks));
    experiment
        .workflow
        .replace_operation(MinMZRangeFilter::new(row.min_mz_range));
    // metabolomics() has no FilterIntensityRelative step, so this is a
    // documented no-op -- matches the CLI's own replace_operation semantics
    // and the GNPS screen's identical result at both levels of this factor.
    experiment
        .workflow
        .replace_operation(FilterIntensityRelative::new(row.min_relative_intensity));

    experiment.ingest_files_parallel(PATH);
    experiment.prepare_clustering();
    let clusters = experiment.cluster();

    let purity = experiment.average_weighted_cluster_purity(&clusters, MZ_TOL, RT_TOL);
    let n10 = experiment.calculate_n10_value(&clusters);
    let n_clustered: usize = clusters.iter().map(|c| c.len()).sum();
    let completeness = n_clustered as f64 / experiment.total_spectra_ingested as f64;
    (purity, n10, n_clustered, completeness, experiment.total_spectra_ingested)
}

fn main() {
    let rows = all_rows();
    println!(
        "epsilon,fragment_tolerance,precursor_tolerance,transform,top_n_peaks,min_peaks,min_mz_range,min_relative_intensity,weighted_purity,n10,n_clustered,completeness,total_spectra,elapsed_s"
    );
    for (i, row) in rows.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (purity, n10, n_clustered, completeness, total) = run_row(row);
        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "{},{},{},{},{},{},{},{},{:.6},{},{},{:.6},{},{:.2}",
            row.epsilon,
            row.fragment_tolerance,
            row.precursor_tolerance,
            if row.transform == Transform::Sqrt { "sqrt" } else { "none" },
            row.top_n_peaks,
            row.min_peaks,
            row.min_mz_range,
            row.min_relative_intensity,
            purity,
            n10,
            n_clustered,
            completeness,
            total,
            elapsed,
        );
        eprintln!("[{}/{}] done in {:.1}s", i + 1, rows.len(), elapsed);
    }
}
