//! AUC-based re-score of the metabolomics 8-factor DOE screen against the
//! real Astral dataset (tests/Astral).
//!
//! Unlike astral_doe_screen.rs (epsilon as an 8th 2-level factor, 256 runs
//! total, 2 epsilon points per combo), this treats epsilon as a nuisance
//! parameter to integrate out: for each of the 128 combinations of the
//! other 7 factors, epsilon is swept across a full 12-point grid and scored
//! by the trapezoidal area under the purity-vs-completeness curve (and
//! purity-vs-N10/total). A combo's AUC is a single number describing how
//! good its whole purity/completeness tradeoff curve is, independent of
//! which single epsilon a user picks -- this is what should rank the other
//! 7 factors; the shipped preset's single epsilon default is then chosen
//! separately (e.g. at the curve's knee) once the best combo is fixed.
//!
//! Runs in-process. Ingestion (which bakes the workflow's peak-filtering
//! factors into `spectra` inside `process_single_file`) only depends on
//! (transform, top_n_peaks, min_peaks, min_mz_range, min_relative_intensity)
//! -- 32 combos -- so those 32 ingestions happen once each; the remaining
//! (fragment_tolerance, precursor_tolerance, epsilon) only affect
//! `prepare_clustering`/`cluster`, so for each of the 32 ingested spectra
//! sets, all 4 x 12 = 48 (fragment_tolerance, precursor_tolerance, epsilon)
//! combinations are re-clustered from a cloned copy of the same ingested
//! spectra, without re-parsing the raw files. This keeps ingestion (the
//! dominant per-run cost measured in astral_doe_screen.rs) below 10% of
//! total runtime instead of ~100%.

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::spectrum::MSNSpectrum;
use oxidems_lib::workflow::{
    FilterIntensityRelative, KeepTopNPeaks, LogTransformIntensity, MassTolerance,
    MinMZRangeFilter, MinNumberOfPeaks, SquareRootTransformIntensity,
};

const MZ_TOL: f64 = 0.01;
const RT_TOL: f32 = 30.0;
const PATH: &str = "tests/Astral";
const EPS_GRID: [f32; 12] = [
    0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transform {
    None,
    Sqrt,
}

#[derive(Clone, Copy, Debug)]
struct OuterCombo {
    transform: Transform,
    top_n_peaks: usize,
    min_peaks: usize,
    min_mz_range: f64,
    min_relative_intensity: f32,
}

#[derive(Clone, Copy, Debug)]
struct InnerCombo {
    fragment_tolerance: f64,
    precursor_tolerance: f64,
}

fn all_outer_combos() -> Vec<OuterCombo> {
    let mut rows = Vec::with_capacity(32);
    for &transform in &[Transform::None, Transform::Sqrt] {
        for &top_n_peaks in &[50usize, 150] {
            for &min_peaks in &[3usize, 5] {
                for &min_mz_range in &[50f64, 100.0] {
                    for &min_relative_intensity in &[0.0f32, 0.01] {
                        rows.push(OuterCombo {
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
    rows
}

fn all_inner_combos() -> Vec<InnerCombo> {
    let mut rows = Vec::with_capacity(4);
    for &fragment_tolerance in &[0.02f64, 0.05] {
        for &precursor_tolerance in &[10f64, 20.0] {
            rows.push(InnerCombo {
                fragment_tolerance,
                precursor_tolerance,
            });
        }
    }
    rows
}

fn ingest_outer(combo: &OuterCombo) -> (SpectralExperiment, Vec<MSNSpectrum>) {
    let mut experiment = SpectralExperiment::metabolomics();
    experiment.workflow.remove_operation::<SquareRootTransformIntensity>();
    experiment.workflow.remove_operation::<LogTransformIntensity>();
    if combo.transform == Transform::Sqrt {
        experiment
            .workflow
            .add_operation(SquareRootTransformIntensity::new());
    }
    experiment
        .workflow
        .replace_operation(KeepTopNPeaks::new(combo.top_n_peaks));
    experiment
        .workflow
        .replace_operation(MinNumberOfPeaks::new(combo.min_peaks));
    experiment
        .workflow
        .replace_operation(MinMZRangeFilter::new(combo.min_mz_range));
    experiment
        .workflow
        .replace_operation(FilterIntensityRelative::new(combo.min_relative_intensity));

    experiment.ingest_files_parallel(PATH);
    let snapshot = experiment.spectra.clone();
    (experiment, snapshot)
}

fn run_point(
    experiment: &mut SpectralExperiment,
    snapshot: &[MSNSpectrum],
    inner: &InnerCombo,
    epsilon: f32,
) -> (f64, usize, usize, f64) {
    experiment.spectra = snapshot.to_vec();
    experiment.engine.hasher.tolerance = MassTolerance::Da(inner.fragment_tolerance);
    experiment.engine.mz_tolerance = MassTolerance::Ppm(inner.precursor_tolerance);
    experiment.engine.epsilon = epsilon;

    experiment.prepare_clustering();
    let clusters = experiment.cluster();

    let purity = experiment.average_weighted_cluster_purity(&clusters, MZ_TOL, RT_TOL);
    let n10 = experiment.calculate_n10_value(&clusters);
    let n_clustered: usize = clusters.iter().map(|c| c.len()).sum();
    let completeness = n_clustered as f64 / experiment.total_spectra_ingested as f64;
    (purity, n10, n_clustered, completeness)
}

fn main() {
    let outer_combos = all_outer_combos();
    let inner_combos = all_inner_combos();
    let total_runs = outer_combos.len() * inner_combos.len() * EPS_GRID.len();
    let mut done = 0usize;

    println!(
        "transform,top_n_peaks,min_peaks,min_mz_range,min_relative_intensity,fragment_tolerance,precursor_tolerance,epsilon,weighted_purity,n10,n_clustered,completeness,total_spectra,elapsed_s"
    );

    for outer in &outer_combos {
        let t_ingest = std::time::Instant::now();
        let (mut experiment, snapshot) = ingest_outer(outer);
        let total = experiment.total_spectra_ingested;
        eprintln!(
            "[ingest] {:?} -> {} spectra in {:.1}s",
            outer,
            total,
            t_ingest.elapsed().as_secs_f64()
        );

        for inner in &inner_combos {
            for &epsilon in &EPS_GRID {
                let t0 = std::time::Instant::now();
                let (purity, n10, n_clustered, completeness) =
                    run_point(&mut experiment, &snapshot, inner, epsilon);
                let elapsed = t0.elapsed().as_secs_f64();
                done += 1;
                println!(
                    "{},{},{},{},{},{},{},{},{:.6},{},{},{:.6},{},{:.2}",
                    if outer.transform == Transform::Sqrt { "sqrt" } else { "none" },
                    outer.top_n_peaks,
                    outer.min_peaks,
                    outer.min_mz_range,
                    outer.min_relative_intensity,
                    inner.fragment_tolerance,
                    inner.precursor_tolerance,
                    epsilon,
                    purity,
                    n10,
                    n_clustered,
                    completeness,
                    total,
                    elapsed,
                );
                eprintln!("[{done}/{total_runs}] done in {elapsed:.1}s");
            }
        }
    }
}
