//! AUC-based re-score of the proteomics 8-factor DOE screen, mirroring
//! astral_doe_auc_screen.rs's structure (see that file's doc comment for the
//! rationale: epsilon swept as a nuisance parameter and integrated out via
//! trapezoidal AUC of the purity-vs-completeness / purity-vs-N10 curve,
//! instead of being one of the 8 screened factors at 2 discrete levels).
//!
//! Ground truth is MS-GF+ PSMs at 1% FDR (USI,PeptideSequence,MS-GF:QValue),
//! joined by (source file, scan number) rather than by re-deriving the USI
//! string -- SpectrumMetadata already carries file_index/scan_number, so the
//! ground truth's own USI strings are parsed apart once into that same key
//! space instead of reconstructing redshift-ms's exact USI format (avoids a
//! whole class of string-formatting mismatch bugs).
//!
//! Purity/completeness formulas mirror
//! ms-cluster-benchmark/doe/evaluate_clustering.py::evaluate exactly:
//!   - weighted_purity: per-cluster majority-identity-count / annotated
//!     cluster size, weighted by each cluster's annotated spectra count.
//!   - weighted_completeness: per-identity (peptide sequence, L/I merged,
//!     >=2 supporting spectra in ground truth) max-single-cluster-count /
//!     that identity's TOTAL annotated count in the ground truth (not just
//!     the clustered subset -- spectra DBSCAN drops as noise count against
//!     the denominator), weighted by each identity's total annotated count.
//!
//! Datasets live under oxidems_lib/tests/<dataset>/ (both the raw MGFs and
//! overall_spectrum_annotations.csv in the same directory) -- this binary
//! takes the data root (e.g. "tests" or "../tests") as argv[1].

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::clustering::cluster::flat_clusters::FlatClusters;
use oxidems_lib::spectrum::MSNSpectrum;
use oxidems_lib::workflow::{
    FilterIntensityRelative, KeepTopNPeaks, LogTransformIntensity, MassTolerance,
    MinMZRangeFilter, MinNumberOfPeaks, SquareRootTransformIntensity,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

const EPS_GRID: [f32; 9] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3];
// Deadline-scoped to the two small Arabidopsis thaliana datasets (163,927
// and 3,106,616 spectra) instead of the full trio -- PXD000561/PXD000865/
// PXD004452 are all ~13M spectra each (not "small"), a full 128-combo x
// 9-eps rescore on any one of them alone is ~16h. PXD000561 etc. deferred
// to future work; see the run script for the ground-truth path convention.
const DATASETS: [&str; 2] = ["PXD023047", "PXD021518"];
const MIN_IDENTITY_SUPPORT: usize = 2;

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
                for &min_mz_range in &[100f64, 250.0] {
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

/// USI format: mzspec:<pxd>:<filename>:scan:<scan_number>
fn load_ground_truth(path: &str) -> HashMap<(String, usize), String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue; // header
        }
        let mut parts = line.splitn(3, ',');
        let usi = match parts.next() {
            Some(u) => u,
            None => continue,
        };
        let peptide = match parts.next() {
            Some(p) => p,
            None => continue,
        };
        let usi_fields: Vec<&str> = usi.split(':').collect();
        if usi_fields.len() < 5 {
            continue;
        }
        let filename = usi_fields[2].to_string();
        let scan_number: usize = match usi_fields[4].parse() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let peptide_merged = peptide.replace('I', "L");
        map.insert((filename, scan_number), peptide_merged);
    }
    map
}

fn ingest_outer(
    data_path: &str,
    combo: &OuterCombo,
) -> (SpectralExperiment, Vec<MSNSpectrum>) {
    let mut experiment = SpectralExperiment::proteomics();
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

    experiment.ingest_files_parallel(data_path);
    let snapshot = experiment.spectra.clone();
    (experiment, snapshot)
}

/// Mirrors evaluate_clustering.py::evaluate exactly (weighted_purity,
/// weighted_completeness), reading ground truth via each spectrum's
/// (file_name, scan_number) instead of a USI string join.
fn score_ms_db(
    experiment: &SpectralExperiment,
    clusters: &FlatClusters,
    ground_truth: &HashMap<(String, usize), String>,
) -> (f64, f64) {
    // identity of each spectrum id, if annotated
    let identity_of = |id: usize| -> Option<&String> {
        let meta = experiment.metadata.get(id)?;
        let fname = experiment.file_names.get(meta.file_index)?;
        ground_truth.get(&(fname.clone(), meta.scan_number?))
    };

    // --- purity ---
    let mut purity_num = 0.0f64;
    let mut purity_den = 0usize;
    for cluster in clusters.iter() {
        let mut counts: HashMap<&String, usize> = HashMap::new();
        let mut annotated = 0usize;
        for &id in cluster {
            if let Some(identity) = identity_of(id) {
                *counts.entry(identity).or_insert(0) += 1;
                annotated += 1;
            }
        }
        if annotated == 0 {
            continue;
        }
        let max_count = counts.values().copied().max().unwrap_or(0);
        purity_num += max_count as f64;
        purity_den += annotated;
    }
    let weighted_purity = if purity_den > 0 {
        purity_num / purity_den as f64
    } else {
        f64::NAN
    };

    // --- completeness ---
    // total annotated count per identity (ground truth, dataset-wide -- not
    // just the clustered subset) and max single-cluster count per identity.
    let mut total_per_identity: HashMap<&String, usize> = HashMap::new();
    for identity in ground_truth.values() {
        *total_per_identity.entry(identity).or_insert(0) += 1;
    }
    let mut max_cluster_per_identity: HashMap<&String, usize> = HashMap::new();
    for cluster in clusters.iter() {
        let mut counts: HashMap<&String, usize> = HashMap::new();
        for &id in cluster {
            if let Some(identity) = identity_of(id) {
                *counts.entry(identity).or_insert(0) += 1;
            }
        }
        for (identity, count) in counts {
            let entry = max_cluster_per_identity.entry(identity).or_insert(0);
            if count > *entry {
                *entry = count;
            }
        }
    }
    let mut completeness_num = 0.0f64;
    let mut completeness_den = 0usize;
    for (identity, &total) in &total_per_identity {
        if total < MIN_IDENTITY_SUPPORT {
            continue;
        }
        let max_count = max_cluster_per_identity.get(identity).copied().unwrap_or(0);
        completeness_num += max_count as f64;
        completeness_den += total;
    }
    let weighted_completeness = if completeness_den > 0 {
        completeness_num / completeness_den as f64
    } else {
        f64::NAN
    };

    (weighted_purity, weighted_completeness)
}

fn run_point(
    experiment: &mut SpectralExperiment,
    snapshot: &[MSNSpectrum],
    ground_truth: &HashMap<(String, usize), String>,
    inner: &InnerCombo,
    epsilon: f32,
) -> (f64, usize, f64) {
    experiment.spectra = snapshot.to_vec();
    experiment.engine.hasher.tolerance = MassTolerance::Da(inner.fragment_tolerance);
    experiment.engine.mz_tolerance = MassTolerance::Ppm(inner.precursor_tolerance);
    experiment.engine.epsilon = epsilon;

    experiment.prepare_clustering();
    let clusters = experiment.cluster();

    let n10 = experiment.calculate_n10_value(&clusters);
    let (purity, completeness) = score_ms_db(experiment, &clusters, ground_truth);
    (purity, n10, completeness)
}

fn main() {
    let data_root = std::env::args()
        .nth(1)
        .expect("usage: proteomics_doe_auc_screen <data_root>");

    let outer_combos = all_outer_combos();
    let inner_combos = all_inner_combos();
    let total_runs = DATASETS.len() * outer_combos.len() * inner_combos.len() * EPS_GRID.len();
    let mut done = 0usize;

    println!(
        "dataset,transform,top_n_peaks,min_peaks,min_mz_range,min_relative_intensity,fragment_tolerance,precursor_tolerance,epsilon,weighted_purity,n10,weighted_completeness,total_spectra,elapsed_s"
    );

    for dataset in &DATASETS {
        let gt_path = format!("{data_root}/{dataset}/overall_spectrum_annotations.csv");
        let t_gt = std::time::Instant::now();
        let ground_truth = load_ground_truth(&gt_path);
        eprintln!(
            "[ground truth] {dataset}: {} annotations in {:.1}s",
            ground_truth.len(),
            t_gt.elapsed().as_secs_f64()
        );
        let data_path = format!("{data_root}/{dataset}");

        for outer in &outer_combos {
            let t_ingest = std::time::Instant::now();
            let (mut experiment, snapshot) = ingest_outer(&data_path, outer);
            let total = experiment.total_spectra_ingested;
            eprintln!(
                "[ingest] {dataset} {:?} -> {} spectra in {:.1}s",
                outer,
                total,
                t_ingest.elapsed().as_secs_f64()
            );

            for inner in &inner_combos {
                for &epsilon in &EPS_GRID {
                    let t0 = std::time::Instant::now();
                    let (purity, n10, completeness) =
                        run_point(&mut experiment, &snapshot, &ground_truth, inner, epsilon);
                    let elapsed = t0.elapsed().as_secs_f64();
                    done += 1;
                    println!(
                        "{},{},{},{},{},{},{},{},{},{:.6},{},{:.6},{},{:.2}",
                        dataset,
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
                        completeness,
                        total,
                        elapsed,
                    );
                    eprintln!("[{done}/{total_runs}] done in {elapsed:.1}s");
                }
            }
        }
    }
}
