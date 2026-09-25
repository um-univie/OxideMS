//! Samples the similarity distribution of *unrelated* spectra, which is what
//! a multiple-comparison correction needs: given N candidates in a partition,
//! the expected number of spurious links at threshold tau is
//! `(N^2 / 2) * p0(tau)`, so holding that at a target requires knowing p0.
//!
//! Unrelated pairs are drawn from different precursor-mass windows, which
//! cannot legitimately cluster together and so are guaranteed non-matching.
//! That is an approximation to the null we actually care about -- different
//! identities at the *same* mass -- and it will understate the tail if
//! near-isobaric peptides resemble each other more than random pairs do.
//! Reported alongside is a same-mass-window sample for comparison.
//!
//! Usage: null_similarity <dataset_dir> <preset> <n_pairs> [fragment_tol_da]

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::clustering::similarities::branching_cosine_sim;
use oxidems_lib::workflow::{BinaryIntensity, MassTolerance, RankTransformIntensity, SquareRootTransformIntensity};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::path::{Path, PathBuf};

fn mgf_dirs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut has_mgf = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            mgf_dirs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mgf") {
            has_mgf = true;
        }
    }
    if has_mgf {
        out.push(dir.to_path_buf());
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dataset_dir = args.next().expect("dataset dir");
    let preset = args.next().unwrap_or_else(|| "proteomics".into());
    let n_pairs: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2_000_000);
    let frag_tol: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0.05);

    let mut experiment = match preset.as_str() {
        "metabolomics" => SpectralExperiment::metabolomics(),
        "proteomics" => SpectralExperiment::proteomics(),
        other => panic!("unknown preset {other}"),
    };
    // The null must be measured under the same similarity the clusterer will
    // use, so the transform has to be applied here too.
    match std::env::var("OXIDEMS_TRANSFORM").as_deref() {
        Ok("rank") => experiment.workflow.add_operation(RankTransformIntensity::new()),
        Ok("binary") => experiment.workflow.add_operation(BinaryIntensity::new()),
        Ok("sqrt") => experiment.workflow.add_operation(SquareRootTransformIntensity::new()),
        _ => {}
    }
    let mut dirs = Vec::new();
    mgf_dirs(Path::new(&dataset_dir), &mut dirs);
    dirs.sort();
    for d in &dirs {
        experiment.ingest_files_parallel(d.to_str().unwrap());
    }
    let mut spectra = std::mem::take(&mut experiment.spectra);
    spectra.sort_unstable_by(|a, b| {
        a.get_neutral_mass()
            .partial_cmp(&b.get_neutral_mass())
            .unwrap()
    });
    let n = spectra.len();
    println!("{n} spectra");

    let tol = MassTolerance::Da(frag_tol);
    let soas: Vec<_> = spectra
        .iter()
        .map(|s| {
            let mut v = s.spectrum.to_soa(tol);
            v.l2_normalize();
            v
        })
        .collect();
    let masses: Vec<f64> = spectra.iter().map(|s| s.get_neutral_mass()).collect();

    let mut rng = StdRng::seed_from_u64(7);
    let sample = |rng: &mut StdRng, same_window: bool| -> Option<f32> {
        let i = rng.random_range(0..n);
        let j = if same_window {
            // another spectrum inside i's precursor window
            let w = masses[i] * 20e-6;
            let lo = masses.partition_point(|&m| m < masses[i] - w);
            let hi = masses.partition_point(|&m| m <= masses[i] + w);
            if hi - lo < 2 {
                return None;
            }
            let mut j = rng.random_range(lo..hi);
            if j == i {
                j = if j + 1 < hi { j + 1 } else { lo };
            }
            j
        } else {
            let j = rng.random_range(0..n);
            // far enough apart in mass that they could never cluster
            if (masses[j] - masses[i]).abs() <= masses[i] * 20e-6 * 100.0 {
                return None;
            }
            j
        };
        Some(branching_cosine_sim(
            &soas[i].mz_bins,
            &soas[i].intensities,
            &soas[j].mz_bins,
            &soas[j].intensities,
        ))
    };

    for (label, same_window) in [("cross-window", false), ("same-window", true)] {
        let mut sims = Vec::with_capacity(n_pairs);
        while sims.len() < n_pairs {
            if let Some(s) = sample(&mut rng, same_window) {
                sims.push(s);
            }
        }
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        println!("\n{label} null, {} pairs", sims.len());
        println!("{:>8} {:>12}", "tau", "p0(>=tau)");
        for tau in [0.5f32, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99] {
            let k = sims.partition_point(|&s| s >= tau);
            println!("{tau:8.3} {:12.3e}", k as f64 / sims.len() as f64);
        }
    }
}
