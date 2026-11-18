//! Where fragmentation tree building goes wrong, measured against known structures.
//!
//! Retrieval accuracy depends on the tree before it depends on anything else: the
//! feature vector is a function of the tree alone, so a peak the tree cannot explain is
//! a peak the fingerprint predictor never sees. This tool takes spectra whose true
//! structure is annotated, builds the tree under the *correct* precursor formula — so
//! that formula identification is not confounded with tree quality — and reports what
//! the tree did with the spectrum.
//!
//! Usage: `diagnose_trees <spectra.mgf> [--max-peaks N] [--ppm P]`

use oxidems_lib::sirius::fingerprint::morgan_identifiers;
use oxidems_lib::sirius::{Adduct, Formula, FragmentationTreeBuilder, TreeSolverConfig};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;

/// Electron rest mass in daltons; the difference between a radical cation and its
/// neutral formula.
const ELECTRON_MASS: f64 = 0.000_548_579_909;
use oxidane::molecular_formula::MolecularFormula;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::molecular_graph::MolecularGraph;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

struct Record {
    name: String,
    formula: Formula,
    precursor_mz: f64,
    peaks: Vec<Peak>,
    positive: bool,
}

fn formula_from_smiles(smiles: &str) -> Option<Formula> {
    let mut mol = std::panic::catch_unwind(|| MolecularGraph::from_smiles_single(smiles).ok())
        .ok()
        .flatten()?;
    mol.add_hydrogens();
    Formula::from_str(&MolecularFormula::from_molecule(&mol).to_string()).ok()
}

fn parse(path: &str) -> Vec<Record> {
    let f = File::open(path).expect("spectra file");
    let mut out = Vec::new();
    let (mut name, mut smiles, mut mz, mut positive) = (String::new(), String::new(), 0.0f64, true);
    let mut peaks: Vec<Peak> = Vec::new();

    for line in BufReader::with_capacity(1 << 22, f).lines().map_while(Result::ok) {
        let line = line.trim();
        if line == "BEGIN IONS" {
            name.clear();
            smiles.clear();
            peaks.clear();
            mz = 0.0;
            positive = true;
            continue;
        }
        if line == "END IONS" {
            if !smiles.is_empty() && mz > 0.0 && peaks.len() >= 3 {
                if let Some(formula) = formula_from_smiles(&smiles) {
                    out.push(Record {
                        name: name.clone(),
                        formula,
                        precursor_mz: mz,
                        peaks: peaks.clone(),
                        positive,
                    });
                }
            }
            continue;
        }
        if let Some(v) = line.strip_prefix("SMILES=") {
            smiles = v.trim().to_string();
        } else if let Some(v) = line
            .strip_prefix("COMPOUND_NAME=")
            .or_else(|| line.strip_prefix("NAME="))
        {
            name = v.trim().to_string();
        } else if let Some(v) = line
            .strip_prefix("PRECURSOR_MZ=")
            .or_else(|| line.strip_prefix("PEPMASS="))
        {
            mz = v.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
        } else if let Some(v) = line.strip_prefix("IONMODE=") {
            positive = !v.trim().starts_with('n') && !v.trim().starts_with('N');
        } else if line.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() >= 2 {
                if let (Ok(m), Ok(i)) = (p[0].parse::<f64>(), p[1].parse::<f32>()) {
                    if m > 0.0 && i > 0.0 {
                        peaks.push(Peak { mz: m, intensity: i });
                    }
                }
            }
        }
    }
    out
}

/// One spectrum's tree, summarized.
struct TreeStat {
    peaks_in: usize,
    peaks_kept: usize,
    nodes: usize,
    explained_peaks: usize,
    explained_intensity: f64,
    /// Intensity thrown away by keeping only the most intense `max_peaks`.
    intensity_discarded_by_truncation: f64,
    /// Fragment assignments whose ring-plus-double-bond equivalent is negative, which
    /// no real molecule can have.
    negative_rdbe: usize,
    /// Assigned fragments that are not subformulas of the precursor.
    non_subformula: usize,
    /// Assignments that fit a radical ion better than a protonated one.
    radicals: usize,
    /// Share of the kept intensity carried by peaks the tree left out.
    unexplained_intensity: f64,
    /// Of that, the share sitting on the precursor m/z itself.
    unexplained_precursor: f64,
    /// Intensity rank of the largest unexplained peak, 1 being the base peak.
    largest_unexplained_rank: Option<usize>,
    mass_errors_ppm: Vec<f64>,
}

fn rdbe(f: &Formula) -> f64 {
    use oxidems_lib::sirius::element::Element;
    let c = f.get(Element::C) as f64;
    let h = f.get(Element::H) as f64;
    let n = f.get(Element::N) as f64;
    let halogens = (f.get(Element::F) + f.get(Element::Cl) + f.get(Element::Br) + f.get(Element::I))
        as f64;
    c - (h + halogens) / 2.0 + n / 2.0 + 1.0
}

fn analyse(rec: &Record, max_peaks: usize, ppm: f64) -> TreeStat {
    let total_intensity: f64 = rec.peaks.iter().map(|p| p.intensity as f64).sum();
    let mut sorted = rec.peaks.clone();
    sorted.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
    let kept = max_peaks.min(sorted.len());
    let kept_intensity: f64 = sorted[..kept].iter().map(|p| p.intensity as f64).sum();
    sorted.truncate(kept);

    let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
        tolerance: MassTolerance::Ppm(ppm),
        adduct: if rec.positive {
            Adduct::protonated()
        } else {
            Adduct::deprotonated()
        },
        ..Default::default()
    });
    let tree = builder.build_tree(&rec.formula, rec.precursor_mz, &sorted);

    let mut negative_rdbe = 0;
    let mut non_subformula = 0;
    let mut radicals = 0usize;
    let mut mass_errors_ppm = Vec::new();
    for node in tree.nodes.iter().skip(1) {
        if rdbe(&node.formula) < 0.0 {
            negative_rdbe += 1;
        }
        if !node.formula.is_subformula_of(&rec.formula) {
            non_subformula += 1;
        }
        // Fragments come in two flavours: even-electron ions, which carry the adduct's
        // proton, and radicals, which do not. EAD makes plenty of the latter. Scoring
        // every assignment as though it were protonated turns each radical into a
        // spurious ~1 Da error, so take whichever interpretation the assignment
        // actually fits.
        let adduct = if rec.positive {
            Adduct::protonated()
        } else {
            Adduct::deprotonated()
        };
        let neutral = node.formula.exact_mass();
        let even = adduct.neutral_mass_to_ion_mz(neutral);
        let radical = if rec.positive {
            neutral - ELECTRON_MASS
        } else {
            neutral + ELECTRON_MASS
        };
        let mut best = f64::INFINITY;
        let mut radical_fit = false;
        for (theoretical, is_radical) in [(even, false), (radical, true)] {
            if theoretical <= 0.0 {
                continue;
            }
            let e = (node.mz - theoretical) / theoretical * 1e6;
            if e.abs() < best.abs() {
                best = e;
                radical_fit = is_radical;
            }
        }
        if best.is_finite() {
            mass_errors_ppm.push(best);
            if radical_fit {
                radicals += 1;
            }
        }
    }

    // Which of the kept peaks did the tree actually use?
    let used: std::collections::HashSet<usize> =
        tree.nodes.iter().filter_map(|n| n.peak_index).collect();
    let kept_total: f64 = sorted.iter().map(|p| p.intensity as f64).sum::<f64>().max(1e-9);
    let mut unexplained = 0.0f64;
    let mut precursor_share = 0.0f64;
    let mut largest_unexplained_rank = None;
    for (rank, peak) in sorted.iter().enumerate() {
        if used.contains(&rank) {
            continue;
        }
        let share = peak.intensity as f64 / kept_total;
        unexplained += share;
        // Within 20 ppm of the precursor: the surviving precursor ion is the tree root,
        // which carries no peak index of its own, so it can look unexplained.
        if ((peak.mz - rec.precursor_mz).abs() / rec.precursor_mz) < 2e-5 {
            precursor_share += share;
        }
        if largest_unexplained_rank.is_none() {
            largest_unexplained_rank = Some(rank + 1);
        }
    }

    TreeStat {
        unexplained_intensity: unexplained,
        unexplained_precursor: precursor_share,
        largest_unexplained_rank,
        peaks_in: rec.peaks.len(),
        peaks_kept: kept,
        nodes: tree.nodes.len(),
        explained_peaks: tree.explained_peaks,
        explained_intensity: tree.explained_intensity as f64,
        intensity_discarded_by_truncation: 1.0 - kept_intensity / total_intensity.max(1e-9),
        negative_rdbe,
        non_subformula,
        radicals,
        mass_errors_ppm,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

mod support;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| support::fail("usage: diagnose_trees <spectra.mgf> [--max-peaks N] [--ppm P]"));
    let mut max_peaks_list = vec![25usize];
    let mut ppm = 20.0f64;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--max-peaks" => {
                max_peaks_list = args[i + 1].split(',').filter_map(|s| s.parse().ok()).collect();
                i += 2;
            }
            "--ppm" => {
                ppm = args[i + 1].parse().unwrap_or(ppm);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let records = parse(path);
    println!("Parsed {} annotated spectra from {path}", records.len());
    let _ = morgan_identifiers("C");

    for &max_peaks in &max_peaks_list {
        let stats: Vec<TreeStat> = records.par_iter().map(|r| analyse(r, max_peaks, ppm)).collect();
        let n = stats.len() as f64;
        let mean = |f: fn(&TreeStat) -> f64| stats.iter().map(f).sum::<f64>() / n;

        println!("\n=== max_peaks {max_peaks}, tolerance {ppm} ppm ===");
        println!(
            "  peaks in spectrum: mean {:.0}, kept {:.0} ({:.1}% of intensity discarded by truncation)",
            mean(|s| s.peaks_in as f64),
            mean(|s| s.peaks_kept as f64),
            mean(|s| s.intensity_discarded_by_truncation) * 100.0
        );
        println!(
            "  tree nodes: mean {:.1} | peaks explained: mean {:.1} ({:.1}% of kept peaks)",
            mean(|s| s.nodes as f64),
            mean(|s| s.explained_peaks as f64),
            mean(|s| s.explained_peaks as f64) / mean(|s| s.peaks_kept as f64) * 100.0
        );
        println!(
            "  intensity explained by the tree: mean {:.1}%, median {:.1}%",
            mean(|s| s.explained_intensity) * 100.0,
            median(stats.iter().map(|s| s.explained_intensity).collect()) * 100.0
        );
        let unexplained_share = stats
            .iter()
            .filter(|s| s.explained_intensity < 0.5)
            .count();
        println!(
            "  spectra where the tree explains under half the intensity: {} ({:.1}%)",
            unexplained_share,
            unexplained_share as f64 / n * 100.0
        );
        println!(
            "  implausible assignments: {} with negative RDBE, {} not a subformula of the precursor",
            stats.iter().map(|s| s.negative_rdbe).sum::<usize>(),
            stats.iter().map(|s| s.non_subformula).sum::<usize>()
        );
        let errs: Vec<f64> = stats.iter().flat_map(|s| s.mass_errors_ppm.clone()).collect();
        if !errs.is_empty() {
            let m = errs.iter().sum::<f64>() / errs.len() as f64;
            let sd = (errs.iter().map(|e| (e - m).powi(2)).sum::<f64>() / errs.len() as f64).sqrt();
            let beyond = errs.iter().filter(|e| e.abs() > ppm).count();
            println!(
                "  assignment mass error: mean {m:+.2} ppm, sd {sd:.2}, |error| > tolerance in {} of {} ({:.1}%)",
                beyond,
                errs.len(),
                beyond as f64 / errs.len() as f64 * 100.0
            );
            println!(
                "  assignments fitting a radical rather than a protonated ion: {} ({:.1}%)",
                stats.iter().map(|s| s.radicals).sum::<usize>(),
                stats.iter().map(|s| s.radicals).sum::<usize>() as f64 / errs.len() as f64 * 100.0
            );
        }
        let ranks: Vec<f64> = stats
            .iter()
            .filter_map(|s| s.largest_unexplained_rank.map(|r| r as f64))
            .collect();
        println!(
            "  unexplained share of kept intensity: {:.1}%, of which {:.1} points sit on the precursor m/z",
            mean(|s| s.unexplained_intensity) * 100.0,
            mean(|s| s.unexplained_precursor) * 100.0
        );
        if !ranks.is_empty() {
            println!(
                "  largest unexplained peak: intensity rank {:.0} (median); it is the base peak in {} of {} spectra",
                median(ranks.clone()),
                stats.iter().filter(|s| s.largest_unexplained_rank == Some(1)).count(),
                stats.len()
            );
        }
    }
}
