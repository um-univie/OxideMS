//! Score the tree scorer against the two ground-truth signals GNPS gives us:
//!
//!   1. formula rank  — the true precursor formula should be the top-scoring
//!      tree among de-novo candidates (hard ground truth: `TRUEFORMULA=`)
//!   2. explanation    — the fragments the true-formula tree assigns should be
//!      real substructures of the molecule (proxy ground truth: `SMILES=`, checked
//!      downstream with RDKit)
//!
//! Scoring parameters are CLI flags so a sweep can look for an operating point
//! that improves both. Writes `<out>/eval.json`; metrics live in
//! `scratch/eval_metrics.py`.
//!
//! Usage:
//!   eval_tree_scoring <mgf> [--limit N] [--ppm P] [--sigma S] [--loss-sigma L] [--out DIR]
//!
//! The candidate alphabet is CHNOPS+FClBr for every spectrum. `--oracle-alphabet`
//! restores the old true-formula-derived tiering for reproducing old numbers only.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{
    Adduct, ChemicalRules, Formula, FormulaGenerator, FragmentationTreeBuilder, SolverKind,
    TreeSolverConfig,
};
use oxidems_lib::sirius::solver::ScoringModel;
use oxidems_lib::sirius::rules::LossGate;
use oxidems_lib::sirius::workflow::{DeisotopeMode, PeakPreprocessing, SiriusClusteringWorkflow, SiriusWorkflowConfig};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::fs;
use std::str::FromStr;

/// SIRIUS's `NoiseThresholdSettings.basePeak`. `Max` is the overall most intense
/// peak, which is what this harness always used; `NotPrecursor` is SIRIUS's shipped
/// default and normalises against the most intense peak that is *not* the surviving
/// precursor. When the precursor is the base peak — the common case — `Max` divides
/// by a larger number and so keeps fewer peaks at the same 0.5 % threshold. Round 29
/// flagged this as the likely reason sparse CASMI spectra lost peaks SIRIUS kept.
#[derive(Clone, Copy, PartialEq, Debug)]
enum BasePeak {
    Max,
    NotPrecursor,
}

fn clean_peaks_with(
    raw: &[Peak],
    precursor_mz: f64,
    max_peaks: usize,
    base_peak: BasePeak,
) -> Vec<Peak> {
    let max_int = match base_peak {
        BasePeak::Max => raw.iter().map(|p| p.intensity).fold(0.0f32, f32::max),
        // SIRIUS treats anything within 0.1 Da of the precursor as the precursor peak.
        BasePeak::NotPrecursor => {
            let m = raw
                .iter()
                .filter(|p| (p.mz - precursor_mz).abs() > 0.1)
                .map(|p| p.intensity)
                .fold(0.0f32, f32::max);
            // A spectrum that is nothing but its precursor still needs a scale.
            if m > 0.0 { m } else { raw.iter().map(|p| p.intensity).fold(0.0f32, f32::max) }
        }
    }
    .max(1e-9);
    let mut peaks: Vec<Peak> = raw
        .iter()
        .filter(|p| p.intensity / max_int >= 0.005)
        .filter(|p| p.mz <= precursor_mz + 5.0)
        .copied()
        .collect();
    peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    peaks.truncate(max_peaks);
    peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
    peaks.dedup_by(|a, b| (a.mz - b.mz).abs() < 1e-4);
    peaks
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' | '\r' | '\t' => out.push(' '),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

struct Rec {
    /// Position in the parsed mgf. The stable key for pairing two runs: GNPS
    /// reuses `NAME=` across spectra, so a name-keyed join silently collapses
    /// them. It is also the compound id SIRIUS writes into its `.ms` files.
    idx: usize,
    name: String,
    smiles: String,
    true_formula: Formula,
    precursor_mz: f64,
    positive: bool,
    /// The adduct this spectrum was actually measured under. Re-deriving it as
    /// protonated in the scoring loop -- which this harness did -- silently made
    /// every non-protonated spectrum unsolvable: the neutral mass used for
    /// candidate enumeration was ~22 Da off for [M+Na]+, so the true formula was
    /// never generated (0/26, `true_in_set` false on every one).
    adduct: Adduct,
    peaks: Vec<Peak>,
    /// MS1 isotope-envelope peaks (`MS1PEAKS=mz:int ...` in the mgf), empty when
    /// the dataset has no MS1.
    ms1: Vec<Peak>,
}

fn parse_ms1(meta: Option<&str>) -> Vec<Peak> {
    let Some(s) = meta else { return Vec::new() };
    s.split_whitespace()
        .filter_map(|tok| {
            let (mz, int) = tok.split_once(':')?;
            Some(Peak { mz: mz.parse().ok()?, intensity: int.parse().ok()? })
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: eval_tree_scoring <mgf> [--limit N] [--ppm P] [--sigma S] [--loss-sigma L] [--out DIR]");

    let d = TreeSolverConfig::default();
    let mut limit = 400usize;
    let mut ppm = 10.0f64;
    let mut sigma = d.mass_sigma_ppm;
    let mut loss_sigma = d.loss_mass_sigma_log;
    let mut learned_losses = d.learned_loss_model;
    let mut nf_scale = d.noise_floor_scale;
    let mut use_isotopes = d.use_isotopes;
    let mut solver = d.solver;
    let mut scoring = d.scoring;
    let mut tree_size = d.tree_size_bonus;
    let mut loss_gate = d.loss_gate;
    let mut exact_top_n = d.exact_top_n;
    let mut profile_out: Option<String> = None;
    let mut max_peaks = 40usize;
    let mut out_dir = "scratch/tree_eval".to_string();
    let mut split: Option<String> = None;
    let mut oracle_alphabet = false;
    let mut no_fluorine = false;
    let mut recalibrate = d.recalibrate;
    let mut peak_noise_median = d.peak_noise_median;
    let mut base_peak = BasePeak::Max;
    let mut preprocess = "harness".to_string();
    let mut adduct_from_mgf = false;
    // Preserve historical controlled-tree comparisons, but label this oracle explicitly.
    let mut precursor_mass_source = "oracle".to_string();
    // Neutral-mass window a spectrum must fall in to be scored. The upper bound is
    // a harness choice, not a library limit -- `FormulaGenerator` is a round-robin
    // DP mod m(H) and has no mass ceiling. It stays at 750 so every number recorded
    // against this harness keeps its composition, but the dark-matter set (round 54)
    // has a median mass of 699 Da and 83 of its 292 compounds sit between 758 and
    // 769, so anything measured there must raise it. Candidate counts grow steeply
    // with mass; expect the runtime to follow.
    let mut min_mass = 80.0f64;
    let mut max_mass = 750.0f64;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--precursor-mass" => {
                precursor_mass_source = args[i + 1].clone();
                assert!(matches!(precursor_mass_source.as_str(), "measured" | "oracle"));
                i += 2;
            }
            "--split" => { split = Some(args[i + 1].clone()); i += 2; }
            "--limit" => { limit = args[i + 1].parse().unwrap(); i += 2; }
            "--ppm" => { ppm = args[i + 1].parse().unwrap(); i += 2; }
            "--sigma" => { sigma = args[i + 1].parse().unwrap(); i += 2; }
            "--loss-sigma" => { loss_sigma = args[i + 1].parse().unwrap(); i += 2; }
            "--hand-losses" => { learned_losses = false; i += 1; }
            "--noise" => { nf_scale = args[i + 1].parse().unwrap(); i += 2; }
            "--no-isotopes" => { use_isotopes = false; i += 1; }
            "--solver" => {
                solver = match args[i + 1].as_str() {
                    "critical" | "critical-path" => SolverKind::CriticalPath,
                    "dagdp" | "dag-dp" => SolverKind::DagDp,
                    "exact" => SolverKind::Exact,
                    "ilp" => SolverKind::Ilp,
                    other => panic!("--solver must be critical|dagdp|exact|ilp (got {other})"),
                };
                i += 2;
            }
            "--scoring" => {
                scoring = match args[i + 1].as_str() {
                    "native" => ScoringModel::Native,
                    "sirius" => ScoringModel::Sirius,
                    other => panic!("--scoring must be native|sirius (got {other})"),
                };
                i += 2;
            }
            "--profile" => { profile_out = Some(args[i + 1].clone()); i += 2; }
            "--exact-top-n" => { exact_top_n = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--loss-gate" => {
                loss_gate = match args[i + 1].as_str() {
                    "legacy" => LossGate::Legacy,
                    "repaired" => LossGate::Repaired,
                    "none" | "sirius" => LossGate::None,
                    other => panic!("--loss-gate must be legacy|repaired|none (got {other})"),
                };
                i += 2;
            }
            "--tree-size" => { tree_size = args[i + 1].parse().unwrap(); i += 2; }
            "--max-peaks" => { max_peaks = args[i + 1].parse().unwrap(); i += 2; }
            "--oracle-alphabet" => { oracle_alphabet = true; i += 1; }
            // CHNOPS+Cl+Br, matching SIRIUS's `-e SBrCl`.
            "--no-fluorine" => { no_fluorine = true; i += 1; }
            "--recalibrate" => { recalibrate = true; i += 1; }
            "--no-peak-noise" => { peak_noise_median = None; i += 1; }
            "--peak-noise" => { peak_noise_median = Some(args[i + 1].parse().unwrap()); i += 2; }
            // Accepted so old sweep scripts keep running; the honest full
            // alphabet is now what they get anyway.
            "--full-alphabet" => {
                eprintln!("--full-alphabet is the default now; flag ignored");
                i += 1;
            }
            // Which preprocessing pipeline builds the peak list. `harness` is what
            // every formula-ID number in this document was measured through;
            // `library-*` is `SiriusWorkflowConfig::preprocess_peaks_for_tree`, the
            // path `evaluate_casmi` and the production workflow actually use. They
            // have never been compared.
            "--preprocess" => {
                preprocess = args[i + 1].clone();
                i += 2;
            }
            // Take each spectrum's measured adduct from the mgf instead of assuming
            // protonation. Every benchmark in this project has assumed [M+H]+.
            "--adduct-from-mgf" => { adduct_from_mgf = true; i += 1; }
            "--min-mass" => { min_mass = args[i + 1].parse().unwrap(); i += 2; }
            "--max-mass" => { max_mass = args[i + 1].parse().unwrap(); i += 2; }
            "--base-peak" => {
                base_peak = match args[i + 1].as_str() {
                    "max" => BasePeak::Max,
                    "not-precursor" | "notprecursor" => BasePeak::NotPrecursor,
                    other => panic!("--base-peak must be max|not-precursor (got {other})"),
                };
                i += 2;
            }
            "--out" => { out_dir = args[i + 1].clone(); i += 2; }
            other => panic!("unknown arg {other}"),
        }
    }

    // `--profile out.svg`: sample this process with pprof (SIGPROF, no perf
    // privileges needed) and write a flamegraph on the way out.
    #[cfg(unix)]
    let guard = profile_out.as_ref().map(|_| {
        pprof::ProfilerGuardBuilder::default()
            .frequency(499)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .expect("start profiler")
    });
    #[cfg(not(unix))]
    if profile_out.is_some() {
        eprintln!("warning: --profile is only supported on Unix targets; ignoring");
    }

    let mut parser = MgfParser::default();
    let records = parser.parse_file(path).expect("parse mgf");
    eprintln!("parsed {} raw spectra", records.len());

    let tolerance = MassTolerance::Ppm(ppm);
    let mut kept: Vec<Rec> = Vec::new();
    for (idx, rec) in records.iter().enumerate() {
        if kept.len() >= limit {
            break;
        }
        if let Some(want) = &split {
            if rec.get_metadata("BENCHMARKSPLIT").map(str::trim) != Some(want.as_str()) {
                continue;
            }
        }
        let Some(tf) = rec
            .get_metadata("TRUEFORMULA")
            .and_then(|s| Formula::from_str(s.trim()).ok())
        else {
            continue;
        };
        let smiles = rec.get_metadata("SMILES").unwrap_or("").trim().to_string();
        if smiles.is_empty() {
            continue;
        }
        let positive = !rec.get_metadata("IONMODE").unwrap_or("Positive").eq_ignore_ascii_case("negative");
        let adduct = if adduct_from_mgf {
            match rec.get_metadata("ADDUCT").map(str::trim).and_then(|a| Adduct::from_str(a).ok()) {
                Some(a) => a,
                // An unparseable label is not silently treated as protonation --
                // that is the very assumption under test. Skip instead.
                None => continue,
            }
        } else if positive {
            Adduct::protonated()
        } else {
            Adduct::deprotonated()
        };
        let Some(pep) = rec
            .get_metadata("PEPMASS")
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
        else {
            continue;
        };
        let exact = tf.exact_mass();
        if !(min_mass..=max_mass).contains(&exact) {
            continue;
        }
        let theoretical_precursor_mz = adduct.neutral_mass_to_ion_mz(exact);
        if (theoretical_precursor_mz - pep).abs() > 0.03 {
            continue;
        }
        let precursor_mz = if precursor_mass_source == "measured" { pep } else { theoretical_precursor_mz };
        let peaks = if preprocess == "harness" {
            clean_peaks_with(rec.peaks(), precursor_mz, max_peaks, base_peak)
        } else {
            let (pp, mode) = match preprocess.as_str() {
                // The new default: the library now runs the same pipeline the
                // harness does, so these two arms must agree.
                "library-simple" => (PeakPreprocessing::Simple, DeisotopeMode::Off),
                "library" => (PeakPreprocessing::Legacy, DeisotopeMode::IsotopeAware),
                "library-legacy" => (PeakPreprocessing::Legacy, DeisotopeMode::Legacy),
                "library-nodeiso" => (PeakPreprocessing::Legacy, DeisotopeMode::Off),
                other => panic!("--preprocess must be harness|library-simple|library|library-legacy|library-nodeiso (got {other})"),
            };
            let mut cfg = SiriusWorkflowConfig::default();
            cfg.max_peaks_per_tree = max_peaks;
            cfg.tolerance = tolerance;
            cfg.deisotope = mode;
            cfg.peak_preprocessing = pp;
            let wf = SiriusClusteringWorkflow::new(cfg);
            match wf.preprocess_peaks_for_tree(rec.peaks(), precursor_mz, if positive { 1 } else { -1 }) {
                Some(p) => p,
                None => Vec::new(),
            }
        };
        // Keep sparse spectra. The old `< 4` skip silently dropped 5 of the 123
        // CASMI challenges — SIRIUS ranks those from the isotope pattern and the
        // precursor mass alone, so dropping them scores our "no answer" as absent
        // rather than as a miss and quietly shrinks the denominator below the one
        // SIRIUS is measured on. A 1-peak spectrum is a legitimate hard case.
        if peaks.is_empty() {
            continue;
        }
        kept.push(Rec {
            idx,
            name: rec.get_metadata("NAME").unwrap_or("unknown").to_string(),
            smiles,
            true_formula: tf,
            precursor_mz,
            positive,
            adduct: adduct.clone(),
            peaks,
            ms1: parse_ms1(rec.get_metadata("MS1PEAKS")),
        });
    }
    let with_ms1 = kept.iter().filter(|r| r.ms1.len() >= 2).count();
    eprintln!("kept {} spectra ({with_ms1} with an MS1 isotope pattern)", kept.len());
    fs::create_dir_all(&out_dir).unwrap();

    let rules = ChemicalRules::default();

    // Build each generator once. `FormulaGenerator::new` runs a Dijkstra over the
    // extended residue table, which a flamegraph put at 38.7 % of this harness's
    // runtime when it was constructed per spectrum inside the loop below. The
    // library caches these behind `LazyLock` for the same reason; the harness was
    // bypassing that by calling the constructors directly.
    let gen_fclbr = FormulaGenerator::chnops_fclbr();
    // SIRIUS's `-e SBrCl`: CHNOPS + Cl + Br, no fluorine. Matching it is the only
    // way to compare runtimes without handing ourselves a larger search.
    let gen_clbr = FormulaGenerator::chnops_halogens();
    let gen_chnops = FormulaGenerator::chnops();
    let gen_chno = FormulaGenerator::chno();

    let results: Vec<String> = kept
        .par_iter()
        .map(|r| {
            let adduct = r.adduct.clone();
            let neutral = adduct.ion_mz_to_neutral_mass(r.precursor_mz);
            let tf = r.true_formula;

            // De-novo candidate alphabet: CHNOPS+FClBr for every spectrum, which
            // is the honest setting and the one comparable to SIRIUS de-novo.
            //
            // `--oracle-alphabet` restores the old tiering, which picked the
            // element set from the *true* formula (CHNO truth -> CHNO-only
            // search). That deletes most decoys before scoring and inflates
            // top-1 by ~2 points; it exists only to reproduce the pre-round-19
            // numbers in tree_compare_FINDINGS.md. Never quote it as a result.
            let candidates: Vec<Formula> = if no_fluorine {
                gen_clbr.generate_candidates(neutral, tolerance, &rules)
            } else if !oracle_alphabet {
                gen_fclbr.generate_candidates(neutral, tolerance, &rules)
            } else if tf.f() + tf.cl() + tf.br() + tf.i() > 0 {
                gen_fclbr.generate_candidates(neutral, tolerance, &rules)
            } else if tf.p() > 0 || tf.s() > 0 {
                gen_chnops.generate_candidates(neutral, tolerance, &rules)
            } else {
                gen_chno.generate_candidates(neutral, tolerance, &rules)
            };
            let n_cands = candidates.len();
            let true_in_set = candidates.contains(&tf);

            let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
                tolerance,
                adduct: adduct.clone(),
                mass_sigma_ppm: sigma,
                loss_mass_sigma_log: loss_sigma,
                learned_loss_model: learned_losses,
                noise_floor_scale: nf_scale,
                use_isotopes,
                solver,
                scoring,
                tree_size_bonus: tree_size,
                loss_gate,
                exact_top_n,
                peak_noise_median,
                recalibrate,
            });

            let mut scored = if r.ms1.len() >= 2 {
                builder.build_trees_for_candidates_with_ms1(
                    &candidates,
                    r.precursor_mz,
                    &r.peaks,
                    &r.ms1,
                )
            } else {
                builder.build_trees_for_candidates(&candidates, r.precursor_mz, &r.peaks)
            };
            scored.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

            let rank = scored
                .iter()
                .position(|(f, _)| *f == tf)
                .map(|p| p + 1)
                .unwrap_or(9999);

            let (winner_f, winner_score) = scored
                .first()
                .map(|(f, t)| (f.to_string(), t.total_score))
                .unwrap_or_default();
            let true_score = scored
                .iter()
                .find(|(f, _)| *f == tf)
                .map(|(_, t)| t.total_score);
            let true_root = scored
                .iter()
                .find(|(f, _)| *f == tf)
                .and_then(|(_, t)| t.nodes.first())
                .map(|n| n.node_score);

            // the true-formula tree, for the substructure check
            let true_tree = scored.iter().find(|(f, _)| *f == tf).map(|(_, t)| t);
            let (expl_int, nodes_json) = match true_tree {
                Some(t) => {
                    let mut nj = String::from("[");
                    for (k, n) in t.nodes.iter().enumerate().skip(1) {
                        if k > 1 {
                            nj.push(',');
                        }
                        write!(nj, "[\"{}\",{:.5},{:.4}]", n.formula, n.mz, n.relative_intensity).unwrap();
                    }
                    nj.push(']');
                    (t.explained_intensity, nj)
                }
                None => (0.0, "[]".to_string()),
            };

            let mut j = String::from("{");
            write!(j, "\"idx\":{},", r.idx).unwrap();
            write!(j, "\"name\":\"{}\",", json_escape(&r.name)).unwrap();
            write!(j, "\"smiles\":\"{}\",", json_escape(&r.smiles)).unwrap();
            write!(j, "\"true_formula\":\"{}\",", tf).unwrap();
            write!(j, "\"adduct\":\"{}\",", if r.positive { "[M+H]+" } else { "[M-H]-" }).unwrap();
            write!(j, "\"rank\":{rank},\"n_cands\":{n_cands},\"true_in_set\":{true_in_set},").unwrap();
            write!(j, "\"n_peaks\":{},", r.peaks.len()).unwrap();
            write!(j, "\"had_ms1\":{},", r.ms1.len() >= 2).unwrap();
            write!(j, "\"winner\":\"{winner_f}\",\"winner_score\":{winner_score:.4},").unwrap();
            match true_score {
                Some(s) => write!(j, "\"true_score\":{s:.4},").unwrap(),
                None => write!(j, "\"true_score\":null,").unwrap(),
            }
            match true_root {
                Some(s) => write!(j, "\"true_root\":{s:.4},").unwrap(),
                None => write!(j, "\"true_root\":null,").unwrap(),
            }
            write!(j, "\"explained_intensity\":{expl_int:.5},").unwrap();
            write!(j, "\"true_nodes\":{nodes_json}}}").unwrap();
            j
        })
        .collect();

    let mut out = String::from("[\n");
    out.push_str(&results.join(",\n"));
    out.push_str("\n]\n");
    fs::write(format!("{out_dir}/eval.json"), out).unwrap();

    #[cfg(unix)]
    if let (Some(g), Some(path)) = (guard, profile_out.as_ref()) {
        let report = g.report().build().expect("build profile");
        let file = fs::File::create(path).expect("create flamegraph");
        report.flamegraph(file).expect("write flamegraph");
        // Also emit folded stacks, which is what a self-time breakdown needs.
        let mut folded = String::new();
        for (frames, count) in &report.data {
            let mut stack: Vec<String> = Vec::new();
            for f in frames.frames.iter().rev() {
                for sym in f.iter().rev() {
                    stack.push(sym.name());
                }
            }
            folded.push_str(&stack.join(";"));
            folded.push(' ');
            folded.push_str(&count.to_string());
            folded.push('\n');
        }
        fs::write(format!("{path}.folded"), folded).expect("write folded stacks");
        eprintln!("wrote flamegraph to {path} (+ .folded)");
    }

    let n = results.len();
    let alphabet = if oracle_alphabet {
        "ORACLE (leaks the answer)"
    } else if no_fluorine {
        "CHNOPS+ClBr (matches SIRIUS -e SBrCl)"
    } else {
        "CHNOPS+FClBr"
    };
    // Round 54.4 could not re-derive the 88.0 % headline because the run that
    // produced it recorded neither its solver nor its shortlist. Every run now
    // writes the full configuration and its own argv next to the results, so a
    // number can always be traced back to the command that made it.
    let top_n_json = exact_top_n.map_or("null".to_string(), |n| n.to_string());
    let noise_json = peak_noise_median.map_or("null".to_string(), |v: f64| v.to_string());
    let cfg_json = format!(
        "{{\"argv\":{:?},\"mgf\":\"{}\",\"n\":{n},\"limit\":{limit},\"ppm\":{ppm},\
         \"sigma\":{sigma},\"loss_sigma\":{loss_sigma},\"learned_losses\":{learned_losses},\
         \"noise_floor_scale\":{nf_scale},\"isotopes\":{use_isotopes},\"solver\":\"{solver:?}\",\
         \"exact_top_n\":{top_n_json},\"scoring\":\"{scoring:?}\",\"tree_size_bonus\":{tree_size},\
         \"loss_gate\":\"{loss_gate:?}\",\"max_peaks\":{max_peaks},\"min_mass\":{min_mass},\
         \"max_mass\":{max_mass},\"alphabet\":\"{alphabet}\",\"preprocess\":\"{preprocess}\",\
         \"base_peak\":\"{base_peak:?}\",\"adduct_from_mgf\":{adduct_from_mgf},\
         \"precursor_mass_source\":\"{precursor_mass_source}\",\"recalibrate\":{recalibrate},\"peak_noise_median\":{noise_json}}}",
        args, path,
    );
    fs::write(format!("{out_dir}/config.json"), &cfg_json).unwrap();
    println!(
        "solver={solver:?} exact_top_n={exact_top_n:?} max_peaks={max_peaks} mass={min_mass}-{max_mass} sigma={sigma} loss_sigma={loss_sigma} learned={learned_losses} isotopes={use_isotopes} scoring={scoring:?} ppm={ppm} alphabet={alphabet} recal={recalibrate} peak_noise={peak_noise_median:?}: wrote {n} records to {out_dir}/eval.json (config -> {out_dir}/config.json)"
    );
}
