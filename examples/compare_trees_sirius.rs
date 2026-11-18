//! Dump our fragmentation trees alongside SIRIUS-ready inputs, for a tree-vs-tree
//! comparison against the reference implementation.
//!
//! Retrieval accuracy is a function of the tree, so the question "are our trees
//! the same as SIRIUS's" is worth answering directly rather than only through
//! downstream formula ranking. This tool takes annotated spectra (GNPS-style MGF
//! with a `TRUEFORMULA=` or `INCHI=` tag), builds our tree under the *correct*
//! precursor formula, and writes:
//!
//!   <out>/ms/<idx>.ms        SIRIUS input, precursor formula pinned via `>formula`
//!   <out>/ours/<idx>.json    our tree, cleaned peaks, metadata
//!   <out>/index.json         the manifest
//!
//! The SIRIUS run and the actual comparison live in `scratch/compare_trees.py`,
//! which drives SIRIUS 5.6.3 over `<out>/ms` and diffs the two tree sets.
//!
//! Usage:
//!   compare_trees_sirius <spectra.mgf> [--limit N] [--ppm P] [--max-peaks M] [--out DIR]

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{Adduct, Formula, FragmentationTreeBuilder, SolverKind, TreeSolverConfig};
use oxidems_lib::sirius::solver::ScoringModel;
use oxidems_lib::sirius::rules::LossGate;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::fmt::Write as _;
use std::fs;
use std::str::FromStr;

fn extract_formula_from_inchi(inchi: &str) -> Option<Formula> {
    let rest = inchi.strip_prefix("InChI=")?;
    let formula_part = rest.split('/').nth(1)?;
    let clean = formula_part.split('.').next().unwrap_or(formula_part);
    Formula::from_str(clean).ok()
}

struct Record {
    idx: usize,
    name: String,
    smiles: String,
    true_formula: Formula,
    precursor_mz: f64,
    positive: bool,
    peaks: Vec<Peak>,
}

fn clean_peaks(raw: &[Peak], precursor_mz: f64, max_peaks: usize) -> Vec<Peak> {
    let max_int = raw.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-9);
    let mut peaks: Vec<Peak> = raw
        .iter()
        .filter(|p| p.intensity / max_int >= 0.005)
        // keep nothing above the precursor + a small isolation margin
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

mod support;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .unwrap_or_else(|| support::fail("usage: compare_trees_sirius <spectra.mgf> [--limit N] [--ppm P] [--max-peaks M] [--out DIR]"));

    let mut limit = 200usize;
    let mut ppm = 10.0f64;
    let mut sigma = TreeSolverConfig::default().mass_sigma_ppm;
    let mut max_peaks = 40usize;
    let mut out_dir = "scratch/tree_compare".to_string();
    let mut solver = SolverKind::CriticalPath;
    // Follow the library default. Hard-coding Native here meant every tree
    // comparison scored our trees with a different objective than the one the
    // pipeline actually ships, and then blamed the difference on SIRIUS.
    let mut scoring = TreeSolverConfig::default().scoring;
    let mut tree_size = TreeSolverConfig::default().tree_size_bonus;
    let mut loss_gate = TreeSolverConfig::default().loss_gate;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                limit = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ppm" => {
                ppm = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--sigma" => {
                sigma = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-peaks" => {
                max_peaks = args[i + 1].parse().unwrap();
                i += 2;
            }
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
            "--out" => {
                out_dir = args[i + 1].clone();
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }

    let mut parser = MgfParser::default();
    let records = parser.parse_file(path).expect("parse mgf");
    eprintln!("parsed {} raw spectra from {path}", records.len());

    let tolerance = MassTolerance::Ppm(ppm);
    let mut kept: Vec<Record> = Vec::new();

    for rec in &records {
        if kept.len() >= limit {
            break;
        }

        let true_formula = rec
            .get_metadata("TRUEFORMULA")
            .and_then(|s| Formula::from_str(s.trim()).ok())
            .or_else(|| rec.get_metadata("FORMULA").and_then(|s| Formula::from_str(s.trim()).ok()))
            .or_else(|| rec.get_metadata("INCHI").and_then(extract_formula_from_inchi));
        let Some(true_formula) = true_formula else { continue };

        // Only the two bread-and-butter adducts; anything else is a separate study.
        let ionmode = rec.get_metadata("IONMODE").unwrap_or("Positive");
        let positive = !ionmode.eq_ignore_ascii_case("negative");
        let adduct = if positive { Adduct::protonated() } else { Adduct::deprotonated() };

        let Some(pepmass) = rec
            .get_metadata("PEPMASS")
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
        else {
            continue;
        };

        let exact = true_formula.exact_mass();
        if !(80.0..=800.0).contains(&exact) {
            continue;
        }
        // The precursor formula is pinned for both tools, so the precursor m/z is a
        // known quantity: use the theoretical [M+H]+ / [M-H]- mass everywhere and
        // keep the measured PEPMASS only as a sanity gate. GNPS PEPMASS is rounded
        // to 3 dp and occasionally just wrong; propagating it would charge one tool
        // a recalibration the other never sees.
        let expected_mz = adduct.neutral_mass_to_ion_mz(exact);
        if (expected_mz - pepmass).abs() > 0.03 {
            continue;
        }
        let precursor_mz = expected_mz;

        let peaks = clean_peaks(rec.peaks(), precursor_mz, max_peaks);
        if peaks.len() < 4 {
            continue;
        }

        let name = rec
            .get_metadata("NAME")
            .or_else(|| rec.get_metadata("COMPOUND_NAME"))
            .unwrap_or("unknown")
            .to_string();
        let smiles = rec.get_metadata("SMILES").unwrap_or("").trim().to_string();

        kept.push(Record {
            idx: kept.len(),
            name,
            smiles,
            true_formula,
            precursor_mz,
            positive,
            peaks,
        });
    }

    eprintln!("kept {} spectra after filtering", kept.len());

    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(format!("{out_dir}/ms")).unwrap();
    fs::create_dir_all(format!("{out_dir}/ours")).unwrap();

    let mut manifest = String::from("[\n");

    for rec in &kept {
        let adduct = if rec.positive { Adduct::protonated() } else { Adduct::deprotonated() };
        let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
            tolerance,
            adduct: adduct.clone(),
            mass_sigma_ppm: sigma,
            solver,
            scoring,
            tree_size_bonus: tree_size,
            loss_gate,
            ..Default::default()
        });
        // Go through the multi-candidate path (with the true formula as the sole
        // candidate) so the spectrum's systematic mass drift is estimated and
        // taken out before the tree is built, as it is in the real workflow.
        let tree = builder
            .build_trees_for_candidates(&[rec.true_formula], rec.precursor_mz, &rec.peaks)
            .pop()
            .map(|(_, t)| t)
            .unwrap_or_else(|| builder.build_tree(&rec.true_formula, rec.precursor_mz, &rec.peaks));

        // --- SIRIUS .ms input, precursor formula pinned so both build under the same root
        let mut ms = String::new();
        writeln!(ms, ">compound {}", rec.idx).unwrap();
        writeln!(ms, ">parentmass {:.6}", rec.precursor_mz).unwrap();
        writeln!(ms, ">formula {}", rec.true_formula).unwrap();
        writeln!(ms, ">ionization {}", if rec.positive { "[M+H]+" } else { "[M-H]-" }).unwrap();
        writeln!(ms, "\n>ms2").unwrap();
        for p in &rec.peaks {
            writeln!(ms, "{:.6} {:.4}", p.mz, p.intensity).unwrap();
        }
        fs::write(format!("{out_dir}/ms/{}.ms", rec.idx), ms).unwrap();

        // --- our tree as JSON
        let mut j = String::new();
        write!(j, "{{").unwrap();
        write!(j, "\"idx\":{},", rec.idx).unwrap();
        write!(j, "\"name\":\"{}\",", json_escape(&rec.name)).unwrap();
        write!(j, "\"true_formula\":\"{}\",", rec.true_formula).unwrap();
        write!(j, "\"precursor_mz\":{},", rec.precursor_mz).unwrap();
        write!(j, "\"adduct\":\"{}\",", if rec.positive { "[M+H]+" } else { "[M-H]-" }).unwrap();
        write!(j, "\"ppm\":{ppm},").unwrap();
        write!(j, "\"explained_intensity\":{},", tree.explained_intensity).unwrap();
        write!(j, "\"explained_peaks\":{},", tree.explained_peaks).unwrap();
        write!(j, "\"total_peaks\":{},", tree.total_peaks).unwrap();
        write!(j, "\"total_score\":{},", tree.total_score).unwrap();

        write!(j, "\"peaks\":[").unwrap();
        for (k, p) in rec.peaks.iter().enumerate() {
            if k > 0 {
                write!(j, ",").unwrap();
            }
            write!(j, "[{:.6},{:.4}]", p.mz, p.intensity).unwrap();
        }
        write!(j, "],").unwrap();

        write!(j, "\"nodes\":[").unwrap();
        for (k, n) in tree.nodes.iter().enumerate() {
            if k > 0 {
                write!(j, ",").unwrap();
            }
            write!(
                j,
                "{{\"id\":{},\"peak_index\":{},\"formula\":\"{}\",\"mz\":{},\"rel_intensity\":{},\"score\":{}}}",
                n.id,
                n.peak_index.map(|v| v as i64).unwrap_or(-1),
                n.formula,
                n.mz,
                n.relative_intensity,
                n.node_score
            )
            .unwrap();
        }
        write!(j, "],").unwrap();

        write!(j, "\"edges\":[").unwrap();
        for (k, e) in tree.edges.iter().enumerate() {
            if k > 0 {
                write!(j, ",").unwrap();
            }
            let pf = &tree.nodes[e.parent_id].formula;
            let cf = &tree.nodes[e.child_id].formula;
            write!(
                j,
                "{{\"parent\":{},\"child\":{},\"parent_formula\":\"{}\",\"child_formula\":\"{}\",\"loss\":\"{}\",\"loss_mass\":{},\"score\":{},\"common\":{}}}",
                e.parent_id, e.child_id, pf, cf, e.loss_formula, e.loss_exact_mass, e.edge_score, e.is_common_loss
            )
            .unwrap();
        }
        write!(j, "]}}").unwrap();

        fs::write(format!("{out_dir}/ours/{}.json", rec.idx), j).unwrap();

        if rec.idx > 0 {
            manifest.push_str(",\n");
        }
        write!(
            manifest,
            "  {{\"idx\":{},\"name\":\"{}\",\"smiles\":\"{}\",\"true_formula\":\"{}\",\"adduct\":\"{}\",\"precursor_mz\":{},\"n_peaks\":{}}}",
            rec.idx,
            json_escape(&rec.name),
            json_escape(&rec.smiles),
            rec.true_formula,
            if rec.positive { "[M+H]+" } else { "[M-H]-" },
            rec.precursor_mz,
            rec.peaks.len()
        )
        .unwrap();
    }
    manifest.push_str("\n]\n");
    fs::write(format!("{out_dir}/index.json"), manifest).unwrap();

    println!("wrote {} cases to {out_dir}/", kept.len());
    println!("next: python3 scratch/compare_trees.py {out_dir} --ppm {ppm}");
}
