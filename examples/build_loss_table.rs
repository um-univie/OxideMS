//! Learn neutral-loss frequencies from labelled GNPS spectra, split by ion mode.
//!
//! Positive and negative ESI fragment differently (protonated even-electron vs
//! deprotonated, different favoured losses), so the loss prior that feeds
//! `score_loss` should be mode-specific and data-derived rather than a single
//! hand-curated list.
//!
//! For every spectrum with a known formula we annotate each peak with its best
//! sub-formula (under the *true* precursor formula), then enumerate every
//! candidate fragmentation-graph edge — an ordered pair (heavier, lighter) whose
//! formula difference is a chemically valid neutral loss — and tally the loss
//! formulas, weighted by child-peak intensity. This is topology-free: it counts
//! what losses are *available*, not what one tree picked.
//!
//! Output: `<out>/loss_table_pos.json`, `<out>/loss_table_neg.json`
//!   { "totals": {edges, weighted, spectra},
//!     "losses": { "<formula>": [n_edges, weighted, n_spectra] },
//!     "root_losses": { "<formula>": [n_edges, weighted, n_spectra] } }
//!
//! Usage:
//!   build_loss_table <gnps.mgf> [--limit N] [--ppm P] [--max-peaks M] [--split tuning|holdout] [--out DIR]

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{Adduct, ChemicalRules, Formula, SpectrumAnnotator};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::str::FromStr;

fn formula_from_inchi(inchi: &str) -> Option<Formula> {
    let rest = inchi.strip_prefix("InChI=")?;
    let part = rest.split('/').nth(1)?;
    Formula::from_str(part.split('.').next().unwrap_or(part)).ok()
}

struct Rec {
    formula: Formula,
    precursor_mz: f64,
    positive: bool,
    peaks: Vec<Peak>,
}

/// loss formula -> (edge count, intensity-weighted count, spectra it appeared in)
type Tally = HashMap<Formula, (u64, f64, u64)>;

#[derive(Default)]
struct Acc {
    losses: Tally,
    root_losses: Tally,
    total_edges: u64,
    total_weighted: f64,
    spectra: u64,
}

impl Acc {
    fn merge(&mut self, other: Acc) {
        for (k, (c, w, s)) in other.losses {
            let e = self.losses.entry(k).or_default();
            e.0 += c;
            e.1 += w;
            e.2 += s;
        }
        for (k, (c, w, s)) in other.root_losses {
            let e = self.root_losses.entry(k).or_default();
            e.0 += c;
            e.1 += w;
            e.2 += s;
        }
        self.total_edges += other.total_edges;
        self.total_weighted += other.total_weighted;
        self.spectra += other.spectra;
    }
}

fn clean_peaks(raw: &[Peak], precursor_mz: f64, max_peaks: usize) -> Vec<Peak> {
    let max_int = raw.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-9);
    let mut peaks: Vec<Peak> = raw
        .iter()
        .filter(|p| p.intensity / max_int >= 0.005 && p.mz <= precursor_mz + 5.0 && p.mz > 0.0)
        .copied()
        .collect();
    peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    peaks.truncate(max_peaks);
    peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
    peaks.dedup_by(|a, b| (a.mz - b.mz).abs() < 1e-4);
    peaks
}

fn process(rec: &Rec, tol: MassTolerance, rules: &ChemicalRules) -> Acc {
    let adduct = if rec.positive { Adduct::protonated() } else { Adduct::deprotonated() };
    let annotator = SpectrumAnnotator::new(2.0).with_adduct(adduct);
    let annotated = annotator.annotate_peaks(&rec.peaks, &rec.formula, tol);

    // node = (formula, child-weight); include the precursor as node 0
    let mut nodes: Vec<(Formula, f64)> = vec![(rec.formula, 1.0)];
    for ap in &annotated {
        if let Some(best) = ap.candidates.first() {
            if best.formula.is_subformula_of(&rec.formula) {
                nodes.push((best.formula, ap.relative_intensity as f64));
            }
        }
    }
    if nodes.len() < 3 {
        return Acc::default();
    }

    let mut acc = Acc { spectra: 1, ..Default::default() };
    let mut seen: std::collections::HashSet<Formula> = std::collections::HashSet::new();
    let mut seen_root: std::collections::HashSet<Formula> = std::collections::HashSet::new();

    for h in 0..nodes.len() {
        for l in 0..nodes.len() {
            if h == l {
                continue;
            }
            let (hf, _) = nodes[h];
            let (lf, lw) = nodes[l];
            let Some(loss) = hf.checked_sub(&lf) else { continue };
            if loss.total_atoms() == 0 || !rules.is_valid_neutral_loss(&loss) {
                continue;
            }
            acc.total_edges += 1;
            acc.total_weighted += lw;
            let e = acc.losses.entry(loss).or_default();
            e.0 += 1;
            e.1 += lw;
            if seen.insert(loss) {
                e.2 += 1;
            }
            if h == 0 {
                let r = acc.root_losses.entry(loss).or_default();
                r.0 += 1;
                r.1 += lw;
                if seen_root.insert(loss) {
                    r.2 += 1;
                }
            }
        }
    }
    acc
}

fn write_table(path: &str, acc: &Acc) {
    let mut j = String::from("{\n");
    write!(
        j,
        "  \"totals\": {{\"edges\": {}, \"weighted\": {:.4}, \"spectra\": {}}},\n",
        acc.total_edges, acc.total_weighted, acc.spectra
    )
    .unwrap();

    let dump = |name: &str, t: &Tally, j: &mut String, last: bool| {
        let mut items: Vec<_> = t.iter().collect();
        items.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        write!(j, "  \"{name}\": {{\n").unwrap();
        for (i, (f, (c, w, s))) in items.iter().enumerate() {
            let comma = if i + 1 == items.len() { "" } else { "," };
            write!(j, "    \"{f}\": [{c}, {w:.4}, {s}]{comma}\n").unwrap();
        }
        write!(j, "  }}{}\n", if last { "" } else { "," }).unwrap();
    };
    dump("losses", &acc.losses, &mut j, false);
    dump("root_losses", &acc.root_losses, &mut j, true);
    j.push_str("}\n");
    fs::write(path, j).unwrap();
}

mod support;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| support::fail("usage: build_loss_table <gnps.mgf> [--limit N] [--ppm P] [--max-peaks M] [--split S] [--out DIR]"));

    let mut limit = 40_000usize;
    let mut ppm = 10.0f64;
    let mut max_peaks = 60usize;
    let mut split: Option<String> = None;
    let mut out_dir = "scratch/loss_table".to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => { limit = args[i + 1].parse().unwrap(); i += 2; }
            "--ppm" => { ppm = args[i + 1].parse().unwrap(); i += 2; }
            "--max-peaks" => { max_peaks = args[i + 1].parse().unwrap(); i += 2; }
            "--split" => { split = Some(args[i + 1].clone()); i += 2; }
            "--out" => { out_dir = args[i + 1].clone(); i += 2; }
            other => panic!("unknown arg {other}"),
        }
    }

    let mut parser = MgfParser::default();
    eprintln!("parsing {path} ...");
    let records = parser.parse_file(path).expect("parse mgf");
    eprintln!("parsed {} spectra", records.len());

    let tol = MassTolerance::Ppm(ppm);
    let (mut pos, mut neg): (Vec<Rec>, Vec<Rec>) = (Vec::new(), Vec::new());

    for r in &records {
        if pos.len() >= limit && neg.len() >= limit {
            break;
        }
        if let Some(want) = &split {
            if r.get_metadata("BENCHMARKSPLIT").map(str::trim) != Some(want.as_str()) {
                // ALL_GNPS has no split tag; only filter when the tag exists somewhere
                if r.get_metadata("BENCHMARKSPLIT").is_some() {
                    continue;
                }
            }
        }
        let formula = r
            .get_metadata("FORMULA")
            .and_then(|s| Formula::from_str(s.trim()).ok())
            .or_else(|| r.get_metadata("INCHI").and_then(formula_from_inchi));
        let Some(formula) = formula else { continue };
        let exact = formula.exact_mass();
        if !(80.0..=1000.0).contains(&exact) {
            continue;
        }
        let ionmode = r.get_metadata("IONMODE").unwrap_or("Positive");
        let positive = !ionmode.eq_ignore_ascii_case("negative");
        let adduct = if positive { Adduct::protonated() } else { Adduct::deprotonated() };
        let Some(pep) = r
            .get_metadata("PEPMASS")
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
        else {
            continue;
        };
        let precursor_mz = adduct.neutral_mass_to_ion_mz(exact);
        if (precursor_mz - pep).abs() > 0.02 {
            continue;
        }
        let raw = r.peaks();
        if raw.len() < 4 || raw.len() > 300 {
            continue;
        }
        let peaks = clean_peaks(raw, precursor_mz, max_peaks);
        if peaks.len() < 4 {
            continue;
        }
        let rec = Rec { formula, precursor_mz, positive, peaks };
        if positive {
            if pos.len() < limit {
                pos.push(rec);
            }
        } else if neg.len() < limit {
            neg.push(rec);
        }
    }
    eprintln!("kept {} positive, {} negative", pos.len(), neg.len());

    let rules = ChemicalRules::loose();
    let fold = |recs: &[Rec]| -> Acc {
        recs.par_iter()
            .map(|r| process(r, tol, &rules))
            .reduce(Acc::default, |mut a, b| {
                a.merge(b);
                a
            })
    };

    fs::create_dir_all(&out_dir).unwrap();
    let acc_pos = fold(&pos);
    let acc_neg = fold(&neg);
    write_table(&format!("{out_dir}/loss_table_pos.json"), &acc_pos);
    write_table(&format!("{out_dir}/loss_table_neg.json"), &acc_neg);
    println!(
        "pos: {} spectra, {} candidate edges, {} distinct losses",
        acc_pos.spectra,
        acc_pos.total_edges,
        acc_pos.losses.len()
    );
    println!(
        "neg: {} spectra, {} candidate edges, {} distinct losses",
        acc_neg.spectra,
        acc_neg.total_edges,
        acc_neg.losses.len()
    );
    println!("wrote {out_dir}/loss_table_{{pos,neg}}.json");
}
