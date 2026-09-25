//! Where does a curator-annotated peak get lost?
//!
//! `fragment_precision` reports one number — ~52 % of MoNA's curator per-peak
//! formulas end up in our tree — and calls it a ceiling. A single rate cannot say
//! *why*, and the three plausible causes want opposite fixes. This walks each
//! annotated peak down a ladder and reports where it falls off:
//!
//!   A  annotated peaks in the truth file
//!   B  the curator's neutral fragment is a subformula of the precursor
//!      (a "no" means the annotation implies atoms the precursor does not have —
//!      a rearrangement, a co-eluting contaminant, a different adduct, or a
//!      curator error. Nothing in a subformula-based tree can ever place it.)
//!   C  the annotation fits its own peak: theoretical ion m/z within tolerance of
//!      the listed m/z
//!   D  the peak survived the peak budget
//!   E  our annotator emitted that formula among the peak's candidates
//!      (B-but-not-E is the generator's element bounds / `ChemicalRules`)
//!   F  the tree solver kept it
//!
//! E -> F is a solver/objective question; B -> E is a decomposition question;
//! anything failing B is not ours to fix. Usage mirrors `fragment_precision`.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{
    Adduct, Element, Formula, FragmentationTreeBuilder, SolverKind, TreeSolverConfig,
};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::str::FromStr;

/// The curator names the ion, our tree names the neutral fragment.
fn neutral_of_ion(f: &Formula, positive: bool) -> Option<Formula> {
    let h = f.get(Element::H);
    let mut out = *f;
    if positive {
        if h == 0 {
            return None;
        }
        out.set(Element::H, h - 1);
    } else {
        out.set(Element::H, h + 1);
    }
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |n: &str, d: &str| -> String {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into())
    };
    let truth_path = get("--truth", "scratch/mona/truth.json");
    let mgf_path = get("--mgf", "scratch/mona/eval.mgf");
    let tree_size: f64 = get("--tree-size", "3").parse().unwrap();
    let max_peaks: usize = get("--max-peaks", "60").parse().unwrap();
    let ppm: f64 = get("--ppm", "20").parse().unwrap();
    let limit: usize = get("--limit", "100000").parse().unwrap();

    let truth: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&truth_path).expect("truth")).unwrap();
    let mut parser = MgfParser::default();
    let records = parser.parse_file(&mgf_path).expect("mgf");

    let (mut a, mut b, mut c, mut d, mut e, mut f) = (0usize, 0, 0, 0, 0, 0);
    let mut n = 0usize;
    // What the B failures look like: which element the annotation exceeds.
    let mut b_fail_element: std::collections::HashMap<&'static str, usize> = Default::default();
    // Intensity rank of the peaks lost at E->F, to see whether the solver is
    // discarding noise or signal.
    let mut lost_at_f_relint: Vec<f32> = Vec::new();
    let mut kept_relint: Vec<f32> = Vec::new();

    for rec in &records {
        if n >= limit {
            break;
        }
        let Some(id) = rec.get_metadata("TITLE").or_else(|| rec.get_metadata("NAME")) else { continue };
        let Some(entry) = truth.get(id.trim()) else { continue };
        let Some(ann) = entry.get("annotated").and_then(|v| v.as_array()) else { continue };
        if ann.is_empty() {
            continue;
        }
        let Some(tf) = entry.get("true_formula").and_then(|v| v.as_str()) else { continue };
        let Ok(precursor) = Formula::from_str(tf) else { continue };
        let positive = entry.get("adduct").and_then(|v| v.as_str()).is_none_or(|s| !s.contains('-'));
        let adduct = if positive { Adduct::protonated() } else { Adduct::deprotonated() };

        let mut peaks: Vec<Peak> = rec.peaks().to_vec();
        peaks.sort_by(|x, y| y.intensity.total_cmp(&x.intensity));
        peaks.truncate(max_peaks);
        let max_int = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);

        let cfg = TreeSolverConfig {
            tolerance: MassTolerance::Ppm(ppm),
            adduct: adduct.clone(),
            tree_size_bonus: tree_size,
            solver: SolverKind::Ilp,
            ..Default::default()
        };
        let builder = FragmentationTreeBuilder::new(cfg.clone());
        let precursor_mz = adduct.neutral_mass_to_ion_mz(precursor.exact_mass());
        let annotated = builder.annotator.annotate_peaks(&peaks, &precursor, cfg.tolerance);
        let tree = builder.build_tree_from_annotated(&precursor, precursor_mz, &peaks, &annotated);
        n += 1;

        for row in ann {
            let (Some(mz), Some(fs)) = (row[0].as_f64(), row[1].as_str()) else { continue };
            a += 1;
            let Ok(ion) = Formula::from_str(fs) else { continue };
            let Some(neutral) = neutral_of_ion(&ion, positive) else { continue };

            // B — representable at all
            if !neutral.is_subformula_of(&precursor) {
                for el in [Element::C, Element::H, Element::N, Element::O, Element::P, Element::S] {
                    if neutral.get(el) > precursor.get(el) {
                        *b_fail_element.entry(el.symbol()).or_default() += 1;
                    }
                }
                continue;
            }
            b += 1;

            // C — the annotation fits its own m/z
            let theo_mz = adduct.neutral_mass_to_ion_mz(neutral.exact_mass());
            let tol_da = mz * ppm * 1e-6;
            if (theo_mz - mz).abs() > tol_da.max(0.002) {
                continue;
            }
            c += 1;

            // D — peak survived the budget
            let Some(pi) = peaks
                .iter()
                .position(|p| (p.mz - mz).abs() <= tol_da.max(0.002))
            else {
                continue;
            };
            d += 1;
            let relint = peaks[pi].intensity / max_int;

            // E — the annotator offered it
            let offered = annotated.iter().any(|ap| {
                (ap.mz - mz).abs() <= tol_da.max(0.002)
                    && ap.candidates.iter().any(|sc| sc.formula == neutral)
            });
            if !offered {
                continue;
            }
            e += 1;

            // F — the solver kept it
            let kept = tree.nodes.iter().skip(1).any(|node| node.formula == neutral);
            if kept {
                f += 1;
                kept_relint.push(relint);
            } else {
                lost_at_f_relint.push(relint);
            }
        }
    }

    let pct = |x: usize| 100.0 * x as f64 / a.max(1) as f64;
    println!("spectra {n}   curator-annotated peaks {a}");
    println!("  B  neutral is a subformula of precursor   {b:6}  {:5.1}%", pct(b));
    println!("  C  annotation fits its own m/z @ {ppm} ppm   {c:6}  {:5.1}%", pct(c));
    println!("  D  peak survived the {max_peaks}-peak budget      {d:6}  {:5.1}%", pct(d));
    println!("  E  our annotator offered the formula     {e:6}  {:5.1}%", pct(e));
    println!("  F  the solver kept it                    {f:6}  {:5.1}%", pct(f));
    println!();
    println!("  stage losses: A-B {}  B-C {}  C-D {}  D-E {}  E-F {}",
             a - b, b - c, c - d, d - e, e - f);
    if !b_fail_element.is_empty() {
        let mut v: Vec<_> = b_fail_element.into_iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        println!("  A-B excess element (annotation has more than the precursor): {v:?}");
    }
    let med = |mut v: Vec<f32>| -> f32 {
        if v.is_empty() { return f32::NAN; }
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    println!("  median rel. intensity  kept {:.3}   dropped at E-F {:.3}",
             med(kept_relint), med(lost_at_f_relint));
}
