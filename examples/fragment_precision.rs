//! Fragment-annotation precision against MoNA's curator per-peak formulas.
//!
//! Raising `tree_size_bonus` makes the solver keep more of the peaks that
//! decompose — 45 % of them are currently discarded — but a bigger tree only helps
//! a structure model if the added vertices carry the *right* formula. This measures
//! that directly against human-assigned annotations rather than a reachability
//! heuristic.
//!
//! Usage: fragment_precision --truth scratch/mona/truth.json --mgf scratch/mona/eval.mgf
//!                           [--tree-size B] [--max-peaks M] [--ppm P]

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{Adduct, Element, Formula, FragmentationTreeBuilder, TreeSolverConfig};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::collections::HashMap;
use std::str::FromStr;

/// The curator names the ion, our tree names the neutral fragment: for [M+H]+ the
/// ion carries one more hydrogen, for [M-H]- one fewer.
fn as_ion(f: &Formula, positive: bool) -> Option<Formula> {
    let h = f.get(Element::H);
    let mut out = *f;
    if positive {
        out.set(Element::H, h + 1);
    } else if h == 0 {
        return None;
    } else {
        out.set(Element::H, h - 1);
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

    let truth: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&truth_path).expect("truth")).unwrap();

    let mut parser = MgfParser::default();
    let records = parser.parse_file(&mgf_path).expect("mgf");

    let (mut matched, mut total_nodes, mut curator_hit, mut curator_total, mut n) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut nodes_sum = 0usize;

    for rec in &records {
        let Some(id) = rec.get_metadata("TITLE").or_else(|| rec.get_metadata("NAME")) else { continue };
        let id = id.trim();
        let Some(entry) = truth.get(id) else { continue };
        let Some(ann) = entry.get("annotated").and_then(|v| v.as_array()) else { continue };
        if ann.is_empty() { continue; }
        let Some(tf) = entry.get("true_formula").and_then(|v| v.as_str()) else { continue };
        let Ok(formula) = Formula::from_str(tf) else { continue };
        let positive = entry.get("adduct").and_then(|v| v.as_str()).is_none_or(|a| !a.contains('-'));
        let adduct = if positive { Adduct::protonated() } else { Adduct::deprotonated() };

        // Curator annotations are ION formulas keyed by m/z.
        let mut want: HashMap<i64, String> = HashMap::new();
        for a in ann {
            if let (Some(mz), Some(f)) = (a[0].as_f64(), a[1].as_str()) {
                want.insert((mz * 100.0).round() as i64, f.to_string());
            }
        }

        let mut peaks: Vec<Peak> = rec.peaks().to_vec();
        peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
        peaks.truncate(max_peaks);
        let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
            tolerance: MassTolerance::Ppm(ppm),
            adduct: adduct.clone(),
            tree_size_bonus: tree_size,
            ..Default::default()
        });
        let precursor_mz = adduct.neutral_mass_to_ion_mz(formula.exact_mass());
        let tree = builder.build_tree(&formula, precursor_mz, &peaks);

        n += 1;
        nodes_sum += tree.nodes.len();
        for node in tree.nodes.iter().skip(1) {
            total_nodes += 1;
            let key = (node.mz * 100.0).round() as i64;
            // The curator names the ion; our node names the neutral fragment.
            for probe in [key - 1, key, key + 1] {
                if let Some(curator) = want.get(&probe) {
                    let ours = as_ion(&node.formula, positive).map(|f| f.to_string());
                    if ours.as_deref() == Some(curator.as_str()) {
                        matched += 1;
                    }
                    break;
                }
            }
        }
        for (mz_key, curator) in &want {
            curator_total += 1;
            let hit = tree.nodes.iter().skip(1).any(|node| {
                let k = (node.mz * 100.0).round() as i64;
                (k - mz_key).abs() <= 1
                    && as_ion(&node.formula, positive).map(|f| f.to_string()).as_deref()
                        == Some(curator.as_str())
            });
            if hit { curator_hit += 1; }
        }
    }

    println!("tree-size {tree_size:<5} max-peaks {max_peaks}  spectra {n}");
    println!("  nodes/tree                {:.1}", nodes_sum as f64 / n.max(1) as f64);
    println!("  precision (our nodes correct)  {}/{} = {:.1}%", matched, total_nodes,
             100.0 * matched as f64 / total_nodes.max(1) as f64);
    println!("  recall (curator peaks found)   {}/{} = {:.1}%", curator_hit, curator_total,
             100.0 * curator_hit as f64 / curator_total.max(1) as f64);
}
