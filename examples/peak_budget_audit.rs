//! Where do a spectrum's peaks go between the file and the tree?
//!
//! Three places a peak can be lost, and they call for different fixes:
//!   1. the peak budget (`--max-peaks`) truncates it before anything looks at it;
//!   2. it has no subformula of the precursor within tolerance — genuinely
//!      unexplainable, or the tolerance is too tight;
//!   3. it decomposes but the tree solver leaves it out.
//!
//! Usage: peak_budget_audit [--limit N] [--max-peaks M] [--ppm P]

use oxidems_lib::sirius::{
    Adduct, Formula, FragmentationTreeBuilder, SpectrumAnnotator, TreeSolverConfig,
};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

mod support;

fn main() {
    support::require("peak_budget_audit", &[("test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp", support::MONA)]);
    let args: Vec<String> = std::env::args().collect();
    let arg = |n: &str, d: usize| -> usize {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let limit = arg("--limit", 2000);
    let max_peaks = arg("--max-peaks", 60);
    let ppm: f64 = args.iter().position(|a| a == "--ppm").and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok()).unwrap_or(20.0);

    let f = File::open("test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp").expect("msp");
    let mut rdr = BufReader::with_capacity(1 << 24, f);
    let (mut formula, mut mz) = (String::new(), 0.0f64);
    let mut peaks: Vec<Peak> = Vec::new();
    let mut line = String::new();
    let (mut raw, mut kept, mut decomposed, mut in_tree, mut n) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut raw_hist: Vec<usize> = Vec::new();

    let tol = MassTolerance::Ppm(ppm);
    // The solver's appetite for vertices. Tuned for formula ranking, where a tight
    // tree is a virtue; structure prediction wants every annotated fragment it can
    // defend, so it is worth knowing how the node count responds.
    let tree_size: f64 = args.iter().position(|a| a == "--tree-size").and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok()).unwrap_or(TreeSolverConfig::default().tree_size_bonus);
    let cfg = TreeSolverConfig { tolerance: tol, adduct: Adduct::protonated(), tree_size_bonus: tree_size, ..Default::default() };
    let builder = FragmentationTreeBuilder::new(cfg);
    let annotator = SpectrumAnnotator::new(2.0).with_adduct(Adduct::protonated());

    loop {
        line.clear();
        if rdr.read_line(&mut line).unwrap_or(0) == 0 { break; }
        let t = line.trim();
        if t.is_empty() {
            if let (Ok(f), true) = (Formula::from_str(&formula), mz > 0.0 && peaks.len() >= 3) {
                if (50.0..=800.0).contains(&f.exact_mass()) {
                    raw += peaks.len();
                    raw_hist.push(peaks.len());
                    let mut p = peaks.clone();
                    p.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
                    p.truncate(max_peaks);
                    kept += p.len();
                    let ann = annotator.annotate_peaks(&p, &f, tol);
                    decomposed += ann.iter().filter(|x| !x.candidates.is_empty()).count();
                    in_tree += builder.build_tree(&f, mz, &p).nodes.len();
                    n += 1;
                    if n >= limit { break; }
                }
            }
            formula.clear(); mz = 0.0; peaks.clear();
            continue;
        }
        if let Some(v) = t.strip_prefix("Formula:") { formula = v.trim().to_string(); }
        else if let Some(v) = t.strip_prefix("PrecursorMZ:") { mz = v.trim().parse().unwrap_or(0.0); }
        else if t.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            let mut it = t.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Ok(m), Ok(i)) = (a.parse::<f64>(), b.parse::<f32>()) {
                    if m > 0.0 && i > 0.0 { peaks.push(Peak { mz: m, intensity: i }); }
                }
            }
        }
    }

    raw_hist.sort_unstable();
    let nf = n as f64;
    println!("spectra audited: {n}   (--max-peaks {max_peaks}, {ppm} ppm, tree-size {tree_size})");
    println!("  raw peaks in file      {:6.1} per spectrum   (median {})", raw as f64 / nf, raw_hist[raw_hist.len()/2]);
    println!("  survive the budget     {:6.1}   -> {:5.1}% lost to --max-peaks", kept as f64 / nf, 100.0 * (raw - kept) as f64 / raw as f64);
    println!("  decompose at all       {:6.1}   -> {:5.1}% of kept have no subformula", decomposed as f64 / nf, 100.0 * (kept - decomposed) as f64 / kept as f64);
    println!("  end up in the tree     {:6.1}   -> {:5.1}% of decomposable dropped by the solver", in_tree as f64 / nf, 100.0 * (decomposed as i64 - in_tree as i64) as f64 / decomposed as f64);
}
