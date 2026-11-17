//! Does `preprocess_peaks_for_tree` destroy chlorine isotope evidence?
//!
//! Step 3 removes satellites at M + k*1.00335 for k in 1..=3. The k=2 target is
//! M + 2.00670. A 37Cl satellite sits at M + 1.99705 — **9.65 mDa away**, and the
//! tolerance is `max(ppm*precursor/1e6, 0.01)`, i.e. never below 10 mDa. So a
//! chlorine M+2 peak is inside the window for "second 13C" on every spectrum, and
//! the intensity guard (<= 1.5x the mono peak) does not save it either: 37Cl comes
//! in at ~32%.
use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{SiriusClusteringWorkflow, SiriusWorkflowConfig};
use oxidems_lib::spectrum::Peak;

const D_CL: f64 = 1.99705;

fn doublets(peaks: &[Peak]) -> usize {
    let mx = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-9);
    let mut n = 0;
    for a in peaks {
        if a.intensity / mx < 0.03 {
            continue;
        }
        for b in peaks {
            let d = b.mz - a.mz;
            if (d - D_CL).abs() < 0.004 {
                let r = b.intensity / a.intensity;
                if (0.24..=0.42).contains(&r) {
                    n += 1;
                    break;
                }
            }
        }
    }
    n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(truth_path), Some(mgf_path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: preproc_audit <truth.json> <eval.mgf>");
        std::process::exit(2)
    };
    let truth: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(truth_path).unwrap()).unwrap();
    let mut parser = MgfParser::default();
    let records = parser.parse_file(mgf_path).expect("mgf");
    let wf = SiriusClusteringWorkflow::new(SiriusWorkflowConfig::default());

    let (mut spectra, mut before, mut after, mut lost_all) = (0usize, 0usize, 0usize, 0usize);
    for rec in &records {
        let Some(id) = rec.get_metadata("TITLE") else { continue };
        let Some(e) = truth.get(id.trim()) else { continue };
        let Some(f) = e.get("true_formula").and_then(|v| v.as_str()) else { continue };
        if !f.contains("Cl") {
            continue;
        }
        let raw: Vec<Peak> = rec.peaks().to_vec();
        let b = doublets(&raw);
        if b == 0 {
            continue;
        }
        let Some(pep) = rec.get_metadata("PEPMASS").and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok()) else { continue };
        let Some(clean) = wf.preprocess_peaks_for_tree(&raw, pep, 1) else { continue };
        let a = doublets(&clean);
        spectra += 1;
        before += b;
        after += a;
        if a == 0 {
            lost_all += 1;
        }
    }
    println!("Cl-containing MoNA spectra that carry a 35Cl/37Cl doublet: {spectra}");
    println!("  doublets before preprocessing : {before}");
    println!("  doublets after  preprocessing : {after}   ({:.0}% destroyed)",
             100.0 * (before - after) as f64 / before.max(1) as f64);
    println!("  spectra left with NO chlorine evidence at all: {lost_all}/{spectra}");
}
