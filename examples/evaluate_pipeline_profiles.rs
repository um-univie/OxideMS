//! Evaluate the application's exact profile implementations, with per-candidate diagnostics.
//! Usage: evaluate_pipeline_profiles INPUT.mgf OUTPUT.jsonl [LIMIT=200]
use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::{Adduct, Formula};
use oxidems_lib::sirius::workflow::{AnnotationProfile, SiriusClusteringWorkflow, SiriusWorkflowConfig};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

mod support;

const USAGE: &str = "usage: evaluate_pipeline_profiles <input.mgf> <output.jsonl> [limit=200]";

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input = args.get(1).unwrap_or_else(|| support::fail(USAGE));
    let output = args.get(2).unwrap_or_else(|| support::fail(USAGE));
    let limit: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let mut parser = MgfParser::default();
    let records = parser.parse_file(input).expect("read MGF");
    let mut selected: Vec<_> = records.iter().filter_map(|record| {
        let formula: Formula = record.get_metadata("TRUEFORMULA").or(record.get_metadata("FORMULA"))?.parse().ok()?;
        let mz: f64 = record.get_metadata("PEPMASS")?.split_whitespace().next()?.parse().ok()?;
        let name = record.get_metadata("NAME").or(record.get_metadata("TITLE")).unwrap_or("");
        Some((xxhash_rust::xxh32::xxh32(name.as_bytes(), 57), record, formula, mz))
    }).collect();
    selected.sort_by_key(|r| r.0);
    if limit > 0 { selected.truncate(limit); }
    let profiles = [AnnotationProfile::FastV1, AnnotationProfile::AccurateV1];
    let workflows: Vec<_> = profiles.iter().map(|&p| SiriusClusteringWorkflow::new(SiriusWorkflowConfig::for_profile(p))).collect();
    let results: Vec<_> = selected.par_iter().map(|(_, record, truth, mz)| {
        let negative = record.get_charge() < 0 || record.get_metadata("IONMODE").is_some_and(|s| s.eq_ignore_ascii_case("negative"));
        let adduct = record.get_metadata("ADDUCT").and_then(|s| s.parse::<Adduct>().ok())
            .unwrap_or_else(|| if negative { Adduct::deprotonated() } else { Adduct::protonated() });
        let ms1: Vec<_> = record.get_metadata("MS1PEAKS").unwrap_or("").split_whitespace().filter_map(|s| {
            let (mz, intensity) = s.split_once(':')?;
            Some(oxidems_lib::spectrum::Peak::new(mz.parse().ok()?, intensity.parse().ok()?))
        }).collect();
        let mut arms = Vec::new();
        for (profile, workflow) in profiles.iter().zip(&workflows) {
            let start = Instant::now();
            let peaks = workflow.preprocess_peaks_for_tree(record.peaks(), *mz, adduct.charge).unwrap_or_default();
            let candidates = workflow.annotate_peaks_with_adduct(&peaks, *mz, &ms1, &adduct);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let rank = candidates.iter().position(|c| c.formula == *truth).map(|i| i+1);
            let rows: Vec<_> = candidates.iter().map(|c| serde_json::json!({
                "formula": c.formula.to_string(), "correct": c.formula == *truth,
                "features": [c.total_score, c.tree_score, c.isotope_score, c.prior_score,
                    c.mass_error_ppm, c.explained_intensity as f64, c.explained_peaks as f64,
                    c.total_peaks as f64, c.tree.nodes.len() as f64, c.exact_mass,
                    c.formula.c() as f64, c.formula.n() as f64, c.formula.o() as f64,
                    c.formula.p() as f64, c.formula.s() as f64],
            })).collect();
            arms.push(serde_json::json!({"profile": profile.name(), "rank": rank,
                "scored": candidates.len(), "elapsed_ms": ms, "candidates": rows}));
        }
        serde_json::json!({"id": record.id, "name": record.get_metadata("NAME"),
            "smiles": record.get_metadata("SMILES"), "split": record.get_metadata("BENCHMARKSPLIT"),
            "true_formula": truth.to_string(), "true_mass": truth.exact_mass(), "adduct": adduct.name,
            "arms": arms})
    }).collect();
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output).unwrap());
    for result in &results { writeln!(writer, "{result}").unwrap(); }
    println!("wrote {} spectra to {output}; timings include candidate generation and tree search", results.len());
}
