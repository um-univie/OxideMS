//! End-to-end SIRIUS + CSI:FingerID + CANOPUS Pipeline Benchmark on GNPS Spectra.
//!
//! Evaluates the integrated pipeline on authentic small-molecule natural product and drug
//! reference spectra from GNPS:
//! 1. MAP Fragmentation Tree & Molecular Formula Optimization
//! 2. 2,214-bit CSI:FingerID Molecular Fingerprint Prediction
//! 3. CANOPUS Hierarchical ClassyFire Chemical Taxonomy Classification (Kingdom -> Superclass -> Class -> Subclass)

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::canopus::CanopusClassifier;
use oxidems_lib::sirius::fingerprint::FingerprintPredictor;
use oxidems_lib::sirius::workflow::{
    AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset,
};
use oxidems_lib::sirius::Adduct;
use oxidems_lib::workflow::MassTolerance;
use std::time::Instant;

mod support;

fn main() {
    support::require("gnps_sirius_pipeline", &[("models/sirius_fingerprints.oxfp", support::MODEL), ("test_files/gnps_eval_100.mgf", support::GNPS_EVAL)]);
    println!("=========================================================================");
    println!("       OxideMS SIRIUS + CSI:FingerID + CANOPUS Pipeline on GNPS          ");
    println!("=========================================================================");

    let model_path = "models/sirius_fingerprints.oxfp";
    println!("\nLoading CSI:FingerID Model Weights from: {}...", model_path);
    let start_load = Instant::now();
    let predictor = FingerprintPredictor::load_from_file(model_path)
        .expect("Failed to load models/sirius_fingerprints.oxfp. Run training first!");
    println!(
        "  Loaded neural model with {} parameters in {:.2}ms",
        predictor.num_parameters(),
        start_load.elapsed().as_secs_f64() * 1000.0
    );

    let gnps_mgf = "test_files/gnps_eval_100.mgf";
    println!("\nIngesting GNPS Evaluation Spectra: {}...", gnps_mgf);
    let mut parser = MgfParser::default();
    let records = parser.parse_file(gnps_mgf).expect("Failed to parse GNPS MGF file");
    println!("  Parsed {} GNPS reference spectra", records.len());

    // Configure the integrated Sirius pipeline with fingerprint prediction and CANOPUS classification
    let mut workflow_config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(15.0),
        top_k_candidates: 3,
        max_peaks_per_tree: 25,
        max_candidates_for_tree_search: 12,
        enable_fingerprint_prediction: true,
        enable_compound_classification: true,
        filter_peaks_above_precursor: true,
        adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
        ..Default::default()
    };

    let workflow = SiriusClusteringWorkflow::new(workflow_config.clone())
        .with_fingerprint_predictor(predictor)
        .with_compound_classification(true);

    println!("\n-------------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<3} | {:<24} | {:<9} | {:<12} | {:<6} | {:<28} | {:<22} | {:<6}",
        "#", "Compound Name", "Prec m/z", "Top Formula", "Score", "ClassyFire Superclass", "ClassyFire Class", "Prob"
    );
    println!("-------------------------------------------------------------------------------------------------------------------------");

    let mut annotated_count = 0;
    let mut total_pipeline_time_ms = 0.0;

    for (idx, record) in records.iter().enumerate() {
        let name = record.get_metadata("NAME").unwrap_or("Unknown");
        let pepmass_str = match record.get_metadata("PEPMASS") {
            Some(s) => s,
            None => continue,
        };
        let pepmass: f64 = match pepmass_str.split_whitespace().next().unwrap_or("0").parse() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
        let adduct = if ionmode.eq_ignore_ascii_case("Negative") {
            Adduct::deprotonated()
        } else {
            Adduct::protonated()
        };

        if record.peaks().len() < 3 {
            continue;
        }

        let t_start = Instant::now();
        let annotation = workflow.annotate_spectrum(record, pepmass, None);
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        total_pipeline_time_ms += elapsed_ms;

        if let Some(ann) = annotation {
            if let Some(top_cand) = ann.top_candidate() {
                annotated_count += 1;

                let formula_str = top_cand.formula.to_string();
                let score = top_cand.total_score;

                let (superclass_str, class_str, prob) = if let Some(ref classif) = top_cand.classification {
                    let sc = classif
                        .superclass
                        .as_ref()
                        .map(|s| s.name.clone())
                        .unwrap_or_else(|| "Unclassified".to_string());
                    let cl = classif
                        .class
                        .as_ref()
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| "Unclassified".to_string());
                    let p = classif
                        .direct_parent
                        .as_ref()
                        .map(|d| d.probability)
                        .unwrap_or(0.0);
                    (sc, cl, p)
                } else {
                    ("N/A".to_string(), "N/A".to_string(), 0.0)
                };

                let short_name: String = name.chars().take(24).collect();
                let short_sc: String = superclass_str.chars().take(28).collect();
                let short_cl: String = class_str.chars().take(22).collect();

                println!(
                    "{:<3} | {:<24} | {:<9.4} | {:<12} | {:<6.1} | {:<28} | {:<22} | {:<5.1}%",
                    annotated_count,
                    short_name,
                    pepmass,
                    formula_str,
                    score,
                    short_sc,
                    short_cl,
                    prob * 100.0
                );
            }
        }
    }

    println!("-------------------------------------------------------------------------------------------------------------------------");
    println!("                                         Pipeline Performance Summary                                                    ");
    println!("-------------------------------------------------------------------------------------------------------------------------");
    println!("  Total Spectra Processed:               {}", records.len());
    println!("  Successfully Annotated:                {} ({:.1}%)", annotated_count, (annotated_count as f64 / records.len() as f64) * 100.0);
    println!("  Avg. Pipeline Latency per Spectrum:    {:.2} ms (Tree MAP + Fingerprint NN + CANOPUS)", total_pipeline_time_ms / annotated_count.max(1) as f64);
    println!("  Pipeline Throughput:                   {:.1} spectra/sec", 1000.0 / (total_pipeline_time_ms / annotated_count.max(1) as f64));
    println!("=========================================================================");
}
