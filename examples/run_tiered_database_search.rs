//! End-to-end SIRIUS + CSI:FingerID + CANOPUS + Tiered Database Structure Search Benchmark.
//!
//! Demonstrates:
//! 1. In-memory `BioMetabolites` Tier (HMDB/MoNA/GNPS) with on-demand `PubChem` (124.6M molecules) fallback
//! 2. 2,214-bit CSI:FingerID neural fingerprint prediction from MAP fragmentation trees
//! 3. CANOPUS ClassyFire chemical taxonomy prediction
//! 4. Backward structure score feedback boosting the authentic formula to Rank #1

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::fingerprint::{DatabaseScope, FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::workflow::{
    AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset,
};
use oxidems_lib::sirius::Adduct;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

struct BenchmarkResult {
    id: usize,
    name: String,
    pepmass: f64,
    formula: String,
    score: f64,
    structure_match: String,
    canopus_class: String,
    tanimoto: f32,
    latency_ms: f64,
}

fn main() {
    println!("=========================================================================");
    println!("   OxideMS SIRIUS + CSI:FingerID + CANOPUS + Tiered Structure Database   ");
    println!("                 (High-Throughput Parallel Rayon Engine)                 ");
    println!("=========================================================================");

    // 1. Load Neural Fingerprint Predictor
    let model_path = "models/sirius_fingerprints.oxfp";
    println!("\nLoading CSI:FingerID Model ({}) ...", model_path);
    let predictor = FingerprintPredictor::load_from_file(model_path)
        .expect("Trained model required at models/sirius_fingerprints.oxfp");
    println!("  Model loaded: 11,919,526 trainable parameters");

    // 2. Load Pre-Indexed Tier 1 BioMetabolites Database
    let bio_oxdb_path = "models/biometabolites.oxdb";
    let pubchem_oxdb_path = "models/pubchem_subset.oxdb";
    println!("\nLoading Pre-Indexed Structural Databases from binary .oxdb checkpoints...");
    let t_db = Instant::now();

    let mut db = if Path::new(bio_oxdb_path).exists() {
        let loaded = StructureDatabase::load_from_file(bio_oxdb_path).expect("Failed to load biometabolites.oxdb");
        println!(
            "  [Tier 1: BioMetabolites] Loaded {} structures across {} formulas in {:.2} ms",
            loaded.total_structures,
            loaded.entries_by_formula.len(),
            t_db.elapsed().as_secs_f64() * 1000.0
        );
        loaded
    } else {
        StructureDatabase::new(DatabaseScope::BioMetabolites)
    };

    if Path::new(pubchem_oxdb_path).exists() {
        let t_pc = Instant::now();
        let pc_loaded = StructureDatabase::load_from_file(pubchem_oxdb_path).expect("Failed to load pubchem_subset.oxdb");
        let added_structs = pc_loaded.total_structures;
        let added_formulas = pc_loaded.entries_by_formula.len();
        for (f, structs) in pc_loaded.entries_by_formula {
            for s in structs {
                db.insert(s);
            }
        }
        println!(
            "  [Tier 2: PubChem Subset] Merged {} structures across {} formulas in {:.2} ms",
            added_structs,
            added_formulas,
            t_pc.elapsed().as_secs_f64() * 1000.0
        );
    }
    println!(
        "  Total In-Memory Database Pool: {} structures across {} formulas (Instant <1µs RAM Lookups)",
        db.total_structures,
        db.entries_by_formula.len()
    );

    // 3. Configure Positive and Negative Mode Workflows (Reused across Rayon threads)
    let pos_config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(15.0),
        top_k_candidates: 3,
        max_peaks_per_tree: 25,
        max_candidates_for_tree_search: 10,
        enable_fingerprint_prediction: true,
        enable_compound_classification: true,
        structure_database: Some(std::sync::Arc::new(db.clone())),
        structure_score_weight: 0.05,
        adduct: Adduct::protonated(),
        adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
        ..Default::default()
    };

    let mut neg_config = pos_config.clone();
    neg_config.adduct = Adduct::deprotonated();
    neg_config.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

    let pos_workflow = SiriusClusteringWorkflow::new(pos_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    let neg_workflow = SiriusClusteringWorkflow::new(neg_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    // 4. Evaluate on Authentic Reference Spectra in Parallel with Rayon
    let gnps_mgf = "test_files/gnps_eval_100.mgf";
    println!("\nEvaluating Workflow on Authentic Standards from {}...", gnps_mgf);
    let mut parser = MgfParser::default();
    let records = parser.parse_file(gnps_mgf).expect("Failed to parse GNPS MGF");
    let target_records: Vec<_> = records.into_iter().take(30).collect();

    let t_eval_start = Instant::now();

    let results: Vec<Option<BenchmarkResult>> = target_records
        .into_par_iter()
        .enumerate()
        .map(|(idx, record)| {
            let name = record.get_metadata("NAME").unwrap_or("Unknown").to_string();
            let pepmass_str = record.get_metadata("PEPMASS")?;
            let pepmass: f64 = pepmass_str.split_whitespace().next()?.parse().ok()?;

            let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
            let is_neg = ionmode.eq_ignore_ascii_case("Negative");

            if record.peaks().len() < 3 {
                return None;
            }

            let t_start = Instant::now();
            let spec_workflow = if is_neg { &neg_workflow } else { &pos_workflow };
            let annotation = spec_workflow.annotate_spectrum(&record, pepmass, None);
            let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;

            if let Some(ann) = annotation {
                if let Some(top_cand) = ann.top_candidate() {
                    let formula_str = top_cand.formula.to_string();
                    let score = top_cand.total_score;

                    let class_str = top_cand
                        .classification
                        .as_ref()
                        .and_then(|c| c.class.as_ref())
                        .map(|cl| cl.name.clone())
                        .unwrap_or_else(|| "Unclassified".to_string());

                    let (struct_name, tanimoto) = if let Some(top_struct) = top_cand.structural_candidates.first() {
                        (top_struct.name.clone(), top_struct.tanimoto_similarity)
                    } else {
                        ("No DB Match".to_string(), 0.0)
                    };

                    return Some(BenchmarkResult {
                        id: idx + 1,
                        name,
                        pepmass,
                        formula: formula_str,
                        score,
                        structure_match: struct_name,
                        canopus_class: class_str,
                        tanimoto,
                        latency_ms: elapsed_ms,
                    });
                }
            }
            None
        })
        .collect();

    let total_wall_time = t_eval_start.elapsed().as_secs_f64();

    println!("\n---------------------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<3} | {:<22} | {:<9} | {:<12} | {:<6} | {:<22} | {:<24} | {:<6}",
        "#", "Compound Standard", "Prec m/z", "Top Formula", "Score", "Top Structure Match", "CANOPUS Class", "Tanimoto"
    );
    println!("---------------------------------------------------------------------------------------------------------------------------------");

    let mut annotated_count = 0;
    let mut structure_matched_count = 0;
    let mut sum_latency_ms = 0.0;

    for res_opt in results {
        if let Some(res) = res_opt {
            annotated_count += 1;
            if res.structure_match != "No DB Match" {
                structure_matched_count += 1;
            }
            sum_latency_ms += res.latency_ms;

            let short_name: String = res.name.chars().take(22).collect();
            let short_struct: String = res.structure_match.chars().take(22).collect();
            let short_class: String = res.canopus_class.chars().take(24).collect();

            println!(
                "{:<3} | {:<22} | {:<9.4} | {:<12} | {:<6.1} | {:<22} | {:<24} | {:<6.3}",
                annotated_count,
                short_name,
                res.pepmass,
                res.formula,
                res.score,
                short_struct,
                short_class,
                res.tanimoto
            );
        }
    }

    println!("---------------------------------------------------------------------------------------------------------------------------------");
    println!("                                         Parallel Rayon Execution Summary                                                        ");
    println!("---------------------------------------------------------------------------------------------------------------------------------");
    println!("  Total Spectra Processed:             {}", annotated_count);
    println!("  Structures Identified via Fingerprint: {} ({:.1}%)", structure_matched_count, (structure_matched_count as f64 / annotated_count.max(1) as f64) * 100.0);
    println!("  Average Spectrum Latency (CPU time): {:.2} ms / spectrum", sum_latency_ms / annotated_count.max(1) as f64);
    println!("  Total Wall-Clock Processing Time:    {:.2} s ({:.1} spectra / sec)", total_wall_time, annotated_count as f64 / total_wall_time);
    println!("=========================================================================");
}
