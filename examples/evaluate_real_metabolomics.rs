//! Benchmark tool for evaluating OxideMS SIRIUS + CSI:FingerID on real Orbitrap metabolomics data.
//!
//! Evaluates authentic high-resolution tandem MS/MS spectra from GNPS / Thermo Orbitrap ID-X dataset
//! against the 632k+ BioMetabolites and PubChem database.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::adduct::Adduct;
use oxidems_lib::sirius::fingerprint::{FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::workflow::{AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset};
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

struct RealMetabolomicsResult {
    id: usize,
    dataset_name: String,
    precursor_mz: f64,
    predicted_formula: String,
    true_formula: Option<String>,
    score: f64,
    top_structure_match: String,
    canopus_class: String,
    tanimoto: f32,
    latency_ms: f64,
}

mod support;

fn main() {
    support::require("evaluate_real_metabolomics", &[("models/sirius_fingerprints.oxfp", support::MODEL), ("models/biometabolites.oxdb", support::STRUCTURE_DB), ("test_files/test2.mgf", support::MSNLIB), ("test_files/gnps_eval_100.mgf", support::GNPS_EVAL)]);
    println!("=========================================================================");
    println!("    OxideMS SIRIUS + CSI:FingerID Real Orbitrap Metabolomics Benchmark   ");
    println!("           Dataset: GNPS MSV000094528 (Thermo Orbitrap ID-X)             ");
    println!("=========================================================================");

    // 1. Load Neural CSI:FingerID Predictor
    let fp_model_path = "models/sirius_fingerprints.oxfp";
    println!("\n[1/3] Loading CSI:FingerID Model ({}) ...", fp_model_path);
    let t_fp = Instant::now();
    let predictor = FingerprintPredictor::load_from_file(fp_model_path)
        .expect("Failed to load CSI:FingerID model weights");
    println!(
        "  CSI:FingerID Model loaded in {:.2} ms ({} neural parameters)",
        t_fp.elapsed().as_secs_f64() * 1000.0,
        predictor.num_parameters()
    );

    // 2. Load Pre-Indexed Databases (BioMetabolites: MoNA + COCONUT + HMDB)
    let bio_oxdb_path = "models/biometabolites.oxdb";
    let pubchem_oxdb_path = "models/pubchem_subset.oxdb";
    println!("\n[2/3] Loading 632k+ Candidate Structure Databases (.oxdb)...");

    let t_load = Instant::now();
    let mut db = StructureDatabase::load_from_file(bio_oxdb_path)
        .expect("Failed to load BioMetabolites .oxdb");
    println!(
        "  [BioMetabolites] Loaded {} structures across {} formulas in {:.2} ms",
        db.total_structures,
        db.entries_by_formula.len(),
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    if Path::new(pubchem_oxdb_path).exists() {
        let t_pc = Instant::now();
        let pc_db = StructureDatabase::load_from_file(pubchem_oxdb_path)
            .expect("Failed to load PubChem .oxdb");
        let added_structs = pc_db.total_structures;
        let added_formulas = pc_db.entries_by_formula.len();
        for (f, mut entries) in pc_db.entries_by_formula {
            db.entries_by_formula.entry(f).or_default().append(&mut entries);
        }
        db.total_structures += added_structs;
        println!(
            "  [PubChem Subset] Merged {} structures across {} formulas in {:.2} ms",
            added_structs,
            added_formulas,
            t_pc.elapsed().as_secs_f64() * 1000.0
        );
    }
    println!(
        "  Total Active Search Pool: {} candidate structures across {} unique formulas",
        db.total_structures,
        db.entries_by_formula.len()
    );

    // 3. Configure Fast Pass ([M+H]+ / [M-H]-) and Multi-Adduct Fallback Workflows
    let pos_fast_config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(15.0),
        top_k_candidates: 5,
        max_peaks_per_tree: 30,
        max_candidates_for_tree_search: 12,
        enable_fingerprint_prediction: true,
        enable_compound_classification: true,
        structure_database: Some(std::sync::Arc::new(db.clone())),
        structure_score_weight: 0.20,
        adduct: Adduct::protonated(),
        adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
        ..Default::default()
    };

    let mut neg_fast_config = pos_fast_config.clone();
    neg_fast_config.adduct = Adduct::deprotonated();
    neg_fast_config.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

    let mut pos_auto_config = pos_fast_config.clone();
    pos_auto_config.adduct_search_strategy = AdductSearchStrategy::AutoPositive;

    let mut neg_auto_config = neg_fast_config.clone();
    neg_auto_config.adduct_search_strategy = AdductSearchStrategy::AutoNegative;

    let pos_fast_wf = SiriusClusteringWorkflow::new(pos_fast_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    let neg_fast_wf = SiriusClusteringWorkflow::new(neg_fast_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    let pos_auto_wf = SiriusClusteringWorkflow::new(pos_auto_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    let neg_auto_wf = SiriusClusteringWorkflow::new(neg_auto_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone())
        .with_compound_classification(true);

    // 4. Ingest and Stream Authentic Orbitrap Metabolomics Spectra
    let real_mgf = "test_files/test2.mgf";
    println!("\n[3/3] Streaming Authentic Orbitrap Spectra from {}...", real_mgf);
    let mut parser = MgfParser::default();
    let records = parser.parse_file(real_mgf).expect("Failed to parse real MGF");
    let target_records: Vec<_> = records.into_iter().take(50).collect();

    println!("  Processing {} real Orbitrap metabolomics spectra with parallel Rayon...", target_records.len());
    let t_eval_start = Instant::now();

    let results: Vec<Option<RealMetabolomicsResult>> = target_records
        .into_par_iter()
        .enumerate()
        .map(|(idx, record)| {
            let name = record.get_metadata("NAME").unwrap_or("Unknown").to_string();
            let true_formula = record.get_metadata("FORMULA").map(|s| s.to_string());
            let pepmass_str = record.get_metadata("PEPMASS")?;
            let pepmass: f64 = pepmass_str.split_whitespace().next()?.parse().ok()?;

            let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
            let is_neg = ionmode.eq_ignore_ascii_case("Negative");

            if record.peaks().len() < 3 {
                return None;
            }

            let t_start = Instant::now();
            let (fast_wf, auto_wf) = if is_neg { (&neg_fast_wf, &neg_auto_wf) } else { (&pos_fast_wf, &pos_auto_wf) };

            // Two-Tier Database Guided Fallback
            let annotation = match fast_wf.annotate_spectrum(&record, pepmass, None) {
                Some(ann) if ann.top_candidate().map(|c| !c.structural_candidates.is_empty()).unwrap_or(false) => Some(ann),
                _ => auto_wf.annotate_spectrum(&record, pepmass, None),
            };
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

                    return Some(RealMetabolomicsResult {
                        id: idx + 1,
                        dataset_name: name,
                        precursor_mz: pepmass,
                        predicted_formula: formula_str,
                        true_formula,
                        score,
                        top_structure_match: struct_name,
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

    // 5. Present Orbitrap Identification Results
    println!("\n---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<3} | {:<14} | {:<8} | {:<12} | {:<12} | {:<6} | {:<24} | {:<22} | {:<6}",
        "#", "Sample ID", "Prec m/z", "Pred Formula", "True Formula", "Score", "Top Structure Match", "CANOPUS Class", "Tanimoto"
    );
    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");

    let mut annotated_count = 0;
    let mut formula_correct_count = 0;
    let mut structure_matched_count = 0;
    let mut sum_latency_ms = 0.0;

    for res_opt in results {
        if let Some(res) = res_opt {
            annotated_count += 1;
            let is_formula_match = res.true_formula.as_deref() == Some(&res.predicted_formula);
            if is_formula_match {
                formula_correct_count += 1;
            }
            if res.top_structure_match != "No DB Match" {
                structure_matched_count += 1;
            }
            sum_latency_ms += res.latency_ms;

            let short_id: String = res.dataset_name.chars().take(14).collect();
            let short_true: String = res.true_formula.unwrap_or_else(|| "-".to_string()).chars().take(12).collect();
            let short_struct: String = res.top_structure_match.chars().take(24).collect();
            let short_class: String = res.canopus_class.chars().take(22).collect();

            println!(
                "{:<3} | {:<14} | {:<8.4} | {:<12} | {:<12} | {:<6.1} | {:<24} | {:<22} | {:<6.3}",
                annotated_count,
                short_id,
                res.precursor_mz,
                res.predicted_formula,
                short_true,
                res.score,
                short_struct,
                short_class,
                res.tanimoto
            );
        }
    }

    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("                                          Orbitrap Real Metabolomics Summary                                                                       ");
    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("  Total Spectra Processed:             {}", annotated_count);
    println!("  Formula Identification Accuracy:     {}/{} ({:.1}%)", formula_correct_count, annotated_count, (formula_correct_count as f64 / annotated_count.max(1) as f64) * 100.0);
    println!("  Structures Matched in 632k DB:       {}/{} ({:.1}%)", structure_matched_count, annotated_count, (structure_matched_count as f64 / annotated_count.max(1) as f64) * 100.0);
    println!("  Average Spectrum Latency (CPU time): {:.2} ms / spectrum", sum_latency_ms / annotated_count.max(1) as f64);
    println!("  Total Wall-Clock Processing Time:    {:.2} s ({:.1} spectra / sec)", total_wall_time, annotated_count as f64 / total_wall_time);
    println!("=========================================================================");

    // 6. Broad GNPS Natural Products & Diverse Metabolites Benchmark
    let gnps_mgf = "test_files/gnps_eval_100.mgf";
    println!("\nEvaluating Diverse Natural Products & Reference Standards from {} (Two-Tier Database Fallback)...", gnps_mgf);
    let gnps_records = parser.parse_file(gnps_mgf).expect("Failed to parse GNPS MGF");
    let t_gnps_start = Instant::now();

    let gnps_results: Vec<Option<RealMetabolomicsResult>> = gnps_records
        .into_par_iter()
        .enumerate()
        .map(|(idx, record)| {
            let name = record.get_metadata("NAME").unwrap_or("Unknown").to_string();
            let true_formula = record.get_metadata("FORMULA").map(|s| s.to_string());
            let pepmass_str = record.get_metadata("PEPMASS")?;
            let pepmass: f64 = pepmass_str.split_whitespace().next()?.parse().ok()?;

            let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
            let is_neg = ionmode.eq_ignore_ascii_case("Negative");

            if record.peaks().len() < 3 {
                return None;
            }

            let t_start = Instant::now();
            let (fast_wf, auto_wf) = if is_neg { (&neg_fast_wf, &neg_auto_wf) } else { (&pos_fast_wf, &pos_auto_wf) };

            // Two-Tier Database Guided Fallback: Fast [M+H]+ first, Auto Multi-Adduct if No DB Match
            let annotation = match fast_wf.annotate_spectrum(&record, pepmass, None) {
                Some(ann) if ann.top_candidate().map(|c| !c.structural_candidates.is_empty()).unwrap_or(false) => Some(ann),
                _ => auto_wf.annotate_spectrum(&record, pepmass, None),
            };
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

                    return Some(RealMetabolomicsResult {
                        id: idx + 1,
                        dataset_name: name,
                        precursor_mz: pepmass,
                        predicted_formula: formula_str,
                        true_formula,
                        score,
                        top_structure_match: struct_name,
                        canopus_class: class_str,
                        tanimoto,
                        latency_ms: elapsed_ms,
                    });
                }
            }
            None
        })
        .collect();

    let gnps_wall_time = t_gnps_start.elapsed().as_secs_f64();
    let mut gnps_count = 0;
    let mut gnps_struct_count = 0;
    let mut gnps_sum_ms = 0.0;

    for r in gnps_results.into_iter().flatten() {
        gnps_count += 1;
        if r.top_structure_match != "No DB Match" {
            gnps_struct_count += 1;
        }
        gnps_sum_ms += r.latency_ms;
    }

    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("                                          GNPS Diverse Standards Evaluation Summary                                                                ");
    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("  Total Spectra Processed:             {}", gnps_count);
    println!("  Structures Matched in 632k DB:       {}/{} ({:.1}%)", gnps_struct_count, gnps_count, (gnps_struct_count as f64 / gnps_count.max(1) as f64) * 100.0);
    println!("  Average Spectrum Latency (CPU time): {:.2} ms / spectrum", gnps_sum_ms / gnps_count.max(1) as f64);
    println!("  Total Wall-Clock Processing Time:    {:.2} s ({:.1} spectra / sec)", gnps_wall_time, gnps_count as f64 / gnps_wall_time);
    println!("=========================================================================");
}
