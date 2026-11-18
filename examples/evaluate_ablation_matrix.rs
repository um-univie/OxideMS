//! Matrix Evaluation Tool for evaluating:
//! 1. Single Adduct + CHNOPS (Baseline)
//! 2. Single Adduct + Halogens (Cl, Br, F, I)
//! 3. Multi-Adduct + CHNOPS
//! 4. Multi-Adduct + Halogens (Full Strategy)
//!
//! Evaluated systematically across GNPS diverse standards (gnps_eval_100.mgf)
//! and Orbitrap metabolomics (test2.mgf).

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::adduct::Adduct;
use oxidems_lib::sirius::fingerprint::{FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::workflow::{AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset};
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

struct MatrixStats {
    name: &'static str,
    total_spectra: usize,
    matched_structures: usize,
    match_percentage: f64,
    avg_tanimoto: f32,
    avg_latency_ms: f64,
    wall_time_s: f64,
    throughput_spec_s: f64,
}

fn run_evaluation(
    name: &'static str,
    pos_wf: &SiriusClusteringWorkflow,
    neg_wf: &SiriusClusteringWorkflow,
    records: &[oxidems_lib::spectrum::MassSpectralRecord],
) -> MatrixStats {
    let t_start = Instant::now();

    let results: Vec<Option<(String, f32, f64)>> = records
        .par_iter()
        .map(|record| {
            let pepmass_str = record.get_metadata("PEPMASS")?;
            let pepmass: f64 = pepmass_str.split_whitespace().next()?.parse().ok()?;
            let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
            let is_neg = ionmode.eq_ignore_ascii_case("Negative");

            if record.peaks().len() < 3 {
                return None;
            }

            let t_spec = Instant::now();
            let wf = if is_neg { neg_wf } else { pos_wf };
            let ann = wf.annotate_spectrum(record, pepmass, None);
            let elapsed_ms = t_spec.elapsed().as_secs_f64() * 1000.0;

            if let Some(annotation) = ann {
                if let Some(top_cand) = annotation.top_candidate() {
                    if let Some(top_struct) = top_cand.structural_candidates.first() {
                        return Some((top_struct.name.clone(), top_struct.tanimoto_similarity, elapsed_ms));
                    }
                    return Some(("No DB Match".to_string(), 0.0, elapsed_ms));
                }
            }
            None
        })
        .collect();

    let wall_time_s = t_start.elapsed().as_secs_f64();
    let mut total_spectra = 0;
    let mut matched_structures = 0;
    let mut sum_tanimoto = 0.0f32;
    let mut sum_latency = 0.0f64;

    for r in results.into_iter().flatten() {
        total_spectra += 1;
        sum_latency += r.2;
        if r.0 != "No DB Match" {
            matched_structures += 1;
            sum_tanimoto += r.1;
        }
    }

    let match_percentage = (matched_structures as f64 / total_spectra.max(1) as f64) * 100.0;
    let avg_tanimoto = if matched_structures > 0 { sum_tanimoto / matched_structures as f32 } else { 0.0 };
    let avg_latency_ms = sum_latency / total_spectra.max(1) as f64;
    let throughput_spec_s = total_spectra as f64 / wall_time_s.max(0.001);

    MatrixStats {
        name,
        total_spectra,
        matched_structures,
        match_percentage,
        avg_tanimoto,
        avg_latency_ms,
        wall_time_s,
        throughput_spec_s,
    }
}

mod support;

fn main() {
    support::require("evaluate_ablation_matrix", &[("models/sirius_fingerprints.oxfp", support::MODEL), ("models/biometabolites.oxdb", support::STRUCTURE_DB), ("test_files/gnps_eval_100.mgf", support::GNPS_EVAL), ("test_files/test2.mgf", support::MSNLIB)]);
    println!("=========================================================================");
    println!("     OxideMS SIRIUS Feature Ablation & Multi-Adduct / Halogen Matrix     ");
    println!("=========================================================================");

    // 1. Load Fingerprint Model & Pre-Indexed Database
    let fp_model_path = "models/sirius_fingerprints.oxfp";
    let predictor = FingerprintPredictor::load_from_file(fp_model_path)
        .expect("Failed to load CSI:FingerID model weights");

    let bio_oxdb_path = "models/biometabolites.oxdb";
    let pubchem_oxdb_path = "models/pubchem_subset.oxdb";
    let mut db = StructureDatabase::load_from_file(bio_oxdb_path)
        .expect("Failed to load BioMetabolites .oxdb");

    if Path::new(pubchem_oxdb_path).exists() {
        let pc_db = StructureDatabase::load_from_file(pubchem_oxdb_path)
            .expect("Failed to load PubChem .oxdb");
        let added_structs = pc_db.total_structures;
        for (f, mut entries) in pc_db.entries_by_formula {
            db.entries_by_formula.entry(f).or_default().append(&mut entries);
        }
        db.total_structures += added_structs;
    }
    println!(
        "Active Database Pool: {} structures across {} unique formulas\n",
        db.total_structures,
        db.entries_by_formula.len()
    );

    // 2. Load Evaluation Datasets
    let mut parser = MgfParser::default();
    let gnps_records = parser.parse_file("test_files/gnps_eval_100.mgf").expect("Failed to parse GNPS MGF");
    let orbitrap_records: Vec<_> = parser
        .parse_file("test_files/test2.mgf")
        .expect("Failed to parse Orbitrap MGF")
        .into_iter()
        .take(50)
        .collect();

    // 3. Base Configuration
    let base_config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(15.0),
        top_k_candidates: 3,
        max_peaks_per_tree: 25,
        max_candidates_for_tree_search: 8,
        enable_fingerprint_prediction: true,
        enable_compound_classification: true,
        structure_database: Some(std::sync::Arc::new(db.clone())),
        structure_score_weight: 0.20,
        ..Default::default()
    };

    println!("=========================================================================================================================");
    println!("                                   GNPS Diverse Standards Matrix Evaluation (99 spectra)                                 ");
    println!("=========================================================================================================================");
    println!(
        "{:<38} | {:<12} | {:<10} | {:<14} | {:<11} | {:<14}",
        "Strategy Configuration", "Matched Rate", "Mean Tanim", "CPU Latency", "Batch Time", "Throughput"
    );
    println!("-------------------------------------------------------------------------------------------------------------------------");

    // [1] Baseline: Single Adduct [M+H]+, CHNOPS only
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct = Adduct::protonated();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::protonated());
        cfg_pos.enable_halogens = false;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct = Adduct::deprotonated();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[1] Single Adduct, CHNOPS (Baseline)", &wf_pos, &wf_neg, &gnps_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }

    // [2] Single Adduct + Halogens
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct = Adduct::protonated();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::protonated());
        cfg_pos.enable_halogens = true;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct = Adduct::deprotonated();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[2] Single Adduct + Halogens (Cl,Br)", &wf_pos, &wf_neg, &gnps_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }

    // [3] Multi-Adduct, CHNOPS only
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::AutoPositive;
        cfg_pos.enable_halogens = false;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::AutoNegative;

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[3] Multi-Adduct, CHNOPS", &wf_pos, &wf_neg, &gnps_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }

    // [4] Multi-Adduct + Halogens (Full)
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::AutoPositive;
        cfg_pos.enable_halogens = true;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::AutoNegative;

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[4] Multi-Adduct + Halogens (Full)", &wf_pos, &wf_neg, &gnps_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }
    println!("=========================================================================================================================\n");

    println!("=========================================================================================================================");
    println!("                                Orbitrap ID-X Real Metabolomics Matrix Evaluation (45 spectra)                            ");
    println!("=========================================================================================================================");
    println!(
        "{:<38} | {:<12} | {:<10} | {:<14} | {:<11} | {:<14}",
        "Strategy Configuration", "Matched Rate", "Mean Tanim", "CPU Latency", "Batch Time", "Throughput"
    );
    println!("-------------------------------------------------------------------------------------------------------------------------");

    // [1] Orbitrap Baseline
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct = Adduct::protonated();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::protonated());
        cfg_pos.enable_halogens = false;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct = Adduct::deprotonated();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[1] Single Adduct, CHNOPS (Baseline)", &wf_pos, &wf_neg, &orbitrap_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }

    // [4] Orbitrap Multi-Adduct + Halogens
    {
        let mut cfg_pos = base_config.clone();
        cfg_pos.adduct_search_strategy = AdductSearchStrategy::AutoPositive;
        cfg_pos.enable_halogens = true;

        let mut cfg_neg = cfg_pos.clone();
        cfg_neg.adduct_search_strategy = AdductSearchStrategy::AutoNegative;

        let wf_pos = SiriusClusteringWorkflow::new(cfg_pos).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());
        let wf_neg = SiriusClusteringWorkflow::new(cfg_neg).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

        let s = run_evaluation("[4] Multi-Adduct + Halogens (Full)", &wf_pos, &wf_neg, &orbitrap_records);
        println!(
            "{:<38} | {:>2}/{:<2} ({:>5.1}%) | {:<10.3} | {:>8.2} ms/sp | {:>7.2} s   | {:>6.1} spec/s",
            s.name, s.matched_structures, s.total_spectra, s.match_percentage, s.avg_tanimoto, s.avg_latency_ms, s.wall_time_s, s.throughput_spec_s
        );
    }
    println!("=========================================================================================================================");
}
