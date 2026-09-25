use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::adduct::Adduct;
use oxidems_lib::sirius::fingerprint::{FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::workflow::{AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset};
use oxidems_lib::workflow::MassTolerance;
use std::path::Path;

fn main() {
    let fp_model_path = "models/sirius_fingerprints.oxfp";
    let predictor = FingerprintPredictor::load_from_file(fp_model_path)
        .expect("Failed to load CSI:FingerID model weights");

    let bio_oxdb_path = "models/biometabolites.oxdb";
    let pubchem_oxdb_path = "models/pubchem_subset.oxdb";
    let mut db = StructureDatabase::load_from_file(bio_oxdb_path)
        .expect("Failed to load BioMetabolites .oxdb");

    if Path::new(pubchem_oxdb_path).exists() {
        let pc_db = StructureDatabase::load_from_file(pubchem_oxdb_path).unwrap();
        let added_structs = pc_db.total_structures;
        for (f, mut entries) in pc_db.entries_by_formula {
            db.entries_by_formula.entry(f).or_default().append(&mut entries);
        }
        db.total_structures += added_structs;
    }

    let mut parser = MgfParser::default();
    let gnps_records = parser.parse_file("test_files/gnps_eval_100.mgf").expect("Failed to parse GNPS MGF");

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

    // WF1: CHNOPS Single Adduct
    let mut cfg1 = base_config.clone();
    cfg1.adduct = Adduct::protonated();
    cfg1.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::protonated());
    cfg1.enable_halogens = false;
    let wf1 = SiriusClusteringWorkflow::new(cfg1).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

    // WF2: Halogens Single Adduct
    let mut cfg2 = base_config.clone();
    cfg2.adduct = Adduct::protonated();
    cfg2.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::protonated());
    cfg2.enable_halogens = true;
    let wf2 = SiriusClusteringWorkflow::new(cfg2).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

    // WF3: CHNOPS Multi Adduct
    let mut cfg3 = base_config.clone();
    cfg3.adduct_search_strategy = AdductSearchStrategy::AutoPositive;
    cfg3.enable_halogens = false;
    let wf3 = SiriusClusteringWorkflow::new(cfg3).with_fingerprint_predictor(predictor.clone()).with_structure_database(db.clone());

    println!("{:<3} | {:<25} | {:<8} | {:<12} | {:<12} | {:<12} | {:<6} | {:<6}",
        "#", "Name", "m/z", "Formula (CHNOPS)", "Formula (Hal)", "Formula (Multi)", "Tan(CH)", "Tan(Hal)");
    println!("{:-<100}", "");

    for (idx, record) in gnps_records.iter().enumerate().take(30) {
        let name = record.get_metadata("NAME").unwrap_or("Unknown").to_string();
        let pepmass_str = match record.get_metadata("PEPMASS") {
            Some(s) => s,
            None => continue,
        };
        let pepmass: f64 = match pepmass_str.split_whitespace().next().and_then(|s| s.parse().ok()) {
            Some(m) => m,
            None => continue,
        };

        let ann1 = wf1.annotate_spectrum(record, pepmass, None);
        let ann2 = wf2.annotate_spectrum(record, pepmass, None);
        let ann3 = wf3.annotate_spectrum(record, pepmass, None);

        let f1 = ann1.as_ref().and_then(|a| a.top_candidate()).map(|c| c.formula.to_string()).unwrap_or("-".to_string());
        let f2 = ann2.as_ref().and_then(|a| a.top_candidate()).map(|c| c.formula.to_string()).unwrap_or("-".to_string());
        let f3 = ann3.as_ref().and_then(|a| a.top_candidate()).map(|c| c.formula.to_string()).unwrap_or("-".to_string());

        let tan1 = ann1.as_ref().and_then(|a| a.top_candidate()).and_then(|c| c.structural_candidates.first()).map(|s| s.tanimoto_similarity).unwrap_or(0.0);
        let tan2 = ann2.as_ref().and_then(|a| a.top_candidate()).and_then(|c| c.structural_candidates.first()).map(|s| s.tanimoto_similarity).unwrap_or(0.0);

        let short_name: String = name.chars().take(25).collect();
        println!("{:<3} | {:<25} | {:<8.3} | {:<12} | {:<12} | {:<12} | {:<6.3} | {:<6.3}",
            idx + 1, short_name, pepmass, f1, f2, f3, tan1, tan2);
    }
}
