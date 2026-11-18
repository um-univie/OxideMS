//! Experiment evaluating the impact of MS1 Isotopic Envelope on Formula & Structure Identification
//! Compares MS2-only vs MS2 + MS1 Isotope Pattern across GNPS Diverse Standards.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::adduct::Adduct;
use oxidems_lib::sirius::fingerprint::{FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::formula::Formula;
use oxidems_lib::sirius::scoring::calculate_isotope_distribution;
use oxidems_lib::sirius::workflow::{AdductSearchStrategy, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

mod support;

fn main() {
    support::require("evaluate_with_ms1_isotopes", &[("models/sirius_fingerprints.oxfp", support::MODEL), ("models/biometabolites.oxdb", support::STRUCTURE_DB), ("test_files/gnps_eval_100.mgf", support::GNPS_EVAL)]);
    println!("=========================================================================");
    println!("     OxideMS SIRIUS Evaluation: MS2-Only vs MS1 Isotope Rescue Pattern   ");
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
        let pc_db = StructureDatabase::load_from_file(pubchem_oxdb_path).unwrap();
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

    // 2. Known Ground Truth Reference Map for GNPS Standards
    let mut ground_truth: HashMap<&str, (&str, &str)> = HashMap::new();
    ground_truth.insert("Pacificanone A", ("C17H22O4", "Positive"));
    ground_truth.insert("Desferrioxamine E", ("C27H48N6O9", "Positive"));
    ground_truth.insert("Staurosporine", ("C28H26N4O3", "Positive"));
    ground_truth.insert("Lymphostin", ("C16H14N4O4", "Positive"));
    ground_truth.insert("Neodysidenin", ("C17H23Cl6NO2", "Positive"));
    ground_truth.insert("epi-Tumonoic acid D", ("C19H25NO", "Positive"));
    ground_truth.insert("Citrinin", ("C13H14O5", "Positive"));
    ground_truth.insert("Proximicin B", ("C22H23N7O5", "Positive"));
    ground_truth.insert("cyclo(L-Leu-L-Pro)", ("C11H18N2O2", "Positive"));
    ground_truth.insert("pateamine", ("C31H45N3O4S", "Positive"));
    ground_truth.insert("cyclo(Phe-Leu)", ("C15H20N2O2", "Positive"));
    ground_truth.insert("cyclo(L-Trp-L-Pro)", ("C16H17N3O2", "Positive"));
    ground_truth.insert("cyclo(L-Val-L-Pro)", ("C10H16N2O2", "Positive"));
    ground_truth.insert("Aerucyclamide B", ("C31H39N7O4S2", "Positive"));
    ground_truth.insert("Aerucyclamide C", ("C32H41N7O4S2", "Positive"));
    ground_truth.insert("Quercetin", ("C15H10O7", "Negative"));
    ground_truth.insert("Acacetin", ("C16H12O5", "Negative"));
    ground_truth.insert("2O-rhamnosyl-swertisin", ("C27H30O14", "Negative"));
    ground_truth.insert("3-O-methylquercetin", ("C16H12O7", "Negative"));
    ground_truth.insert("Apigetrin", ("C21H20O10", "Negative"));
    ground_truth.insert("Avicularin", ("C20H18O11", "Negative"));
    ground_truth.insert("Baicalein", ("C15H10O5", "Negative"));
    ground_truth.insert("Baicalin", ("C21H18O11", "Negative"));
    ground_truth.insert("Formononetin", ("C16H12O4", "Negative"));
    ground_truth.insert("Genistein", ("C15H10O5", "Negative"));
    ground_truth.insert("Hesperetin", ("C16H14O6", "Negative"));
    ground_truth.insert("Isovitexin", ("C21H20O10", "Negative"));
    ground_truth.insert("Kaempferol", ("C15H10O6", "Negative"));
    ground_truth.insert("Luteolin", ("C15H10O6", "Negative"));
    ground_truth.insert("Naringenin", ("C15H12O5", "Negative"));

    // 3. Configure Sirius Workflow
    let pos_config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(20.0),
        top_k_candidates: 5,
        max_peaks_per_tree: 25,
        max_candidates_for_tree_search: 8,
        enable_fingerprint_prediction: true,
        enable_compound_classification: true,
        structure_database: Some(std::sync::Arc::new(db.clone())),
        structure_score_weight: 0.20,
        adduct: Adduct::protonated(),
        adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
        isotope_weight: 2.5,
        ..Default::default()
    };

    let mut neg_config = pos_config.clone();
    neg_config.adduct = Adduct::deprotonated();
    neg_config.adduct_search_strategy = AdductSearchStrategy::Single(Adduct::deprotonated());

    let pos_wf = SiriusClusteringWorkflow::new(pos_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone());

    let neg_wf = SiriusClusteringWorkflow::new(neg_config)
        .with_fingerprint_predictor(predictor.clone())
        .with_structure_database(db.clone());

    // 4. Load Records
    let mut parser = MgfParser::default();
    let records = parser.parse_file("test_files/gnps_eval_100.mgf").expect("Failed to parse GNPS MGF");

    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<3} | {:<25} | {:<12} | {:<14} | {:<14} | {:<6} | {:<6} | {:<22}",
        "#", "Compound Name", "True Formula", "MS2-Only Pred", "MS1+MS2 Pred", "Resc?", "Tanim", "Top DB Match"
    );
    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");

    let mut total_with_truth = 0;
    let mut ms2_correct = 0;
    let mut ms1_rescued_correct = 0;
    let mut db_matched_with_ms1 = 0;

    for (idx, record) in records.iter().enumerate() {
        let name_meta = record.get_metadata("NAME").unwrap_or("Unknown");
        let base_name = name_meta.split(" M+").next().unwrap_or(name_meta).split(" M-").next().unwrap_or(name_meta).trim();

        let (true_formula_str, _expected_mode) = match ground_truth.get(base_name) {
            Some(&(f, m)) => (f, m),
            None => continue,
        };

        total_with_truth += 1;
        let true_formula = Formula::from_str(true_formula_str).expect("Valid true formula");
        let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
        let is_neg = ionmode.eq_ignore_ascii_case("Negative");
        let adduct = if is_neg { Adduct::deprotonated() } else { Adduct::protonated() };

        let pepmass: f64 = adduct.neutral_mass_to_ion_mz(true_formula.exact_mass());
        let wf = if is_neg { &neg_wf } else { &pos_wf };

        // Test 1: MS2 Only (no MS1 isotope pattern)
        let ann_ms2 = wf.annotate_spectrum(record, pepmass, None);
        let pred_ms2_formula = ann_ms2
            .as_ref()
            .and_then(|a| a.top_candidate())
            .map(|c| c.formula.to_string())
            .unwrap_or_else(|| "None".to_string());

        let is_ms2_correct = pred_ms2_formula == true_formula_str;
        if is_ms2_correct {
            ms2_correct += 1;
        }

        // Test 2: MS2 + Theoretical MS1 Isotopic Pattern
        let iso_peaks = calculate_isotope_distribution(&true_formula, 4);
        let ms1_envelope: Vec<Peak> = iso_peaks
            .into_iter()
            .map(|p| Peak {
                mz: adduct.neutral_mass_to_ion_mz(p.mz),
                intensity: (p.abundance * 10000.0) as f32,
            })
            .collect();

        let ann_ms1 = wf.annotate_spectrum(record, pepmass, Some(&ms1_envelope));
        let pred_ms1_cand = ann_ms1.as_ref().and_then(|a| a.top_candidate());
        let pred_ms1_formula = pred_ms1_cand
            .map(|c| c.formula.to_string())
            .unwrap_or_else(|| "None".to_string());

        let is_ms1_correct = pred_ms1_formula == true_formula_str;
        if is_ms1_correct {
            ms1_rescued_correct += 1;
        }

        let (db_match_name, tanimoto) = if let Some(cand) = pred_ms1_cand {
            if let Some(top_struct) = cand.structural_candidates.first() {
                db_matched_with_ms1 += 1;
                (top_struct.name.clone(), top_struct.tanimoto_similarity)
            } else {
                ("No DB Match".to_string(), 0.0)
            }
        } else {
            ("None".to_string(), 0.0)
        };

        let rescued_str = if !is_ms2_correct && is_ms1_correct {
            "RESCUED"
        } else if is_ms1_correct {
            "YES"
        } else {
            "NO"
        };

        let short_name: String = base_name.chars().take(25).collect();
        let short_db: String = db_match_name.chars().take(22).collect();

        println!(
            "{:<3} | {:<25} | {:<12} | {:<14} | {:<14} | {:<6} | {:<6.3} | {:<22}",
            total_with_truth,
            short_name,
            true_formula_str,
            pred_ms2_formula,
            pred_ms1_formula,
            rescued_str,
            tanimoto,
            short_db
        );
    }

    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("                                                MS1 Isotope Pattern Impact Summary                                                 ");
    println!("---------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("  Total Standard Compounds Evaluated:       {}", total_with_truth);
    println!("  MS2-Only Formula Accuracy:                {}/{} ({:.1}%)", ms2_correct, total_with_truth, (ms2_correct as f64 / total_with_truth.max(1) as f64) * 100.0);
    println!("  MS1 + MS2 Formula Accuracy:               {}/{} ({:.1}%)", ms1_rescued_correct, total_with_truth, (ms1_rescued_correct as f64 / total_with_truth.max(1) as f64) * 100.0);
    println!("  Structures Matched in 634k DB with MS1:   {}/{} ({:.1}%)", db_matched_with_ms1, total_with_truth, (db_matched_with_ms1 as f64 / total_with_truth.max(1) as f64) * 100.0);
    println!("===================================================================================================================================");
}
