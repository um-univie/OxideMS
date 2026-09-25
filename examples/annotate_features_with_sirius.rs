//! End-to-End Feature Finding & SIRIUS Structure Annotation Pipeline.
//!
//! Bridges high-performance LC-MS untargeted feature finding with the SIRIUS pipeline:
//! 1. Ingests mzML runs and extracts Struct-of-Arrays MS1 survey scans.
//! 2. Detects chromatographic regions of interest (ROI), resolves elution peaks via CWT,
//!    and deconvolutes isotopic patterns and charge states into quantified LC-MS features.
//! 3. Links MS2 fragmentation spectra to detected features within chromatographic elution bounds.
//! 4. Extracts authentic survey-scan MS1 isotopic envelopes at peak apexes.
//! 5. Ranks molecular formulas and fragmentation trees, then searches available
//!    structure databases using the Rust predictor or an optional DGAT bundle.
//! 6. Exports candidate annotations, feature quantities, and a run manifest.
//! Chemical classification is disabled; scores are not calibrated probabilities.

use oxidems_lib::feature_extraction::{FeatureFinder, FeatureFinderConfig, Ms1Run};
use oxidems_lib::io::mzml::MzmlParser;
use oxidems_lib::sirius::fingerprint::{FingerprintPredictor, StructureDatabase};
use oxidems_lib::sirius::workflow::{
    AdductSearchStrategy, AnnotationProfile, SiriusClusteringWorkflow, SiriusWorkflowConfig, WorkflowPreset,
};
use oxidems_lib::sirius::Adduct;
use oxidems_lib::spectrum::{MassSpectralRecord, Peak};
use oxidems_lib::workflow::MassTolerance;

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Holds combined feature quantification and SIRIUS structural identification.
#[derive(Debug, Clone)]
struct AnnotatedFeatureResult {
    feature_id: usize,
    monoisotopic_mz: f64,
    charge: i8,
    apex_rt: f32,
    rt_start: f32,
    rt_end: f32,
    area: f32,
    apex_intensity: f32,
    fwhm: f32,
    snr: f32,
    isotope_count: usize,
    linked_ms2_count: usize,
    isolation_purity: Option<f32>,
    // SIRIUS identification results
    top_formula: Option<String>,
    adduct: Option<String>,
    exact_mass: Option<f64>,
    mass_error_ppm: Option<f64>,
    total_score: Option<f64>,
    tree_score: Option<f64>,
    isotope_score: Option<f64>,
    prior_score: Option<f64>,
    explained_intensity: Option<f32>,
    tree_nodes: Option<usize>,
    canopus_superclass: Option<String>,
    canopus_class: Option<String>,
    canopus_subclass: Option<String>,
    top_structure_name: Option<String>,
    top_structure_inchikey: Option<String>,
    top_structure_smiles: Option<String>,
    tanimoto_similarity: Option<f32>,
    mod_platt_score: Option<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_file = "/scratch/muhlir/data/mass_spec_data/MSV000080554_mzML/A10_GA10_01_7891.mzML";
    let mzml_path = args.get(1).map(|s| s.as_str()).unwrap_or(default_file);
    let max_annotate: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let output_csv = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("annotated_features_sirius.csv");

    println!("=========================================================================");
    println!("    OxideMS End-to-End Feature Finding & SIRIUS Structural Annotation    ");
    println!("=========================================================================\n");

    println!("Input mzML File:    {}", mzml_path);
    println!("Annotation Limit:   {} top MS2-linked features", max_annotate);
    println!("Output CSV:         {}\n", output_csv);

    let profile = match args.get(4).map(String::as_str).unwrap_or("fast-v1") {
        "fast-v1" => AnnotationProfile::FastV1,
        "accurate-v1" => AnnotationProfile::AccurateV1,
        value => panic!("Unknown profile {value}; use fast-v1 or accurate-v1"),
    };
    let dgat_checkpoint = std::env::var("OXIDEMS_DGAT_CHECKPOINT").ok();
    let dgat_requests = std::sync::Mutex::new(Vec::<serde_json::Value>::new());
    println!("Search profile: {} | fingerprint backend: {}", profile.name(),
        if dgat_checkpoint.is_some() { "DGAT batch" } else { "Rust ResNet if available" });

    // 1. Ingest mzML file
    println!("Step 1: Parsing mzML spectra (MS1 survey + MS2 fragmentation)...");
    let t_parse = Instant::now();
    let mut parser = MzmlParser::default();
    let records = match parser.parse_file(mzml_path) {
        Ok(recs) => recs,
        Err(err) => {
            eprintln!("Error opening mzML file '{}': {}", mzml_path, err);
            return;
        }
    };
    let parse_duration = t_parse.elapsed();

    let polarities: std::collections::HashSet<_> = records.iter().filter(|r| r.ms_level == 1)
        .filter_map(|r| r.get_metadata("IONMODE")).collect();
    assert!(polarities.len() <= 1, "Mixed-polarity MS1 runs must be split by polarity before feature finding");
    let ms1_count = records.iter().filter(|r| r.ms_level == 1).count();
    let ms2_count = records.iter().filter(|r| r.ms_level == 2).count();
    println!(
        "  Parsed {} total spectra ({} MS1, {} MS2) in {:.2} ms ({:.0} spectra/sec)",
        records.len(),
        ms1_count,
        ms2_count,
        parse_duration.as_secs_f64() * 1000.0,
        records.len() as f64 / parse_duration.as_secs_f64().max(1e-6)
    );

    // Build index of MS2 records by record ID for O(1) retrieval
    let records_by_id: HashMap<usize, &MassSpectralRecord> = records
        .iter()
        .filter(|r| r.ms_level >= 2)
        .map(|r| (r.id, r))
        .collect();

    // 2. High-Performance Feature Finding
    println!("\nStep 2: Detecting chromatographic features & deconvoluting isotopes...");
    let t_ff = Instant::now();

    // Select preset based on filename/profile
    let finder_config = if mzml_path.to_lowercase().contains("orbitrap")
        || mzml_path.to_lowercase().contains("astral")
    {
        println!("  Using Instrument Preset: Orbitrap / Astral (5 ppm)");
        FeatureFinderConfig::metabolomics_orbitrap()
    } else {
        println!("  Using Instrument Preset: High-Res Q-TOF (12 ppm)");
        FeatureFinderConfig::metabolomics_qtof()
    };

    let finder = FeatureFinder::new(finder_config);
    let ms1_run = Ms1Run::from_records(&records);
    let mut feature_table = finder.find_features_from_run(&ms1_run);
    let links = feature_table.link_ms2_records_with_tolerance(&records, finder.config.mass_tolerance);
    let assignments: HashMap<_, _> = links.iter().map(|m| ((m.feature_id, m.scan_id), m)).collect();
    let feature_index: HashMap<_, _> = feature_table.features.iter().map(|f| (f.id, f)).collect();
    let purities: HashMap<_, _> = links.iter().map(|link| {
        let purity = records_by_id.get(&link.scan_id).and_then(|r| isolation_purity(
            &ms1_run, feature_index[&link.feature_id], r, finder.config.mass_tolerance));
        ((link.feature_id, link.scan_id), purity)
    }).collect();
    let ff_duration = t_ff.elapsed();

    let ms2_linked_features: Vec<&oxidems_lib::feature_extraction::Feature> = feature_table
        .features
        .iter()
        .filter(|f| !f.linked_ms2_ids.is_empty())
        .collect();

    println!(
        "  Found {} quantified features in {:.2} ms",
        feature_table.len(),
        ff_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Features with linked MS2 spectra: {} ({:.1}%)",
        ms2_linked_features.len(),
        (ms2_linked_features.len() as f64 / feature_table.len().max(1) as f64) * 100.0
    );

    // 3. Build MS1 Run for survey-scan isotopic envelope extraction
    // Reuse the MS1 run used for feature finding.

    // 4. Load CSI:FingerID Model and Structure Database
    println!("\nStep 3: Initializing SIRIUS + CSI:FingerID + BioMetabolites Engine...");
    let fp_model_path = "models/sirius_fingerprints.oxfp";
    let bio_db_path = "models/biometabolites.oxdb";
    let pubchem_db_path = "models/pubchem_subset.oxdb";

    let t_models = Instant::now();
    let predictor = if dgat_checkpoint.is_none() && Path::new(fp_model_path).exists() {
        match FingerprintPredictor::load_from_file(fp_model_path) {
            Ok(p) => {
                println!(
                    "  [CSI:FingerID] Loaded neural model ({} parameters)",
                    p.num_parameters()
                );
                Some(p)
            }
            Err(e) => {
                eprintln!("  [Warning] Failed to load {}: {}", fp_model_path, e);
                None
            }
        }
    } else {
        println!("  Rust ResNet inactive; DGAT batch backend: {}", dgat_checkpoint.is_some());
        None
    };

    let structure_db = if Path::new(bio_db_path).exists() {
        match StructureDatabase::load_from_file(bio_db_path) {
            Ok(mut db) => {
                println!(
                    "  [BioMetabolites DB] Loaded {} structures across {} formulas",
                    db.total_structures,
                    db.entries_by_formula.len()
                );
                if Path::new(pubchem_db_path).exists() {
                    if let Ok(pc_db) = StructureDatabase::load_from_file(pubchem_db_path) {
                        let pc_structs = pc_db.total_structures;
                        for (_, structs) in pc_db.entries_by_formula {
                            for s in structs {
                                db.insert(s);
                            }
                        }
                        println!(
                            "  [PubChem Subset DB] Merged {} additional structures",
                            pc_structs
                        );
                    }
                }
                Some(db)
            }
            Err(e) => {
                eprintln!("  [Warning] Failed to load {}: {}", bio_db_path, e);
                None
            }
        }
    } else {
        println!("  [Notice] BioMetabolites DB not found at {}. Proceeding with de novo formulas.", bio_db_path);
        None
    };
    println!("  Engine initialized in {:.2} ms", t_models.elapsed().as_secs_f64() * 1000.0);

    // 5. Configure SIRIUS Workflow
    let workflow_config = SiriusWorkflowConfig {
        adduct_search_strategy: if profile == AnnotationProfile::AccurateV1 { AdductSearchStrategy::AutoPositive } else { AdductSearchStrategy::TwoTierPositive },
        ..SiriusWorkflowConfig::for_profile(profile)
    };

    let mut workflow = SiriusClusteringWorkflow::new(workflow_config);
    if let Some(p) = predictor {
        workflow = workflow.with_fingerprint_predictor(p);
    }
    if let Some(db) = structure_db {
        workflow = workflow.with_structure_database(db);
    }
    // The current taxonomy implementation is a heuristic, not a trained CANOPUS model.
    workflow = workflow.with_compound_classification(false);
    let mut negative = SiriusClusteringWorkflow::new(SiriusWorkflowConfig {
        enable_fingerprint_prediction: false,
        adduct_search_strategy: if profile == AnnotationProfile::AccurateV1 { AdductSearchStrategy::AutoNegative } else { AdductSearchStrategy::TwoTierNegative },
        ..workflow.config.clone()
    });
    if let Some(ref p) = workflow.fingerprint_predictor {
        negative = negative.with_fingerprint_predictor(p.clone());
    }

    // 6. Annotate Features with Linked MS2 Spectra
    // Sort features by chromatographic area descending
    let mut candidate_features: Vec<oxidems_lib::feature_extraction::Feature> = feature_table
        .features
        .into_iter()
        .filter(|f| !f.linked_ms2_ids.is_empty())
        .collect();
    candidate_features.sort_by(|a, b| b.total_area.total_cmp(&a.total_area));

    let to_annotate = if max_annotate > 0 {
        candidate_features.truncate(max_annotate);
        &candidate_features[..]
    } else {
        &candidate_features[..]
    };

    println!(
        "\nStep 4: Annotating {} top chromatographic features with SIRIUS...",
        to_annotate.len()
    );
    let t_annotate = Instant::now();

    // Process features in parallel across Rayon worker threads
    let mut annotated_results: Vec<AnnotatedFeatureResult> = to_annotate
        .par_iter()
        .map(|feat| {
            // Pick best linked MS2 record (richest fragmentation scan)
            let best_record = feat
                .linked_ms2_ids
                .iter()
                .filter_map(|id| records_by_id.get(id))
                .max_by(|a, b| {
                    // Prefer unambiguous precursor assignments, then apex proximity.
                    let quality = |r: &&MassSpectralRecord| {
                        let ambiguous = assignments.get(&(feat.id, r.id)).is_some_and(|m| m.ambiguous);
                        let rt = r.get_metadata("RTINSECONDS").and_then(|v| v.parse::<f32>().ok()).unwrap_or(f32::INFINITY);
                        let purity = purities.get(&(feat.id, r.id)).copied().flatten().unwrap_or(0.5);
                        (!ambiguous, purity, -(rt - feat.apex_rt).abs())
                    };
                    let qa = quality(a); let qb = quality(b);
                    qa.0.cmp(&qb.0).then(qa.1.total_cmp(&qb.1)).then(qa.2.total_cmp(&qb.2))
                });

            let mut ann_result = AnnotatedFeatureResult {
                feature_id: feat.id,
                monoisotopic_mz: feat.monoisotopic_mz,
                charge: feat.charge,
                apex_rt: feat.apex_rt,
                rt_start: feat.rt_start,
                rt_end: feat.rt_end,
                area: feat.total_area,
                apex_intensity: feat.apex_intensity,
                fwhm: feat.fwhm,
                snr: feat.snr,
                isotope_count: feat.isotope_count,
                linked_ms2_count: feat.linked_ms2_ids.len(),
                isolation_purity: best_record.and_then(|r| purities.get(&(feat.id, r.id)).copied().flatten()),
                top_formula: None,
                adduct: None,
                exact_mass: None,
                mass_error_ppm: None,
                total_score: None,
                tree_score: None,
                isotope_score: None,
                prior_score: None,
                explained_intensity: None,
                tree_nodes: None,
                canopus_superclass: None,
                canopus_class: None,
                canopus_subclass: None,
                top_structure_name: None,
                top_structure_inchikey: None,
                top_structure_smiles: None,
                tanimoto_similarity: None,
                mod_platt_score: None,
            };

            let Some(raw_rec) = best_record else {
                return ann_result;
            };

            // Ensure MS2 record reflects feature's deconvoluted charge
            let mut record = (*raw_rec).clone();
            if record.get_charge() == 0 && feat.charge != 0 {
                let sign = if record.get_metadata("IONMODE").is_some_and(|m| m.eq_ignore_ascii_case("negative")) { -1 } else { 1 };
                record.set_charge(sign * feat.charge.abs());
            }

            // Extract MS1 survey-scan isotopic envelope at peak apex
            let ms1_envelope = extract_survey_isotope_envelope(
                &ms1_run,
                feat.apex_rt,
                feat.monoisotopic_mz,
                feat.charge,
                (feat.rt_start, feat.rt_end),
                finder.config.mass_tolerance,
            );

            // Execute SIRIUS MAP formula tree optimization & structure identification
            let active_workflow = if record.get_charge() < 0 { &negative } else { &workflow };
            let annotation = active_workflow.annotate_spectrum(
                &record,
                feat.monoisotopic_mz,
                if !ms1_envelope.is_empty() {
                    Some(&ms1_envelope)
                } else {
                    None
                },
            );

            if let Some(ann) = annotation {
                if let Some(top) = ann.top_candidate() {
                    if dgat_checkpoint.is_some() && !top.tree.edges.is_empty() {
                        if let Some(ref db) = active_workflow.structure_database {
                            let (nodes, edges, edge_feats) = oxidems_lib::sirius::fingerprint::features::tree_graph(&top.tree, record.get_charge());
                            let candidates: Vec<_> = db.get_candidates_for_context(&top.formula, active_workflow.config.sample_context)
                                .iter().map(|c| serde_json::json!({"name": c.name, "smiles": c.smiles, "inchi_key": c.inchi_key})).collect();
                            dgat_requests.lock().unwrap().push(serde_json::json!({"id": feat.id,
                                "graph_nodes": nodes, "graph_edges": edges, "graph_edge_feats": edge_feats,
                                "candidates": candidates}));
                        }
                    }
                    ann_result.top_formula = Some(top.formula.to_string());
                    ann_result.adduct = Some(top.adduct.name.clone());
                    ann_result.exact_mass = Some(top.exact_mass);
                    ann_result.mass_error_ppm = Some(top.mass_error_ppm);
                    ann_result.total_score = Some(top.total_score);
                    ann_result.tree_score = Some(top.tree_score);
                    ann_result.isotope_score = Some(top.isotope_score);
                    ann_result.prior_score = Some(top.prior_score);
                    ann_result.explained_intensity = Some(top.explained_intensity);
                    ann_result.tree_nodes = Some(top.tree.nodes.len());

                    if let Some(ref classif) = top.classification {
                        ann_result.canopus_superclass =
                            classif.superclass.as_ref().map(|s| s.name.clone());
                        ann_result.canopus_class =
                            classif.class.as_ref().map(|c| c.name.clone());
                        ann_result.canopus_subclass =
                            classif.subclass.as_ref().map(|s| s.name.clone());
                    }

                    if let Some(top_struct) = top.structural_candidates.first() {
                        ann_result.top_structure_name = Some(top_struct.name.clone());
                        ann_result.top_structure_inchikey = top_struct.inchi_key.clone();
                        ann_result.top_structure_smiles = Some(top_struct.smiles.clone());
                        ann_result.tanimoto_similarity = Some(top_struct.tanimoto_similarity);
                        ann_result.mod_platt_score = Some(top_struct.mod_platt_score);
                    }
                }
            }

            ann_result
        })
        .collect();

    if let Some(ref checkpoint) = dgat_checkpoint {
        let request_path = format!("{output_csv}.graphs.jsonl");
        let prediction_path = format!("{output_csv}.structures.jsonl");
        let mut requests = dgat_requests.into_inner().unwrap();
        requests.sort_by_key(|r| r["id"].as_u64().unwrap());
        let mut out = BufWriter::new(File::create(&request_path).expect("create inference requests"));
        for request in requests { writeln!(out, "{request}").unwrap(); }
        out.flush().unwrap();
        let python = std::env::var("OXIDEMS_PYTHON").unwrap_or_else(|_| ".venv-ml/bin/python".into());
        let mut command = std::process::Command::new(python);
        command.arg("scratch/ml/predict_graphs.py");
        if let Ok(data) = std::env::var("OXIDEMS_DGAT_DATA") { command.args(["--data", &data]); }
        let status = command
            .args(["--checkpoint", checkpoint,
                   "--input", &request_path, "--output", &prediction_path])
            .env("OXFP_VOCAB", std::env::var("OXFP_VOCAB").unwrap_or_else(|_| "scratch/ml/vocab_db.json".into()))
            .env("OXFP_RDK_BITS", std::env::var("OXFP_RDK_BITS").unwrap_or_else(|_| "4096".into()))
            .status().expect("start DGAT batch inference");
        assert!(status.success(), "DGAT inference failed; no successful annotated CSV will be exported");
        for line in std::fs::read_to_string(&prediction_path).unwrap().lines() {
            let prediction: serde_json::Value = serde_json::from_str(line).unwrap();
            let id = prediction["id"].as_u64().unwrap() as usize;
            if let Some(result) = annotated_results.iter_mut().find(|r| r.feature_id == id) {
                if let Some(top) = prediction["candidates"].as_array().and_then(|c| c.first()) {
                    result.top_structure_name = top["name"].as_str().map(str::to_owned);
                    result.top_structure_smiles = top["smiles"].as_str().map(str::to_owned);
                    result.top_structure_inchikey = top["inchi_key"].as_str().map(str::to_owned);
                    if prediction["score_kind"] == "mod_platt" { result.mod_platt_score = top["score"].as_f64(); }
                }
            }
        }
    }
    let manifest = serde_json::json!({"schema": 1, "profile": profile.name(), "input": mzml_path,
        "max_features": max_annotate, "tolerance": workflow.config.tolerance,
        "peak_budget": workflow.config.max_peaks_per_tree,
        "precursor_cap": if workflow.config.max_candidates_for_tree_search == usize::MAX { None } else { Some(workflow.config.max_candidates_for_tree_search) },
        "alphabet": "CHNOPS+ClBr", "exact_shortlist": workflow.tree_builder.config().exact_top_n,
        "fingerprint_checkpoint": dgat_checkpoint.as_deref().unwrap_or(fp_model_path),
        "ambiguous_links": links.iter().filter(|m| m.ambiguous).count(), "links": links});
    std::fs::write(format!("{output_csv}.manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let annotate_duration = t_annotate.elapsed();
    let successful_annotations = annotated_results
        .iter()
        .filter(|r| r.top_formula.is_some())
        .count();

    println!(
        "  Annotated {}/{} features in {:.2} s ({:.1} ms/feature)",
        successful_annotations,
        to_annotate.len(),
        annotate_duration.as_secs_f64(),
        annotate_duration.as_secs_f64() * 1000.0 / to_annotate.len().max(1) as f64
    );

    // 7. Display Summary Results Table
    println!("\n=================================================================================================================================");
    println!("                                                ANNOTATED METABOLOMICS FEATURES                                                  ");
    println!("=================================================================================================================================");
    println!(
        "{:<4} | {:<7} | {:<9} | {:<7} | {:<12} | {:<8} | {:<5} | {:<6} | {:<20} | {:<24} | {:<5}",
        "ID", "RT (s)", "m/z", "Area", "Formula", "Adduct", "ppm", "Score", "Compound Class", "Top Structure Match", "Tanim"
    );
    println!("---------------------------------------------------------------------------------------------------------------------------------");

    for res in &annotated_results {
        let formula_str = res.top_formula.as_deref().unwrap_or("Unassigned");
        let adduct_str = res.adduct.as_deref().unwrap_or("-");
        let ppm_str = res
            .mass_error_ppm
            .map(|p| format!("{:.1}", p))
            .unwrap_or_else(|| "-".to_string());
        let score_str = res
            .total_score
            .map(|s| format!("{:.1}", s))
            .unwrap_or_else(|| "-".to_string());
        let class_str = res
            .canopus_class
            .as_deref()
            .or_else(|| res.canopus_superclass.as_deref())
            .unwrap_or("N/A");
        let short_class: String = class_str.chars().take(20).collect();
        let struct_str = res.top_structure_name.as_deref().unwrap_or("N/A");
        let short_struct: String = struct_str.chars().take(24).collect();
        let tanim_str = res
            .tanimoto_similarity
            .map(|t| format!("{:.2}", t))
            .unwrap_or_else(|| "-".to_string());

        let area_formatted = if res.area >= 1e6 {
            format!("{:.1}M", res.area / 1e6)
        } else if res.area >= 1e3 {
            format!("{:.1}k", res.area / 1e3)
        } else {
            format!("{:.0}", res.area)
        };

        println!(
            "{:<4} | {:<7.1} | {:<9.4} | {:<7} | {:<12} | {:<8} | {:<5} | {:<6} | {:<20} | {:<24} | {:<5}",
            res.feature_id,
            res.apex_rt,
            res.monoisotopic_mz,
            area_formatted,
            formula_str,
            adduct_str,
            ppm_str,
            score_str,
            short_class,
            short_struct,
            tanim_str
        );
    }
    println!("=================================================================================================================================\n");

    // 8. Detailed View for Top 3 Confident Features
    println!("Detailed candidate annotations:");
    let top_detailed = annotated_results
        .iter()
        .filter(|r| r.top_formula.is_some() && r.top_structure_name.is_some())
        .take(3);

    for (i, res) in top_detailed.enumerate() {
        println!("-------------------------------------------------------------------------");
        println!(
            "[{}] Feature #{} | RT: {:.1}s ({:.2} min) | m/z: {:.4} | Charge: {:+}",
            i + 1,
            res.feature_id,
            res.apex_rt,
            res.apex_rt / 60.0,
            res.monoisotopic_mz,
            res.charge
        );
        println!(
            "    Chromatography: RT Range: [{:.1} - {:.1}s] | FWHM: {:.1}s | SNR: {:.1} | Area: {:.0}",
            res.rt_start, res.rt_end, res.fwhm, res.snr, res.area
        );
        println!(
            "    SIRIUS Formula: {} {} (Exact: {:.4} Da, Error: {:.2} ppm)",
            res.top_formula.as_deref().unwrap_or("?"),
            res.adduct.as_deref().unwrap_or(""),
            res.exact_mass.unwrap_or(0.0),
            res.mass_error_ppm.unwrap_or(0.0)
        );
        println!(
            "    Scoring:        Total: {:.2} | Tree: {:.2} | Isotope: {:.2} | Prior: {:.2}",
            res.total_score.unwrap_or(0.0),
            res.tree_score.unwrap_or(0.0),
            res.isotope_score.unwrap_or(0.0),
            res.prior_score.unwrap_or(0.0)
        );
        println!(
            "    Fragmentation:  Explained Int: {:.1}% | Tree Vertices: {}",
            res.explained_intensity.unwrap_or(0.0) * 100.0,
            res.tree_nodes.unwrap_or(0)
        );
        println!(
            "    CANOPUS Class:  Superclass: {} | Class: {} | Subclass: {}",
            res.canopus_superclass.as_deref().unwrap_or("N/A"),
            res.canopus_class.as_deref().unwrap_or("N/A"),
            res.canopus_subclass.as_deref().unwrap_or("N/A")
        );
        println!(
            "    Structure:      {} (Tanimoto: {}, ModPlatt: {})",
            res.top_structure_name.as_deref().unwrap_or("N/A"),
            res.tanimoto_similarity.map(|v| format!("{v:.2}")).unwrap_or_else(|| "N/A".into()),
            res.mod_platt_score.map(|v| format!("{v:.2}")).unwrap_or_else(|| "N/A".into())
        );
        if let Some(ref ik) = res.top_structure_inchikey {
            println!("    InChIKey:       {}", ik);
        }
        if let Some(ref sm) = res.top_structure_smiles {
            println!("    SMILES:         {}", sm);
        }
    }
    println!("-------------------------------------------------------------------------\n");

    // 9. Export to CSV
    export_to_csv(output_csv, &annotated_results).expect("Failed to export annotated CSV");
    println!("Successfully exported full annotations to: {}", output_csv);
}

/// MS1 intensity fraction attributable to this feature inside the measured isolation window.
/// Missing window metadata produces None, not an assumed purity of 100%.
fn isolation_purity(run: &Ms1Run, feat: &oxidems_lib::feature_extraction::Feature,
                    record: &MassSpectralRecord, tolerance: MassTolerance) -> Option<f32> {
    let center: f64 = record.get_metadata("ISOLATION_TARGET")?.parse().ok()?;
    let lower: f64 = record.get_metadata("ISOLATION_LOWER")?.parse().ok()?;
    let upper: f64 = record.get_metadata("ISOLATION_UPPER")?.parse().ok()?;
    let rt: f32 = record.get_metadata("RTINSECONDS")?.parse().ok()?;
    if lower < 0.0 || upper < 0.0 || run.num_scans() == 0 { return None; }
    let pos = run.scan_rts.partition_point(|&r| r <= rt);
    let scan = [pos.saturating_sub(1), pos.min(run.num_scans()-1)].into_iter()
        .min_by(|&a, &b| (run.scan_rts[a]-rt).abs().total_cmp(&(run.scan_rts[b]-rt).abs()))?;
    let (mzs, ints) = run.scan_peaks(scan);
    let lo = mzs.partition_point(|&m| m < center-lower);
    let hi = mzs.partition_point(|&m| m <= center+upper);
    let mut total = 0.0; let mut assigned = 0.0;
    for i in lo..hi {
        if !ints[i].is_finite() || ints[i] <= 0.0 { continue; }
        total += ints[i];
        let z = feat.charge.unsigned_abs().max(1) as f64;
        if (0..feat.isotope_count.max(1)).any(|k| {
            let target = feat.monoisotopic_mz + k as f64 * 1.003355/z;
            (mzs[i]-target).abs() <= tolerance.tolerance_at(target)
        }) { assigned += ints[i]; }
    }
    if total > 0.0 { Some((assigned/total).clamp(0.0, 1.0)) } else { None }
}

/// Extracts survey-scan isotopic envelope around a given precursor m/z and apex retention time.
fn extract_survey_isotope_envelope(
    run: &Ms1Run,
    rt: f32,
    precursor_mz: f64,
    charge: i8,
    rt_bounds: (f32, f32),
    tolerance: MassTolerance,
) -> Vec<Peak> {
    if run.scan_rts.is_empty() { return Vec::new(); }
    let pos = run.scan_rts.partition_point(|&scan_rt| scan_rt <= rt);
    let mut scans: Vec<_> = (pos.saturating_sub(2)..(pos + 2).min(run.num_scans()))
        .filter(|&i| run.scan_rts[i] >= rt_bounds.0 && run.scan_rts[i] <= rt_bounds.1).collect();
    scans.sort_by(|&a, &b| (run.scan_rts[a] - rt).abs().total_cmp(&(run.scan_rts[b] - rt).abs()));
    scans.truncate(3);
    let z = charge.unsigned_abs().max(1) as f64;
    let tol = tolerance.tolerance_at(precursor_mz);
    let mut envelope: Vec<Peak> = Vec::new();
    for scan in scans {
        let (mzs, ints) = run.scan_peaks(scan);
        let lo = mzs.partition_point(|&m| m < precursor_mz - tol);
        let hi = mzs.partition_point(|&m| m <= precursor_mz + tol);
        let anchor = (lo..hi).filter(|&i| ints[i] > 0.0)
            .min_by(|&a, &b| (mzs[a] - precursor_mz).abs().total_cmp(&(mzs[b] - precursor_mz).abs()));
        let Some(anchor) = anchor else { continue; };
        // Retain fine isotope structure (including Cl/Br), rather than searching
        // exclusively at multiples of 13C spacing. The scorer matches the peaks.
        let end = mzs.partition_point(|&m| m <= mzs[anchor] + 4.1 / z);
        for i in anchor..end {
            if ints[i] <= 0.0 { continue; }
            if let Some(p) = envelope.iter_mut().find(|p| (p.mz - mzs[i]).abs() <= tolerance.tolerance_at(mzs[i])) {
                let total = p.intensity + ints[i];
                p.mz = (p.mz * p.intensity as f64 + mzs[i] * ints[i] as f64) / total as f64;
                p.intensity = total;
            } else { envelope.push(Peak::new(mzs[i], ints[i])); }
        }
    }
    envelope.sort_by(|a, b| a.mz.total_cmp(&b.mz));
    envelope
}

/// Exports annotated feature results to a standardized CSV file.
fn export_to_csv(path: &str, results: &[AnnotatedFeatureResult]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "feature_id,rt,rt_min,rt_start,rt_end,monoisotopic_mz,charge,area,apex_intensity,fwhm,snr,isotopes,ms2_count,formula,adduct,exact_mass,mass_error_ppm,total_score,tree_score,isotope_score,prior_score,explained_intensity,tree_nodes,canopus_superclass,canopus_class,canopus_subclass,top_structure_name,top_structure_inchikey,top_structure_smiles,tanimoto_similarity,mod_platt_score,isolation_purity"
    )?;

    for r in results {
        writeln!(
            writer,
            "{},{:.2},{:.4},{:.2},{:.2},{:.6},{},{:.2},{:.2},{:.2},{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.feature_id,
            r.apex_rt,
            r.apex_rt / 60.0,
            r.rt_start,
            r.rt_end,
            r.monoisotopic_mz,
            r.charge,
            r.area,
            r.apex_intensity,
            r.fwhm,
            r.snr,
            r.isotope_count,
            r.linked_ms2_count,
            escape_csv(r.top_formula.as_deref().unwrap_or("")),
            escape_csv(r.adduct.as_deref().unwrap_or("")),
            r.exact_mass.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            r.mass_error_ppm.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.total_score.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.tree_score.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.isotope_score.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.prior_score.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.explained_intensity.map(|v| format!("{:.3}", v)).unwrap_or_default(),
            r.tree_nodes.map(|v| v.to_string()).unwrap_or_default(),
            escape_csv(r.canopus_superclass.as_deref().unwrap_or("")),
            escape_csv(r.canopus_class.as_deref().unwrap_or("")),
            escape_csv(r.canopus_subclass.as_deref().unwrap_or("")),
            escape_csv(r.top_structure_name.as_deref().unwrap_or("")),
            escape_csv(r.top_structure_inchikey.as_deref().unwrap_or("")),
            escape_csv(r.top_structure_smiles.as_deref().unwrap_or("")),
            r.tanimoto_similarity.map(|v| format!("{:.3}", v)).unwrap_or_default(),
            r.mod_platt_score.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            r.isolation_purity.map(|v| format!("{:.4}", v)).unwrap_or_default(),
        )?;
    }

    writer.flush()?;
    Ok(())
}

fn escape_csv(val: &str) -> String {
    if val.contains(',') || val.contains('"') || val.contains('\n') {
        format!("\"{}\"", val.replace('"', "\"\""))
    } else {
        val.to_string()
    }
}
