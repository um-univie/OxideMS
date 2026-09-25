//! Train the CSI:FingerID-style molecular fingerprint predictor on MS/MS reference standards.
//!
//! Parses authentic MS/MS reference spectra (e.g. MoNA LC-MS/MS or GNPS) with known SMILES,
//! builds MAP fragmentation trees via OxideMS's fast DP solver, generates ground-truth
//! 2,214-bit molecular fingerprints via `oxidane` (MACCS + Morgan/ECFP4), extracts
//! 5,147-d tree feature vectors, and trains the ResNet predictor with AdamW and compound loss.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example train_fingerprint_model -- [OPTIONS]
//!
//! Options:
//!   --msp <PATH>        Path to MoNA MSP file (default: test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp)
//!   --max-samples <N>   Maximum samples to extract (default: 3000)
//!   --epochs <N>        Training epochs (default: 35)
//!   --batch-size <N>    Batch size (default: 64)
//!   --lr <F>            Learning rate (default: 0.001)
//!   --output <PATH>     Output model path (default: models/sirius_fingerprints.oxfp)
//! ```

use oxidems_lib::sirius::fingerprint::{
    BinaryFingerprint, DatasetManifest, FingerprintDictionary, FingerprintPredictor,
    FingerprintSearcher, MorganVocabulary, MORGAN_OVERFLOW_SLOTS, Split, SplitRatios, TOTAL_BITS,
    TOTAL_FEATURES, TrainingConfig, TrainingSample, TreeFeatureExtractor, assign_split,
    morgan_identifiers, spectrum_key, structure_key, train,
};
use oxidems_lib::sirius::{Adduct, ChemicalRules, Formula, FragmentationTreeBuilder, SiriusClusteringWorkflow, TreeSolverConfig};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::collections::HashMap;
use std::time::Instant;

/// Raw unfeaturized record parsed from MoNA MSP.
struct RawMspRecord {
    name: String,
    formula_str: String,
    smiles: String,
    precursor_mz: f64,
    charge: i8,
    peaks: Vec<Peak>,
}

fn parse_mona_msp(path: &Path, max_records: usize) -> Vec<RawMspRecord> {
    let file = File::open(path).expect("Could not open MSP file");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut records = Vec::new();
    let mut current_name = String::new();
    let mut current_formula = String::new();
    let mut current_smiles = String::new();
    let mut current_mz = 0.0f64;
    let mut current_charge = 1i8;
    let mut current_peaks = Vec::new();

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        if line.is_empty() {
            if !current_smiles.is_empty() && !current_formula.is_empty() && !current_peaks.is_empty() && current_mz > 0.0 {
                records.push(RawMspRecord {
                    name: std::mem::take(&mut current_name),
                    formula_str: std::mem::take(&mut current_formula),
                    smiles: std::mem::take(&mut current_smiles),
                    precursor_mz: current_mz,
                    charge: current_charge,
                    peaks: std::mem::take(&mut current_peaks),
                });

                if records.len() >= max_records {
                    break;
                }
            }
            current_mz = 0.0;
            current_charge = 1;
            continue;
        }

        if let Some(val) = line.strip_prefix("Name:") {
            current_name = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("Formula:") {
            current_formula = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("PrecursorMZ:") {
            current_mz = val.trim().parse::<f64>().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("Ion_mode:") {
            if val.trim().starts_with('N') || val.trim().starts_with('-') {
                current_charge = -1;
            } else {
                current_charge = 1;
            }
        } else if line.starts_with("Comments:") {
            if let Some(idx) = line.find("SMILES=") {
                let rest = &line[idx + 7..];
                let end = rest.find('"').or_else(|| rest.find(' ')).unwrap_or(rest.len());
                current_smiles = rest[..end].trim().to_string();
            }
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz), Ok(intensity)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if mz > 0.0 && intensity > 0.0 {
                        current_peaks.push(Peak { mz, intensity });
                    }
                }
            }
        }
    }

    records
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut msp_path = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp".to_string();
    let mut max_samples = 3000usize;
    let mut epochs = 35usize;
    let mut batch_size = 128usize;
    let mut lr = 0.0003f64;
    let mut hidden_dim = 1024usize;
    let mut num_blocks = 4usize;
    let mut split_seed = 42u64;
    let mut no_vocabulary = false;
    let mut exclude_manifest: Option<String> = None;
    let mut dropout = 0.0f32;
    let mut feature_dropout = 0.0f32;
    let mut output_model = "models/sirius_fingerprints.oxfp".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--msp" => {
                msp_path = args[i + 1].clone();
                i += 2;
            }
            "--max-samples" => {
                max_samples = args[i + 1].parse().unwrap_or(max_samples);
                i += 2;
            }
            "--epochs" => {
                epochs = args[i + 1].parse().unwrap_or(epochs);
                i += 2;
            }
            "--batch-size" => {
                batch_size = args[i + 1].parse().unwrap_or(batch_size);
                i += 2;
            }
            "--lr" => {
                lr = args[i + 1].parse().unwrap_or(lr);
                i += 2;
            }
            "--hidden-dim" => {
                hidden_dim = args[i + 1].parse().unwrap_or(hidden_dim);
                i += 2;
            }
            // Structures to keep out of training entirely, so a benchmark built from
            // them is answerable. Without this, an evaluation set that overlaps the
            // reference library measures recall of memorized compounds.
            "--exclude-manifest" => {
                exclude_manifest = Some(args[i + 1].clone());
                i += 2;
            }
            "--no-vocabulary" => {
                no_vocabulary = true;
                i += 1;
            }
            "--split-seed" => {
                split_seed = args[i + 1].parse().unwrap_or(split_seed);
                i += 2;
            }
            "--dropout" => {
                dropout = args[i + 1].parse().unwrap_or(dropout);
                i += 2;
            }
            "--feature-dropout" => {
                feature_dropout = args[i + 1].parse().unwrap_or(feature_dropout);
                i += 2;
            }
            "--num-blocks" => {
                num_blocks = args[i + 1].parse().unwrap_or(num_blocks);
                i += 2;
            }
            "--output" => {
                output_model = args[i + 1].clone();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("=========================================================================");
    println!("           OxideMS CSI:FingerID Molecular Fingerprint Training           ");
    println!("=========================================================================");
    println!("  MSP Input:       {}", msp_path);
    println!("  Target Samples:  {}", max_samples);
    println!("  Epochs:          {}", epochs);
    println!("  Batch Size:      {}", batch_size);
    println!("  Learning Rate:   {}", lr);
    println!("  Hidden Dim:      {}", hidden_dim);
    println!("  ResNet Blocks:   {}", num_blocks);
    println!("  Output Model:    {}", output_model);
    println!("=========================================================================\n");

    // ── 1. Ingest MSP Reference Library ─────────────────────────────────────
    let t_start = Instant::now();
    println!("Step 1/4: Parsing MS/MS Reference Spectra from {}...", msp_path);
    // Exactly max_samples records, in file order. Parsing a surplus and then racing
    // the parallel extractor to fill a quota made the dataset depend on thread timing,
    // so no two runs trained on the same spectra and no two results were comparable.
    let raw_records = parse_mona_msp(Path::new(&msp_path), max_samples);
    println!(
        "  Parsed {} candidate records with SMILES in {:.2}s",
        raw_records.len(),
        t_start.elapsed().as_secs_f64()
    );

    let held_out = exclude_manifest.as_ref().map(|p| {
        let m = DatasetManifest::load(p).expect("Failed to read the exclusion manifest");
        println!(
            "  Holding {} structures out of training entirely (from {p}).",
            m.structure_keys.len()
        );
        m
    });
    let raw_records: Vec<_> = match &held_out {
        Some(m) => {
            let before = raw_records.len();
            let kept: Vec<_> = raw_records
                .into_iter()
                .filter(|r| !m.contains_structure(&r.smiles))
                .collect();
            println!("  Dropped {} of {before} records as held out.", before - kept.len());
            kept
        }
        None => raw_records,
    };

    // ── 2. Morgan vocabulary, counted on the training partition only ────────
    // The vocabulary decides what a Morgan bit means, so it is part of the model, and
    // counting it over validation or test structures would leak them into the layout
    // the model is fitted in. Formula-disjoint assignment is known before any
    // fingerprint is computed, so the training partition can be isolated first.
    let ratios = SplitRatios::default();
    let t_vocab = Instant::now();
    let mut identifier_counts: HashMap<u32, usize> = HashMap::new();
    for counts in raw_records
        .par_iter()
        .filter(|rec| assign_split(&rec.formula_str, ratios, split_seed) == Split::Train)
        .filter_map(|rec| morgan_identifiers(&rec.smiles))
        .collect::<Vec<_>>()
    {
        for id in counts {
            *identifier_counts.entry(id).or_default() += 1;
        }
    }
    // An empty vocabulary means plain modulo folding — the layout before dedicated
    // slots existed. Kept as a flag so the comparison between the two can be repeated
    // under whatever the rest of the pipeline currently does, rather than against a
    // model trained by an older one.
    let vocabulary = if no_vocabulary {
        MorganVocabulary::new()
    } else {
        MorganVocabulary::from_counts(&identifier_counts)
    };
    let mut dictionary = FingerprintDictionary::new();
    dictionary.set_morgan_vocabulary(vocabulary.clone());
    println!(
        "\nStep 2/4: Morgan vocabulary from {} training identifiers -> {} dedicated slots + {} folded ({:.2}s)",
        identifier_counts.len(),
        vocabulary.ids().len(),
        MORGAN_OVERFLOW_SLOTS,
        t_vocab.elapsed().as_secs_f64()
    );

    // ── 3. Parallel Fragmentation Tree & Fingerprint Generation ─────────────
    println!("\nStep 3/5: Computing MAP Fragmentation Trees & Ground-Truth Fingerprints (Rayon parallel)...");
    let t_feat = Instant::now();

    let workflow = SiriusClusteringWorkflow::default();

    struct RawSample {
        formula_str: String,
        /// 2D structure and spectrum identities, recorded so the manifest can state
        /// exactly what this run consumed.
        structure_key: Option<String>,
        spectrum_key: u64,
        features: Vec<f32>,
        fingerprint: Vec<bool>,
    }

    let raw_samples: Vec<RawSample> = raw_records
        .par_iter()
        .filter_map(|rec| {
            let formula = Formula::from_str(&rec.formula_str).ok()?;
            let exact_mass = formula.exact_mass();

            if !(50.0..=900.0).contains(&exact_mass) || rec.peaks.is_empty() {
                return None;
            }

            let fp = dictionary.ground_truth_fingerprint(&rec.smiles)?;

            let clean_peaks = workflow.preprocess_peaks_for_tree(&rec.peaks, rec.precursor_mz, rec.charge)?;

            let adduct = if rec.charge < 0 {
                Adduct::deprotonated()
            } else {
                Adduct::protonated()
            };

            let config = TreeSolverConfig {
                tolerance: MassTolerance::Ppm(20.0),
                adduct,
                ..Default::default()
            };
            let builder = FragmentationTreeBuilder::new(config);

            let tree = builder.build_tree(&formula, rec.precursor_mz, &clean_peaks);

            if tree.nodes.len() < 2 {
                return None;
            }

            let extractor = TreeFeatureExtractor::new();
            let features = extractor.extract(&tree).features;

            Some(RawSample {
                formula_str: rec.formula_str.clone(),
                structure_key: structure_key(&rec.smiles),
                spectrum_key: spectrum_key(rec.precursor_mz, &rec.peaks),
                features,
                fingerprint: fp,
            })
        })
        .collect();

    println!(
        "  Extracted {} valid training pairs (Features [{} dims] -> Fingerprints [{} bits]) in {:.2}s ({:.1} samples/s)",
        raw_samples.len(),
        TOTAL_FEATURES,
        TOTAL_BITS,
        t_feat.elapsed().as_secs_f64(),
        raw_samples.len() as f64 / t_feat.elapsed().as_secs_f64().max(0.001)
    );

    if raw_samples.is_empty() {
        eprintln!("Error: No valid training samples could be extracted!");
        return;
    }

    // ── 4. Formula-Disjoint Train / Validation / Test Splitting ──────────────
    // The partition of a formula is a hash of the formula itself, so every spectrum of
    // every isomer of it lands together, and it lands in the same place no matter how
    // many samples this run collected or in what order.
    let mut train_samples = Vec::new();
    let mut val_samples = Vec::new();
    let mut test_samples = Vec::new();
    let mut formulas_in: std::collections::HashMap<Split, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    // Everything this run consumed, in any partition. Validation and test spectra never
    // reach the optimizer, but they choose the checkpoint and the reported numbers, so
    // an independent benchmark has to stay clear of them too.
    let mut manifest = DatasetManifest::new();

    for s in raw_samples {
        let split = assign_split(&s.formula_str, ratios, split_seed);
        formulas_in.entry(split).or_default().insert(s.formula_str.clone());

        if let Some(key) = s.structure_key {
            manifest.structure_keys.insert(key);
        }
        manifest.spectrum_keys.insert(s.spectrum_key);

        let sample = TrainingSample {
            features: s.features,
            fingerprint: s.fingerprint,
        };
        match split {
            Split::Train => train_samples.push(sample),
            Split::Validation => val_samples.push(sample),
            Split::Test => test_samples.push(sample),
        }
    }

    let n_formulas = |sp: Split| formulas_in.get(&sp).map_or(0, |f| f.len());
    println!(
        "  Formula-disjoint split (seed {}): {} train ({} formulas) | {} val ({} formulas) | {} test ({} formulas)",
        split_seed,
        train_samples.len(),
        n_formulas(Split::Train),
        val_samples.len(),
        n_formulas(Split::Validation),
        test_samples.len(),
        n_formulas(Split::Test)
    );

    // ── 4. Train ResNet Fingerprint Predictor ────────────────────────────────
    println!("\nStep 4/5: Training Deep ResNet Fingerprint Predictor (Candle AdamW + Compound Loss)...");

    let train_config = TrainingConfig {
        learning_rate: lr,
        min_learning_rate: lr * 0.05,
        epochs,
        batch_size,
        weight_decay: 1e-3,
        tanimoto_loss_weight: 1.0,
        validation_fraction: 0.1,
        log_interval: 1,
        patience: 4,
        cosine_scheduler: true,
        seed: 42,
        dropout,
        feature_dropout,
    };

    let mut predictor = FingerprintPredictor::new_with_dims(TOTAL_FEATURES, hidden_dim, num_blocks, TOTAL_BITS);
    // Recorded before training, not after, so that no path can produce weights whose
    // fingerprint layout was left unstated.
    predictor.resnet.morgan_vocabulary = vocabulary.ids().to_vec();
    println!("  Model: ResNet-{} ({} Residual Blocks + LayerNorm + GELU, {} parameters)", hidden_dim, num_blocks, predictor.num_parameters());
    println!("  Loss:  Asymmetric Loss (ASL γ=2.0, m=0.05) + Soft-Tanimoto (λ=1.00)");
    println!("  Reg:   dropout {dropout:.2} (residual branches) + feature dropout {feature_dropout:.2} (occupied bins), weight decay 1e-3");

    let t_train = Instant::now();
    let history = oxidems_lib::sirius::fingerprint::training::train_with_val_set(
        &mut predictor,
        &train_samples,
        &val_samples,
        &train_config,
    );
    println!(
        "  Training completed in {:.2}s (Best Val Tanimoto: {:.4} at epoch {})",
        t_train.elapsed().as_secs_f64(),
        history.best_valid_tanimoto,
        history.best_epoch
    );

    // ── 4. Independent Test Evaluation ──────────────────────────────────────
    println!("\nStep 5/5: Evaluating on Held-Out Test Set (N = {})...", test_samples.len());

    let mut total_tan_05 = 0.0f64;
    let mut total_tan_03 = 0.0f64;
    let mut total_cont_tan = 0.0f64;
    let mut total_bit_acc = 0.0f64;
    let mut total_bce = 0.0f64;

    for sample in &test_samples {
        let predicted = predictor.predict(&sample.features);

        let mut sample_bce = 0.0f64;
        let mut bit_correct = 0usize;

        for i in 0..TOTAL_BITS {
            let p = (predicted.probabilities[i] as f64).clamp(1e-7, 1.0 - 1e-7);
            let t = if sample.fingerprint[i] { 1.0 } else { 0.0 };
            sample_bce -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();

            let pred_bit = p >= 0.5;
            if pred_bit == sample.fingerprint[i] {
                bit_correct += 1;
            }
        }
        total_bce += sample_bce / TOTAL_BITS as f64;
        total_bit_acc += bit_correct as f64 / TOTAL_BITS as f64;

        let candidate_fp = BinaryFingerprint::from_bits(sample.fingerprint.clone());
        total_tan_05 += FingerprintSearcher::tanimoto_similarity(&predicted, &candidate_fp, 0.5);
        total_tan_03 += FingerprintSearcher::tanimoto_similarity(&predicted, &candidate_fp, 0.3);
        total_cont_tan += FingerprintSearcher::continuous_tanimoto_similarity(&predicted, &candidate_fp);
    }

    let n_t = test_samples.len().max(1) as f64;
    let mean_bce = total_bce / n_t;
    let mean_bit_acc = (total_bit_acc / n_t) * 100.0;

    println!("-------------------------------------------------------------------------");
    println!("                     Test Set Performance Metrics                        ");
    println!("-------------------------------------------------------------------------");
    println!("  Mean Binary Cross-Entropy (BCE): {:.6}", mean_bce);
    println!("  Mean Bit-level Accuracy:         {:.2}%", mean_bit_acc);
    println!("  Mean Binary Tanimoto (th=0.5):   {:.4}", total_tan_05 / n_t);
    println!("  Mean Binary Tanimoto (th=0.3):   \x1b[32m{:.4}\x1b[0m", total_tan_03 / n_t);
    println!("  Mean Continuous Tanimoto:        \x1b[32m{:.4}\x1b[0m", total_cont_tan / n_t);
    println!("-------------------------------------------------------------------------");

    // ── 5. Save Trained Checkpoint ──────────────────────────────────────────
    if let Some(parent) = Path::new(&output_model).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out_file = File::create(&output_model).expect("Failed to create output model file");
    predictor.save_weights(&mut out_file).expect("Failed to write model weights");
    println!("\nSaved trained model checkpoint to: {}\n", output_model);

    // The manifest is what lets a later benchmark prove it is disjoint from this run.
    // Reconstructing the training set from the MSP would not do: it would have to
    // reproduce every filtering decision made above, and any drift shows up as a
    // benchmark that quietly scores compounds the model already knows.
    let manifest_path = format!("{output_model}.manifest");
    manifest
        .save(&manifest_path)
        .expect("Failed to write dataset manifest");
    println!(
        "Wrote dataset manifest to: {} ({} structures, {} spectra)\n",
        manifest_path,
        manifest.structure_keys.len(),
        manifest.spectrum_keys.len()
    );
}
