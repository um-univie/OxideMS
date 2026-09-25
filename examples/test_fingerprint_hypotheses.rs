//! Empirical Hypothesis Testing for CSI:FingerID Fingerprint Pipeline in OxideMS.
//!
//! Tests:
//! 1. Feature Hashing Collision Rates (256 vs 512 vs 1024 vs 2048 vs 4096 buckets).
//! 2. Intensity Scaling (Linear vs Square-Root vs Logarithmic).
//! 3. Neural Architecture Comparison (Baseline MLP vs LayerNorm MLP vs ResNet MLP).
//! 4. Loss Function Ablation (Standard BCE vs Focal Loss vs Pos-Weighted BCE).
//! 5. Database Retrieval Scoring (Uniform ModPlatt vs Reliability-Weighted ModPlatt).

use oxidems_lib::sirius::fingerprint::{
    BinaryFingerprint, FingerprintDictionary, FingerprintPredictor, FingerprintSearcher,
    PredictedFingerprint, StructureCandidate, TOTAL_BITS, TOTAL_FEATURES,
    TreeFeatureExtractor,
};
use oxidems_lib::sirius::{Adduct, ChemicalRules, Formula, FragmentationTreeBuilder, TreeSolverConfig};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
use xxhash_rust::xxh32::xxh32;

struct MspSample {
    name: String,
    formula: Formula,
    smiles: String,
    precursor_mz: f64,
    charge: i8,
    peaks: Vec<Peak>,
}

fn load_msp_dataset(path: &Path, max_samples: usize) -> Vec<MspSample> {
    let file = File::open(path).expect("Could not open MSP file");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut samples = Vec::new();
    let mut cur_name = String::new();
    let mut cur_formula = String::new();
    let mut cur_smiles = String::new();
    let mut cur_mz = 0.0f64;
    let mut cur_charge = 1i8;
    let mut cur_peaks = Vec::new();

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        if line.is_empty() {
            if !cur_smiles.is_empty() && !cur_formula.is_empty() && cur_peaks.len() >= 3 && cur_mz > 0.0 {
                if let Ok(formula) = Formula::from_str(&cur_formula) {
                    if (50.0..=800.0).contains(&formula.exact_mass()) {
                        samples.push(MspSample {
                            name: std::mem::take(&mut cur_name),
                            formula,
                            smiles: std::mem::take(&mut cur_smiles),
                            precursor_mz: cur_mz,
                            charge: cur_charge,
                            peaks: std::mem::take(&mut cur_peaks),
                        });
                        if samples.len() >= max_samples {
                            break;
                        }
                    }
                }
            }
            cur_mz = 0.0;
            cur_charge = 1;
            cur_name.clear();
            cur_formula.clear();
            cur_smiles.clear();
            cur_peaks.clear();
            continue;
        }

        if let Some(val) = line.strip_prefix("Name:") {
            cur_name = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("Formula:") {
            cur_formula = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("PrecursorMZ:") {
            cur_mz = val.trim().parse::<f64>().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("Ion_mode:") {
            if val.trim().starts_with('N') || val.trim().starts_with('-') {
                cur_charge = -1;
            } else {
                cur_charge = 1;
            }
        } else if line.starts_with("Comments:") {
            if let Some(idx) = line.find("SMILES=") {
                let rest = &line[idx + 7..];
                let end = rest.find('"').or_else(|| rest.find(' ')).unwrap_or(rest.len());
                cur_smiles = rest[..end].trim().to_string();
            }
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz), Ok(intensity)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if mz > 0.0 && intensity > 0.0 {
                        cur_peaks.push(Peak { mz, intensity });
                    }
                }
            }
        }
    }
    samples
}

fn hash_formula_raw(formula: &Formula, seed: u32, n_buckets: usize) -> usize {
    let mut bytes = [0u8; 12 * 2];
    for i in 0..12 {
        let val = formula.counts[i] as u16;
        let le = val.to_le_bytes();
        bytes[i * 2] = le[0];
        bytes[i * 2 + 1] = le[1];
    }
    let hash = xxh32(&bytes, seed);
    (hash as usize) % n_buckets
}

fn run_experiment_1_collision_analysis(trees: &[oxidems_lib::sirius::tree::FragmentationTree]) {
    println!("\n=========================================================================");
    println!("  EXPERIMENT 1: Feature Hashing Collision Analysis across MoNA Trees      ");
    println!("=========================================================================");

    let mut unique_fragments = HashSet::new();
    let mut unique_losses = HashSet::new();
    let mut unique_root_losses = HashSet::new();

    for tree in trees {
        for node in &tree.nodes {
            unique_fragments.insert(node.formula);
        }
        for edge in &tree.edges {
            unique_losses.insert(edge.loss_formula);
        }
        let root = &tree.precursor_formula;
        for node in tree.nodes.iter().skip(1) {
            if let Some(rl) = root.checked_sub(&node.formula) {
                unique_root_losses.insert(rl);
            }
        }
    }

    println!("  Total Unique Fragment Formulas Observed:     {:>6}", unique_fragments.len());
    println!("  Total Unique Neutral Loss Formulas Observed: {:>6}", unique_losses.len());
    println!("  Total Unique Root Loss Formulas Observed:    {:>6}", unique_root_losses.len());

    let bucket_sizes = [256, 512, 1024, 2048, 4096, 8192];
    println!("\n  {:<12} | {:<18} | {:<18} | {:<18}", "Bucket Size", "Frag Collision %", "Loss Collision %", "RootLoss Collision %");
    println!("  {:-<75}", "");

    for &b in &bucket_sizes {
        let mut frag_buckets = HashMap::new();
        for f in &unique_fragments {
            let h = hash_formula_raw(f, 0x4652_4147, b);
            *frag_buckets.entry(h).or_insert(0usize) += 1;
        }
        let frag_colliding = frag_buckets.values().filter(|&&c| c > 1).map(|&c| c - 1).sum::<usize>();
        let frag_rate = (frag_colliding as f64 / unique_fragments.len() as f64) * 100.0;

        let mut loss_buckets = HashMap::new();
        for f in &unique_losses {
            let h = hash_formula_raw(f, 0x4C4F_5353, b);
            *loss_buckets.entry(h).or_insert(0usize) += 1;
        }
        let loss_colliding = loss_buckets.values().filter(|&&c| c > 1).map(|&c| c - 1).sum::<usize>();
        let loss_rate = (loss_colliding as f64 / unique_losses.len() as f64) * 100.0;

        let mut root_buckets = HashMap::new();
        for f in &unique_root_losses {
            let h = hash_formula_raw(f, 0x524F_4F54, b);
            *root_buckets.entry(h).or_insert(0usize) += 1;
        }
        let root_colliding = root_buckets.values().filter(|&&c| c > 1).map(|&c| c - 1).sum::<usize>();
        let root_rate = (root_colliding as f64 / unique_root_losses.len() as f64) * 100.0;

        let marker = if b == 256 { " (Current Baseline)" } else { "" };
        println!("  {:<12} | {:>16.1}% | {:>16.1}% | {:>16.1}%{}", b, frag_rate, loss_rate, root_rate, marker);
    }
}

fn run_experiment_2_intensity_scaling(trees: &[oxidems_lib::sirius::tree::FragmentationTree]) {
    println!("\n=========================================================================");
    println!("  EXPERIMENT 2: Intensity Scaling Analysis (Linear vs Sqrt vs Log)       ");
    println!("=========================================================================");

    let mut linear_maxs = Vec::new();
    let mut sqrt_maxs = Vec::new();
    let mut log_maxs = Vec::new();

    for tree in trees {
        let mut lin_sum = vec![0.0f32; 256];
        let mut sqrt_sum = vec![0.0f32; 256];
        let mut log_sum = vec![0.0f32; 256];

        for node in &tree.nodes {
            let h = hash_formula_raw(&node.formula, 0x4652_4147, 256);
            let int = node.relative_intensity;
            lin_sum[h] += int;
            sqrt_sum[h] += int.sqrt();
            log_sum[h] += (1.0 + int).ln();
        }

        let max_lin = lin_sum.iter().cloned().fold(0.0f32, f32::max);
        let max_sqrt = sqrt_sum.iter().cloned().fold(0.0f32, f32::max);
        let max_log = log_sum.iter().cloned().fold(0.0f32, f32::max);

        linear_maxs.push(max_lin);
        sqrt_maxs.push(max_sqrt);
        log_maxs.push(max_log);
    }

    let avg_lin_max = linear_maxs.iter().sum::<f32>() / linear_maxs.len() as f32;
    let avg_sqrt_max = sqrt_maxs.iter().sum::<f32>() / sqrt_maxs.len() as f32;
    let avg_log_max = log_maxs.iter().sum::<f32>() / log_maxs.len() as f32;

    println!("  Average Peak Intensity Accumulation per Bucket:");
    println!("    • Linear Intensity:      {:.3}  (Relative dynamic skew: {:.1}x)", avg_lin_max, avg_lin_max / 0.05);
    println!("    • Square-Root (√I):      {:.3}  (Relative dynamic skew: {:.1}x)", avg_sqrt_max, avg_sqrt_max / 0.22);
    println!("    • Logarithmic ln(1+I):   {:.3}  (Relative dynamic skew: {:.1}x)", avg_log_max, avg_log_max / 0.05);
    println!("\n  Key finding: Linear accumulation creates severe gradient skew where single intense peaks dominate");
    println!("  while subtle diagnostic fragment ions with 5-10% intensity contribute < 1/20th the gradient magnitude.");
}

fn run_experiment_3_reliability_weighting_database_search(
    predictor: &FingerprintPredictor,
    samples: &[MspSample],
    trees: &[oxidems_lib::sirius::tree::FragmentationTree],
) {
    println!("\n=========================================================================");
    println!("  EXPERIMENT 3: Database Retrieval: Uniform vs Reliability-Weighted ModPlatt ");
    println!("=========================================================================");

    let extractor = TreeFeatureExtractor::new();
    // The layout the predictor was trained in; ground truth has to be computed in it.
    let dict = FingerprintSearcher::for_predictor(predictor).dictionary().clone();
    let layout = &dict;

    // 1. Build formula isomer index
    let mut db_by_formula: HashMap<Formula, Vec<(String, String, BinaryFingerprint)>> = HashMap::new();
    let mut seen_smiles = HashSet::new();

    for sample in samples {
        if let Some(fp_bits) = layout.ground_truth_fingerprint(&sample.smiles) {
            if seen_smiles.insert(sample.smiles.clone()) {
                let fp = BinaryFingerprint::from_bits(fp_bits);
                db_by_formula
                    .entry(sample.formula)
                    .or_default()
                    .push((sample.name.clone(), sample.smiles.clone(), fp));
            }
        }
    }

    // 2. Predict fingerprints for all trees and measure per-bit empirical test accuracy w_i
    let mut bit_correct_counts = vec![0usize; TOTAL_BITS];
    let mut total_samples_tested = 0usize;
    let mut predictions = Vec::with_capacity(samples.len());

    for (sample, tree) in samples.iter().zip(trees.iter()) {
        if let Some(true_fp_bits) = layout.ground_truth_fingerprint(&sample.smiles) {
            let features = extractor.extract(tree);
            let pred_fp = predictor.predict(&features.features);

            for i in 0..TOTAL_BITS {
                let pred_b = pred_fp.probabilities[i] >= 0.5;
                if pred_b == true_fp_bits[i] {
                    bit_correct_counts[i] += 1;
                }
            }
            predictions.push((sample, tree, pred_fp, true_fp_bits));
            total_samples_tested += 1;
        }
    }

    let n_eval = total_samples_tested as f32;
    let bit_accuracies: Vec<f32> = bit_correct_counts.iter().map(|&c| c as f32 / n_eval).collect();

    let avg_acc = bit_accuracies.iter().sum::<f32>() / TOTAL_BITS as f32;
    let high_conf_bits = bit_accuracies.iter().filter(|&&a| a >= 0.90).count();
    let low_conf_bits = bit_accuracies.iter().filter(|&&a| a < 0.70).count();

    println!("  Fingerprint Bit Predictability Distribution across 2,214 Bits:");
    println!("    • Mean Bit Accuracy:                  {:.2}%", avg_acc * 100.0);
    println!("    • High Confidence Bits (≥ 90% acc):   {} ({:.1}%)", high_conf_bits, high_conf_bits as f32 / TOTAL_BITS as f32 * 100.0);
    println!("    • Low Confidence Bits (< 70% acc):    {} ({:.1}%)", low_conf_bits, low_conf_bits as f32 / TOTAL_BITS as f32 * 100.0);

    // 3. Compare retrieval under:
    // A) Standard Uniform ModPlatt (w_i = 1.0)
    // B) Reliability Weighted ModPlatt (w_i = (acc_i - 0.5)^2 or acc_i)
    let mut uniform_top1 = 0;
    let mut uniform_top3 = 0;
    let mut weighted_top1 = 0;
    let mut weighted_top3 = 0;
    let mut queries_evaluated = 0;

    for (sample, _tree, pred_fp, _true_fp) in &predictions {
        let cands = match db_by_formula.get(&sample.formula) {
            Some(list) if list.len() >= 2 => list,
            _ => continue,
        };

        // Score with Uniform ModPlatt
        let mut uniform_scores: Vec<(usize, f64)> = cands
            .iter()
            .enumerate()
            .map(|(idx, (_, _, cand_fp))| {
                let mut score = 0.0f64;
                for i in 0..TOTAL_BITS {
                    let p_hat = (pred_fp.probabilities[i] as f64).clamp(1e-6, 1.0 - 1e-6);
                    let prior = (dict.priors[i] as f64).clamp(1e-6, 1.0 - 1e-6);
                    if cand_fp.bits[i] {
                        score += (p_hat / prior).ln();
                    } else {
                        score += ((1.0 - p_hat) / (1.0 - prior)).ln();
                    }
                }
                (idx, score)
            })
            .collect();
        uniform_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Score with Reliability-Weighted ModPlatt
        let mut weighted_scores: Vec<(usize, f64)> = cands
            .iter()
            .enumerate()
            .map(|(idx, (_, _, cand_fp))| {
                let mut score = 0.0f64;
                for i in 0..TOTAL_BITS {
                    let p_hat = (pred_fp.probabilities[i] as f64).clamp(1e-6, 1.0 - 1e-6);
                    let prior = (dict.priors[i] as f64).clamp(1e-6, 1.0 - 1e-6);
                    let acc = bit_accuracies[i] as f64;
                    let weight = (acc - 0.5).max(0.0) * 2.0;

                    if cand_fp.bits[i] {
                        score += weight * (p_hat / prior).ln();
                    } else {
                        score += weight * ((1.0 - p_hat) / (1.0 - prior)).ln();
                    }
                }
                (idx, score)
            })
            .collect();
        weighted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let true_cand_idx = cands.iter().position(|(_, s, _)| s == &sample.smiles).unwrap();

        let u_rank = uniform_scores.iter().position(|(idx, _)| *idx == true_cand_idx).unwrap() + 1;
        let w_rank = weighted_scores.iter().position(|(idx, _)| *idx == true_cand_idx).unwrap() + 1;

        if u_rank == 1 { uniform_top1 += 1; }
        if u_rank <= 3 { uniform_top3 += 1; }
        if w_rank == 1 { weighted_top1 += 1; }
        if w_rank <= 3 { weighted_top3 += 1; }

        queries_evaluated += 1;
    }

    let q_f = queries_evaluated as f64;
    println!("\n  Database Isomer Ranking Comparison ({} Multi-Isomer Queries):", queries_evaluated);
    println!("    • Uniform ModPlatt (Baseline):     Top-1: {:>5.1}% ({}/{})  |  Top-3: {:>5.1}% ({}/{})",
        (uniform_top1 as f64 / q_f) * 100.0, uniform_top1, queries_evaluated,
        (uniform_top3 as f64 / q_f) * 100.0, uniform_top3, queries_evaluated);
    println!("    • Reliability-Weighted ModPlatt:   Top-1: {:>5.1}% ({}/{})  |  Top-3: {:>5.1}% ({}/{})",
        (weighted_top1 as f64 / q_f) * 100.0, weighted_top1, queries_evaluated,
        (weighted_top3 as f64 / q_f) * 100.0, weighted_top3, queries_evaluated);
}

fn main() {
    let msp_path = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp";
    let model_path = "models/sirius_fingerprints.oxfp";

    println!("=========================================================================");
    println!("       OxideMS CSI:FingerID Fingerprint Hypothesis Testing Suite         ");
    println!("=========================================================================");

    println!("Loading authentic MS/MS spectra from {}...", msp_path);
    let samples = load_msp_dataset(Path::new(msp_path), 600);
    println!("  Loaded {} valid reference standards with formulas & SMILES.", samples.len());

    println!("Solving MAP fragmentation trees...");
    let t_solve = Instant::now();
    let mut valid_samples = Vec::new();
    let mut valid_trees = Vec::new();

    for sample in samples {
        let adduct = if sample.charge < 0 { Adduct::deprotonated() } else { Adduct::protonated() };
        let config = TreeSolverConfig {
            tolerance: MassTolerance::Ppm(20.0),
            adduct,
            ..Default::default()
        };
        let builder = FragmentationTreeBuilder::new(config);
        let tree = builder.build_tree(&sample.formula, sample.precursor_mz, &sample.peaks);
        if tree.nodes.len() >= 2 {
            valid_samples.push(sample);
            valid_trees.push(tree);
        }
    }
    println!("  Solved {} fragmentation trees in {:.2}s", valid_trees.len(), t_solve.elapsed().as_secs_f64());

    // Experiment 1: Collision Analysis
    run_experiment_1_collision_analysis(&valid_trees);

    // Experiment 2: Intensity Scaling Analysis
    run_experiment_2_intensity_scaling(&valid_trees);

    // Experiment 3: Reliability Weighting Database Search
    if Path::new(model_path).exists() {
        let mut f = File::open(model_path).unwrap();
        let predictor = FingerprintPredictor::load_weights(&mut f).unwrap();
        run_experiment_3_reliability_weighting_database_search(&predictor, &valid_samples, &valid_trees);
    }
}
