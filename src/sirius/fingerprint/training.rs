//! Training pipeline for the fingerprint prediction ResNet using Candle.
//!
//! Implements mini-batch training with AdamW, Cosine Annealing learning rate schedule,
//! and compound loss (Binary Cross-Entropy + Differentiable Soft-Tanimoto Loss).
//!
//! # Training Data
//!
//! Training samples consist of:
//! - **Input**: Tree feature vectors (5,147-d) extracted by [`TreeFeatureExtractor`]
//! - **Target**: Ground-truth binary fingerprints (2,214 bits) computed from known molecular
//!   structures (SMILES) using `oxidane`

use super::dictionary::{
    FingerprintDictionary, MACCS_BITS, MACCS_OFFSET, MORGAN_BITS, MORGAN_OFFSET, TOTAL_BITS,
};
use super::features::{TreeFeatureExtractor, TOTAL_FEATURES};
use super::predictor::{FingerprintPredictor, FingerprintResNet, PredictedFingerprint};
use super::search::FingerprintSearcher;
use crate::sirius::formula::Formula;
use crate::sirius::solver::FragmentationTreeBuilder;
use crate::spectrum::Peak;
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

/// Generates a training sample from an MS/MS spectrum and known SMILES structure.
///
/// `dictionary` states which fingerprint layout the ground truth is computed in; it has
/// to be the same one the resulting model will be scored under.
pub fn generate_training_sample(
    tree_builder: &FragmentationTreeBuilder,
    extractor: &TreeFeatureExtractor,
    dictionary: &FingerprintDictionary,
    smiles: &str,
    formula: &Formula,
    precursor_mz: f64,
    peaks: &[Peak],
) -> Option<TrainingSample> {
    let fp = dictionary.ground_truth_fingerprint(smiles)?;
    let tree = tree_builder.build_tree(formula, precursor_mz, peaks);
    let features = extractor.extract(&tree).features;
    Some(TrainingSample {
        features,
        fingerprint: fp,
    })
}

// ── Training sample ─────────────────────────────────────────────────────────

/// A single training sample for fingerprint prediction.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Tree feature vector of length [`TOTAL_FEATURES`].
    pub features: Vec<f32>,
    /// Ground-truth binary fingerprint of length [`TOTAL_BITS`].
    pub fingerprint: Vec<bool>,
}

// ── Training configuration ──────────────────────────────────────────────────

/// Configuration for the training loop.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Initial learning rate for AdamW optimizer.
    pub learning_rate: f64,
    /// Minimum learning rate for cosine annealing schedule.
    pub min_learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Weight decay for AdamW L2 regularization.
    pub weight_decay: f64,
    /// Weight for Soft-Tanimoto Loss (beta).
    pub tanimoto_loss_weight: f64,
    /// Random seed.
    pub seed: u64,
    /// Print progress every N epochs (0 = silent).
    pub log_interval: usize,
    /// Fraction of samples to hold out for validation.
    pub validation_fraction: f32,
    /// Early stopping patience (stop if validation Tanimoto fails to improve for N epochs, 0 = disabled).
    pub patience: usize,
    /// Whether to use cosine annealing learning rate schedule.
    pub cosine_scheduler: bool,
    /// Probability of dropping each hidden unit inside a residual branch during
    /// training (0 = disabled, the default).
    ///
    /// The network has far more parameters than the training set has spectra, so this
    /// looks like it should help, but measurement says otherwise: at 0.1 it left top-1
    /// retrieval on 100 MoNA isomer queries unchanged (73%), slightly lowered test
    /// Tanimoto, and cost 36% more training time by suppressing early stopping. Kept as
    /// a knob for larger training sets, off until it earns its place.
    pub dropout: f32,
    /// Probability of zeroing each *non-zero* input feature during training
    /// (0 = disabled, the default — see [`dropout`](Self::dropout) for why).
    ///
    /// Zeroing feature dimensions uniformly would be nearly a no-op: the 5,147-d tree
    /// feature vector is sparse, so most dimensions are already zero. Dropping the
    /// occupied ones instead simulates the fragment that a noisier spectrum would not
    /// have produced, which is the variation the model actually has to survive.
    pub feature_dropout: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            min_learning_rate: 1e-5,
            epochs: 40,
            batch_size: 128,
            weight_decay: 1e-3,
            tanimoto_loss_weight: 1.0,
            seed: 42,
            log_interval: 1,
            validation_fraction: 0.1,
            patience: 4,
            cosine_scheduler: true,
            dropout: 0.0,
            feature_dropout: 0.0,
        }
    }
}

// ── Training history ────────────────────────────────────────────────────────

/// Training progress record.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Per-epoch average training loss.
    pub train_losses: Vec<f32>,
    /// Per-epoch average validation loss.
    pub valid_losses: Vec<f32>,
    /// Per-epoch validation Tanimoto similarity.
    pub valid_tanimotos: Vec<f32>,
    /// Best validation loss achieved.
    pub best_valid_loss: f32,
    /// Best validation Tanimoto achieved.
    pub best_valid_tanimoto: f32,
    /// Epoch at which best validation loss was achieved.
    pub best_epoch: usize,
}

// ── Compound Loss Calculation (Asymmetric Loss + Soft-Tanimoto) ─────────────

fn compute_compound_loss(
    pred: &Tensor,
    target: &Tensor,
    tanimoto_weight: f64,
) -> CandleResult<Tensor> {
    let eps = 1e-7f64;

    let p = pred.clamp(eps, 1.0 - eps)?;
    let y = target;
    let one_minus_y = y.affine(-1.0, 1.0)?;

    // Positive loss: - y * log(p)
    let log_p = p.log()?;
    let loss_pos = (y * &log_p)?.neg()?;

    // Asymmetric negative loss with margin shifting (m=0.05) and focal attenuation:
    // When p <= 0.05, p_shifted = 0, so the negative loss and gradient are zeroed out!
    // This stops 98% negative background bits from drowning out rare positive functional groups.
    let p_shifted = p.affine(1.0, -0.05)?.clamp(0.0, 1.0)?;
    let one_minus_pm = p_shifted.affine(-1.0, 1.0)?.clamp(eps, 1.0)?;
    let log_one_minus_pm = one_minus_pm.log()?;
    let focal_neg = (&p_shifted * &p_shifted)?; // gamma_neg = 2.0
    let loss_neg = (((&one_minus_y * &focal_neg)? * &log_one_minus_pm))?.neg()?;

    let asl_loss = (&loss_pos + &loss_neg)?.mean_all()?;

    // Soft Tanimoto Loss (Intersection-over-Union directly optimized)
    let intersection = (pred * y)?.sum_keepdim(1)?;
    let sum_p = pred.sum_keepdim(1)?;
    let sum_y = y.sum_keepdim(1)?;
    let union = ((&sum_p + &sum_y)? - &intersection)?.affine(1.0, eps)?;
    let soft_tan = (&intersection / &union)?;
    let tan_loss = soft_tan.affine(-1.0, 1.0)?.mean_all()?;

    let total = (&asl_loss + (&tan_loss * tanimoto_weight)?)?;
    Ok(total)
}

// ── Training loop ───────────────────────────────────────────────────────────

/// Trains the `FingerprintResNet` model with explicitly supplied train and validation sets.
pub fn train_resnet_with_val_set(
    model: &mut FingerprintResNet,
    train_samples: &[TrainingSample],
    valid_samples: &[TrainingSample],
    config: &TrainingConfig,
) -> TrainingHistory {
    assert!(!train_samples.is_empty(), "Training set must not be empty");

    let device = model.device.clone();

    let opt_params = candle_nn::ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(model.varmap.all_vars(), opt_params)
        .expect("Failed to create AdamW optimizer");

    let mut history = TrainingHistory {
        train_losses: Vec::with_capacity(config.epochs),
        valid_losses: Vec::with_capacity(config.epochs),
        valid_tanimotos: Vec::with_capacity(config.epochs),
        best_valid_loss: f32::INFINITY,
        best_valid_tanimoto: 0.0,
        best_epoch: 0,
    };

    let in_dim = model.in_dim;
    let out_dim = model.out_dim;

    let mut best_weights: Option<Vec<u8>> = None;
    let mut patience_counter = 0usize;

    // Batches are reshuffled every epoch: a fixed sample order makes every epoch see the
    // exact same gradients batch for batch, which correlates the updates and wastes the
    // stochasticity AdamW relies on.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order: Vec<usize> = (0..train_samples.len()).collect();

    for epoch in 1..=config.epochs {
        // Cosine Annealing Learning Rate Schedule
        let current_lr = if config.cosine_scheduler && config.epochs > 1 {
            let progress = (epoch - 1) as f64 / (config.epochs - 1) as f64;
            config.min_learning_rate
                + 0.5
                    * (config.learning_rate - config.min_learning_rate)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
        } else {
            config.learning_rate
        };
        opt.set_learning_rate(current_lr);

        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0;

        order.shuffle(&mut rng);

        for chunk in order.chunks(config.batch_size) {
            let b_size = chunk.len();
            let mut flat_x = Vec::with_capacity(b_size * in_dim);
            let mut flat_y = Vec::with_capacity(b_size * out_dim);

            for &idx in chunk {
                let s = &train_samples[idx];
                let feat_start = flat_x.len();
                flat_x.extend_from_slice(&s.features);
                if config.feature_dropout > 0.0 {
                    let p = config.feature_dropout as f64;
                    for v in &mut flat_x[feat_start..] {
                        if *v != 0.0 && rng.random_bool(p) {
                            *v = 0.0;
                        }
                    }
                }
                for &b in &s.fingerprint {
                    flat_y.push(if b { 1.0f32 } else { 0.0f32 });
                }
            }

            let batch_x = Tensor::from_vec(flat_x, (b_size, in_dim), &device).unwrap();
            let batch_y = Tensor::from_vec(flat_y, (b_size, out_dim), &device).unwrap();

            let pred = model.forward_train(&batch_x, config.dropout).unwrap();
            let loss =
                compute_compound_loss(&pred, &batch_y, config.tanimoto_loss_weight).unwrap();

            opt.backward_step(&loss).unwrap();

            epoch_loss += loss.to_scalar::<f32>().unwrap() as f64;
            batch_count += 1;
        }

        let avg_train_loss = (epoch_loss / batch_count as f64) as f32;
        history.train_losses.push(avg_train_loss);

        // Validation evaluation in batches
        if !valid_samples.is_empty() {
            let mut val_loss_sum = 0.0f64;
            let mut val_batch_count = 0;
            let mut total_tan = 0.0f64;
            let v_batch_size = 256;

            for v_chunk in valid_samples.chunks(v_batch_size) {
                let vb_size = v_chunk.len();
                let mut flat_vx = Vec::with_capacity(vb_size * in_dim);
                let mut flat_vy = Vec::with_capacity(vb_size * out_dim);

                for s in v_chunk {
                    flat_vx.extend_from_slice(&s.features);
                    for &b in &s.fingerprint {
                        flat_vy.push(if b { 1.0f32 } else { 0.0f32 });
                    }
                }

                let v_batch_x = Tensor::from_vec(flat_vx, (vb_size, in_dim), &device).unwrap();
                let v_batch_y = Tensor::from_vec(flat_vy, (vb_size, out_dim), &device).unwrap();

                let v_pred = model.forward(&v_batch_x).unwrap();
                let v_loss =
                    compute_compound_loss(&v_pred, &v_batch_y, config.tanimoto_loss_weight).unwrap();
                val_loss_sum += v_loss.to_scalar::<f32>().unwrap() as f64;
                val_batch_count += 1;

                let pred_2d = v_pred.to_vec2::<f32>().unwrap();
                for (i, s) in v_chunk.iter().enumerate() {
                    let p_fp = PredictedFingerprint {
                        probabilities: pred_2d[i].clone(),
                    };
                    let t_fp = super::dictionary::BinaryFingerprint::from_bits(s.fingerprint.clone());
                    total_tan += FingerprintSearcher::tanimoto_similarity(&p_fp, &t_fp, 0.5);
                }
            }

            let val_loss_val = (val_loss_sum / val_batch_count as f64) as f32;
            history.valid_losses.push(val_loss_val);

            let mean_tan = (total_tan / valid_samples.len() as f64) as f32;
            history.valid_tanimotos.push(mean_tan);

            let is_best = mean_tan > history.best_valid_tanimoto;
            if is_best {
                history.best_valid_tanimoto = mean_tan;
                history.best_valid_loss = val_loss_val;
                history.best_epoch = epoch;
                patience_counter = 0;

                let mut buf = Vec::new();
                model.save_weights(&mut buf).unwrap();
                best_weights = Some(buf);
            } else {
                patience_counter += 1;
            }

            if config.log_interval > 0 && (epoch % config.log_interval == 0 || epoch == config.epochs) {
                let marker = if is_best { " [BEST]" } else { "" };
                println!(
                    "  [Epoch {:>2}/{}] LR: {:.2e} | Train Loss: {:.6} | Val Loss: {:.6} | Val Tanimoto: {:.4}{}",
                    epoch, config.epochs, current_lr, avg_train_loss, val_loss_val, mean_tan, marker
                );
            }

            if config.patience > 0 && patience_counter >= config.patience {
                println!(
                    "  Early stopping triggered at epoch {} (validation metric did not improve for {} epochs). Restoring checkpoint from epoch {}.",
                    epoch, config.patience, history.best_epoch
                );
                break;
            }
        }
    }

    if let Some(buf) = best_weights {
        if let Ok(loaded) = FingerprintResNet::load_weights(&mut &buf[..], device) {
            *model = loaded;
        }
    }

    // Record the training-set bit frequencies on the model. ModPlatt database search
    // needs them for its prior correction, and they are only knowable here; without
    // them the searcher falls back to uniform 0.5 priors and the correction vanishes.
    // Set after the checkpoint restore, which replaces the whole model.
    if out_dim == TOTAL_BITS {
        let mut counts = vec![0usize; TOTAL_BITS];
        for s in train_samples {
            for (i, &bit) in s.fingerprint.iter().enumerate() {
                if bit {
                    counts[i] += 1;
                }
            }
        }
        let mut dict = FingerprintDictionary::new();
        dict.set_priors_from_counts(&counts, train_samples.len());
        model.bit_priors = dict.priors;

        // Measure how well the restored model predicts each bit, on data it was not
        // fitted to. Database search weights each bit by this, so bits the model never
        // learned stop contributing confident noise to the ranking. Like the priors,
        // this is only knowable here.
        if !valid_samples.is_empty() {
            let weights = measure_bit_reliability(model, valid_samples);
            let live = weights.iter().filter(|&&w| w > 0.0).count();
            println!(
                "  Per-bit reliability measured on {} validation spectra: {}/{} bits carry signal (MCC > 0).",
                valid_samples.len(),
                live,
                TOTAL_BITS
            );
            model.bit_weights = weights;

            // Fit the Platt step the scoring function is named for. The network's raw
            // outputs are systematically low — the loss is asymmetric and 96% of bits
            // are unset — so the probabilities ModPlatt consumes are on the wrong
            // scale. This is the last thing done, on the restored checkpoint, using
            // data the optimizer never touched.
            let platt = fit_platt_calibration(model, valid_samples);
            let fitted = platt.chunks_exact(2).filter(|c| c != &[1.0, 0.0]).count();
            println!(
                "  Platt calibration fitted on {} validation spectra: {}/{} bits recalibrated.",
                valid_samples.len(),
                fitted,
                TOTAL_BITS
            );
            model.bit_platt = platt;
        }
    }

    history
}

/// Fits one logistic recalibration per fingerprint bit on held-out data.
///
/// Returns `[a₀, b₀, a₁, b₁, …]` for `σ(aᵢ · logit(p) + bᵢ)`; see
/// [`FingerprintResNet::bit_platt`]. Bits that are constant in the validation set get
/// the identity `(1, 0)`, since there is nothing to fit and guessing would be worse
/// than leaving the network's own estimate alone.
///
/// Targets are smoothed as in Lin, Lin & Weng (2007) — `(N₊+1)/(N₊+2)` and
/// `1/(N₋+2)` instead of 1 and 0 — which keeps the fit finite when a bit is nearly
/// separable, the common case here.
fn fit_platt_calibration(model: &FingerprintResNet, valid_samples: &[TrainingSample]) -> Vec<f32> {
    let device = model.device.clone();
    let in_dim = model.in_dim;
    let n = valid_samples.len();

    // Raw probabilities for the whole validation set, bit-major so each bit's fit
    // reads a contiguous slice.
    let mut by_bit = vec![0.0f32; TOTAL_BITS * n];
    let mut labels = vec![false; TOTAL_BITS * n];

    let mut offset = 0usize;
    for chunk in valid_samples.chunks(256) {
        let mut flat_x = Vec::with_capacity(chunk.len() * in_dim);
        for s in chunk {
            flat_x.extend_from_slice(&s.features);
        }
        let batch_x = Tensor::from_vec(flat_x, (chunk.len(), in_dim), &device).unwrap();
        let probs = model.forward(&batch_x).unwrap().to_vec2::<f32>().unwrap();

        for (row, s) in probs.iter().zip(chunk) {
            for i in 0..TOTAL_BITS {
                by_bit[i * n + offset] = row[i];
                labels[i * n + offset] = s.fingerprint[i];
            }
            offset += 1;
        }
    }

    let coefficients: Vec<[f32; 2]> = (0..TOTAL_BITS)
        .into_par_iter()
        .map(|i| {
            let probs = &by_bit[i * n..(i + 1) * n];
            let ys = &labels[i * n..(i + 1) * n];
            fit_one_bit(probs, ys)
        })
        .collect();

    coefficients.into_iter().flatten().collect()
}

/// Newton fit of `σ(a·x + b)` against smoothed targets, where `x = logit(p)`.
fn fit_one_bit(probs: &[f32], ys: &[bool]) -> [f32; 2] {
    const IDENTITY: [f32; 2] = [1.0, 0.0];

    let n_pos = ys.iter().filter(|&&y| y).count();
    let n_neg = ys.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return IDENTITY;
    }

    let hi = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
    let lo = 1.0 / (n_neg as f64 + 2.0);

    let xs: Vec<f64> = probs
        .iter()
        .map(|&p| {
            let c = (p as f64).clamp(1e-6, 1.0 - 1e-6);
            (c / (1.0 - c)).ln()
        })
        .collect();
    let ts: Vec<f64> = ys.iter().map(|&y| if y { hi } else { lo }).collect();

    let nll = |a: f64, b: f64| -> f64 {
        xs.iter()
            .zip(&ts)
            .map(|(&x, &t)| {
                let f = a * x + b;
                // log(1 + e^f) computed without overflowing on large |f|.
                let softplus = if f > 0.0 { f + (-f).exp().ln_1p() } else { f.exp().ln_1p() };
                softplus - t * f
            })
            .sum()
    };

    let (mut a, mut b) = (1.0f64, 0.0f64);
    let mut current = nll(a, b);

    for _ in 0..40 {
        let (mut g_a, mut g_b) = (0.0f64, 0.0f64);
        let (mut h_aa, mut h_ab, mut h_bb) = (0.0f64, 0.0f64, 0.0f64);

        for (&x, &t) in xs.iter().zip(&ts) {
            let p = 1.0 / (1.0 + (-(a * x + b)).exp());
            let d = p - t;
            let w = p * (1.0 - p);
            g_a += d * x;
            g_b += d;
            h_aa += w * x * x;
            h_ab += w * x;
            h_bb += w;
        }

        // Ridge on the diagonal: many bits are nearly separable, which drives the
        // Hessian toward singular.
        h_aa += 1e-9;
        h_bb += 1e-9;

        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-12 || !det.is_finite() {
            break;
        }
        let step_a = (h_bb * g_a - h_ab * g_b) / det;
        let step_b = (h_aa * g_b - h_ab * g_a) / det;

        // Backtracking: an undamped Newton step can overshoot badly on separable bits.
        let mut scale = 1.0f64;
        let mut improved = false;
        for _ in 0..10 {
            let candidate = nll(a - scale * step_a, b - scale * step_b);
            if candidate < current {
                a -= scale * step_a;
                b -= scale * step_b;
                current = candidate;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved || (step_a.abs() + step_b.abs()) < 1e-8 {
            break;
        }
    }

    if !a.is_finite() || !b.is_finite() {
        return IDENTITY;
    }
    // A calibration this extreme is a fit to noise, not a correction.
    [
        (a.clamp(0.05, 20.0)) as f32,
        (b.clamp(-20.0, 20.0)) as f32,
    ]
}

/// Scores every fingerprint bit by how well `model` predicts it on held-out data.
///
/// Returns the per-bit weights that database search will scale ModPlatt contributions
/// by; see [`FingerprintDictionary::set_weights_from_confusion`] for what they mean.
fn measure_bit_reliability(
    model: &FingerprintResNet,
    valid_samples: &[TrainingSample],
) -> Vec<f32> {
    let device = model.device.clone();
    let in_dim = model.in_dim;

    let mut tp = vec![0usize; TOTAL_BITS];
    let mut fp = vec![0usize; TOTAL_BITS];
    let mut tn = vec![0usize; TOTAL_BITS];
    let mut fn_ = vec![0usize; TOTAL_BITS];

    for chunk in valid_samples.chunks(256) {
        let mut flat_x = Vec::with_capacity(chunk.len() * in_dim);
        for s in chunk {
            flat_x.extend_from_slice(&s.features);
        }
        let batch_x = Tensor::from_vec(flat_x, (chunk.len(), in_dim), &device).unwrap();
        let probs = model.forward(&batch_x).unwrap().to_vec2::<f32>().unwrap();

        for (row, s) in probs.iter().zip(chunk) {
            for i in 0..TOTAL_BITS {
                match (row[i] >= 0.5, s.fingerprint[i]) {
                    (true, true) => tp[i] += 1,
                    (true, false) => fp[i] += 1,
                    (false, false) => tn[i] += 1,
                    (false, true) => fn_[i] += 1,
                }
            }
        }
    }

    let mut dict = FingerprintDictionary::new();
    dict.set_weights_from_confusion(&tp, &fp, &tn, &fn_);
    dict.weights
}

/// Trains the `FingerprintResNet` model using an internal validation split.
pub fn train_resnet(
    model: &mut FingerprintResNet,
    samples: &[TrainingSample],
    config: &TrainingConfig,
) -> TrainingHistory {
    assert!(!samples.is_empty(), "Training set must not be empty");
    let n_valid = (samples.len() as f32 * config.validation_fraction) as usize;
    let n_train = samples.len() - n_valid;
    train_resnet_with_val_set(model, &samples[..n_train], &samples[n_train..], config)
}

/// Trains the unified `FingerprintPredictor` using Candle with an explicit validation set.
pub fn train_with_val_set(
    predictor: &mut FingerprintPredictor,
    train_samples: &[TrainingSample],
    valid_samples: &[TrainingSample],
    config: &TrainingConfig,
) -> TrainingHistory {
    let history = train_resnet_with_val_set(&mut predictor.resnet, train_samples, valid_samples, config);
    predictor.fast_engine = std::sync::Arc::new(super::predictor::FastResNetInference::from_resnet(&predictor.resnet));
    history
}

/// Trains the unified `FingerprintPredictor` using Candle.
pub fn train(
    predictor: &mut FingerprintPredictor,
    samples: &[TrainingSample],
    config: &TrainingConfig,
) -> TrainingHistory {
    let history = train_resnet(&mut predictor.resnet, samples, config);
    predictor.fast_engine = std::sync::Arc::new(super::predictor::FastResNetInference::from_resnet(&predictor.resnet));
    history
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_fit_corrects_a_systematically_low_predictor() {
        // A bit that is set a third of the time, but whose raw probabilities all sit
        // well below the 0.5 threshold — exactly the failure the fit exists to undo.
        let probs: Vec<f32> = (0..900)
            .map(|i| if i % 3 == 0 { 0.30 } else { 0.05 })
            .collect();
        let ys: Vec<bool> = (0..900).map(|i| i % 3 == 0).collect();

        let [a, b] = fit_one_bit(&probs, &ys);

        let calibrated = |p: f32| {
            let c = p.clamp(1e-6, 1.0 - 1e-6);
            let logit = (c / (1.0 - c)).ln();
            1.0 / (1.0 + (-(a * logit + b)).exp())
        };

        assert!(
            calibrated(0.30) > 0.5,
            "a bit that is always set at p = 0.30 must calibrate above the threshold, got {}",
            calibrated(0.30)
        );
        assert!(
            calibrated(0.05) < 0.5,
            "a bit that is never set at p = 0.05 must stay below, got {}",
            calibrated(0.05)
        );
    }

    #[test]
    fn test_platt_fit_is_identity_for_a_constant_bit() {
        // Nothing to learn from a bit that is never set; the network's own estimate is
        // the best available answer.
        let probs = vec![0.1f32; 100];
        assert_eq!(fit_one_bit(&probs, &vec![false; 100]), [1.0, 0.0]);
        assert_eq!(fit_one_bit(&probs, &vec![true; 100]), [1.0, 0.0]);
    }

    #[test]
    fn test_training_records_bit_priors() {
        let mut model = FingerprintResNet::new();
        let mut fingerprint = vec![false; TOTAL_BITS];
        fingerprint[0] = true;

        let sample = TrainingSample {
            features: vec![0.5f32; TOTAL_FEATURES],
            fingerprint,
        };

        let config = TrainingConfig {
            epochs: 1,
            batch_size: 1,
            validation_fraction: 0.0,
            log_interval: 0,
            ..Default::default()
        };
        train_resnet(&mut model, &[sample], &config);

        // Laplace-smoothed over a single sample: set bit -> 2/3, unset bit -> 1/3.
        assert_eq!(model.bit_priors.len(), TOTAL_BITS);
        assert!((model.bit_priors[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((model.bit_priors[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_training_toy_step() {
        let mut model = FingerprintResNet::new();
        let sample = TrainingSample {
            features: vec![0.5f32; TOTAL_FEATURES],
            fingerprint: vec![true; TOTAL_BITS],
        };

        let config = TrainingConfig {
            epochs: 2,
            batch_size: 1,
            validation_fraction: 0.0,
            log_interval: 0,
            ..Default::default()
        };

        let history = train_resnet(&mut model, &[sample], &config);
        assert_eq!(history.train_losses.len(), 2);
    }
}
