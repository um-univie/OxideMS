//! Pure-Rust Deep ResNet Predictor for Molecular Fingerprints using Candle.
//!
//! Maps fragmentation tree feature vectors (5,147 dims) to predicted molecular
//! fingerprint probabilities (2,214 bits).
//!
//! # Architecture
//!
//! ```text
//! Input(5,147) ──→ Linear(5,147, 1024) ──→ LayerNorm ──→ GELU
//!                       │
//!         ┌─────────────┴─────────────┐
//!         ▼ (4 Residual Blocks)       │ (skip)
//!    LayerNorm ──→ Linear ──→ GELU ──→ Linear ──→ (+)
//!                       │
//!         LayerNorm ──→ Linear(1024, 2214) ──→ Sigmoid
//! ```
//!
//! # References
//! - Dührkop, K. et al. (2015). Searching molecular structure databases with tandem
//!   mass spectra using CSI:FingerID. PNAS, 112(41), 12580–12585.

use super::dictionary::TOTAL_BITS;
use super::features::TOTAL_FEATURES;
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder, VarMap};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

/// Magic bytes for weight file identification.
const WEIGHT_MAGIC: &[u8; 4] = b"OXFP";

/// Weight file format versions.
///
/// Each version appends a trailing block to the one before it, so loading an older
/// file simply stops early and leaves the newer statistics empty.
///
/// - `2`: ResNet with LayerNorm & GELU.
/// - `3`: same, plus training-set bit priors.
/// - `4`: same, plus per-bit reliability weights.
/// - `5`: same, plus per-bit Platt calibration coefficients.
/// - `6`: same, plus the Morgan vocabulary the fingerprints are defined by.
const WEIGHT_VERSION_VOCABULARY: u32 = 6;
const WEIGHT_VERSION_PLATT: u32 = 5;
const WEIGHT_VERSION_WEIGHTS: u32 = 4;
const WEIGHT_VERSION_PRIORS: u32 = 3;
const WEIGHT_VERSION_NO_PRIORS: u32 = 2;

/// Version written by [`FingerprintResNet::save_weights`].
const WEIGHT_VERSION_CURRENT: u32 = WEIGHT_VERSION_VOCABULARY;

/// Default hidden layer dimension.
pub const DEFAULT_HIDDEN_DIM: usize = 1024;
pub const DEFAULT_NUM_BLOCKS: usize = 4;

// ── Predicted fingerprint ───────────────────────────────────────────────────

/// Predicted molecular fingerprint as a vector of bit probabilities.
#[derive(Debug, Clone)]
pub struct PredictedFingerprint {
    /// Probability of each fingerprint bit being set, `P(bit=1) ∈ [0, 1]`.
    /// Length = [`TOTAL_BITS`].
    pub probabilities: Vec<f32>,
}

impl PredictedFingerprint {
    /// Converts to a hard binary fingerprint using a threshold (default 0.5).
    pub fn to_binary(&self, threshold: f32) -> Vec<bool> {
        self.probabilities.iter().map(|&p| p >= threshold).collect()
    }

    /// Returns the number of bits with probability above the given threshold.
    pub fn count_above(&self, threshold: f32) -> usize {
        self.probabilities.iter().filter(|&&p| p >= threshold).count()
    }


}

// ── ResNet Block Layer ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ResBlock {
    pub ln1: LayerNorm,
    pub fc1: Linear,
    pub ln2: LayerNorm,
    pub fc2: Linear,
}

impl ResBlock {
    pub fn new(dim: usize, vs: VarBuilder) -> CandleResult<Self> {
        let ln1 = layer_norm(dim, LayerNormConfig::default(), vs.pp("ln1"))?;
        let fc1 = linear(dim, dim, vs.pp("fc1"))?;
        let ln2 = layer_norm(dim, LayerNormConfig::default(), vs.pp("ln2"))?;
        let fc2 = linear(dim, dim, vs.pp("fc2"))?;
        Ok(Self { ln1, fc1, ln2, fc2 })
    }

    /// Runs the residual branch, optionally dropping `dropout` of the hidden units.
    ///
    /// Dropout applies inside the branch only, so the identity path always carries the
    /// full signal. Pass `0.0` at inference.
    pub fn forward(&self, x: &Tensor, dropout: f32) -> CandleResult<Tensor> {
        let h = self.fc1.forward(&self.ln1.forward(x)?)?.gelu()?;
        let h = if dropout > 0.0 {
            candle_nn::ops::dropout(&h, dropout)?
        } else {
            h
        };
        let h = self.fc2.forward(&self.ln2.forward(&h)?)?;
        x + h
    }
}

// ── Candle-based Fingerprint ResNet Model ────────────────────────────────────

#[derive(Clone)]
pub struct FingerprintResNet {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub out_dim: usize,

    pub in_proj: Linear,
    pub in_ln: LayerNorm,
    pub blocks: Vec<ResBlock>,
    pub out_ln: LayerNorm,
    pub out_head: Linear,

    /// Frequency of each fingerprint bit in the training set, stored alongside the
    /// weights because it is a product of the same training run. Database search needs
    /// it for the ModPlatt prior correction; the forward pass does not use it.
    /// Empty when the model was trained before priors were recorded.
    pub bit_priors: Vec<f32>,

    /// Per-bit reliability of this model, measured on the validation split at the end
    /// of the run that produced these weights. Database search scales each bit's
    /// ModPlatt contribution by it; the forward pass does not use it.
    /// Empty when the model was trained before reliability was recorded.
    pub bit_weights: Vec<f32>,

    /// Per-bit Platt calibration, as `[a₀, b₀, a₁, b₁, …]`: the raw sigmoid output for
    /// bit `i` maps to `σ(aᵢ · logit(p) + bᵢ)`. Fitted on the validation split.
    ///
    /// **Recorded, but not applied.** Calibration was fitted and benchmarked, and it
    /// made retrieval worse: on 500 GNPS queries drawn from compounds absent from the
    /// training library, top-1 fell from 48.6% to 47.8% with priors and from 47.2% to
    /// 45.0% without them, so it is not merely redundant with the prior correction.
    ///
    /// The reasoning that motivated it was wrong. Averaged over all bits, the mean
    /// probability assigned to bits that are set sits below 0.5, which looks like an
    /// under-confident predictor; but that average mixes bits whose base rates differ
    /// by orders of magnitude. Calibrating each bit against its own base rate correctly
    /// pulls the rare ones *down*, and the predicted bit count fell from 54.5 to 34.1
    /// against a true 76.1.
    ///
    /// Kept so the experiment stays reproducible — see [`apply_platt`], and
    /// `examples/evaluate_database_search.rs --platt`. A refinement worth trying is
    /// refusing to fit bits with too few positives in the validation split, which is
    /// the most likely source of the overfitting.
    ///
    /// Empty for models trained before calibration was fitted.
    pub bit_platt: Vec<f32>,

    /// The circular identifiers that own a Morgan bit in this model's fingerprints.
    ///
    /// Unlike the priors, weights and calibration, this is not a scoring parameter but
    /// part of what a bit *means*. A candidate fingerprint computed under a different
    /// vocabulary is not comparable with this model's predictions at all, so it travels
    /// with the weights and is installed into any dictionary used alongside them.
    ///
    /// Empty for models trained against plain modulo folding.
    pub morgan_vocabulary: Vec<u32>,

    pub varmap: VarMap,
    pub device: Device,
}

impl FingerprintResNet {
    /// Creates a new ResNet model initialized on the specified device.
    pub fn new_with_dims(
        in_dim: usize,
        hidden_dim: usize,
        num_blocks: usize,
        out_dim: usize,
        device: Device,
    ) -> CandleResult<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let in_proj = linear(in_dim, hidden_dim, vs.pp("in_proj"))?;
        let in_ln = layer_norm(hidden_dim, LayerNormConfig::default(), vs.pp("in_ln"))?;

        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(ResBlock::new(hidden_dim, vs.pp(format!("res_{i}")))?);
        }

        let out_ln = layer_norm(hidden_dim, LayerNormConfig::default(), vs.pp("out_ln"))?;
        let out_head = linear(hidden_dim, out_dim, vs.pp("out_head"))?;

        Ok(Self {
            in_dim,
            hidden_dim,
            num_blocks,
            out_dim,
            in_proj,
            in_ln,
            blocks,
            out_ln,
            out_head,
            bit_priors: Vec::new(),
            bit_weights: Vec::new(),
            bit_platt: Vec::new(),
            morgan_vocabulary: Vec::new(),
            varmap,
            device,
        })
    }

    /// Creates a new default ResNet predictor (5,147 -> 1024 -> 2,214).
    pub fn new() -> Self {
        Self::new_with_dims(
            TOTAL_FEATURES,
            DEFAULT_HIDDEN_DIM,
            DEFAULT_NUM_BLOCKS,
            TOTAL_BITS,
            Device::Cpu,
        )
        .expect("Failed to initialize default FingerprintResNet")
    }

    /// Forward pass through the ResNet network returning unnormalized logits.
    pub fn forward_logits(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward_logits_with_dropout(x, 0.0)
    }

    /// Forward pass returning Sigmoid probabilities.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let logits = self.forward_logits(x)?;
        candle_nn::ops::sigmoid(&logits)
    }

    /// Forward pass with dropout active in every residual branch.
    ///
    /// Only the training loop should call this, and only on training batches:
    /// dropout is a stochastic regularizer, so validation and inference must go through
    /// [`forward`](Self::forward), which runs the same network with `dropout = 0`.
    pub fn forward_train(&self, x: &Tensor, dropout: f32) -> CandleResult<Tensor> {
        let logits = self.forward_logits_with_dropout(x, dropout)?;
        candle_nn::ops::sigmoid(&logits)
    }

    fn forward_logits_with_dropout(&self, x: &Tensor, dropout: f32) -> CandleResult<Tensor> {
        let mut h = self.in_ln.forward(&self.in_proj.forward(x)?)?.gelu()?;
        for block in &self.blocks {
            h = block.forward(&h, dropout)?;
        }
        self.out_head.forward(&self.out_ln.forward(&h)?)
    }

    /// Predicts fingerprint probabilities for a single feature vector.
    pub fn predict(&self, features: &[f32]) -> PredictedFingerprint {
        assert_eq!(
            features.len(),
            self.in_dim,
            "Input feature dimension mismatch: expected {}, got {}",
            self.in_dim,
            features.len()
        );

        let input_tensor =
            Tensor::from_slice(features, (1, self.in_dim), &self.device).expect("Tensor creation failed");

        let probs_tensor = self.forward(&input_tensor).expect("Forward pass failed");
        let probs = probs_tensor
            .squeeze(0)
            .expect("Squeeze failed")
            .to_vec1::<f32>()
            .expect("Tensor to vec failed");

        PredictedFingerprint {
            probabilities: probs,
        }
    }

    /// Predicts fingerprint probabilities for a batch of feature vectors.
    pub fn predict_batch(&self, features: &[f32], batch_size: usize) -> Vec<PredictedFingerprint> {
        let n_samples = features.len() / self.in_dim;
        assert_eq!(
            n_samples, batch_size,
            "Batch size mismatch with input slice length"
        );

        let input_tensor =
            Tensor::from_slice(features, (batch_size, self.in_dim), &self.device).expect("Tensor creation failed");

        let probs_tensor = self.forward(&input_tensor).expect("Forward pass failed");
        let probs_2d = probs_tensor.to_vec2::<f32>().expect("Tensor to vec2 failed");

        probs_2d
            .into_iter()
            .map(|row| PredictedFingerprint {
                probabilities: row,
            })
            .collect()
    }

    /// Saves the model weights to a writer in `.oxfp` binary format.
    pub fn save_weights<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(WEIGHT_MAGIC)?;
        writer.write_all(&WEIGHT_VERSION_CURRENT.to_le_bytes())?;
        writer.write_all(&(self.in_dim as u32).to_le_bytes())?;
        writer.write_all(&(self.hidden_dim as u32).to_le_bytes())?;
        writer.write_all(&(self.num_blocks as u32).to_le_bytes())?;
        writer.write_all(&(self.out_dim as u32).to_le_bytes())?;

        let data = self.varmap.data().lock().unwrap();
        writer.write_all(&(data.len() as u32).to_le_bytes())?;

        // Sorted so that saving the same model twice produces byte-identical files.
        let mut entries: Vec<_> = data.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (name, var) in entries {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            let tensor = var.as_tensor();
            let shape = tensor.dims();
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u32).to_le_bytes())?;
            }

            let flat_floats = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            writer.write_all(&f32_to_le_bytes(&flat_floats))?;
        }
        drop(data);

        writer.write_all(&(self.bit_priors.len() as u32).to_le_bytes())?;
        writer.write_all(&f32_to_le_bytes(&self.bit_priors))?;

        writer.write_all(&(self.bit_weights.len() as u32).to_le_bytes())?;
        writer.write_all(&f32_to_le_bytes(&self.bit_weights))?;

        writer.write_all(&(self.bit_platt.len() as u32).to_le_bytes())?;
        writer.write_all(&f32_to_le_bytes(&self.bit_platt))?;

        writer.write_all(&(self.morgan_vocabulary.len() as u32).to_le_bytes())?;
        for &id in &self.morgan_vocabulary {
            writer.write_all(&id.to_le_bytes())?;
        }

        Ok(())
    }

    /// Loads model weights from a reader.
    pub fn load_weights<R: Read>(reader: &mut R, device: Device) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != WEIGHT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes in weight file",
            ));
        }

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let version = u32::from_le_bytes(u32_buf);

        if !matches!(
            version,
            WEIGHT_VERSION_VOCABULARY
                | WEIGHT_VERSION_PLATT
                | WEIGHT_VERSION_WEIGHTS
                | WEIGHT_VERSION_PRIORS
                | WEIGHT_VERSION_NO_PRIORS
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported model version: {version}"),
            ));
        }

        reader.read_exact(&mut u32_buf)?;
        let in_dim = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let hidden_dim = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let num_blocks = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let out_dim = u32::from_le_bytes(u32_buf) as usize;

        let mut model = Self::new_with_dims(in_dim, hidden_dim, num_blocks, out_dim, device.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))?;

        reader.read_exact(&mut u32_buf)?;
        let num_tensors = u32::from_le_bytes(u32_buf) as usize;

        let data = model.varmap.data().lock().unwrap();

        for _ in 0..num_tensors {
            reader.read_exact(&mut u32_buf)?;
            let name_len = u32::from_le_bytes(u32_buf) as usize;
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;

            reader.read_exact(&mut u32_buf)?;
            let rank = u32::from_le_bytes(u32_buf) as usize;
            let mut shape = Vec::with_capacity(rank);
            let mut total_elements = 1;
            for _ in 0..rank {
                reader.read_exact(&mut u32_buf)?;
                let d = u32::from_le_bytes(u32_buf) as usize;
                shape.push(d);
                total_elements *= d;
            }

            let float_buf = read_f32_block(reader, total_elements)?;

            if let Some(var) = data.get(&name) {
                let loaded_tensor =
                    Tensor::from_vec(float_buf, shape, &device).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))?;
                var.set(&loaded_tensor)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))?;
            }
        }

        drop(data);

        if version >= WEIGHT_VERSION_PRIORS {
            reader.read_exact(&mut u32_buf)?;
            let n_priors = u32::from_le_bytes(u32_buf) as usize;
            model.bit_priors = read_f32_block(reader, n_priors)?;
        }

        if version >= WEIGHT_VERSION_WEIGHTS {
            reader.read_exact(&mut u32_buf)?;
            let n_weights = u32::from_le_bytes(u32_buf) as usize;
            model.bit_weights = read_f32_block(reader, n_weights)?;
        }

        if version >= WEIGHT_VERSION_PLATT {
            reader.read_exact(&mut u32_buf)?;
            let n_platt = u32::from_le_bytes(u32_buf) as usize;
            model.bit_platt = read_f32_block(reader, n_platt)?;
        }

        if version >= WEIGHT_VERSION_VOCABULARY {
            reader.read_exact(&mut u32_buf)?;
            let n_vocab = u32::from_le_bytes(u32_buf) as usize;
            let mut bytes = vec![0u8; n_vocab * 4];
            reader.read_exact(&mut bytes)?;
            model.morgan_vocabulary = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
        }

        Ok(model)
    }

    /// Returns the total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let data = self.varmap.data().lock().unwrap();
        data.values().map(|v| v.as_tensor().elem_count()).sum()
    }

    /// Loads the model from a file path.
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
        Self::load_weights(&mut reader, Device::Cpu)
    }
}

/// Encodes a slice of `f32` as little-endian bytes for one bulk write.
/// Rewrites raw sigmoid outputs through their per-bit Platt logistic, in place.
///
/// A no-op unless `platt` holds two coefficients per probability.
///
/// Nothing in the prediction path calls this: calibration was measured and did not pay
/// off (see [`FingerprintResNet::bit_platt`]). It stays available so the experiment can
/// be repeated against a model that carries coefficients.
pub fn apply_platt(probs: &mut [f32], platt: &[f32]) {
    if platt.len() != probs.len() * 2 {
        return;
    }
    for (i, p) in probs.iter_mut().enumerate() {
        let clamped = p.clamp(1e-6, 1.0 - 1e-6);
        let logit = (clamped / (1.0 - clamped)).ln();
        let z = platt[2 * i] * logit + platt[2 * i + 1];
        *p = fast_sigmoid(z);
    }
}

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Reads `count` little-endian `f32` values in one bulk read.
fn read_f32_block<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ── Fast Zero-Allocation SIMD Inference Engine ──────────────────────────────

#[derive(Clone)]
pub struct FastResBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct FastResNetInference {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub out_dim: usize,

    pub w_in: Vec<f32>,
    pub b_in: Vec<f32>,
    pub ln_in_w: Vec<f32>,
    pub ln_in_b: Vec<f32>,

    pub blocks: Vec<FastResBlockWeights>,

    pub ln_out_w: Vec<f32>,
    pub ln_out_b: Vec<f32>,
    pub w_out: Vec<f32>,
    pub b_out: Vec<f32>,
}

impl FastResNetInference {
    pub fn from_resnet(resnet: &FingerprintResNet) -> Self {
        let in_dim = resnet.in_dim;
        let hidden_dim = resnet.hidden_dim;
        let num_blocks = resnet.num_blocks;
        let out_dim = resnet.out_dim;

        let w_in = resnet.in_proj.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        let b_in = resnet.in_proj.bias().unwrap().to_vec1::<f32>().unwrap();
        let ln_in_w = resnet.in_ln.weight().to_vec1::<f32>().unwrap();
        let ln_in_b = resnet.in_ln.bias().unwrap().to_vec1::<f32>().unwrap();

        let mut blocks = Vec::with_capacity(num_blocks);
        for b in &resnet.blocks {
            blocks.push(FastResBlockWeights {
                ln1_w: b.ln1.weight().to_vec1::<f32>().unwrap(),
                ln1_b: b.ln1.bias().unwrap().to_vec1::<f32>().unwrap(),
                fc1_w: b.fc1.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect(),
                fc1_b: b.fc1.bias().unwrap().to_vec1::<f32>().unwrap(),
                ln2_w: b.ln2.weight().to_vec1::<f32>().unwrap(),
                ln2_b: b.ln2.bias().unwrap().to_vec1::<f32>().unwrap(),
                fc2_w: b.fc2.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect(),
                fc2_b: b.fc2.bias().unwrap().to_vec1::<f32>().unwrap(),
            });
        }

        let ln_out_w = resnet.out_ln.weight().to_vec1::<f32>().unwrap();
        let ln_out_b = resnet.out_ln.bias().unwrap().to_vec1::<f32>().unwrap();
        let w_out = resnet.out_head.weight().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        let b_out = resnet.out_head.bias().unwrap().to_vec1::<f32>().unwrap();

        Self {
            in_dim,
            hidden_dim,
            num_blocks,
            out_dim,
            w_in,
            b_in,
            ln_in_w,
            ln_in_b,
            blocks,
            ln_out_w,
            ln_out_b,
            w_out,
            b_out,
        }
    }

    /// Fast forward pass with pre-allocated scratch buffers. Zero heap allocations.
    pub fn predict_scratch(
        &self,
        features: &[f32],
        h_main: &mut [f32],
        h_tmp1: &mut [f32],
        h_tmp2: &mut [f32],
        output: &mut [f32],
    ) {
        let h_dim = self.hidden_dim;

        // 1. in_proj
        mat_vec_mul(&self.w_in, features, &self.b_in, h_main, h_dim, self.in_dim);

        // 2. in_ln + GELU
        layer_norm_inplace(h_main, &self.ln_in_w, &self.ln_in_b);
        gelu_inplace(h_main);

        // 3. Residual Blocks
        for b in &self.blocks {
            h_tmp1.copy_from_slice(h_main);
            layer_norm_inplace(h_tmp1, &b.ln1_w, &b.ln1_b);

            mat_vec_mul(&b.fc1_w, h_tmp1, &b.fc1_b, h_tmp2, h_dim, h_dim);
            gelu_inplace(h_tmp2);

            h_tmp1.copy_from_slice(h_tmp2);
            layer_norm_inplace(h_tmp1, &b.ln2_w, &b.ln2_b);

            mat_vec_mul(&b.fc2_w, h_tmp1, &b.fc2_b, h_tmp2, h_dim, h_dim);

            for i in 0..h_dim {
                h_main[i] += h_tmp2[i];
            }
        }

        // 4. out_ln
        layer_norm_inplace(h_main, &self.ln_out_w, &self.ln_out_b);

        // 5. out_head + Sigmoid
        mat_vec_mul(&self.w_out, h_main, &self.b_out, output, self.out_dim, h_dim);
        for p in output.iter_mut() {
            *p = fast_sigmoid(*p);
        }
    }

    /// Single prediction returning a new vector.
    pub fn predict(&self, features: &[f32]) -> PredictedFingerprint {
        let mut h_main = vec![0.0f32; self.hidden_dim];
        let mut h_tmp1 = vec![0.0f32; self.hidden_dim];
        let mut h_tmp2 = vec![0.0f32; self.hidden_dim];
        let mut output = vec![0.0f32; self.out_dim];

        self.predict_scratch(features, &mut h_main, &mut h_tmp1, &mut h_tmp2, &mut output);
        PredictedFingerprint {
            probabilities: output,
        }
    }
}

#[inline]
fn mat_vec_mul(weights: &[f32], input: &[f32], bias: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    output.copy_from_slice(bias);

    let mat_w = faer::MatRef::from_row_major_slice(weights, rows, cols);
    let mat_x = faer::MatRef::from_column_major_slice(input, cols, 1);
    let mut mat_y = faer::MatMut::from_column_major_slice_mut(output, rows, 1);

    faer::linalg::matmul::matmul(
        mat_y.as_mut(),
        faer::Accum::Add,
        mat_w,
        mat_x,
        1.0,
        faer::Par::Seq,
    );
}

#[inline]
fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    for i in 0..x.len() {
        x[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}

#[inline]
fn gelu_inplace(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    for v in x.iter_mut() {
        let x_val = *v;
        let tanh_arg = SQRT_2_OVER_PI * (x_val + 0.044715 * x_val * x_val * x_val);
        *v = 0.5 * x_val * (1.0 + tanh_arg.tanh());
    }
}

#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ── Top-level Predictor Interface ───────────────────────────────────────────

/// Unified predictor interface for molecular fingerprints with compiled SIMD acceleration.
#[derive(Clone)]
pub struct FingerprintPredictor {
    pub resnet: FingerprintResNet,
    pub fast_engine: std::sync::Arc<FastResNetInference>,
}

impl FingerprintPredictor {
    /// Creates a new default predictor instance.
    pub fn new(_seed: u64) -> Self {
        let resnet = FingerprintResNet::new();
        let fast_engine = std::sync::Arc::new(FastResNetInference::from_resnet(&resnet));
        Self { resnet, fast_engine }
    }

    /// Creates a new predictor with custom dimensions.
    pub fn new_with_dims(in_dim: usize, hidden_dim: usize, num_blocks: usize, out_dim: usize) -> Self {
        let resnet = FingerprintResNet::new_with_dims(in_dim, hidden_dim, num_blocks, out_dim, Device::Cpu).expect("Failed to create ResNet");
        let fast_engine = std::sync::Arc::new(FastResNetInference::from_resnet(&resnet));
        Self { resnet, fast_engine }
    }

    /// Creates predictor from an existing FingerprintResNet.
    pub fn from_resnet(resnet: FingerprintResNet) -> Self {
        let fast_engine = std::sync::Arc::new(FastResNetInference::from_resnet(&resnet));
        Self { resnet, fast_engine }
    }

    /// Returns the total number of parameters.
    #[inline]
    pub fn num_parameters(&self) -> usize {
        self.resnet.num_parameters()
    }

    /// Returns the training-set bit priors recorded with the weights, or an empty
    /// slice for models trained before priors were recorded.
    #[inline]
    pub fn bit_priors(&self) -> &[f32] {
        &self.resnet.bit_priors
    }

    /// Returns the per-bit reliability weights measured when this model was trained,
    /// or an empty slice for models trained before reliability was recorded.
    #[inline]
    pub fn bit_weights(&self) -> &[f32] {
        &self.resnet.bit_weights
    }

    /// Returns the Morgan vocabulary this model's fingerprints are defined by, or an
    /// empty slice for models trained against plain modulo folding.
    #[inline]
    pub fn morgan_vocabulary(&self) -> &[u32] {
        &self.resnet.morgan_vocabulary
    }

    /// Returns the per-bit Platt calibration coefficients, or an empty slice for models
    /// trained before calibration was fitted.
    #[inline]
    pub fn bit_platt(&self) -> &[f32] {
        &self.resnet.bit_platt
    }

    /// Predicts fingerprint probabilities for a feature vector.
    ///
    /// Returns the network's own probabilities. The recorded Platt calibration is
    /// deliberately not applied — see [`FingerprintResNet::bit_platt`] for the
    /// measurement that decided that — and a caller who wants it must ask for it
    /// through [`apply_platt`].
    #[inline]
    pub fn predict(&self, features: &[f32]) -> PredictedFingerprint {
        self.fast_engine.predict(features)
    }

    /// Predicts fingerprint probabilities for a batch.
    #[inline]
    pub fn predict_batch(&self, features: &[f32], batch_size: usize) -> Vec<PredictedFingerprint> {
        self.resnet.predict_batch(features, batch_size)
    }

    /// Saves model weights to a writer.
    pub fn save_weights<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.resnet.save_weights(writer)
    }

    /// Loads model weights from a reader.
    pub fn load_weights<R: Read>(reader: &mut R) -> io::Result<Self> {
        let resnet = FingerprintResNet::load_weights(reader, Device::Cpu)?;
        Ok(Self::from_resnet(resnet))
    }

    /// Loads model from a file.
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let resnet = FingerprintResNet::load_from_file(path)?;
        Ok(Self::from_resnet(resnet))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet_creation() {
        let model = FingerprintResNet::new();
        assert_eq!(model.in_dim, TOTAL_FEATURES);
        assert_eq!(model.hidden_dim, DEFAULT_HIDDEN_DIM);
        assert_eq!(model.num_blocks, DEFAULT_NUM_BLOCKS);
        assert_eq!(model.out_dim, TOTAL_BITS);
    }

    #[test]
    fn test_resnet_forward() {
        let model = FingerprintResNet::new();
        let input = vec![0.5f32; TOTAL_FEATURES];
        let output = model.predict(&input);

        assert_eq!(output.probabilities.len(), TOTAL_BITS);
        for &p in &output.probabilities {
            assert!(p >= 0.0 && p <= 1.0, "Probability {p} out of [0, 1] range");
        }
    }

    #[test]
    fn test_resnet_save_load_roundtrip() {
        let model = FingerprintResNet::new();
        let input = vec![0.3f32; TOTAL_FEATURES];
        let expected = model.predict(&input);

        let mut buffer = Vec::new();
        model.save_weights(&mut buffer).unwrap();

        let loaded = FingerprintResNet::load_weights(&mut &buffer[..], Device::Cpu).unwrap();
        let actual = loaded.predict(&input);

        assert_eq!(expected.probabilities.len(), actual.probabilities.len());
        for i in 0..TOTAL_BITS {
            assert!(
                (expected.probabilities[i] - actual.probabilities[i]).abs() < 1e-6,
                "Mismatch at bit {i}"
            );
        }
    }

    #[test]
    fn test_bit_priors_survive_save_load() {
        let mut model = FingerprintResNet::new();
        model.bit_priors = (0..TOTAL_BITS).map(|i| (i % 100) as f32 / 100.0).collect();

        let mut buffer = Vec::new();
        model.save_weights(&mut buffer).unwrap();

        let loaded = FingerprintResNet::load_weights(&mut &buffer[..], Device::Cpu).unwrap();
        assert_eq!(loaded.bit_priors, model.bit_priors);
    }

    #[test]
    fn test_platt_calibration_is_recorded_but_not_applied() {
        let mut model = FingerprintResNet::new();
        // a = 1, b = +2 shifts every bit strongly upward.
        model.bit_platt = (0..TOTAL_BITS).flat_map(|_| [1.0f32, 2.0]).collect();

        let mut buffer = Vec::new();
        model.save_weights(&mut buffer).unwrap();
        let loaded = FingerprintResNet::load_weights(&mut &buffer[..], Device::Cpu).unwrap();
        assert_eq!(loaded.bit_platt, model.bit_platt);

        // Prediction must ignore the coefficients: they are recorded, not applied.
        let input = vec![0.3f32; TOTAL_FEATURES];
        let via_predictor = FingerprintPredictor::from_resnet(loaded);
        let predicted = via_predictor.predict(&input);
        let mut manually_calibrated = predicted.probabilities.clone();
        apply_platt(&mut manually_calibrated, via_predictor.bit_platt());
        assert!(
            manually_calibrated
                .iter()
                .zip(&predicted.probabilities)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "applying the coefficients should change something, so predict() ignoring them is meaningful"
        );

        let mut probs = vec![0.2f32, 0.5, 0.8];
        let platt = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0];
        apply_platt(&mut probs, &platt);
        for (before, after) in [0.2f32, 0.5, 0.8].iter().zip(&probs) {
            assert!(after > before, "b = +2 must raise {before} but gave {after}");
        }
        // Identity coefficients must leave probabilities alone.
        let mut identity = vec![0.2f32, 0.5, 0.8];
        apply_platt(&mut identity, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        for (a, b) in identity.iter().zip([0.2f32, 0.5, 0.8]) {
            assert!((a - b).abs() < 1e-4, "identity changed {b} to {a}");
        }
    }

    #[test]
    fn test_bit_weights_survive_save_load() {
        let mut model = FingerprintResNet::new();
        model.bit_weights = (0..TOTAL_BITS).map(|i| (i % 7) as f32 / 7.0).collect();

        let mut buffer = Vec::new();
        model.save_weights(&mut buffer).unwrap();

        let loaded = FingerprintResNet::load_weights(&mut &buffer[..], Device::Cpu).unwrap();
        assert_eq!(loaded.bit_weights, model.bit_weights);
    }

    /// Rewrites a freshly saved file as an older version by dropping the trailing
    /// blocks that version did not have, and returns what loading it produces.
    fn load_as_older_version(model: &FingerprintResNet, version: u32) -> FingerprintResNet {
        let mut buffer = Vec::new();
        model.save_weights(&mut buffer).unwrap();

        // Each empty trailing block is a single u32 length of zero.
        let trailing_blocks = match version {
            5 => 1,
            4 => 2,
            3 => 3,
            2 => 4,
            v => panic!("unhandled version {v}"),
        };
        buffer[4..8].copy_from_slice(&version.to_le_bytes());
        buffer.truncate(buffer.len() - 4 * trailing_blocks);

        FingerprintResNet::load_weights(&mut &buffer[..], Device::Cpu).unwrap()
    }

    #[test]
    fn test_older_versions_load_with_missing_statistics_empty() {
        let model = FingerprintResNet::new();
        assert!(
            model.bit_priors.is_empty()
                && model.bit_weights.is_empty()
                && model.bit_platt.is_empty()
                && model.morgan_vocabulary.is_empty()
        );

        let input = vec![0.3f32; TOTAL_FEATURES];
        let expected = model.predict(&input);

        for version in [2u32, 3, 4, 5] {
            let loaded = load_as_older_version(&model, version);
            assert!(loaded.bit_priors.is_empty(), "version {version}");
            assert!(loaded.bit_weights.is_empty(), "version {version}");
            assert!(loaded.bit_platt.is_empty(), "version {version}");
            assert!(loaded.morgan_vocabulary.is_empty(), "version {version}");

            // The network itself must be unchanged: old files still predict identically.
            let actual = loaded.predict(&input);
            for i in 0..TOTAL_BITS {
                assert!(
                    (expected.probabilities[i] - actual.probabilities[i]).abs() < 1e-6,
                    "version {version} diverged at bit {i}"
                );
            }
        }
    }

    #[test]
    fn test_save_is_byte_reproducible() {
        let model = FingerprintResNet::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        model.save_weights(&mut a).unwrap();
        model.save_weights(&mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_predictor_wrapper_save_load() {
        let predictor = FingerprintPredictor::new(42);
        let input = vec![0.2f32; TOTAL_FEATURES];
        let expected = predictor.predict(&input);

        let mut buffer = Vec::new();
        predictor.save_weights(&mut buffer).unwrap();

        let loaded = FingerprintPredictor::load_weights(&mut &buffer[..]).unwrap();
        let actual = loaded.predict(&input);

        assert_eq!(expected.probabilities.len(), actual.probabilities.len());
        for i in 0..TOTAL_BITS {
            assert!(
                (expected.probabilities[i] - actual.probabilities[i]).abs() < 1e-6,
                "Mismatch at bit {i}"
            );
        }
    }
}
