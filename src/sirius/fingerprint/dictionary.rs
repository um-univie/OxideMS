//! Molecular fingerprint dictionary for CSI:FingerID-style prediction.
//!
//! Defines the layout and metadata for the merged fingerprint vector that
//! the neural network predicts. Each bit position corresponds to a specific
//! molecular substructure query (MACCS key, Morgan circular environment, etc.).
//!
//! # Fingerprint Layout
//!
//! | Range        | Type       | Count | Description                              |
//! |--------------|------------|-------|------------------------------------------|
//! | `[0, 166)`   | MACCS      | 166   | MDL/Symyx structural keys                |
//! | `[166, 2214)`| Morgan     | 2048  | ECFP4 circular fingerprint (folded)      |
//!
//! # Ground Truth Generation
//!
//! Ground truth fingerprints are computed from known molecular structures (SMILES)
//! using the `oxidane` crate's fingerprint module at training time. At inference,
//! only the predicted probabilities and the bit layout are needed.

use crate::sirius::formula::Formula;
use oxidane::molecule::fingerprint::{circular_fingerprint, maccs_keys};
use oxidane::molecule::molecular_graph::MolecularGraph;
use std::collections::HashMap;

/// Total number of MACCS structural key bits.
pub const MACCS_BITS: usize = 166;

/// Total number of Morgan/ECFP4 fingerprint bits.
pub const MORGAN_BITS: usize = 2048;

/// Morgan bits given to a single circular identifier each.
///
/// Modulo folding puts roughly a dozen unrelated substructures on every bit — the
/// training corpus uses about 60,000 distinct circular identifiers, and there are only
/// 2,048 buckets. That costs almost nothing when *comparing* two known structures,
/// since one molecule contributes only ~50 identifiers and rarely collides with itself,
/// but it is ruinous when *predicting*: the network has to learn "any one of twelve
/// unrelated environments is present". Measured on the validation split, it managed
/// MCC > 0 on 82% of the 166 curated MACCS keys and on only 28% of the folded Morgan
/// bits.
///
/// Dedicated slots for the most frequent identifiers make each of those bits mean one
/// thing. The frequent set transfers across libraries — a vocabulary counted on MoNA
/// covered 73.4% of identifier occurrences in GNPS structures absent from MoNA, against
/// 75.8% within MoNA itself — so this is a property of chemistry rather than of the
/// library it was counted in.
pub const MORGAN_VOCAB_SLOTS: usize = 1792;

/// Morgan bits left as a folded remainder for identifiers outside the vocabulary.
///
/// About a quarter of a molecule's identifiers are too rare to earn a slot (half of all
/// identifiers in the corpus occur in exactly one molecule). Folding them into a small
/// tail keeps a trace of them without letting them muddy the dedicated bits.
pub const MORGAN_OVERFLOW_SLOTS: usize = MORGAN_BITS - MORGAN_VOCAB_SLOTS;

/// Morgan fingerprint radius (radius 2 = ECFP4).
pub const MORGAN_RADIUS: usize = 2;

/// Offset of MACCS bits in the merged fingerprint vector.
pub const MACCS_OFFSET: usize = 0;

/// Offset of Morgan bits in the merged fingerprint vector.
pub const MORGAN_OFFSET: usize = MACCS_BITS;

/// Total number of fingerprint bits predicted by the model.
pub const TOTAL_BITS: usize = MACCS_BITS + MORGAN_BITS;

/// Type of a fingerprint bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerprintType {
    /// MACCS 166 structural keys (MDL/Symyx).
    Maccs,
    /// Morgan circular fingerprint (ECFP-style).
    Morgan,
}

/// Metadata for a single fingerprint bit position.
#[derive(Debug, Clone)]
pub struct FingerprintBitInfo {
    /// Global index in the merged fingerprint vector `[0, TOTAL_BITS)`.
    pub index: usize,
    /// Which fingerprint type this bit belongs to.
    pub fp_type: FingerprintType,
    /// Local index within the fingerprint type (e.g., MACCS key 42 or Morgan bucket 1023).
    pub local_index: usize,
}

/// The circular identifiers that earn a dedicated Morgan bit.
///
/// Built by counting identifiers over a training corpus and keeping the most frequent
/// [`MORGAN_VOCAB_SLOTS`]; see that constant for why. An empty vocabulary means plain
/// modulo folding over the whole Morgan block, which is what models predating the
/// vocabulary were trained against.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MorganVocabulary {
    ids: Vec<u32>,
    index: HashMap<u32, usize>,
}

impl MorganVocabulary {
    /// An empty vocabulary: every identifier folds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps the most frequent identifiers, ties broken by identifier so that the same
    /// corpus always yields the same vocabulary.
    pub fn from_counts(counts: &HashMap<u32, usize>) -> Self {
        let mut ranked: Vec<(u32, usize)> = counts.iter().map(|(&id, &c)| (id, c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(MORGAN_VOCAB_SLOTS);
        Self::from_ids(ranked.into_iter().map(|(id, _)| id).collect())
    }

    /// Builds a vocabulary from an explicit, already-ordered identifier list.
    pub fn from_ids(ids: Vec<u32>) -> Self {
        let index = ids.iter().enumerate().map(|(slot, &id)| (id, slot)).collect();
        Self { ids, index }
    }

    /// The identifiers, in slot order.
    #[inline]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Whether every identifier folds instead of getting a slot.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The Morgan-block bit this identifier owns, if it has one.
    #[inline]
    pub fn slot(&self, id: u32) -> Option<usize> {
        self.index.get(&id).copied()
    }
}

/// The circular identifiers of a structure, for counting a corpus into a vocabulary.
///
/// Returns `None` for SMILES that cannot be parsed. Malformed library annotations make
/// the parser panic on occasion, and a corpus scan must not die on one record.
pub fn morgan_identifiers(smiles: &str) -> Option<Vec<u32>> {
    let mol = std::panic::catch_unwind(|| MolecularGraph::from_smiles_single(smiles).ok())
        .ok()
        .flatten()?;
    let fp = std::panic::catch_unwind(|| circular_fingerprint(&mol, MORGAN_RADIUS)).ok()?;
    Some(fp.fingerprint.ids().collect())
}

/// The merged molecular fingerprint dictionary.
///
/// Defines the complete set of fingerprint bits that the predictor outputs.
/// Used for interpreting predicted probabilities and computing database search scores.
#[derive(Debug, Clone)]
pub struct FingerprintDictionary {
    /// Per-bit metadata for all `TOTAL_BITS` positions.
    pub bits: Vec<FingerprintBitInfo>,
    /// Prior probability of each bit being set, estimated from the training set.
    /// Used in ModPlatt scoring. Values in `[0, 1]`.
    pub priors: Vec<f32>,
    /// Per-bit weight scaling that bit's contribution to the ModPlatt score.
    ///
    /// All 1.0 unless something sets them, which is the unweighted score. Nothing in
    /// the production path does: reliability weighting was measured and rejected, see
    /// [`FingerprintSearcher::for_predictor`]. These exist so that the experiment stays
    /// reproducible.
    ///
    /// [`FingerprintSearcher::for_predictor`]: super::search::FingerprintSearcher::for_predictor
    pub weights: Vec<f32>,
    /// Which circular identifiers own a Morgan bit outright.
    ///
    /// This is part of what a bit *means*, not a scoring parameter: a fingerprint
    /// computed under one vocabulary cannot be compared with one computed under
    /// another. It travels with the model that was trained against it, which is why
    /// [`ground_truth_fingerprint`](Self::ground_truth_fingerprint) is a method here
    /// rather than a free function — a caller cannot compute a fingerprint without
    /// having said which layout it is in.
    pub morgan_vocabulary: MorganVocabulary,
}

impl FingerprintDictionary {
    /// Creates a new dictionary with uniform priors (0.5) and uniform weights (1.0).
    ///
    /// The priors should be replaced with those of the run that trained the model whose
    /// predictions will be scored — see [`FingerprintSearcher::for_predictor`].
    ///
    /// [`FingerprintSearcher::for_predictor`]: super::search::FingerprintSearcher::for_predictor
    pub fn new() -> Self {
        let mut bits = Vec::with_capacity(TOTAL_BITS);

        // MACCS keys [0, 166)
        for i in 0..MACCS_BITS {
            bits.push(FingerprintBitInfo {
                index: MACCS_OFFSET + i,
                fp_type: FingerprintType::Maccs,
                local_index: i,
            });
        }

        // Morgan/ECFP4 bits [166, 2214)
        for i in 0..MORGAN_BITS {
            bits.push(FingerprintBitInfo {
                index: MORGAN_OFFSET + i,
                fp_type: FingerprintType::Morgan,
                local_index: i,
            });
        }

        let priors = vec![0.5; TOTAL_BITS];
        let weights = vec![1.0; TOTAL_BITS];

        Self {
            bits,
            priors,
            weights,
            morgan_vocabulary: MorganVocabulary::new(),
        }
    }

    /// Installs the Morgan vocabulary this dictionary's fingerprints are defined by.
    pub fn set_morgan_vocabulary(&mut self, vocabulary: MorganVocabulary) {
        self.morgan_vocabulary = vocabulary;
    }

    /// Computes the ground-truth fingerprint of a structure under this layout.
    ///
    /// Returns `None` if the SMILES cannot be parsed.
    ///
    /// The MACCS block is 166 curated keys. The Morgan block is ECFP4: identifiers in
    /// [`morgan_vocabulary`](Self::morgan_vocabulary) take their own bit, and the rest
    /// fold into the tail of [`MORGAN_OVERFLOW_SLOTS`] bits. With an empty vocabulary
    /// the whole block folds, which is the layout models before the vocabulary were
    /// trained against.
    pub fn ground_truth_fingerprint(&self, smiles: &str) -> Option<Vec<bool>> {
        let mol = std::panic::catch_unwind(|| MolecularGraph::from_smiles_single(smiles).ok())
            .ok()
            .flatten()?;
        let mut bits = vec![false; TOTAL_BITS];

        // MACCS keys occupy indices 1..=166 of oxidane's 167-bit vector.
        let maccs = std::panic::catch_unwind(|| maccs_keys(&mol)).ok()?;
        for key in 1..=MACCS_BITS {
            if maccs.get(key) {
                bits[MACCS_OFFSET + key - 1] = true;
            }
        }

        let morgan = std::panic::catch_unwind(|| circular_fingerprint(&mol, MORGAN_RADIUS)).ok()?;
        if self.morgan_vocabulary.is_empty() {
            let folded = morgan.fingerprint.fold(MORGAN_BITS);
            for i in 0..MORGAN_BITS {
                if folded.get(i) {
                    bits[MORGAN_OFFSET + i] = true;
                }
            }
        } else {
            for id in morgan.fingerprint.ids() {
                let local = match self.morgan_vocabulary.slot(id) {
                    Some(slot) => slot,
                    // Too rare to have earned a slot: fold into the tail.
                    None => MORGAN_VOCAB_SLOTS + (id as usize) % MORGAN_OVERFLOW_SLOTS,
                };
                bits[MORGAN_OFFSET + local] = true;
            }
        }

        Some(bits)
    }

    /// Sets the prior probabilities from training set bit frequency counts.
    ///
    /// `bit_counts[i]` = number of training molecules where bit `i` is set.
    /// `total_molecules` = total number of training molecules.
    ///
    /// Applies Laplace smoothing to avoid zero priors: `(count + 1) / (total + 2)`.
    pub fn set_priors_from_counts(&mut self, bit_counts: &[usize], total_molecules: usize) {
        assert_eq!(
            bit_counts.len(),
            TOTAL_BITS,
            "bit_counts length must match TOTAL_BITS"
        );
        let total = total_molecules as f32;
        for i in 0..TOTAL_BITS {
            // Laplace smoothing
            self.priors[i] = (bit_counts[i] as f32 + 1.0) / (total + 2.0);
        }
    }

    /// Sets the prior probabilities directly.
    pub fn set_priors(&mut self, priors: Vec<f32>) {
        assert_eq!(priors.len(), TOTAL_BITS, "priors length must match TOTAL_BITS");
        self.priors = priors;
    }

    /// Sets the per-bit reliability weights directly. Values are clamped to `[0, 1]`.
    pub fn set_weights(&mut self, weights: Vec<f32>) {
        assert_eq!(weights.len(), TOTAL_BITS, "weights length must match TOTAL_BITS");
        self.weights = weights.into_iter().map(|w| w.clamp(0.0, 1.0)).collect();
    }

    /// Sets the per-bit reliability weights from validation-set confusion counts.
    ///
    /// The weight is the Matthews correlation coefficient of the bit, floored at zero:
    ///
    /// ```text
    /// MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    /// ```
    ///
    /// Plain accuracy is unusable here: most of the 2,214 bits are set in only a few
    /// percent of molecules, so a model that always answers 0 scores above 95% on them
    /// while carrying no information. MCC is zero for exactly that predictor, so bits
    /// the model has not actually learned fall out of the score instead of contributing
    /// a confident-looking constant.
    ///
    /// Each slice is indexed by bit and holds the count over the validation set.
    pub fn set_weights_from_confusion(
        &mut self,
        true_pos: &[usize],
        false_pos: &[usize],
        true_neg: &[usize],
        false_neg: &[usize],
    ) {
        assert_eq!(true_pos.len(), TOTAL_BITS, "confusion counts must match TOTAL_BITS");
        for i in 0..TOTAL_BITS {
            let (tp, fp) = (true_pos[i] as f64, false_pos[i] as f64);
            let (tn, fn_) = (true_neg[i] as f64, false_neg[i] as f64);
            let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
            let mcc = if denom > 0.0 {
                (tp * tn - fp * fn_) / denom
            } else {
                // A bit that is constant in the validation set, or that the model never
                // predicts either way, is undefined under MCC and uninformative here.
                0.0
            };
            self.weights[i] = mcc.max(0.0) as f32;
        }
    }

    /// Returns the total number of bits in the dictionary.
    #[inline]
    pub fn total_bits(&self) -> usize {
        TOTAL_BITS
    }

    /// Returns the fingerprint type and local index for a global bit position.
    pub fn bit_info(&self, global_index: usize) -> &FingerprintBitInfo {
        &self.bits[global_index]
    }

    /// Returns the range of global indices for MACCS bits.
    #[inline]
    pub fn maccs_range(&self) -> std::ops::Range<usize> {
        MACCS_OFFSET..MACCS_OFFSET + MACCS_BITS
    }

    /// Returns the range of global indices for Morgan bits.
    #[inline]
    pub fn morgan_range(&self) -> std::ops::Range<usize> {
        MORGAN_OFFSET..MORGAN_OFFSET + MORGAN_BITS
    }
}

impl Default for FingerprintDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// A ground-truth or candidate binary fingerprint vector.
///
/// Stores `TOTAL_BITS` boolean values packed into a `Vec<bool>`.
/// Used for training labels and database candidate fingerprints.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryFingerprint {
    /// One boolean per bit position. Length = `TOTAL_BITS`.
    pub bits: Vec<bool>,
}

impl BinaryFingerprint {
    /// Creates a zero fingerprint (all bits false).
    pub fn zeros() -> Self {
        Self {
            bits: vec![false; TOTAL_BITS],
        }
    }

    /// Creates a fingerprint from a pre-computed bit vector.
    pub fn from_bits(bits: Vec<bool>) -> Self {
        assert_eq!(bits.len(), TOTAL_BITS, "bits length must match TOTAL_BITS");
        Self { bits }
    }

    /// Sets a bit at the given global index.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        self.bits[index] = value;
    }

    /// Gets a bit at the given global index.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        self.bits[index]
    }

    /// Counts the number of set (true) bits.
    pub fn popcount(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Computes the Tanimoto similarity to another fingerprint.
    pub fn tanimoto(&self, other: &BinaryFingerprint) -> f64 {
        let mut intersection = 0usize;
        let mut union = 0usize;
        for i in 0..TOTAL_BITS {
            let a = self.bits[i];
            let b = other.bits[i];
            if a || b {
                union += 1;
            }
            if a && b {
                intersection += 1;
            }
        }
        if union == 0 {
            return 1.0;
        }
        intersection as f64 / union as f64
    }

    /// Number of bytes required to pack `TOTAL_BITS` into contiguous byte array.
    pub const PACKED_BYTES: usize = (TOTAL_BITS + 7) / 8;

    /// Packs the 2,214-bit fingerprint into a compact contiguous 277-byte slice.
    pub fn to_packed_bytes(&self) -> [u8; Self::PACKED_BYTES] {
        let mut packed = [0u8; Self::PACKED_BYTES];
        for i in 0..TOTAL_BITS {
            if self.bits[i] {
                packed[i / 8] |= 1 << (i % 8);
            }
        }
        packed
    }

    /// Unpacks a 277-byte slice into a 2,214-bit `BinaryFingerprint`.
    pub fn from_packed_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Self::PACKED_BYTES, "bytes slice too short for fingerprint");
        let mut bits = vec![false; TOTAL_BITS];
        for i in 0..TOTAL_BITS {
            bits[i] = (bytes[i / 8] & (1 << (i % 8))) != 0;
        }
        Self { bits }
    }
}

// ── Chemical Formula Constraint Masking ──────────────────────────────────────

/// Applies strict chemical formula constraints to predicted fingerprint probabilities.
///
/// If the candidate molecular formula lacks specific elements or features (e.g. 0 Nitrogen,
/// 0 Oxygen, 0 Sulfur, 0 Halogens, 0 DBE, Carbon < 6), any corresponding substructure


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_creation() {
        let dict = FingerprintDictionary::new();
        assert_eq!(dict.total_bits(), TOTAL_BITS);
        assert_eq!(dict.bits.len(), TOTAL_BITS);
        assert_eq!(dict.priors.len(), TOTAL_BITS);
    }

    #[test]
    fn test_dictionary_layout() {
        let dict = FingerprintDictionary::new();

        // MACCS bits
        for i in 0..MACCS_BITS {
            let info = dict.bit_info(i);
            assert_eq!(info.fp_type, FingerprintType::Maccs);
            assert_eq!(info.local_index, i);
        }

        // Morgan bits
        for i in 0..MORGAN_BITS {
            let info = dict.bit_info(MORGAN_OFFSET + i);
            assert_eq!(info.fp_type, FingerprintType::Morgan);
            assert_eq!(info.local_index, i);
        }
    }

    #[test]
    fn test_ground_truth_under_both_morgan_layouts() {
        // Caffeine.
        let smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C";

        let folded = FingerprintDictionary::new();
        let folded_bits = folded.ground_truth_fingerprint(smiles).expect("caffeine parses");
        assert_eq!(folded_bits.len(), TOTAL_BITS);
        assert!(folded_bits.iter().any(|&b| b));

        // Give caffeine's own identifiers the vocabulary, so every one of them lands in
        // a dedicated slot and none reaches the folded tail.
        let ids = morgan_identifiers(smiles).expect("caffeine parses");
        assert!(!ids.is_empty());
        let mut counts = HashMap::new();
        for (rank, &id) in ids.iter().enumerate() {
            counts.insert(id, ids.len() - rank);
        }
        let vocabulary = MorganVocabulary::from_counts(&counts);
        let mut dict = FingerprintDictionary::new();
        dict.set_morgan_vocabulary(vocabulary.clone());

        let bits = dict.ground_truth_fingerprint(smiles).expect("caffeine parses");
        assert_eq!(bits.len(), TOTAL_BITS);

        // MACCS is untouched by the Morgan layout.
        assert_eq!(bits[..MACCS_BITS], folded_bits[..MACCS_BITS]);

        // Every set Morgan bit is a dedicated slot, and exactly the ones this molecule
        // owns; nothing spilled into the folded tail.
        let expected: std::collections::HashSet<usize> =
            ids.iter().map(|&id| vocabulary.slot(id).unwrap()).collect();
        let actual: std::collections::HashSet<usize> = (0..MORGAN_BITS)
            .filter(|&i| bits[MORGAN_OFFSET + i])
            .collect();
        assert_eq!(actual, expected, "vocabulary slots must be exactly the molecule's identifiers");
        assert!(
            actual.iter().all(|&i| i < MORGAN_VOCAB_SLOTS),
            "no identifier should have reached the folded tail"
        );

        // A structure whose identifiers are absent from the vocabulary still gets bits,
        // in the folded tail rather than nowhere.
        let other = dict.ground_truth_fingerprint("CCCCCCCCCCCCCCCCCC(=O)O").expect("stearic acid parses");
        assert!(
            (MORGAN_VOCAB_SLOTS..MORGAN_BITS).any(|i| other[MORGAN_OFFSET + i]),
            "identifiers outside the vocabulary must fold into the tail"
        );
    }

    #[test]
    fn test_vocabulary_is_deterministic_and_capped() {
        let mut counts = HashMap::new();
        for id in 0..(MORGAN_VOCAB_SLOTS as u32 * 2) {
            counts.insert(id, (id % 7) as usize + 1);
        }
        let a = MorganVocabulary::from_counts(&counts);
        let b = MorganVocabulary::from_counts(&counts);
        assert_eq!(a.ids().len(), MORGAN_VOCAB_SLOTS);
        assert_eq!(a, b, "the same corpus must always give the same vocabulary");
        assert_eq!(a.slot(a.ids()[0]), Some(0));
        assert!(MorganVocabulary::new().is_empty());
    }

    #[test]
    fn test_weights_from_confusion() {
        let mut dict = FingerprintDictionary::new();
        let (mut tp, mut fp, mut tn, mut fn_) = (
            vec![0usize; TOTAL_BITS],
            vec![0usize; TOTAL_BITS],
            vec![0usize; TOTAL_BITS],
            vec![0usize; TOTAL_BITS],
        );

        // Bit 0: predicted perfectly on a balanced bit.
        tp[0] = 50;
        tn[0] = 50;

        // Bit 1: the always-zero predictor on a rare bit. 95% accurate, no information.
        tn[1] = 95;
        fn_[1] = 5;

        // Bit 2: predicted exactly backwards.
        fp[2] = 50;
        fn_[2] = 50;

        // Bit 3: half the positives found, no false alarms.
        tp[3] = 25;
        fn_[3] = 25;
        tn[3] = 50;

        dict.set_weights_from_confusion(&tp, &fp, &tn, &fn_);

        assert!((dict.weights[0] - 1.0).abs() < 1e-6, "perfect bit: {}", dict.weights[0]);
        assert_eq!(dict.weights[1], 0.0, "constant predictor must carry no weight");
        assert_eq!(dict.weights[2], 0.0, "anti-correlated bit must be floored at zero");
        assert!(
            dict.weights[3] > 0.0 && dict.weights[3] < 1.0,
            "partial bit: {}",
            dict.weights[3]
        );
    }

    #[test]
    fn test_priors_from_counts() {
        let mut dict = FingerprintDictionary::new();
        let mut counts = vec![0usize; TOTAL_BITS];
        counts[0] = 100; // MACCS key 0 set in 100/200 molecules
        counts[1] = 0;   // MACCS key 1 never set

        dict.set_priors_from_counts(&counts, 200);

        // With Laplace: (100 + 1) / (200 + 2) = 101/202 ≈ 0.5
        assert!((dict.priors[0] - 101.0 / 202.0).abs() < 1e-6);
        // With Laplace: (0 + 1) / (200 + 2) = 1/202 ≈ 0.005
        assert!((dict.priors[1] - 1.0 / 202.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_fingerprint_tanimoto() {
        let mut a = BinaryFingerprint::zeros();
        let mut b = BinaryFingerprint::zeros();

        a.set(0, true);
        a.set(1, true);
        a.set(2, true);

        b.set(1, true);
        b.set(2, true);
        b.set(3, true);

        // intersection = {1, 2} = 2, union = {0, 1, 2, 3} = 4
        let tanimoto = a.tanimoto(&b);
        assert!((tanimoto - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_binary_fingerprint_popcount() {
        let mut fp = BinaryFingerprint::zeros();
        assert_eq!(fp.popcount(), 0);
        fp.set(10, true);
        fp.set(100, true);
        fp.set(1000, true);
        assert_eq!(fp.popcount(), 3);
    }
}
