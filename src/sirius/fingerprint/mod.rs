//! Molecular fingerprint prediction from fragmentation trees (CSI:FingerID-style).
//!
//! This module implements a complete pipeline for predicting molecular fingerprints
//! directly from MAP fragmentation trees computed by the SIRIUS solver. The predicted
//! fingerprints enable structure identification by scoring candidate molecules
//! from structural databases (PubChem, HMDB, etc.).
//!
//! # Pipeline Overview
//!
//! ```text
//! FragmentationTree ──→ TreeFeatureExtractor ──→ Feature Vector (5,147-d)
//!                                                        │
//!                                        FingerprintPredictor (ResNet)
//!                                                        │
//!                                                        ▼
//!                                          PredictedFingerprint (2214 probabilities)
//!                                                        │
//!                                           FingerprintSearcher (ModPlatt)
//!                                                        │
//!                                                        ▼
//!                                          Ranked Structure Candidates
//! ```
//!
//! # Architecture
//!
//! - **Feature Extraction**: Hashes fragment formulas, neutral loss formulas, and
//!   root-to-descendant losses into fixed-size buckets using xxhash, combined with
//!   27 tree-level scalar statistics.
//! - **Prediction**: A pure-Rust residual network with LayerNorm, GELU hidden
//!   activations and sigmoid output, producing per-bit probabilities. Dropout in the
//!   residual branches and dropout of occupied input features are available and off by
//!   default; see [`TrainingConfig::dropout`](training::TrainingConfig::dropout).
//! - **Database Search**: ModPlatt log-likelihood scoring with prior correction
//!   for ranking candidate structures against predicted fingerprints.
//!
//! # Model statistics
//!
//! Scoring needs two per-bit quantities that only the training run can know, so both
//! are measured there and stored in the model file next to the weights:
//!
//! - **priors** `πᵢ` — the frequency of bit `i` in the training set, which ModPlatt
//!   divides out so that common substructures cannot dominate the score.
//! - **reliability weights** `wᵢ` — the Matthews correlation of bit `i` on the
//!   validation split. Recorded as a diagnostic only: on the current model just 158 of
//!   2,214 bits clear MCC > 0, and scoring with these weights cost 27 points of top-1
//!   retrieval accuracy, so scoring does not use them. See
//!   [`FingerprintSearcher::for_predictor`].
//!
//! # Formula constraints, removed
//!
//! A predicted fingerprint used to be masked against the candidate's molecular formula:
//! MACCS keys that require nitrogen were zeroed for a nitrogen-free formula, and so on,
//! from hand-written lists of key indices. Those lists were wrong. Checked against the
//! ground-truth fingerprints of 2,660 library structures, the mask zeroed a bit that
//! the structure actually sets in 99.6% of molecules — MACCS key 150 alone in 1,783 of
//! them. Removing it raised predicted-vs-true Tanimoto on the CASMI 2016 answers from
//! 0.305 to 0.341 and top-10 retrieval from 18.9% to 26.8%.
//!
//! The idea is sound — a formula does forbid some keys — but it has to be derived from
//! the key definitions rather than typed out.
//!
//! Build a searcher with [`FingerprintSearcher::for_predictor`] to pick up the priors.
//! A model file that predates either statistic loads with the neutral value for it.
//!
//! # References
//!
//! - Dührkop, K. et al. (2015). Searching molecular structure databases with
//!   tandem mass spectra using CSI:FingerID. PNAS, 112(41), 12580–12585.
//! - Dührkop, K. et al. (2021). Systematic classification of unknown metabolites
//!   using high-resolution fragmentation mass spectra. Nature Biotechnology, 39(4), 462–471.

pub mod dataset;
pub mod db;
pub mod dictionary;
pub mod features;
pub mod predictor;
pub mod search;
pub mod training;

pub use dataset::{assign_split, spectrum_key, structure_key, DatasetManifest, Split, SplitRatios};
pub use db::{DatabaseScope, DatabaseStructure, SampleContext, StructureDatabase};
pub use dictionary::{
    morgan_identifiers, BinaryFingerprint, FingerprintBitInfo,
    FingerprintDictionary, FingerprintType, MorganVocabulary, MACCS_BITS, MORGAN_BITS,
    MORGAN_OVERFLOW_SLOTS, MORGAN_VOCAB_SLOTS, TOTAL_BITS,
};
pub use features::{TreeFeatureExtractor, TreeFeatures, TOTAL_FEATURES, tree_graph};
pub use predictor::{
    apply_platt, FingerprintPredictor, FingerprintResNet, PredictedFingerprint, ResBlock,
};
pub use search::{
    CandidateScore, FingerprintSearcher, ModPlattTable, ScoredStructure, StructureCandidate,
};
pub use training::{
    generate_training_sample, train, train_resnet,
    train_resnet_with_val_set, train_with_val_set, TrainingConfig, TrainingHistory, TrainingSample,
};
