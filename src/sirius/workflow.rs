//! End-to-end Spectral Clustering & SIRIUS Molecular Formula Annotation Workflow.
//!
//! Ingests mzML or MGF spectral data files, runs high-performance DBSCAN clustering
//! using domain-tuned presets (Metabolomics or Proteomics), reconstructs consensus spectra,
//! and annotates each cluster with candidate molecular formulas, MS1 isotopic envelope scores,
//! and optimal MAP fragmentation trees.

use super::adduct::Adduct;
use super::canopus::{CanopusClassifier, CompoundClassification};
use super::element::{Element, ALL_ELEMENTS};
use super::fingerprint::{
    DatabaseScope, DatabaseStructure, FingerprintDictionary, FingerprintPredictor,
    FingerprintSearcher, PredictedFingerprint, SampleContext, StructureDatabase,
    TreeFeatureExtractor,
};
use super::formula::Formula;
use super::generator::FormulaGenerator;
use super::rules::ChemicalRules;
use super::scoring::{FormulaScorer, calculate_isotope_distribution};
use super::solver::{FragmentationTreeBuilder, TreeSolverConfig};
use super::tree::FragmentationTree;
use crate::clustering::cluster::experiment::SpectralExperiment;
use crate::clustering::cluster::flat_clusters::FlatClusters;
use crate::io::mzml::{MzmlParser, MzmlParserConfig};
use crate::spectrum::{MassSpectralRecord, Peak};
use crate::workflow::MassTolerance;
use rayon::prelude::*;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;
use std::sync::Arc;

/// Preset configuration for spectral preprocessing and clustering parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowPreset {
    /// Metabolomics preset (epsilon = 0.5, min_mz_range = 50.0, relaxed filtering)
    Metabolomics,
    /// Proteomics preset (epsilon = 0.05, min_mz_range = 250.0, strict filtering)
    Proteomics,
}

/// Strategy for generating precursor adduct hypotheses during cluster / spectrum annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum AdductSearchStrategy {
    /// Evaluate only the configured primary adduct (e.g. [M+H]+ or [M-H]-).
    Single(Adduct),
    /// Automatically test all common positive ESI adducts ([M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, [M+H-H2O]+).
    AutoPositive,
    /// Automatically test all common negative ESI adducts ([M-H]-, [M+Cl]-, [M+HCOO]-, [M+CH3COO]-).
    AutoNegative,
    /// Fast primary [M+H]+ evaluation with automatic multi-adduct fallback if no confident identification
    TwoTierPositive,
    /// Fast primary [M-H]- evaluation with automatic multi-adduct fallback if no confident identification
    TwoTierNegative,
    /// Test a user-specified custom list of candidate adducts.
    Custom(Vec<Adduct>),
}

/// Heuristic identification confidence level; thresholds are not calibrated probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// High confidence: decisive score gap (Δscore >= 4.0) and high explained intensity (>= 60%)
    High,
    /// Medium confidence: clear score gap (Δscore >= 1.5) and moderate explained intensity (>= 30%)
    Medium,
    /// Low confidence: small score gap or low explained intensity
    Low,
    /// Ambiguous: close score tie with second-best candidate (Δscore < 0.5)
    Ambiguous,
}

/// Comprehensive confidence evaluation for a top candidate formula
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentificationConfidence {
    pub level: ConfidenceLevel,
    pub score_gap: f64,
    pub normalized_score_gap: f64,
    /// Legacy score-gap sigmoid, not an empirically calibrated posterior.
    pub posterior_probability: f64,
    pub explained_intensity: f32,
}

/// A candidate structural isomer retrieved from a database and ranked by ModPlatt Bayesian scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateStructureMatch {
    pub name: String,
    pub smiles: String,
    pub inchi_key: Option<String>,
    pub tanimoto_similarity: f32,
    pub mod_platt_score: f64,
}

/// A ranked candidate molecular formula annotation for a spectral cluster.
#[derive(Debug, Clone)]
pub struct AnnotatedFormulaCandidate {
    pub formula: Formula,
    pub adduct: Adduct,
    pub exact_mass: f64,
    pub mass_error_ppm: f64,
    pub mass_error_da: f64,
    pub tree_score: f64,
    pub isotope_score: f64,
    pub prior_score: f64,
    pub adduct_prior_score: f64,
    pub total_score: f64,
    pub explained_intensity: f32,
    pub explained_peaks: usize,
    pub total_peaks: usize,
    pub tree: FragmentationTree,
    pub predicted_fingerprint: Option<super::fingerprint::PredictedFingerprint>,
    pub classification: Option<super::canopus::CompoundClassification>,
    pub structural_candidates: Vec<CandidateStructureMatch>,
}

/// Comprehensive annotation results for a single spectral cluster or consensus spectrum.
#[derive(Debug, Clone)]
pub struct SiriusClusterAnnotation {
    pub cluster_id: usize,
    pub member_count: usize,
    pub precursor_mz: f64,
    pub neutral_mass: f64,
    pub retention_time_seconds: f32,
    pub charge: i8,
    pub adduct: Adduct,
    pub top_candidates: Vec<AnnotatedFormulaCandidate>,
    pub consensus_peaks: Vec<Peak>,
    pub ms1_envelope: Vec<Peak>,
    /// True when `ms1_envelope` came from a real MS1 survey scan; false when it
    /// fell back to the surviving-precursor window of the MS2 consensus (MGF
    /// input, or mzML without MS1).
    pub ms1_from_survey_scan: bool,
}

impl AnnotatedFormulaCandidate {
    /// The theoretical isotope pattern of this candidate as it would appear in
    /// the MS1 spectrum: `(ion m/z, relative abundance)` for M, M+1, M+2, M+3,
    /// abundances normalised to sum 1. Pair with the cluster's `ms1_envelope`
    /// (the observed peaks) to render an observed-vs-theoretical fit.
    pub fn theoretical_isotope_pattern(&self) -> Vec<(f64, f64)> {
        calculate_isotope_distribution(&self.formula, 4)
            .into_iter()
            .map(|p| (self.adduct.neutral_mass_to_ion_mz(p.mz), p.abundance))
            .collect()
    }
}

impl SiriusClusterAnnotation {
    /// Returns the top-1 ranked formula candidate, if any.
    pub fn top_candidate(&self) -> Option<&AnnotatedFormulaCandidate> {
        self.top_candidates.first()
    }

    /// Computes heuristic identification confidence for the top candidate.
    pub fn confidence(&self) -> Option<IdentificationConfidence> {
        let top1 = self.top_candidates.first()?;
        let top2_score = self.top_candidates.get(1).map(|c| c.total_score);
        let score_gap = match top2_score {
            Some(s2) => top1.total_score - s2,
            None => 0.0,
        };
        let norm_gap = score_gap / top1.total_score.abs().max(1.0);
        let posterior = 1.0 / (1.0 + (-score_gap / 2.0).exp());
        let expl_int = top1.explained_intensity;

        let level = if score_gap >= 4.0 && expl_int >= 0.60 {
            ConfidenceLevel::High
        } else if score_gap >= 1.5 && expl_int >= 0.30 {
            ConfidenceLevel::Medium
        } else if score_gap < 0.5 && self.top_candidates.len() >= 2 {
            ConfidenceLevel::Ambiguous
        } else {
            ConfidenceLevel::Low
        };

        Some(IdentificationConfidence {
            level,
            score_gap,
            normalized_score_gap: norm_gap,
            posterior_probability: posterior,
            explained_intensity: expl_int,
        })
    }
}

/// Summary report of an end-to-end workflow execution.
#[derive(Debug, Clone)]
pub struct SiriusWorkflowResult {
    pub annotations: Vec<SiriusClusterAnnotation>,
    pub total_spectra_ingested: usize,
    pub total_clusters_formed: usize,
    pub total_annotated_clusters: usize,
    pub elapsed_time_seconds: f64,
    pub throughput_spectra_per_sec: f64,
}

/// Policy for handling spectra or clusters with unknown charge (charge == 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownChargePolicy {
    /// Default to the configured adduct (e.g. [M+H]+ for metabolomics, [M+2H]2+ for proteomics)
    DefaultAdduct,
    /// Test multiple charge hypotheses (e.g. +1 and +2) and let tree/isotope scoring select the best charge
    MultiChargeHypothesis,
    /// Drop spectra with unknown charge (charge == 0)
    FilterOut,
}

/// Versioned application/benchmark search budgets. AccurateV1 scores all generated
/// precursor candidates before the exact-tree shortlist; FastV1 caps them at 128.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationProfile { FastV1, AccurateV1 }

impl AnnotationProfile {
    pub fn name(self) -> &'static str {
        match self { Self::FastV1 => "fast-v1", Self::AccurateV1 => "accurate-v1" }
    }
}

impl SiriusWorkflowConfig {
    pub fn for_profile(profile: AnnotationProfile) -> Self {
        Self {
            tolerance: MassTolerance::Ppm(10.0),
            max_peaks_per_tree: if profile == AnnotationProfile::AccurateV1 { 60 } else { 30 },
            max_candidates_for_tree_search: if profile == AnnotationProfile::AccurateV1 { usize::MAX } else { 128 },
            enable_halogens: true,
            // Formula ranking is spectral until structure boosts are validated.
            structure_score_weight: 0.0,
            isotope_weight: 1.0,
            ..Self::default()
        }
    }
}

/// Which peak-preprocessing pipeline `preprocess_peaks_for_tree` runs.
///
/// Round 46 compared them head to head on 1779 MassBank spectra, everything else
/// fixed. `Simple` won **62/43 overall and 37/8 on chlorinated compounds
/// (p = 1.5e-5)**, because `Legacy`'s adaptive floor, deisotoping and
/// diversity-aware selection between them discard nearly half the peaks — 16.0
/// kept per spectrum against 30.1 — and peak count is what bounds how much of a
/// spectrum a tree can explain.
///
/// `Legacy` is kept so that result stays reproducible, not because it is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeakPreprocessing {
    /// Relative-intensity threshold, precursor ceiling, top-N by intensity, dedupe.
    #[default]
    Simple,
    /// Adaptive noise floor + deisotoping + entropy gate + diversity selection.
    Legacy,
}

/// How `preprocess_peaks_for_tree` treats M+k satellites.
///
/// The legacy rule accepts anything within `max(ppm·precursor/1e6, 0.01)` of
/// `M + k·1.00335` for k in 1..=3. The k=2 target is M + 2.00670, and a 37Cl
/// satellite sits at M + 1.99705 — **9.65 mDa away**, so it is always inside a
/// window that can never be tighter than 10 mDa. Measured on MoNA: 95 % of
/// 35Cl/37Cl doublets destroyed, 52 of 65 chlorinated spectra left with no
/// halogen evidence at all. `IsotopeAware` assigns each satellite to whichever
/// of the two spacings it is actually closer to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeisotopeMode {
    /// No pre-tree deisotoping.
    Off,
    /// The original rule: any M + k·1.00335 within tolerance is a 13C satellite.
    Legacy,
    /// Separate 13C2 (M+2.00670) from 37Cl/81Br (M+1.99705/M+1.99795).
    #[default]
    IsotopeAware,
}

/// Configuration options for the integrated clustering and SIRIUS annotation workflow.
#[derive(Debug, Clone)]
pub struct SiriusWorkflowConfig {
    pub preset: WorkflowPreset,
    pub tolerance: MassTolerance,
    pub adduct: Adduct,
    pub adduct_search_strategy: AdductSearchStrategy,
    pub rules: ChemicalRules,
    pub isotope_weight: f64,
    pub top_k_candidates: usize,
    pub min_cluster_size: usize,
    pub max_peaks_per_tree: usize,
    pub max_candidates_for_tree_search: usize,
    pub consensus_bin_width_da: f64,
    pub filter_unknown_charge: bool,
    pub unknown_charge_policy: UnknownChargePolicy,
    pub candidate_charges: Vec<i8>,
    pub filter_peaks_above_precursor: bool,
    /// See [`DeisotopeMode`]. Only consulted by [`PeakPreprocessing::Legacy`].
    pub deisotope: DeisotopeMode,
    /// See [`PeakPreprocessing`].
    pub peak_preprocessing: PeakPreprocessing,
    pub enable_fingerprint_prediction: bool,
    pub enable_compound_classification: bool,
    pub fingerprint_model_path: Option<PathBuf>,
    pub structure_database: Option<Arc<StructureDatabase>>,
    pub structure_score_weight: f64,
    pub sample_context: SampleContext,
    pub enable_halogens: bool,
}

impl Default for SiriusWorkflowConfig {
    fn default() -> Self {
        Self {
            preset: WorkflowPreset::Metabolomics,
            tolerance: MassTolerance::Ppm(15.0),
            adduct: Adduct::protonated(),
            adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
            rules: ChemicalRules::default(),
            isotope_weight: 1.5,
            top_k_candidates: 5,
            min_cluster_size: 1,
            max_peaks_per_tree: 30,
            max_candidates_for_tree_search: 25,
            consensus_bin_width_da: 0.01,
            filter_unknown_charge: false,
            unknown_charge_policy: UnknownChargePolicy::DefaultAdduct,
            candidate_charges: vec![1, 2],
            filter_peaks_above_precursor: true,
            deisotope: DeisotopeMode::default(),
            peak_preprocessing: PeakPreprocessing::default(),
            enable_fingerprint_prediction: false,
            enable_compound_classification: false,
            fingerprint_model_path: None,
            structure_database: None,
            structure_score_weight: 0.05,
            sample_context: SampleContext::AllBio,
            enable_halogens: false,
        }
    }
}

/// Integrated Spectral Clustering & SIRIUS Molecular Formula Annotation Workflow.
pub struct SiriusClusteringWorkflow {
    pub config: SiriusWorkflowConfig,
    pub gen_chno: FormulaGenerator<4>,
    pub gen_chnos: FormulaGenerator<5>,
    pub gen_chnops: FormulaGenerator<6>,
    pub gen_halogens: FormulaGenerator<8>,
    pub tree_builder: FragmentationTreeBuilder,
    pub scorer: FormulaScorer,
    pub fingerprint_predictor: Option<FingerprintPredictor>,
    pub canopus_classifier: CanopusClassifier,
    pub structure_database: Option<Arc<StructureDatabase>>,
    pub fingerprint_searcher: FingerprintSearcher,
}

impl Default for SiriusClusteringWorkflow {
    fn default() -> Self {
        Self::new(SiriusWorkflowConfig::default())
    }
}

impl SiriusClusteringWorkflow {
    /// Creates a new workflow instance with the specified configuration.
    pub fn new(config: SiriusWorkflowConfig) -> Self {
        let tree_config = TreeSolverConfig {
            tolerance: config.tolerance,
            adduct: config.adduct.clone(),
            ..Default::default()
        };

        let ppm = match config.tolerance {
            MassTolerance::Ppm(p) => p,
            MassTolerance::Da(d) => (d / 300.0) * 1e6,
        };

        let predictor = if config.enable_fingerprint_prediction {
            if let Some(ref path) = config.fingerprint_model_path {
                FingerprintPredictor::load_from_file(path).ok()
            } else if Path::new("models/sirius_fingerprints.oxfp").exists() {
                FingerprintPredictor::load_from_file("models/sirius_fingerprints.oxfp").ok()
            } else {
                None
            }
        } else {
            None
        };

        let db = config.structure_database.clone();
        let searcher = Self::searcher_for(predictor.as_ref());

        Self {
            gen_chno: FormulaGenerator::chno(),
            gen_chnos: FormulaGenerator::chnos(),
            gen_chnops: FormulaGenerator::chnops(),
            gen_halogens: FormulaGenerator::chnops_halogens(),
            tree_builder: FragmentationTreeBuilder::new(tree_config),
            scorer: FormulaScorer::new(ppm, 0.05),
            fingerprint_predictor: predictor,
            canopus_classifier: CanopusClassifier::new(),
            structure_database: db,
            fingerprint_searcher: searcher,
            config,
        }
    }

    /// Enables or disables halogen candidate formula generation (Cl, Br, F, I, Si, B).
    pub fn with_halogens(mut self, enable: bool) -> Self {
        self.config.enable_halogens = enable;
        self
    }

    /// Sets a preloaded CSI:FingerID fingerprint predictor.
    pub fn with_fingerprint_predictor(mut self, predictor: FingerprintPredictor) -> Self {
        self.config.enable_fingerprint_prediction = true;
        self.fingerprint_searcher = Self::searcher_for(Some(&predictor));
        self.fingerprint_predictor = Some(predictor);
        self
    }

    /// Builds the database searcher for a given predictor.
    ///
    /// The priors and reliability weights ModPlatt scores against belong to the model
    /// doing the predicting, so this defers to [`FingerprintSearcher::for_predictor`].
    /// With no predictor there is nothing to search with, and the neutral searcher is
    /// only a placeholder.
    fn searcher_for(predictor: Option<&FingerprintPredictor>) -> FingerprintSearcher {
        match predictor {
            Some(p) => FingerprintSearcher::for_predictor(p),
            None => FingerprintSearcher::new(FingerprintDictionary::new()),
        }
    }

    /// Sets a structural candidate database for isomer identification and formula score boosting.
    pub fn with_structure_database(mut self, db: StructureDatabase) -> Self {
        let db = Arc::new(db);
        self.structure_database = Some(db.clone());
        self.config.structure_database = Some(db);
        self
    }

    /// Sets the weighting factor for structure match score boosting on formulas (default: 0.05).
    pub fn with_structure_score_weight(mut self, weight: f64) -> Self {
        self.config.structure_score_weight = weight;
        self
    }

    /// Sets the biological sample context / origin for candidate structure filtering (e.g. `SampleContext::Human`, `SampleContext::Plant`).
    pub fn with_sample_context(mut self, context: SampleContext) -> Self {
        self.config.sample_context = context;
        self
    }

    /// Enables or disables CANOPUS ClassyFire compound classification.
    pub fn with_compound_classification(mut self, enable: bool) -> Self {
        self.config.enable_compound_classification = enable;
        self
    }

    /// Loads CSI:FingerID model weights from a specified binary `.oxfp` path.
    pub fn with_fingerprint_model_path<P: AsRef<Path>>(mut self, path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let predictor = FingerprintPredictor::load_from_file(p)?;
        self.config.enable_fingerprint_prediction = true;
        self.config.fingerprint_model_path = Some(p.to_path_buf());
        self.fingerprint_searcher = Self::searcher_for(Some(&predictor));
        self.fingerprint_predictor = Some(predictor);
        Ok(self)
    }

    /// Sets the workflow preset (Metabolomics or Proteomics).
    pub fn with_preset(mut self, preset: WorkflowPreset) -> Self {
        self.config.preset = preset;
        self
    }

    /// Sets the mass tolerance for precursor matching and fragment annotation.
    pub fn with_tolerance(mut self, tolerance: MassTolerance) -> Self {
        self.config.tolerance = tolerance;
        let mut tree_config = self.tree_builder.config().clone();
        tree_config.tolerance = tolerance;
        self.tree_builder = FragmentationTreeBuilder::new(tree_config);
        let ppm = match tolerance { MassTolerance::Ppm(p) => p, MassTolerance::Da(d) => d / 300.0 * 1e6 };
        self.scorer = FormulaScorer::new(ppm, 0.05);
        self
    }

    /// Sets the ion adduct mode (e.g. `Adduct::protonated()` or `Adduct::deprotonated()`).
    pub fn with_adduct(mut self, adduct: Adduct) -> Self {
        self.config.adduct = adduct.clone();
        self.config.adduct_search_strategy = AdductSearchStrategy::Single(adduct.clone());
        let mut tree_config = self.tree_builder.config().clone();
        tree_config.adduct = adduct;
        self.tree_builder = FragmentationTreeBuilder::new(tree_config);
        self
    }

    /// Sets the adduct search strategy (e.g. Single, AutoPositive, AutoNegative, Custom).
    pub fn with_adduct_strategy(mut self, strategy: AdductSearchStrategy) -> Self {
        self.config.adduct_search_strategy = strategy;
        self
    }

    /// Automatically search all standard positive ESI adducts ([M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, [M+H-H2O]+).
    pub fn with_auto_positive_adducts(mut self) -> Self {
        self.config.adduct_search_strategy = AdductSearchStrategy::AutoPositive;
        self
    }

    /// Automatically search all standard negative ESI adducts ([M-H]-, [M+Cl]-, [M+HCOO]-, [M+CH3COO]-).
    pub fn with_auto_negative_adducts(mut self) -> Self {
        self.config.adduct_search_strategy = AdductSearchStrategy::AutoNegative;
        self
    }

    /// Fast primary [M+H]+ evaluation with automatic multi-adduct fallback if no confident identification
    pub fn with_two_tier_positive_adducts(mut self) -> Self {
        self.config.adduct_search_strategy = AdductSearchStrategy::TwoTierPositive;
        self
    }

    /// Fast primary [M-H]- evaluation with automatic multi-adduct fallback if no confident identification
    pub fn with_two_tier_negative_adducts(mut self) -> Self {
        self.config.adduct_search_strategy = AdductSearchStrategy::TwoTierNegative;
        self
    }

    /// Sets whether to filter out peaks at or above precursor m/z (+ 0.5 Da) prior to fragment tree search.
    pub fn with_filter_peaks_above_precursor(mut self, filter: bool) -> Self {
        self.config.filter_peaks_above_precursor = filter;
        self
    }

    /// Executes the full workflow on an mzML or MGF file.
    pub fn run_on_file<P: AsRef<Path>>(&self, path: P) -> io::Result<SiriusWorkflowResult> {
        let t_start = Instant::now();
        let path_str = path.as_ref().to_string_lossy().to_string();

        // 1. Ingest and configure experiment
        let mut experiment = match self.config.preset {
            WorkflowPreset::Metabolomics => SpectralExperiment::metabolomics(),
            WorkflowPreset::Proteomics => SpectralExperiment::proteomics(),
        };
        if self.config.filter_unknown_charge {
            experiment.workflow.add_operation(crate::workflow::DropUnknownCharge::new());
        }

        experiment.ingest_file(&path_str);
        let total_ingested = experiment.total_spectra_ingested;

        if total_ingested == 0 {
            return Ok(SiriusWorkflowResult {
                annotations: Vec::new(),
                total_spectra_ingested: 0,
                total_clusters_formed: 0,
                total_annotated_clusters: 0,
                elapsed_time_seconds: t_start.elapsed().as_secs_f64(),
                throughput_spectra_per_sec: 0.0,
            });
        }

        // 2. Cluster spectra
        experiment.prepare_clustering();
        let clusters = experiment.cluster();
        let total_clusters = clusters.num_clusters();

        // 3. Annotate clusters in parallel across all CPU cores
        let annotations: Vec<SiriusClusterAnnotation> = clusters
            .iter()
            .enumerate()
            .par_bridge()
            .filter(|(_, members)| members.len() >= self.config.min_cluster_size)
            .filter_map(|(cluster_id, members)| {
                self.annotate_cluster(&experiment, cluster_id, members)
            })
            .collect();

        let elapsed = t_start.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            total_ingested as f64 / elapsed
        } else {
            0.0
        };

        let total_annotated = annotations.len();
        Ok(SiriusWorkflowResult {
            annotations,
            total_spectra_ingested: total_ingested,
            total_clusters_formed: total_clusters,
            total_annotated_clusters: total_annotated,
            elapsed_time_seconds: elapsed,
            throughput_spectra_per_sec: throughput,
        })
    }

    /// Annotates a single cluster from the experiment, reconstructing consensus and MS1 envelope.
    pub fn annotate_cluster(
        &self,
        experiment: &SpectralExperiment,
        cluster_id: usize,
        members: &[usize],
    ) -> Option<SiriusClusterAnnotation> {
        let first_id = *members.first()?;
        let meta = experiment.metadata.get(first_id)?;

        let precursor_mz = meta.precursor_mz;
        if precursor_mz <= 0.0 {
            return None;
        }

        let rt = meta.retention_time_in_seconds;
        let charge = meta.charge;

        if charge == 0 && (self.config.filter_unknown_charge || self.config.unknown_charge_policy == UnknownChargePolicy::FilterOut) {
            return None;
        }

        let effective_adducts: Vec<Adduct> = if charge > 1 {
            vec![Adduct::protonated_with_charge(charge)]
        } else if charge < -1 {
            vec![Adduct::deprotonated_with_charge(charge.abs())]
        } else if charge != 0 {
            match &self.config.adduct_search_strategy {
                AdductSearchStrategy::Single(adduct) => vec![adduct.clone()],
                AdductSearchStrategy::AutoPositive => Adduct::common_positive(),
                AdductSearchStrategy::AutoNegative => Adduct::common_negative(),
                AdductSearchStrategy::TwoTierPositive => vec![Adduct::protonated()],
                AdductSearchStrategy::TwoTierNegative => vec![Adduct::deprotonated()],
                AdductSearchStrategy::Custom(list) => list.clone(),
            }
        } else {
            match self.config.unknown_charge_policy {
                UnknownChargePolicy::MultiChargeHypothesis => {
                    self.config
                        .candidate_charges
                        .iter()
                        .map(|&z| if z > 0 { Adduct::protonated_with_charge(z) } else { Adduct::deprotonated_with_charge(z.abs()) })
                        .collect()
                }
                _ => match &self.config.adduct_search_strategy {
                    AdductSearchStrategy::Single(adduct) => vec![adduct.clone()],
                    AdductSearchStrategy::AutoPositive => Adduct::common_positive(),
                    AdductSearchStrategy::AutoNegative => Adduct::common_negative(),
                    AdductSearchStrategy::TwoTierPositive => vec![Adduct::protonated()],
                    AdductSearchStrategy::TwoTierNegative => vec![Adduct::deprotonated()],
                    AdductSearchStrategy::Custom(list) => list.clone(),
                },
            }
        };

        // Build consensus spectrum or fallback to first member
        let consensus = experiment.build_consensus_spectrum(members, self.config.consensus_bin_width_da);
        let consensus_peaks = match consensus {
            Some(rec) => rec.peaks().to_vec(),
            None => {
                if let Some(spec) = experiment.spectra.get(first_id) {
                    Vec::new()
                } else {
                    Vec::new()
                }
            }
        };

        // Smart preprocessing: adaptive noise, deisotoping, entropy gating, diversity selection
        let effective_charge = if charge != 0 { charge } else { 1 };
        let clean_peaks = match self.preprocess_peaks_for_tree(&consensus_peaks, precursor_mz, effective_charge) {
            Some(peaks) => peaks,
            None => return None, // Spectrum too low-quality for tree construction
        };

        // Real MS1 precursor isotope envelope from the survey scan that triggered
        // this DDA event. Falls back to the MS2-consensus window (the surviving
        // precursor cluster) for MGF input or mzML without MS1.
        let ms1_real =
            experiment.ms1_envelope_for(meta.file_index, rt, precursor_mz, effective_charge);
        let ms1_from_survey_scan = ms1_real.len() >= 2;
        let ms1_for_scoring: &[Peak] = if ms1_from_survey_scan {
            &ms1_real
        } else {
            &consensus_peaks
        };

        let mut all_candidates = Vec::new();
        for adduct in &effective_adducts {
            let cands = self.annotate_peaks_with_adduct(
                &clean_peaks,
                precursor_mz,
                ms1_for_scoring,
                adduct,
            );
            all_candidates.extend(cands);
        }

        // Two-Tier fallback: if primary adduct yielded poor explanation, fall back to multi-adduct search
        let needs_fallback = match &self.config.adduct_search_strategy {
            AdductSearchStrategy::TwoTierPositive => {
                all_candidates.is_empty()
                    || all_candidates.first().map_or(true, |c| c.explained_intensity < 0.30 || c.tree_score <= 0.0)
            }
            AdductSearchStrategy::TwoTierNegative => {
                all_candidates.is_empty()
                    || all_candidates.first().map_or(true, |c| c.explained_intensity < 0.30 || c.tree_score <= 0.0)
            }
            _ => false,
        };

        if needs_fallback {
            let fallback_adducts = match &self.config.adduct_search_strategy {
                AdductSearchStrategy::TwoTierPositive => {
                    let mut list = Adduct::common_positive();
                    list.retain(|a| *a != Adduct::protonated());
                    list
                }
                AdductSearchStrategy::TwoTierNegative => {
                    let mut list = Adduct::common_negative();
                    list.retain(|a| *a != Adduct::deprotonated());
                    list
                }
                _ => Vec::new(),
            };
            for adduct in &fallback_adducts {
                let cands = self.annotate_peaks_with_adduct(
                    &clean_peaks,
                    precursor_mz,
                    ms1_for_scoring,
                    adduct,
                );
                all_candidates.extend(cands);
            }
        }

        all_candidates.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        all_candidates.truncate(self.config.top_k_candidates);

        let selected_adduct = all_candidates
            .first()
            .map(|c| c.adduct.clone())
            .unwrap_or_else(|| effective_adducts.first().cloned().unwrap_or_else(Adduct::protonated));
        let neutral_mass = selected_adduct.ion_mz_to_neutral_mass(precursor_mz);
        let z_abs = selected_adduct.charge.abs().max(1) as f64;
        // Prefer the real MS1 survey-scan envelope; fall back to the surviving
        // precursor cluster in the MS2 consensus.
        let ms1_envelope: Vec<Peak> = if ms1_real.len() >= 2 {
            ms1_real
        } else {
            consensus_peaks
                .iter()
                .filter(|p| {
                    (p.mz >= precursor_mz - (0.5 / z_abs))
                        && (p.mz <= precursor_mz + (4.5 / z_abs))
                })
                .copied()
                .collect()
        };

        Some(SiriusClusterAnnotation {
            cluster_id,
            member_count: members.len(),
            precursor_mz,
            neutral_mass,
            retention_time_seconds: rt,
            charge: if charge != 0 { charge } else { selected_adduct.charge },
            adduct: selected_adduct,
            top_candidates: all_candidates,
            consensus_peaks,
            ms1_envelope,
            ms1_from_survey_scan,
        })
    }

    // ── Smart Spectral Preprocessing ────────────────────────────────────────

    /// Preprocesses raw peaks for fragmentation tree construction with adaptive
    /// noise estimation, charge-aware precursor filtering, deisotoping, entropy
    /// gating, and diversity-aware peak selection.
    ///
    /// Returns `None` if the spectrum is too low-quality to produce a meaningful tree.
    /// The pipeline every formula-ID number in `tree_compare_FINDINGS.md` was
    /// measured through: a relative-intensity floor, the precursor ceiling, the
    /// most intense `max_peaks_per_tree`, and a dedupe. Nothing adaptive.
    fn preprocess_peaks_simple(&self, raw: &[Peak], precursor_mz: f64, charge: i8) -> Option<Vec<Peak>> {
        let max_int = raw.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-9);
        let mut peaks: Vec<Peak> = raw
            .iter()
            .filter(|p| p.intensity / max_int >= 0.005)
            .filter(|p| !self.config.filter_peaks_above_precursor || p.mz <= precursor_mz * charge.unsigned_abs().max(1) as f64 + 5.0)
            .copied()
            .collect();
        if peaks.is_empty() {
            return None;
        }
        peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
        peaks.truncate(self.config.max_peaks_per_tree);
        peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
        peaks.dedup_by(|a, b| (a.mz - b.mz).abs() < 1e-4);
        Some(peaks)
    }

    pub fn preprocess_peaks_for_tree(
        &self,
        raw_peaks: &[Peak],
        precursor_mz: f64,
        charge: i8,
    ) -> Option<Vec<Peak>> {
        if raw_peaks.is_empty() {
            return None;
        }

        if self.config.peak_preprocessing == PeakPreprocessing::Simple {
            return self.preprocess_peaks_simple(raw_peaks, precursor_mz, charge);
        }

        let mut peaks = raw_peaks.to_vec();
        peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));

        let max_int = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);

        // 1. Adaptive noise floor estimation
        //    Use the median of the bottom 50% of peak intensities as the noise baseline,
        //    then set threshold at 3× that level. Falls back to 0.5% relative intensity
        //    if there aren't enough peaks for a robust estimate.
        let noise_threshold = {
            let mut sorted_ints: Vec<f32> = peaks.iter().map(|p| p.intensity).collect();
            sorted_ints.sort_by(|a, b| a.total_cmp(b));

            if sorted_ints.len() >= 8 {
                let bottom_half = &sorted_ints[..sorted_ints.len() / 2];
                let median_noise = bottom_half[bottom_half.len() / 2];
                // Threshold = max(3× median noise, 0.2% of base peak)
                (median_noise * 3.0).max(max_int * 0.002)
            } else {
                max_int * 0.005
            }
        };

        peaks.retain(|p| p.intensity >= noise_threshold);

        if peaks.is_empty() {
            return None;
        }

        // 2. Charge-aware precursor filter
        //    For singly charged ions (z=1), fragments cannot exceed precursor m/z (+ isotope tolerance).
        //    For multiply-charged precursors (|z| > 1), fragments with lower charge states (e.g. z=1)
        //    can have m/z values up to the full neutral precursor mass + adduct.
        if self.config.filter_peaks_above_precursor {
            let z_abs = charge.unsigned_abs().max(1) as f64;
            let precursor_ceiling = if z_abs > 1.0 {
                (precursor_mz * z_abs) + 2.5
            } else {
                precursor_mz + 2.5
            };
            peaks.retain(|p| p.mz <= precursor_ceiling);
        }

        if peaks.is_empty() {
            return None;
        }

        // 3. Pre-tree deisotoping
        //    Remove isotope satellite peaks and fold their intensity into the
        //    monoisotopic peak. This prevents isotope peaks from consuming tree
        //    vertices without adding structural information.
        let z_max = charge.unsigned_abs().max(1) as u32;
        let iso_tol = match self.config.tolerance {
            MassTolerance::Da(d) => d.max(0.01),
            MassTolerance::Ppm(ppm) => (ppm * precursor_mz / 1e6).max(0.01),
        };

        if self.config.deisotope != DeisotopeMode::Off {
            let n = peaks.len();
            let mut is_isotope = vec![false; n];
            let c13_spacing = 1.00335; // C13-C12 mass difference
            // 37Cl-35Cl and 81Br-79Br. Both sit ~9.6 mDa below 2x the 13C step, so
            // under `Legacy` they are indistinguishable from a second 13C.
            const HALOGEN_STEPS: [f64; 2] = [1.99705, 1.99795];
            let aware = self.config.deisotope == DeisotopeMode::IsotopeAware;

            for i in 0..n {
                if is_isotope[i] {
                    continue;
                }
                // Look for isotope satellites at M + k*(1.003/z) for z in 1..=z_max, k in 1..=3
                for z in 1..=z_max {
                    let spacing = c13_spacing / z as f64;
                    let mut found_any = false;
                    for k in 1..=3u32 {
                        let target_mz = peaks[i].mz + spacing * k as f64;
                        // Binary search for the target
                        if let Ok(j) | Err(j) = peaks.binary_search_by(|p| p.mz.total_cmp(&target_mz)) {
                            // Check neighbors around insertion point
                            for &idx in &[j.wrapping_sub(1), j, j + 1] {
                                if idx < n && !is_isotope[idx] && idx != i {
                                    let diff = (peaks[idx].mz - target_mz).abs();
                                    // A satellite is only 13C if it is closer to the
                                    // 13C target than to a halogen step; otherwise it
                                    // is real element evidence and must survive.
                                    let halogen_closer = aware
                                        && HALOGEN_STEPS.iter().any(|h| {
                                            (peaks[idx].mz - (peaks[i].mz + h)).abs() < diff
                                        });
                                    if diff <= iso_tol && !halogen_closer {
                                        // Isotope peak should be less intense than monoisotopic
                                        if peaks[idx].intensity <= peaks[i].intensity * 1.5 {
                                            is_isotope[idx] = true;
                                            found_any = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if found_any {
                        break; // Found isotopes at this charge state, don't check higher charges
                    }
                }
            }

            // Remove isotope peaks (don't fold intensity — tree doesn't need it)
            let mut idx = 0;
            peaks.retain(|_| {
                let keep = !is_isotope[idx];
                idx += 1;
                keep
            });
        }

        if peaks.is_empty() {
            return None;
        }

        // 4. Spectral entropy gating
        //    Skip spectra dominated by a single peak (near-zero entropy) — these
        //    can't yield meaningful fragmentation trees.
        {
            let total_int: f32 = peaks.iter().map(|p| p.intensity).sum();
            if total_int > 0.0 {
                let entropy: f32 = peaks
                    .iter()
                    .map(|p| {
                        let frac = p.intensity / total_int;
                        if frac > 0.0 { -frac * frac.ln() } else { 0.0 }
                    })
                    .sum();
                let normalized_entropy = entropy / (peaks.len() as f32).ln().max(1e-6);

                // If > 95% of intensity is in a single peak, entropy ≈ 0
                if normalized_entropy < 0.05 && peaks.len() >= 3 {
                    return None;
                }
            }
        }

        // 5. Diversity-aware peak selection
        //    Instead of just taking the top-N by intensity, ensure we select peaks
        //    spread across the full m/z range to maximize structural coverage.
        let max_peaks = self.config.max_peaks_per_tree;
        if peaks.len() > max_peaks {
            // Split m/z range into windows and guarantee representation from each
            let mz_min = peaks.first().map(|p| p.mz).unwrap_or(0.0);
            let mz_max = peaks.last().map(|p| p.mz).unwrap_or(1.0);
            let mz_range = (mz_max - mz_min).max(1.0);

            // Use sqrt(max_peaks) windows, reserve 1 peak per window
            let n_windows = (max_peaks as f64).sqrt().ceil() as usize;
            let window_width = mz_range / n_windows as f64;

            let mut selected = Vec::with_capacity(max_peaks);
            let mut used = vec![false; peaks.len()];

            // Phase 1: Select the most intense peak from each m/z window
            for w in 0..n_windows {
                let w_lo = mz_min + w as f64 * window_width;
                let w_hi = w_lo + window_width;

                let best = peaks
                    .iter()
                    .enumerate()
                    .filter(|(j, p)| !used[*j] && p.mz >= w_lo && p.mz < w_hi)
                    .max_by(|(_, a), (_, b)| a.intensity.total_cmp(&b.intensity));

                if let Some((j, _)) = best {
                    selected.push(peaks[j]);
                    used[j] = true;
                }
            }

            // Phase 2: Fill remaining slots with highest-intensity unselected peaks
            let remaining = max_peaks.saturating_sub(selected.len());
            if remaining > 0 {
                let mut unselected: Vec<(usize, &Peak)> = peaks
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| !used[*j])
                    .collect();
                unselected.sort_by(|(_, a), (_, b)| b.intensity.total_cmp(&a.intensity));

                for (j, p) in unselected.into_iter().take(remaining) {
                    selected.push(*p);
                }
            }

            selected.sort_by(|a, b| a.mz.total_cmp(&b.mz));
            peaks = selected;
        }

        if peaks.is_empty() {
            return None;
        }

        Some(peaks)
    }

    /// Annotates candidate precursor formulas, MS1 isotope fit, and fragmentation trees for a specific adduct hypothesis.
    pub fn annotate_peaks_with_adduct(
        &self,
        clean_peaks: &[Peak],
        precursor_mz: f64,
        ms1_peaks: &[Peak],
        adduct: &Adduct,
    ) -> Vec<AnnotatedFormulaCandidate> {
        let neutral_mass = adduct.ion_mz_to_neutral_mass(precursor_mz);
        let z_abs = adduct.charge.abs().max(1) as f64;
        let ms1_envelope: Vec<Peak> = ms1_peaks
            .iter()
            .filter(|p| (p.mz >= precursor_mz - (0.5 / z_abs)) && (p.mz <= precursor_mz + (4.5 / z_abs)))
            .copied()
            .collect();

        let mut candidate_precursors = if self.config.enable_halogens {
            self.gen_halogens.generate_candidates(
                neutral_mass,
                self.config.tolerance,
                &self.config.rules,
            )
        } else {
            let mut cands = self.gen_chno.generate_candidates(
                neutral_mass,
                self.config.tolerance,
                &self.config.rules,
            );
            let chnops_cands = self.gen_chnops.generate_candidates(
                neutral_mass,
                self.config.tolerance,
                &self.config.rules,
            );
            // O(n) dedup via HashSet instead of O(n²) contains()
            let existing: std::collections::HashSet<Formula> = cands.iter().copied().collect();
            for cand in chnops_cands {
                if !existing.contains(&cand) {
                    cands.push(cand);
                }
            }
            cands
        };

        if candidate_precursors.is_empty() {
            return Vec::new();
        }

        if candidate_precursors.len() > self.config.max_candidates_for_tree_search {
            candidate_precursors.sort_by_cached_key(|formula| {
                let prior = self.scorer.score_prior(formula);
                let iso = if !ms1_envelope.is_empty() {
                    self.tree_builder.isotope_score(formula, &ms1_envelope, precursor_mz, adduct)
                } else { 0.0 };
                let mass = self.scorer.score_mass_accuracy(formula.exact_mass(), neutral_mass);
                // Ordered f64 bits, descending, matching total_cmp without recomputation.
                let score = prior + iso * self.config.isotope_weight + mass;
                let bits = score.to_bits() as i64;
                std::cmp::Reverse(bits ^ (((bits >> 63) as u64 >> 1) as i64))
            });
            candidate_precursors.truncate(self.config.max_candidates_for_tree_search);
        }

        let scored_trees = self.tree_builder.build_trees_with_adduct_and_ms1(
            &candidate_precursors,
            precursor_mz,
            clean_peaks,
            &ms1_envelope,
            adduct,
        );

        let mut preliminary_candidates = Vec::with_capacity(scored_trees.len());
        for (formula, tree) in scored_trees {
            let exact_mass = formula.exact_mass();
            let mass_err_da = (exact_mass - neutral_mass).abs();
            let mass_err_ppm = (mass_err_da / neutral_mass) * 1e6;

            let iso_score = if !ms1_envelope.is_empty() {
                self.tree_builder.isotope_score(&formula, &ms1_envelope, precursor_mz, adduct)
            } else {
                0.0
            };

            let prior_score = self.scorer.score_prior(&formula);
            let adduct_prior = adduct.prior_score();
            // `tree.total_score` already carries the precursor's mass accuracy and
        // chemical prior and isotope likelihood in its root. Only a non-unit
        // isotope weight adjustment and the adduct prior are new evidence here.
        let prelim_score = tree.total_score + (iso_score * (self.config.isotope_weight - 1.0)) + adduct_prior;

            preliminary_candidates.push((formula, tree, exact_mass, mass_err_ppm, mass_err_da, iso_score, prior_score, adduct_prior, prelim_score));
        }

        // Sort by preliminary score descending
        preliminary_candidates.sort_by(|a, b| b.8.total_cmp(&a.8));

        // Evaluate neural network & database lookup only on the top candidate hypotheses
        let top_eval_limit = (self.config.top_k_candidates + 2).min(preliminary_candidates.len());
        let mut candidates = Vec::with_capacity(preliminary_candidates.len());

        for (i, (formula, tree, exact_mass, mass_err_ppm, mass_err_da, iso_score, prior_score, adduct_prior, prelim_score)) in preliminary_candidates.into_iter().enumerate() {
            let should_run_nn = i < top_eval_limit;

            let (predicted_fingerprint, classification) = if should_run_nn {
                if let Some(ref predictor) = self.fingerprint_predictor {
                    let extractor = TreeFeatureExtractor::default();
                    let features = extractor.extract(&tree);
                    let mut pred_fp = predictor.predict(&features.features);
                    let class = if self.config.enable_compound_classification {
                        Some(self.canopus_classifier.classify(&pred_fp))
                    } else {
                        None
                    };
                    (Some(pred_fp), class)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let mut structural_candidates = Vec::new();
            let mut structure_boost = 0.0f64;

            if let Some(ref pred_fp) = predicted_fingerprint {
                if let Some(ref db) = self.structure_database {
                    structural_candidates = db.query_and_rank(
                        &formula,
                        pred_fp,
                        &self.fingerprint_searcher,
                        self.config.sample_context,
                        5,
                    );
                    if let Some(top_match) = structural_candidates.first() {
                        if top_match.mod_platt_score > 0.0 {
                            structure_boost = (top_match.mod_platt_score / 50.0) * self.config.structure_score_weight;
                        }
                    }
                }
            }

            let total_score = prelim_score + structure_boost;

            candidates.push(AnnotatedFormulaCandidate {
                formula,
                adduct: adduct.clone(),
                exact_mass,
                mass_error_ppm: mass_err_ppm,
                mass_error_da: mass_err_da,
                tree_score: tree.total_score - iso_score,
                isotope_score: iso_score,
                prior_score,
                adduct_prior_score: adduct_prior,
                total_score,
                explained_intensity: tree.explained_intensity,
                explained_peaks: tree.explained_peaks,
                total_peaks: tree.total_peaks,
                tree,
                predicted_fingerprint,
                classification,
                structural_candidates,
            });
        }

        candidates.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        candidates
    }

    /// Annotates an arbitrary single spectrum with candidate formulas, isotopic envelope, and trees.
    pub fn annotate_spectrum(
        &self,
        record: &MassSpectralRecord,
        precursor_mz: f64,
        ms1_envelope: Option<&[Peak]>,
    ) -> Option<SiriusClusterAnnotation> {
        let charge = record.get_charge();

        if charge == 0 && (self.config.filter_unknown_charge || self.config.unknown_charge_policy == UnknownChargePolicy::FilterOut) {
            return None;
        }

        let known_adduct = record.get_metadata("ADDUCT").or(record.get_metadata("PRECURSORTYPE"))
            .and_then(|name| name.parse::<Adduct>().ok());
        let effective_adducts: Vec<Adduct> = if let Some(ref adduct) = known_adduct {
            vec![adduct.clone()]
        } else if charge > 1 {
            vec![Adduct::protonated_with_charge(charge)]
        } else if charge < -1 {
            vec![Adduct::deprotonated_with_charge(charge.abs())]
        } else if charge != 0 {
            match &self.config.adduct_search_strategy {
                AdductSearchStrategy::Single(adduct) => vec![adduct.clone()],
                AdductSearchStrategy::AutoPositive => Adduct::common_positive(),
                AdductSearchStrategy::AutoNegative => Adduct::common_negative(),
                AdductSearchStrategy::TwoTierPositive => vec![Adduct::protonated()],
                AdductSearchStrategy::TwoTierNegative => vec![Adduct::deprotonated()],
                AdductSearchStrategy::Custom(list) => list.clone(),
            }
        } else {
            match self.config.unknown_charge_policy {
                UnknownChargePolicy::MultiChargeHypothesis => {
                    self.config
                        .candidate_charges
                        .iter()
                        .map(|&z| if z > 0 { Adduct::protonated_with_charge(z) } else { Adduct::deprotonated_with_charge(z.abs()) })
                        .collect()
                }
                _ => match &self.config.adduct_search_strategy {
                    AdductSearchStrategy::Single(adduct) => vec![adduct.clone()],
                    AdductSearchStrategy::AutoPositive => Adduct::common_positive(),
                    AdductSearchStrategy::AutoNegative => Adduct::common_negative(),
                    AdductSearchStrategy::TwoTierPositive => vec![Adduct::protonated()],
                    AdductSearchStrategy::TwoTierNegative => vec![Adduct::deprotonated()],
                    AdductSearchStrategy::Custom(list) => list.clone(),
                },
            }
        };

        // Smart preprocessing: adaptive noise, deisotoping, entropy gating, diversity selection
        let effective_charge = known_adduct.as_ref().map(|a| a.charge).unwrap_or(if charge != 0 { charge } else { 1 });
        let clean_peaks = match self.preprocess_peaks_for_tree(record.peaks(), precursor_mz, effective_charge) {
            Some(peaks) => peaks,
            None => return None, // Spectrum too low-quality for tree construction
        };

        let ms1_peaks = ms1_envelope.unwrap_or(&[]);
        let mut all_candidates = Vec::new();
        for adduct in &effective_adducts {
            let cands = self.annotate_peaks_with_adduct(
                &clean_peaks,
                precursor_mz,
                ms1_peaks,
                adduct,
            );
            all_candidates.extend(cands);
        }

        // Two-Tier fallback: if primary adduct yielded poor explanation, fall back to multi-adduct search
        let needs_fallback = match &self.config.adduct_search_strategy {
            AdductSearchStrategy::TwoTierPositive => {
                all_candidates.is_empty()
                    || all_candidates.first().map_or(true, |c| c.explained_intensity < 0.30 || c.tree_score <= 0.0)
            }
            AdductSearchStrategy::TwoTierNegative => {
                all_candidates.is_empty()
                    || all_candidates.first().map_or(true, |c| c.explained_intensity < 0.30 || c.tree_score <= 0.0)
            }
            _ => false,
        };

        if needs_fallback && known_adduct.is_none() {
            let fallback_adducts = match &self.config.adduct_search_strategy {
                AdductSearchStrategy::TwoTierPositive => {
                    let mut list = Adduct::common_positive();
                    list.retain(|a| *a != Adduct::protonated());
                    list
                }
                AdductSearchStrategy::TwoTierNegative => {
                    let mut list = Adduct::common_negative();
                    list.retain(|a| *a != Adduct::deprotonated());
                    list
                }
                _ => Vec::new(),
            };
            for adduct in &fallback_adducts {
                let cands = self.annotate_peaks_with_adduct(
                    &clean_peaks,
                    precursor_mz,
                    ms1_peaks,
                    adduct,
                );
                all_candidates.extend(cands);
            }
        }

        all_candidates.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        all_candidates.truncate(self.config.top_k_candidates);

        let selected_adduct = all_candidates
            .first()
            .map(|c| c.adduct.clone())
            .unwrap_or_else(|| effective_adducts.first().cloned().unwrap_or_else(Adduct::protonated));
        let neutral_mass = selected_adduct.ion_mz_to_neutral_mass(precursor_mz);

        Some(SiriusClusterAnnotation {
            cluster_id: 0,
            member_count: 1,
            precursor_mz,
            neutral_mass,
            retention_time_seconds: 0.0,
            charge: if charge != 0 { charge } else { selected_adduct.charge },
            adduct: selected_adduct,
            top_candidates: all_candidates,
            consensus_peaks: record.peaks().to_vec(),
            ms1_envelope: ms1_peaks.to_vec(),
            ms1_from_survey_scan: !ms1_peaks.is_empty(),
        })
    }
}
