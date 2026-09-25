// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Contains code ported from SIRIUS <https://github.com/sirius-ms/sirius>,
// which is licensed under the GNU Lesser General Public License v3 or later:
//
//   Copyright (C) 2013-2020 Kai Dührkop, Markus Fleischauer, Marcus Ludwig,
//   Martin A. Hoffman, Fleming Kretschmer and Sebastian Böcker, Chair of
//   Bioinformatics, Friedrich-Schiller University Jena.
//   Modifications Copyright (C) 2026 Manuel Uhlir: ported to Rust and adapted
//   for OxideMS.
//
// This file is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See LICENSES/LGPL-3.0-or-later.txt and
// LICENSES/GPL-3.0-or-later.txt in the repository root, and ATTRIBUTION.md.

use super::adduct::Adduct;
use super::annotation::{AnnotatedPeak, SpectrumAnnotator};
use super::formula::Formula;
use super::rules::{ChemicalRules, LossGate};
use super::scoring::FormulaScorer;
use super::tree::{FragmentationTree, TreeEdge, TreeNode};
use crate::spectrum::Peak;
use crate::workflow::MassTolerance;
use std::collections::{HashMap, HashSet};

/// Configuration parameters for Fragmentation Tree building and scoring
#[derive(Debug, Clone)]
pub struct TreeSolverConfig {
    pub tolerance: MassTolerance,
    pub adduct: Adduct,
    /// Expected mass accuracy of the instrument, in ppm. This is how tightly a
    /// candidate's theoretical mass is expected to match a measured one, and is
    /// deliberately not `tolerance`: tolerance is how wide a net the decomposer
    /// casts, sigma is how much a deviation inside that net counts against a
    /// candidate. Tying the two together, as this used to, made every candidate
    /// inside the tolerance window score almost alike. SIRIUS searches wide and
    /// scores at sigma = 10 ppm.
    pub mass_sigma_ppm: f64,
    /// Width, in log space, of the neutral-loss mass prior. Smaller is stricter
    /// about large one-step losses. See `loss_mass_penalty`. Ignored when
    /// `learned_loss_model` is set — the learned frequencies already encode size.
    pub loss_mass_sigma_log: f64,
    /// Score neutral losses with the data-learned, ionization-mode-specific model
    /// (`loss_model`) instead of the 35-entry hand list + blunt size penalty. A
    /// large loss then keeps its bonus only if that exact formula is a frequent
    /// observed loss (a hexose is; an arbitrary alkyl chunk a decoy needs is not).
    pub learned_loss_model: bool,
    /// Multiplier on the per-fragment noise floor subtracted from every vertex
    /// weight. < 1.0 admits more low-intensity peaks (higher fragment recall,
    /// lower precision); > 1.0 keeps only strong evidence. Measured against MoNA
    /// curator annotations, recall sits at ~43% at 1.0 and this is the lever that
    /// moves it, not the loss model.
    pub noise_floor_scale: f64,
    /// Fold the MS1 isotope-envelope log-likelihood into the root score when an
    /// isotope pattern is supplied (`build_trees_for_candidates_with_ms1`). No
    /// effect on MS2-only calls. The M+1 carbon bump and the M+2 step of Cl/Br/S
    /// are a strong, orthogonal formula discriminator — SIRIUS's headline CASMI
    /// accuracy leans on it.
    pub use_isotopes: bool,
    /// Which algorithm selects the tree out of the scored fragment graph.
    pub solver: SolverKind,
    /// Which objective scores the fragment graph.
    pub scoring: ScoringModel,
    /// Constant per-fragment bonus for [`ScoringModel::Sirius`] — SIRIUS's
    /// `TreeSizeScorer`, which it auto-tunes in `Beautification`. A vertex is
    /// worth including when its own scorers plus this clear zero. Higher = bigger
    /// trees / more explained intensity. No effect under `ScoringModel::Native`.
    pub tree_size_bonus: f64,
    /// Two-pass candidate solving: score every candidate with the fast
    /// Critical-Path heuristic, then re-solve only the top N with
    /// [`Self::solver`]. `None` runs the configured solver on every candidate.
    ///
    /// An exact solve per candidate is unaffordable de-novo — a wide alphabet puts
    /// thousands of candidates on a spectrum, and an ILP each does not finish.
    /// SIRIUS has the same structure: it shortlists, then runs the exact solver on
    /// `extractTopResults(results, max(numberOfResultsToKeep, 10), 5)`.
    ///
    /// The shortlist is the risk: a true formula that the heuristic ranks below N
    /// never gets upgraded, so N trades runtime against recall of the reranker.
    pub exact_top_n: Option<usize>,
    /// Which neutral losses may become graph edges. See
    /// [`super::rules::LossGate`] — SIRIUS has no such gate at all.
    pub loss_gate: LossGate,
    /// SIRIUS's `MedianNoiseIntensity`, relative to the base peak, feeding
    /// `ClippedPeakIsNoiseScorer` — the Pareto noise term that makes an intense
    /// fragment worth more as evidence than a trace one. `None` leaves the
    /// [`ScoringModel::Sirius`] vertex weight with no intensity term at all, which
    /// is what it had before.
    ///
    /// A parameter rather than a constant: SIRIUS 5.6.3 defaults the annotation to
    /// `NaN` and nothing in the distribution sets it, so there is no faithful value
    /// to copy. 0.005, our own pipeline's intensity floor, is the best of the
    /// values swept (0.003 / 0.005 / 0.01 / 0.02).
    ///
    /// **Off by default — it trades ranking for tree completeness.** With
    /// `Some(0.005)` the true tree's explained intensity rises sharply (CASMI
    /// 93.3 -> 96.7 % mean, GNPS holdout 77.0 -> 88.7 %) because intense peaks
    /// become worth including. Formula *ranking* pays for it: CASMI top-1 is a wash
    /// (1 spectrum each way) but GNPS holdout drops **-9.7 pts [-12.3, -7.1]**,
    /// 64 lost against 6 won, McNemar p < 0.001.
    ///
    /// The base-peak bonus is `-ln(beta) ~ 11.5` against vertex scores of ~4, so it
    /// dominates the objective and every candidate's tree grows to take in whatever
    /// it can explain — which helps a decoy with many decomposable peaks more than
    /// it helps the truth. SIRIUS sums this with `TreeSizeScorer` into one peak
    /// score and can carry it because its peak processing and decomposition limits
    /// differ; ours cannot at a 12-peak budget. Turn it on when the fragment
    /// annotation matters more than the formula ranking. See FINDINGS round 23.
    pub peak_noise_median: Option<f64>,
    /// Re-fit a per-candidate linear m/z calibration against that candidate's own
    /// tree and rebuild it on the corrected masses, keeping whichever tree scores
    /// higher. SIRIUS's `HypothesenDrivenRecalibration`.
    ///
    /// **Off by default, on measurement — it is harmful on every dataset tried.**
    /// CASMI 2016: -1 spectrum top-1 at 12 peaks and at 40, gaining none. GNPS
    /// holdout: **-7.0 pts [-9.3, -4.7]**, 47 spectra lost against 5 won,
    /// McNemar p < 0.001.
    ///
    /// The reason is that there is nothing left to correct. After the drift pass
    /// above, CASMI residuals have a median offset of +0.11 ppm with only 2 of 80
    /// spectra past 1 ppm, and a residual slope of -0.10 ppm/300 Da over an IQR of
    /// [-1.90, +0.45] — symmetric around zero, so per-spectrum noise rather than an
    /// instrument calibration curve. Fitting two free parameters to ~9 reference
    /// points that already sit inside +/-0.5 ppm bends the masses toward whichever
    /// hypothesis is being tested, and a decoy benefits as readily as the truth.
    ///
    /// Note that GNPS is the drifted data, not the clean case — the global median
    /// pass is what handles the 19 ppm negative-mode drift, and it handles it
    /// first. Kept flag-guarded so the negative result stays reproducible rather
    /// than being re-derived; delete it if nothing earns it. See FINDINGS round 22.
    pub recalibrate: bool,
}

/// The scoring objective the tree is optimised against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoringModel {
    /// This crate's own objective (intensity-weighted vertices + hand/learned
    /// loss dict + adaptive noise floor), tuned round-by-round against GNPS /
    /// MoNA / CASMI.
    #[default]
    Native,
    /// SIRIUS's fragmentation-tree objective with its distributed parameter
    /// tables, ported from SIRIUS (`sirius_scoring`). No intensity term; a
    /// fixed tree-size bonus replaces the noise floor.
    Sirius,
}

/// Tree-selection algorithm for the Maximum Colorful Subtree problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverKind {
    /// Critical-Path heuristic (Dührkop et al., WABI 2018). Fast, approximate.
    #[default]
    CriticalPath,
    /// Pick the best candidate per peak, then the exact max-weight rooted subtree
    /// of the resulting single-colour-per-vertex DAG by dynamic programming
    /// (O(V·E)). Exact given the vertex selection; the colourful constraint is
    /// satisfied trivially.
    DagDp,
    /// Exact Maximum Colorful Subtree as an integer program, solved with `microlp`
    /// — pure Rust, so the default build needs no cmake, no C++ toolchain and no
    /// libclang.
    /// Same optimum as [`SolverKind::Exact`], reached without the colour cap: the
    /// fragment graph is a DAG (an edge strictly decreases fragment mass), so the
    /// compact formulation of Rauf et al. 2013 needs no subtour elimination and
    /// stays polynomial in size. This is how SIRIUS solves it — its own log lists
    /// `Treebuilder priorities: [CLP, GUROBI, CPLEX]`.
    Ilp,
    /// Exact Maximum Colorful Subtree by the Böcker–Rasche colour-subset DP
    /// (O(3^k·V + 2^k·E)), the true optimum SIRIUS's ILP also targets. Falls back
    /// to [`SolverKind::CriticalPath`] above `EXACT_COLOR_CAP` colours.
    Exact,
}

/// Colour (distinct-peak) ceiling for [`SolverKind::Exact`]; above this the
/// subset DP's 3^k term is impractical and it falls back to Critical-Path.
/// 14 -> 3^14 ≈ 4.8M subset ops per candidate formula, fine for a pinned tree,
/// slow for a full de-novo candidate sweep — use a small `--max-peaks` there.
pub const EXACT_COLOR_CAP: usize = 14;

// SIRIUS `FasterTreeComputationInstance` beautification constants (from the
// SIRIUS 5.6.3 release). The tree-size bonus starts at the profile's `TreeSizeScorer`
// score and is raised in unit steps until a top tree is "high quality" or the
// bonus reaches `MAX_TREESIZE`.
/// Profile `TreeSizeScorer.score`.
const SIRIUS_TREE_SIZE_START: f64 = -0.5;
/// `TREE_SIZE_INCREASE`.
const SIRIUS_TREE_SIZE_STEP: f64 = 1.0;
/// `MAX_TREESIZE` — the bonus never exceeds this.
const SIRIUS_TREE_SIZE_MAX: f64 = 2.5;
/// `MIN_NUMBER_OF_TREES_CHECK_FOR_INTENSITY`.
const SIRIUS_BEAUTIFY_TOP_N: usize = 5;
/// `MIN_EXPLAINED_INTENSITY`.
const SIRIUS_MIN_EXPLAINED_INTENSITY: f64 = 0.7;
/// `MIN_NUMBER_OF_EXPLAINED_PEAKS` — the node-count target is capped at this.
const SIRIUS_MIN_EXPLAINED_PEAKS: usize = 15;

impl Default for TreeSolverConfig {
    fn default() -> Self {
        Self {
            tolerance: MassTolerance::Ppm(20.0),
            adduct: Adduct::protonated(),
            mass_sigma_ppm: 2.0,
            // 0.45: on the blind CASMI 2016 set the tight large-loss prior gains
            // +6 pts formula top-1 over the learned model; 0.45 vs 0.31 keeps
            // explained intensity at parity on MoNA. See tree_compare_FINDINGS.md
            // round 7.
            loss_mass_sigma_log: 0.45,
            // The data-learned loss table (round 5) is GNPS-overfit: a wash with
            // the hand list on GNPS-shaped data, 6 pts worse on the blind CASMI
            // set. Still available behind this flag.
            learned_loss_model: false,
            noise_floor_scale: 1.0,
            use_isotopes: true,
            // Round 15: SIRIUS's published objective + the exact Maximum Colorful
            // Subtree optimum are the default. On the blind CASMI 2016 set this is
            // +2.6 formula top-1 over Native+Critical, +7 on the GNPS holdout, at
            // parity on explained intensity, and the trees line up with SIRIUS
            // 5.6.3 (loss-edge Jaccard 0.41 -> 0.62).
            //
            // The optimum is reached through the ILP, not `SolverKind::Exact`.
            // `Exact` falls back to Critical-Path above `EXACT_COLOR_CAP` = 14
            // colours, and at a realistic 60-peak budget nearly every candidate is
            // over that line, so the `Exact` default was in fact the heuristic.
            // MassBank, 60 peaks, 10 ppm, non-F, everything else equal: 85.8 % with
            // `Exact` against 88.1 % here, n=1784, 57 w / 17 l, exact McNemar
            // p = 0.000, and on the true formula the ILP finds a strictly
            // higher-objective tree on 1111 of 1768 spectra and a worse one on 1.
            // Round 54.4 recorded that 2.3-point gap as an unexplained
            // non-reproduction of the published number; this is its cause.
            solver: SolverKind::Ilp,
            scoring: ScoringModel::Sirius,
            tree_size_bonus: 3.0,
            // An ILP per candidate is unaffordable de-novo, so shortlist with the
            // heuristic and re-solve the leaders exactly -- SIRIUS's own structure.
            // 20 is the value the published 88.1 % was produced at.
            exact_top_n: Some(20),
            loss_gate: LossGate::default(),
            peak_noise_median: None,
            recalibrate: false,
        }
    }
}

/// How many times to re-estimate the drift. One pass corrects an ordinary offset;
/// a second finishes a drift larger than the tolerance, where the first estimate is
/// necessarily capped by the tolerance window.
const MAX_RECALIBRATION_PASSES: usize = 2;

/// Anchors needed before a drift estimate is trusted. Too few and the median is
/// just one mis-annotated peak.
const MIN_RECALIBRATION_ANCHORS: usize = 5;

/// Drift below this is not worth a second decomposition pass, in ppm.
const MIN_RECALIBRATION_DRIFT_PPM: f64 = 1.0;

/// How far to widen the tolerance when the ordinary pass left too few anchors to
/// estimate a drift from. Paid only by spectra that look drifted or sparse.
const RECALIBRATION_SEARCH_FACTOR: f64 = 3.0;

/// Scale a tolerance, whichever unit it is expressed in.
fn widen(tolerance: MassTolerance, factor: f64) -> MassTolerance {
    match tolerance {
        MassTolerance::Ppm(ppm) => MassTolerance::Ppm(ppm * factor),
        MassTolerance::Da(da) => MassTolerance::Da(da * factor),
    }
}

/// Peaks with at least one decomposition, and so able to anchor a drift estimate.
fn anchor_count(annotated: &[AnnotatedPeak]) -> usize {
    annotated.iter().filter(|p| !p.candidates.is_empty()).count()
}

/// Estimate the spectrum's systematic mass drift, in ppm, from the peaks that
/// decomposed: each peak's best explanation is an anchor, and the median of their
/// signed deviations is the offset. Returns `None` when there is too little
/// evidence, or too little drift to be worth acting on.
///
/// A drifted spectrum charges every candidate the same mass penalty, so formula
/// *ranking* survives it — but the tree does not. At 15 ppm tolerance a 19 ppm
/// drift leaves most peaks with no decomposition at all and the tree collapses to
/// its root, which is what several negative-mode GNPS spectra were doing. SIRIUS
/// recalibrates each spectrum against its own best theoretical explanations before
/// scoring; this is the same idea, run once against the union bound of all
/// candidates rather than per candidate.
fn estimate_drift_ppm(annotated: &[AnnotatedPeak]) -> Option<f64> {
    let mut deviations: Vec<f64> = annotated
        .iter()
        .filter_map(|p| {
            // `candidates` is sorted by score, and for a fragment that is mass
            // accuracy and chemical prior — the best explanation available.
            let best = p.candidates.first()?;
            Some(best.mass_error_ppm)
        })
        .collect();

    if deviations.len() < MIN_RECALIBRATION_ANCHORS {
        return None;
    }

    deviations.sort_by(f64::total_cmp);
    let mid = deviations.len() / 2;
    let median = if deviations.len() % 2 == 0 {
        (deviations[mid - 1] + deviations[mid]) / 2.0
    } else {
        deviations[mid]
    };

    (median.abs() >= MIN_RECALIBRATION_DRIFT_PPM).then_some(median)
}

/// SIRIUS `HypothesenDrivenRecalibration` defaults, from the SIRIUS 5.6.3 release: the
/// strategy is `MedianSlope(Deviation(2 ppm, 5e-4 Da), minNumberOfPeaks = 8,
/// threshold = 0.01)`, inheriting `AbstractRecalibrationStrategy`'s
/// `maxDeviation = Deviation(10 ppm, 1e-3 Da)`. Only `maxDeviation` and
/// `minNumberOfPeaks` reach the median-slope path: the epsilon array
/// `MedianSlope::recalibrate` builds is dead once it calls
/// `getMedianSubsetFairDistributed` rather than `maxIntervalStabbing`.
const RECAL_MAX_DEV_PPM: f64 = 10.0;
const RECAL_MAX_DEV_DA: f64 = 1e-3;
const RECAL_MIN_REFERENCE_PEAKS: usize = 8;

/// `getMedianSubsetFairDistributed`: bin the reference peaks by `floor(mz / 100)`
/// and take the most intense `max(5, |median bin|)` out of each, so that a dense
/// cluster of low-mass fragments cannot carry the slope on its own.
const RECAL_BIN_WIDTH_DA: f64 = 100.0;
const RECAL_MIN_PER_BIN: usize = 5;

/// `extractTopResults(results, max(numberOfResultsToKeep, 10), 5)`: recalibration
/// is a per-candidate hypothesis and costs a re-decomposition each, so SIRIUS runs
/// it on the leaders only.
const RECAL_TOP_N: usize = 10;

/// Median of a slice, averaging the two middles on an even count — the same
/// convention `estimate_drift_ppm` uses.
fn median_of(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

/// `MedianSlope.getMedianSubsetFairDistributed`. `refs` are
/// `(measured mz, theoretical mz, relative intensity)`.
fn fair_distributed_subset(refs: &[(f64, f64, f32)]) -> Vec<(f64, f64)> {
    if refs.is_empty() {
        return Vec::new();
    }
    let max_mz = refs.iter().map(|r| r.0).fold(0.0f64, f64::max);
    let n_bins = ((max_mz / RECAL_BIN_WIDTH_DA).ceil() as usize).max(1);
    let mut bins: Vec<Vec<(f64, f64, f32)>> = vec![Vec::new(); n_bins];
    for &r in refs {
        let b = ((r.0 / RECAL_BIN_WIDTH_DA).floor() as usize).min(n_bins - 1);
        bins[b].push(r);
    }
    for bin in &mut bins {
        // MedianSlope$1: intensity, descending.
        bin.sort_by(|a, b| b.2.total_cmp(&a.2));
    }
    let mut sizes: Vec<usize> = bins.iter().map(Vec::len).collect();
    sizes.sort_unstable();
    let limit = RECAL_MIN_PER_BIN.max(sizes[sizes.len() / 2]);
    bins.iter()
        .flat_map(|bin| bin.iter().take(limit).map(|&(x, y, _)| (x, y)))
        .collect()
}

/// `MzRecalibration.getMedianLinearRecalibration`: a Theil-Sen fit — the median
/// of the pairwise slopes, then the median intercept given that slope. Returns
/// `(intercept, slope)` for `f(mz) = intercept + slope * mz`.
///
/// Robust by construction: a single mis-assigned fragment shifts a least-squares
/// line but not a median of pairwise slopes.
fn median_linear_recalibration(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return None;
    }
    let mut slopes = Vec::with_capacity(points.len() * (points.len() - 1) / 2);
    for i in 0..points.len() - 1 {
        for j in i + 1..points.len() {
            let dx = points[j].0 - points[i].0;
            if dx != 0.0 {
                slopes.push((points[j].1 - points[i].1) / dx);
            }
        }
    }
    let slope = median_of(&mut slopes)?;
    let mut intercepts: Vec<f64> = points.iter().map(|&(x, y)| y - slope * x).collect();
    let intercept = median_of(&mut intercepts)?;
    (slope.is_finite() && intercept.is_finite()).then_some((intercept, slope))
}

/// Fit a recalibration for one candidate against its own tree: every tree node is
/// a hypothesis that this measured peak *is* that formula, so the pair (measured
/// m/z, theoretical m/z) is a reference point.
///
/// `None` when the tree offers too little evidence, in which case SIRIUS returns
/// the identity function and the candidate keeps its original tree.
fn recalibration_from_tree(tree: &FragmentationTree, adduct: &Adduct) -> Option<(f64, f64)> {
    let parent_mz = tree.nodes.iter().map(|n| n.mz).fold(f64::NEG_INFINITY, f64::max);
    let refs: Vec<(f64, f64, f32)> = tree
        .nodes
        .iter()
        .filter_map(|n| {
            let theo = adduct.neutral_mass_to_ion_mz(n.formula.exact_mass());
            // `AbstractRecalibrationStrategy.preprocess`: the parent peak is kept
            // unconditionally; any other reference deviating by more than
            // `maxDeviation` is a bad hypothesis and is dropped from the fit.
            if n.mz < parent_mz - 0.5 {
                let allowed = (RECAL_MAX_DEV_PPM * n.mz * 1e-6).max(RECAL_MAX_DEV_DA);
                if (n.mz - theo).abs() > allowed {
                    return None;
                }
            }
            Some((n.mz, theo, n.relative_intensity))
        })
        .collect();

    let subset = fair_distributed_subset(&refs);
    if subset.len() < RECAL_MIN_REFERENCE_PEAKS {
        return None;
    }
    median_linear_recalibration(&subset)
}

/// Mode of the neutral loss mass distribution, 55.84 Da: Böcker & Dührkop fit a
/// log-normal with mu = 4.02, sigma = 0.31 to observed losses (J Cheminform 2016).
const LOSS_MASS_MODE_DA: f64 = 55.84;

/// Penalty for a neutral loss being implausibly large for a single fragmentation
/// step, as the log-normal loss mass prior's log-density relative to its mode.
///
/// Applied on one side only. Below the mode the prior would penalise water,
/// ammonia and CO2 — the losses the curated dictionary exists to reward — and
/// those scores are calibrated against no size penalty at all. Above the mode
/// there was effectively no penalty before: the old linear term charged 0.015
/// for a 300 Da loss, so a whole sugar chain leaving in one step cost nothing.
///
/// The default is Böcker & Dührkop's fitted sigma = 0.31. An earlier note here
/// claimed ranking accuracy was flat from 0.31 to 1.0 and kept 1.0 for tree size;
/// re-measured on 600 held-out GNPS spectra (after the fragment-alphabet and
/// radical-convention fixes) that is no longer true — de-novo formula top-1 runs
/// 67.9% at 0.31 vs 64.7% at 1.0, and intensity-weighted fragment-substructure
/// precision 83.3% vs 81.6%, for a ~5-point drop in explained intensity (86% vs
/// 91%, still SIRIUS-level). Tightening the large-loss prior stops decoy
/// precursor formulas from fitting peaks through implausible one-step losses.
/// See `examples/eval_tree_scoring.rs`.
fn loss_mass_penalty(loss_mass: f64, sigma_log: f64) -> f64 {
    if loss_mass <= LOSS_MASS_MODE_DA || sigma_log <= 0.0 {
        return 0.0;
    }
    let z = (loss_mass / LOSS_MASS_MODE_DA).ln() / sigma_log;
    -0.5 * z * z
}

/// Internal candidate vertex in the fragmentation graph
#[derive(Debug, Clone)]
struct GraphVertex {
    node_id: usize,
    peak_idx: Option<usize>, // None for root
    formula: Formula,
    mz: f64,
    neutral_mass: f64,
    intensity: f32,
    rel_intensity: f32,
    weight: f64,
}

/// Internal candidate directed edge in the fragmentation graph
#[derive(Debug, Clone)]
struct GraphEdge {
    from: usize,
    to: usize,
    loss_formula: Formula,
    loss_mass: f64,
    mass_error_da: f64,
    weight: f64,
    is_common_loss: bool,
}

/// Maximum A Posteriori (MAP) Fragmentation Tree Builder.
/// Finds the optimal fragmentation tree explaining an MS/MS spectrum
/// under a candidate precursor molecular formula.
pub struct FragmentationTreeBuilder {
    pub annotator: SpectrumAnnotator,
    scorer: FormulaScorer,
    rules: ChemicalRules,
    config: TreeSolverConfig,
}

impl Default for FragmentationTreeBuilder {
    fn default() -> Self {
        Self::new(TreeSolverConfig::default())
    }
}

impl FragmentationTreeBuilder {
    /// The isotope term used by this solver, shared with workflow prescoring.
    pub fn isotope_score(&self, precursor_formula: &Formula, ms1_peaks: &[Peak], precursor_mz: f64, adduct: &Adduct) -> f64 {
        if self.config.use_isotopes && ms1_peaks.len() >= 2 {
            if self.config.scoring == ScoringModel::Sirius {
                super::sirius_scoring::isotope_score(
                    precursor_formula,
                    ms1_peaks,
                    precursor_mz,
                    adduct,
                )
            } else {
                self.scorer.score_isotope_pattern_with_adduct(
                    precursor_formula,
                    ms1_peaks,
                    precursor_mz,
                    adduct,
                    self.config.tolerance,
                )
            }
        } else {
            0.0
        }
    }

    /// Candidate search with the same MS1 evidence and adduct as final ranking.
    pub fn build_trees_with_adduct_and_ms1(&self, formulas: &[Formula], precursor_mz: f64, peaks: &[Peak], ms1_peaks: &[Peak], adduct: &Adduct) -> Vec<(Formula, FragmentationTree)> {
        self.build_trees_for_candidates_inner(formulas, precursor_mz, peaks, ms1_peaks, adduct)
    }

    /// Effective solver settings for run manifests and benchmark parity checks.
    pub fn config(&self) -> &TreeSolverConfig {
        &self.config
    }

    pub fn new(config: TreeSolverConfig) -> Self {
        // A Da tolerance has to be expressed in ppm for the annotator and scorer,
        // which are ppm-based. Referencing it to 300 Da matches how `workflow`
        // converts the same setting; substituting a flat 20 ppm, as this used to,
        // left the solver running two different tolerances at once.
        let adduct = config.adduct.clone();
        Self {
            annotator: SpectrumAnnotator::new(config.mass_sigma_ppm)
                .with_adduct(adduct)
                .with_learned_losses(config.learned_loss_model),
            scorer: FormulaScorer::new(config.mass_sigma_ppm, 0.05),
            rules: ChemicalRules { loss_gate: config.loss_gate, ..ChemicalRules::loose() },
            config,
        }
    }

    /// Build optimal fragmentation trees for multiple candidate precursor formulas,
    /// decomposing fragment peaks once against the tight union bound of all candidates.
    pub fn build_trees_for_candidates(
        &self,
        precursor_formulas: &[Formula],
        precursor_mz: f64,
        peaks: &[Peak],
    ) -> Vec<(Formula, FragmentationTree)> {
        self.build_trees_for_candidates_with_adduct(
            precursor_formulas,
            precursor_mz,
            peaks,
            &self.config.adduct,
        )
    }

    /// As `build_trees_for_candidates`, plus an MS1 isotope-envelope peak list
    /// (the survivor precursor cluster: M, M+1, M+2, ...). When `use_isotopes` is
    /// set and the list holds at least two peaks in the isotope window, the
    /// isotope-pattern log-likelihood is added to each candidate's root score.
    pub fn build_trees_for_candidates_with_ms1(
        &self,
        precursor_formulas: &[Formula],
        precursor_mz: f64,
        peaks: &[Peak],
        ms1_peaks: &[Peak],
    ) -> Vec<(Formula, FragmentationTree)> {
        self.build_trees_for_candidates_inner(
            precursor_formulas,
            precursor_mz,
            peaks,
            ms1_peaks,
            &self.config.adduct,
        )
    }

    /// Build optimal fragmentation trees for multiple candidate precursor formulas with a specific adduct/charge state.
    pub fn build_trees_for_candidates_with_adduct(
        &self,
        precursor_formulas: &[Formula],
        precursor_mz: f64,
        peaks: &[Peak],
        adduct: &Adduct,
    ) -> Vec<(Formula, FragmentationTree)> {
        self.build_trees_for_candidates_inner(precursor_formulas, precursor_mz, peaks, &[], adduct)
    }

    fn build_trees_for_candidates_inner(
        &self,
        precursor_formulas: &[Formula],
        precursor_mz: f64,
        peaks: &[Peak],
        ms1_peaks: &[Peak],
        adduct: &Adduct,
    ) -> Vec<(Formula, FragmentationTree)> {
        if precursor_formulas.is_empty() {
            return Vec::new();
        }

        // Element-wise maximum union bound across all candidate precursors
        let mut max_formula = Formula::empty();
        for f in precursor_formulas {
            for el in &super::element::ALL_ELEMENTS {
                let cnt = f.get(*el);
                if cnt > max_formula.get(*el) {
                    max_formula.set(*el, cnt);
                }
            }
        }

        // Single-pass bounded peak decomposition against union bound using effective adduct
        let mut annotator = self.annotator.clone();
        annotator = annotator.with_adduct(adduct.clone());
        let mut precomputed_peaks =
            annotator.annotate_peaks(peaks, &max_formula, self.config.tolerance);

        // Take the systematic mass drift out of the spectrum before the trees are
        // built. Up to two passes: when the drift exceeds the tolerance the first
        // estimate is capped by the window and understates it, and a second pass
        // finishes the correction.
        let mut corrected: Vec<Peak> = Vec::new();
        for _ in 0..MAX_RECALIBRATION_PASSES {
            let drift_ppm = match estimate_drift_ppm(&precomputed_peaks) {
                Some(drift) => Some(drift),
                // A drift larger than the tolerance is precisely the case where too
                // few peaks decomposed to estimate it from. Widen the net, but only
                // for the estimate, and only when the ordinary pass came up short.
                None if anchor_count(&precomputed_peaks) < MIN_RECALIBRATION_ANCHORS
                    && peaks.len() >= MIN_RECALIBRATION_ANCHORS =>
                {
                    let source: &[Peak] = if corrected.is_empty() { peaks } else { &corrected };
                    let wide = widen(self.config.tolerance, RECALIBRATION_SEARCH_FACTOR);
                    estimate_drift_ppm(&annotator.annotate_peaks(source, &max_formula, wide))
                }
                None => None,
            };

            let Some(drift_ppm) = drift_ppm else { break };

            let source: &[Peak] = if corrected.is_empty() { peaks } else { &corrected };
            corrected = source
                .iter()
                .map(|p| Peak { mz: p.mz * (1.0 + drift_ppm * 1e-6), intensity: p.intensity })
                .collect();

            let mut reannotated =
                annotator.annotate_peaks(&corrected, &max_formula, self.config.tolerance);
            // Scores are against the corrected masses; the m/z reported back to the
            // caller stays the one the instrument actually measured.
            for (annotated, original) in reannotated.iter_mut().zip(peaks) {
                annotated.mz = original.mz;
            }
            precomputed_peaks = reannotated;
        }

        let filtered_for = |cand: &Formula| -> Vec<AnnotatedPeak> {
            precomputed_peaks
                .iter()
                .map(|p| AnnotatedPeak {
                    peak_index: p.peak_index,
                    mz: p.mz,
                    intensity: p.intensity,
                    relative_intensity: p.relative_intensity,
                    candidates: p
                        .candidates
                        .iter()
                        .filter(|c| c.formula.is_subformula_of(cand))
                        .copied()
                        .collect(),
                })
                .filter(|p| !p.candidates.is_empty())
                .collect()
        };
        let all_filtered: Vec<Vec<AnnotatedPeak>> =
            precursor_formulas.iter().map(filtered_for).collect();

        // SIRIUS `Beautification` (`FasterTreeComputationInstance`): the
        // tree-size bonus is tuned per spectrum, not fixed. Start at the profile
        // default and raise it in steps of 1.0 (fast Critical-Path passes) until
        // one of the top-5 candidate trees is "high quality" — explains ≥ 70 %
        // intensity with ≥ min(#peaks − 2, 15) nodes — or the bonus reaches
        // MAX_TREESIZE. Then build every candidate once at that bonus with the
        // configured solver. Under the Native objective the fixed
        // `config.tree_size_bonus` is used directly.
        let beautified_bonus = if self.config.scoring == ScoringModel::Sirius {
            let mut bonus = SIRIUS_TREE_SIZE_START;
            loop {
                let trial: Vec<FragmentationTree> = precursor_formulas
                    .iter()
                    .zip(&all_filtered)
                    .map(|(cand, fp)| {
                        self.build_tree_from_annotated_tuned(
                            cand,
                            precursor_mz,
                            peaks,
                            fp,
                            ms1_peaks,
                            adduct,
                            bonus,
                            SolverKind::CriticalPath,
                        )
                    })
                    .collect();
                let mut order: Vec<usize> = (0..trial.len()).collect();
                order.sort_by(|&a, &b| trial[b].total_score.total_cmp(&trial[a].total_score));
                let high_quality = order.iter().take(SIRIUS_BEAUTIFY_TOP_N).any(|&i| {
                    let t = &trial[i];
                    let need = t.total_peaks.saturating_sub(2).min(SIRIUS_MIN_EXPLAINED_PEAKS);
                    t.explained_intensity >= SIRIUS_MIN_EXPLAINED_INTENSITY as f32
                        && t.nodes.len() >= need
                });
                if high_quality || bonus >= SIRIUS_TREE_SIZE_MAX {
                    break bonus;
                }
                bonus += SIRIUS_TREE_SIZE_STEP;
            }
        } else {
            self.config.tree_size_bonus
        };

        // Pass 1. With a shortlist configured this is the cheap heuristic over every
        // candidate; without one it is already the final answer.
        let first_pass = match self.config.exact_top_n {
            Some(_) => SolverKind::CriticalPath,
            None => self.config.solver,
        };
        let mut results = Vec::with_capacity(precursor_formulas.len());
        for (cand, filtered_peaks) in precursor_formulas.iter().zip(&all_filtered) {
            let tree = self.build_tree_from_annotated_tuned(
                cand,
                precursor_mz,
                peaks,
                filtered_peaks,
                ms1_peaks,
                adduct,
                beautified_bonus,
                first_pass,
            );
            results.push((*cand, tree));
        }

        // Pass 2. Re-solve the leaders exactly. Critical-Path never over-scores a
        // tree — it is a restriction of the same objective — so the upgrade can
        // only raise a candidate's score, and a candidate the heuristic already
        // ranked below N could not have overtaken the leaders by less than the
        // gap it would have to close.
        if let Some(n) = self.config.exact_top_n {
            if first_pass != self.config.solver {
                let mut order: Vec<usize> = (0..results.len()).collect();
                order.sort_by(|&a, &b| {
                    results[b].1.total_score.total_cmp(&results[a].1.total_score)
                });
                for &i in order.iter().take(n) {
                    let cand = results[i].0;
                    results[i].1 = self.build_tree_from_annotated_tuned(
                        &cand,
                        precursor_mz,
                        peaks,
                        &all_filtered[i],
                        ms1_peaks,
                        adduct,
                        beautified_bonus,
                        self.config.solver,
                    );
                }
            }
        }

        // SIRIUS `HypothesenDrivenRecalibration` (`FasterTreeComputationInstance
        // .recalibrate`): each leading candidate's own tree is a set of hypotheses
        // about which peak is which formula, so it can be fitted for a linear m/z
        // calibration and the tree rebuilt on the corrected masses. The drift
        // correction above removes a constant offset every candidate shares; this
        // removes the slope, which is what actually separates near-isobars.
        //
        // Keep the better of the two trees, as SIRIUS does — a recalibration that
        // does not pay for itself is discarded rather than trusted.
        if self.config.recalibrate {
            let mut order: Vec<usize> = (0..results.len()).collect();
            order.sort_by(|&a, &b| results[b].1.total_score.total_cmp(&results[a].1.total_score));
            for &i in order.iter().take(RECAL_TOP_N) {
                let cand = results[i].0;
                let Some((intercept, slope)) = recalibration_from_tree(&results[i].1, adduct)
                else {
                    continue;
                };
                let recal: Vec<Peak> = peaks
                    .iter()
                    .map(|p| Peak { mz: intercept + slope * p.mz, intensity: p.intensity })
                    .collect();
                let mut reannotated =
                    annotator.annotate_peaks(&recal, &cand, self.config.tolerance);
                // Score against the corrected masses, report the measured ones —
                // the same convention the drift pass uses.
                for (annotated, original) in reannotated.iter_mut().zip(peaks) {
                    annotated.mz = original.mz;
                }
                reannotated.retain(|p| !p.candidates.is_empty());
                let recal_precursor = intercept + slope * precursor_mz;
                let tree = self.build_tree_from_annotated_tuned(
                    &cand,
                    recal_precursor,
                    peaks,
                    &reannotated,
                    ms1_peaks,
                    adduct,
                    beautified_bonus,
                    self.config.solver,
                );
                if tree.total_score > results[i].1.total_score {
                    results[i].1 = tree;
                }
            }
        }

        results
    }

    /// Construct maximum a posteriori (MAP) fragmentation tree for a single candidate precursor formula
    pub fn build_tree(
        &self,
        precursor_formula: &Formula,
        precursor_mz: f64,
        peaks: &[Peak],
    ) -> FragmentationTree {
        let annotated_peaks = self.annotator.annotate_peaks(peaks, precursor_formula, self.config.tolerance);
        self.build_tree_from_annotated(precursor_formula, precursor_mz, peaks, &annotated_peaks)
    }

    /// Construct fragmentation tree from pre-filtered annotated peaks
    pub fn build_tree_from_annotated(
        &self,
        precursor_formula: &Formula,
        precursor_mz: f64,
        peaks: &[Peak],
        annotated_peaks: &[AnnotatedPeak],
    ) -> FragmentationTree {
        self.build_tree_from_annotated_with_adduct(
            precursor_formula,
            precursor_mz,
            peaks,
            annotated_peaks,
            &self.config.adduct,
        )
    }

    /// Construct fragmentation tree from pre-filtered annotated peaks using a specific adduct/charge state
    pub fn build_tree_from_annotated_with_adduct(
        &self,
        precursor_formula: &Formula,
        precursor_mz: f64,
        peaks: &[Peak],
        annotated_peaks: &[AnnotatedPeak],
        adduct: &Adduct,
    ) -> FragmentationTree {
        self.build_tree_from_annotated_inner(
            precursor_formula,
            precursor_mz,
            peaks,
            annotated_peaks,
            &[],
            adduct,
        )
    }

    fn build_tree_from_annotated_inner(
        &self,
        precursor_formula: &Formula,
        precursor_mz: f64,
        peaks: &[Peak],
        annotated_peaks: &[AnnotatedPeak],
        ms1_peaks: &[Peak],
        adduct: &Adduct,
    ) -> FragmentationTree {
        self.build_tree_from_annotated_tuned(
            precursor_formula,
            precursor_mz,
            peaks,
            annotated_peaks,
            ms1_peaks,
            adduct,
            self.config.tree_size_bonus,
            self.config.solver,
        )
    }

    /// As [`Self::build_tree_from_annotated_inner`], with an explicit tree-size
    /// bonus and solver — the SIRIUS beautification loop sweeps the bonus and
    /// uses the fast solver for the trial passes.
    #[allow(clippy::too_many_arguments)]
    fn build_tree_from_annotated_tuned(
        &self,
        precursor_formula: &Formula,
        precursor_mz: f64,
        peaks: &[Peak],
        annotated_peaks: &[AnnotatedPeak],
        ms1_peaks: &[Peak],
        adduct: &Adduct,
        tree_size_bonus: f64,
        solver: SolverKind,
    ) -> FragmentationTree {
        let predicted_precursor_mz = adduct.neutral_mass_to_ion_mz(precursor_formula.exact_mass());

        // MS1 isotope-envelope evidence for this candidate formula, added once to
        // the root. Zero unless an isotope pattern was supplied and enabled.
        // Under the SIRIUS objective, use SIRIUS's own 4-scorer + best-prefix
        // isotope model (the M+2 step is what separates a Cl formula from its
        // near-isobaric P decoy — see FINDINGS round 16).
        let root_isotope = self.isotope_score(precursor_formula, ms1_peaks, precursor_mz, adduct);

        if peaks.is_empty() {
            let root_node = TreeNode {
                id: 0,
                peak_index: None,
                formula: *precursor_formula,
                mz: precursor_mz,
                intensity: 100.0,
                relative_intensity: 1.0,
                node_score: self.scorer.score_mass_accuracy(predicted_precursor_mz, precursor_mz)
                    + root_isotope,
            };
            return FragmentationTree {
                precursor_formula: *precursor_formula,
                precursor_mz,
                nodes: vec![root_node],
                edges: Vec::new(),
                total_score: 0.0,
                explained_intensity: 0.0,
                explained_peaks: 0,
                total_peaks: 0,
            };
        }

        // 1. Build Vertices: Root (Precursor) + Sub-formula annotations for each peak
        let max_intensity = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);
        let total_spectral_intensity: f32 = peaks.iter().map(|p| p.intensity).sum();

        let mut vertices: Vec<GraphVertex> = Vec::new();

        let sirius = self.config.scoring == ScoringModel::Sirius;
        // SIRIUS's mass-deviation term: ln(erfc(|Δm| / (sd·√2))). sd is the
        // instrument's absolute std deviation at the peak m/z.
        let sirius_sd = |mz: f64| (mz * self.config.mass_sigma_ppm * 1e-6).max(1e-3);

        // Root vertex 0
        let root_prior = self.scorer.score_prior(precursor_formula);
        let root_neutral = adduct.ion_mz_to_neutral_mass(precursor_mz);
        let root_mass_score = if sirius {
            super::sirius_scoring::mass_deviation_score(
                predicted_precursor_mz - precursor_mz,
                sirius_sd(precursor_mz),
            )
        } else {
            self.scorer.score_mass_accuracy(predicted_precursor_mz, precursor_mz)
        };
        let root_weight = if sirius {
            super::sirius_scoring::vertex_score(
                precursor_formula,
                precursor_formula.exact_mass(),
                root_mass_score,
                true,
            ) + root_isotope
        } else {
            root_mass_score + root_prior + root_isotope
        };
        vertices.push(GraphVertex {
            node_id: 0,
            peak_idx: None,
            formula: *precursor_formula,
            mz: precursor_mz,
            neutral_mass: root_neutral,
            intensity: max_intensity,
            rel_intensity: 1.0,
            // The root's node score, like any other node's: how well the candidate
            // explains the precursor mass, plus its chemical prior, plus the MS1
            // isotope-envelope log-likelihood when one was supplied. Counted once
            // here and nowhere else -- `workflow` used to add `score_prior` a
            // second time on top of the tree score, giving the prior three times
            // the weight of the fragment evidence the tree is supposed to be
            // judged on.
            weight: root_weight,
        });

        // Adaptive noise floor: scales with peak density to preserve genuine peaks in sparse spectra
        let noise_floor = self.config.noise_floor_scale
            * if peaks.len() <= 5 {
                1.5
            } else if peaks.len() <= 15 {
                2.0
            } else {
                2.5
            };

        // `ClippedPeakIsNoiseScorer` scores each peak against the max intensity of
        // the merged peak list, so the base peak is the reference regardless of
        // which peaks decomposed.
        let max_rel = annotated_peaks
            .iter()
            .map(|p| p.relative_intensity as f64)
            .fold(0.0f64, f64::max);

        // Sort fragment vertices descending by exact formula mass so that any parent u must precede v (u < v)
        let mut fragment_vertices: Vec<GraphVertex> = Vec::new();
        for ap in annotated_peaks {
            // One peak, one noise score — shared by every formula that explains it.
            let peak_noise = match self.config.peak_noise_median {
                Some(median) => super::sirius_scoring::peak_is_noise_score(
                    ap.relative_intensity as f64,
                    max_rel,
                    median,
                ),
                None => 0.0,
            };
            for cand in &ap.candidates {
                let vertex_weight = if sirius {
                    // No intensity term (SIRIUS's objective has none); the fixed
                    // tree-size bonus stands in for the adaptive noise floor.
                    let md = super::sirius_scoring::mass_deviation_score(
                        cand.mass_error_da,
                        sirius_sd(ap.mz),
                    );
                    // CommonRootLossScorer: bonus for the cumulative loss from
                    // the precursor down to this fragment.
                    let root_loss = precursor_formula
                        .checked_sub(&cand.formula)
                        .map_or(0.0, |l| super::sirius_scoring::root_loss_score(&l));
                    super::sirius_scoring::vertex_score(
                        &cand.formula,
                        cand.formula.exact_mass(),
                        md,
                        false,
                    ) + root_loss
                        + tree_size_bonus
                        + peak_noise
                } else {
                    // Log-intensity scaling (log-odds evidence) minus the
                    // adaptive noise floor.
                    let int_score = ((ap.relative_intensity as f64) * 100.0 + 1.0).ln() * 1.2;
                    int_score + cand.mass_score * 0.4 + cand.prior_score - noise_floor
                };

                fragment_vertices.push(GraphVertex {
                    node_id: 0,
                    peak_idx: Some(ap.peak_index),
                    formula: cand.formula,
                    mz: ap.mz,
                    neutral_mass: cand.formula.exact_mass(),
                    intensity: ap.intensity,
                    rel_intensity: ap.relative_intensity,
                    weight: vertex_weight,
                });
            }
        }
        // `neutral_mass` is this vertex's `formula.exact_mass()`, computed once when
        // the vertex was built. Recomputing it inside the comparator cost 2 calls
        // per comparison, and inside the edge loop below it cost O(V^2) — together
        // 23.7 % of self time on the de-novo profile.
        fragment_vertices.sort_by(|a, b| b.neutral_mass.total_cmp(&a.neutral_mass));

        for (i, v) in fragment_vertices.into_iter().enumerate() {
            let mut v = v;
            v.node_id = i + 1;
            vertices.push(v);
        }

        // 2. Build Candidate Directed Edges u -> v (u < v guaranteed by descending mass order)
        let mut edges: Vec<GraphEdge> = Vec::new();
        let num_vertices = vertices.len();

        for v in 1..num_vertices {
            for u in 0..v {
                if vertices[u].neutral_mass <= vertices[v].neutral_mass {
                    continue;
                }

                // If both are fragment candidates for the same peak, skip
                if let (Some(p_u), Some(p_v)) = (vertices[u].peak_idx, vertices[v].peak_idx) {
                    if p_u == p_v {
                        continue;
                    }
                }

                let form_u = &vertices[u].formula;
                let form_v = &vertices[v].formula;

                if let Some(loss_f) = form_u.checked_sub(form_v) {
                    if loss_f.total_atoms() == 0 {
                        // If u == 0 (Root) and root formula equals fragment formula,
                        // this represents dissociation of the adduct modifier (e.g. [M+HCOO]- -> [M-H]- + HCOOH or [M+NH4]+ -> [M+H]+ + NH3)
                        if u == 0 && vertices[u].neutral_mass > vertices[v].neutral_mass {
                            let obs_delta = vertices[u].neutral_mass - vertices[v].neutral_mass;
                            if obs_delta > 0.5 {
                                let edge_weight = 2.5;
                                edges.push(GraphEdge {
                                    from: u,
                                    to: v,
                                    loss_formula: loss_f,
                                    loss_mass: obs_delta,
                                    mass_error_da: 0.0,
                                    weight: edge_weight,
                                    is_common_loss: true,
                                });
                            }
                        }
                        continue;
                    }

                    // Strict chemical loss validity (Senior's rules + forbidden atoms)
                    if !self.rules.is_valid_neutral_loss(&loss_f) {
                        continue;
                    }

                    let loss_mass = loss_f.exact_mass();

                    // Mass deviation is evidence about a *peak*, and it is already
                    // scored once at the node. An edge's loss is exactly
                    // parent - child, so the loss formula's mass and the difference
                    // of the two theoretical masses agree by construction: scoring
                    // that agreement, as this used to, measured nothing (the term sat
                    // at its maximum on every fragment-to-fragment edge) and the
                    // tolerance check guarding it could never fail. What an edge
                    // carries is chemical plausibility.
                    let is_common = self.annotator.is_common_loss(&loss_f);
                    let edge_weight = if sirius {
                        super::sirius_scoring::edge_score(&loss_f, loss_mass)
                    } else {
                        let loss_score = self.annotator.score_loss(&loss_f);
                        // The learned model scores every loss by its observed
                        // per-mode frequency and already gives unlisted large
                        // losses a penalty, so the blunt size prior is only for
                        // the hand list.
                        let size_penalty = if self.config.learned_loss_model {
                            0.0
                        } else {
                            loss_mass_penalty(loss_mass, self.config.loss_mass_sigma_log)
                        };
                        loss_score + size_penalty
                    };

                    edges.push(GraphEdge {
                        from: u,
                        to: v,
                        loss_formula: loss_f,
                        loss_mass,
                        mass_error_da: 0.0,
                        weight: edge_weight,
                        is_common_loss: is_common,
                    });
                }
            }
        }

        // 3. Maximum Colorful Subtree search (Critical Path heuristic).
        //
        // Picking the best incoming edge per vertex and *then* thinning to one
        // vertex per peak does not give a tree: the winner for one peak is often
        // parented by a candidate that lost its own peak's contest, and there is
        // no honest way to reattach it afterwards. Instead the tree is grown so
        // that it is connected and colorful at every step, which is what makes
        // `loss_formula == parent.formula - child.formula` an invariant rather
        // than something to be patched up at the end.
        //
        // Each peak is a color and may be used at most once, so this is the
        // Maximum Colorful Subtree problem (NP-hard). We use the Critical Path
        // heuristic of Dührkop et al., WABI 2018: repeatedly find the highest
        // scoring downward path that starts at a vertex already in the tree and
        // visits only unused colors, and append that whole path.
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
        for (i, edge) in edges.iter().enumerate() {
            out_edges[edge.from].push(i);
        }

        // The surviving precursor ion is usually present as a peak, and it is the base
        // peak more often than not. The root explains it, so it has to be attributed to
        // the root: counting it in the spectrum's total but never in the explained total
        // understated `explained_intensity` by the largest single peak. Measured on 949
        // annotated spectra, that alone accounted for 35.5 of the 40.1 percentage points
        // that looked unexplained.
        let precursor_peak = peaks
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                self.config
                    .tolerance
                    .is_within(p.mz, precursor_mz)
            })
            .max_by(|(_, a), (_, b)| a.intensity.partial_cmp(&b.intensity).unwrap_or(std::cmp::Ordering::Equal));

        // Colors (peak indices) the root already claims: the precursor peak, so
        // no fragment can be credited with that intensity a second time.
        let root_colors: HashSet<usize> = precursor_peak.map(|(i, _)| i).into_iter().collect();

        let (tree_vertices, parent_edge) = match solver {
            SolverKind::CriticalPath => {
                select_critical_path(&vertices, &edges, &out_edges, num_vertices, &root_colors)
            }
            SolverKind::DagDp => {
                select_dag_dp(&vertices, &edges, num_vertices, &root_colors)
            }
            SolverKind::Exact => {
                select_exact(&vertices, &edges, &out_edges, num_vertices, &root_colors)
            }
            SolverKind::Ilp => {
                select_ilp(&vertices, &edges, num_vertices, &root_colors)
                    .unwrap_or_else(|| {
                        select_exact(&vertices, &edges, &out_edges, num_vertices, &root_colors)
                    })
            }
        };

        // 4. Construct Final Fragmentation Tree
        let mut final_nodes = Vec::with_capacity(tree_vertices.len());
        let mut final_edges = Vec::with_capacity(tree_vertices.len().saturating_sub(1));
        let mut graph_to_tree_id: HashMap<usize, usize> = HashMap::with_capacity(tree_vertices.len());

        for (tree_id, &v_id) in tree_vertices.iter().enumerate() {
            graph_to_tree_id.insert(v_id, tree_id);
        }

        // Root node
        final_nodes.push(TreeNode {
            id: 0,
            peak_index: precursor_peak.map(|(i, _)| i),
            formula: vertices[0].formula,
            mz: vertices[0].mz,
            intensity: precursor_peak.map_or(vertices[0].intensity, |(_, p)| p.intensity),
            relative_intensity: 1.0,
            node_score: vertices[0].weight,
        });

        let mut total_explained_intensity: f32 = precursor_peak.map_or(0.0, |(_, p)| p.intensity);
        let mut tree_score = vertices[0].weight;

        for (tree_id, &v_id) in tree_vertices.iter().enumerate().skip(1) {
            let v = &vertices[v_id];

            final_nodes.push(TreeNode {
                id: tree_id,
                peak_index: v.peak_idx,
                formula: v.formula,
                mz: v.mz,
                intensity: v.intensity,
                relative_intensity: v.rel_intensity,
                node_score: v.weight,
            });

            total_explained_intensity += v.intensity;
            tree_score += v.weight;

            let edge_idx = parent_edge[v_id].expect("every vertex but the root is appended with its parent edge");
            let edge = &edges[edge_idx];

            final_edges.push(TreeEdge {
                parent_id: graph_to_tree_id[&edge.from],
                child_id: tree_id,
                loss_formula: edge.loss_formula,
                loss_exact_mass: edge.loss_mass,
                mass_error_da: edge.mass_error_da,
                edge_score: edge.weight,
                is_common_loss: edge.is_common_loss,
            });

            tree_score += edge.weight;
        }

        let explained_fraction = if total_spectral_intensity > 0.0 {
            total_explained_intensity / total_spectral_intensity
        } else {
            0.0
        };

        FragmentationTree {
            precursor_formula: *precursor_formula,
            precursor_mz,
            nodes: final_nodes,
            edges: final_edges,
            total_score: tree_score,
            explained_intensity: explained_fraction,
            explained_peaks: tree_vertices.len() - 1 + usize::from(precursor_peak.is_some()),
            total_peaks: peaks.len(),
        }
    }

    fn annotator_is_common_loss(&self, loss: &Formula) -> bool {
        self.annotator.is_common_loss(loss)
    }
}

// ── Tree-selection algorithms ───────────────────────────────────────────────
//
// Each takes the scored fragment graph — `vertices` (index 0 = root, the rest
// fragments in descending formula-mass order, so index order is a topological
// order and every edge runs low index -> high index), `edges`, and the colors
// (peak indices) the root already claims — and returns `(tree_vertices,
// parent_edge)`: `tree_vertices[0] == 0` (root), and `parent_edge[v]` is the
// index of v's incoming edge for every non-root v in the tree.

/// Critical-Path heuristic (Dührkop et al., WABI 2018): repeatedly append the
/// highest-scoring downward color-disjoint path starting from a vertex already
/// in the tree.
fn select_critical_path(
    vertices: &[GraphVertex],
    edges: &[GraphEdge],
    out_edges: &[Vec<usize>],
    num_vertices: usize,
    root_colors: &HashSet<usize>,
) -> (Vec<usize>, Vec<Option<usize>>) {
    let mut in_tree = vec![false; num_vertices];
    in_tree[0] = true;
    let mut used_colors: HashSet<usize> = root_colors.clone();
    let mut parent_edge: Vec<Option<usize>> = vec![None; num_vertices];
    let mut tree_vertices: Vec<usize> = vec![0];

    loop {
        let mut path_score = vec![0.0f64; num_vertices];
        let mut path_step: Vec<Option<usize>> = vec![None; num_vertices];

        for u in (0..num_vertices).rev() {
            for &edge_idx in &out_edges[u] {
                let edge = &edges[edge_idx];
                let v = edge.to;
                let color = match vertices[v].peak_idx {
                    Some(c) => c,
                    None => continue,
                };
                if in_tree[v] || used_colors.contains(&color) {
                    continue;
                }
                let gain = edge.weight + vertices[v].weight + path_score[v];
                if gain > path_score[u] {
                    path_score[u] = gain;
                    path_step[u] = Some(edge_idx);
                }
            }
        }

        let start = tree_vertices
            .iter()
            .copied()
            .filter(|&u| path_step[u].is_some())
            .max_by(|&a, &b| path_score[a].total_cmp(&path_score[b]));

        let start = match start {
            Some(u) if path_score[u] > 0.0 => u,
            _ => break,
        };

        let mut u = start;
        while let Some(edge_idx) = path_step[u] {
            let v = edges[edge_idx].to;
            let color = match vertices[v].peak_idx {
                Some(c) => c,
                None => break,
            };
            if in_tree[v] || !used_colors.insert(color) {
                break;
            }
            in_tree[v] = true;
            parent_edge[v] = Some(edge_idx);
            tree_vertices.push(v);
            u = v;
        }
    }

    (tree_vertices, parent_edge)
}

/// Pick the single best-scoring candidate per peak, then take the exact
/// max-weight rooted subtree of the resulting one-vertex-per-color DAG by a
/// reverse-topological DP. Exact when the DAG has no reconvergence (a fragment
/// reachable from two different retained parents); a relaxation otherwise, in
/// which case the materialised tree still scores honestly (each vertex gets one
/// parent) but may be slightly sub-optimal.
fn select_dag_dp(
    vertices: &[GraphVertex],
    edges: &[GraphEdge],
    num_vertices: usize,
    root_colors: &HashSet<usize>,
) -> (Vec<usize>, Vec<Option<usize>>) {
    // Best candidate vertex per color.
    let mut best_for_color: HashMap<usize, usize> = HashMap::new();
    for (i, v) in vertices.iter().enumerate().skip(1) {
        let Some(c) = v.peak_idx else { continue };
        if root_colors.contains(&c) {
            continue;
        }
        best_for_color
            .entry(c)
            .and_modify(|cur| {
                if v.weight > vertices[*cur].weight {
                    *cur = i;
                }
            })
            .or_insert(i);
    }
    let keep: HashSet<usize> = best_for_color
        .values()
        .copied()
        .chain(std::iter::once(0usize))
        .collect();

    // Retained edges, grouped by parent, deduped so a parent has at most one
    // edge to any given child (the best-weight one).
    let mut child_edges: Vec<HashMap<usize, usize>> = vec![HashMap::new(); num_vertices];
    for (ei, e) in edges.iter().enumerate() {
        if !keep.contains(&e.from) || !keep.contains(&e.to) {
            continue;
        }
        child_edges[e.from]
            .entry(e.to)
            .and_modify(|cur| {
                if e.weight > edges[*cur].weight {
                    *cur = ei;
                }
            })
            .or_insert(ei);
    }

    // best[v] = weight of the optimal subtree rooted at v (v included).
    // Reverse-topological order = descending vertex index.
    let mut best = vec![f64::NEG_INFINITY; num_vertices];
    for v in (0..num_vertices).rev() {
        if !keep.contains(&v) {
            continue;
        }
        let mut s = vertices[v].weight;
        for (&_child, &ei) in &child_edges[v] {
            let c = edges[ei].to;
            let contrib = edges[ei].weight + best[c];
            if contrib > 0.0 {
                s += contrib;
            }
        }
        best[v] = s;
    }

    // Materialise: BFS from the root, attach a child when its edge contribution
    // is positive and it is still unclaimed.
    let mut parent_edge: Vec<Option<usize>> = vec![None; num_vertices];
    let mut tree_vertices: Vec<usize> = vec![0];
    let mut claimed = vec![false; num_vertices];
    claimed[0] = true;
    let mut queue = std::collections::VecDeque::from([0usize]);
    while let Some(v) = queue.pop_front() {
        let mut kids: Vec<usize> = child_edges[v].values().copied().collect();
        // Deterministic: heaviest (lowest index) child first.
        kids.sort_by_key(|&ei| edges[ei].to);
        for ei in kids {
            let c = edges[ei].to;
            if claimed[c] {
                continue;
            }
            if edges[ei].weight + best[c] > 0.0 {
                claimed[c] = true;
                parent_edge[c] = Some(ei);
                tree_vertices.push(c);
                queue.push_back(c);
            }
        }
    }

    (tree_vertices, parent_edge)
}

/// Exact Maximum Colorful Subtree as an integer program.
///
/// One binary per edge. The fragment graph is a DAG — an edge always goes from a
/// heavier fragment to a lighter one — so the compact formulation of Rauf et al.
/// (2013) is exact without subtour elimination:
///
/// * **colourful**: for each colour, at most one selected edge may enter a vertex
///   of that colour. This also gives each vertex at most one parent, so a separate
///   in-degree constraint would be redundant.
/// * **connectivity**: an edge may leave `u` only if some edge enters `u`. The
///   root is always reachable and so is exempt.
///
/// A vertex is paid for by its incoming edge, which is sound because at most one
/// is ever selected. The root's own weight is a constant and stays out.
///
/// Returns `None` if the model is empty or the solver does not reach an optimum,
/// so the caller can fall back rather than silently return a worse tree.
fn select_ilp(
    vertices: &[GraphVertex],
    edges: &[GraphEdge],
    num_vertices: usize,
    root_colors: &HashSet<usize>,
) -> Option<(Vec<usize>, Vec<Option<usize>>)> {
    use good_lp::{Expression, Solution, SolverModel, constraint, variable, variables};

    // Edges that may be selected at all: the target must carry a colour the root
    // has not already consumed.
    let usable: Vec<usize> = (0..edges.len())
        .filter(|&e| {
            vertices[edges[e].to]
                .peak_idx
                .is_some_and(|c| !root_colors.contains(&c))
        })
        .collect();
    if usable.is_empty() {
        return Some((vec![0], vec![None; num_vertices]));
    }

    let mut vars = variables!();
    let x: Vec<_> = usable.iter().map(|_| vars.add(variable().binary())).collect();

    let objective: Expression = usable
        .iter()
        .zip(&x)
        .map(|(&e, xi)| (edges[e].weight + vertices[edges[e].to].weight) * *xi)
        .sum();

    // Connectivity is written per edge against the full incoming sum. Naming a
    // `reached[v]` variable instead trades those nonzeros for one extra column and
    // row per vertex; measured, that is 32 % slower here (25.1 s vs 19.0 s on the
    // CASMI de-novo sweep), because these graphs have small in-degree and microlp
    // pays more for the extra rows than the nonzeros cost.
    let mut incoming: Vec<Expression> = vec![Expression::from(0.0); num_vertices];
    // Colours are dense peak indices, so a Vec beats hashing every edge.
    let n_colors = vertices.iter().filter_map(|v| v.peak_idx).max().map_or(0, |m| m + 1);
    let mut by_color: Vec<Option<Expression>> = vec![None; n_colors];
    let mut touched: Vec<usize> = Vec::new();
    for (&e, xi) in usable.iter().zip(&x) {
        incoming[edges[e].to] += *xi;
        if let Some(c) = vertices[edges[e].to].peak_idx {
            match &mut by_color[c] {
                Some(sum) => *sum += *xi,
                slot @ None => {
                    *slot = Some(Expression::from(*xi));
                    touched.push(c);
                }
            }
        }
    }

    let mut model = vars.maximise(objective).using(good_lp::microlp);
    for c in touched {
        if let Some(sum) = by_color[c].take() {
            model = model.with(constraint!(sum <= 1));
        }
    }
    for (&e, xi) in usable.iter().zip(&x) {
        let u = edges[e].from;
        if u != 0 {
            model = model.with(constraint!(*xi - incoming[u].clone() <= 0));
        }
    }

    let solution = model.solve().ok()?;

    let mut parent_edge: Vec<Option<usize>> = vec![None; num_vertices];
    for (&e, xi) in usable.iter().zip(&x) {
        if solution.value(*xi) > 0.5 {
            parent_edge[edges[e].to] = Some(e);
        }
    }

    // Walk down from the root so the returned vertices are exactly the connected
    // component the solution describes, in parent-before-child order.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for v in 0..num_vertices {
        if let Some(e) = parent_edge[v] {
            children[edges[e].from].push(v);
        }
    }
    let mut tree_vertices = vec![0usize];
    let mut stack = vec![0usize];
    let mut seen = vec![false; num_vertices];
    seen[0] = true;
    while let Some(u) = stack.pop() {
        for &v in &children[u] {
            if !seen[v] {
                seen[v] = true;
                tree_vertices.push(v);
                stack.push(v);
            }
        }
    }
    // Anything the solver selected but left unreachable is not part of the tree.
    for v in 0..num_vertices {
        if !seen[v] {
            parent_edge[v] = None;
        }
    }
    tree_vertices.sort_unstable();
    Some((tree_vertices, parent_edge))
}

/// Exact Maximum Colorful Subtree by the Böcker–Rasche colour-subset DP. Returns
/// the true optimum SIRIUS's ILP also targets. Falls back to Critical-Path above
/// [`EXACT_COLOR_CAP`] colours (the 3^k term).
fn select_exact(
    vertices: &[GraphVertex],
    edges: &[GraphEdge],
    out_edges: &[Vec<usize>],
    num_vertices: usize,
    root_colors: &HashSet<usize>,
) -> (Vec<usize>, Vec<Option<usize>>) {
    // Distinct colours in play (peaks with a surviving vertex, minus the root's).
    let mut color_list: Vec<usize> = Vec::new();
    let mut color_bit: HashMap<usize, usize> = HashMap::new();
    for v in vertices.iter().skip(1) {
        let Some(c) = v.peak_idx else { continue };
        if root_colors.contains(&c) {
            continue;
        }
        if !color_bit.contains_key(&c) {
            color_bit.insert(c, color_list.len());
            color_list.push(c);
        }
    }
    let k = color_list.len();
    if k > EXACT_COLOR_CAP {
        return select_critical_path(vertices, edges, out_edges, num_vertices, root_colors);
    }
    if k == 0 {
        return (vec![0], vec![None; num_vertices]);
    }

    // vertices participating in the DP: root + fragments whose colour is in play.
    let vbit = |v: usize| -> Option<usize> {
        vertices[v].peak_idx.and_then(|c| color_bit.get(&c).copied())
    };
    let full: u32 = if k == 32 { u32::MAX } else { (1u32 << k) - 1 };

    // g[v][S] = best score of a colourful subtree rooted at v covering exactly
    // colour-set S (v's own colour bit ∈ S), with a backpointer. Multiple
    // children are handled by extending the set of subtrees rooted at v one
    // child at a time to a fixed point — no explicit merge step is needed.
    #[derive(Clone, Copy)]
    enum Back {
        Leaf,
        Child(usize, u32), // (edge_idx to the child, child's colour subset)
    }
    let mut g: Vec<HashMap<u32, (f64, Back)>> = vec![HashMap::new(); num_vertices];

    // Children before parents: vertices are descending mass, so descending index
    // is light -> heavy.
    for v in (0..num_vertices).rev() {
        let vb = if v == 0 { None } else { vbit(v) };
        if v != 0 && vb.is_none() {
            continue;
        }
        let base_set: u32 = vb.map_or(0, |b| 1u32 << b);
        let mut local: HashMap<u32, (f64, Back)> = HashMap::new();
        local.insert(base_set, (vertices[v].weight, Back::Leaf));

        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<(u32, f64)> = local.iter().map(|(&s, &(sc, _))| (s, sc)).collect();
            for &edge_idx in &out_edges[v] {
                let c = edges[edge_idx].to;
                if g[c].is_empty() {
                    continue;
                }
                let child_sets: Vec<(u32, f64)> =
                    g[c].iter().map(|(&s, &(sc, _))| (s, sc)).collect();
                for &(s_here, sc_here) in &snapshot {
                    for &(s_child, sc_child) in &child_sets {
                        if s_here & s_child != 0 {
                            continue;
                        }
                        let s_new = s_here | s_child;
                        let sc_new = sc_here + edges[edge_idx].weight + sc_child;
                        if local.get(&s_new).map_or(true, |&(cur, _)| sc_new > cur + 1e-9) {
                            local.insert(s_new, (sc_new, Back::Child(edge_idx, s_child)));
                            changed = true;
                        }
                    }
                }
            }
        }
        g[v] = local;
    }

    let best_set = g[0]
        .iter()
        .max_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        .map(|(&s, _)| s)
        .unwrap_or(0);

    // Reconstruct: at (v, S) the backpointer says "S was built by attaching
    // child (via edge ei) with subset cs to (v, S & !cs)".
    let mut parent_edge: Vec<Option<usize>> = vec![None; num_vertices];
    let mut tree_vertices: Vec<usize> = vec![0];
    let mut stack: Vec<(usize, u32)> = vec![(0, best_set)];
    while let Some((v, mut s)) = stack.pop() {
        while let Some(&(_, Back::Child(ei, cs))) = g[v].get(&s) {
            let c = edges[ei].to;
            parent_edge[c] = Some(ei);
            tree_vertices.push(c);
            stack.push((c, cs));
            s &= !cs;
        }
    }

    (tree_vertices, parent_edge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::Peak;
    use std::str::FromStr;

    /// The Theil-Sen fit must recover a known distortion exactly, and must not be
    /// moved by a single mis-assigned reference. Both matter: a recalibration is
    /// fitted against hypotheses, and one wrong fragment formula in the tree is the
    /// ordinary case, not the exceptional one.
    #[test]
    fn median_linear_recalibration_recovers_a_known_distortion_despite_an_outlier() {
        // f(mz) = 0.002 + (1 - 3e-6) * mz: a 0.002 Da offset and a -3 ppm slope.
        let (want_intercept, want_slope) = (0.002, 1.0 - 3e-6);
        let theoretical = [110.0, 145.5, 180.25, 223.0, 260.5, 301.75, 340.0, 388.5, 421.0];
        let mut points: Vec<(f64, f64)> = theoretical
            .iter()
            .map(|&t| ((t - want_intercept) / want_slope, t))
            .collect();
        let (intercept, slope) = median_linear_recalibration(&points).unwrap();
        assert!((slope - want_slope).abs() < 1e-9, "slope {slope} != {want_slope}");
        assert!((intercept - want_intercept).abs() < 1e-6, "intercept {intercept}");

        // One reference assigned the wrong formula, 40 mDa out.
        points[4].1 += 0.04;
        let (intercept, slope) = median_linear_recalibration(&points).unwrap();
        assert!(
            (slope - want_slope).abs() < 1e-8 && (intercept - want_intercept).abs() < 1e-4,
            "a single outlier moved the fit to ({intercept}, {slope})"
        );
    }

    /// A dense cluster of low-mass fragments must not carry the fit on its own —
    /// the slope is only as good as the m/z range the references span.
    #[test]
    fn fair_distributed_subset_caps_each_hundred_dalton_bin() {
        // 9 references under 100 Da, 2 between 100 and 200, 1 between 200 and 300.
        let mut refs: Vec<(f64, f64, f32)> =
            (0..9).map(|i| (10.0 + i as f64 * 5.0, 10.0 + i as f64 * 5.0, i as f32)).collect();
        refs.push((120.0, 120.0, 1.0));
        refs.push((150.0, 150.0, 2.0));
        refs.push((250.0, 250.0, 1.0));

        let subset = fair_distributed_subset(&refs);
        let low = subset.iter().filter(|p| p.0 < 100.0).count();
        // limit = max(5, median bin size) = max(5, 2) = 5.
        assert_eq!(low, 5, "the crowded bin should be capped at 5, got {low}");
        assert_eq!(subset.len(), 5 + 2 + 1);
        // and it keeps the most intense of the crowd, not the first five.
        let kept: Vec<f64> = subset.iter().filter(|p| p.0 < 100.0).map(|p| p.0).collect();
        assert!(kept.contains(&50.0), "kept {kept:?}, expected the high-intensity end");
    }

    #[test]
    fn test_surviving_precursor_peak_counts_as_explained() {
        // Caffeine, protonated, with the precursor surviving as the base peak — the
        // usual case, and the one that made explained_intensity understate every tree.
        let formula = Formula::from_str("C8H10N4O2").unwrap();
        let precursor_mz = 195.0877;
        let peaks = vec![
            Peak { mz: precursor_mz, intensity: 100.0 },
            Peak { mz: 138.0662, intensity: 30.0 },
            Peak { mz: 110.0713, intensity: 10.0 },
        ];

        let builder = FragmentationTreeBuilder::new(TreeSolverConfig::default());
        let tree = builder.build_tree(&formula, precursor_mz, &peaks);

        assert_eq!(
            tree.root().peak_index,
            Some(0),
            "the root should be attributed to the surviving precursor peak"
        );
        assert!(
            tree.explained_intensity > 0.7,
            "the precursor alone carries 71% of the intensity, so the explained fraction \
             must exceed that; got {}",
            tree.explained_intensity
        );
    }

    #[test]
    fn test_absent_precursor_peak_leaves_the_root_unattributed() {
        // A spectrum whose precursor did not survive: nothing to attribute, and the
        // explained fraction must not be inflated by pretending otherwise.
        let formula = Formula::from_str("C8H10N4O2").unwrap();
        let peaks = vec![
            Peak { mz: 138.0662, intensity: 30.0 },
            Peak { mz: 110.0713, intensity: 10.0 },
        ];

        let builder = FragmentationTreeBuilder::new(TreeSolverConfig::default());
        let tree = builder.build_tree(&formula, 195.0877, &peaks);

        assert_eq!(tree.root().peak_index, None);
        assert!(tree.explained_intensity <= 1.0);
    }

    /// Two independent exact methods must agree. The ILP and the Böcker-Rasche DP
    /// solve the same Maximum Colorful Subtree, so on any instance the DP can take
    /// (<= `EXACT_COLOR_CAP` colours) their objective values must be identical —
    /// not merely both >= the heuristic.
    #[test]
    fn ilp_and_subset_dp_reach_the_same_optimum() {
        let cfg = |s: SolverKind| TreeSolverConfig {
            solver: s,
            tolerance: MassTolerance::Ppm(20.0),
            ..Default::default()
        };
        let cases: [(&str, f64, &[(f64, f32)]); 3] = [
            (
                "C21H25NO4",
                356.18563,
                &[(292.133205, 0.5), (324.159425, 15.0), (338.175075, 100.0)],
            ),
            (
                "C8H10N4O2",
                195.0877,
                &[(138.0662, 30.0), (110.0713, 10.0), (163.0608, 55.0)],
            ),
            (
                "C9H22O4P2S4",
                384.994891,
                &[(96.9508, 8.0), (114.9613, 38.0), (142.9385, 100.0), (170.9698, 27.0)],
            ),
        ];
        for (f, mz, peaks) in cases {
            let formula = Formula::from_str(f).unwrap();
            let pk: Vec<Peak> = peaks.iter().map(|&(m, i)| Peak { mz: m, intensity: i }).collect();
            let dp = FragmentationTreeBuilder::new(cfg(SolverKind::Exact)).build_tree(&formula, mz, &pk);
            let ilp = FragmentationTreeBuilder::new(cfg(SolverKind::Ilp)).build_tree(&formula, mz, &pk);
            assert!(
                (dp.total_score - ilp.total_score).abs() < 1e-6,
                "{f}: DP {} != ILP {}",
                dp.total_score,
                ilp.total_score
            );
            let mut seen = std::collections::HashSet::new();
            for n in &ilp.nodes {
                if let Some(pi) = n.peak_index {
                    assert!(seen.insert(pi), "{f}: ILP tree reuses peak {pi}");
                }
            }
        }
    }

    #[test]
    fn exact_solver_never_scores_below_critical_path() {
        // The Böcker–Rasche subset DP returns the true optimum, so its tree score
        // must be >= the Critical-Path heuristic's on the same graph, and its
        // tree must still be valid (rooted, colourful, correct edge labels).
        let cfg = |s: SolverKind| TreeSolverConfig {
            solver: s,
            tolerance: MassTolerance::Ppm(20.0),
            ..Default::default()
        };
        let cases: [(&str, f64, &[(f64, f32)]); 3] = [
            (
                "C21H25NO4",
                356.18563,
                &[(292.133205, 0.5), (324.159425, 15.0), (338.175075, 100.0)],
            ),
            (
                "C8H10N4O2",
                195.0877,
                &[(138.0662, 30.0), (110.0713, 10.0), (163.0608, 55.0)],
            ),
            (
                "C9H22O4P2S4",
                384.994891,
                &[(96.9508, 8.0), (114.9613, 38.0), (142.9385, 100.0), (170.9698, 27.0)],
            ),
        ];
        for (f, mz, peaks) in cases {
            let formula = Formula::from_str(f).unwrap();
            let pk: Vec<Peak> = peaks.iter().map(|&(m, i)| Peak { mz: m, intensity: i }).collect();
            let t_cp = FragmentationTreeBuilder::new(cfg(SolverKind::CriticalPath))
                .build_tree(&formula, mz, &pk);
            let t_ex = FragmentationTreeBuilder::new(cfg(SolverKind::Exact))
                .build_tree(&formula, mz, &pk);
            assert!(
                t_ex.total_score >= t_cp.total_score - 1e-6,
                "{f}: exact {} < critical {}",
                t_ex.total_score,
                t_cp.total_score
            );
            // colourful: no two nodes share a peak index
            let mut seen = std::collections::HashSet::new();
            for n in &t_ex.nodes {
                if let Some(pi) = n.peak_index {
                    assert!(seen.insert(pi), "{f}: exact tree reuses peak {pi}");
                }
            }
            assert_eq!(t_ex.edges.len(), t_ex.nodes.len() - 1, "{f}: exact tree not a tree");
        }
    }
}
