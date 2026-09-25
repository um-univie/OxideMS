// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Ported from SIRIUS <https://github.com/sirius-ms/sirius>,
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

//! SIRIUS's fragmentation-tree scoring objective, ported from the SIRIUS source.
//!
//! The MAP tree objective and every scorer below are described in
//!   Böcker, S. & Dührkop, K. "Fragmentation trees reloaded."
//!   J. Cheminform. 8, 5 (2016). (open access)
//! and the earlier Böcker & Rasche (Bioinformatics 2008) / Rasche et al. (2011).
//!
//! The numeric parameters (loss- and fragment-score dictionaries, the
//! log-normal loss-size fit, the radical / strange-element whitelists) are the
//! empirically-learned tables shipped in the SIRIUS open-source distribution's
//! default instrument profile
//!   `fragmentation_tree_construction/src/main/resources/profiles/default.json`
//! (LGPL-3.0-or-later, <https://github.com/sirius-ms/sirius>). They are
//! transcribed verbatim into `sirius_scoring_tables.rs` by
//! `scratch/gen_sirius_scoring.py`. See `ATTRIBUTION.md`.
//!
//! The scorers are ported from the SIRIUS source and adapted to work on our
//! `Formula` type, so this file is licensed under the LGPL-3.0-or-later (see
//! the notice above and `ATTRIBUTION.md`).
//!
//! Ported here: the loss (edge) and fragment/root (vertex) scorers of the
//! default profile, plus `CommonRootLossScorer` (cumulative precursor→fragment
//! loss bonus) and, at the root, the trained-SVM `ChemicalPriorScorer`
//! (`sirius_svm`), which only fires for candidates ≥ 100 Da.
//!
//! Not ported: the auto-tuning `Beautification` of the tree-size bonus (a fixed
//! `TreeSolverConfig::tree_size_bonus` stands in), adduct-switch losses, the
//! `CommonLossEdgeScorer` 2-step loss recombinator, `MultimereLossScorer`.

use super::formula::Formula;
use super::sirius_scoring_tables as t;
use crate::spectrum::Peak;
use rustc_hash::FxHashMap as HashMap;
use std::str::FromStr;
use std::sync::LazyLock;

fn parse_map(entries: &[(&str, f64)]) -> HashMap<Formula, f64> {
    entries
        .iter()
        .filter_map(|(s, v)| Formula::from_str(s).ok().map(|f| (f, *v)))
        .collect()
}
fn parse_set(entries: &[&str]) -> std::collections::HashSet<Formula> {
    entries.iter().filter_map(|s| Formula::from_str(s).ok()).collect()
}

static COMMON_LOSSES: LazyLock<HashMap<Formula, f64>> = LazyLock::new(|| parse_map(t::COMMON_LOSSES));
static IMPLAUSIBLE: LazyLock<std::collections::HashSet<Formula>> =
    LazyLock::new(|| parse_set(t::IMPLAUSIBLE_LOSSES));
static KNOWN_RADICALS: LazyLock<std::collections::HashSet<Formula>> =
    LazyLock::new(|| parse_set(t::KNOWN_RADICALS));
static STRANGE_LOSSES: LazyLock<std::collections::HashSet<Formula>> =
    LazyLock::new(|| parse_set(t::STRANGE_ELEMENT_LOSSES));
static COMMON_FRAGMENTS: LazyLock<HashMap<Formula, f64>> =
    LazyLock::new(|| parse_map(t::COMMON_FRAGMENTS));
static COMMON_ROOT_LOSSES: LazyLock<HashMap<Formula, f64>> =
    LazyLock::new(|| parse_map(t::COMMON_ROOT_LOSSES));
static COMMON_LOSS_RECOMBINED: LazyLock<HashMap<Formula, f64>> =
    LazyLock::new(|| parse_map(t::COMMON_LOSS_RECOMBINED));
static STRANGE_FRAGMENTS: LazyLock<std::collections::HashSet<Formula>> =
    LazyLock::new(|| parse_set(t::STRANGE_FRAGMENT_WHITELIST));

/// Complementary error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let tt = 1.0 / (1.0 + 0.5 * z);
    let ans = tt
        * (-z * z - 1.26551223
            + tt * (1.00002368
                + tt * (0.37409196
                    + tt * (0.09678418
                        + tt * (-0.18628806
                            + tt * (0.27886807
                                + tt * (-1.13520398
                                    + tt * (1.48851587
                                        + tt * (-0.82215223 + tt * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// `MassDeviationVertexScorer`: `ln(erfc(|Δm| / (sd·√2)))`, floored at −100.
/// `sd` is the instrument's absolute standard deviation at that m/z.
pub fn mass_deviation_score(delta_mass: f64, sd: f64) -> f64 {
    if sd <= 0.0 {
        return 0.0;
    }
    let s = erfc(delta_mass.abs() / (sd * std::f64::consts::SQRT_2)).ln();
    if !s.is_finite() || s < -100.0 { -100.0 } else { s }
}

fn doubled_rdbe(f: &Formula) -> f64 {
    2.0 * f.dbe()
}

/// `LossSizeScorer`: log-normal on the loss's exact mass.
/// `score = ln(max(1e-12, density(mass))) − normalization`.
pub fn loss_size_score(loss_mass: f64) -> f64 {
    let mu = t::LOSS_SIZE_MEAN;
    let sigma = t::LOSS_SIZE_VARIANCE.sqrt();
    let x = loss_mass.max(1e-9);
    // log-normal density
    let z = (x.ln() - mu) / sigma;
    let density = (-0.5 * z * z).exp() / (x * sigma * (2.0 * std::f64::consts::PI).sqrt());
    density.max(1e-12).ln() - t::LOSS_SIZE_NORMALIZATION
}

/// `FreeRadicalEdgeScorer` on a loss formula.
fn free_radical_score(loss: &Formula) -> f64 {
    let s = if KNOWN_RADICALS.contains(loss) {
        t::KNOWN_RADICAL_LOSS
    } else if loss.is_radical() {
        t::RADICAL_LOSS_PENALTY
    } else {
        0.0
    };
    s - t::FREE_RADICAL_NORMALIZATION
}

/// `DBELossScorer`: penalise a loss with negative (doubled) RDBE.
fn dbe_loss_score(loss: &Formula) -> f64 {
    let rdbe = doubled_rdbe(loss);
    if rdbe < 0.0 {
        (rdbe.abs() * t::DBE_LOSS_SCORE).max((0.05_f64).ln())
    } else {
        0.0
    }
}

/// `PhosphorScorer` on a loss (or non-root fragment): a phosphorus wants at
/// least one O or one S per P.
fn phosphor_score(f: &Formula) -> f64 {
    let p = f.p();
    if p > 0 && f.o() < p && f.s() < p {
        (0.25_f64).ln()
    } else {
        0.0
    }
}

/// `PureCarbonNitrogenLossScorer`: penalise a loss made only of C and/or N
/// (single Na/Cl/K losses are excluded, but we don't decompose those anyway).
fn pure_cn_loss_score(loss: &Formula) -> f64 {
    let cn = loss.c() + loss.n();
    let total = loss.total_atoms();
    if total > 0 && cn >= total {
        t::PURE_CN_LOSS_PENALTY
    } else {
        0.0
    }
}

/// `StrangeElementLossScorer`: bonus for a listed organo-S/Cl/Br/P/I loss.
fn strange_element_loss_score(loss: &Formula) -> f64 {
    if STRANGE_LOSSES.contains(loss) {
        t::STRANGE_ELEMENT_LOSS_BONUS
    } else {
        0.0
    }
}

/// `CommonLossEdgeScorer.getScoreForFormulaWithLossSize`:
/// `max(recombinatedList[L], commonLosses[L]) − normalization + lossSize(L)`,
/// where both maps default to 0 for an absent key. `recombinatedList` is the
/// `MinimalScoreRecombinator` output (a loss explained as two common losses in
/// one step, penalty −1) — see `COMMON_LOSS_RECOMBINED` in the tables.
fn common_loss_score(loss: &Formula, loss_mass: f64) -> f64 {
    let common = if IMPLAUSIBLE.contains(loss) {
        (0.01_f64).ln()
    } else {
        COMMON_LOSSES.get(loss).copied().unwrap_or(0.0)
    };
    let recombined = COMMON_LOSS_RECOMBINED.get(loss).copied().unwrap_or(0.0);
    recombined.max(common) - t::COMMON_LOSS_NORMALIZATION + loss_size_score(loss_mass)
}

/// Full SIRIUS edge (loss) score for an edge parent → child with the given
/// neutral loss formula and its exact mass.
pub fn edge_score(loss: &Formula, loss_mass: f64) -> f64 {
    common_loss_score(loss, loss_mass)
        + free_radical_score(loss)
        + dbe_loss_score(loss)
        + phosphor_score(loss)
        + pure_cn_loss_score(loss)
        + strange_element_loss_score(loss)
}

fn is_chno(f: &Formula) -> bool {
    f.p() == 0
        && f.s() == 0
        && f.f() == 0
        && f.cl() == 0
        && f.br() == 0
        && f.i() == 0
        && f.get(super::element::Element::Si) == 0
        && f.get(super::element::Element::B) == 0
}

/// `CommonFragmentsScore`: dict bonus minus normalization; unseen → −norm.
fn common_fragment_score(f: &Formula) -> f64 {
    match COMMON_FRAGMENTS.get(f) {
        Some(v) => v - t::COMMON_FRAGMENT_NORMALIZATION,
        None => -t::COMMON_FRAGMENT_NORMALIZATION,
    }
}

/// `StrangeElementSmallFragmentScorer` + `StrangeElementFragmentScorer`.
fn strange_fragment_score(f: &Formula, mass: f64) -> f64 {
    if mass <= t::SMALL_STRANGE_FRAGMENT_MAX_MASS && !is_chno(f) {
        return t::SMALL_STRANGE_FRAGMENT_BONUS;
    }
    if STRANGE_FRAGMENTS.contains(f) {
        return t::STRANGE_FRAGMENT_BONUS;
    }
    if !is_chno(f) && mass >= t::STRANGE_FRAGMENT_MIN_MASS {
        return t::STRANGE_FRAGMENT_PENALTY;
    }
    0.0
}

// `FragmentSizeScorer` (`2·max(0, 1−mass/200)` per fragment vertex) and
// `MultimereLossScorer` (`+10`/`+2` when the loss equals the child formula) were
// ported and benchmarked (FINDINGS round 18): both are net-negative on the CASMI
// blind honest-alphabet set (−0.9 pts top-1), so they are deliberately left out.

/// `CommonRootLossScorer`: a bonus keyed on the *cumulative* neutral loss from
/// the precursor down to this fragment (not the single edge loss). Listed total
/// losses — water, formic acid, HCl, sugars, phosphate — get their learned
/// score; everything else 0.
pub fn root_loss_score(cumulative_loss: &Formula) -> f64 {
    if cumulative_loss.total_atoms() == 0 {
        return 0.0;
    }
    match COMMON_ROOT_LOSSES.get(cumulative_loss) {
        Some(v) => v - t::COMMON_ROOT_LOSS_NORMALIZATION,
        None => 0.0,
    }
}

/// SIRIUS fragment-vertex score. `mass_dev` is [`mass_deviation_score`] for this
/// peak/formula; `is_root` selects the precursor-specific scorers. For the root,
/// the trained-SVM chemical prior ([`super::sirius_svm`]) is folded in — it only
/// contributes for candidates ≥ 100 Da.
pub fn vertex_score(f: &Formula, mass: f64, mass_dev: f64, is_root: bool) -> f64 {
    if is_root {
        let mut s = mass_dev;
        // PhosphorScorer, root variant: 2 O/S per P.
        if f.p() > 0 && (f.s() + f.o()) < 2 * f.p() {
            s += (0.05_f64).ln();
        }
        // StrangeElementRootScorer: penalty per Na/K/Cl/Br if not CHNO.
        if !is_chno(f) {
            let strange = f.cl() + f.br();
            s += strange as f64 * t::STRANGE_ROOT_PENALTY.abs().copysign(t::STRANGE_ROOT_PENALTY);
        }
        // ChemicalPriorScorer (trained linear SVM formula prior).
        s += super::sirius_svm::chemical_prior_score(f, mass);
        return s;
    }
    mass_dev + common_fragment_score(f) + strange_fragment_score(f, mass) + phosphor_score(f)
}

// ── MS1 isotope-pattern scoring ─────────────────────────────────────────────
//
// A port of SIRIUS's `IsotopePatternAnalysis` — `extractPattern` +
// `scoreFormulas` (`IsotopePatternAnalysis.java`), `getIsotopicMassWindow`
// (`PeriodicTable.java`) — with the default profile's four `patternScorers`:
//   MassDifferenceDeviationScorer  ln erfc on the isotope-peak *spacing*
//   MassDeviationScorer            ln erfc on the absolute isotope-peak m/z
//   NormalDistributedIntensityScorer  Gaussian log-odds on the intensity error
//   MissingPeakScorer              −λ·I (λ = 50) per theoretical peak ≥ 5 % that
//                                  the trusted prefix leaves out
// Each writes a cumulative-over-peaks score into `scoreBuffer`; the pattern
// score is the best prefix (`max_k scoreBuffer[k]`) — SIRIUS decides how many
// isotope peaks to trust, and because the buffer is sized to the *extracted*
// measured pattern the trailing entries stay 0, so trusting nothing floors the
// score at 0.
//
// Parameters are the SIRIUS `default.json` / config values: 10 ppm
// `allowedMassDeviation` + `standardMassDeviation`, 5 ppm `massDifferenceDeviation`,
// σ_abs 0.02 / σ_rel 0.08, λ 50 / threshold 0.05, the [1,2,3] / [0.2,0.1,0.01]
// intensity-dependency knots, `minimalIntensityToConsider` 0.01. Ported from
// the named classes; see `ATTRIBUTION.md`.

const MS1_ALLOWED_PPM: f64 = 10.0;
const MS1_STD_PPM: f64 = 10.0;
const MS1_DIFF_PPM: f64 = 5.0;
const ISO_INT_ABS_ERR: f64 = 0.02;
const ISO_INT_REL_ERR: f64 = 0.08;
const MISSING_LAMBDA: f64 = 50.0;
const MISSING_THRESHOLD: f64 = 0.05;
const ISO_MIN_INTENSITY: f64 = 0.01;
/// Elements whose heavy isotopes bound the per-offset mass-defect window — the
/// mass-defect extremes are ²H (+) and ³⁴S (−); Cl/Br sit inside that. Matches
/// SIRIUS's default `FormulaConstraints` alphabet plus the common halogens.
const ISO_WINDOW_ELEMENTS: [super::element::Element; 8] = {
    use super::element::Element::{Br, Cl, C, H, N, O, P, S};
    [C, H, N, O, P, S, Cl, Br]
};

/// `PiecewiseLinearFunctionIntensityDependency([1,2,3], [0.2,0.1,0.01])`:
/// widen the mass tolerance up to 3× for weak isotope peaks.
fn iso_intensity_multiplier(intensity: f64) -> f64 {
    const BAGS: [f64; 3] = [0.2, 0.1, 0.01];
    const VALUES: [f64; 3] = [1.0, 2.0, 3.0];
    let mut i = 0;
    while i < BAGS.len() {
        if intensity > BAGS[i] {
            break;
        }
        i += 1;
    }
    if i >= BAGS.len() {
        return VALUES[2];
    }
    if i == 0 {
        return VALUES[0];
    }
    let p = (intensity - BAGS[i]) / (BAGS[i - 1] - BAGS[i]);
    p * VALUES[i - 1] + (1.0 - p) * VALUES[i]
}

fn safe_ln(x: f64) -> f64 {
    x.max(1e-300).ln()
}

/// `PeriodicTable.getIsotopicMassWindow`: the `[lo, hi]` m/z interval the
/// `peak_offset`-th isotope peak of an M0 at `m0_mz` could fall in, given the
/// alphabet's isotope mass defects, expanded by `allowedMassDeviation`.
fn isotopic_mass_window(m0_mz: f64, peak_offset: usize) -> (f64, f64) {
    let mut min_diff = f64::INFINITY;
    let mut max_diff = f64::NEG_INFINITY;
    for el in ISO_WINDOW_ELEMENTS {
        let mono = el.monoisotopic_mass();
        for iso in super::element::get_element_isotopes(el).as_slice() {
            let i = iso.mass_number_delta;
            if i == 0 {
                continue;
            }
            if i > peak_offset {
                break;
            }
            // (mass defect of this isotope step) scaled by how many such steps
            // make up `peak_offset` — integer division, exactly as in SIRIUS.
            let mut diff = (iso.mass - mono) - i as f64;
            diff *= (peak_offset / i) as f64;
            min_diff = min_diff.min(diff);
            max_diff = max_diff.max(diff);
        }
    }
    let centre = m0_mz + peak_offset as f64;
    if !min_diff.is_finite() {
        return (centre, centre);
    }
    let a = centre + min_diff;
    let b = centre + max_diff;
    (a - MS1_ALLOWED_PPM * a * 1e-6, b + MS1_ALLOWED_PPM * b * 1e-6)
}

/// `IsotopePatternAnalysis.extractPattern`: anchor M0 on the most intense peak
/// within `allowedMassDeviation` of `m0_target`, then for k = 1..5 merge every
/// peak in the k-th isotopic mass window (intensity-weighted centroid, summed
/// intensity), stopping at the first empty window.
fn extract_pattern(ms1_peaks: &[Peak], m0_target: f64) -> Vec<(f64, f64)> {
    let allowed = MS1_ALLOWED_PPM * m0_target * 1e-6;
    let Some(m0) = ms1_peaks
        .iter()
        .filter(|p| (p.mz - m0_target).abs() <= allowed)
        .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
    else {
        return Vec::new();
    };
    let mut out = vec![(m0.mz as f64, m0.intensity as f64)];
    for k in 1..=5 {
        let (mut lo, mut hi) = isotopic_mass_window(out[0].0, k);
        lo -= MS1_DIFF_PPM * lo * 1e-6;
        hi += MS1_DIFF_PPM * hi * 1e-6;
        let mut mz_acc = 0.0;
        let mut int_acc = 0.0;
        for p in ms1_peaks {
            let mz = p.mz as f64;
            if mz >= lo && mz <= hi {
                let int = p.intensity as f64;
                mz_acc += mz * int;
                int_acc += int;
            }
        }
        if int_acc <= 0.0 {
            break;
        }
        out.push((mz_acc / int_acc, int_acc));
    }
    out
}

/// `ClippedPeakIsNoiseScorer` from the SIRIUS default profile: a Pareto model of
/// the noise intensity distribution, `ParetoDistribution$EstimateByMedian` with
/// `xmin = 0.002` and `beta = 1e-05`.
const NOISE_PARETO_XMIN: f64 = 0.002;
const NOISE_BETA: f64 = 1e-5;

/// `ParetoDistribution.EstimateByMedian.extimateByMedian`: the shape that puts the
/// distribution's median at the observed median noise intensity.
fn pareto_k(median_noise: f64) -> f64 {
    std::f64::consts::LN_2 / (median_noise / NOISE_PARETO_XMIN).ln()
}

/// Survival function `1 - F(x)` of `ParetoDistribution`: the probability that a
/// noise peak is at least this intense. Zero density below `xmin`, so everything
/// down there is indistinguishable from noise.
fn pareto_sf(x: f64, k: f64) -> f64 {
    if x < NOISE_PARETO_XMIN { 1.0 } else { (NOISE_PARETO_XMIN / x).powf(k) }
}

/// `ClippedPeakIsNoiseScorer.score`: how much better than noise this peak's
/// intensity is, as `-ln((sf(x) - sf(1) + beta) / (1 - sf(1) + beta))` where
/// `x = relative intensity / max relative intensity`.
///
/// Zero at the noise floor and rising monotonically to `-ln(beta / (1 - sf(1) +
/// beta))` at the base peak — the term that makes an intense fragment worth more
/// as evidence than a trace one. Under [`ScoringModel::Sirius`] the vertex weight
/// had no intensity term at all before this: a 0.1 % peak and a 90 % peak scored
/// alike.
///
/// `median_noise` is SIRIUS's `MedianNoiseIntensity` annotation, relative to the
/// base peak. It has no faithful value to copy — 5.6.3 defaults it to `NaN` and
/// nothing in the distribution sets it, so it is a parameter here, not a constant.
pub fn peak_is_noise_score(rel_intensity: f64, max_rel_intensity: f64, median_noise: f64) -> f64 {
    if rel_intensity <= 0.0 || max_rel_intensity <= 0.0 || median_noise <= NOISE_PARETO_XMIN {
        return 0.0;
    }
    let k = pareto_k(median_noise);
    let p1 = pareto_sf(1.0, k);
    let px = pareto_sf(rel_intensity / max_rel_intensity, k);
    let score = -((px - p1 + NOISE_BETA) / (1.0 - p1 + NOISE_BETA)).ln();
    if score.is_finite() { score } else { 0.0 }
}

/// SIRIUS MS1 isotope-pattern score for a candidate precursor formula against an
/// observed MS1 envelope (a peak list spanning the precursor and its isotopes).
/// Added once to the root vertex weight. Returns 0 when there is no usable
/// evidence.
pub fn isotope_score(
    formula: &Formula,
    ms1_peaks: &[Peak],
    precursor_mz: f64,
    adduct: &super::adduct::Adduct,
) -> f64 {
    if ms1_peaks.len() < 2 {
        return 0.0;
    }

    // Theoretical pattern → ion m/z, normalised to M0 = 1, ≤ 10 peaks, trailing
    // peaks below the cutoff dropped (`patternGenerator.simulatePattern`).
    let theo_raw = super::scoring::calculate_isotope_distribution(formula, 9);
    let Some(first) = theo_raw.first() else { return 0.0 };
    if first.abundance <= 0.0 {
        return 0.0;
    }
    let t0 = first.abundance;
    let mut theo: Vec<(f64, f64)> = Vec::with_capacity(10);
    for tp in &theo_raw {
        let rel = tp.abundance / t0;
        if rel < ISO_MIN_INTENSITY || theo.len() == 10 {
            break;
        }
        theo.push((adduct.neutral_mass_to_ion_mz(tp.mz), rel));
    }
    if theo.len() < 2 {
        return 0.0;
    }
    // `scoreFormulas` puts BOTH patterns on `Normalization.Max(1)`. The trim
    // above is relative to M0 (that is what `simulatePattern`'s probability
    // cutoff is relative to), but M0 is only the tallest peak for a pattern with
    // no heavy halogen: Br2 puts ~1.95 at M+2, Cl4 ~1.3. Leaving `theo` on M0 = 1
    // while `meas` is on max = 1 compared the two on different scales, and the
    // MissingPeakScorer then charged lambda * 1.95 for isotope peaks that were
    // present and correct. C13H18Br2N2O scored -141 and ranked 968th on its own
    // textbook Br2 envelope.
    let tmax = theo.iter().map(|p| p.1).fold(0.0f64, f64::max);
    if tmax <= 0.0 {
        return 0.0;
    }
    for p in &mut theo {
        p.1 /= tmax;
    }

    // Measured pattern, extracted once (candidate-independent positions).
    let mut meas = extract_pattern(ms1_peaks, precursor_mz);
    if meas.len() < 2 {
        return 0.0;
    }
    // scoreFormulas normalisation: Sum(1); reject / trim on the 0.01 cutoff;
    // then Max(1).
    let sum: f64 = meas.iter().map(|p| p.1).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    for p in &mut meas {
        p.1 /= sum;
    }
    if meas[0].1 < ISO_MIN_INTENSITY {
        return 0.0;
    }
    while meas.last().is_some_and(|p| p.1 < ISO_MIN_INTENSITY) {
        meas.pop();
    }
    if meas.len() < 2 {
        return 0.0;
    }
    let mmax = meas.iter().map(|p| p.1).fold(0.0f64, f64::max);
    for p in &mut meas {
        p.1 /= mmax;
    }
    // `theo` was put on the same Max(1) normalization above.

    // scoreBuffer is sized to the full extracted measured pattern; the per-peak
    // scorers run only over the aligned prefix `n`.
    let n_scored = meas.len().min(theo.len());
    let mut buf = vec![0.0f64; meas.len()];
    let sqrt2 = std::f64::consts::SQRT_2;

    // MassDifferenceDeviationScorer — spacing relative to M0.
    {
        let (mz0, th0) = (meas[0].0, theo[0].0);
        let mut score = 0.0;
        for i in 1..n_scored {
            let d_meas = meas[i].0 - mz0;
            let d_theo = theo[i].0 - th0;
            let sd = MS1_DIFF_PPM * meas[i].0 * 1e-6 * iso_intensity_multiplier(meas[i].1);
            score += safe_ln(erfc((d_theo - d_meas).abs() / (sqrt2 * sd)));
            buf[i] += score;
        }
    }
    // MassDeviationScorer — absolute isotope-peak m/z.
    {
        let mut score = 0.0;
        for i in 0..n_scored {
            let sd = MS1_STD_PPM * meas[i].0 * 1e-6 * iso_intensity_multiplier(meas[i].1);
            score += safe_ln(erfc((theo[i].0 - meas[i].0).abs() / (sqrt2 * sd)));
            buf[i] += score;
        }
    }
    // NormalDistributedIntensityScorer (LOGODDS).
    {
        let mut score = 0.0;
        for i in 1..n_scored {
            let mi = meas[i].1.max(1e-4);
            let ti = theo[i].1;
            let delta = mi - ti;
            let var = ISO_INT_ABS_ERR * ISO_INT_ABS_ERR + mi * mi * ISO_INT_REL_ERR * ISO_INT_REL_ERR;
            let denom = 2.0 * std::f64::consts::PI * mi * ISO_INT_REL_ERR * ISO_INT_ABS_ERR;
            let peak_p = (-(delta * delta) / (2.0 * var)).exp() / denom;
            score += safe_ln(peak_p);
            let sigma = mi * 2.0 * ISO_INT_REL_ERR + 2.0 * ISO_INT_ABS_ERR;
            let ref_p = (-(sigma * sigma) / (2.0 * var)).exp() / denom;
            score -= safe_ln(ref_p);
            buf[i] += score;
        }
    }
    // MissingPeakScorer — over the full theoretical pattern.
    {
        let mut score = 0.0;
        for k in (0..theo.len()).rev() {
            let ti = theo[k].1;
            if ti >= MISSING_THRESHOLD {
                if k < buf.len() {
                    buf[k] += score;
                }
                score -= ti * MISSING_LAMBDA;
            }
        }
    }

    let best = buf.into_iter().fold(f64::NEG_INFINITY, f64::max);
    if best.is_finite() { best } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::Peak;
    use std::str::FromStr;

    /// The M+2 step separates a chlorinated formula from its near-isobaric
    /// phosphorus decoy: given a Cl compound's real MS1 envelope, the true
    /// formula must out-score the P swap by a wide margin.
    #[test]
    fn isotope_score_separates_chlorine_from_phosphorus() {
        let add = super::super::adduct::Adduct::protonated();
        // Challenge-179 MS1 envelope (m/z:intensity), true C14H9ClF2N2O2.
        let ms1: Vec<Peak> = [
            (309.997125, 453500.91),
            (311.039058, 44056308.0),
            (312.042508, 7124229.0),
            (312.066159, 4558227.0),
            (313.036101, 14442985.0),
            (313.069366, 598857.69),
            (314.039491, 1985078.25),
            (314.063203, 2801868.0),
        ]
        .iter()
        .map(|&(mz, i)| Peak { mz, intensity: i })
        .collect();

        let truth = Formula::from_str("C14H9ClF2N2O2").unwrap();
        let decoy = Formula::from_str("C13H9F2N2O3P").unwrap();
        let pmz = add.neutral_mass_to_ion_mz(truth.exact_mass());
        let s_true = isotope_score(&truth, &ms1, pmz, &add);
        let s_decoy = isotope_score(&decoy, &ms1, pmz, &add);
        assert!(
            s_true > s_decoy + 3.0,
            "Cl truth {s_true:.2} should beat P decoy {s_decoy:.2} by >3"
        );
    }

    /// A pattern whose tallest peak is not M0. Br2 puts ~1.95 at M+2, so
    /// normalising the theoretical pattern on M0 while the measured one is on its
    /// max compared the two on different scales and made the MissingPeakScorer
    /// charge lambda for peaks that were present. The true formula scored -141 on
    /// its own textbook envelope and ranked 968th of 972 candidates.
    #[test]
    fn isotope_score_handles_a_pattern_whose_tallest_peak_is_not_m0() {
        let add = super::super::adduct::Adduct::protonated();
        // Challenge-094 MS1 envelope, true C13H18Br2N2O. Includes the co-eluting
        // 376.239 / 377.243 ion the real survey scan carries.
        let ms1: Vec<Peak> = [
            (376.23942, 0.3107),
            (376.98573, 0.5239),
            (377.09013, 0.0109),
            (377.24269, 0.0745),
            (377.98921, 0.0632),
            (378.98363, 1.0000),
            (379.98741, 0.1135),
            (380.98156, 0.4721),
            (381.98503, 0.0680),
        ]
        .iter()
        .map(|&(mz, i)| Peak { mz, intensity: i })
        .collect();

        let truth = Formula::from_str("C13H18Br2N2O").unwrap();
        let decoy = Formula::from_str("C9H9BrF4N4O3").unwrap();
        let pmz = 376.985865;
        let s_true = isotope_score(&truth, &ms1, pmz, &add);
        let s_decoy = isotope_score(&decoy, &ms1, pmz, &add);
        assert!(
            s_true > -15.0,
            "a correct Br2 formula on its own envelope must not be crushed: {s_true:.2}"
        );
        assert!(
            s_true > s_decoy + 3.0,
            "Br2 truth {s_true:.2} should beat the single-Br decoy {s_decoy:.2} by >3"
        );
    }

    /// The noise term must be zero at the noise floor, monotone in intensity, and
    /// bounded at the base peak — it is added to every fragment vertex, so an
    /// unbounded or non-monotone term would quietly reshape every tree.
    #[test]
    fn peak_is_noise_score_rises_monotonically_from_the_noise_floor() {
        let median = 0.005;
        // At or below xmin = 0.002 a peak is indistinguishable from noise.
        assert_eq!(peak_is_noise_score(0.001, 1.0, median), 0.0);
        assert_eq!(peak_is_noise_score(0.002, 1.0, median), 0.0);

        let mut previous = 0.0;
        for rel in [0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0] {
            let score = peak_is_noise_score(rel, 1.0, median);
            assert!(score.is_finite(), "score at {rel} is not finite");
            assert!(score > previous, "score fell from {previous} to {score} at {rel}");
            previous = score;
        }
        // The base peak's bonus is -ln(beta / (1 - sf(1) + beta)), ~11.5 here.
        assert!((5.0..20.0).contains(&previous), "base-peak bonus {previous} out of range");

        // Scored relative to the base peak, not in absolute terms.
        let a = peak_is_noise_score(0.05, 0.1, median);
        let b = peak_is_noise_score(0.5, 1.0, median);
        assert!((a - b).abs() < 1e-12, "{a} != {b}: the score must scale with the base peak");

        // A median at or below xmin gives no usable shape, so the term is off.
        assert_eq!(peak_is_noise_score(1.0, 1.0, 0.002), 0.0);
    }

    #[test]
    fn isotope_score_is_zero_without_a_pattern() {
        let add = super::super::adduct::Adduct::protonated();
        let f = Formula::from_str("C8H10N4O2").unwrap();
        assert_eq!(isotope_score(&f, &[], 195.0877, &add), 0.0);
        let one = [Peak { mz: 195.0877, intensity: 1.0 }];
        assert_eq!(isotope_score(&f, &one, 195.0877, &add), 0.0);
    }
}
