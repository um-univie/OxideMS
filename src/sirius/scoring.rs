use super::element::{Element, get_element_isotopes, ALL_ELEMENTS};
use super::formula::Formula;
use crate::spectrum::Peak;
use crate::workflow::MassTolerance;
use std::cmp::Ordering;

/// Theoretical isotopic peak in a simulated mass distribution
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TheoreticalIsotopePeak {
    pub mz: f64,
    pub abundance: f64,          // Fractional abundance (sums to 1.0 across cluster)
    pub nominal_delta: usize,    // 0 = M, 1 = M+1, 2 = M+2, ...
}

/// A candidate molecular formula with detailed probabilistic scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredFormula {
    pub formula: Formula,
    pub exact_mass: f64,
    pub mass_error_da: f64,
    pub mass_error_ppm: f64,
    pub mass_score: f64,
    pub isotope_score: f64,
    pub prior_score: f64,
    pub total_score: f64,
}

impl ScoredFormula {
    pub fn new(
        formula: Formula,
        target_mass: f64,
        mass_score: f64,
        isotope_score: f64,
        prior_score: f64,
    ) -> Self {
        let exact_mass = formula.exact_mass();
        let mass_error_da = exact_mass - target_mass;
        let mass_error_ppm = (mass_error_da / target_mass) * 1_000_000.0;
        let total_score = mass_score + isotope_score + prior_score;

        Self {
            formula,
            exact_mass,
            mass_error_da,
            mass_error_ppm,
            mass_score,
            isotope_score,
            prior_score,
            total_score,
        }
    }
}

impl PartialOrd for ScoredFormula {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Higher score is better (descending sort)
        other.total_score.partial_cmp(&self.total_score)
    }
}

/// Log-odds tie-break penalty applied to a radical (odd-electron) fragment
/// formula in `rank_fragment_candidates`. Small on purpose: the real cost of a
/// radical fragment is charged on its tree edge (radical neutral loss), so this
/// only needs to break a mass tie in the candidate list, not evict the fragment.
const RADICAL_FRAGMENT_PENALTY: f64 = -0.5;

/// Computes theoretical isotope distributions using polynomial expansion
pub fn calculate_isotope_distribution(formula: &Formula, max_nominal_delta: usize) -> Vec<TheoreticalIsotopePeak> {
    let max_k = max_nominal_delta + 1;
    let mut dist = vec![(0.0f64, 0.0f64); max_k];
    dist[0] = (0.0, 1.0); // Start with 1.0 at M+0

    for &el in &ALL_ELEMENTS {
        let count = formula.get(el);
        if count == 0 {
            continue;
        }

        let isotopes = get_element_isotopes(el);

        // Convolve 'count' times with the element's isotope distribution
        for _ in 0..count {
            let mut next_dist = vec![(0.0f64, 0.0f64); max_k];

            for (curr_k, &(curr_mass_prob, curr_prob)) in dist.iter().enumerate() {
                if curr_prob <= 1e-12 {
                    continue;
                }
                // `dist` bins accumulate (sum of mass*prob, sum of prob); the
                // representative mass of a bin is the ratio. Adding `iso.mass` to
                // the raw mass*prob accumulator, as this used to, is only correct
                // for the very first element (bin 0 starts at mass 0) and produces
                // nonsense isotope-peak masses for any multi-element formula.
                let curr_mass = curr_mass_prob / curr_prob;

                for iso in isotopes.as_slice() {
                    let next_k = curr_k + iso.mass_number_delta;
                    if next_k < max_k {
                        let p = curr_prob * iso.abundance;
                        let m = curr_mass + iso.mass;
                        next_dist[next_k].0 += m * p;
                        next_dist[next_k].1 += p;
                    }
                }
            }

            dist = next_dist;
        }
    }

    let mut result = Vec::with_capacity(max_k);
    let total_abundance: f64 = dist.iter().map(|(_, p)| *p).sum();

    for (k, &(mass_sum, prob)) in dist.iter().enumerate() {
        if prob > 1e-7 && total_abundance > 0.0 {
            let avg_mz = mass_sum / prob;
            result.push(TheoreticalIsotopePeak {
                mz: avg_mz,
                abundance: prob / total_abundance,
                nominal_delta: k,
            });
        }
    }

    result
}

/// Probabilistic Formula Scorer
#[derive(Debug, Clone)]
pub struct FormulaScorer {
    pub mass_sigma_ppm: f64,
    pub mass_sigma_min_da: f64,
    pub isotope_sigma_int: f64,
    pub max_isotope_peaks: usize,
    pub missing_peak_penalty: f64,
}

impl Default for FormulaScorer {
    fn default() -> Self {
        Self {
            mass_sigma_ppm: 3.0,          // 3 ppm standard deviation for high-res Orbitrap
            mass_sigma_min_da: 0.001,     // minimum 1 mDa uncertainty
            isotope_sigma_int: 0.10,      // 10% realistic experimental intensity uncertainty
            max_isotope_peaks: 5,         // Compare up to M+4
            missing_peak_penalty: -5.0,
        }
    }
}

impl FormulaScorer {
    pub fn new(mass_sigma_ppm: f64, isotope_sigma_int: f64) -> Self {
        Self {
            mass_sigma_ppm,
            mass_sigma_min_da: 0.001,
            isotope_sigma_int,
            max_isotope_peaks: 5,
            missing_peak_penalty: -5.0,
        }
    }

    /// Compute Gaussian log-likelihood for mass accuracy:
    /// ln N(delta_m | 0, sigma^2)
    pub fn score_mass_accuracy(&self, formula_mass: f64, observed_mass: f64) -> f64 {
        let delta_m = formula_mass - observed_mass;
        let sigma = (observed_mass * self.mass_sigma_ppm * 1e-6).max(self.mass_sigma_min_da);
        let z = delta_m / sigma;
        -0.5 * (z * z) - (sigma * (2.0 * std::f64::consts::PI).sqrt()).ln()
    }

    /// Compute isotope pattern log-likelihood comparing observed spectral peaks against theoretical pattern for a specific adduct / charge state.
    /// Scales isotope peak delta spacing and window by 1 / |z|.
    pub fn score_isotope_pattern_with_adduct(
        &self,
        formula: &Formula,
        observed_peaks: &[Peak],
        precursor_mz: f64,
        adduct: &super::adduct::Adduct,
        tolerance: MassTolerance,
    ) -> f64 {
        let z = adduct.charge.abs().max(1) as f64;
        let min_range = precursor_mz - (0.5 / z);
        let max_range = precursor_mz + (self.max_isotope_peaks as f64 + 0.5) / z;
        let iso_peaks: Vec<&Peak> = observed_peaks
            .iter()
            .filter(|p| p.mz >= min_range && p.mz <= max_range)
            .collect();

        // If no peaks exist in the isotope window, return 0.0
        if iso_peaks.is_empty() {
            return 0.0;
        }

        let theor_pattern = calculate_isotope_distribution(formula, self.max_isotope_peaks);
        if theor_pattern.first().map_or(true, |t| t.abundance <= 0.0) {
            return 0.0;
        }

        // Check if the precursor was observed in the MS1 envelope within tolerance
        let precursor_observed = iso_peaks.iter().any(|p| {
            let err = (p.mz - precursor_mz).abs();
            match tolerance {
                MassTolerance::Da(da) => err <= da,
                MassTolerance::Ppm(ppm) => (err / precursor_mz) * 1e6 <= ppm,
            }
        });

        // Halogen/sulfur fine-structure presence (e.g. 37Cl, 81Br, 34S doublets split ~9-11 mDa from 13C2)
        let has_fine_structure = formula.cl() > 0 || formula.br() > 0 || formula.s() > 0;

        let match_peak = |theor_mz: f64, k: usize| -> Option<&&Peak> {
            let base_tol_da = match tolerance {
                MassTolerance::Da(da) => da,
                MassTolerance::Ppm(ppm) => (ppm * theor_mz * 1e-6).max(0.001),
            };
            let eff_tol_da = if k >= 2 && has_fine_structure {
                base_tol_da.max(0.012 / z)
            } else {
                base_tol_da
            };

            iso_peaks
                .iter()
                .filter(|p| (p.mz - theor_mz).abs() <= eff_tol_da)
                .min_by(|a, b| (a.mz - theor_mz).abs().total_cmp(&(b.mz - theor_mz).abs()))
        };

        // Anchor on the monoisotopic peak. Score isotope *ratios to M0* rather
        // than fractions of the whole window — robust to unrelated co-eluting
        // ions a wide MS1 slice picks up.
        let m0_mz = adduct.neutral_mass_to_ion_mz(theor_pattern[0].mz);
        let Some(m0_peak) = match_peak(m0_mz, 0) else {
            // If the precursor was observed in the MS1 envelope, but this candidate
            // failed to match it within tolerance, penalize heavily.
            // If the precursor was not detected in the MS1 envelope at all, no usable evidence (0.0).
            return if precursor_observed {
                (self.missing_peak_penalty * 2.0).clamp(-15.0, 0.0)
            } else {
                0.0
            };
        };
        let m0_int = m0_peak.intensity as f64;
        if m0_int <= 0.0 {
            return 0.0;
        }
        let theor0 = theor_pattern[0].abundance;

        // M0 mass accuracy: a real bonus for the precursor sitting where the
        // candidate says it should.
        let m0_score = 0.15 * self.score_mass_accuracy(m0_mz, m0_peak.mz).min(6.0);

        // Evaluate isotope steps with bounded Huber loss and prefix optimal score
        let mut prefix_scores = Vec::with_capacity(theor_pattern.len());
        let mut current_score = m0_score;
        prefix_scores.push(current_score);

        for (k, theor) in theor_pattern.iter().enumerate().skip(1) {
            let theor_ratio = theor.abundance / theor0;
            if theor_ratio < 1e-3 {
                continue;
            }
            let theor_mz = adduct.neutral_mass_to_ion_mz(theor.mz);
            match match_peak(theor_mz, k) {
                Some(p) => {
                    let obs_ratio = p.intensity as f64 / m0_int;
                    let sigma = (self.isotope_sigma_int * (1.0 + theor_ratio)).max(0.03);
                    let zdev = (obs_ratio - theor_ratio) / sigma;
                    let abs_z = zdev.abs();

                    // Bounded Huber loss: Gaussian for |z| <= 2, linear up to |z| = 4, capped at -5.0.
                    // Protects against co-eluting chemical contaminants (e.g. solvent/adduct clusters).
                    let huber_pen = if abs_z <= 2.0 {
                        -0.5 * abs_z * abs_z
                    } else if abs_z <= 4.0 {
                        -2.0 - 1.5 * (abs_z - 2.0)
                    } else {
                        -5.0
                    };
                    current_score += huber_pen;

                    // Mass accuracy score for isotope peak
                    let mass_acc = if k >= 2 && has_fine_structure {
                        // Widen mass accuracy sigma in quadrature for resolved fine structure
                        let delta_m = theor_mz - p.mz;
                        let base_sigma = (p.mz * self.mass_sigma_ppm * 1e-6).max(self.mass_sigma_min_da);
                        let eff_sigma = (base_sigma * base_sigma + (0.006 / z) * (0.006 / z)).sqrt();
                        let z_m = delta_m / eff_sigma;
                        -0.5 * (z_m * z_m) - (eff_sigma * (2.0 * std::f64::consts::PI).sqrt()).ln()
                    } else {
                        self.score_mass_accuracy(theor_mz, p.mz)
                    };
                    current_score += 0.1 * mass_acc.min(6.0);
                }
                None if theor_ratio > 0.10 => {
                    // Expected a visible isotope peak, didn't see it (e.g. the Cl/Br M+2 step)
                    current_score += -3.0 * theor_ratio.min(1.0);
                }
                None => {}
            }
            prefix_scores.push(current_score);
        }

        // Prefix-optimal evaluation: allow stopping before noisy trailing minor peaks,
        // but deduct penalty for any major (> 15%) theoretical peak omitted from the prefix.
        let mut best_score = f64::NEG_INFINITY;
        for (cut_k, &p_score) in prefix_scores.iter().enumerate() {
            let mut missing_tail_pen = 0.0;
            for theor in theor_pattern.iter().skip(cut_k + 1) {
                let theor_ratio = theor.abundance / theor0;
                if theor_ratio > 0.15 {
                    missing_tail_pen += 3.0 * theor_ratio.min(1.0);
                }
            }
            let total_k = p_score - missing_tail_pen;
            if total_k > best_score {
                best_score = total_k;
            }
        }

        // Bounded: a decisive isotope mismatch (wrong halogen count) should
        // clearly outvote the tree, but a noisy real-world survey scan must not
        // be able to drive the score to -100 the way fraction-of-window scoring
        // did.
        best_score.clamp(-15.0, 6.0)
    }

    /// Compute isotope pattern log-likelihood comparing observed spectral peaks against theoretical pattern.
    /// Defaults to singly charged protonated adduct [M+H]+.
    pub fn score_isotope_pattern(
        &self,
        formula: &Formula,
        observed_peaks: &[Peak],
        target_mass: f64,
        tolerance: MassTolerance,
    ) -> f64 {
        self.score_isotope_pattern_with_adduct(
            formula,
            observed_peaks,
            target_mass,
            &super::adduct::Adduct::protonated(),
            tolerance,
        )
    }

    /// Natural chemical prior score based on DBE, elemental composition, and heteroatom ratios
    /// Consistent with SIRIUS / Seven Golden Rules prior distributions
    pub fn score_prior(&self, formula: &Formula) -> f64 {
        let mut prior = 0.0;
        let c = formula.c();
        let n = formula.n();
        let o = formula.o();
        let p = formula.p();
        let s = formula.s();

        // 1. DBE prior
        let dbe = formula.dbe();
        if dbe < 0.0 {
            prior -= 5.0;
        } else if dbe > 20.0 {
            prior -= (dbe - 20.0) * 0.3;
        }

        // 2. Heteroatom priors (Phosphorus and Sulfur in natural metabolites)
        // Phosphorus is rare (~3-5% of metabolites) and Sulfur is uncommon (~10%)
        // Baseline penalties prevent spurious P/S candidate formulas from outcompeting CHNO formulas
        if p > 0 {
            prior -= 1.5 + (p - 1) as f64 * 1.8;
            // Non-phosphate P (less than 3 oxygens per phosphorus) is extremely rare in nature
            if o < 3 * p {
                prior -= 2.0;
            }
        }
        if s > 0 {
            prior -= 0.8 + (s - 1) as f64 * 1.2;
        }
        if p > 0 && s > 0 {
            prior -= 2.0;
        }

        // 3. Nitrogen-to-Carbon ratio prior (Seven Golden Rules for natural metabolites)
        if c > 0 {
            let n_c = n as f64 / c as f64;
            if n_c > 0.55 {
                prior -= (n_c - 0.55) * 3.0;
            }
            if n_c > 0.8 {
                prior -= (n_c - 0.8) * 4.0;
            }
            let o_c = o as f64 / c as f64;
            if o_c > 1.2 {
                prior -= (o_c - 1.2) * 1.5;
            }
        }

        // 4. Halogens prior
        let halogens = formula.f() + formula.cl() + formula.br() + formula.i();
        if halogens > 4 {
            prior -= (halogens - 4) as f64 * 1.5;
        }

        prior
    }

    /// Chemical prior for a *fragment* sub-formula.
    ///
    /// The whole-molecule heteroatom-ratio terms in `score_prior` (N/C, O/C) are
    /// Seven-Golden-Rules statistics about intact natural products, and applying
    /// them to fragments suppresses exactly the small, N- or O-rich ions that
    /// dominate real CID spectra — triazolium (C2H3N3, N/C 1.5), imidazolium,
    /// protonated nucleobases, acylium/oxocarbenium (O/C 1-2). SIRIUS does not
    /// charge a fragment for its elemental ratios; it only rules out the
    /// impossible (negative DBE) and rewards known fragment ions. This keeps the
    /// DBE and rare-heteroatom guards and drops the ratio penalties.
    pub fn score_prior_fragment(&self, formula: &Formula) -> f64 {
        let mut prior = 0.0;

        // A fragment tree is always built under a *known* precursor formula, and
        // a fragment is a subformula of it — so its P / S / halogen content is
        // already sanctioned by that formula. The absolute heteroatom penalties
        // that live in `score_prior` (whose job is to keep spurious P/S formulas
        // from winning a de-novo *precursor* search) are wrong here: applied to
        // fragments they made whole compound classes untreeable — every P/S
        // fragment vertex of an organophosphate went negative and dropped below
        // the noise floor, collapsing the tree to its root. SIRIUS conditions the
        // fragment prior on the molecular formula; not conditioning at all is the
        // equivalent. Only genuinely impossible compositions are charged.
        let dbe = formula.dbe();
        if dbe < 0.0 {
            prior -= 5.0;
        } else if dbe > 20.0 {
            prior -= (dbe - 20.0) * 0.3;
        }

        prior
    }

    /// Score and rank a list of candidate formulas against an observed isotope envelope.
    pub fn rank_candidates(
        &self,
        candidates: &[Formula],
        target_mass: f64,
        observed_peaks: &[Peak],
        tolerance: MassTolerance,
    ) -> Vec<ScoredFormula> {
        self.rank(candidates, target_mass, Some((observed_peaks, tolerance)))
    }

    /// Score and rank candidate sub-formulas for a single MS/MS fragment peak.
    ///
    /// A fragment has no isotope envelope to be scored against: the peak list holds
    /// one peak per fragment, so an isotope pattern score derived from it is not
    /// evidence about the candidate, it is a penalty proportional to the candidate's
    /// M+1/M+2 abundance — that is, a penalty for being a larger formula. Fragment
    /// candidates are scored on mass accuracy and chemical prior only.
    ///
    /// A radical (odd-electron) fragment formula gets a small tie-break penalty.
    /// The chemical implausibility of a radical fragment is already charged on its
    /// tree edge — a radical fragment can only hang off a non-radical parent by a
    /// radical neutral loss, which `SpectrumAnnotator::score_loss` penalises the
    /// way SIRIUS's free-radical scorer does. This term only settles a mass tie
    /// between the even-electron and radical readings of the same peak in the
    /// annotator's candidate list, without pushing a genuine radical fragment out
    /// of the tree.
    pub fn rank_fragment_candidates(
        &self,
        candidates: &[Formula],
        target_mass: f64,
    ) -> Vec<ScoredFormula> {
        let mut scored = self.rank(candidates, target_mass, None);
        for s in &mut scored {
            // `rank` charged the whole-molecule prior; swap in the fragment prior.
            let delta = self.score_prior_fragment(&s.formula) - self.score_prior(&s.formula);
            let radical = if s.formula.is_radical() { RADICAL_FRAGMENT_PENALTY } else { 0.0 };
            s.prior_score += delta + radical;
            s.total_score += delta + radical;
        }
        scored.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        scored
    }

    fn rank(
        &self,
        candidates: &[Formula],
        target_mass: f64,
        envelope: Option<(&[Peak], MassTolerance)>,
    ) -> Vec<ScoredFormula> {
        let mut scored: Vec<ScoredFormula> = candidates
            .iter()
            .map(|&f| {
                let mass_score = self.score_mass_accuracy(f.exact_mass(), target_mass);
                let isotope_score = match envelope {
                    Some((peaks, tolerance)) => {
                        self.score_isotope_pattern(&f, peaks, target_mass, tolerance)
                    }
                    None => 0.0,
                };
                let prior_score = self.score_prior(&f);
                ScoredFormula::new(f, target_mass, mass_score, isotope_score, prior_score)
            })
            .collect();

        // Sort descending by total score
        scored.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn isotope_distribution_masses_are_physical_for_multi_element_formulas() {
        // Regression: the polynomial expansion used to add `iso.mass` to a
        // running (mass*prob) accumulator, so every isotope peak past the first
        // element came out at a nonsense m/z (~150 for a 377 Da molecule) and the
        // isotope scorer marked every real peak "missing".
        let f = Formula::from_str("C13H18Br2N2O").unwrap();
        let dist = calculate_isotope_distribution(&f, 4);
        let mono = f.exact_mass();

        // M+0 sits at the monoisotopic mass; each nominal step is ~1 Da heavier
        // and strictly increasing.
        assert!((dist[0].mz - mono).abs() < 0.05, "M+0 at {} want {mono}", dist[0].mz);
        for w in dist.windows(2) {
            let step = w[1].mz - w[0].mz;
            assert!((0.9..1.1).contains(&step), "nominal step {step} out of range");
        }
        // Di-bromine signature: M+2 is the tallest peak, M+4 comparable to M+0.
        let ab: Vec<f64> = dist.iter().map(|p| p.abundance).collect();
        assert!(ab[2] > ab[0] && ab[2] > ab[4], "M+2 should dominate: {ab:?}");
        assert!(ab[4] > 0.5 * ab[0], "M+4 should be substantial for Br2: {ab:?}");
        assert!((ab.iter().sum::<f64>() - 1.0).abs() < 1e-6, "abundances normalized");
    }

    #[test]
    fn isotope_score_rewards_the_correct_halogen_count() {
        // A di-bromo envelope (M : M+2 : M+4 ~= 1 : 2 : 1) should score the real
        // Br2 formula well above a same-nominal-mass formula with no M+2 step.
        let scorer = FormulaScorer::new(2.0, 0.05);
        let adduct = super::super::adduct::Adduct::protonated();
        let tol = MassTolerance::Ppm(10.0);
        let truth = Formula::from_str("C13H18Br2N2O").unwrap();
        let prec_mz = adduct.neutral_mass_to_ion_mz(truth.exact_mass());

        // Synthetic MS1 cluster from the true theoretical pattern.
        let ms1: Vec<Peak> = calculate_isotope_distribution(&truth, 4)
            .iter()
            .map(|p| Peak {
                mz: adduct.neutral_mass_to_ion_mz(p.mz),
                intensity: (p.abundance * 1000.0) as f32,
            })
            .collect();

        let s_truth = scorer.score_isotope_pattern_with_adduct(&truth, &ms1, prec_mz, &adduct, tol);
        // Same precursor m/z, a formula one Br lighter + two Cl (near-isobaric):
        // it has to explain the huge observed M+2 with a much weaker Cl step.
        let decoy = Formula::from_str("C13H18BrCl2N2O").unwrap();
        let s_decoy = scorer.score_isotope_pattern_with_adduct(&decoy, &ms1, prec_mz, &adduct, tol);
        assert!(s_truth > 0.0, "truth should score positive: {s_truth}");
        assert!(s_truth > s_decoy + 3.0, "truth {s_truth} vs decoy {s_decoy}");
    }

    #[test]
    fn contaminant_on_minor_satellite_does_not_destroy_halogen_score() {
        // Challenge-113 regression: a C6H5Cl2N compound with genuine M0, M+1, M+2, M+4
        // peaks has a co-eluting chemical contaminant at M+3 (138% of M0).
        // Bounded Huber loss and prefix optimal evaluation must preserve the true
        // formula's high score and outrank a decoy.
        let scorer = FormulaScorer::default();
        let adduct = super::super::adduct::Adduct::protonated();
        let tol = MassTolerance::Ppm(20.0);
        let truth = Formula::from_str("C6H5Cl2N").unwrap();
        let prec_mz = adduct.neutral_mass_to_ion_mz(truth.exact_mass());

        let ms1 = vec![
            Peak { mz: 161.986973, intensity: 8871631.0 },   // M0 (err -1.3 ppm)
            Peak { mz: 162.989110, intensity: 571234.9 },    // M+1
            Peak { mz: 163.983980, intensity: 5486951.0 },   // M+2 (Cl2 doublet)
            Peak { mz: 164.984310, intensity: 12304382.0 },  // M+3 contaminant (138% of M0!)
            Peak { mz: 165.981090, intensity: 924600.1 },    // M+4
        ];

        let s_truth = scorer.score_isotope_pattern_with_adduct(&truth, &ms1, prec_mz, &adduct, tol);
        // Decoy C3H5Cl3N is 1 Da lighter, fails M0 match completely.
        let decoy = Formula::from_str("C3H5Cl3N").unwrap();
        let s_decoy = scorer.score_isotope_pattern_with_adduct(&decoy, &ms1, prec_mz, &adduct, tol);

        assert!(s_truth > 0.0, "truth must score positive despite M+3 contaminant: {s_truth}");
        assert!(s_truth > s_decoy + 5.0, "truth ({s_truth}) must decisively beat decoy ({s_decoy})");
    }

    #[test]
    fn halogen_fine_structure_tolerance_broadening() {
        // A chlorinated compound has M+2 splitting between 37Cl and 13C2 (~9.7 mDa).
        // At high resolving power, centroiding places the 37Cl peak ~9 mDa away
        // from the unresolved isobaric centroid. Tight 10 ppm tolerance at 160 Da is 1.6 mDa.
        // The fine-structure broadening must ensure the genuine 37Cl peak matches and scores well.
        let scorer = FormulaScorer::default();
        let adduct = super::super::adduct::Adduct::protonated();
        let tight_tol = MassTolerance::Ppm(10.0);
        let truth = Formula::from_str("C6H5Cl2N").unwrap();
        let prec_mz = adduct.neutral_mass_to_ion_mz(truth.exact_mass());

        let dist = calculate_isotope_distribution(&truth, 4);
        let m0_mz = adduct.neutral_mass_to_ion_mz(dist[0].mz);
        let m2_theor_mz = adduct.neutral_mass_to_ion_mz(dist[2].mz);

        // Resolved 37Cl peak is shifted by 8 mDa from the isobaric centroid
        let ms1 = vec![
            Peak { mz: m0_mz, intensity: 10000.0 },
            Peak { mz: m2_theor_mz - 0.008, intensity: 6400.0 },
        ];

        let s_truth = scorer.score_isotope_pattern_with_adduct(&truth, &ms1, prec_mz, &adduct, tight_tol);
        assert!(s_truth > 0.0, "resolved 37Cl peak must be matched and score positive: {s_truth}");
    }

    #[test]
    fn unmatched_precursor_when_observed_is_penalized() {
        let scorer = FormulaScorer::default();
        let adduct = super::super::adduct::Adduct::protonated();
        let tol = MassTolerance::Ppm(10.0);
        let truth = Formula::from_str("C6H12O6").unwrap();
        let prec_mz = adduct.neutral_mass_to_ion_mz(truth.exact_mass());

        let ms1 = vec![
            Peak { mz: prec_mz, intensity: 10000.0 },
        ];

        // Candidate with mismatched formula (100 ppm away)
        let mismatch = Formula::from_str("C6H12O5S").unwrap();
        let s_mismatch = scorer.score_isotope_pattern_with_adduct(&mismatch, &ms1, prec_mz, &adduct, tol);
        assert!(s_mismatch <= -5.0, "mismatched candidate when precursor is present should be penalized: {s_mismatch}");

        // Envelope with no precursor peak at all (uninformative)
        let ms1_empty_prec = vec![
            Peak { mz: prec_mz + 50.0, intensity: 10000.0 },
        ];
        let s_uninformative = scorer.score_isotope_pattern_with_adduct(&truth, &ms1_empty_prec, prec_mz, &adduct, tol);
        assert_eq!(s_uninformative, 0.0, "missing precursor evidence should be neutral (0.0)");
    }
}
