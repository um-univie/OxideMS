use super::adduct::Adduct;
use super::element::NUM_ELEMENTS;
use super::formula::Formula;
use super::generator::FormulaGenerator;
use super::rules::ChemicalRules;
use super::scoring::{FormulaScorer, ScoredFormula};
use crate::spectrum::Peak;
use crate::workflow::MassTolerance;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// The three fragment decomposers, built once. Their Round-Robin DP tables cost
/// real time to construct (Dijkstra over a table keyed by the scaled mass of
/// hydrogen), and `FragmentationTreeBuilder::new` — hence `SpectrumAnnotator::new`
/// — is called once per spectrum in batch workflows, so building them per
/// annotator regressed throughput badly. Shared behind `Arc`, cloning an
/// annotator is now a refcount bump.
static GEN_CHNOPS: LazyLock<Arc<FormulaGenerator<6>>> =
    LazyLock::new(|| Arc::new(FormulaGenerator::chnops()));
static GEN_FCLBR: LazyLock<Arc<FormulaGenerator<9>>> =
    LazyLock::new(|| Arc::new(FormulaGenerator::chnops_fclbr()));
static GEN_EXTENDED: LazyLock<Arc<FormulaGenerator<{ NUM_ELEMENTS }>>> =
    LazyLock::new(|| Arc::new(FormulaGenerator::extended()));

/// Does this formula contain only the six elements the fast CHNOPS decomposer
/// knows about? Fragments are constrained to be subformulas of the precursor, so
/// the precursor's element set is the fragment alphabet — mirroring how SIRIUS
/// decomposes fragments over exactly the elements present in the precursor
/// formula rather than a fixed alphabet.
fn is_chnops_only(f: &Formula) -> bool {
    f.f() == 0 && f.cl() == 0 && f.br() == 0 && f.i() == 0 && f.si() == 0 && f.b() == 0
}

/// Precursor carries I, Si or B — the rare tail that needs the full-alphabet
/// decomposer. F/Cl/Br are handled by a cheaper nine-element generator.
fn needs_full_alphabet(f: &Formula) -> bool {
    use super::element::Element;
    f.get(Element::I) > 0 || f.get(Element::Si) > 0 || f.get(Element::B) > 0
}

/// Candidate sub-formula annotation for a specific fragment peak in an MS/MS spectrum
#[derive(Debug, Clone)]
pub struct AnnotatedPeak {
    pub peak_index: usize,
    pub mz: f64,
    pub intensity: f32,
    pub relative_intensity: f32,
    pub candidates: Vec<ScoredFormula>,
}

/// Annotates an MS/MS spectrum with sub-formula candidates for every peak.
#[derive(Clone)]
pub struct SpectrumAnnotator {
    /// Fast six-element decomposer for the common case.
    generator: Arc<FormulaGenerator<6>>,
    /// Nine-element decomposer (CHNOPS + F/Cl/Br) for halogenated precursors.
    generator_fclbr: Arc<FormulaGenerator<9>>,
    /// Full-alphabet decomposer, for the rare precursor carrying I, Si or B.
    /// Shared behind an `Arc` so cloning an annotator (per spectrum in the
    /// batched tree search) does not copy its multi-megabyte DP table.
    generator_extended: Arc<FormulaGenerator<{ NUM_ELEMENTS }>>,
    scorer: FormulaScorer,
    rules: ChemicalRules,
    adduct: Adduct,
    common_losses: HashMap<Formula, f64>,
    /// Use the data-learned, mode-specific loss model instead of the hand list.
    use_learned_losses: bool,
}

impl Default for SpectrumAnnotator {
    fn default() -> Self {
        let mut annotator = Self {
            generator: GEN_CHNOPS.clone(),
            generator_fclbr: GEN_FCLBR.clone(),
            generator_extended: GEN_EXTENDED.clone(),
            scorer: FormulaScorer::default(),
            rules: ChemicalRules::loose(),
            adduct: Adduct::protonated(),
            common_losses: HashMap::new(),
            use_learned_losses: true,
        };
        annotator.init_common_losses();
        annotator
    }
}

impl SpectrumAnnotator {
    pub fn new(tolerance_ppm: f64) -> Self {
        let mut annotator = Self {
            generator: GEN_CHNOPS.clone(),
            generator_fclbr: GEN_FCLBR.clone(),
            generator_extended: GEN_EXTENDED.clone(),
            scorer: FormulaScorer::new(tolerance_ppm, 0.05),
            rules: ChemicalRules::loose(),
            adduct: Adduct::protonated(),
            common_losses: HashMap::new(),
            use_learned_losses: true,
        };
        annotator.init_common_losses();
        annotator
    }

    /// Toggle the data-learned loss model (on by default). Off falls back to the
    /// 35-entry hand-curated common-loss list.
    pub fn with_learned_losses(mut self, enabled: bool) -> Self {
        self.use_learned_losses = enabled;
        self
    }

    pub fn with_adduct(mut self, adduct: Adduct) -> Self {
        self.adduct = adduct;
        self
    }

    fn init_common_losses(&mut self) {
        let common = [
            ("H2O", 3.0),         // Water: 18.010565
            ("CO", 2.5),          // Carbon monoxide: 27.994915
            ("CO2", 3.0),         // Carbon dioxide: 43.989829
            ("NH3", 2.5),         // Ammonia: 17.026549
            ("H2S", 2.2),         // Hydrogen sulfide: 33.987721
            ("HCN", 2.2),         // Hydrogen cyanide: 27.010899
            ("C2H4", 2.0),        // Ethylene: 28.031300
            ("CH2O", 2.2),        // Formaldehyde: 30.010565
            ("CH4O", 2.2),        // Methanol: 32.026215
            ("CH4", 1.8),         // Methane: 16.031300
            ("C2H2", 1.8),        // Acetylene: 26.015650
            ("H3PO4", 3.0),       // Phosphoric acid: 97.976896
            ("HPO3", 2.2),        // Metaphosphoric acid: 79.966331
            ("C2H4O2", 2.5),      // Acetic acid: 60.021129
            ("C3H6", 2.0),        // Propene: 42.046950
            ("C4H8", 2.0),        // Isobutene: 56.062600
            ("C2H4O", 2.0),       // Acetaldehyde: 44.026215
            ("C3H6O", 2.0),       // Acetone: 58.041865
            ("CH2O2", 2.5),       // Formic acid: 46.005479
            ("SO2", 2.5),         // Sulfur dioxide: 63.961901
            ("SO3", 2.2),         // Sulfur trioxide: 79.956815
            ("N2", 2.2),          // Nitrogen: 28.006148
            ("C6H6", 1.8),        // Benzene: 78.046950
            ("C6H6O", 2.2),       // Phenol: 94.041865
            ("C7H6O2", 2.2),      // Benzoic acid: 122.036779
            ("C6H10O5", 3.0),     // Hexose loss (glucose/galactose): 162.052823
            ("C5H8O4", 2.5),      // Pentose loss (ribose/xylose): 132.042259
            ("C2H7N", 2.0),       // Dimethylamine: 45.057849
            ("CH5N", 2.0),        // Methylamine: 31.042199
            ("HNCO", 2.0),        // Isocyanic acid: 43.005814
            ("C2H2O", 2.0),       // Ketene: 42.010565
            ("H2", 1.5),          // Hydrogen: 2.015650
            ("CH2", 1.2),         // Carbene / methylene chain: 14.015650
            ("C3H4", 1.2),        // Propyne/allene: 40.031300
            ("C4H6", 1.2),        // Butadiene: 54.046950
            ("C5H8", 1.2),        // Isoprene: 68.062600
        ];

        for (formula_str, bonus) in common {
            if let Ok(f) = formula_str.parse::<Formula>() {
                self.common_losses.insert(f, bonus);
            }
        }
    }

    /// Whether this loss is a recognised common loss (curated list, or — with the
    /// learned model — a listed loss for the current ionization mode).
    pub fn is_common_loss(&self, loss: &Formula) -> bool {
        if self.use_learned_losses {
            return super::loss_model::LossModel::global().is_listed(loss, self.adduct.charge >= 0);
        }
        self.common_losses.contains_key(loss)
    }

    /// Log-odds score for a neutral loss. With the learned model this is the
    /// mode-specific observed frequency of that exact loss formula in GNPS data;
    /// otherwise the hand-curated bonus (or a flat radical penalty).
    pub fn score_loss(&self, loss: &Formula) -> f64 {
        if self.use_learned_losses {
            return super::loss_model::LossModel::global().score(loss, self.adduct.charge >= 0);
        }
        if let Some(&bonus) = self.common_losses.get(loss) {
            return bonus;
        }
        if loss.is_radical() {
            -3.5
        } else {
            0.0
        }
    }

    /// Annotates every fragment peak in an MS/MS spectrum with candidate sub-formulas
    /// derived from a candidate precursor formula F_prec.
    ///
    /// The fragment alphabet is the precursor's element set: a plain CHNOPS
    /// precursor uses the fast six-element decomposer, a halogenated one the full
    /// alphabet. Absent elements are pruned by the subformula bound regardless,
    /// so the only effect of picking the wider generator is that Cl/Br/F/I/Si/B
    /// fragments become reachable at all.
    pub fn annotate_peaks(
        &self,
        peaks: &[Peak],
        precursor_formula: &Formula,
        tolerance: MassTolerance,
    ) -> Vec<AnnotatedPeak> {
        if is_chnops_only(precursor_formula) {
            self.decompose_with(self.generator.as_ref(), peaks, precursor_formula, tolerance)
        } else if needs_full_alphabet(precursor_formula) {
            self.decompose_with(
                self.generator_extended.as_ref(),
                peaks,
                precursor_formula,
                tolerance,
            )
        } else {
            self.decompose_with(self.generator_fclbr.as_ref(), peaks, precursor_formula, tolerance)
        }
    }

    fn decompose_with<const N: usize>(
        &self,
        generator: &FormulaGenerator<N>,
        peaks: &[Peak],
        precursor_formula: &Formula,
        tolerance: MassTolerance,
    ) -> Vec<AnnotatedPeak> {
        let max_intensity = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);

        let mut annotated = Vec::with_capacity(peaks.len());

        for (idx, peak) in peaks.iter().enumerate() {
            let mut scored_candidates = Vec::new();

            let max_z = self.adduct.charge.abs().max(1);

            for z_f in 1..=max_z {
                // 1. Even-electron fragment ion candidate search with charge z_f (protonated/deprotonated)
                let frag_adduct = if self.adduct.charge < 0 {
                    Adduct::deprotonated_with_charge(z_f)
                } else {
                    Adduct::protonated_with_charge(z_f)
                };
                let neutral_even = frag_adduct.ion_mz_to_neutral_mass(peak.mz);
                if neutral_even > 0.0 {
                    let cands_even = generator.generate_subformulas(
                        neutral_even,
                        precursor_formula,
                        tolerance,
                        &self.rules,
                    );
                    let scored_even = self.scorer.rank_fragment_candidates(&cands_even, neutral_even);
                    scored_candidates.extend(scored_even);
                }

                // 2. Parent adduct retention candidate search (e.g. sodiated, ammoniated, or formate fragments)
                if self.adduct != frag_adduct {
                    let neutral_parent_adduct = self.adduct.ion_mz_to_neutral_mass(peak.mz);
                    if neutral_parent_adduct > 0.0 {
                        let cands_parent = generator.generate_subformulas(
                            neutral_parent_adduct,
                            precursor_formula,
                            tolerance,
                            &self.rules,
                        );
                        let scored_parent = self.scorer.rank_fragment_candidates(&cands_parent, neutral_parent_adduct);
                        scored_candidates.extend(scored_parent);
                    }
                }

                // 3. Bare-ion (odd-electron) candidate search: the peak's m/z *is*
                //    the fragment's neutral mass, no protonation. Only meaningful
                //    for radical precursor adducts ([M]+•, EAD, ...). For an
                //    even-electron precursor this branch was the source of the
                //    systematic disagreement with SIRIUS: it re-explains every
                //    protonated fragment with a formula one hydrogen heavier
                //    (C10H8N -> C10H9N) that fits its own target perfectly and
                //    escapes the radical penalty because the extra H makes the
                //    ion composition itself even-electron. SIRIUS always reports
                //    the neutral fragment (ion minus the adduct) and penalises it
                //    when that neutral is a radical — which is exactly what
                //    branch 1 plus `rank_fragment_candidates` now does.
                if self.adduct.is_radical {
                    let neutral_rad = peak.mz * (z_f as f64);
                    if neutral_rad > 0.0 {
                        let cands_radical = generator.generate_subformulas(
                            neutral_rad,
                            precursor_formula,
                            tolerance,
                            &self.rules,
                        );
                        let scored_radical =
                            self.scorer.rank_fragment_candidates(&cands_radical, neutral_rad);
                        scored_candidates.extend(scored_radical);
                    }
                }
            }

            // Deduplicate by formula (keeping higher score)
            scored_candidates.sort_by(|a, b| {
                if a.formula == b.formula {
                    b.total_score.total_cmp(&a.total_score)
                } else {
                    a.formula.cmp(&b.formula)
                }
            });

            let mut deduplicated = Vec::new();
            for cand in scored_candidates {
                if deduplicated.last().map(|prev: &ScoredFormula| prev.formula) != Some(cand.formula) {
                    deduplicated.push(cand);
                }
            }

            // Re-sort by total score descending
            deduplicated.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));

            annotated.push(AnnotatedPeak {
                peak_index: idx,
                mz: peak.mz,
                intensity: peak.intensity,
                relative_intensity: peak.intensity / max_intensity,
                candidates: deduplicated,
            });
        }

        annotated
    }
}
