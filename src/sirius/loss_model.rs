//! Data-derived neutral-loss scores, learned from ~118k labelled GNPS spectra
//! (~59k positive, ~59k negative), split by ionization mode. Positive- and
//! negative-mode ESI fragment differently: protonated species shed neutral
//! amines and water, deprotonated species decarboxylate (CO2 loss is ~14x more
//! common in negative mode) and lose HF, sugars and glucuronic acid.
//!
//! For each spectrum the fragment peaks were annotated under the true precursor
//! formula and every precursor->fragment neutral loss tallied, weighted by
//! fragment intensity. Score = clamp(0.9 * ln(share / ref), -1.0, 3.2) with ref
//! fixed so water scores 3.0. Only losses seen in >=25 spectra with a positive
//! score are listed; anything else falls back to a mild penalty in
//! `LossModel::score`.
//!
//! Regenerate: `cargo run --release --example build_loss_table -- <ALL_GNPS.mgf>`
//! then `python3 scratch/gen_loss_model.py > src/sirius/loss_model.rs`.


/// Neutral-loss bonuses for positive-mode [M+H] fragmentation.
pub(crate) static LOSS_SCORES_POSITIVE: &[(&str, f64)] = &[
    ("H2O", 3.00),
    ("H3N", 2.54),
    ("CH2O2", 2.54),
    ("C4H8", 1.78),
    ("C2H4O2", 1.65),
    ("C2H7N", 1.52),
    ("C5H8O4", 1.50),
    ("C4H9NO", 1.46),
    ("CH4", 1.41),
    ("C6H10O5", 1.32),
    ("C9H11NO2", 1.31),
    ("CO2", 1.30),
    ("C7H13NO3", 1.29),
    ("C6H13NO2", 1.27),
    ("C10H11NO3", 1.26),
    ("CO", 1.25),
    ("C3H7N", 1.25),
    ("CH4O", 1.24),
    ("C2H5NO", 1.23),
    ("C3H6", 1.16),
    ("C4H11N", 1.14),
    ("C4H9N", 1.12),
    ("C5H11N", 1.11),
    ("C2H6O", 1.10),
    ("C12H12N2O3", 1.09),
    ("C2H5N", 1.08),
    ("C2H2O", 1.07),
    ("C2H4O", 1.06),
    ("C3H6O2", 1.03),
    ("C2H6", 1.02),
    ("C3H6O", 1.02),
    ("C9H11NO", 0.97),
    ("C2H4", 0.96),
    ("CH3NO", 0.94),
    ("C5H11NO2", 0.92),
    ("C8H7N", 0.88),
    ("C3H9N", 0.87),
    ("C5H12N2", 0.81),
    ("C24H38O3", 0.80),
    ("C5H8O3", 0.76),
    ("C5H8O2", 0.75),
    ("C24H38O4", 0.75),
    ("C4H10O", 0.75),
    ("C3H4O3", 0.74),
    ("C11H12N2O2", 0.73),
    ("C3H7NO2", 0.72),
    ("C6H11NO", 0.72),
    ("C8H9NO2", 0.72),
    ("C12H20O9", 0.71),
    ("C3H8O", 0.66),
    ("C4H9NO2", 0.65),
    ("C10H13NO2", 0.63),
    ("C4H6O2", 0.63),
    ("C7H15NO2", 0.62),
    ("C5H9NO2", 0.62),
    ("C2H5NO2", 0.62),
    ("C4H7NO2", 0.62),
    ("C4H8O4", 0.60),
    ("C5H11NO", 0.60),
    ("C2H3NO2", 0.59),
    ("C5H10O2", 0.59),
    ("C3H5NO2", 0.59),
    ("C4H8O2", 0.58),
    ("C2H6O3", 0.58),
    ("CH3NO2", 0.56),
    ("C6H13NO", 0.55),
    ("C2H2O3", 0.52),
    ("C5H8O", 0.52),
    ("C6H7N", 0.51),
    ("C5H9NO", 0.51),
    ("C5H11NO2S", 0.51),
    ("C3H7NO", 0.51),
    ("C2H3NO", 0.50),
    ("C3H5NO", 0.50),
    ("C2H7NO", 0.50),
    ("C4H11NO", 0.48),
    ("C9H11N", 0.47),
    ("C6H10", 0.47),
    ("C5H9N", 0.47),
    ("C8H9NO", 0.46),
    ("CHN", 0.46),
    ("CHNO", 0.46),
    ("C9H9NO", 0.44),
    ("C6H11NO2", 0.44),
    ("C12H14N2O2", 0.44),
    ("C6H11NO3", 0.44),
    ("C6H11NO3S", 0.41),
    ("C3H8O2", 0.41),
    ("C9H10O2", 0.40),
    ("C5H9NO3", 0.40),
    ("C4H11NO2", 0.40),
    ("C6H14N2", 0.40),
    ("C6H10O", 0.40),
    ("C6H10O4", 0.38),
    ("C5H8", 0.37),
    ("C8H8", 0.37),
    ("C4H8O", 0.34),
    ("C4H7NO3", 0.34),
    ("C7H15NO", 0.34),
    ("C10H15NO", 0.34),
    ("C7H11NO", 0.33),
    ("C7H7NO2", 0.33),
    ("C6H15NO", 0.33),
    ("C3H4O2", 0.32),
    ("C4H7NO", 0.32),
    ("C7H9N", 0.31),
    ("C6H9NO3", 0.31),
    ("C3H9NO", 0.30),
    ("C2H6O2", 0.30),
    ("C13H14N2O3", 0.30),
    ("C4H8O3", 0.27),
    ("C11H13NO3", 0.27),
    ("C8H8O3", 0.26),
    ("C7H13NO", 0.26),
    ("C3H8O3", 0.24),
    ("C8H8O", 0.24),
    ("C8H11N", 0.24),
    ("C5H9NO4", 0.24),
    ("C10H12N2", 0.23),
    ("C5H10O5", 0.23),
    ("CH3N", 0.23),
    ("C8H15NO3", 0.20),
    ("C11H15NO", 0.20),
    ("C6H6O", 0.19),
    ("C3H6O3", 0.19),
    ("C6H8O6", 0.19),
    ("C3H9NO2", 0.17),
    ("C7H5NO", 0.17),
    ("C7H13N3O4", 0.16),
    ("C9H12O3", 0.16),
    ("C6H10O3", 0.15),
    ("C7H11NO2", 0.15),
    ("C9H13NO", 0.15),
    ("C12H17NO2", 0.15),
    ("C11H15NO2", 0.14),
    ("C7H13NO2", 0.14),
    ("C6H13N", 0.14),
    ("C5H13NO", 0.14),
    ("C6H12O2", 0.13),
    ("C5H10", 0.12),
    ("C4H6O3", 0.12),
    ("C8H15NO2", 0.12),
    ("C4H6O", 0.12),
    ("C12H15NO2", 0.12),
    ("C3H6N2", 0.12),
    ("C5H13NO2", 0.12),
    ("C8H8O2", 0.11),
    ("C6H6", 0.11),
    ("C7H9NO", 0.10),
];

/// Neutral-loss bonuses for negative-mode [M-H] fragmentation.
pub(crate) static LOSS_SCORES_NEGATIVE: &[(&str, f64)] = &[
    ("CO2", 3.20),
    ("H2O", 2.46),
    ("C6H10O5", 2.17),
    ("CHNO", 1.97),
    ("C2H2O", 1.88),
    ("C2H4O2", 1.78),
    ("HF", 1.70),
    ("CH4O", 1.69),
    ("C3H4O2", 1.55),
    ("C2H6", 1.49),
    ("CH2O2", 1.49),
    ("C4H8O4", 1.43),
    ("C2H2O2", 1.36),
    ("CH4", 1.33),
    ("C2H4O", 1.32),
    ("C6H8O6", 1.25),
    ("CO", 1.24),
    ("C4H6O2", 1.24),
    ("C9H6O3", 1.23),
    ("CH2O", 1.22),
    ("C13H12N2O3", 1.20),
    ("C5H8O4", 1.18),
    ("O2S", 1.16),
    ("C9H8O2", 1.10),
    ("C6H12O6", 1.07),
    ("C2H4O3", 1.05),
    ("C2H2OS", 1.05),
    ("C3H6O3", 1.04),
    ("C3H2O3", 1.01),
    ("C3H4O3", 0.98),
    ("C3H6O", 0.96),
    ("C11H11NO3", 0.93),
    ("CH2O2S", 0.92),
    ("C9H7N", 0.90),
    ("C12H20O9", 0.90),
    ("C24H38O3", 0.89),
    ("N2", 0.88),
    ("H3N", 0.81),
    ("C2H2O3", 0.80),
    ("C2H4", 0.80),
    ("HCl", 0.78),
    ("C24H38O4", 0.77),
    ("C12H20O10", 0.75),
    ("C7H10O7", 0.75),
    ("C3H6O2", 0.74),
    ("C6H10O4", 0.74),
    ("C10H11NO", 0.72),
    ("C2H3NO", 0.72),
    ("C4H6O3", 0.72),
    ("C3H4O4", 0.71),
    ("C4H4O2", 0.69),
    ("C9H8O3", 0.69),
    ("C9H9NO", 0.67),
    ("C7H11NO3", 0.67),
    ("C14H14N2O3", 0.67),
    ("C7H4O2", 0.66),
    ("C9H10O2", 0.66),
    ("C9H8O4", 0.65),
    ("C4H10O", 0.63),
    ("C5H8O2", 0.63),
    ("C8H6O3", 0.61),
    ("C7H11NO2", 0.60),
    ("C8H8O2", 0.60),
    ("O3S", 0.60),
    ("C5H7NO3", 0.59),
    ("C11H12N2O2", 0.54),
    ("H2S", 0.54),
    ("C7H8O", 0.53),
    ("C12H13NO3", 0.53),
    ("C11H18O9", 0.53),
    ("C4H5NO3", 0.52),
    ("C8H13NO3", 0.51),
    ("C9H9NO2", 0.50),
    ("C7H6O2", 0.49),
    ("C3H5NO2", 0.49),
    ("C6H10O2", 0.49),
    ("C6H6O3", 0.49),
    ("C7H4O4", 0.48),
    ("C7H6O", 0.47),
    ("C2HNO2", 0.47),
    ("C2H6O2", 0.47),
    ("C2H6O", 0.47),
    ("C6H9NO2", 0.45),
    ("CHN", 0.45),
    ("C13H15NO3", 0.44),
    ("C8H7NO", 0.43),
    ("C10H9NO3", 0.43),
    ("C15H20O9", 0.42),
    ("C7H5NO", 0.41),
    ("CH4O2", 0.39),
    ("C8H9NO", 0.39),
    ("C12H12N2O2", 0.38),
    ("C10H10O4", 0.38),
    ("C7H10O3", 0.38),
    ("C4H4O3", 0.38),
    ("C10H10O2", 0.38),
    ("C10H11NO2", 0.38),
    ("CH2N2", 0.37),
    ("C5H6O4", 0.36),
    ("CHFO2", 0.36),
    ("C9H10O4", 0.36),
    ("C9H10O5", 0.36),
    ("C5H8O5", 0.36),
    ("C8H8O", 0.34),
    ("C9H10O3", 0.34),
    ("C10H10O3", 0.33),
    ("C5H8O3", 0.32),
    ("C5H10O2", 0.31),
    ("C11H10N2O", 0.31),
    ("CH3NO2", 0.30),
    ("C4H6O4", 0.30),
    ("C13H14O9", 0.29),
    ("C15H20O8", 0.29),
    ("C10H12O3", 0.27),
    ("C2H2O5", 0.27),
    ("C7H12O6", 0.26),
    ("C6H11NO2", 0.25),
    ("C44H70O16", 0.25),
    ("C7H12O2", 0.25),
    ("CHClO2", 0.25),
    ("C3H3NO3", 0.25),
    ("C9H15NO3", 0.24),
    ("C10H9NO2", 0.24),
    ("C4H4O4", 0.24),
    ("C3H2O2", 0.24),
    ("C12H14N2O2", 0.23),
    ("C6H7NO3", 0.23),
    ("C4H10", 0.22),
    ("C4H8O2", 0.21),
    ("C6H9NO3", 0.21),
    ("C16H22O10", 0.20),
    ("C9H13NO2", 0.19),
    ("C6H6O2", 0.19),
    ("C2H2O4", 0.18),
    ("C5H9NO2", 0.17),
    ("C11H13NO", 0.17),
    ("C7H4O3", 0.17),
    ("C8H13NO2", 0.16),
    ("C9H11NO2", 0.16),
    ("C9H11NO", 0.15),
    ("C10H13N3O2", 0.14),
    ("C6H8O2", 0.14),
    ("C8H10O2", 0.14),
    ("C5H9NO", 0.14),
    ("C4H7NO2", 0.13),
    ("C9H12O3", 0.13),
    ("C8H6O4", 0.13),
    ("C10H12O2", 0.13),
    ("C16H20O10", 0.12),
    ("C7H6O3", 0.12),
    ("C6H12O2", 0.11),
    ("CH4O2S", 0.11),
    ("C7H14O5", 0.11),
    ("C3H5NO", 0.11),
    ("C8H6O", 0.10),
];

use super::formula::Formula;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::LazyLock;

/// Score a candidate neutral loss by how often that exact formula is observed as
/// a real fragmentation loss in labelled GNPS data, resolved per ionization
/// mode. Replaces the 35-entry hand-curated common-loss list and the blunt
/// large-loss size penalty with one data-derived, formula-specific term: a big
/// common loss (a hexose) keeps its bonus, a big rare loss (an arbitrary alkyl
/// chunk that only a wrong precursor formula needs) does not.
pub(crate) struct LossModel {
    pos: HashMap<Formula, f64>,
    neg: HashMap<Formula, f64>,
}

/// Loss formulas below this mass are always plausible one-step losses; unlisted
/// ones are not penalised. Above it, an unlisted loss gets `UNLISTED_LARGE`.
const SMALL_LOSS_DA: f64 = 40.0;
const UNLISTED_LARGE: f64 = -1.2;
const RADICAL_LOSS: f64 = -3.5;

impl LossModel {
    fn build(table: &[(&str, f64)]) -> HashMap<Formula, f64> {
        table
            .iter()
            .filter_map(|(f, s)| Formula::from_str(f).ok().map(|f| (f, *s)))
            .collect()
    }

    pub(crate) fn global() -> &'static LossModel {
        static MODEL: LazyLock<LossModel> = LazyLock::new(|| LossModel {
            pos: LossModel::build(LOSS_SCORES_POSITIVE),
            neg: LossModel::build(LOSS_SCORES_NEGATIVE),
        });
        &MODEL
    }

    pub(crate) fn score(&self, loss: &Formula, positive: bool) -> f64 {
        let table = if positive { &self.pos } else { &self.neg };
        if let Some(&s) = table.get(loss) {
            return s;
        }
        if loss.is_radical() {
            return RADICAL_LOSS;
        }
        if loss.exact_mass() <= SMALL_LOSS_DA {
            0.0
        } else {
            UNLISTED_LARGE
        }
    }

    pub(crate) fn is_listed(&self, loss: &Formula, positive: bool) -> bool {
        let table = if positive { &self.pos } else { &self.neg };
        table.contains_key(loss)
    }
}
