//! Adduct definitions, ion modes, and exact mass conversions for mass spectrometry.
//!
//! Sourced from NIST / CODATA fundamental constants via `chemistry_consts`.
//!
//! # References
//! - **CODATA 2018 Recommended Values**:
//!   Tiesinga, E., Mohr, P.J., Newell, D.B., and Taylor, B.N. (2021). *CODATA recommended values of the
//!   fundamental physical constants: 2018*. Rev. Mod. Phys. 93, 025010.
//! - **NIST Physical Reference Data**: <https://physics.nist.gov/cuu/Constants/index.html>

use super::element::Element;
use super::formula::Formula;
use chemistry_consts::{ELECTRON_MASS_AMU, MONOISOTOPIC_MASSES, PROTON_MASS_AMU};
use std::str::FromStr;

/// Electron mass in unified atomic mass units (Da/amu).
/// CODATA 2018: 5.48579909065e-4 Da.
pub const ELECTRON_MASS: f64 = ELECTRON_MASS_AMU;

/// Proton mass in unified atomic mass units (Da/amu).
/// CODATA 2018: 1.007276466621 Da.
pub const PROTON_MASS: f64 = PROTON_MASS_AMU;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IonMode {
    Positive,
    Negative,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Adduct {
    pub name: String,
    pub charge: i8,
    pub mass_shift: f64,
    pub is_radical: bool,
    pub formula_delta: Option<Formula>,
    pub formula_delta_negative: bool, // true if formula is subtracted (e.g. [M-H]-)
}

impl Adduct {
    pub fn protonated() -> Self {
        Self::protonated_with_charge(1)
    }

    /// Multiply protonated positive ion mode adduct [M+zH]z+
    pub fn protonated_with_charge(z: i8) -> Self {
        let charge = z.abs().max(1);
        let name = if charge == 1 {
            "[M+H]+".to_string()
        } else {
            format!("[M+{}H]{}+", charge, charge)
        };
        Self {
            name,
            charge,
            mass_shift: charge as f64 * PROTON_MASS,
            is_radical: false,
            formula_delta: Some(Formula::chnops(0, charge as usize, 0, 0, 0, 0)),
            formula_delta_negative: false,
        }
    }

    pub fn deprotonated() -> Self {
        Self::deprotonated_with_charge(1)
    }

    /// Multiply deprotonated negative ion mode adduct [M-zH]z-
    pub fn deprotonated_with_charge(z: i8) -> Self {
        let charge = z.abs().max(1);
        let name = if charge == 1 {
            "[M-H]-".to_string()
        } else {
            format!("[M-{}H]{}-", charge, charge)
        };
        Self {
            name,
            charge: -charge,
            mass_shift: -(charge as f64 * PROTON_MASS),
            is_radical: false,
            formula_delta: Some(Formula::chnops(0, charge as usize, 0, 0, 0, 0)),
            formula_delta_negative: true,
        }
    }

    pub fn positive_radical() -> Self {
        Self {
            name: "[M]+.".to_string(),
            charge: 1,
            mass_shift: -ELECTRON_MASS,
            is_radical: true,
            formula_delta: None,
            formula_delta_negative: false,
        }
    }

    pub fn negative_radical() -> Self {
        Self {
            name: "[M]-.".to_string(),
            charge: -1,
            mass_shift: ELECTRON_MASS,
            is_radical: true,
            formula_delta: None,
            formula_delta_negative: false,
        }
    }

    pub fn sodium() -> Self {
        // [M+Na]+ = M + 23Na (22.9897692820 Da) - e-
        Self {
            name: "[M+Na]+".to_string(),
            charge: 1,
            mass_shift: MONOISOTOPIC_MASSES[11] - ELECTRON_MASS,
            is_radical: false,
            formula_delta: None,
            formula_delta_negative: false,
        }
    }

    pub fn potassium() -> Self {
        // [M+K]+ = M + 39K (38.9637064864 Da) - e-
        Self {
            name: "[M+K]+".to_string(),
            charge: 1,
            mass_shift: MONOISOTOPIC_MASSES[19] - ELECTRON_MASS,
            is_radical: false,
            formula_delta: None,
            formula_delta_negative: false,
        }
    }

    pub fn ammonium() -> Self {
        // [M+NH4]+ = M + 14N + 4*1H - e-
        let nh4_mass = MONOISOTOPIC_MASSES[7] + 4.0 * MONOISOTOPIC_MASSES[1];
        Self {
            name: "[M+NH4]+".to_string(),
            charge: 1,
            mass_shift: nh4_mass - ELECTRON_MASS,
            is_radical: false,
            formula_delta: Some(Formula::chnops(0, 4, 1, 0, 0, 0)),
            formula_delta_negative: false,
        }
    }

    pub fn chloride() -> Self {
        // [M+Cl]- = M + 35Cl (34.9688526820 Da) + e-
        Self {
            name: "[M+Cl]-".to_string(),
            charge: -1,
            mass_shift: MONOISOTOPIC_MASSES[17] + ELECTRON_MASS,
            is_radical: false,
            formula_delta: {
                let mut f = Formula::empty();
                f.set(Element::Cl, 1);
                Some(f)
            },
            formula_delta_negative: false,
        }
    }

    pub fn formate() -> Self {
        // [M+HCOO]- = M + 12C + 1H + 2*16O + e-
        let hcoo_mass = MONOISOTOPIC_MASSES[6] + MONOISOTOPIC_MASSES[1] + 2.0 * MONOISOTOPIC_MASSES[8];
        Self {
            name: "[M+HCOO]-".to_string(),
            charge: -1,
            mass_shift: hcoo_mass + ELECTRON_MASS,
            is_radical: false,
            formula_delta: Some(Formula::chnops(1, 1, 0, 2, 0, 0)),
            formula_delta_negative: false,
        }
    }

    pub fn acetate() -> Self {
        // [M+CH3COO]- = M + 2*12C + 3*1H + 2*16O + e-
        let ch3coo_mass = 2.0 * MONOISOTOPIC_MASSES[6] + 3.0 * MONOISOTOPIC_MASSES[1] + 2.0 * MONOISOTOPIC_MASSES[8];
        Self {
            name: "[M+CH3COO]-".to_string(),
            charge: -1,
            mass_shift: ch3coo_mass + ELECTRON_MASS,
            is_radical: false,
            formula_delta: Some(Formula::chnops(2, 3, 0, 2, 0, 0)),
            formula_delta_negative: false,
        }
    }

    /// Water-loss protonated positive ion mode adduct [M+H-H2O]+
    pub fn water_loss_protonated() -> Self {
        let h2o_mass = 2.0 * MONOISOTOPIC_MASSES[1] + MONOISOTOPIC_MASSES[8];
        Self {
            name: "[M+H-H2O]+".to_string(),
            charge: 1,
            mass_shift: PROTON_MASS - h2o_mass,
            is_radical: false,
            formula_delta: None,
            formula_delta_negative: false,
        }
    }

    /// Returns the standard list of common positive ESI adducts
    pub fn common_positive() -> Vec<Self> {
        vec![
            Self::protonated(),
            Self::sodium(),
            Self::potassium(),
            Self::ammonium(),
            Self::water_loss_protonated(),
        ]
    }

    /// Returns the standard list of common negative ESI adducts
    pub fn common_negative() -> Vec<Self> {
        vec![
            Self::deprotonated(),
            Self::chloride(),
            Self::formate(),
            Self::acetate(),
        ]
    }

    /// Converts an observed precursor m/z to the corresponding neutral molecule monoisotopic mass
    #[inline]
    pub fn ion_mz_to_neutral_mass(&self, ion_mz: f64) -> f64 {
        let z = self.charge.abs() as f64;
        (ion_mz * z) - self.mass_shift
    }

    /// Converts a neutral molecule monoisotopic mass to the predicted ion m/z
    #[inline]
    pub fn neutral_mass_to_ion_mz(&self, neutral_mass: f64) -> f64 {
        let z = self.charge.abs() as f64;
        (neutral_mass + self.mass_shift) / z
    }

    /// Empirical prior log-odds score for standard ESI adduct formation.
    pub fn prior_score(&self) -> f64 {
        match self.name.as_str() {
            "[M+H]+" | "[M-H]-" => 0.0,
            "[M+Na]+" => -0.5,
            "[M+NH4]+" => -0.5,
            "[M+Cl]-" => -0.5,
            "[M+HCOO]-" => -0.5,
            "[M+H-H2O]+" => -0.8,
            "[M+CH3COO]-" => -0.8,
            "[M+K]+" => -1.2,
            "[M]+." | "[M]-." => -2.0,
            _ => -1.0,
        }
    }
}

impl FromStr for Adduct {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let clean = s.trim().replace(' ', "");
        match clean.as_str() {
            "[M+H]+" | "[M+H]" | "M+H" => Ok(Adduct::protonated()),
            "[M+2H]2+" | "[M+2H]++" | "M+2H" => Ok(Adduct::protonated_with_charge(2)),
            "[M+3H]3+" | "[M+3H]+++" | "M+3H" => Ok(Adduct::protonated_with_charge(3)),
            "[M+4H]4+" | "[M+4H]++++" | "M+4H" => Ok(Adduct::protonated_with_charge(4)),
            "[M-H]-" | "[M-H]" | "M-H" => Ok(Adduct::deprotonated()),
            "[M-2H]2-" | "[M-2H]--" | "M-2H" => Ok(Adduct::deprotonated_with_charge(2)),
            "[M-3H]3-" | "[M-3H]---" | "M-3H" => Ok(Adduct::deprotonated_with_charge(3)),
            "[M+Na]+" | "[M+Na]" | "M+Na" => Ok(Adduct::sodium()),
            "[M+K]+" | "[M+K]" | "M+K" => Ok(Adduct::potassium()),
            "[M+NH4]+" | "[M+NH4]" | "M+NH4" => Ok(Adduct::ammonium()),
            "[M+H-H2O]+" | "[M-H2O+H]+" | "[M+H-H2O]" | "[M-H2O+H]" => Ok(Adduct::water_loss_protonated()),
            "[M+Cl]-" | "[M+Cl]" | "M+Cl" => Ok(Adduct::chloride()),
            "[M+HCOO]-" | "[M+FA-H]-" | "[M+HCO2]-" => Ok(Adduct::formate()),
            "[M+CH3COO]-" | "[M+Ac-H]-" => Ok(Adduct::acetate()),
            "[M]+." | "[M]+" | "M+" => Ok(Adduct::positive_radical()),
            "[M]-." | "[M]-" | "M-" => Ok(Adduct::negative_radical()),
            other => Err(format!("Unrecognized adduct: '{}'", other)),
        }
    }
}
