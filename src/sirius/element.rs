//! Chemical element definitions, exact monoisotopic masses, nominal valences,
//! and isotope abundance distributions sourced directly from `chemistry_consts` (crates.io).
//!
//! # References
//! - **NIST Physical Reference Data**: Coursey, J.S., Schwab, D.J., Tsai, J.J., and Dragoset, R.A. (2015),
//!   *Atomic Weights and Isotopic Compositions (version 4.1)*. National Institute of Standards and Technology,
//!   Gaithersburg, MD. <https://www.physics.nist.gov/cgi-bin/Compositions/stand_alone.pl>
//! - **IUPAC / CIAAW**: Meija, J. et al. (2016), *Atomic weights of the elements 2013 (IUPAC Technical Report)*.
//!   Pure Appl. Chem., 88(3), 265-291. DOI: 10.1515/pac-2015-0305
//! - **CODATA 2018 Recommended Values of the Fundamental Physical Constants**:
//!   Tiesinga, E., Mohr, P.J., Newell, D.B., and Taylor, B.N. (2021). Rev. Mod. Phys. 93, 025010.

use chemistry_consts::{ISOTOPES, MONOISOTOPIC_MASSES};

/// Standard elements used in organic, metabolite, and small-molecule mass spectrometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    P,
    S,
    F,
    Cl,
    Br,
    I,
    Si,
    B,
}

impl Element {
    /// Returns the IUPAC atomic number (Z) for this element.
    #[inline(always)]
    pub const fn atomic_number(&self) -> usize {
        match self {
            Element::H => 1,
            Element::C => 6,
            Element::N => 7,
            Element::O => 8,
            Element::P => 15,
            Element::S => 16,
            Element::F => 9,
            Element::Cl => 17,
            Element::Br => 35,
            Element::I => 53,
            Element::Si => 14,
            Element::B => 5,
        }
    }

    /// Monoisotopic mass sourced directly from `chemistry_consts::MONOISOTOPIC_MASSES`.
    #[inline(always)]
    pub fn monoisotopic_mass(&self) -> f64 {
        MONOISOTOPIC_MASSES[self.atomic_number()]
    }

    /// Standard nominal valence for Senior's rules and DBE calculation.
    #[inline(always)]
    pub const fn valence(&self) -> usize {
        match self {
            Element::H => 1,
            Element::C => 4,
            Element::N => 3,
            Element::O => 2,
            Element::P => 3,
            Element::S => 2,
            Element::F => 1,
            Element::Cl => 1,
            Element::Br => 1,
            Element::I => 1,
            Element::Si => 4,
            Element::B => 3,
        }
    }

    /// Standard element symbol string.
    #[inline(always)]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::P => "P",
            Element::S => "S",
            Element::F => "F",
            Element::Cl => "Cl",
            Element::Br => "Br",
            Element::I => "I",
            Element::Si => "Si",
            Element::B => "B",
        }
    }
}

pub const NUM_ELEMENTS: usize = 12;

pub const ALL_ELEMENTS: [Element; NUM_ELEMENTS] = [
    Element::H,
    Element::C,
    Element::N,
    Element::O,
    Element::P,
    Element::S,
    Element::F,
    Element::Cl,
    Element::Br,
    Element::I,
    Element::Si,
    Element::B,
];

/// Exact IUPAC Monoisotopic masses sourced directly from `chemistry_consts::MONOISOTOPIC_MASSES`.
/// Note: H is at index 0 because the Round-Robin DP table requires
/// the lightest element to be at index 0.
pub const ELEMENT_MASSES: [f64; NUM_ELEMENTS] = [
    MONOISOTOPIC_MASSES[Element::H.atomic_number()],  // H (Z=1)
    MONOISOTOPIC_MASSES[Element::C.atomic_number()],  // C (Z=6)
    MONOISOTOPIC_MASSES[Element::N.atomic_number()],  // N (Z=7)
    MONOISOTOPIC_MASSES[Element::O.atomic_number()],  // O (Z=8)
    MONOISOTOPIC_MASSES[Element::P.atomic_number()],  // P (Z=15)
    MONOISOTOPIC_MASSES[Element::S.atomic_number()],  // S (Z=16)
    MONOISOTOPIC_MASSES[Element::F.atomic_number()],  // F (Z=9)
    MONOISOTOPIC_MASSES[Element::Cl.atomic_number()], // Cl (Z=17)
    MONOISOTOPIC_MASSES[Element::Br.atomic_number()], // Br (Z=35)
    MONOISOTOPIC_MASSES[Element::I.atomic_number()],  // I (Z=53)
    MONOISOTOPIC_MASSES[Element::Si.atomic_number()], // Si (Z=14)
    MONOISOTOPIC_MASSES[Element::B.atomic_number()],  // B (Z=5)
];

/// Standard nominal valences for Senior's rules and DBE calculation.
pub const ELEMENT_VALENCES: [usize; NUM_ELEMENTS] = [
    Element::H.valence(),
    Element::C.valence(),
    Element::N.valence(),
    Element::O.valence(),
    Element::P.valence(),
    Element::S.valence(),
    Element::F.valence(),
    Element::Cl.valence(),
    Element::Br.valence(),
    Element::I.valence(),
    Element::Si.valence(),
    Element::B.valence(),
];

/// Standard element symbol strings.
pub const ELEMENT_SYMBOLS: [&str; NUM_ELEMENTS] = [
    Element::H.symbol(),
    Element::C.symbol(),
    Element::N.symbol(),
    Element::O.symbol(),
    Element::P.symbol(),
    Element::S.symbol(),
    Element::F.symbol(),
    Element::Cl.symbol(),
    Element::Br.symbol(),
    Element::I.symbol(),
    Element::Si.symbol(),
    Element::B.symbol(),
];

/// Natural isotope data: (mass, abundance, nominal mass delta from monoisotopic peak)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Isotope {
    pub mass: f64,
    pub abundance: f64,
    pub mass_number_delta: usize, // e.g. 0 for M, 1 for M+1, 2 for M+2
}

#[inline(always)]
pub const fn get_element_index(el: Element) -> usize {
    match el {
        Element::H => 0,
        Element::C => 1,
        Element::N => 2,
        Element::O => 3,
        Element::P => 4,
        Element::S => 5,
        Element::F => 6,
        Element::Cl => 7,
        Element::Br => 8,
        Element::I => 9,
        Element::Si => 10,
        Element::B => 11,
    }
}

/// Fixed-size container for an element's natural isotopes (at most 4 stable isotopes per element).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementIsotopes {
    pub isotopes: [Isotope; 4],
    pub count: usize,
}

impl ElementIsotopes {
    #[inline]
    pub fn as_slice(&self) -> &[Isotope] {
        &self.isotopes[..self.count]
    }
}

/// Dynamically extracts natural isotopes for an element directly from `chemistry_consts::ISOTOPES[Z]`.
/// Sourced from NIST / CODATA.
pub fn get_element_isotopes(el: Element) -> ElementIsotopes {
    let z = el.atomic_number();
    let mono_mass = MONOISOTOPIC_MASSES[z];
    let mono_nominal = mono_mass.round() as isize;

    let mut isotopes = [Isotope {
        mass: 0.0,
        abundance: 0.0,
        mass_number_delta: 0,
    }; 4];
    let mut count = 0;

    for opt_iso in &ISOTOPES[z] {
        if let Some(iso) = opt_iso {
            let delta = (iso.mass.round() as isize - mono_nominal).unsigned_abs();
            isotopes[count] = Isotope {
                mass: iso.mass,
                abundance: iso.abundance,
                mass_number_delta: delta,
            };
            count += 1;
        }
    }

    ElementIsotopes { isotopes, count }
}
