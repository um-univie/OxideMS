use super::element::{
    Element, ALL_ELEMENTS, ELEMENT_MASSES, ELEMENT_SYMBOLS, ELEMENT_VALENCES, NUM_ELEMENTS,
    get_element_index,
};
use std::fmt;
use std::ops::{Add, Sub};
use std::str::FromStr;

/// A compact, high-performance representation of a chemical molecular formula.
/// Stores atom counts for the 12 standard organic/metabolite elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Formula {
    pub counts: [usize; NUM_ELEMENTS],
}

impl Formula {
    pub const fn new(counts: [usize; NUM_ELEMENTS]) -> Self {
        Self { counts }
    }

    pub const fn empty() -> Self {
        Self {
            counts: [0; NUM_ELEMENTS],
        }
    }

    pub const fn chnops(c: usize, h: usize, n: usize, o: usize, p: usize, s: usize) -> Self {
        let mut counts = [0; NUM_ELEMENTS];
        counts[0] = h;
        counts[1] = c;
        counts[2] = n;
        counts[3] = o;
        counts[4] = p;
        counts[5] = s;
        Self { counts }
    }

    #[inline(always)]
    pub fn get(&self, el: Element) -> usize {
        self.counts[get_element_index(el)]
    }

    #[inline(always)]
    pub fn set(&mut self, el: Element, count: usize) {
        self.counts[get_element_index(el)] = count;
    }

    #[inline(always)]
    pub fn h(&self) -> usize {
        self.counts[0]
    }
    #[inline(always)]
    pub fn c(&self) -> usize {
        self.counts[1]
    }
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.counts[2]
    }
    #[inline(always)]
    pub fn o(&self) -> usize {
        self.counts[3]
    }
    #[inline(always)]
    pub fn p(&self) -> usize {
        self.counts[4]
    }
    #[inline(always)]
    pub fn s(&self) -> usize {
        self.counts[5]
    }
    #[inline(always)]
    pub fn f(&self) -> usize {
        self.counts[6]
    }
    #[inline(always)]
    pub fn cl(&self) -> usize {
        self.counts[7]
    }
    #[inline(always)]
    pub fn br(&self) -> usize {
        self.counts[8]
    }
    #[inline(always)]
    pub fn i(&self) -> usize {
        self.counts[9]
    }
    #[inline(always)]
    pub fn si(&self) -> usize {
        self.counts[10]
    }
    #[inline(always)]
    pub fn b(&self) -> usize {
        self.counts[11]
    }

    /// Total number of atoms
    #[inline]
    pub fn total_atoms(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Exact monoisotopic mass in Da
    #[inline]
    pub fn exact_mass(&self) -> f64 {
        let mut mass = 0.0;
        for i in 0..NUM_ELEMENTS {
            mass += self.counts[i] as f64 * ELEMENT_MASSES[i];
        }
        mass
    }

    /// Nominal mass: the sum of each element's integer mass number.
    ///
    /// Rounding the total exact mass instead, as this used to, diverges from the
    /// nominal mass once a formula is large enough for the mass defects to add up
    /// to half a dalton — around C40H80 — which would silently flip the nitrogen
    /// rule's parity test.
    #[inline]
    pub fn nominal_mass(&self) -> u32 {
        let mut nominal = 0u32;
        for i in 0..NUM_ELEMENTS {
            nominal += self.counts[i] as u32 * ELEMENT_MASSES[i].round() as u32;
        }
        nominal
    }

    /// Sum of standard valences: sum(count_i * valence_i)
    #[inline]
    pub fn sum_valences(&self) -> usize {
        let mut sum = 0;
        for i in 0..NUM_ELEMENTS {
            sum += self.counts[i] * ELEMENT_VALENCES[i];
        }
        sum
    }

    /// Whether this formula is an odd-electron (radical) species.
    ///
    /// A closed-shell neutral molecule pairs every electron, which makes the sum
    /// of its valences even; an odd sum is exactly what a radical is. This is a
    /// property of the formula, never something a caller gets to assert.
    #[inline]
    pub fn is_radical(&self) -> bool {
        self.sum_valences() % 2 != 0
    }

    /// Maximum valence among present elements
    #[inline]
    pub fn max_valence(&self) -> usize {
        let mut max_v = 0;
        for i in 0..NUM_ELEMENTS {
            if self.counts[i] > 0 {
                max_v = max_v.max(ELEMENT_VALENCES[i]);
            }
        }
        max_v
    }

    /// Ring and Double Bond Equivalents (DBE / RDBE)
    /// DBE = 1 + 1/2 * sum(count_i * (valence_i - 2))
    /// For standard organic: 1 + C + Si + (N + P + B - H - Halogens)/2
    #[inline]
    pub fn dbe(&self) -> f64 {
        let halogens = self.f() + self.cl() + self.br() + self.i();
        1.0 + self.c() as f64 + self.si() as f64
            + 0.5 * (self.n() as f64 + self.p() as f64 + self.b() as f64
                - self.h() as f64 - halogens as f64)
    }

    /// Check if self is a sub-formula of another formula (i.e. self <= other for all elements)
    #[inline]
    pub fn is_subformula_of(&self, other: &Formula) -> bool {
        for i in 0..NUM_ELEMENTS {
            if self.counts[i] > other.counts[i] {
                return false;
            }
        }
        true
    }

    /// Checked subtraction: returns Some(diff) if other is a sub-formula of self, otherwise None.
    pub fn checked_sub(&self, other: &Formula) -> Option<Formula> {
        let mut diff = [0; NUM_ELEMENTS];
        for i in 0..NUM_ELEMENTS {
            if self.counts[i] < other.counts[i] {
                return None;
            }
            diff[i] = self.counts[i] - other.counts[i];
        }
        Some(Formula::new(diff))
    }
}

impl Add for Formula {
    type Output = Formula;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = [0; NUM_ELEMENTS];
        for i in 0..NUM_ELEMENTS {
            res[i] = self.counts[i] + rhs.counts[i];
        }
        Formula::new(res)
    }
}

impl Sub for Formula {
    type Output = Option<Formula>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hill notation: C first, then H, then remaining elements in alphabetical order
        if self.c() > 0 {
            write!(f, "C")?;
            if self.c() > 1 {
                write!(f, "{}", self.c())?;
            }
        }
        if self.h() > 0 {
            write!(f, "H")?;
            if self.h() > 1 {
                write!(f, "{}", self.h())?;
            }
        }

        // Remaining elements alphabetically: B, Br, Cl, F, I, N, O, P, S, Si
        let alphabetical_order = [
            Element::B,
            Element::Br,
            Element::Cl,
            Element::F,
            Element::I,
            Element::N,
            Element::O,
            Element::P,
            Element::S,
            Element::Si,
        ];

        for el in alphabetical_order {
            let count = self.get(el);
            if count > 0 {
                write!(f, "{}", ELEMENT_SYMBOLS[get_element_index(el)])?;
                if count > 1 {
                    write!(f, "{}", count)?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseFormulaError(pub String);

impl fmt::Display for ParseFormulaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to parse formula: {}", self.0)
    }
}

impl std::error::Error for ParseFormulaError {}

impl FromStr for Formula {
    type Err = ParseFormulaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut formula = Formula::empty();
        let s = s.trim();
        if s.is_empty() {
            return Ok(formula);
        }

        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if !c.is_ascii_uppercase() {
                return Err(ParseFormulaError(format!("Expected uppercase element symbol, found '{}'", c)));
            }

            let mut symbol = String::with_capacity(2);
            symbol.push(c);

            // Check for lowercase second letter (e.g. Cl, Br, Si)
            if let Some(&next_c) = chars.peek() {
                if next_c.is_ascii_lowercase() {
                    symbol.push(chars.next().unwrap());
                }
            }

            // Find matching element
            let el = match symbol.as_str() {
                "H" => Element::H,
                "C" => Element::C,
                "N" => Element::N,
                "O" => Element::O,
                "P" => Element::P,
                "S" => Element::S,
                "F" => Element::F,
                "Cl" => Element::Cl,
                "Br" => Element::Br,
                "I" => Element::I,
                "Si" => Element::Si,
                "B" => Element::B,
                unknown => return Err(ParseFormulaError(format!("Unsupported or unknown element '{}'", unknown))),
            };

            // Parse count digits
            let mut count_str = String::new();
            while let Some(&next_c) = chars.peek() {
                if next_c.is_ascii_digit() {
                    count_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            let count: usize = if count_str.is_empty() {
                1
            } else {
                count_str.parse().map_err(|_| ParseFormulaError(format!("Invalid atom count '{}'", count_str)))?
            };

            let current = formula.get(el);
            formula.set(el, current + count);
        }

        Ok(formula)
    }
}
