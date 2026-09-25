use super::element::{
    Element, ALL_ELEMENTS, ELEMENT_MASSES, ELEMENT_VALENCES, NUM_ELEMENTS, get_element_index,
};
use super::formula::Formula;
use super::rules::ChemicalRules;
use crate::workflow::MassTolerance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const SCALE_FACTOR: f64 = 100_000.0;
const PPM_FACTOR: f64 = 1_000_000.0;

/// Fast Round-Robin Dynamic Programming formula generator.
/// Implements the Frobenius coin problem algorithm from Böcker & Lipták (2007)
/// to decompose exact masses into molecular formulas with branch-and-bound pruning.
#[derive(Clone, Debug)]
pub struct FormulaGenerator<const N: usize> {
    dp_tables: Vec<Vec<usize>>,
    scaled_weights: [usize; N],
    exact_masses: [f64; N],
    valences: [usize; N],
    element_map: [Element; N],
    pub integer_padding: usize,
}

impl FormulaGenerator<6> {
    /// Generator for standard organic metabolomics (CHNOPS)
    /// Lightest element H must be at index 0.
    pub fn chnops() -> Self {
        let elements = [
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::P,
            Element::S,
        ];
        Self::from_elements(&elements)
    }
}

impl FormulaGenerator<4> {
    /// Generator for standard natural products and metabolites (CHNO)
    pub fn chno() -> Self {
        let elements = [Element::H, Element::C, Element::N, Element::O];
        Self::from_elements(&elements)
    }
}

impl FormulaGenerator<5> {
    /// Generator for CHNOS (CHNO + Sulfur)
    pub fn chnos() -> Self {
        let elements = [Element::H, Element::C, Element::N, Element::O, Element::S];
        Self::from_elements(&elements)
    }
}

impl FormulaGenerator<8> {
    /// Generator for standard halogenated natural products (CHNOPS + Cl + Br)
    pub fn chnops_halogens() -> Self {
        let elements = [
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::P,
            Element::S,
            Element::Cl,
            Element::Br,
        ];
        Self::from_elements(&elements)
    }
}

impl FormulaGenerator<9> {
    /// CHNOPS + F, Cl, Br — covers essentially every halogenated small molecule
    /// (pesticides, halogenated drugs and natural products) without paying for the
    /// I/Si/B columns of the full alphabet.
    pub fn chnops_fclbr() -> Self {
        let elements = [
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::P,
            Element::S,
            Element::F,
            Element::Cl,
            Element::Br,
        ];
        Self::from_elements(&elements)
    }
}

impl FormulaGenerator<NUM_ELEMENTS> {
    /// Generator for extended element set (CHNOPS + F, Cl, Br, I, Si, B)
    pub fn extended() -> Self {
        Self::from_elements(&ALL_ELEMENTS)
    }
}

impl<const N: usize> FormulaGenerator<N> {
    /// Build Round-Robin DP tables for a given element alphabet.
    /// Elements MUST have the lightest element at index 0 (typically Hydrogen).
    pub fn from_elements(elements: &[Element; N]) -> Self {
        let mut exact_masses = [0.0; N];
        let mut valences = [0; N];
        let mut scaled_weights = [0; N];

        for (i, &el) in elements.iter().enumerate() {
            let idx = get_element_index(el);
            exact_masses[i] = ELEMENT_MASSES[idx];
            valences[i] = ELEMENT_VALENCES[idx];
            scaled_weights[i] = (exact_masses[i] * SCALE_FACTOR).round() as usize;
        }

        let w0 = scaled_weights[0]; // Lightest element (e.g. H)
        assert!(w0 > 0, "Weight of lightest element must be positive");

        // d[i][r] = minimum scaled mass to reach remainder r modulo w0 using elements 0..=i
        let mut d = vec![vec![usize::MAX; w0]; N];
        d[0][0] = 0;

        for i in 1..N {
            d[i] = d[i - 1].clone();
            let mut heap = BinaryHeap::new();

            for r in 0..w0 {
                if d[i][r] != usize::MAX {
                    heap.push(Reverse((d[i][r], r)));
                }
            }

            // Dijkstra's algorithm to fill the table for current element
            while let Some(Reverse((cost, r))) = heap.pop() {
                if cost > d[i][r] {
                    continue;
                }
                let next_cost = cost + scaled_weights[i];
                let next_r = next_cost % w0;

                if next_cost < d[i][next_r] {
                    d[i][next_r] = next_cost;
                    heap.push(Reverse((next_cost, next_r)));
                }
            }
        }

        Self {
            dp_tables: d,
            scaled_weights,
            exact_masses,
            valences,
            element_map: *elements,
            integer_padding: 20,
        }
    }

    /// Set custom integer rounding padding for branch-and-bound search
    pub fn with_integer_padding(mut self, padding: usize) -> Self {
        self.integer_padding = padding;
        self
    }

    /// Generates unconstrained candidate formulas for a target mass
    pub fn generate_candidates(
        &self,
        target_mass: f64,
        tolerance: MassTolerance,
        rules: &ChemicalRules,
    ) -> Vec<Formula> {
        let max_error_da = match tolerance {
            MassTolerance::Da(da) => da,
            MassTolerance::Ppm(ppm) => target_mass * (ppm / PPM_FACTOR),
        };

        let min_integer_mass =
            (((target_mass - max_error_da) * SCALE_FACTOR).floor() as usize).saturating_sub(self.integer_padding);
        let max_integer_mass =
            ((target_mass + max_error_da) * SCALE_FACTOR).ceil() as usize + self.integer_padding;

        // Stoichiometric upper bounds for target mass
        let mut bounds_array = [0; N];
        for i in 0..N {
            let max_stoich = (target_mass / self.exact_masses[i]).floor() as usize;
            let max_chem = match self.element_map[i] {
                Element::Cl => 8,
                Element::Br => 4,
                Element::F => 10,
                Element::I => 4,
                Element::P => 6,
                Element::S => 8,
                _ => usize::MAX,
            };
            bounds_array[i] = max_stoich.min(max_chem);
        }

        let mut raw_results = Vec::new();
        let w0 = self.scaled_weights[0];
        let mut current = [0; N];

        // Search the interval using Round-Robin reachability bounds
        for m in min_integer_mass..=max_integer_mass {
            if self.dp_tables[N - 1][m % w0] <= m {
                self.branch_and_bound(
                    N - 1,
                    m,
                    &mut current,
                    &mut raw_results,
                    Some(&bounds_array),
                );
            }
        }

        // Exact floating-point filtering & chemical validation
        let mut candidates = Vec::new();
        for counts in raw_results {
            let formula = self.counts_to_formula(&counts);
            let exact_mass = formula.exact_mass();
            let mass_err = (exact_mass - target_mass).abs();

            let within_tol = match tolerance {
                MassTolerance::Da(da) => mass_err <= da,
                MassTolerance::Ppm(ppm) => (mass_err / target_mass) * PPM_FACTOR <= ppm,
            };

            if within_tol && rules.is_valid(&formula) {
                candidates.push(formula);
            }
        }

        // Sort by absolute mass error
        candidates.sort_by(|a, b| {
            let err_a = (a.exact_mass() - target_mass).abs();
            let err_b = (b.exact_mass() - target_mass).abs();
            err_a.total_cmp(&err_b)
        });

        candidates
    }

    /// Generates candidate sub-formulas for a fragment mass bounded by a parent/precursor formula.
    /// Guarantees that all returned formulas F satisfy F <= max_bounds.
    pub fn generate_subformulas(
        &self,
        target_mass: f64,
        max_bounds: &Formula,
        tolerance: MassTolerance,
        rules: &ChemicalRules,
    ) -> Vec<Formula> {
        let max_error_da = match tolerance {
            MassTolerance::Da(da) => da,
            MassTolerance::Ppm(ppm) => target_mass * (ppm / PPM_FACTOR),
        };

        // Convert max_bounds into array mapped to self.element_map
        let mut bounds_array = [0; N];
        for (i, &el) in self.element_map.iter().enumerate() {
            bounds_array[i] = max_bounds.get(el);
        }

        let min_integer_mass =
            (((target_mass - max_error_da) * SCALE_FACTOR).floor() as usize).saturating_sub(self.integer_padding);
        let max_integer_mass =
            ((target_mass + max_error_da) * SCALE_FACTOR).ceil() as usize + self.integer_padding;

        let mut raw_results = Vec::new();
        let w0 = self.scaled_weights[0];
        let mut current = [0; N];

        for m in min_integer_mass..=max_integer_mass {
            if self.dp_tables[N - 1][m % w0] <= m {
                self.branch_and_bound(
                    N - 1,
                    m,
                    &mut current,
                    &mut raw_results,
                    Some(&bounds_array),
                );
            }
        }

        let mut candidates = Vec::new();
        for counts in raw_results {
            let formula = self.counts_to_formula(&counts);
            let exact_mass = formula.exact_mass();
            let mass_err = (exact_mass - target_mass).abs();

            let within_tol = match tolerance {
                MassTolerance::Da(da) => mass_err <= da,
                MassTolerance::Ppm(ppm) => (mass_err / target_mass) * PPM_FACTOR <= ppm,
            };

            if within_tol && rules.is_valid(&formula) {
                candidates.push(formula);
            }
        }

        candidates.sort_by(|a, b| {
            let err_a = (a.exact_mass() - target_mass).abs();
            let err_b = (b.exact_mass() - target_mass).abs();
            err_a.total_cmp(&err_b)
        });

        candidates
    }

    fn branch_and_bound(
        &self,
        i: usize,
        m: usize,
        current: &mut [usize; N],
        results: &mut Vec<[usize; N]>,
        bounds: Option<&[usize; N]>,
    ) {
        if m == 0 {
            results.push(*current);
            return;
        }

        if i == 0 {
            let w0 = self.scaled_weights[0];
            if m % w0 == 0 {
                let count = m / w0;
                if let Some(b) = bounds {
                    if current[0] + count > b[0] {
                        return;
                    }
                }
                current[0] += count;
                results.push(*current);
                current[0] -= count;
            }
            return;
        }

        // Branch 1: Skip current element i
        if self.dp_tables[i - 1][m % self.scaled_weights[0]] <= m {
            self.branch_and_bound(
                i - 1,
                m,
                current,
                results,
                bounds,
            );
        }

        // Branch 2: Take current element i (if within upper bound)
        let weight_i = self.scaled_weights[i];
        let can_take = if let Some(b) = bounds {
            current[i] < b[i]
        } else {
            true
        };

        if can_take && m >= weight_i {
            let next_m = m - weight_i;
            if self.dp_tables[i][next_m % self.scaled_weights[0]] <= next_m {
                current[i] += 1;
                self.branch_and_bound(
                    i,
                    next_m,
                    current,
                    results,
                    bounds,
                );
                current[i] -= 1;
            }
        }
    }

    fn counts_to_formula(&self, counts: &[usize; N]) -> Formula {
        let mut f = Formula::empty();
        for (i, &el) in self.element_map.iter().enumerate() {
            f.set(el, counts[i]);
        }
        f
    }
}
