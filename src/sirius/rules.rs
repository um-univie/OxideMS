use super::formula::Formula;

/// How hard a gate stands between a candidate neutral loss and the graph.
///
/// SIRIUS has no such gate. `LossValidator` ships with no concrete implementation
/// and the default profile installs none; loss plausibility is carried entirely by
/// the four loss scorers (`FreeRadicalEdgeScorer`, `DBELossScorer`,
/// `PureCarbonNitrogenLossScorer`, `StrangeElementLossScorer`) plus
/// `LossSizeScorer`, all of which `sirius_scoring::edge_score` already has. Our
/// gate is therefore an *extra* filter, and round 24 measured what it costs: 6.2 %
/// of the loss edges SIRIUS actually uses cannot exist in our graph, and the
/// spectra that touches have a loss-edge Jaccard of 0.387 against 0.636 for the
/// rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossGate {
    /// The original hand-rolled whitelist. Rejects methylamine (`CH5N`, through an
    /// `H/C <= 4` hydrocarbon ceiling applied to a nitrogen-bearing loss), `C2O2`
    /// (the `H == 0` branch is a five-entry whitelist), and `S`.
    Legacy,
    /// The same rules with the measured defects removed: no `H/C` bounds — the RDBE
    /// window already covers saturation, and covers it correctly for heteroatoms —
    /// no `H == 0` whitelist, no pure-heteroatom ban. What survives is what is
    /// defensible: a loss must have atoms, an RDBE in [-0.5, 25], and enough
    /// valence to hold itself together.
    #[default]
    Repaired,
    /// No gate, as SIRIUS. Any subformula difference is an edge, and the loss
    /// scorers decide what it is worth.
    None,
}

/// Evaluates chemical feasibility of a molecular formula using Senior's rules,
/// Seven Golden Rules (Kind & Fiehn 2007), and DBE bounds.
#[derive(Debug, Clone)]
pub struct ChemicalRules {
    /// Which neutral losses may become graph edges at all. See [`LossGate`].
    pub loss_gate: LossGate,
    pub check_senior: bool,
    pub check_golden_ratios: bool,
    pub check_dbe: bool,
    pub allow_radicals: bool,
    pub min_h_c: f64,
    pub max_h_c: f64,
    pub max_n_c: f64,
    pub max_o_c: f64,
    pub max_p_c: f64,
    pub max_s_c: f64,
    pub max_halogens_c: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
}

impl Default for ChemicalRules {
    fn default() -> Self {
        Self {
            loss_gate: LossGate::default(),
            check_senior: true,
            check_golden_ratios: true,
            check_dbe: true,
            allow_radicals: false,
            min_h_c: 0.1,
            max_h_c: 4.0,
            max_n_c: 2.0,
            max_o_c: 2.5,
            max_p_c: 0.5,
            max_s_c: 1.0,
            max_halogens_c: 3.0,
            min_dbe: -0.5,
            max_dbe: 40.0,
        }
    }
}

impl ChemicalRules {
    /// Rules for annotating MS/MS fragment peaks.
    ///
    /// Senior's rules are off here deliberately, and this is the one preset where
    /// that is right: they describe a complete neutral molecule, and a fragment
    /// annotation is not one — it is the neutral core of an ion, which may be a
    /// carbocation or an acylium with no closed-shell neutral counterpart.
    /// Measured over 807 labelled GNPS spectra, enforcing them on fragments costs
    /// about a point of top-1 precursor accuracy (81.7% -> 80.7% tuning,
    /// 78.4% -> 77.4% holdout). `default()`, which generates whole precursor
    /// molecules, keeps them.
    ///
    /// Until `is_valid` stopped taking the caller's word for whether a formula was
    /// a radical, `allow_radicals: true` switched these off as a side effect, along
    /// with the DBE parity and nitrogen-rule tests, for every candidate.
    pub fn loose() -> Self {
        Self {
            loss_gate: LossGate::default(),
            check_senior: false,
            check_golden_ratios: true,
            check_dbe: true,
            allow_radicals: true,
            min_h_c: 0.0,
            max_h_c: 5.0,
            max_n_c: 3.0,
            max_o_c: 3.0,
            max_p_c: 1.0,
            max_s_c: 2.0,
            max_halogens_c: 4.0,
            min_dbe: -1.0,
            max_dbe: 60.0,
        }
    }

    /// Validates if a molecular formula is chemically plausible.
    ///
    /// Whether the formula is a radical is derived from it, not passed in.
    /// `allow_radicals` is a policy — may an odd-electron species be returned at
    /// all — and used to be conflated with the property, so a rule set that
    /// permitted radicals skipped Senior's rules, the DBE parity test and the
    /// nitrogen rule for *every* candidate, radical or not.
    pub fn is_valid(&self, formula: &Formula) -> bool {
        if formula.total_atoms() == 0 {
            return false;
        }

        let is_radical = formula.is_radical();
        if is_radical && !self.allow_radicals {
            return false;
        }

        // 1. Senior's Rules
        if self.check_senior && !self.senior_rules(formula) {
            return false;
        }

        // 2. Ring Double Bond Equivalents (DBE)
        //
        // DBE = 1 + (sum_valences - 2 * atoms) / 2, so a half-integer DBE and an
        // odd valence sum are the same statement: the old separate "DBE must be
        // an integer for non-radicals" test was Senior's parity clause again.
        if self.check_dbe {
            let dbe = formula.dbe();
            if dbe < self.min_dbe || dbe > self.max_dbe {
                return false;
            }
        }

        // The nitrogen rule used to be tested here. It is not missing: for this
        // element set it is the radical test above, written differently.
        //
        // Nitrogen is the only element whose valence and nominal mass have
        // different parity (3 and 14; every other element's sum is even), so
        //
        //     nominal_mass = sum_valences + N   (mod 2)
        //
        // and the rule's claim, `nominal_mass = N (mod 2)`, reduces to
        // `sum_valences = 0 (mod 2)` — exactly "not a radical". Checking it again
        // could never reject a formula the radical test admitted.
        // `test_nitrogen_rule_is_the_radical_test` pins the parity table this rests on.

        // 3. Seven Golden Rules (Elemental Ratios)
        if self.check_golden_ratios && !self.golden_ratios(formula) {
            return false;
        }

        true
    }

    /// Senior's Rules:
    /// 1. Sum of valences is even for closed-shell molecules.
    /// 2. Sum of valences >= 2 * max_valence.
    /// 3. Sum of valences >= 2 * (total_atoms - 1).
    pub fn senior_rules(&self, formula: &Formula) -> bool {
        // Rule 1 is the radical test, and whether a radical is acceptable is the
        // rule set's policy. Rules 2 and 3 are about whether the atoms can be
        // connected at all, which is true of radicals as much as anything else —
        // skipping them whenever radicals were permitted, as this used to, left
        // Senior's rules doing nothing in the whole fragment-annotation path.
        if formula.is_radical() && !self.allow_radicals {
            return false;
        }

        Self::senior_connectivity(formula)
    }

    /// Senior's second and third conditions: the highest-valence atom can have its
    /// bonds satisfied, and there is enough valence to bond every atom into one
    /// connected structure. Independent of electron parity, and so of any policy
    /// about radicals.
    pub fn senior_connectivity(formula: &Formula) -> bool {
        let total_atoms = formula.total_atoms();
        if total_atoms <= 1 {
            return true;
        }

        formula.sum_valences() >= 2 * formula.max_valence() && Self::is_connectable(formula)
    }

    /// Senior's third condition alone: enough valence to bond every atom into one
    /// connected species. Equivalently DBE >= 0, which is why anything passing this
    /// has a non-negative ring-and-double-bond count.
    ///
    /// This is the part that holds without exception. Senior's second condition does
    /// not: it rejects carbon monoxide, whose dative bond satisfies the carbon, and
    /// methylene — species listed as permissible losses a few lines below and
    /// rewarded by the common-loss dictionary. Small stable molecules like those are
    /// exactly what leaves during collision-induced dissociation, so neutral losses
    /// are held to this condition only.
    pub fn is_connectable(formula: &Formula) -> bool {
        let total_atoms = formula.total_atoms();
        total_atoms <= 1 || formula.sum_valences() >= 2 * (total_atoms - 1)
    }

    /// Golden Rules elemental ratios
    pub fn golden_ratios(&self, formula: &Formula) -> bool {
        let c = formula.c();
        if c == 0 {
            return formula.total_atoms() <= 15;
        }

        let c_f = c as f64;
        let h_c = formula.h() as f64 / c_f;
        let n_c = formula.n() as f64 / c_f;
        let o_c = formula.o() as f64 / c_f;
        let p_c = formula.p() as f64 / c_f;
        let s_c = formula.s() as f64 / c_f;
        let halogens = (formula.f() + formula.cl() + formula.br() + formula.i()) as f64;
        let hal_c = halogens / c_f;

        if h_c < self.min_h_c || h_c > self.max_h_c {
            return false;
        }
        if n_c > self.max_n_c {
            return false;
        }
        if o_c > self.max_o_c {
            return false;
        }
        if p_c > self.max_p_c {
            return false;
        }
        if s_c > self.max_s_c {
            return false;
        }
        if hal_c > self.max_halogens_c {
            return false;
        }

        true
    }

    /// Evaluates whether a candidate neutral loss formula is chemically feasible during MS/MS fragmentation.
    pub fn is_valid_neutral_loss(&self, loss: &Formula) -> bool {
        match self.loss_gate {
            LossGate::None => loss.total_atoms() > 0,
            LossGate::Repaired => self.loss_is_plausible(loss),
            LossGate::Legacy => self.loss_passes_legacy_whitelist(loss),
        }
    }

    /// [`LossGate::Repaired`]: the defensible core of the old whitelist.
    fn loss_is_plausible(&self, loss: &Formula) -> bool {
        if loss.total_atoms() == 0 {
            return false;
        }
        let dbe = loss.dbe();
        if !(-0.5..=25.0).contains(&dbe) {
            return false;
        }
        // No heteroatom-ratio caps. Those are Seven-Golden-Rules limits for whole
        // molecules, and a small neutral loss is exactly where they misfire:
        // O/C <= 2.5 rejects carbonic acid, which is H2O + CO2 leaving together.
        // The same category error as the H/C ceiling that rejected methylamine.
        Self::is_connectable(loss)
    }

    fn loss_passes_legacy_whitelist(&self, loss: &Formula) -> bool {
        let total_atoms = loss.total_atoms();
        if total_atoms == 0 {
            return false;
        }

        let c = loss.c();
        let h = loss.h();
        let n = loss.n();
        let o = loss.o();
        let p = loss.p();
        let s = loss.s();
        let halogens = loss.f() + loss.cl() + loss.br() + loss.i();

        // 1. Inorganic / Non-carbon neutral losses (C == 0)
        if c == 0 {
            // Known permissible inorganic species:
            // H2, H2O, NH3, N2, H2S, SO2, SO3, H3PO4, HPO3, HCl, HBr, HF, HI
            // Simple radicals: OH, NO, NO2, H, SH, Cl, Br, F, I, NH2
            if h == 2 && total_atoms == 2 { return true; } // H2
            if h == 2 && o == 1 && total_atoms == 3 { return true; } // H2O
            if h == 3 && n == 1 && total_atoms == 4 { return true; } // NH3
            if n == 2 && total_atoms == 2 { return true; } // N2
            if h == 2 && s == 1 && total_atoms == 3 { return true; } // H2S
            if s == 1 && o == 2 && total_atoms == 3 { return true; } // SO2
            if s == 1 && o == 3 && total_atoms == 4 { return true; } // SO3
            if p == 1 && o == 4 && h == 3 && total_atoms == 8 { return true; } // H3PO4
            if p == 1 && o == 3 && h == 1 && total_atoms == 5 { return true; } // HPO3
            if h == 1 && halogens == 1 && total_atoms == 2 { return true; } // Hydrogen halides
            if n == 1 && o == 1 && total_atoms == 2 { return true; } // NO
            if n == 1 && o == 2 && total_atoms == 3 { return true; } // NO2
            if o == 1 && h == 1 && total_atoms == 2 { return true; } // OH
            if o == 2 && total_atoms == 2 { return true; } // O2
            if s == 1 && h == 1 && total_atoms == 2 { return true; } // SH
            if n == 1 && h == 2 && total_atoms == 3 { return true; } // NH2
            if total_atoms == 1 && (h == 1 || halogens == 1) { return true; } // H or Halogen radical

            // General Senior's check for larger inorganic polyatomic losses
            if total_atoms > 8 {
                return false;
            }

            // Forbid pure heteroatom clusters (pure N>2, pure O>2, pure P, pure S)
            if (n > 2 && h == 0 && o == 0) || (o > 2 && h == 0 && n == 0) || (s > 0 && h == 0 && o == 0) || (p > 0 && h == 0 && o == 0) {
                return false;
            }

            return Self::is_connectable(loss);
        }

        // 2. Carbon-containing neutral losses (C > 0)
        // If H == 0: must be an oxygenated/halogenated carbon species (CO, CO2, C3O2, CS2, CCl4, etc.)
        if h == 0 {
            if c == 1 && o == 1 && total_atoms == 2 { return true; } // CO
            if c == 1 && o == 2 && total_atoms == 3 { return true; } // CO2
            if c == 3 && o == 2 && total_atoms == 5 { return true; } // C3O2
            if c == 1 && s == 2 && total_atoms == 3 { return true; } // CS2
            if c == 1 && halogens == 4 && total_atoms == 5 { return true; } // CX4
            // Pure carbon clusters (C1, C2, C3, C5...) without heteroatoms are strictly forbidden in CID
            return false;
        }

        // If H > 0:
        let c_f = c as f64;
        let h_c = h as f64 / c_f;
        if h_c < 0.33 || h_c > 4.0 {
            return false;
        }

        let dbe = loss.dbe();
        if dbe < -0.5 || dbe > 25.0 {
            return false;
        }

        let o_c = o as f64 / c_f;
        let n_c = n as f64 / c_f;
        let p_c = p as f64 / c_f;
        let s_c = s as f64 / c_f;

        if o_c > 2.5 || n_c > 2.0 || p_c > 0.5 || s_c > 1.0 {
            return false;
        }

        Self::is_connectable(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sirius::element::{ALL_ELEMENTS, ELEMENT_MASSES, ELEMENT_VALENCES};
    use std::str::FromStr;

    /// The three losses round 24 caught the legacy whitelist rejecting, each of
    /// which SIRIUS uses in trees it actually builds.
    #[test]
    fn repaired_loss_gate_admits_the_losses_the_whitelist_wrongly_refused() {
        let legacy = ChemicalRules { loss_gate: LossGate::Legacy, ..ChemicalRules::loose() };
        let repaired = ChemicalRules { loss_gate: LossGate::Repaired, ..ChemicalRules::loose() };

        for (f, why) in [
            ("CH5N", "methylamine, refused by an H/C <= 4 hydrocarbon ceiling"),
            ("C2O2", "two carbonyls, refused by the H == 0 whitelist"),
            ("S", "refused by the pure-heteroatom ban"),
            ("CH2O3", "carbonic acid"),
            ("CS", "refused by the H == 0 whitelist"),
        ] {
            let loss = Formula::from_str(f).unwrap();
            assert!(!legacy.is_valid_neutral_loss(&loss), "{f} should show the legacy defect ({why})");
            assert!(repaired.is_valid_neutral_loss(&loss), "{f} must be admitted: {why}");
        }

        // Still a gate, not an absence of one: an RDBE far below -0.5 stays out.
        let absurd = Formula::from_str("CH10").unwrap();
        assert!(!repaired.is_valid_neutral_loss(&absurd), "CH10 has RDBE -3, it must stay out");
        assert!(!repaired.is_valid_neutral_loss(&Formula::from_str("").unwrap_or_default()));
    }

    /// `LossGate::None` is SIRIUS's behaviour: no gate at all beyond a non-empty
    /// loss, with plausibility left to `sirius_scoring::edge_score`.
    #[test]
    fn none_loss_gate_admits_anything_with_atoms() {
        let none = ChemicalRules { loss_gate: LossGate::None, ..ChemicalRules::loose() };
        for f in ["CH5N", "C2O2", "S", "CH10", "N9"] {
            assert!(none.is_valid_neutral_loss(&Formula::from_str(f).unwrap()), "{f}");
        }
        assert!(!none.is_valid_neutral_loss(&Formula::default()));
    }

    /// `is_valid` drops the nitrogen rule because it is the radical test restated.
    ///
    /// Nitrogen is the only ordinary element whose valence and nominal mass differ
    /// in parity (3 and 14), which makes `nominal_mass = sum_valences + N (mod 2)`,
    /// so the rule's claim `nominal_mass = N (mod 2)` reduces to "the valence sum is
    /// even" — exactly "not a radical".
    ///
    /// Boron is the exception, and it argues for the removal rather than against it:
    /// this crate's boron mass comes from `chemistry_consts` as 10-B, and at nominal
    /// mass 10 the parity coincidence breaks. Where the two tests then disagree it is
    /// the nitrogen rule that is wrong — it rejects closed-shell boranes. No
    /// generator alphabet includes boron.
    ///
    /// This pins the parity table the reduction rests on: a new element with odd
    /// valence+nominal parity would quietly turn the nitrogen rule back into a real
    /// constraint, and this test is what would notice.
    #[test]
    fn test_nitrogen_rule_is_the_radical_test() {
        let odd: Vec<&str> = ALL_ELEMENTS
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                (ELEMENT_VALENCES[*i] + ELEMENT_MASSES[*i].round() as usize) % 2 == 1
            })
            .map(|(_, el)| el.symbol())
            .collect();

        assert_eq!(
            odd,
            vec!["N", "B"],
            "the reduction holds only while N and B are the only elements with odd              valence+nominal parity; {:?} now break it",
            odd
        );

        // The two tests agree on everything the generators can produce.
        for text in ["N", "C6H6", "C9H8O4", "C21H25NO4", "C6H7", "C13H14N3O5", "CH3", "H2O"] {
            let f = Formula::from_str(text).unwrap();
            let nitrogen_rule_holds = f.nominal_mass() % 2 == f.n() as u32 % 2;
            assert_eq!(
                nitrogen_rule_holds,
                !f.is_radical(),
                "{}: the nitrogen rule and the radical test must agree",
                f
            );
        }

        // And where they part company, on boron, the rule is the one that is wrong.
        let borane = Formula::from_str("BH3").unwrap();
        assert!(!borane.is_radical(), "BH3 is closed-shell");
        assert_ne!(
            borane.nominal_mass() % 2,
            borane.n() as u32 % 2,
            "the nitrogen rule would have rejected BH3"
        );
    }
}
