//! Deep diagnostic: For each failure, compare the correct vs wrong tree in detail.
//! Shows exactly where and why the scoring goes wrong.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::adduct::Adduct;
use oxidems_lib::sirius::formula::Formula;
use oxidems_lib::sirius::generator::FormulaGenerator;
use oxidems_lib::sirius::rules::ChemicalRules;
use oxidems_lib::sirius::scoring::FormulaScorer;
use oxidems_lib::sirius::solver::{FragmentationTreeBuilder, TreeSolverConfig};
use oxidems_lib::sirius::tree::FragmentationTree;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::collections::HashMap;
use std::str::FromStr;

fn build_ground_truth() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut gt: HashMap<&str, (&str, &str)> = HashMap::new();
    gt.insert("Pacificanone A", ("C17H22O4", "Positive"));
    gt.insert("Desferrioxamine E", ("C27H48N6O9", "Positive"));
    gt.insert("Staurosporine", ("C28H26N4O3", "Positive"));
    gt.insert("Lymphostin", ("C16H14N4O4", "Positive"));
    gt.insert("Neodysidenin", ("C17H23Cl6NO2", "Positive"));
    gt.insert("epi-Tumonoic acid D", ("C19H25NO", "Positive"));
    gt.insert("Citrinin", ("C13H14O5", "Positive"));
    gt.insert("Proximicin B", ("C22H23N7O5", "Positive"));
    gt.insert("cyclo(L-Leu-L-Pro)", ("C11H18N2O2", "Positive"));
    gt.insert("pateamine", ("C31H45N3O4S", "Positive"));
    gt.insert("cyclo(Phe-Leu)", ("C15H20N2O2", "Positive"));
    gt.insert("cyclo(L-Trp-L-Pro)", ("C16H17N3O2", "Positive"));
    gt.insert("cyclo(L-Val-L-Pro)", ("C10H16N2O2", "Positive"));
    gt.insert("Aerucyclamide B", ("C31H39N7O4S2", "Positive"));
    gt.insert("Aerucyclamide C", ("C32H41N7O4S2", "Positive"));
    gt.insert("Quercetin", ("C15H10O7", "Negative"));
    gt.insert("Acacetin", ("C16H12O5", "Negative"));
    gt.insert("2O-rhamnosyl-swertisin", ("C27H30O14", "Negative"));
    gt.insert("3-O-methylquercetin", ("C16H12O7", "Negative"));
    gt.insert("Apigetrin", ("C21H20O10", "Negative"));
    gt.insert("Avicularin", ("C20H18O11", "Negative"));
    gt.insert("Baicalein", ("C15H10O5", "Negative"));
    gt.insert("Baicalin", ("C21H18O11", "Negative"));
    gt.insert("Formononetin", ("C16H12O4", "Negative"));
    gt.insert("Genistein", ("C15H10O5", "Negative"));
    gt.insert("Hesperetin", ("C16H14O6", "Negative"));
    gt.insert("Isovitexin", ("C21H20O10", "Negative"));
    gt.insert("Kaempferol", ("C15H10O6", "Negative"));
    gt.insert("Luteolin", ("C15H10O6", "Negative"));
    gt.insert("Naringenin", ("C15H12O5", "Negative"));
    gt
}

fn clean_spectrum(peaks: &[Peak], precursor_mz: f64, max_peaks: usize) -> Vec<Peak> {
    let max_int = peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);
    let mut clean: Vec<Peak> = peaks
        .iter()
        .filter(|p| (p.intensity / max_int) >= 0.005)
        .filter(|p| p.mz <= precursor_mz + 0.5)
        .copied()
        .collect();
    clean.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    clean.truncate(max_peaks);
    clean.sort_by(|a, b| a.mz.total_cmp(&b.mz));
    clean
}

mod support;

fn main() {
    support::require("diagnose_failures", &[("test_files/gnps_eval_100.mgf", support::GNPS_EVAL)]);
    println!("================================================================================");
    println!("     Deep Failure Diagnostic: Why does the correct formula lose?");
    println!("================================================================================\n");

    let ground_truth = build_ground_truth();
    let mut parser = MgfParser::default();
    let records = parser.parse_file("test_files/gnps_eval_100.mgf").expect("Failed to parse MGF");

    let gen_chnops = FormulaGenerator::chnops();
    let tolerance = MassTolerance::Ppm(15.0);
    let rules = ChemicalRules::default();
    let scorer = FormulaScorer::new(15.0, 0.05);

    for record in &records {
        let raw_name = match record.get_metadata("NAME") {
            Some(n) => n,
            None => continue,
        };
        let base_name = raw_name
            .trim_end_matches(" M+H")
            .trim_end_matches(" M-H")
            .trim_end_matches(" [M+H]+")
            .trim_end_matches(" [M-H]-");

        let (formula_str, ionmode) = match ground_truth.get(base_name) {
            Some(x) => *x,
            None => continue,
        };

        let true_formula = match Formula::from_str(formula_str) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let pepmass_str = match record.get_metadata("PEPMASS") {
            Some(s) => s,
            None => continue,
        };
        let pepmass: f64 = match pepmass_str.split_whitespace().next().unwrap_or("0").parse() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let adduct = if ionmode.eq_ignore_ascii_case("Negative") {
            Adduct::deprotonated()
        } else {
            Adduct::protonated()
        };

        let neutral_mass = adduct.ion_mz_to_neutral_mass(pepmass);
        let exact_mass = true_formula.exact_mass();

        if (neutral_mass - exact_mass).abs() > 0.05 || record.peaks().len() < 3 {
            continue;
        }

        let clean_peaks = clean_spectrum(record.peaks(), pepmass, 30);
        if clean_peaks.len() < 2 {
            continue;
        }

        // Generate candidates
        let candidate_precursors = gen_chnops.generate_candidates(neutral_mass, tolerance, &rules);

        let oracle = candidate_precursors.contains(&true_formula);

        let tree_config = TreeSolverConfig {
            tolerance,
            adduct: adduct.clone(),
            ..Default::default()
        };
        let tree_builder = FragmentationTreeBuilder::new(tree_config);

        let scored_trees = tree_builder.build_trees_for_candidates_with_adduct(
            &candidate_precursors,
            pepmass,
            &clean_peaks,
            &adduct,
        );

        let mut ranked: Vec<(Formula, FragmentationTree)> = scored_trees;
        ranked.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

        let rank = ranked.iter().position(|(f, _)| *f == true_formula).map(|p| p + 1).unwrap_or(999);

        // Only show failures
        if rank == 1 {
            continue;
        }

        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("  COMPOUND: {} ({})", base_name, ionmode);
        println!("  Precursor m/z: {:.6}  |  Neutral mass: {:.6}  |  True exact mass: {:.6}", pepmass, neutral_mass, exact_mass);
        println!("  Mass error: {:.4} Da = {:.2} ppm", (neutral_mass - exact_mass).abs(), ((neutral_mass - exact_mass).abs() / neutral_mass) * 1e6);
        println!("  True formula: {}  |  Rank: #{}  |  Oracle: {}  |  Total candidates: {}", true_formula, rank, oracle, candidate_precursors.len());
        println!("  Clean peaks: {} (from {} raw)", clean_peaks.len(), record.peaks().len());
        println!();

        // Show top-3 ranked formulas vs true formula
        println!("  ── Top 3 Ranked Candidates ──");
        for (i, (formula, tree)) in ranked.iter().take(3).enumerate() {
            let mass_err_ppm = ((formula.exact_mass() - neutral_mass).abs() / neutral_mass) * 1e6;
            let prior = scorer.score_prior(formula);
            let is_correct = *formula == true_formula;
            let marker = if is_correct { " ← CORRECT" } else { "" };
            println!("  #{}: {} (mass err: {:.2} ppm, prior: {:.2}, tree: {:.2}, expl: {:.1}%, peaks: {}/{}){}",
                i + 1, formula, mass_err_ppm, prior, tree.total_score,
                tree.explained_intensity * 100.0, tree.explained_peaks, tree.total_peaks, marker);

            // Show tree structure
            for line in tree.format_tree().lines().take(12) {
                println!("      {}", line);
            }
            println!();
        }

        // If true formula is ranked, show its score too
        if oracle && rank > 3 && rank < 999 {
            let (ref tf, ref ttree) = ranked[rank - 1];
            let prior = scorer.score_prior(tf);
            let mass_err_ppm = ((tf.exact_mass() - neutral_mass).abs() / neutral_mass) * 1e6;
            println!("  ── True Formula (Rank #{}) ──", rank);
            println!("  #{}: {} (mass err: {:.2} ppm, prior: {:.2}, tree: {:.2}, expl: {:.1}%, peaks: {}/{})",
                rank, tf, mass_err_ppm, prior, ttree.total_score,
                ttree.explained_intensity * 100.0, ttree.explained_peaks, ttree.total_peaks);
            for line in ttree.format_tree().lines().take(8) {
                println!("      {}", line);
            }
            println!();
        }

        // Score breakdown: why does wrong beat right?
        if oracle {
            let (ref wrong_f, ref wrong_t) = ranked[0];
            let right_idx = rank - 1;
            let (ref right_f, ref right_t) = ranked[right_idx];

            let wrong_prior = scorer.score_prior(wrong_f);
            let right_prior = scorer.score_prior(right_f);

            println!("  ── Score Breakdown: Why #{} beats #{} ──", wrong_f, right_f);
            println!("    Component            | Wrong (#1)     | Correct (#{})", rank);
            println!("    ---------------------|----------------|----------------");
            println!("    Tree score            | {:>14.3} | {:>14.3}", wrong_t.total_score, right_t.total_score);
            println!("    Prior score            | {:>14.3} | {:>14.3}", wrong_prior, right_prior);
            println!("    Explained intensity    | {:>13.1}% | {:>13.1}%", wrong_t.explained_intensity * 100.0, right_t.explained_intensity * 100.0);
            println!("    Explained peaks        | {:>14} | {:>14}", wrong_t.explained_peaks, right_t.explained_peaks);
            println!("    Delta tree score       | {:>+14.3} |", wrong_t.total_score - right_t.total_score);
            println!("    Delta prior            | {:>+14.3} |", wrong_prior - right_prior);

            // Check element composition differences
            println!();
            println!("    Wrong formula elements: C={} H={} N={} O={} P={} S={}",
                wrong_f.c(), wrong_f.h(), wrong_f.n(), wrong_f.o(), wrong_f.p(), wrong_f.s());
            println!("    Correct formula elements: C={} H={} N={} O={} P={} S={}",
                right_f.c(), right_f.h(), right_f.n(), right_f.o(), right_f.p(), right_f.s());

            let wrong_has_p = wrong_f.p() > 0;
            let right_has_p = right_f.p() > 0;
            let wrong_has_s = wrong_f.s() > 0;
            let right_has_s = right_f.s() > 0;

            if wrong_has_p && !right_has_p {
                println!("    ⚠ SPURIOUS P: Wrong formula has P but correct formula doesn't!");
            }
            if wrong_has_s && !right_has_s {
                println!("    ⚠ SPURIOUS S: Wrong formula has S but correct formula doesn't!");
            }
        } else {
            println!("  ⚠ TRUE FORMULA NOT IN CANDIDATE LIST (Oracle=NO)");
            println!("    True formula {} exact mass: {:.6}", true_formula, exact_mass);
            println!("    Target neutral mass: {:.6}", neutral_mass);
            println!("    True elements: C={} H={} N={} O={} P={} S={} Cl={} Br={} F={} I={}",
                true_formula.c(), true_formula.h(), true_formula.n(), true_formula.o(),
                true_formula.p(), true_formula.s(), true_formula.cl(), true_formula.br(),
                true_formula.f(), true_formula.i());

            // Is the formula actually generable by CHNOPS?
            let has_halogens = true_formula.cl() > 0 || true_formula.br() > 0 || true_formula.f() > 0 || true_formula.i() > 0;
            if has_halogens {
                println!("    → HALOGENATED compound! CHNOPS generator cannot produce this formula.");
            }

            // Check if mass is within tolerance
            let mass_err_ppm = ((exact_mass - neutral_mass).abs() / neutral_mass) * 1e6;
            println!("    Mass error from precursor: {:.2} ppm", mass_err_ppm);
            if mass_err_ppm > 15.0 {
                println!("    → Mass error exceeds tolerance, adduct assumption might be wrong.");
            }

            // Check if rules reject it
            let valid = rules.is_valid(&true_formula);
            println!("    ChemicalRules validation: {}", if valid { "PASS" } else { "FAIL — rules reject this formula" });
        }

        println!();
    }

    println!("\nDiagnostic complete.");
}
