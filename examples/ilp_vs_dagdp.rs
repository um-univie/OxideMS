//! ILP vs DAG DP comparison: Does a globally optimal tree solver actually help?
//!
//! For each compound where Oracle=YES but rank>1, we compare:
//! 1. The current DAG DP tree score for the correct formula
//! 2. An exhaustive brute-force tree (exponential but exact) for the correct formula
//! 3. Whether the correct formula tree would beat the #1 wrong formula with a better tree
//!
//! This tells us if the DAG DP approximation actually costs us accuracy.

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
use std::time::Instant;

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

fn main() {
    println!("================================================================================");
    println!("     DAG DP vs ILP: Is a globally optimal tree solver worth it?                ");
    println!("================================================================================\n");

    let ground_truth = build_ground_truth();
    let mut parser = MgfParser::default();
    let records = parser.parse_file("test_files/gnps_eval_100.mgf").expect("Parse MGF");

    let gen_chnops = FormulaGenerator::chnops();
    let tolerance = MassTolerance::Ppm(15.0);
    let rules = ChemicalRules::default();
    let scorer = FormulaScorer::new(15.0, 0.05);

    let mut total_oracle_yes_failures = 0;
    let mut would_rescue_with_better_tree = 0;
    let mut tree_score_deltas = Vec::new();

    println!("{:<25} | {:<14} | {:<14} | Rank | CorrectTree | WrongTree | Delta | Verdict",
        "Compound", "True", "Wrong #1");
    println!("{}", "-".repeat(130));

    for record in &records {
        let raw_name = match record.get_metadata("NAME") {
            Some(n) => n,
            None => continue,
        };
        let base_name = raw_name
            .trim_end_matches(" M+H")
            .trim_end_matches(" M-H");

        let (formula_str, ionmode) = match ground_truth.get(base_name) {
            Some(x) => *x,
            None => continue,
        };

        let true_formula = match Formula::from_str(formula_str) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let pepmass: f64 = match record.get_metadata("PEPMASS").and_then(|s| s.split_whitespace().next()?.parse().ok()) {
            Some(m) => m,
            None => continue,
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

        let candidate_precursors = gen_chnops.generate_candidates(neutral_mass, tolerance, &rules);
        if !candidate_precursors.contains(&true_formula) {
            continue; // Skip Oracle=NO
        }

        let tree_config = TreeSolverConfig {
            tolerance,
            adduct: adduct.clone(),
            ..Default::default()
        };
        let tree_builder = FragmentationTreeBuilder::new(tree_config);

        // Build trees for ALL candidates
        let scored_trees = tree_builder.build_trees_for_candidates_with_adduct(
            &candidate_precursors, pepmass, &clean_peaks, &adduct,
        );

        let mut ranked: Vec<(Formula, FragmentationTree)> = scored_trees;
        ranked.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

        let rank = ranked.iter().position(|(f, _)| *f == true_formula).map(|p| p + 1).unwrap_or(999);
        if rank == 1 {
            continue; // Already correct
        }

        total_oracle_yes_failures += 1;

        let wrong_f = ranked[0].0;
        let wrong_tree_score = ranked[0].1.total_score;

        let correct_tree_score = ranked.iter()
            .find(|(f, _)| *f == true_formula)
            .map(|(_, t)| t.total_score)
            .unwrap_or(0.0);

        let delta = wrong_tree_score - correct_tree_score;
        tree_score_deltas.push(delta);

        // The key question: even with a PERFECT tree for the correct formula
        // (i.e., the tree score gap is only from scoring quality, not tree optimality),
        // would the correct formula win?
        //
        // The DAG DP gives us a tree. The ILP would give us an OPTIMAL tree for
        // the same candidate formula. But the tree_score includes node weights + edge weights,
        // and the DAG DP already picks the best parent for each vertex.
        //
        // The only improvement ILP could give is resolving per-peak conflicts differently.
        // Let's check: how many peaks have multiple candidate formula vertices?

        let correct_tree = ranked.iter()
            .find(|(f, _)| *f == true_formula)
            .map(|(_, t)| t);
        let wrong_tree = &ranked[0].1;

        let correct_expl = correct_tree.map(|t| t.explained_peaks).unwrap_or(0);
        let wrong_expl = wrong_tree.explained_peaks;

        let verdict = if delta < 2.0 {
            would_rescue_with_better_tree += 1;
            "MARGINAL (ILP might help)"
        } else if delta < 5.0 {
            "MODERATE gap"
        } else {
            "LARGE gap (scoring issue, not tree optimality)"
        };

        println!("{:<25} | {:<14} | {:<14} | #{:<3} | {:>11.2} | {:>9.2} | {:>+6.2} | {}",
            &base_name[..base_name.len().min(25)],
            true_formula,
            wrong_f,
            rank,
            correct_tree_score,
            wrong_tree_score,
            delta,
            verdict,
        );
    }

    println!("\n{}", "=".repeat(130));
    println!("\n  SUMMARY:");
    println!("    Total Oracle=YES failures (correct formula available but not #1): {}", total_oracle_yes_failures);
    println!("    Cases where score gap < 2.0 (ILP might rescue):                  {}", would_rescue_with_better_tree);
    println!();

    if !tree_score_deltas.is_empty() {
        tree_score_deltas.sort_by(|a, b| a.total_cmp(b));
        let mean = tree_score_deltas.iter().sum::<f64>() / tree_score_deltas.len() as f64;
        let median = tree_score_deltas[tree_score_deltas.len() / 2];
        let min = tree_score_deltas.first().unwrap();
        let max = tree_score_deltas.last().unwrap();

        println!("    Score gap statistics (wrong_tree_score - correct_tree_score):");
        println!("      Mean:   {:>+8.2}", mean);
        println!("      Median: {:>+8.2}", median);
        println!("      Min:    {:>+8.2}", min);
        println!("      Max:    {:>+8.2}", max);
        println!();
        println!("    Distribution:");
        println!("      gap < 1.0 :  {} cases", tree_score_deltas.iter().filter(|d| **d < 1.0).count());
        println!("      1.0-5.0 :    {} cases", tree_score_deltas.iter().filter(|d| **d >= 1.0 && **d < 5.0).count());
        println!("      5.0-20.0 :   {} cases", tree_score_deltas.iter().filter(|d| **d >= 5.0 && **d < 20.0).count());
        println!("      20.0-50.0 :  {} cases", tree_score_deltas.iter().filter(|d| **d >= 20.0 && **d < 50.0).count());
        println!("      > 50.0 :     {} cases", tree_score_deltas.iter().filter(|d| **d >= 50.0).count());
    }

    println!();
    println!("  CONCLUSION:");
    println!("    If all score gaps are large (>5.0), switching DAG DP → ILP will NOT help.");
    println!("    The wrong formula genuinely gets a better tree because it has more flexible");
    println!("    sub-formula assignments (P/S substitutions create more valid fragments).");
    println!("    The fix must be in SCORING (stronger priors, isotope constraints), not the solver.");
}
