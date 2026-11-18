//! Systematic ablation benchmark: Baseline → each improvement individually.
//!
//! Measures Top-1/3/5 accuracy, mean score gap, explained intensity, and timing
//! for the GNPS eval-100 reference spectra.

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

/// Ground truth: compound name → (formula_str, ionmode)
fn build_ground_truth() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut gt: HashMap<&str, (&str, &str)> = HashMap::new();
    // Positive mode compounds
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
    // Negative mode compounds
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

/// Prepare clean peaks from a record (standard decontamination pipeline)
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

/// Results for a single spectrum evaluation
#[derive(Debug, Clone)]
struct SpecResult {
    name: String,
    true_formula: Formula,
    predicted_formula: Formula,
    rank: usize,          // 1-indexed rank of true formula (999 if not found)
    top1_score: f64,
    top2_score: f64,
    score_gap: f64,
    explained_intensity: f32,
    explained_peaks: usize,
    total_peaks: usize,
    n_candidates: usize,
    time_ms: f64,
    oracle_hit: bool,     // was true formula in candidate list at all?
}

/// Aggregate metrics over a set of spectrum results
#[derive(Debug, Clone)]
struct AggregateMetrics {
    label: String,
    n: usize,
    top1: usize,
    top3: usize,
    top5: usize,
    oracle: usize,
    mean_score_gap: f64,
    mean_explained_int: f64,
    mean_time_ms: f64,
    total_time_ms: f64,
}

impl AggregateMetrics {
    fn from_results(label: &str, results: &[SpecResult]) -> Self {
        let n = results.len();
        let top1 = results.iter().filter(|r| r.rank == 1).count();
        let top3 = results.iter().filter(|r| r.rank <= 3).count();
        let top5 = results.iter().filter(|r| r.rank <= 5).count();
        let oracle = results.iter().filter(|r| r.oracle_hit).count();
        let mean_score_gap = results.iter().map(|r| r.score_gap).sum::<f64>() / n.max(1) as f64;
        let mean_explained = results.iter().map(|r| r.explained_intensity as f64).sum::<f64>() / n.max(1) as f64;
        let total_time: f64 = results.iter().map(|r| r.time_ms).sum();
        let mean_time = total_time / n.max(1) as f64;
        Self {
            label: label.to_string(),
            n,
            top1,
            top3,
            top5,
            oracle,
            mean_score_gap,
            mean_explained_int: mean_explained,
            mean_time_ms: mean_time,
            total_time_ms: total_time,
        }
    }

    fn print(&self) {
        let pct = |x: usize| (x as f64 / self.n.max(1) as f64) * 100.0;
        println!("  {:<36} | Top-1: {:>5.1}% ({:>2}/{:>2}) | Top-3: {:>5.1}% | Top-5: {:>5.1}% | Oracle: {:>5.1}% | ScoreGap: {:>5.2} | ExplInt: {:>5.1}% | {:.2} ms/sp",
            self.label,
            pct(self.top1), self.top1, self.n,
            pct(self.top3),
            pct(self.top5),
            pct(self.oracle),
            self.mean_score_gap,
            self.mean_explained_int * 100.0,
            self.mean_time_ms,
        );
    }
}

/// Run a single configuration variant and return per-spectrum results.
///
/// `config_name`: label for this configuration
/// `tolerance_ppm`: mass tolerance in ppm
/// `max_candidates`: max candidate formulas to evaluate with tree search
/// `max_peaks`: max peaks per tree
/// `use_chnops_only`: if true, CHNOPS generator; else match to ground truth elements
/// `noise_floor_override`: if Some(x), override the implicit noise floor
fn run_evaluation(
    config_name: &str,
    records: &[oxidems_lib::spectrum::MassSpectralRecord],
    ground_truth: &HashMap<&str, (&str, &str)>,
    tolerance_ppm: f64,
    max_candidates: usize,
    max_peaks: usize,
    scorer_ppm: f64,
) -> Vec<SpecResult> {
    let tolerance = MassTolerance::Ppm(tolerance_ppm);
    let rules = ChemicalRules::default();
    let gen_chnops = FormulaGenerator::chnops();
    let gen_chno = FormulaGenerator::chno();
    let gen_chnos = FormulaGenerator::chnos();

    let mut results = Vec::new();

    for record in records {
        let raw_name = match record.get_metadata("NAME") {
            Some(n) => n,
            None => continue,
        };
        // Match ground truth: try exact name, then try stripping " M+H" / " M-H" suffix
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

        // Skip if precursor doesn't match standard adduct within 0.05 Da
        if (neutral_mass - exact_mass).abs() > 0.05 {
            continue;
        }

        // Skip if too few peaks
        if record.peaks().len() < 3 {
            continue;
        }

        let clean_peaks = clean_spectrum(record.peaks(), pepmass, max_peaks);
        if clean_peaks.len() < 2 {
            continue;
        }

        // Generate candidates - use element profile matching ground truth
        let has_s = true_formula.s() > 0;
        let has_p = true_formula.p() > 0;
        let has_halogens = true_formula.cl() > 0 || true_formula.br() > 0
            || true_formula.f() > 0 || true_formula.i() > 0;

        // Tiered generation: generate CHNO first, then merge CHNOPS candidates
        let mut candidate_precursors = gen_chno.generate_candidates(neutral_mass, tolerance, &rules);
        let chnops_cands = gen_chnops.generate_candidates(neutral_mass, tolerance, &rules);
        for c in chnops_cands {
            if !candidate_precursors.contains(&c) {
                candidate_precursors.push(c);
            }
        }

        let oracle_hit = candidate_precursors.contains(&true_formula);

        // Truncate to max_candidates if needed (using mass error sorting as proxy for pre-tree ranking)
        if candidate_precursors.len() > max_candidates {
            let scorer = FormulaScorer::new(scorer_ppm, 0.05);
            candidate_precursors.sort_by(|a, b| {
                let err_a = (a.exact_mass() - neutral_mass).abs();
                let err_b = (b.exact_mass() - neutral_mass).abs();
                let prior_a = scorer.score_prior(a);
                let prior_b = scorer.score_prior(b);
                let score_a = prior_a - (err_a / neutral_mass) * 1e6;
                let score_b = prior_b - (err_b / neutral_mass) * 1e6;
                score_b.total_cmp(&score_a)
            });
            candidate_precursors.truncate(max_candidates);
        }

        let t0 = Instant::now();

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

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let rank = ranked
            .iter()
            .position(|(f, _)| *f == true_formula)
            .map(|p| p + 1)
            .unwrap_or(999);

        let top1_score = ranked.first().map(|(_, t)| t.total_score).unwrap_or(0.0);
        let top2_score = ranked.get(1).map(|(_, t)| t.total_score).unwrap_or(0.0);
        let score_gap = top1_score - top2_score;

        let best_tree = ranked.first().map(|(_, t)| t);
        let expl_int = best_tree.map(|t| t.explained_intensity).unwrap_or(0.0);
        let expl_peaks = best_tree.map(|t| t.explained_peaks).unwrap_or(0);
        let total_peaks = best_tree.map(|t| t.total_peaks).unwrap_or(0);

        let predicted = ranked.first().map(|(f, _)| *f).unwrap_or(Formula::empty());

        results.push(SpecResult {
            name: base_name.to_string(),
            true_formula,
            predicted_formula: predicted,
            rank,
            top1_score,
            top2_score,
            score_gap,
            explained_intensity: expl_int,
            explained_peaks: expl_peaks,
            total_peaks,
            n_candidates: candidate_precursors.len(),
            time_ms: elapsed_ms,
            oracle_hit,
        });
    }

    results
}

mod support;

fn main() {
    support::require("ablation_improvements", &[("test_files/gnps_eval_100.mgf", support::GNPS_EVAL)]);
    println!("================================================================================");
    println!("     OxideMS SIRIUS: Systematic Improvement Ablation Benchmark");
    println!("================================================================================");
    println!();

    let ground_truth = build_ground_truth();

    let mut parser = MgfParser::default();
    let records = parser
        .parse_file("test_files/gnps_eval_100.mgf")
        .expect("Failed to parse gnps_eval_100.mgf");
    println!("Loaded {} spectra from gnps_eval_100.mgf", records.len());
    println!("Ground truth entries: {}", ground_truth.len());
    println!();

    // ──────────────────────────────────────────────────
    //  Baseline: Current defaults
    // ──────────────────────────────────────────────────
    println!("Running evaluations...\n");

    let baseline = run_evaluation(
        "Baseline",
        &records,
        &ground_truth,
        15.0,   // tolerance_ppm
        15,     // max_candidates
        30,     // max_peaks
        15.0,   // scorer_ppm
    );
    let baseline_metrics = AggregateMetrics::from_results("[A] BASELINE (defaults)", &baseline);

    // ──────────────────────────────────────────────────
    //  Improvement 2: Dynamic candidate budget (25 instead of 15)
    // ──────────────────────────────────────────────────
    let imp2 = run_evaluation(
        "MaxCandidates=25",
        &records,
        &ground_truth,
        15.0,
        25,     // ← increased from 15
        30,
        15.0,
    );
    let imp2_metrics = AggregateMetrics::from_results("[B] max_candidates: 15 → 25", &imp2);

    // ──────────────────────────────────────────────────
    //  Improvement 2b: Even more candidates (40)
    // ──────────────────────────────────────────────────
    let imp2b = run_evaluation(
        "MaxCandidates=40",
        &records,
        &ground_truth,
        15.0,
        40,     // ← 40
        30,
        15.0,
    );
    let imp2b_metrics = AggregateMetrics::from_results("[C] max_candidates: 15 → 40", &imp2b);

    // ──────────────────────────────────────────────────
    //  Improvement 2c: No truncation at all (100)
    // ──────────────────────────────────────────────────
    let imp2c = run_evaluation(
        "MaxCandidates=100 (no trunc)",
        &records,
        &ground_truth,
        15.0,
        100,
        30,
        15.0,
    );
    let imp2c_metrics = AggregateMetrics::from_results("[D] max_candidates: 15 → 100 (oracle)", &imp2c);

    // ──────────────────────────────────────────────────
    //  Improvement: Tighter tolerance (10 ppm)
    // ──────────────────────────────────────────────────
    let imp_tol10 = run_evaluation(
        "Tolerance=10ppm",
        &records,
        &ground_truth,
        10.0,   // ← tighter
        15,
        30,
        10.0,
    );
    let imp_tol10_metrics = AggregateMetrics::from_results("[E] tolerance: 15 → 10 ppm", &imp_tol10);

    // ──────────────────────────────────────────────────
    //  Improvement: Wider tolerance (20 ppm)
    // ──────────────────────────────────────────────────
    let imp_tol20 = run_evaluation(
        "Tolerance=20ppm",
        &records,
        &ground_truth,
        20.0,   // ← wider
        15,
        30,
        20.0,
    );
    let imp_tol20_metrics = AggregateMetrics::from_results("[F] tolerance: 15 → 20 ppm", &imp_tol20);

    // ──────────────────────────────────────────────────
    //  Improvement: More peaks per tree (50)
    // ──────────────────────────────────────────────────
    let imp_peaks50 = run_evaluation(
        "MaxPeaks=50",
        &records,
        &ground_truth,
        15.0,
        15,
        50,     // ← more peaks
        15.0,
    );
    let imp_peaks50_metrics = AggregateMetrics::from_results("[G] max_peaks: 30 → 50", &imp_peaks50);

    // ──────────────────────────────────────────────────
    //  Improvement: Fewer peaks per tree (20)
    // ──────────────────────────────────────────────────
    let imp_peaks20 = run_evaluation(
        "MaxPeaks=20",
        &records,
        &ground_truth,
        15.0,
        15,
        20,     // ← fewer peaks
        15.0,
    );
    let imp_peaks20_metrics = AggregateMetrics::from_results("[H] max_peaks: 30 → 20", &imp_peaks20);

    // ──────────────────────────────────────────────────
    //  Combined: 25 candidates + tighter scorer
    // ──────────────────────────────────────────────────
    let combined = run_evaluation(
        "Combined: 25 cands + 10ppm scorer",
        &records,
        &ground_truth,
        15.0,   // keep 15 ppm tolerance for generation
        25,     // more candidates
        30,
        10.0,   // tighter scorer
    );
    let combined_metrics = AggregateMetrics::from_results("[I] COMBINED: 25 cands + 10ppm scorer", &combined);

    // ──────────────────────────────────────────────────
    //  Print all results
    // ──────────────────────────────────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("                                                ABLATION RESULTS SUMMARY                                                                          ");
    println!("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!();

    baseline_metrics.print();
    println!("  ---");
    imp2_metrics.print();
    imp2b_metrics.print();
    imp2c_metrics.print();
    println!("  ---");
    imp_tol10_metrics.print();
    imp_tol20_metrics.print();
    println!("  ---");
    imp_peaks50_metrics.print();
    imp_peaks20_metrics.print();
    println!("  ---");
    combined_metrics.print();

    println!();
    println!("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");

    // ──────────────────────────────────────────────────
    //  Detailed failure analysis for baseline
    // ──────────────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────");
    println!("   FAILURE ANALYSIS (Baseline)");
    println!("──────────────────────────────────────────────────────");

    let mut failures: Vec<&SpecResult> = baseline.iter().filter(|r| r.rank != 1).collect();
    failures.sort_by_key(|r| r.rank);

    if failures.is_empty() {
        println!("  No failures! Perfect top-1 accuracy.");
    } else {
        println!("  {:<25} | {:<14} | {:<14} | Rank | ScoreGap | Cands | Oracle | ExplInt",
            "Compound", "True Formula", "Predicted", );
        println!("  {}", "-".repeat(120));
        for r in &failures {
            let oracle_str = if r.oracle_hit { "YES" } else { "NO" };
            println!("  {:<25} | {:<14} | {:<14} | #{:<3} | {:>8.2} | {:>5} | {:<6} | {:.1}%",
                &r.name[..r.name.len().min(25)],
                r.true_formula.to_string(),
                r.predicted_formula.to_string(),
                r.rank,
                r.score_gap,
                r.n_candidates,
                oracle_str,
                r.explained_intensity * 100.0,
            );
        }
    }

    // ──────────────────────────────────────────────────
    //  Check: Does more candidates rescue failures?
    // ──────────────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────");
    println!("   RESCUE ANALYSIS: Which failures are fixed by more candidates?");
    println!("──────────────────────────────────────────────────────");

    for fail in &failures {
        let rescued_in_25 = imp2.iter().find(|r| r.name == fail.name).map(|r| r.rank).unwrap_or(999);
        let rescued_in_40 = imp2b.iter().find(|r| r.name == fail.name).map(|r| r.rank).unwrap_or(999);
        let rescued_in_100 = imp2c.iter().find(|r| r.name == fail.name).map(|r| r.rank).unwrap_or(999);

        if rescued_in_25 == 1 || rescued_in_40 == 1 || rescued_in_100 == 1 {
            println!("  ✓ {} : baseline rank #{} → 25-cands #{} → 40-cands #{} → 100-cands #{}",
                fail.name, fail.rank, rescued_in_25, rescued_in_40, rescued_in_100);
        } else {
            println!("  ✗ {} : NOT rescued (rank stays #{} → #{} → #{} → #{})",
                fail.name, fail.rank, rescued_in_25, rescued_in_40, rescued_in_100);
        }
    }

    println!();
    println!("Done.");
}
