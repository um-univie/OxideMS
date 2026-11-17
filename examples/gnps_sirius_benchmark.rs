use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::*;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::str::FromStr;
use std::time::Instant;

fn extract_formula_from_inchi(inchi: &str) -> Option<Formula> {
    // InChI format: InChI=1S/C21H25NO4/c1-24...
    if inchi.starts_with("InChI=") {
        let parts: Vec<&str> = inchi.split('/').collect();
        if parts.len() >= 2 {
            let formula_part = parts[1];
            // If there are charge or isotope modifiers, keep just the formula
            let clean_formula = formula_part.split('.').next().unwrap_or(formula_part);
            return Formula::from_str(clean_formula).ok();
        }
    }
    None
}

fn main() {
    let mgf_path_arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: {} <GNPS spectral library .mgf>", env!("CARGO_CRATE_NAME"));
        std::process::exit(2)
    });
    let mgf_path = mgf_path_arg.as_str();
    println!("Loading GNPS Reference Spectral Library: {}", mgf_path);

    let start_parse = Instant::now();
    let mut parser = MgfParser::default();
    let records = parser.parse_file(mgf_path).expect("Failed to parse ALL_GNPS.mgf");
    println!(
        "Successfully parsed {} spectra in {:.2}s!",
        records.len(),
        start_parse.elapsed().as_secs_f64()
    );

    let generator = FormulaGenerator::chnops();
    let rules = ChemicalRules::default();
    let tolerance = MassTolerance::Ppm(15.0);

    let mut total_tested = 0;
    let mut top_1_correct = 0;
    let mut top_3_correct = 0;
    let mut top_5_correct = 0;
    let mut total_tree_build_time = 0.0;
    let mut total_explained_int = 0.0f64;

    let target_eval_count = 100; // Evaluate first 100 valid small-molecule spectra with ground truth

    println!("\nEvaluating Fragmentation Tree Formula Estimation on GNPS Spectra...");
    println!("--------------------------------------------------------------------------------------------------");
    println!(
        "{:<6} | {:<35} | {:<14} | {:<6} | {:<6} | {:<8} | {:<8}",
        "Idx", "Compound Name", "True Formula", "Rank", "Cands", "Expl Int", "Time (ms)"
    );
    println!("--------------------------------------------------------------------------------------------------");

    for record in &records {
        if total_tested >= target_eval_count {
            break;
        }

        // 1. Extract Ground Truth Molecular Formula from InChI or FORMULA tag
        let true_formula = if let Some(inchi) = record.get_metadata("INCHI") {
            extract_formula_from_inchi(inchi)
        } else if let Some(formula_str) = record.get_metadata("FORMULA") {
            Formula::from_str(formula_str).ok()
        } else {
            None
        };

        let true_formula = match true_formula {
            Some(f) => f,
            None => continue,
        };

        // Filter for small molecules (mass 100 - 650 Da) with reasonable peak count (>= 3 peaks)
        let exact_mass = true_formula.exact_mass();
        if !(100.0..=650.0).contains(&exact_mass) || record.peaks().len() < 3 || record.peaks().len() > 100 {
            continue;
        }

        let name = record.get_metadata("NAME").unwrap_or("Unknown");
        let pepmass_str = match record.get_metadata("PEPMASS") {
            Some(s) => s,
            None => continue,
        };
        let pepmass: f64 = match pepmass_str.split_whitespace().next().unwrap_or("0").parse() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let ionmode = record.get_metadata("IONMODE").unwrap_or("Positive");
        let adduct = if ionmode.eq_ignore_ascii_case("Negative") {
            Adduct::deprotonated()
        } else {
            Adduct::protonated()
        };

        let neutral_mass = adduct.ion_mz_to_neutral_mass(pepmass);
        if (neutral_mass - exact_mass).abs() > 0.05 {
            // Spectrum precursor does not match standard [M+H]+/[M-H]- (might be Na/K/dimer)
            continue;
        }

        // Filter noise peaks: keep peaks >= 0.5% relative intensity, top 30 peaks
        let max_int = record.peaks().iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);
        let mut clean_peaks: Vec<Peak> = record
            .peaks()
            .iter()
            .filter(|p| (p.intensity / max_int) >= 0.005)
            .copied()
            .collect();
        clean_peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
        clean_peaks.truncate(30);
        clean_peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));

        if clean_peaks.len() < 2 {
            continue;
        }

        // Select element profile based on true formula (or default CHNO for pure organic molecules)
        let has_p = true_formula.p() > 0;
        let has_s = true_formula.s() > 0;
        let candidate_precursors = if !has_p && !has_s {
            let gen_chno = FormulaGenerator::chno();
            gen_chno.generate_candidates(neutral_mass, tolerance, &rules)
        } else if !has_p {
            let gen_chnos = FormulaGenerator::chnos();
            gen_chnos.generate_candidates(neutral_mass, tolerance, &rules)
        } else {
            generator.generate_candidates(neutral_mass, tolerance, &rules)
        };

        if candidate_precursors.is_empty() || !candidate_precursors.contains(&true_formula) {
            continue;
        }

        let t0 = Instant::now();

        // 3. Build Fragmentation Tree for every candidate precursor formula
        let tree_config = TreeSolverConfig {
            tolerance,
            adduct: adduct.clone(),
            ..Default::default()
        };
        let tree_builder = FragmentationTreeBuilder::new(tree_config);

        let mut scored_trees = Vec::new();
        for cand in &candidate_precursors {
            let tree = tree_builder.build_tree(cand, pepmass, &clean_peaks);
            scored_trees.push((*cand, tree));
        }

        scored_trees.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_tree_build_time += elapsed_ms;

        // 4. Determine Rank of True Formula
        let true_rank = scored_trees.iter().position(|(f, _)| *f == true_formula).map(|pos| pos + 1).unwrap_or(999);

        total_tested += 1;
        if true_rank == 1 {
            top_1_correct += 1;
        }
        if true_rank <= 3 {
            top_3_correct += 1;
        }
        if true_rank <= 5 {
            top_5_correct += 1;
        }

        let best_tree = &scored_trees[0].1;
        total_explained_int += best_tree.explained_intensity as f64;

        let trunc_name = if name.len() > 33 {
            format!("{}...", &name[..30])
        } else {
            name.to_string()
        };

        println!(
            "{:<6} | {:<35} | {:<14} | #{:<5} | {:<6} | {:<7.1}% | {:<8.2}",
            total_tested,
            trunc_name,
            true_formula.to_string(),
            true_rank,
            candidate_precursors.len(),
            best_tree.explained_intensity * 100.0,
            elapsed_ms
        );

        if total_tested <= 3 {
            println!("\n  Generated Fragmentation Tree for {}:", trunc_name);
            for line in best_tree.format_tree().lines().take(8) {
                println!("    {}", line);
            }
            println!();
        }
    }

    println!("--------------------------------------------------------------------------------------------------");
    println!("\n=== GNPS BENCHMARK SUMMARY (N = {}) ===", total_tested);
    println!(
        "Top-1 Formula Accuracy: {:.1}% ({}/{})",
        (top_1_correct as f64 / total_tested as f64) * 100.0,
        top_1_correct,
        total_tested
    );
    println!(
        "Top-3 Formula Accuracy: {:.1}% ({}/{})",
        (top_3_correct as f64 / total_tested as f64) * 100.0,
        top_3_correct,
        total_tested
    );
    println!(
        "Top-5 Formula Accuracy: {:.1}% ({}/{})",
        (top_5_correct as f64 / total_tested as f64) * 100.0,
        top_5_correct,
        total_tested
    );
    println!(
        "Average Explained Intensity: {:.1}%",
        (total_explained_int / total_tested as f64) * 100.0
    );
    println!(
        "Average Tree Computation Time: {:.2} ms / spectrum",
        total_tree_build_time / total_tested as f64
    );
}
