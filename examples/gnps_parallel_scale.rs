use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::*;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::str::FromStr;
use std::time::Instant;

fn extract_formula_from_inchi(inchi: &str) -> Option<Formula> {
    if inchi.starts_with("InChI=") {
        let parts: Vec<&str> = inchi.split('/').collect();
        if parts.len() >= 2 {
            let formula_part = parts[1];
            let clean_formula = formula_part.split('.').next().unwrap_or(formula_part);
            return Formula::from_str(clean_formula).ok();
        }
    }
    None
}

struct EvalRecord {
    pub name: String,
    pub true_formula: Formula,
    pub pepmass: f64,
    pub adduct: Adduct,
    pub peaks: Vec<Peak>,
}

fn main() {
    let gnps_path_arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: {} <GNPS spectral library .mgf>", env!("CARGO_CRATE_NAME"));
        std::process::exit(2)
    });
    let gnps_path = gnps_path_arg.as_str();

    println!("=========================================================================");
    println!("  OxideMS Parallel GNPS Scale Benchmark (24 Cores)");
    println!("=========================================================================\n");

    let mut parser = MgfParser::default();
    println!("Ingesting GNPS reference library (4.3 GB)...");
    let t0 = Instant::now();
    let records = parser.parse_file(gnps_path).expect("Failed to parse ALL_GNPS.mgf");
    println!("Ingested {} spectra in {:.2}s.\n", records.len(), t0.elapsed().as_secs_f64());

    let target_count = 500;
    let mut eval_records: Vec<EvalRecord> = Vec::new();

    for record in &records {
        if eval_records.len() >= target_count {
            break;
        }

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

        let exact_mass = true_formula.exact_mass();
        if !(100.0..=650.0).contains(&exact_mass) || record.peaks().len() < 3 || record.peaks().len() > 100 {
            continue;
        }

        let name = record.get_metadata("NAME").unwrap_or("Unknown").to_string();
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
            continue;
        }

        // Clean noise peaks: keep >= 0.5% relative intensity, top 30
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

        eval_records.push(EvalRecord {
            name,
            true_formula,
            pepmass,
            adduct,
            peaks: clean_peaks,
        });
    }

    println!("Evaluating {} GNPS spectra across 24 parallel worker threads...\n", eval_records.len());

    let rules = ChemicalRules::default();
    let tolerance = MassTolerance::Ppm(15.0);

    let t_solve = Instant::now();
    let results: Vec<(usize, usize, f64)> = eval_records
        .par_iter()
        .map(|r| {
            let neutral_mass = r.adduct.ion_mz_to_neutral_mass(r.pepmass);
            let has_p = r.true_formula.p() > 0;
            let has_s = r.true_formula.s() > 0;

            let candidate_precursors = if !has_p && !has_s {
                let gen_chno = FormulaGenerator::chno();
                gen_chno.generate_candidates(neutral_mass, tolerance, &rules)
            } else if !has_p {
                let gen_chnos = FormulaGenerator::chnos();
                gen_chnos.generate_candidates(neutral_mass, tolerance, &rules)
            } else {
                let gen_chnops = FormulaGenerator::chnops();
                gen_chnops.generate_candidates(neutral_mass, tolerance, &rules)
            };

            let tree_config = TreeSolverConfig {
                tolerance,
                adduct: r.adduct.clone(),
                ..Default::default()
            };
            let tree_builder = FragmentationTreeBuilder::new(tree_config);

            let mut scored_trees = tree_builder.build_trees_for_candidates(&candidate_precursors, r.pepmass, &r.peaks);

            scored_trees.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

            let rank = scored_trees
                .iter()
                .position(|(f, _)| *f == r.true_formula)
                .map(|pos| pos + 1)
                .unwrap_or(999);

            let expl_int = scored_trees.first().map(|(_, t)| t.explained_intensity as f64).unwrap_or(0.0);
            (rank, candidate_precursors.len(), expl_int)
        })
        .collect();

    let total_time = t_solve.elapsed();

    let n = eval_records.len();
    let mut top1 = 0;
    let mut top3 = 0;
    let mut top5 = 0;
    let mut total_expl_int = 0.0;
    let mut total_cands = 0;

    for &(rank, num_cands, expl_int) in &results {
        if rank == 1 { top1 += 1; }
        if rank <= 3 { top3 += 1; }
        if rank <= 5 { top5 += 1; }
        total_expl_int += expl_int;
        total_cands += num_cands;
    }

    println!("=========================================================================");
    println!("              LARGE-SCALE GNPS BENCHMARK RESULTS (N = {})", n);
    println!("=========================================================================");
    println!("Top-1 Formula Accuracy:        {:.1}% ({}/{})", (top1 as f64 / n as f64) * 100.0, top1, n);
    println!("Top-3 Formula Accuracy:        {:.1}% ({}/{})", (top3 as f64 / n as f64) * 100.0, top3, n);
    println!("Top-5 Formula Accuracy:        {:.1}% ({}/{})", (top5 as f64 / n as f64) * 100.0, top5, n);
    println!("Average Candidate Space:       {:.1} formulas / precursor", total_cands as f64 / n as f64);
    println!("Average Explained Intensity:   {:.1}%", (total_expl_int / n as f64) * 100.0);
    println!("Total Tree Build Time:         {:.2} s (across 24 cores)", total_time.as_secs_f64());
    println!("Throughput:                    {:.1} spectra / sec ({:.2} ms / spectrum)", n as f64 / total_time.as_secs_f64(), (total_time.as_secs_f64() * 1000.0) / n as f64);
    println!("=========================================================================\n");
}
