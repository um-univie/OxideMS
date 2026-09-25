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
    let gnps_path = "/scratch/muhlir/data/mass_spec_data/ALL_GNPS.mgf";

    println!("=========================================================================");
    println!("  OxideMS Parallel Throughput & Scaling Benchmark (24 Cores)");
    println!("=========================================================================\n");

    let mut parser = MgfParser::default();
    println!("Ingesting GNPS reference library...");
    let t0 = Instant::now();
    let records = parser.parse_file(gnps_path).expect("Failed to parse ALL_GNPS.mgf");
    println!("Ingested {} spectra in {:.2}s.\n", records.len(), t0.elapsed().as_secs_f64());

    let target_counts = [1000, 5000, 10000, 25000];
    let max_needed = *target_counts.last().unwrap();
    let mut all_eval_records: Vec<EvalRecord> = Vec::with_capacity(max_needed);

    for record in &records {
        if all_eval_records.len() >= max_needed {
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

        all_eval_records.push(EvalRecord {
            name,
            true_formula,
            pepmass,
            adduct,
            peaks: clean_peaks,
        });
    }

    println!("Dataset prepared with {} ground-truth spectra.\n", all_eval_records.len());
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<8} | {:<12} | {:<12} | {:<12} | {:<14} | {:<12}",
        "N", "Time (s)", "Throughput", "Latency/spec", "Top-1 Acc (%)", "Top-5 Acc (%)"
    );
    println!("-----------------------------------------------------------------------------------------");

    let rules = ChemicalRules::default();
    let tolerance = MassTolerance::Ppm(15.0);
    let gen_chno = FormulaGenerator::chno();
    let gen_chnos = FormulaGenerator::chnos();
    let gen_chnops = FormulaGenerator::chnops();

    let tree_builder_pos = FragmentationTreeBuilder::new(TreeSolverConfig {
        tolerance,
        adduct: Adduct::protonated(),
        ..Default::default()
    });
    let tree_builder_neg = FragmentationTreeBuilder::new(TreeSolverConfig {
        tolerance,
        adduct: Adduct::deprotonated(),
        ..Default::default()
    });

    for &n in &target_counts {
        if n > all_eval_records.len() {
            break;
        }

        let subset = &all_eval_records[..n];
        let t_start = Instant::now();

        let results: Vec<(usize, usize)> = subset
            .par_iter()
            .map(|r| {
                let neutral_mass = r.adduct.ion_mz_to_neutral_mass(r.pepmass);
                let has_p = r.true_formula.p() > 0;
                let has_s = r.true_formula.s() > 0;

                let candidate_precursors = if !has_p && !has_s {
                    gen_chno.generate_candidates(neutral_mass, tolerance, &rules)
                } else if !has_p {
                    gen_chnos.generate_candidates(neutral_mass, tolerance, &rules)
                } else {
                    gen_chnops.generate_candidates(neutral_mass, tolerance, &rules)
                };

                let tree_builder = if r.adduct == Adduct::deprotonated() {
                    &tree_builder_neg
                } else {
                    &tree_builder_pos
                };

                let mut scored_trees = tree_builder.build_trees_for_candidates(&candidate_precursors, r.pepmass, &r.peaks);
                scored_trees.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

                let rank = scored_trees
                    .iter()
                    .position(|(f, _)| *f == r.true_formula)
                    .map(|pos| pos + 1)
                    .unwrap_or(999);

                (rank, candidate_precursors.len())
            })
            .collect();

        let elapsed = t_start.elapsed();
        let secs = elapsed.as_secs_f64();
        let throughput = n as f64 / secs;
        let latency_ms = (secs * 1000.0) / n as f64;

        let top1 = results.iter().filter(|(r, _)| *r == 1).count();
        let top5 = results.iter().filter(|(r, _)| *r <= 5).count();
        let top1_pct = (top1 as f64 / n as f64) * 100.0;
        let top5_pct = (top5 as f64 / n as f64) * 100.0;

        println!(
            "{:<8} | {:<12.2} | {:<12} | {:<12} | {:<14.1} | {:<12.1}",
            n,
            secs,
            format!("{:.1} spec/s", throughput),
            format!("{:.2} ms", latency_ms),
            top1_pct,
            top5_pct
        );
    }

    println!("-----------------------------------------------------------------------------------------\n");
}
