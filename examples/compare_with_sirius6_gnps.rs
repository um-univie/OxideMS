use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::*;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
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
    pub mgf_raw: String,
}

fn main() {
    let gnps_path = "/scratch/muhlir/data/mass_spec_data/ALL_GNPS.mgf";
    let subset_mgf = "test_files/gnps_eval_100.mgf";
    let sirius_out_dir = "scratch/sirius6_eval_out";

    println!("=========================================================================");
    println!("  OxideMS vs SIRIUS 6.0.5 Real MS/MS Benchmark (GNPS Library)");
    println!("=========================================================================\n");

    // 1. Extract 100 diverse evaluation spectra with ground truth from GNPS
    let mut parser = MgfParser::default();
    println!("Step 1: Ingesting GNPS reference library...");
    let t0 = Instant::now();
    let records = parser.parse_file(gnps_path).expect("Failed to parse ALL_GNPS.mgf");
    println!("Ingested {} spectra in {:.2}s.\n", records.len(), t0.elapsed().as_secs_f64());

    let target_count = 100;
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

        // Clean peaks
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

        // Generate MGF record string for SIRIUS 6
        let mut mgf_raw = format!(
            "BEGIN IONS\nNAME={}\nPEPMASS={:.6}\nCHARGE={}\nIONMODE={}\n",
            name.replace('\n', " "),
            pepmass,
            if ionmode.eq_ignore_ascii_case("Negative") { "-1" } else { "1" },
            ionmode
        );
        for p in &clean_peaks {
            mgf_raw.push_str(&format!("{:.6} {:.1}\n", p.mz, p.intensity));
        }
        mgf_raw.push_str("END IONS\n\n");

        eval_records.push(EvalRecord {
            name,
            true_formula,
            pepmass,
            adduct,
            peaks: clean_peaks,
            mgf_raw,
        });
    }

    // Write subset MGF for SIRIUS 6
    let _ = fs::create_dir_all("test_files");
    let mut mgf_file = File::create(subset_mgf).expect("Failed to create subset MGF");
    for r in &eval_records {
        mgf_file.write_all(r.mgf_raw.as_bytes()).unwrap();
    }
    println!("Step 2: Created benchmark MGF with {} spectra at '{}'", eval_records.len(), subset_mgf);

    // =========================================================================
    // 2. Run OxideMS in Parallel (Rayon)
    // =========================================================================
    println!("\nStep 3: Running OxideMS Parallel Fragmentation Tree Solver (24 Cores)...");
    let rules = ChemicalRules::default();
    let tolerance = MassTolerance::Ppm(15.0);

    let t_oxidems = Instant::now();
    let oxidems_results: Vec<(usize, usize, f64)> = eval_records
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

            let mut scored_trees = Vec::new();
            for cand in &candidate_precursors {
                let tree = tree_builder.build_tree(cand, r.pepmass, &r.peaks);
                scored_trees.push((*cand, tree));
            }

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

    let oxidems_time = t_oxidems.elapsed();
    println!("OxideMS finished in {:.2}s ({:.2} ms / spectrum)!", oxidems_time.as_secs_f64(), (oxidems_time.as_secs_f64() * 1000.0) / eval_records.len() as f64);

    // =========================================================================
    // 3. Run SIRIUS 6.0.5 CLI
    // =========================================================================
    println!("\nStep 4: Running official SIRIUS 6.0.5 on the same dataset...");
    let _ = fs::remove_dir_all(sirius_out_dir);

    let t_sirius = Instant::now();
    let sirius_bin = "/scratch/muhlir/bin/sirius/bin/sirius";
    let sirius_output = Command::new(sirius_bin)
        .arg("-i")
        .arg(subset_mgf)
        .arg("-o")
        .arg(sirius_out_dir)
        .arg("--ignore-formula")
        .arg("formula")
        .arg("-c=10")
        .arg("--ppm-max=15")
        .arg("--ppm-max-ms2=15")
        .output()
        .expect("Failed to execute sirius CLI");

    let sirius_time = t_sirius.elapsed();
    println!("SIRIUS 6 finished in {:.2}s ({:.2} ms / spectrum)!", sirius_time.as_secs_f64(), (sirius_time.as_secs_f64() * 1000.0) / eval_records.len() as f64);

    // 4. Parse SIRIUS 6 Output
    // SIRIUS 6 creates formula_identifications.tsv or subdirectories under sirius_out_dir
    let mut sirius_ranks = vec![999; eval_records.len()];
    let summary_tsv = format!("{}/formula_identifications.tsv", sirius_out_dir);

    if let Ok(file) = File::open(&summary_tsv) {
        let reader = BufReader::new(file);
        for line in reader.lines().flatten() {
            let cols: Vec<&str> = line.split('\t').collect();
            // Expected columns: id/compound, formula, rank, score...
            if cols.len() >= 3 && cols[0] != "id" {
                if let Ok(idx) = cols[0].parse::<usize>() {
                    if idx < eval_records.len() {
                        let cand_formula_str = cols[1];
                        if let Ok(cand_f) = Formula::from_str(cand_formula_str) {
                            if cand_f == eval_records[idx].true_formula {
                                if let Ok(rank) = cols[2].parse::<usize>() {
                                    sirius_ranks[idx] = rank;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // 5. Side-by-Side Comparison
    // =========================================================================
    let n = eval_records.len();
    let mut oxidems_top1 = 0;
    let mut oxidems_top3 = 0;
    let mut oxidems_top5 = 0;

    let mut sirius_top1 = 0;
    let mut sirius_top3 = 0;
    let mut sirius_top5 = 0;

    println!("\n===================================================================================================");
    println!(
        "{:<4} | {:<30} | {:<12} | {:<12} | {:<12}",
        "Idx", "Compound", "Formula", "OxideMS Rank", "SIRIUS 6 Rank"
    );
    println!("---------------------------------------------------------------------------------------------------");

    for i in 0..n.min(30) {
        let (ox_rank, _, _) = oxidems_results[i];
        let sir_rank = sirius_ranks[i];

        let trunc_name = if eval_records[i].name.len() > 28 {
            format!("{}...", &eval_records[i].name[..26])
        } else {
            eval_records[i].name.clone()
        };

        println!(
            "{:<4} | {:<30} | {:<12} | #{:<11} | #{:<11}",
            i + 1,
            trunc_name,
            eval_records[i].true_formula.to_string(),
            ox_rank,
            if sir_rank == 999 { ">10 / None".to_string() } else { sir_rank.to_string() }
        );
    }

    for i in 0..n {
        let (ox_rank, _, _) = oxidems_results[i];
        let sir_rank = sirius_ranks[i];

        if ox_rank == 1 { oxidems_top1 += 1; }
        if ox_rank <= 3 { oxidems_top3 += 1; }
        if ox_rank <= 5 { oxidems_top5 += 1; }

        if sir_rank == 1 { sirius_top1 += 1; }
        if sir_rank <= 3 { sirius_top3 += 1; }
        if sir_rank <= 5 { sirius_top5 += 1; }
    }

    println!("---------------------------------------------------------------------------------------------------");
    println!("\n=========================================================================");
    println!("                     FINAL HEAD-TO-HEAD SUMMARY (N = {})", n);
    println!("=========================================================================");
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Benchmark Metric", "OxideMS", "SIRIUS 6.0.5"
    );
    println!("-------------------------------------------------------------------------");
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Top-1 Accuracy",
        format!("{:.1}% ({}/{})", (oxidems_top1 as f64 / n as f64) * 100.0, oxidems_top1, n),
        format!("{:.1}% ({}/{})", (sirius_top1 as f64 / n as f64) * 100.0, sirius_top1, n)
    );
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Top-3 Accuracy",
        format!("{:.1}% ({}/{})", (oxidems_top3 as f64 / n as f64) * 100.0, oxidems_top3, n),
        format!("{:.1}% ({}/{})", (sirius_top3 as f64 / n as f64) * 100.0, sirius_top3, n)
    );
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Top-5 Accuracy",
        format!("{:.1}% ({}/{})", (oxidems_top5 as f64 / n as f64) * 100.0, oxidems_top5, n),
        format!("{:.1}% ({}/{})", (sirius_top5 as f64 / n as f64) * 100.0, sirius_top5, n)
    );
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Total Computation Time",
        format!("{:.2} s", oxidems_time.as_secs_f64()),
        format!("{:.2} s", sirius_time.as_secs_f64())
    );
    println!(
        "{:<30} | {:<20} | {:<20}",
        "Throughput",
        format!("{:.1} spectra / sec", n as f64 / oxidems_time.as_secs_f64()),
        format!("{:.1} spectra / sec", n as f64 / sirius_time.as_secs_f64())
    );
    println!("=========================================================================\n");
}
