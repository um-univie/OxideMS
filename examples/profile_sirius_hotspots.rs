use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::sirius::*;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
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

fn main() {
    let gnps_path_arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: {} <GNPS spectral library .mgf>", env!("CARGO_CRATE_NAME"));
        std::process::exit(2)
    });
    let gnps_path = gnps_path_arg.as_str();
    println!("=========================================================================");
    println!("  OxideMS Algorithmic Hotspot Profiler (Sub-Millisecond Instrumentation)");
    println!("=========================================================================\n");

    let mut parser = MgfParser::default();
    let records = parser.parse_file(gnps_path).expect("Failed to parse ALL_GNPS.mgf");

    let target_count = 500;
    let mut eval_spectra = Vec::new();

    for record in &records {
        if eval_spectra.len() >= target_count {
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

        eval_spectra.push((true_formula, pepmass, adduct, clean_peaks));
    }

    println!("Profiling {} real GNPS spectra in single-threaded mode...\n", eval_spectra.len());

    let rules = ChemicalRules::default();
    let tolerance = MassTolerance::Ppm(15.0);

    let mut time_precursor_gen_us: u128 = 0;
    let mut time_union_bound_us: u128 = 0;
    let mut time_fragment_decomp_us: u128 = 0;
    let mut time_subformula_filter_us: u128 = 0;
    let mut time_edge_building_us: u128 = 0;
    let mut time_dag_dp_solve_us: u128 = 0;
    let mut time_total_us: u128 = 0;

    let mut total_precursors_tested: usize = 0;
    let mut total_peaks_decomposed: usize = 0;
    let mut total_edges_evaluated: usize = 0;

    let t_all_start = Instant::now();

    let gen_chno = FormulaGenerator::chno();
    let gen_chnos = FormulaGenerator::chnos();
    let gen_chnops = FormulaGenerator::chnops();

    for (true_formula, pepmass, adduct, peaks) in &eval_spectra {
        let t_spec_start = Instant::now();
        let neutral_mass = adduct.ion_mz_to_neutral_mass(*pepmass);
        let has_p = true_formula.p() > 0;
        let has_s = true_formula.s() > 0;

        // 1. Precursor Generation
        let t0 = Instant::now();
        let candidate_precursors = if !has_p && !has_s {
            gen_chno.generate_candidates(neutral_mass, tolerance, &rules)
        } else if !has_p {
            gen_chnos.generate_candidates(neutral_mass, tolerance, &rules)
        } else {
            gen_chnops.generate_candidates(neutral_mass, tolerance, &rules)
        };
        time_precursor_gen_us += t0.elapsed().as_micros();
        total_precursors_tested += candidate_precursors.len();

        let tree_config = TreeSolverConfig {
            tolerance,
            adduct: adduct.clone(),
            ..Default::default()
        };
        let tree_builder = FragmentationTreeBuilder::new(tree_config);

        // 2. Union Bounding
        let t1 = Instant::now();
        let mut max_formula = Formula::empty();
        for f in &candidate_precursors {
            for el in &ALL_ELEMENTS {
                let cnt = f.get(*el);
                if cnt > max_formula.get(*el) {
                    max_formula.set(*el, cnt);
                }
            }
        }
        time_union_bound_us += t1.elapsed().as_micros();

        // 3. Fragment Peak Decomposition
        let t2 = Instant::now();
        let precomputed_peaks = tree_builder.annotator.annotate_peaks(peaks, &max_formula, tolerance);
        time_fragment_decomp_us += t2.elapsed().as_micros();
        total_peaks_decomposed += peaks.len();

        // 4. Multi-precursor tree building
        for cand in &candidate_precursors {
            let t3 = Instant::now();
            let filtered_peaks: Vec<AnnotatedPeak> = precomputed_peaks
                .iter()
                .map(|p| AnnotatedPeak {
                    peak_index: p.peak_index,
                    mz: p.mz,
                    intensity: p.intensity,
                    relative_intensity: p.relative_intensity,
                    candidates: p
                        .candidates
                        .iter()
                        .filter(|c| c.formula.is_subformula_of(cand))
                        .copied()
                        .collect(),
                })
                .filter(|p| !p.candidates.is_empty())
                .collect();
            time_subformula_filter_us += t3.elapsed().as_micros();

            // Build Graph Vertices & Edges
            let t4 = Instant::now();
            let tree = tree_builder.build_tree_from_annotated(cand, *pepmass, peaks, &filtered_peaks);
            time_dag_dp_solve_us += t4.elapsed().as_micros();
            total_edges_evaluated += tree.edges.len();
        }

        time_total_us += t_spec_start.elapsed().as_micros();
    }

    let total_time_sec = t_all_start.elapsed().as_secs_f64();
    let total_us_f = time_total_us as f64;

    println!("=========================================================================");
    println!("                    EXECUTION BREAKDOWN (N = {})", eval_spectra.len());
    println!("=========================================================================");
    println!(
        "{:<35} | {:<12} | {:<12} | {:<10}",
        "Subsystem / Operation", "Time (ms)", "Avg / Spec", "% of Total"
    );
    println!("-------------------------------------------------------------------------");
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | {:<9.1}%",
        "1. Precursor Formula Gen (DP)",
        (time_precursor_gen_us as f64) / 1000.0,
        (time_precursor_gen_us as f64) / (eval_spectra.len() as f64 * 1000.0),
        ((time_precursor_gen_us as f64) / total_us_f) * 100.0
    );
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | {:<9.1}%",
        "2. Union Bounding (F_max)",
        (time_union_bound_us as f64) / 1000.0,
        (time_union_bound_us as f64) / (eval_spectra.len() as f64 * 1000.0),
        ((time_union_bound_us as f64) / total_us_f) * 100.0
    );
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | {:<9.1}%",
        "3. Fragment Peak DP Decomposition",
        (time_fragment_decomp_us as f64) / 1000.0,
        (time_fragment_decomp_us as f64) / (eval_spectra.len() as f64 * 1000.0),
        ((time_fragment_decomp_us as f64) / total_us_f) * 100.0
    );
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | {:<9.1}%",
        "4. Sub-formula Filtering (is_sub)",
        (time_subformula_filter_us as f64) / 1000.0,
        (time_subformula_filter_us as f64) / (eval_spectra.len() as f64 * 1000.0),
        ((time_subformula_filter_us as f64) / total_us_f) * 100.0
    );
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | {:<9.1}%",
        "5. Graph Building & DAG DP Solve",
        (time_dag_dp_solve_us as f64) / 1000.0,
        (time_dag_dp_solve_us as f64) / (eval_spectra.len() as f64 * 1000.0),
        ((time_dag_dp_solve_us as f64) / total_us_f) * 100.0
    );
    println!("-------------------------------------------------------------------------");
    println!(
        "{:<35} | {:<12.2} | {:<12.2} | 100.0%",
        "TOTAL INSTRUMENTED TIME",
        (time_total_us as f64) / 1000.0,
        (time_total_us as f64) / (eval_spectra.len() as f64 * 1000.0)
    );
    println!("=========================================================================\n");
}
