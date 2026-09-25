use oxidems_lib::sirius::*;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::time::Instant;

struct StandardSpectrum {
    name: String,
    true_formula: Formula,
    precursor_mz: f64,
    adduct: Adduct,
    peaks: Vec<Peak>,
}

fn parse_ms_file(path: &str, default_formula: &str) -> Option<StandardSpectrum> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);

    let mut name = String::new();
    let mut precursor_mz = 0.0;
    let mut adduct = Adduct::protonated();
    let mut peaks = Vec::new();
    let mut in_peaks = false;

    for line in reader.lines().flatten() {
        let trimmed = line.trim();
        if trimmed.starts_with(">compound") {
            name = trimmed.trim_start_matches(">compound").trim().to_string();
        } else if trimmed.starts_with(">parentmass") {
            precursor_mz = trimmed.trim_start_matches(">parentmass").trim().parse().unwrap_or(0.0);
        } else if trimmed.starts_with(">ionization") {
            let ion_str = trimmed.trim_start_matches(">ionization").trim();
            if ion_str.contains("-") {
                adduct = Adduct::deprotonated();
            }
        } else if trimmed.starts_with(">collision") || trimmed.starts_with(">ms2") {
            in_peaks = true;
        } else if trimmed.starts_with('>') {
            in_peaks = false;
        } else if in_peaks && !trimmed.is_empty() {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz), Ok(intensity)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    peaks.push(Peak { mz, intensity });
                }
            }
        }
    }

    if precursor_mz == 0.0 || peaks.is_empty() {
        return None;
    }

    let true_formula = Formula::from_str(default_formula).ok()?;
    Some(StandardSpectrum {
        name,
        true_formula,
        precursor_mz,
        adduct,
        peaks,
    })
}

fn main() {
    println!("=========================================================================");
    println!("  OxideMS Formula Estimation & Fragmentation Tree on Clinical Standards");
    println!("=========================================================================\n");

    let standards = [
        (
            "/scratch/muhlir/downloads/ms-pred/data/exp_specs/clinical/hercynine_standard.ms",
            "C9H15N3O2",
            "Hercynine",
        ),
        (
            "/scratch/muhlir/downloads/ms-pred/data/exp_specs/clinical/gaba_arg_standard.ms",
            "C10H21N5O3",
            "GABA-Arg",
        ),
        (
            "/scratch/muhlir/downloads/ms-pred/data/exp_specs/clinical/lpc19-0_standard.ms",
            "C27H56NO7P",
            "LPC 19:0 (Lipid)",
        ),
    ];

    let tolerance = MassTolerance::Ppm(15.0);
    let rules = ChemicalRules::default();

    for (path, formula_str, label) in standards {
        println!("-------------------------------------------------------------------------");
        println!("Analyzing Standard: {} (Ground Truth: {})", label, formula_str);
        println!("File: {}", path);

        let spec = match parse_ms_file(path, formula_str) {
            Some(s) => s,
            None => {
                println!("Failed to parse standard file.\n");
                continue;
            }
        };

        let neutral_mass = spec.adduct.ion_mz_to_neutral_mass(spec.precursor_mz);

        // Clean peaks
        let max_int = spec.peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);
        let mut clean_peaks: Vec<Peak> = spec
            .peaks
            .iter()
            .filter(|p| (p.intensity / max_int) >= 0.005)
            .copied()
            .collect();
        clean_peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
        clean_peaks.truncate(30);
        clean_peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));

        let t0 = Instant::now();

        // 1. Candidate Precursor Generation (Extended if P present)
        let candidate_precursors = if spec.true_formula.p() > 0 {
            let generator = FormulaGenerator::chnops();
            generator.generate_candidates(neutral_mass, tolerance, &rules)
        } else {
            let generator = FormulaGenerator::chno();
            generator.generate_candidates(neutral_mass, tolerance, &rules)
        };

        // 2. Multi-Candidate Tree Building & Ranking
        let tree_config = TreeSolverConfig {
            tolerance,
            adduct: spec.adduct.clone(),
            ..Default::default()
        };
        let tree_builder = FragmentationTreeBuilder::new(tree_config);

        let mut scored_trees = tree_builder.build_trees_for_candidates(
            &candidate_precursors,
            spec.precursor_mz,
            &clean_peaks,
        );
        scored_trees.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

        let elapsed = t0.elapsed();

        let rank = scored_trees
            .iter()
            .position(|(f, _)| *f == spec.true_formula)
            .map(|pos| pos + 1)
            .unwrap_or(999);

        println!("Candidate Precursors: {}", candidate_precursors.len());
        println!("Ground Truth Rank:    #{}", rank);
        println!("Analysis Time:        {:.2} ms", elapsed.as_secs_f64() * 1000.0);

        println!("\nTop-5 Ranked Formula Candidates:");
        for (i, (f, tree)) in scored_trees.iter().take(5).enumerate() {
            let is_truth = *f == spec.true_formula;
            println!(
                "  #{:<2} {:<15} | Score: {:<7.2} | Expl Int: {:<5.1}% | Nodes: {:<2} | Loss Edges: {:<2} {}",
                i + 1,
                f.to_string(),
                tree.total_score,
                tree.explained_intensity,
                tree.nodes.len(),
                tree.edges.len(),
                if is_truth { "<- [TRUE FORMULA]" } else { "" }
            );
        }

        if let Some((_, best_tree)) = scored_trees.first() {
            println!("\nOptimal Fragmentation Tree Hierarchy:");
            println!("{}", best_tree.format_tree());
        }
        println!();
    }
}
