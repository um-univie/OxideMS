use oxidems_lib::sirius::*;
use oxidems_lib::workflow::MassTolerance;
use std::time::Instant;

fn main() {
    let mzml_path: String = std::env::args().nth(1).unwrap_or_else(|| {
        "/scratch/muhlir/data/mass_spec_data/MSV000080554_mzML/A10_GA10_01_7891.mzML".to_string()
    });
    let mzml_path = mzml_path.as_str();

    println!("=========================================================================");
    println!("  OxideMS End-to-End mzML Clustering & SIRIUS Formula Workflow");
    println!("=========================================================================\n");

    println!("Target mzML File: {}", mzml_path);

    let config = SiriusWorkflowConfig {
        preset: WorkflowPreset::Metabolomics,
        tolerance: MassTolerance::Ppm(15.0),
        adduct: Adduct::protonated(),
        adduct_search_strategy: AdductSearchStrategy::Single(Adduct::protonated()),
        rules: ChemicalRules::default(),
        isotope_weight: 1.5,
        top_k_candidates: 5,
        min_cluster_size: 1,
        max_peaks_per_tree: 30,
        max_candidates_for_tree_search: 15,
        consensus_bin_width_da: 0.01,
        filter_unknown_charge: false,
        unknown_charge_policy: UnknownChargePolicy::DefaultAdduct,
        candidate_charges: vec![1, 2],
        filter_peaks_above_precursor: true,
        ..Default::default()
    };

    let workflow = SiriusClusteringWorkflow::new(config);

    println!("Executing end-to-end DBSCAN clustering + SIRIUS formula annotation...");
    let t0 = Instant::now();
    let result = workflow.run_on_file(mzml_path).expect("Failed to execute workflow on mzML file");
    let elapsed = t0.elapsed();

    println!("\n=========================================================================");
    println!("                           WORKFLOW SUMMARY");
    println!("=========================================================================");
    println!("Total Spectra Ingested:   {}", result.total_spectra_ingested);
    println!("Total Clusters Formed:    {}", result.total_clusters_formed);
    println!("Total Annotated Clusters: {}", result.total_annotated_clusters);
    println!("Total Workflow Time:      {:.2} s", elapsed.as_secs_f64());
    println!("Overall Throughput:       {:.1} spectra / sec", result.throughput_spectra_per_sec);
    println!("=========================================================================\n");

    println!("Top Representative Annotated Clusters:");
    for (i, cluster) in result.annotations.iter().take(5).enumerate() {
        println!("-------------------------------------------------------------------------");
        println!(
            "Cluster #{:<2} | Members: {:<3} | Precursor m/z: {:<9.4} | RT: {:<6.1}s | Charge: {:<+2}",
            cluster.cluster_id,
            cluster.member_count,
            cluster.precursor_mz,
            cluster.retention_time_seconds,
            cluster.charge
        );

        if let Some(top) = cluster.top_candidate() {
            println!(
                "  Top Formula:     {} (Exact Mass: {:.4} Da, Error: {:.2} ppm)",
                top.formula,
                top.exact_mass,
                top.mass_error_ppm
            );
            println!(
                "  Scores:          Total: {:<6.2} | Tree: {:<6.2} | Isotope: {:<6.2} | Prior: {:<6.2}",
                top.total_score,
                top.tree_score,
                top.isotope_score,
                top.prior_score
            );
            println!(
                "  Tree Statistics: Explained Int: {:.1}% | Explained Peaks: {}/{} | Tree Nodes: {}",
                top.explained_intensity,
                top.explained_peaks,
                top.total_peaks,
                top.tree.nodes.len()
            );

            // MS1 isotope-envelope fit (real survey scan when the mzML has MS1,
            // otherwise the surviving precursor cluster in the MS2 consensus).
            let env = &cluster.ms1_envelope;
            let theo = top.theoretical_isotope_pattern();
            // Anchor observed intensities on the monoisotopic peak, the way the
            // scorer does — ratios to M0, not fractions of the whole window.
            let near = |mz: f64, p: &oxidems_lib::spectrum::Peak| (p.mz - mz).abs() / mz < 15e-6;
            let m0_obs = theo.first().and_then(|(mz, _)| {
                env.iter()
                    .filter(|p| near(*mz, p))
                    .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
                    .map(|p| p.intensity as f64)
            });
            let src = if cluster.ms1_from_survey_scan { "MS1 survey scan" } else { "MS2-consensus fallback" };
            match (env.len() >= 2, m0_obs) {
                (true, Some(m0)) if m0 > 0.0 => {
                    let (theo0, _) = (theo[0].1, ());
                    println!("\n  MS1 Isotope Fit [{src}] ({} peaks, score {:.2}):", env.len(), top.isotope_score);
                    for (k, (mz, ab)) in theo.iter().take(3).enumerate() {
                        let obs = env
                            .iter()
                            .filter(|p| near(*mz, p))
                            .map(|p| p.intensity as f64 / m0)
                            .next();
                        let theo_ratio = ab / theo0;
                        match obs {
                            Some(o) => println!("    M+{k}  m/z {mz:8.4}  theo {:5.2}x  obs {:5.2}x", theo_ratio, o),
                            None => println!("    M+{k}  m/z {mz:8.4}  theo {:5.2}x  obs   --  (not found)", theo_ratio),
                        }
                    }
                }
                _ => println!("\n  MS1 Isotope Fit: no usable envelope (score {:.2})", top.isotope_score),
            }

            if !top.tree.nodes.is_empty() {
                println!("\n  Optimal Fragmentation Tree Hierarchy:");
                let tree_str = top.tree.format_tree();
                for line in tree_str.lines().take(12) {
                    println!("    {}", line);
                }
                if tree_str.lines().count() > 12 {
                    println!("    ... [truncated for display]");
                }
            }
        } else {
            println!("  No valid formula candidate found within tolerance.");
        }
        println!();
    }
}
