//! Regenerates spectral_comparison_Astral_eps.svg and prints per-epsilon
//! purity/completeness (parsed by publication/generate_astral_purity_completeness.py
//! into figures/astral_purity_completeness.png) from the real Astral MGF
//! data in tests/Astral, using the tuned metabolomics preset.
//!
//! MS-RT proxy tolerances (0.01 Da precursor, 30s retention time) match the
//! narrower window main.tex uses for metabolomics validation (vs. 65s for
//! proteomics), reflecting the narrower chromatographic peaks of small
//! molecules.

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::plotting::SpectrumPlotter;

const MZ_TOL: f64 = 0.01;
const RT_TOL: f32 = 30.0;

fn main() {
    let path = "tests/Astral";
    let eps_values = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    // Used to pick a representative matched pair for the mirror plot.
    let mut mirror_written = false;

    for &eps in &eps_values {
        let mut experiment = SpectralExperiment::metabolomics();
        experiment.engine.epsilon = eps;

        let t0 = std::time::Instant::now();
        experiment.ingest_files_parallel(path);
        let spectra_for_mirror = experiment.spectra.clone();
        experiment.prepare_clustering();
        let clusters = experiment.cluster();

        let purity = experiment.average_weighted_cluster_purity(&clusters, MZ_TOL, RT_TOL);
        let n10 = experiment.calculate_n10_value(&clusters);
        let n_clustered: usize = clusters.iter().map(|c| c.len()).sum();
        let completeness = n_clustered as f64 / experiment.total_spectra_ingested as f64;

        println!(
            "eps={eps}: weighted_purity={purity:.4} n10={n10} n_clusters={} n_clustered={n_clustered} completeness={completeness:.4} total_spectra={} elapsed={:.2}s",
            clusters.num_clusters(),
            experiment.total_spectra_ingested,
            t0.elapsed().as_secs_f64()
        );


        // At eps=0.5 (the tuned preset value), also pull a real matched pair
        // for the mirror plot: the two largest members of the biggest
        // multi-spectrum cluster.
        if !mirror_written && (eps - 0.5).abs() < 1e-9 {
            if let Some(cluster) = clusters.iter().max_by_key(|c| c.len()) {
                if cluster.len() >= 2 {
                    let s1 = &spectra_for_mirror[cluster[0]];
                    let s2 = &spectra_for_mirror[cluster[1]];
                    let plotter = SpectrumPlotter::new()
                        .dimensions(1200, 600)
                        .match_tolerance(0.05)
                        .title("Astral Metabolomics: Matched Cluster Members");
                    plotter
                        .plot_mirror(
                            &s1.spectrum,
                            &s2.spectrum,
                            "publication/figures/spectral_comparison_Astral_eps.png",
                        )
                        .expect("failed to write mirror plot");
                    println!(
                        "Wrote spectral_comparison_Astral_eps.png from cluster of size {}",
                        cluster.len()
                    );
                    mirror_written = true;
                }
            }
        }
    }
}
