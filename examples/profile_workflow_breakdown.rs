use oxidems_lib::sirius::*;
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::workflow::MassTolerance;
use std::time::Instant;

fn main() {
    let mzml_path = "/scratch/muhlir/data/mass_spec_data/MSV000080554_mzML/A10_GA10_01_7891.mzML";

    println!("=========================================================================");
    println!("  OxideMS Workflow Phase Bottleneck Profiling");
    println!("=========================================================================\n");

    let t0 = Instant::now();

    // 1. Ingestion
    let t_ingest = Instant::now();
    let mut experiment = SpectralExperiment::metabolomics();
    experiment.ingest_file(mzml_path);
    let ingest_duration = t_ingest.elapsed();
    println!("1. Ingest mzML: {:.4} s ({} spectra)", ingest_duration.as_secs_f64(), experiment.total_spectra_ingested);

    // 2. Clustering Preparation & Execution
    let t_cluster = Instant::now();
    experiment.prepare_clustering();
    let clusters = experiment.cluster();
    let cluster_duration = t_cluster.elapsed();
    println!("2. DBSCAN Clustering: {:.4} s ({} clusters)", cluster_duration.as_secs_f64(), clusters.num_clusters());

    // 3. Consensus Building
    let t_consensus = Instant::now();
    let mut consensus_count = 0;
    for (i, cluster) in clusters.iter().enumerate() {
        if let Some(rec) = experiment.build_consensus_spectrum(cluster, 0.01) {
            consensus_count += 1;
        }
    }
    let consensus_duration = t_consensus.elapsed();
    println!("3. Consensus Spectrum Reconstruction: {:.4} s ({} built)", consensus_duration.as_secs_f64(), consensus_count);

    // 4. Candidate Formula Gen & Tree Solving per Cluster
    let workflow = SiriusClusteringWorkflow::new(SiriusWorkflowConfig::default());
    let mut total_candidate_count = 0usize;
    let mut gen_time_total = 0.0;
    let mut tree_time_total = 0.0;

    println!("\nProfiling all {} clusters for formula generation and tree building:", clusters.num_clusters());
    for (i, cluster) in clusters.iter().enumerate() {
        let first_id = cluster[0];
        let meta = &experiment.metadata[first_id];
        let precursor_mz = meta.precursor_mz;
        let neutral_mass = workflow.config.adduct.ion_mz_to_neutral_mass(precursor_mz);

        let consensus = experiment.build_consensus_spectrum(cluster, 0.01);
        let mut clean_peaks = consensus.map(|c| c.peaks().to_vec()).unwrap_or_default();
        let max_int = clean_peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max).max(1e-6);
        clean_peaks.retain(|p| (p.intensity / max_int) >= 0.005);
        clean_peaks.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
        clean_peaks.truncate(workflow.config.max_peaks_per_tree);
        clean_peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));

        let t_gen = Instant::now();
        let mut candidate_precursors = workflow.gen_chnops.generate_candidates(
            neutral_mass,
            workflow.config.tolerance,
            &workflow.config.rules,
        );
        let n_raw_candidates = candidate_precursors.len();

        // Pre-rank by mass accuracy + prior before building expensive DAG trees
        if candidate_precursors.len() > 15 {
            candidate_precursors.sort_by(|a, b| {
                let err_a = (a.exact_mass() - neutral_mass).abs();
                let err_b = (b.exact_mass() - neutral_mass).abs();
                let prior_a = workflow.scorer.score_prior(a);
                let prior_b = workflow.scorer.score_prior(b);
                let score_a = prior_a - (err_a / neutral_mass) * 1e6;
                let score_b = prior_b - (err_b / neutral_mass) * 1e6;
                score_b.total_cmp(&score_a)
            });
            candidate_precursors.truncate(15);
        }

        let gen_elapsed = t_gen.elapsed().as_secs_f64();
        gen_time_total += gen_elapsed;
        total_candidate_count += candidate_precursors.len();

        let t_tree = Instant::now();
        let scored_trees = workflow.tree_builder.build_trees_for_candidates(
            &candidate_precursors,
            precursor_mz,
            &clean_peaks,
        );
        let tree_elapsed = t_tree.elapsed().as_secs_f64();
        tree_time_total += tree_elapsed;

        if gen_elapsed + tree_elapsed > 0.05 || n_raw_candidates > 100 {
            println!(
                "  Cluster #{:<3} (m/z {:<7.2}, n_peaks: {:<2}): {:<4} raw -> {:<2} pruned | Gen: {:<6.4}s | Tree: {:<6.4}s",
                i, precursor_mz, clean_peaks.len(), n_raw_candidates, candidate_precursors.len(), gen_elapsed, tree_elapsed
            );
        }
    }

    println!("\n=========================================================================");
    println!("Total Elapsed: {:.4} s", t0.elapsed().as_secs_f64());
    println!("=========================================================================");
}
