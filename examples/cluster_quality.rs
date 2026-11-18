use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::plotting::ClusterQualityAnalyzer;

mod support;

fn main() {
    let path = support::arg_or(1, "tests/PXD000561_pilot");
    support::require("cluster_quality", &[(&path, support::PXD000561)]);
    let path = path.as_str();

    let t0 = std::time::Instant::now();
    let mut experiment = SpectralExperiment::proteomics();
    experiment.ingest_files_parallel(path);
    println!(
        "Ingested {} spectra from {path} in {:.3}s",
        experiment.spectra.len(),
        t0.elapsed().as_secs_f64()
    );

    // `prepare_clustering` drains `experiment.spectra` into the engine's batches, so
    // keep a copy around for the (optional) quality-metric pass afterwards.
    let t1 = std::time::Instant::now();
    let spectra_for_quality = experiment.spectra.clone();
    println!("Cloned spectra for quality pass in {:.3}s", t1.elapsed().as_secs_f64());

    let t2 = std::time::Instant::now();
    experiment.prepare_clustering();
    let clusters = experiment.cluster();
    println!(
        "Formed {} clusters covering {} spectra in {:.3}s",
        clusters.num_clusters(),
        clusters.total_elements(),
        t2.elapsed().as_secs_f64()
    );

    let t3 = std::time::Instant::now();
    let quality = experiment.calculate_cluster_quality_metrics(&clusters, &spectra_for_quality);
    println!("Computed quality metrics in {:.3}s", t3.elapsed().as_secs_f64());

    let avg_cohesions: Vec<f64> = quality.iter().map(|q| q.avg_cohesion).collect();
    let min_cohesions: Vec<f64> = quality.iter().map(|q| q.min_cohesion).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    println!(
        "Mean avg-cohesion: {:.4}  |  Mean min-cohesion: {:.4}",
        mean(&avg_cohesions),
        mean(&min_cohesions)
    );
    println!(
        "Clusters with avg-cohesion < 0.8: {}",
        avg_cohesions.iter().filter(|&&c| c < 0.8).count()
    );

    let t4 = std::time::Instant::now();
    let out_path = "cluster_quality.png";
    ClusterQualityAnalyzer::new(&quality)
        .plot_quality_overview(out_path)
        .expect("failed to render quality plot");
    println!("Wrote {out_path} in {:.3}s", t4.elapsed().as_secs_f64());

    println!("Total: {:.3}s", t0.elapsed().as_secs_f64());
}
