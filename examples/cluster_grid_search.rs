use itertools::Itertools;
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;

mod support;

fn main() {
    let start_time = std::time::Instant::now();
    let mut spectral_experiment = SpectralExperiment::proteomics();
    spectral_experiment.engine.exact = true;

    let path = support::arg_or(1, "tests/Astral");
    support::require("cluster_grid_search", &[(&path, support::ASTRAL)]);
    let path = path.as_str();

    spectral_experiment.ingest_files_parallel(path);

    println!(
        "Total spectra parsed: {}, after {:.3} s",
        spectral_experiment.total_spectra_ingested,
        start_time.elapsed().as_secs_f32(),
    );
    println!(
        "Spectra remaining after filtering: {}, after {:.3} s",
        spectral_experiment.spectra.len(),
        start_time.elapsed().as_secs_f32(),
    );

    spectral_experiment.prepare_clustering();

    let epsilon_values = [0.1, 0.3, 0.6];
    let hash_sizes = [512, 800, 1024];

    for (epsilon_value, hash_size) in epsilon_values
        .into_iter()
        .cartesian_product(hash_sizes.into_iter())
    {
        spectral_experiment.engine.epsilon = epsilon_value;
        spectral_experiment.engine.set_hash_size(hash_size);
        #[cfg(feature = "gpu")]
        spectral_experiment.engine.enable_gpu(1);
        spectral_experiment.spectra = Vec::new();

        let clusters = spectral_experiment.cluster();

        println!(
            "eps={} hash={}: {} clusters, max_size={}",
            epsilon_value,
            hash_size,
            clusters.iter().count(),
            clusters.max_cluster_size(),
        );

        spectral_experiment.write_results(
            &clusters,
            format!(
                "exact_cluster_assignments_{}_{}.csv",
                epsilon_value, hash_size
            )
            .as_str(),
        );
    }
}
