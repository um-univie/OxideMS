use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::set_global_thread_limit;

mod support;

fn main() {
    let path = support::arg_or(1, "tests/Astral");
    support::require("basic_clustering", &[(&path, support::ASTRAL)]);
    let path = path.as_str();
    let mut spectral_experiment = SpectralExperiment::default();
    spectral_experiment.engine.exact = true;

    spectral_experiment.ingest_files_parallel(path);
    spectral_experiment.prepare_clustering();

    let clusters = spectral_experiment.cluster();

    spectral_experiment.write_results(&clusters, "redshift.csv");
}
