use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::plotting::ClusterQualityAnalyzer;
use std::path::{Path, PathBuf};

fn find_mgf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_mgf_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mgf") {
            out.push(path);
        }
    }
}

mod support;

fn main() {
    let dataset_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../msgf_full_run".to_string());
    support::require("cluster_quality_low_mem", &[(&dataset_dir, "directory of MGF files to cluster (first argument)")]);
    let spill_dir = PathBuf::from("target/low_mem_spill");
    let out_csv = "low_mem_clusters.csv";
    let out_png = "low_mem_cluster_quality.png";

    let mut mgf_files = Vec::new();
    find_mgf_files(Path::new(&dataset_dir), &mut mgf_files);
    mgf_files.sort();
    println!("Found {} mgf files under {dataset_dir}", mgf_files.len());

    let mut experiment = SpectralExperiment::proteomics();

    let t0 = std::time::Instant::now();
    let mut skipped = 0usize;
    for (i, path) in mgf_files.iter().enumerate() {
        if let Err(e) = experiment.ingest_file_low_mem(path.to_str().unwrap(), &spill_dir) {
            eprintln!("skipping {}: {e}", path.display());
            skipped += 1;
            continue;
        }
        if (i + 1) % 100 == 0 || i + 1 == mgf_files.len() {
            println!(
                "  ingested {}/{} files ({} spectra so far, {} skipped) [{:.1}s]",
                i + 1,
                mgf_files.len(),
                experiment.total_spectra_ingested,
                skipped,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "Ingested {} spectra total across {} files in {:.1}s",
        experiment.total_spectra_ingested,
        mgf_files.len(),
        t0.elapsed().as_secs_f64()
    );

    let t1 = std::time::Instant::now();
    let (total_clustered, num_clusters, quality) = experiment
        .cluster_and_write_low_mem_with_quality(out_csv)
        .expect("low-mem clustering failed");
    println!(
        "Formed {num_clusters} clusters covering {total_clustered} spectra in {:.1}s",
        t1.elapsed().as_secs_f64()
    );

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

    ClusterQualityAnalyzer::new(&quality)
        .plot_quality_overview(out_png)
        .expect("failed to render quality plot");
    println!("Wrote {out_csv} and {out_png}");
    println!("Total wall time: {:.1}s", t0.elapsed().as_secs_f64());
}
