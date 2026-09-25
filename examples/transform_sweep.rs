//! Clusters one dataset under a chosen intensity transform and epsilon, and
//! writes a standard cluster CSV for scoring against a sequence-level ground
//! truth.
//!
//! The shipped presets use no intensity transform, a choice made by a DOE
//! scored on the MS-RT proxy. That proxy was since measured to miss ~78% of
//! real peptide-identity confusions, so the choice is worth re-testing
//! against MS-DB truth: a transform compresses the dominant peaks and gives
//! the minor, diagnostic fragments proportionally more weight, which is the
//! axis on which isobaric sequence variants differ.
//!
//! A transform changes the whole similarity distribution, so runs are not
//! comparable at fixed epsilon -- sweep epsilon per transform and compare
//! the purity/completeness curves.
//!
//! Usage: transform_sweep <dataset_dir> <preset> <none|sqrt|log> <epsilon> <out.csv>

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::workflow::{
    BinaryIntensity, LogTransformIntensity, RankTransformIntensity,
    SquareRootTransformIntensity,
};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn mgf_dirs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut has_mgf = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            mgf_dirs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mgf") {
            has_mgf = true;
        }
    }
    if has_mgf {
        out.push(dir.to_path_buf());
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dataset_dir = args.next().expect("dataset dir");
    let preset = args.next().expect("preset");
    let transform = args.next().expect("none|sqrt|log");
    let epsilon: f32 = args.next().and_then(|a| a.parse().ok()).expect("epsilon");
    let out_path = args.next().expect("output csv");

    let mut experiment = match preset.as_str() {
        "metabolomics" => SpectralExperiment::metabolomics(),
        "proteomics" => SpectralExperiment::proteomics(),
        other => panic!("unknown preset {other}"),
    };
    experiment.engine.epsilon = epsilon;

    // Appended after the preset's filters. `KeepTopNPeaks` selects by rank,
    // and both transforms are monotone, so placement relative to it does not
    // change which peaks survive -- only their weights in the similarity.
    match transform.as_str() {
        "none" => {}
        "sqrt" => experiment
            .workflow
            .add_operation(SquareRootTransformIntensity::new()),
        "log" => experiment
            .workflow
            .add_operation(LogTransformIntensity::new()),
        "rank" => experiment
            .workflow
            .add_operation(RankTransformIntensity::new()),
        "binary" => experiment
            .workflow
            .add_operation(BinaryIntensity::new()),
        "rank" => experiment
            .workflow
            .add_operation(RankTransformIntensity::new()),
        "binary" => experiment
            .workflow
            .add_operation(BinaryIntensity::new()),
        other => panic!("unknown transform {other}"),
    }

    // Adaptive threshold, with a null tail matched to whatever similarity is
    // in use: "tau:log10p,tau:log10p,..."
    if let Ok(t) = std::env::var("OXIDEMS_NULL_TAIL") {
        let table: Vec<(f32, f32)> = t
            .split(',')
            .filter_map(|kv| {
                let (a, b) = kv.split_once(':')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .collect();
        println!("null tail: {} points", table.len());
        experiment.engine.null_tail = Some(table);
    }
    if let Ok(a) = std::env::var("OXIDEMS_ADAPTIVE_ALPHA") {
        if let Ok(v) = a.parse::<f32>() {
            experiment.engine.adaptive_alpha = Some(v);
            println!("adaptive_alpha={v}");
        }
    }
    let mut dirs = Vec::new();
    mgf_dirs(Path::new(&dataset_dir), &mut dirs);
    dirs.sort();
    for d in &dirs {
        experiment.ingest_files_parallel(d.to_str().unwrap());
    }

    let file_scan: Vec<(usize, usize)> = experiment
        .metadata
        .iter()
        .map(|m| (m.file_index, m.scan_number))
        .collect();
    let file_names = experiment.file_names.clone();
    let spectra = experiment.spectra.clone();

    let t = std::time::Instant::now();
    experiment.prepare_clustering();
    let clusters = experiment.cluster();
    println!(
        "transform={transform} epsilon={epsilon} clusters={} clustered={} in {:.1}s",
        clusters.num_clusters(),
        clusters.total_elements(),
        t.elapsed().as_secs_f64()
    );

    let mut out = BufWriter::new(std::fs::File::create(&out_path).unwrap());
    writeln!(
        out,
        "identifier,precursor_charge,precursor_mz,retention_time,cluster"
    )
    .unwrap();
    for (cid, cluster) in clusters.iter().enumerate() {
        if cluster.len() < 2 {
            continue;
        }
        for &idx in cluster {
            let (fi, scan) = file_scan[idx];
            let s = &spectra[idx];
            writeln!(
                out,
                "mzspec:USI000000:{}:scan:{scan},{},{},{},{cid}",
                file_names[fi],
                s.charge(),
                s.precursor_mz,
                s.retention_time_in_seconds
            )
            .unwrap();
        }
    }
    println!("wrote {out_path}");
}
