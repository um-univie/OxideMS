//! Demo / benchmark for the low-memory streaming ingest path:
//! `SpectralExperiment::ingest_file_low_mem` +
//! `SpectralExperiment::finalize_low_mem_partitions` +
//! `SpectralExperiment::load_partition`.
//!
//! Unlike the normal `ingest_file` path, this never builds one big
//! `Vec<MSNSpectrum>` for the whole dataset: each rayon worker spills its own
//! parsed+filtered records straight to its own shard file as it produces
//! them, and only a small per-spectrum key stays resident until partitioning
//! is computed at the end. See `clustering::cluster::mz_partition` and
//! `io::mgf::MgfParser::parse_file_low_mem` for the implementation, and
//! `clustering::cluster::experiment::low_mem_tests` for the correctness test
//! (partitioning content is checked against
//! `ClusterEngine::split_on_mz_and_charge` directly, not re-derived here).
//!
//! Usage:
//!   cargo run --release --example partition_spill_prototype -- \
//!       <SPILL_DIR> <FILE_OR_DIR>...

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn mem_mb(field: &str) -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}

fn checkpoint(label: &str, t0: Instant) {
    println!(
        "[{:>20}] rss={:>6} MB   peak={:>6} MB   elapsed={:>7.2}s",
        label,
        mem_mb("VmRSS:"),
        mem_mb("VmHWM:"),
        t0.elapsed().as_secs_f32()
    );
}

mod support;

fn main() {
    let mut args = std::env::args().skip(1);
    let spill_dir = args
        .next()
        .unwrap_or_else(|| support::fail("usage: partition_spill_prototype <SPILL_DIR> <FILE_OR_DIR>..."));
    let inputs: Vec<String> = args.collect();
    assert!(!inputs.is_empty(), "need at least one input file or directory");
    let spill_dir = Path::new(&spill_dir);

    let t0 = Instant::now();
    checkpoint("start", t0);

    let mut experiment = SpectralExperiment::metabolomics();
    for path_str in &inputs {
        let path = Path::new(path_str);
        if path.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                let entry_path = entry.unwrap().path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("mgf") {
                    experiment
                        .ingest_file_low_mem(entry_path.to_str().unwrap(), spill_dir)
                        .unwrap();
                }
            }
        } else {
            experiment.ingest_file_low_mem(path_str, spill_dir).unwrap();
        }
    }
    checkpoint("after streaming ingest", t0);
    println!(
        "parsed {} spectra, kept {} keys",
        experiment.total_spectra_ingested,
        experiment.low_mem_keys.len()
    );

    let partition_paths = experiment.finalize_low_mem_partitions().unwrap();
    checkpoint("after finalize partitions", t0);
    println!("{} partition files", partition_paths.len());

    let mut sizes: Vec<usize> = Vec::with_capacity(partition_paths.len());
    let mut total_spectra = 0usize;
    for (path, _) in &partition_paths {
        let spectra = SpectralExperiment::load_partition(path).unwrap();
        total_spectra += spectra.len();
        sizes.push(spectra.len());
    }
    checkpoint("after loading every partition back", t0);

    sizes.sort_unstable();
    if !sizes.is_empty() {
        println!(
            "partition sizes (spectra): min={} max={} median={} mean={:.1} total={}",
            sizes.first().unwrap(),
            sizes.last().unwrap(),
            sizes[sizes.len() / 2],
            total_spectra as f64 / sizes.len() as f64,
            total_spectra
        );
    }

    let spill_bytes: u64 = fs::read_dir(spill_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    let partitions_bytes: u64 = fs::read_dir(spill_dir.join("partitions"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    println!(
        "disk: {:.2} GB shards + {:.2} GB partitions",
        spill_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        partitions_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

    checkpoint("done", t0);
}
