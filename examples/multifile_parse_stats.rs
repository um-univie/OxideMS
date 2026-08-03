//! Compares two axes of parallelism for parsing a *set* of files, not just
//! one: intra-file (parallelize across the spectra within one file, files
//! processed one after another -- what parse_stats.rs measures) vs.
//! file-level (parallelize across files themselves, each file parsed
//! single-threaded internally). A real pipeline ingesting a folder of many
//! files can pick either axis, or both -- this measures which is actually
//! better, and whether combining them helps or just oversubscribes the CPU.
//!
//! Same warm-cache + repeated-stats discipline as parse_stats.rs: every file
//! is read once, untimed, before any timed rep, and every mode is repeated
//! N times with mean/median/stdev reported, not a single run.
//!
//! Usage:
//!   multifile_parse_stats <mgf|mzml> <reps> <file1> [file2] [file3] ...
use oxidems_lib::io::mgf::{MgfParser, MgfParserConfig};
use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};
use rayon::prelude::*;
use std::time::Instant;

fn warm_page_cache(path: &str) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = vec![0u8; 8 << 20];
    while f.read(&mut buf).unwrap() > 0 {}
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 { (v[mid - 1] + v[mid]) / 2.0 } else { v[mid] }
}
fn stdev(values: &[f64], m: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

/// (spectra, peaks) parsing one file, with the given intra-file parallelism.
fn parse_one(format: &str, path: &str, intra_parallel: bool) -> (usize, usize) {
    match format {
        "mgf" => {
            let mut parser = MgfParser::new(MgfParserConfig {
                use_parallel_parsing: intra_parallel,
                ..MgfParserConfig::default()
            });
            let records = parser.parse_file(path).unwrap();
            let peaks = records.iter().map(|r| r.peaks().len()).sum();
            (records.len(), peaks)
        }
        "mzml" => {
            let mut parser = MzmlParser::new(MzmlParserConfig {
                use_parallel_parsing: intra_parallel,
                ..MzmlParserConfig::default()
            });
            let records = parser.parse_file(path).unwrap();
            let peaks = records.iter().map(|r| r.peaks().len()).sum();
            (records.len(), peaks)
        }
        other => panic!("unknown format {other}"),
    }
}

/// Runs one full pass over every file in `paths` under one (files_parallel,
/// intra_parallel) combination, returning (elapsed_seconds, total_spectra,
/// total_peaks).
fn run_mode(format: &str, paths: &[String], files_parallel: bool, intra_parallel: bool) -> (f64, usize, usize) {
    let start = Instant::now();
    let (spectra, peaks): (usize, usize) = if files_parallel {
        paths
            .par_iter()
            .map(|p| parse_one(format, p, intra_parallel))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        paths
            .iter()
            .map(|p| parse_one(format, p, intra_parallel))
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    };
    (start.elapsed().as_secs_f64(), spectra, peaks)
}

fn report(label: &str, times: &[f64], total_mb: f64, spectra: usize, peaks: usize) {
    let m = mean(times);
    let med = median(times);
    let sd = stdev(times, m);
    println!(
        "{label:<32} n={:>2} mean_s={m:.4} median_s={med:.4} stdev_s={sd:.4} ({:.0}% of mean)  \
         -> {:.0} spectra/s, {:.0} peaks/s, {:.1} MB/s",
        times.len(),
        if m > 0.0 { 100.0 * sd / m } else { 0.0 },
        spectra as f64 / med,
        peaks as f64 / med,
        total_mb / med,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: multifile_parse_stats <mgf|mzml> <reps> <file1> [file2] ...");
        std::process::exit(2);
    }
    let format = args[1].clone();
    let reps: usize = args[2].parse().expect("reps must be an integer");
    let paths: Vec<String> = args[3..].to_vec();

    let total_mb: f64 = paths
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0))
        .sum::<f64>()
        / (1024.0 * 1024.0);

    println!("warming page cache for {} file(s), {:.1} MB total ...", paths.len(), total_mb);
    for p in &paths {
        warm_page_cache(p);
    }

    println!(
        "format={format} files={} reps={reps} rayon_threads={}\n",
        paths.len(),
        rayon::current_num_threads()
    );

    // One untimed warmup pass per mode combination first (see parse_stats.rs
    // rationale), then the real timed reps.
    for &(files_parallel, intra_parallel, label) in &[
        (false, false, "files=seq  intra=seq  (baseline)"),
        (false, true, "files=seq  intra=par  (per-file parallel)"),
        (true, false, "files=par  intra=seq  (file-level parallel)"),
        (true, true, "files=par  intra=par  (both, oversubscription risk)"),
    ] {
        let _ = run_mode(&format, &paths, files_parallel, intra_parallel);
        let mut times = Vec::with_capacity(reps);
        let mut last = (0usize, 0usize);
        for _ in 0..reps {
            let (t, spectra, peaks) = run_mode(&format, &paths, files_parallel, intra_parallel);
            times.push(t);
            last = (spectra, peaks);
        }
        report(label, &times, total_mb, last.0, last.1);
    }
}
