//! Repeats parsing a file N times, always from a warm OS page cache, and
//! reports mean/median/stdev -- the same discipline the Python benchmark
//! harness (oxidems_py/bench/scaling_bench.py) uses: a single timed run is
//! at the mercy of whatever the disk was doing that moment, and once the
//! file is warm in page cache, disk I/O stops being the thing being
//! measured at all -- parsing throughput is. This also makes MGF and mzML
//! comparable despite being different files/formats: every number reported
//! is warm-cache, repeated, and given with a spread, not a single cherry-
//! pickable run.
//!
//! Usage:
//!   parse_stats mgf   <file.mgf>   [reps] [--sequential]
//!   parse_stats mzml  <file.mzML>  [reps] [--sequential]
use oxidems_lib::io::mgf::{MgfParser, MgfParserConfig};
use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};
use std::time::Instant;

fn warm_page_cache(path: &str) -> std::io::Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn stdev(values: &[f64], m: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

struct RepResult {
    seconds: f64,
    spectra: usize,
    peaks: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sequential = args.iter().any(|a| a == "--sequential");
    let quickxml = args.iter().any(|a| a == "--quickxml");
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    if positional.len() < 2 {
        eprintln!("Usage: parse_stats <mgf|mzml> <file> [reps] [--sequential] [--quickxml]");
        eprintln!("  --quickxml: mzml only -- use the quick_xml fallback tokenizer instead of the default hand-rolled one");
        std::process::exit(2);
    }
    let format = positional[0].as_str();
    let path = positional[1].as_str();
    let reps: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    let file_size_mb = std::fs::metadata(path).map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);

    println!("warming page cache for {path} ...");
    warm_page_cache(path).expect("failed to warm page cache");

    // One untimed rep first and discarded: confirms everything works and
    // finishes touching every page via the real parse path (not just a
    // linear read, which the mzml indexed path in particular doesn't do)
    // before any timed rep starts.
    let _ = run_once(format, path, sequential, quickxml);

    println!(
        "running {reps} timed rep(s), format={format} mode={} tokenizer={} ...",
        if sequential { "sequential" } else { "parallel" },
        if quickxml { "quickxml" } else { "fast" },
    );
    let results: Vec<RepResult> =
        (0..reps).map(|_| run_once(format, path, sequential, quickxml)).collect();

    let times: Vec<f64> = results.iter().map(|r| r.seconds).collect();
    let m = mean(&times);
    let med = median(&times);
    let sd = stdev(&times, m);
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spectra = results[0].spectra;
    let peaks = results[0].peaks;

    println!(
        "\nformat={format} mode={} file_mb={file_size_mb:.1} spectra={spectra} peaks={peaks}",
        if sequential { "sequential" } else { "parallel" }
    );
    println!(
        "n={reps} mean_s={m:.4} median_s={med:.4} stdev_s={sd:.4} min_s={min:.4} max_s={max:.4}"
    );
    println!(
        "throughput (median): {:.0} spectra/s, {:.0} peaks/s, {:.1} MB/s",
        spectra as f64 / med,
        peaks as f64 / med,
        file_size_mb / med
    );
}

fn run_once(format: &str, path: &str, sequential: bool, quickxml: bool) -> RepResult {
    let start = Instant::now();
    let (spectra, peaks) = match format {
        "mgf" => {
            let mut parser = MgfParser::new(MgfParserConfig {
                use_parallel_parsing: !sequential,
                ..MgfParserConfig::default()
            });
            let records = parser.parse_file(path).unwrap();
            let peaks = records.iter().map(|r| r.peaks().len()).sum();
            (records.len(), peaks)
        }
        "mzml" => {
            let mut parser = MzmlParser::new(MzmlParserConfig {
                use_parallel_parsing: !sequential,
                use_fast_tokenizer: !quickxml,
                ..MzmlParserConfig::default()
            });
            let records = parser.parse_file(path).unwrap();
            let peaks = records.iter().map(|r| r.peaks().len()).sum();
            (records.len(), peaks)
        }
        other => panic!("unknown format {other}, expected mgf or mzml"),
    };
    RepResult {
        seconds: start.elapsed().as_secs_f64(),
        spectra,
        peaks,
    }
}
