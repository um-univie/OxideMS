use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};
use std::time::Instant;

mod support;

fn main() {
    eprintln!("rayon threads available: {}", rayon::current_num_threads());
    let args: Vec<String> = std::env::args().collect();
    let sequential = args.iter().any(|a| a == "--sequential");
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| support::fail("usage: mzml_timing <file.mzML> [--sequential]"));

    let mut parser = MzmlParser::new(MzmlParserConfig {
        parse_metadata: true,
        use_parallel_parsing: !sequential,
        ..MzmlParserConfig::default()
    });

    let start = Instant::now();
    let records = parser.parse_file(path).unwrap();
    let elapsed = start.elapsed();

    let total_peaks: usize = records.iter().map(|r| r.peaks().len()).sum();
    let with_precursor = records
        .iter()
        .filter(|r| r.get_metadata("PEPMASS").is_some())
        .count();
    let with_charge = records.iter().filter(|r| r.get_charge() != 0).count();

    println!(
        "mode={} spectra={} total_peaks={} with_precursor_mz={} with_charge={} elapsed={:?}",
        if sequential { "sequential" } else { "parallel" },
        records.len(),
        total_peaks,
        with_precursor,
        with_charge,
        elapsed
    );

    if let Some(first) = records.first() {
        println!(
            "first spectrum: scan={:?} charge={} precursor_mz={:?} rt_s={:?} n_peaks={}",
            first.get_metadata("SCANS"),
            first.get_charge(),
            first.get_metadata("PEPMASS"),
            first.get_metadata("RTINSECONDS"),
            first.peaks().len()
        );
    }
}
