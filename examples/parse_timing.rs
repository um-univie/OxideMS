use oxidems_lib::io::mgf::MgfParser;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut parser = MgfParser::default();
    let start = Instant::now();
    let records = parser.parse_file(&path).unwrap();
    let elapsed = start.elapsed();
    let total_peaks: usize = records.iter().map(|r| r.peaks().len()).sum();
    println!("spectra={} total_peaks={} elapsed={:?}", records.len(), total_peaks, elapsed);
}
