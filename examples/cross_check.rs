use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};

mod support;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| support::fail("usage: cross_check <file.mzML>"));

    let mut fast = MzmlParser::new(MzmlParserConfig { use_fast_tokenizer: true, ..MzmlParserConfig::default() });
    let fast_records = fast.parse_file(&path).unwrap();

    let mut quickxml = MzmlParser::new(MzmlParserConfig { use_fast_tokenizer: false, ..MzmlParserConfig::default() });
    let quickxml_records = quickxml.parse_file(&path).unwrap();

    assert_eq!(fast_records.len(), quickxml_records.len(), "count mismatch");
    let mut mismatches = 0;
    for (a, b) in fast_records.iter().zip(quickxml_records.iter()) {
        let same = a.get_charge() == b.get_charge()
            && a.get_metadata("PEPMASS") == b.get_metadata("PEPMASS")
            && a.get_metadata("RTINSECONDS") == b.get_metadata("RTINSECONDS")
            && a.peaks().len() == b.peaks().len()
            && a.peaks().iter().zip(b.peaks().iter()).all(|(pa, pb)| {
                (pa.mz - pb.mz).abs() < 1e-9 && (pa.intensity - pb.intensity).abs() < 1e-4
            });
        if !same { mismatches += 1; }
    }
    println!("{}: {} spectra, {} mismatches", path, fast_records.len(), mismatches);
}
