//! Same-tolerance, same-results comparison of exhaustive and indexed linking.
use oxidems_lib::feature_extraction::{Feature, FeatureTable};
use oxidems_lib::feature_extraction::raw_data::Ms2PrecursorInfo;
use oxidems_lib::workflow::MassTolerance;
use std::time::Instant;

fn main() {
    let features: Vec<_> = (0..50_000).map(|id| Feature {
        id, monoisotopic_mz: 100.0 + id as f64 * 0.02, apex_mz: 100.0 + id as f64 * 0.02,
        charge: 1, apex_rt: 10.0, rt_start: 8.0, rt_end: 12.0,
        monoisotopic_area: 100.0, total_area: 100.0, apex_intensity: 100.0,
        fwhm: 2.0, snr: 10.0, isotope_count: 1, linked_ms2_ids: vec![],
    }).collect();
    let scans: Vec<_> = (0..2_000).map(|i| Ms2PrecursorInfo {
        scan_id: i, precursor_mz: features[(i * 7919) % features.len()].monoisotopic_mz,
        rt: 10.0, charge: 1,
    }).collect();
    let tolerance = MassTolerance::Ppm(10.0);
    let start = Instant::now();
    let mut expected = Vec::new();
    for scan in &scans {
        for feature in &features {
            if scan.rt >= feature.rt_start - 1.0 && scan.rt <= feature.rt_end + 1.0
                && (feature.monoisotopic_mz - scan.precursor_mz).abs() <= tolerance.tolerance_at(scan.precursor_mz) {
                expected.push((scan.scan_id, feature.id));
            }
        }
    }
    let exhaustive = start.elapsed();
    let mut table = FeatureTable::new(features);
    let start = Instant::now();
    let links = table.link_ms2_precursors_with_tolerance(&scans, tolerance);
    let indexed = start.elapsed();
    let actual: Vec<_> = links.iter().map(|l| (l.scan_id, l.feature_id)).collect();
    assert_eq!(expected, actual);
    println!("features=50000 scans=2000 identical_links={} exhaustive_ms={:.3} indexed_ms={:.3} speedup={:.2}",
        links.len(), exhaustive.as_secs_f64()*1000.0, indexed.as_secs_f64()*1000.0,
        exhaustive.as_secs_f64()/indexed.as_secs_f64());
}
