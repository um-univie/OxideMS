use oxidems_lib::feature_extraction::*;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(mzml_path) = args.get(1) else {
        eprintln!("usage: run_feature_finder <input .mzML>");
        std::process::exit(2)
    };

    println!("=========================================================================");
    println!("             OxideMS High-Performance Feature Finder                     ");
    println!("=========================================================================\n");

    println!("Target mzML File: {}", mzml_path);

    // Auto-detect or select instrument preset
    let is_astral = mzml_path.to_lowercase().contains("astral");
    let is_orbitrap = mzml_path.to_lowercase().contains("orbitrap") || is_astral;

    let config = if is_astral {
        println!("Preset Selected:  Orbitrap Astral (high scan rate, narrow FWHM)");
        FeatureFinderConfig::astral()
    } else if is_orbitrap {
        println!("Preset Selected:  Orbitrap Metabolomics (5 ppm, clean baseline)");
        FeatureFinderConfig::metabolomics_orbitrap()
    } else {
        println!("Preset Selected:  Q-TOF Metabolomics (10 ppm, Bruker / Agilent)");
        FeatureFinderConfig::metabolomics_qtof()
    };

    let finder = FeatureFinder::new(config);

    println!("\nExecuting end-to-end selective ingestion & feature finding pipeline...");
    let t0 = Instant::now();
    let table = finder
        .find_features_from_file(mzml_path)
        .expect("Failed to execute feature finder on mzML");
    let total_elapsed = t0.elapsed();

    println!("\n=========================================================================");
    println!("                           PERFORMANCE SUMMARY                           ");
    println!("=========================================================================");
    println!("Total End-to-End Time:    {:.3} s", total_elapsed.as_secs_f64());
    println!("MS1 Scan Throughput:      {:.1} scans / sec", table.total_ms1_scans as f64 / total_elapsed.as_secs_f64().max(1e-4));
    println!("Overall Spectra Rate:     {:.0} spectra / sec", (table.total_ms1_scans + table.total_ms2_scans) as f64 / total_elapsed.as_secs_f64().max(1e-4));
    println!("=========================================================================\n");

    table.print_summary();

    // Show top 10 features by area
    let mut sorted_features = table.features.clone();
    sorted_features.sort_by(|a, b| b.total_area.total_cmp(&a.total_area));

    println!("\nTop 10 Most Abundant Features Detected:");
    println!("{:<6} | {:<10} | {:<8} | {:<5} | {:<12} | {:<6} | {:<5} | {:<7}",
        "ID", "m/z", "RT (s)", "z", "Total Area", "FWHM", "SNR", "MS2 IDs");
    println!("-------------------------------------------------------------------------");
    for f in sorted_features.iter().take(10) {
        let ms2_count = f.linked_ms2_ids.len();
        let ms2_preview = if ms2_count > 0 {
            format!("{} spectra", ms2_count)
        } else {
            "-".to_string()
        };
        println!("{:<6} | {:<10.4} | {:<8.1} | {:<+5} | {:<12.1} | {:<6.2} | {:<5.1} | {:<7}",
            f.id, f.monoisotopic_mz, f.apex_rt, f.charge, f.total_area, f.fwhm, f.snr, ms2_preview);
    }

    // Save feature table to CSV
    let output_csv = "features_output.csv";
    table.write_to_file(output_csv).expect("Failed to write CSV");
    println!("\nFeature table exported to '{}' ({} rows)", output_csv, table.len());
}
