use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::plotting::SpectrumAnalyzer;
use oxidems_lib::spectrum::MSNSpectrum;
use std::path::Path;
use std::{env, fs};

fn load_from_path(path: &str) -> Vec<MSNSpectrum> {
    let mut parser = MgfParser::default();
    let p = Path::new(path);

    if p.is_dir() {
        let mut all = Vec::new();
        let entries = fs::read_dir(p).expect("cannot read directory");
        for entry in entries.flatten() {
            let fp = entry.path();
            if fp.extension().and_then(|e| e.to_str()) == Some("mgf") {
                let path_str = fp.to_string_lossy().to_string();
                match parser.parse_file(&path_str) {
                    Ok(records) => {
                        all.extend(records.into_iter().map(|r| r.to_msn()));
                    }
                    Err(e) => eprintln!("skip {path_str}: {e}"),
                }
            }
        }
        all
    } else {
        parser
            .parse_file(path)
            .expect("cannot parse file")
            .into_iter()
            .map(|r| r.to_msn())
            .collect()
    }
}

fn print_stats(spectra: &[MSNSpectrum]) {
    let n = spectra.len();
    if n == 0 {
        println!("No spectra loaded.");
        return;
    }

    let peak_counts: Vec<usize> = spectra.iter().map(|s| s.spectrum.peaks.len()).collect();
    let total_peaks: usize = peak_counts.iter().sum();
    let min_peaks = *peak_counts.iter().min().unwrap();
    let max_peaks = *peak_counts.iter().max().unwrap();
    let mean_peaks = total_peaks as f64 / n as f64;

    let mzs: Vec<f64> = spectra.iter().map(|s| s.precursor_mz).collect();
    let min_mz = mzs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mz = mzs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_mz = mzs.iter().sum::<f64>() / n as f64;

    let rts: Vec<f32> = spectra.iter().map(|s| s.retention_time_in_seconds).collect();
    let min_rt = rts.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_rt = rts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let charges: Vec<i8> = spectra.iter().map(|s| s.charge()).collect();
    let mut charge_counts: std::collections::BTreeMap<i8, usize> =
        std::collections::BTreeMap::new();
    for c in &charges {
        *charge_counts.entry(*c).or_insert(0) += 1;
    }

    let total_intensity: f64 = spectra
        .iter()
        .flat_map(|s| s.spectrum.peaks.iter())
        .map(|p| p.intensity as f64)
        .sum();

    let mut all_intensities: Vec<f32> = spectra
        .iter()
        .flat_map(|s| s.spectrum.peaks.iter())
        .map(|p| p.intensity)
        .collect();
    all_intensities.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_intensity = if all_intensities.is_empty() {
        0.0
    } else {
        all_intensities[all_intensities.len() / 2]
    };

    println!("=== Dataset Overview ===");
    println!("Spectra:          {n}");
    println!("Total peaks:      {total_peaks}");
    println!("Peaks per spec:   min={min_peaks}  max={max_peaks}  mean={mean_peaks:.1}");
    println!("Precursor m/z:    min={min_mz:.2}  max={max_mz:.2}  mean={mean_mz:.2}");
    println!("Retention time:   min={min_rt:.1}s  max={max_rt:.1}s");
    println!(
        "Total intensity:  {:.3e}",
        total_intensity
    );
    println!("Median peak int:  {median_intensity:.1}");
    println!("\nCharge state distribution:");
    for (charge, count) in &charge_counts {
        let pct = *count as f64 / n as f64 * 100.0;
        println!("  z={charge:+}: {count} ({pct:.1}%)");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = match args.get(1) {
        Some(p) => p.as_str(),
        None => {
            eprintln!("Usage: dataset_overview <file.mgf or directory/>");
            std::process::exit(1);
        }
    };

    let out_dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");

    println!("Loading spectra from: {path}");
    let spectra = load_from_path(path);
    println!("Loaded {} spectra\n", spectra.len());

    print_stats(&spectra);

    if spectra.is_empty() {
        return;
    }

    let analyzer = SpectrumAnalyzer::new(&spectra);

    let plots: &[(&str, fn(&SpectrumAnalyzer, &str) -> Result<(), Box<dyn std::error::Error>>)] = &[
        ("mz_rt.png", |a, p| a.plot_mz_rt(p)),
        ("mz_kde.png", |a, p| a.plot_mz_kde(p)),
        ("peak_counts.png", |a, p| a.plot_peak_counts(p)),
        ("feature_map.png", |a, p| a.plot_feature_map(p)),
        ("feature_map_log.png", |a, p| a.plot_feature_map_log(p)),
        ("tic.png", |a, p| a.plot_tic(p)),
        ("charge_distribution.png", |a, p| a.plot_charge_distribution(p)),
        ("mz_rt_by_charge.png", |a, p| a.plot_mz_rt_by_charge(p)),
        ("neutral_mass_rt.png", |a, p| a.plot_neutral_mass_rt(p)),
        ("kendrick_mass_defect.png", |a, p| {
            a.plot_kendrick_mass_defect(p)
        }),
        ("intensity_distribution.png", |a, p| {
            a.plot_intensity_distribution(p)
        }),
        ("3d_landscape.png", |a, p| a.plot_3d_landscape(p)),
    ];

    fs::create_dir_all(out_dir).expect("cannot create output directory");
    println!("\nGenerating plots in: {out_dir}");
    for (filename, plot_fn) in plots {
        let out_path = format!("{out_dir}/{filename}");
        match plot_fn(&analyzer, &out_path) {
            Ok(()) => println!("  wrote {out_path}"),
            Err(e) => eprintln!("  failed {out_path}: {e}"),
        }
    }

    println!("\nDone.");
}
