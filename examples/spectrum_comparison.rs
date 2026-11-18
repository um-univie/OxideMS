use oxidems_lib::clustering::cluster::SpectralExperiment;
use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::workflow::{FilterIntensityRelative, MinMZRangeFilter, WorkFlowManager};
use oxidems_lib::{BinnedSpectrumConfig, MassSpectralRecord, SpectrumComparison};

use oxidems_lib::plotting::SpectrumPlotter;

mod support;

fn main() {
    support::require("spectrum_comparison", &[("tests/Astral/029_Sa04_Methanol_POS.mgf", support::ASTRAL)]);
    let mut peaks1: Vec<(f32, f32)> = vec![
        (75.0, 100.0),
        (100.0, 1000.0),
        (130.0, 500.0),
        (150.0, 500.0),
        (200.0, 800.0),
        (250.0, 300.0),
    ];
    let mut peaks2: Vec<(f32, f32)> = vec![
        (100.2, 1000.0),
        (150.1, 480.0),
        (200.3, 750.0),
        (250.1, 280.0),
        (350.1, 360.0),
    ];

    peaks1.iter_mut().for_each(|peak| peak.1 /= 1000.0);
    peaks2.iter_mut().for_each(|peak| peak.1 /= 1000.0);

    let peaks3: Vec<(f32, f32)> = vec![(300.0, 1000.0), (350.0, 600.0), (400.0, 400.0)];

    let spectrum1 = MassSpectralRecord::from_tuples(&peaks1);
    let spectrum2 = MassSpectralRecord::from_tuples(&peaks2);
    let spectrum3 = MassSpectralRecord::from_tuples(&peaks3);

    let fast_config = BinnedSpectrumConfig::fast();
    let similarity_1_2 =
        SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &fast_config).unwrap();
    let similarity_1_3 =
        SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum3, &fast_config).unwrap();

    let plotter = SpectrumPlotter::new()
        .dimensions(1200, 600)
        .match_tolerance(0.5)
        .title("Mirror Plot");

    plotter
        .plot_mirror(
            &spectrum1.to_mass_spectrum(),
            &spectrum2.to_mass_spectrum(),
            "mirror.png",
        )
        .unwrap();

    println!("=== Spectrum Comparison ===");
    println!(
        "Fast cosine similarity (spectrum1 vs spectrum2): {:.4}",
        similarity_1_2
    );
    println!(
        "Fast cosine similarity (spectrum1 vs spectrum3): {:.4}",
        similarity_1_3
    );

    let precise_similarity_1_2 =
        SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum2, 0.5);
    let precise_similarity_1_3 =
        SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum3, 0.5);

    println!(
        "Precise peak similarity (spectrum1 vs spectrum2): {:.4}",
        precise_similarity_1_2
    );
    println!(
        "Precise peak similarity (spectrum1 vs spectrum3): {:.4}",
        precise_similarity_1_3
    );

    println!("\n=== Speed vs Precision Comparison ===");
    let precise_config = BinnedSpectrumConfig::precise();
    let similarity_precise =
        SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &precise_config)
            .unwrap();

    println!("Fast config similarity:    {:.6}", similarity_1_2);
    println!("Precise config similarity: {:.6}", similarity_precise);
    println!("Fast config bin width:    {} Da", fast_config.bin_width);
    println!("Precise config bin width: {} Da", precise_config.bin_width);

    println!("\n=== Bin Centering Demonstration ===");
    let default_config = BinnedSpectrumConfig::default();
    let mixed_peaks: Vec<(f32, f32)> = vec![(100.2, 300.0), (100.3, 700.0)];
    let mixed_spectrum = MassSpectralRecord::from_tuples(&mixed_peaks);
    let binned_mixed = mixed_spectrum.to_binned(&default_config);

    let weighted_center = binned_mixed.bin_center(0);
    let expected_weighted = (100.2 * 300.0 + 100.3 * 700.0) / (300.0 + 700.0);
    println!("Weighted bin center: {:.3} Da", weighted_center);
    println!("Expected average:    {:.3} Da", expected_weighted);
    println!("Total intensity:     {:.0}", binned_mixed.intensities()[0]);

    let mut experiment = SpectralExperiment::metabolomics();
    let mut parser = MgfParser::default();
    let spectra = parser
        .parse_file("tests/Astral/029_Sa04_Methanol_POS.mgf")
        .unwrap();
    let max_spectrum = spectra
        .into_iter()
        .max_by(|a, b| a.peaks().len().cmp(&b.peaks().len()))
        .unwrap();

    let spectrum2 = max_spectrum.clone();
    let mut spectral_vec = vec![max_spectrum];
    experiment
        .workflow
        .remove_operation::<FilterIntensityRelative>();
    experiment.workflow.remove_operation::<MinMZRangeFilter>();
    experiment.workflow.execute(&mut spectral_vec);
    plotter
        .plot_mirror(
            &spectral_vec[0].to_mass_spectrum(),
            &spectrum2.to_mass_spectrum(),
            "cleanup_mirror.png",
        )
        .unwrap();
}
