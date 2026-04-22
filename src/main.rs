#![allow(warnings)]

#[cfg(feature = "ml")]
use burn::backend::Autodiff;
#[cfg(feature = "ml")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "ml")]
use burn::data::dataloader::DataLoaderBuilder;
#[cfg(feature = "ml")]
use burn::data::dataset::{Dataset, InMemDataset};
#[cfg(feature = "ml")]
use ms_tools::machine_learning;
#[cfg(feature = "ml")]
use ms_tools::machine_learning::dataloader::{MsBatcher, MsTrainingItem};

#[cfg(feature = "ml")]
use ms_tools::machine_learning::model::train_model;

use nohash_hasher::IntMap;
use oxidems_lib::clustering::kde::{
    Kde, deconvolve_gaussian, detect_peaks_cwt, epanechnikov_kernel, gaussian_kernel,
};
use oxidems_lib::plotting::SpectrumAnalyzer;
use plotters::prelude::*;
use rand::prelude::SliceRandom;
use std::str::FromStr;

use std::fs::File;
use std::io::Write;

// Define your concrete backend types
// Wgpu uses your GPU (Mac Metal, Windows DX/Vulkan, Linux Vulkan) out of the box
#[cfg(feature = "ml")]
type MyBackend = Wgpu<f32, i32>;
#[cfg(feature = "ml")]
type MyAutodiffBackend = Autodiff<MyBackend>;

use itertools::Itertools;
// use oxidems_lib::clustering::cluster::SpectralExperiment;
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::clustering::ms_rt::*;
#[cfg(feature = "plotting")]
use oxidems_lib::plotting::SpectrumPlotter;
use oxidems_lib::{
    BinnedSpectrumConfig, MSNSpectrum, MassSpectralRecord, MassSpectrum, SpectrumComparison,
    clustering::{cluster::ClusterEngine, ms_rt::*},
    io::mgf::{MgfParser, MgfParserConfig},
    workflow::*,
};

// --- 2. Plotting Logic ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prefix = "tests/".to_string();
    let folder_name = "PXD023047";
    // let folder_name = "PXD021518";
    let path = prefix + folder_name;

    let mut spectral_experiment = SpectralExperiment::proteomics();
    spectral_experiment.engine.exact = true;

    spectral_experiment.ingest_files_parallel(path.as_str());
    spectral_experiment.prepare_clustering();

    let clusters = spectral_experiment.cluster();

    // 1. Prepare the sorted data (simulating a bimodal distribution)
    let mut data: Vec<f64> = clusters
        .iter()
        .max_by(|cluster1, cluster2| cluster1.len().cmp(&cluster2.len()))
        .unwrap()
        .into_iter()
        .map(|id| spectral_experiment.metadata[*id].precursor_mz)
        .collect();

    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let steps = 300;
    let x_min = data[0];
    let x_max = *data.iter().last().unwrap();
    let step_size = (x_max - x_min) / (steps as f64);

    println!("Data: {:?}", data);
    let bandwidth = 0.1;
    let mut kde_gaussian = Kde::new(&data, bandwidth, gaussian_kernel);
    kde_gaussian.estimate_bandwidth_scott();
    let mut kde_epan = Kde::new(&data, bandwidth, epanechnikov_kernel);
    kde_epan.estimate_bandwidth_scott();

    let root = BitMapBackend::new("kde_output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create a 2D Cartesian coordinate system
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Kernel Density Estimation (Bandwidth = {})", bandwidth),
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        // X-axis from 0 to 7, Y-axis from 0 to 0.5 (density)
        .build_cartesian_2d(x_min..x_max, 0f64..1.0f64)?;

    chart
        .configure_mesh()
        .x_desc("Value")
        .y_desc("Density")
        .draw()?;

    let mut gaussian_points = Vec::with_capacity(steps + 1);
    let mut epan_points = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let x = x_min + (i as f64) * step_size;
        gaussian_points.push((x, kde_gaussian.estimate(x)));
        epan_points.push((x, kde_epan.estimate_finite_support(x)));
    }
    let gaussian_max = *gaussian_points
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    gaussian_points
        .iter_mut()
        .for_each(|point| point.1 /= gaussian_max.1);
    let epan_max = *epan_points
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    epan_points
        .iter_mut()
        .for_each(|point| point.1 /= epan_max.1);

    // Run the deconvolution!
    // Since we normalized the max height to 1.0, a noise threshold of 0.05
    // means "ignore peaks that are less than 5% of the height of the main peak".
    let deconvolved_peaks = deconvolve_gaussian(
        &gaussian_points,
        kde_gaussian.bandwidth,
        10,   // Extract up to 10 distinct clusters
        0.05, // 5% noise threshold
    );

    // Assuming your gaussian_points has 300 steps...
    // If a cluster spans roughly 5% of your total X-axis, that's ~15 steps.
    // Try a scale related to that width.
    let target_scale = 300.0 * (kde_gaussian.bandwidth / (x_max - x_min));
    println!("Target scale: {}", target_scale);

    let cwt_peaks = detect_peaks_cwt(&epan_points, target_scale as f64);

    println!("CWT Detected {} peaks:", cwt_peaks.len());
    for peak_x in cwt_peaks {
        println!("Peak found at: {:.5}", peak_x);
    }

    // println!("Deconvolved Clusters found: {}", deconvolved_peaks.len());
    // for (i, peak) in deconvolved_peaks.iter().enumerate() {
    //     println!("Cluster {}: center = {:.5}, relative height = {:.2}", i + 1, peak.x, peak.height);
    // }

    // let gaussian_criticals = kde_epan.detect_ridges(300);
    // let epan_criticals = kde_gaussian.detect_ridges(300);

    // 4. Draw Gaussian Series
    chart
        .draw_series(LineSeries::new(gaussian_points, &BLUE.mix(0.8)))?
        .label("Gaussian Kernel")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // 5. Draw Epanechnikov Series
    chart
        .draw_series(LineSeries::new(epan_points, &RED.mix(0.8)))?
        .label("Epanechnikov Kernel")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 6. Draw Rug Plot (Data points on the X-axis)
    chart
        .draw_series(data.iter().map(|&x| {
            // Draw small vertical ticks from y=0.0 to y=0.02
            PathElement::new(vec![(x, 0.0), (x, 0.02)], &BLACK)
        }))?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    // 7. Render Legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    // Ensure everything is flushed to the file
    root.present()?;
    println!("Plot successfully saved to 'kde_output.png'");

    Ok(())
}

// fn main() {
//     let start_time = std::time::Instant::now();
//     let mut spectral_experiment = SpectralExperiment::proteomics();
//     spectral_experiment.engine.exact = true;
//
//     let folder_name = "PXD000561";
//     // let folder_name = "Astral";
//     let prefix = "tests/".to_string();
//     // let folder_name = "PXD023047";
//     // let folder_name = "PXD021518";
//     let path = prefix + folder_name;
//
//     // spectral_experiment.ingest_files(path.as_str());
//     spectral_experiment.ingest_files_parallel(path.as_str());
//
//     println!(
//         "Total spectra parsed: {}, after {:.3} s",
//         spectral_experiment.total_spectra_ingested,
//         start_time.elapsed().as_secs_f32(),
//     );
//
//     println!(
//         "Spectra remaining after filtering: {}, after {:.3} s",
//         spectral_experiment.spectra.len(),
//         start_time.elapsed().as_secs_f32(),
//     );
//
//     let mz_tol = 0.01;
//     let rt_tol = 65.0;
//
//     spectral_experiment.prepare_clustering();
//     // let mut cluster_sizes = Vec::new();
//     // let mut parameters = Vec::new();
//     // let mut n10_values = Vec::new();
//     // let mut average_purities = Vec::new();
//     // let mut run_times = Vec::new();
//     // let epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0];
//     let epsilon_values = [0.3];
//
//     // let mut current: f32 = 0.004;
//     // let epsilon_values = (1..9)
//     //     .map(|int| {
//     //         current *= 2.0;
//     //         current.min(1.0)
//     //     })
//     //     .collect::<Vec<f32>>();
//
//     // let epsilon_values = [0.1];
//     // let epsilon_values = [0.1];
//
//     // let hash_sizes = [32, 64, 128, 256, 512, 800, 1024, 2048, 4096];
//     let hash_sizes = [800];
//
//     for (epsilon_value, hash_size) in epsilon_values
//         .into_iter()
//         .cartesian_product(hash_sizes.into_iter())
//     {
//         let mut start_time = std::time::Instant::now();
//         spectral_experiment.engine.epsilon = epsilon_value;
//         spectral_experiment.engine.set_hash_size(hash_size);
//         #[cfg(feature = "gpu")]
//         spectral_experiment.engine.enable_gpu(1);
//         spectral_experiment.spectra = Vec::new();
//
//         let clusters = spectral_experiment.cluster();
//         spectral_experiment.write_results(
//             &clusters,
//             format!(
//                 "exact_cluster_assignments_{}_{}.csv",
//                 epsilon_value, hash_size
//             )
//             .as_str(),
//         );
//
//         // println!(
//         //     "Done with clustering after {:.3}s",
//         //     start_time.elapsed().as_secs_f32()
//         // );
//         // let mut end_time = start_time.elapsed().as_secs_f32();
//         // run_times.push(end_time);
//
//         // parameters.push(format!(
//         //     "Epsilon: {} Hash Size: {}",
//         //     epsilon_value, hash_size as f32
//         // ));
//         // cluster_sizes.push(
//         //     clusters
//         //         .iter()
//         //         .map(|cluster| cluster.len())
//         //         .collect::<Vec<usize>>(),
//         // );
//
//         // println!(
//         //     "Max cluster size: {}\nAvg cluster size: {}\nMin cluster size: {}\nNumber of clusters: {}\nTotal number of spectra clustered: {}",
//         //     clusters.max_cluster_size(),
//         //     clusters.avg_cluster_size(),
//         //     clusters.min_cluster_size(),
//         //     clusters.iter().count(),
//         //     clusters.iter().map(|cluster| cluster.len()).sum::<usize>()
//         // );
//
//         // let avg_weighted_purity =
//         //     spectral_experiment.average_weighted_cluster_purity(&clusters, mz_tol, rt_tol);
//         // let avg_purity = spectral_experiment.average_cluster_purity(&clusters, mz_tol, rt_tol);
//         // average_purities.push(avg_purity);
//
//         // let n10 = spectral_experiment.calculate_n10_value(&clusters);
//         // n10_values.push(n10);
//         // println!(
//         //     "Average Weighted purity {:.5}, Average purity: {:.5}, N10 Value: {}",
//         //     avg_weighted_purity, avg_purity, n10
//         // );
//         // // println!("Run time: {}", end_time);
//     }
//
//     // println!("Run times: {:?}", run_times);
//
//     // plot_multiple_cluster_distributions(
//     //     &cluster_sizes,
//     //     &parameters,
//     //     spectral_experiment.total_spectra_ingested,
//     //     format!("plots/spectral_comparison_{}_eps.svg", folder_name).as_str(),
//     // );
//
//     // plot_purity_vs_n10(
//     //     &average_purities,
//     //     &n10_values,
//     //     &parameters,
//     //     format!("plots/purity_plot_{}_eps.svg", folder_name).as_str(),
//     // );
//
//     // let analyzer = SpectrumAnalyzer::new(&spectral_experiment.spectra);
//     // let prefix = "raw";
//     // analyzer.plot_mz_kde(format!("plots/mz_kde_{}.png", prefix).as_str());
//     // analyzer.plot_peak_counts(format!("plots/peak_counts_{}.png", prefix).as_str());
//     // analyzer.plot_feature_map_log(format!("plots/feature_map_log_{}.png", prefix).as_str());
//     // analyzer.plot_tic(format!("plots/total_tic_{}.png", prefix).as_str());
//     // analyzer.plot_kendrick_mass_defect(format!("plots/mass_defect_{}.png", prefix).as_str());
//     // analyzer.plot_3d_landscape(format!("plots/intensity_landscape_{}.png", prefix).as_str());
//     // analyzer.plot_intensity_distribution(
//     //     format!("plots/intensity_distribution_{}.png", prefix).as_str(),
//     // );
//
//     // analyzer.plot_charge_distribution(format!("plots/charge_distribution_{}.png", prefix).as_str());
//     // analyzer.plot_mz_rt_by_charge(format!("plots/mz_rt_charge_{}.png", prefix).as_str());
//     // analyzer.plot_neutral_mass_rt(format!("plots/neutral_mass_rt_charge_{}.png", prefix).as_str());
//
//     // let analyzer = SpectrumAnalyzer::new(&msn_spectra);
//     // let prefix = "filtered";
//     // analyzer.plot_mz_kde(format!("mz_kde_{}.png", prefix).as_str());
//     // analyzer.plot_peak_counts(format!("peak_counts_{}.png", prefix).as_str());
//     // analyzer.plot_feature_map_log(format!("feature_map_log_{}.png", prefix).as_str());
//     // analyzer.plot_tic(format!("total_tic_{}.png", prefix).as_str());
//
//     //     let start_time = std::time::Instant::now();
//
//     //     let mz_tolerance = MassTolerance::Ppm(20.0);
//     //     let mut cluster_engine = ClusterEngine::default(); //, Some(msn_spectra.len()));
//     //     cluster_engine.split_on_mz_and_charge(&mut msn_spectra);
//
//     //     println!(
//     //         "Done with splitting spectra and writing temp files: {}s",
//     //         start_time.elapsed().as_secs_f32()
//     //     );
//
//     //     let spectral_map: IntMap<usize, &MSNSpectrum> = msn_spectra
//     //         .iter()
//     //         .map(|spectrum| (spectrum.id as usize, spectrum))
//     //         .collect();
//     //     let start_time = std::time::Instant::now();
//
//     //     let mut csv_file = File::create("clustering_results.csv").expect("Unable to create file");
//     //     writeln!(
//     //     csv_file,
//     //     "epsilon,hash_size,num_clusters,min_cluster_size,avg_cluster_size,max_cluster_size,total_spectra,processed_spectra,clustered_spectra,avg_weighted_purity,cluster_purities,cluster_sizes"
//     // ).expect("Failed to write CSV header");
//
//     //     let epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3];
//     //     let hash_sizes = [32, 64, 128, 256, 512, 800, 1024, 2048];
//
//     //     for (epsilon_value, hash_size) in epsilon_values
//     //         .into_iter()
//     //         .cartesian_product(hash_sizes.into_iter())
//     //     {
//     //         cluster_engine.epsilon = epsilon_value;
//     //         cluster_engine.set_hash_size(hash_size);
//     //         cluster_and_export(
//     //             &mut cluster_engine,
//     //             epsilon_value,
//     //             hash_size,
//     //             &spectral_map,
//     //             total_spectra,
//     //             msn_spectra.len(),
//     //             &file_boundaries,
//     //             &mut csv_file,
//     //         );
//     //     }
// }
//
// fn cluster_and_plot(
//     experiment: &mut SpectralExperiment,
//     epsilon: f32,
//     hash_size: usize,
//     spectral_map: &IntMap<usize, &MSNSpectrum>,
//     total_spectra: usize,
//     processed_spectra: usize,
//     file_boundaries: &[(String, usize)],
// ) {
//     let clusters = experiment.cluster();
//
//     let mut count = 0;
//     let mut clustered_spectra = 0;
//     for clusterlabel in clusters.iter() {
//         count += 1;
//         clustered_spectra += clusterlabel.len();
//     }
//
//     let average_cluster_size = clusters.avg_cluster_size();
//     let min_cluster_size = clusters.min_cluster_size();
//     let max_cluster_size = clusters.max_cluster_size();
//     println!("Number of clusters found: {}", count);
//     println!("Smallest cluster size:  {}", min_cluster_size);
//     println!("Average cluster size:  {}", average_cluster_size);
//     println!("Largest cluster size: {}", max_cluster_size);
//     println!("Number of spectra read: {}", total_spectra);
//     println!(
//         "Number of spectra processed: {} ({:.0}%)",
//         processed_spectra,
//         processed_spectra * 100 / total_spectra
//     );
//     println!(
//         "Number of spectra clustered: {} ({:.0}%)",
//         clustered_spectra,
//         clustered_spectra * 100 / total_spectra
//     );
//
//     let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();
//
//     plot_cluster_distribution(
//         &mut cluster_sizes,
//         total_spectra,
//         format!(
//             "plots/cluster_distributions_after_refinement_eps_{}_hash_len_{}.svg",
//             epsilon, hash_size
//         )
//         .as_str(),
//     );
//
//     plot_cluster_size_distribution(
//         &mut cluster_sizes,
//         format!(
//             "plots/cluster_size_distributions_after_refinement_eps_{}_hash_len_{}.svg",
//             epsilon, hash_size
//         )
//         .as_str(),
//     );
//     let mut cluster_purities: Vec<f64> = clusters
//         .iter()
//         .map(|cluster| experiment.calculate_cluster_purity(cluster, 0.01, 65.0))
//         .collect();
//
//     let results = &cluster_sizes
//         .iter()
//         .zip(cluster_purities.iter())
//         .map(|(x, y)| (*x, *y))
//         .collect::<Vec<(usize, f64)>>();
//
//     let stats = bin_results(&results);
//
//     println!(
//         "Average weighted purity after refinement at eps {} and hash len {}: {}",
//         epsilon,
//         hash_size,
//         calculate_average_purity(&results)
//     );
//     plot_msrt_benchmarks(
//         &stats,
//         format!(
//             "purity_plot_after_refinement_eps_{}_hash_len_{}.png",
//             epsilon, hash_size
//         )
//         .as_str(),
//     );
// }
//
// // ---------------------------------------------------------
//
// fn cluster_and_export(
//     experiment: &mut SpectralExperiment,
//     epsilon: f32,
//     hash_size: usize,
//     total_spectra: usize,
//     processed_spectra: usize,
//     file_boundaries: &[(String, usize)],
//     csv_file: &mut File, // Accept the CSV file
// ) {
//     let clusters = experiment.cluster();
//
//     let mut count = 0;
//     let mut clustered_spectra = 0;
//     for clusterlabel in clusters.iter() {
//         count += 1;
//         clustered_spectra += clusterlabel.len();
//     }
//
//     let average_cluster_size = clusters.avg_cluster_size();
//     let min_cluster_size = clusters.min_cluster_size();
//     let max_cluster_size = clusters.max_cluster_size();
//
//     // Print to console (optional, you can remove these if it clutters standard output)
//     print!("Processed eps: {}, hash: {}", epsilon, hash_size);
//
//     let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();
//     cluster_sizes.sort_by(|a, b| b.cmp(a));
//
//     let mut cumulative = 0;
//     let mut n10_value = 0;
//     for &cluster_size in cluster_sizes.iter() {
//         cumulative += cluster_size;
//         if cumulative >= (total_spectra * 10) / 100 && n10_value == 0 {
//             n10_value = cluster_size;
//         }
//     }
//
//     /* --- Optional: Commented out original Rust plots to save time ---
//     plot_cluster_distribution(...);
//     plot_cluster_size_distribution(...);
//     */
//
//     let mut cluster_purities: Vec<f64> = clusters
//         .iter()
//         .map(|cluster| experiment.calculate_cluster_purity(cluster, 0.01, 65.0))
//         .collect();
//
//     let results = &cluster_sizes
//         .iter()
//         .zip(cluster_purities.iter())
//         .map(|(x, y)| (*x, *y))
//         .collect::<Vec<(usize, f64)>>();
//
//     let avg_weighted_purity = calculate_average_purity(&results);
//
//     /* --- Optional: Commented out original msrt benchmark plot ---
//     let stats = bin_results(&results);
//     plot_msrt_benchmarks(...);
//     */
//
//     println!(
//         " Purity: {}, N10: {}, Average Cluster size: {}, Clustered Spectra: {}",
//         avg_weighted_purity, n10_value, average_cluster_size, cumulative
//     );
//
//     // 3. Write the metrics for this specific grid search coordinate to the CSV
//     writeln!(
//         csv_file,
//         "{},{},{},{},{},{},{},{},{},{},{:?},{:?}",
//         epsilon,
//         hash_size,
//         count,
//         min_cluster_size,
//         average_cluster_size,
//         max_cluster_size,
//         total_spectra,
//         processed_spectra,
//         clustered_spectra,
//         avg_weighted_purity,
//         cluster_purities,
//         cluster_sizes
//     )
//     .expect("Failed to write row to CSV");
// }
//
// // println!(
// //     "Done with clustering spectra: {}s",
// //     start_time.elapsed().as_secs_f32()
// // );
//
// // println!("Total time: {}s", total_time.elapsed().as_secs_f32());
//
// // let start_time = std::time::Instant::now();
//
// // let parser: MgfParser = MgfParser::default().with_parse_metadata(false);
// // let spectra: Vec<MassSpectralRecord> = parser.parse_file("test_files/test2.mgf").unwrap();
// // println!(
// //     "Time elapsed for parsing: {}",
// //     start_time.elapsed().as_millis() as f32 / 1000.0
// // );
// // let start_time = std::time::Instant::now();
// // let mut mass_spectra: Vec<MassSpectrum> = spectra
// //     .par_iter()
// //     .map(|spectrum| spectrum.to_mass_spectrum())
// //     .collect();
// // println!(
// //     "Time elapsed for conversion: {}",
// //     start_time.elapsed().as_micros() as f32 / 1000000.0
// // );
//
// // let start_time = std::time::Instant::now();
// // operation.operate(&mut mass_spectra);
//
// // println!(
// //     "Time elapsed for normalization: {}",
// //     start_time.elapsed().as_micros() as f32 / 1000000.0
// // );
//
// // let operation = FilterIntensityRelative::new(0.01);
//
// // let mut mass_spectra: Vec<MassSpectrum> = spectra
// //     .par_iter()
// //     .map(|spectrum| spectrum.to_mass_spectrum())
// //     .collect();
// // let start_time = std::time::Instant::now();
// // operation.operate(&mut mass_spectra);
//
// // println!(
// //     "Time elapsed for filtering 1% intensities: {}",
// //     start_time.elapsed().as_micros() as f32 / 1000000.0
// // );
//
// // let mut mass_spectra: Vec<MassSpectrum> = spectra
// //     .par_iter()
// //     .map(|spectrum| spectrum.to_mass_spectrum())
// //     .collect();
// // let operation = KeepTopNPeaks::new(50);
//
// // let start_time = std::time::Instant::now();
// // operation.operate(&mut mass_spectra);
//
// // println!(Add the changed_flag to your bindings. Then, capture the return value of atomicMin. In WGSL, atomicMin returns the original value that was in the memory address. If the original value is greater than the neighbor's value, it means we actually modified the label, so we trigger the flag.
// //     "Time elapsed for keep top 50 peaks: {}",
// //     start_time.elapsed().as_micros() as f32 / 1000000.0
// // );
// // //
// // #[cfg(feature = "ml")]
// // {
// //     // 1. Initialize the GPU device
// //     let device = WgpuDevice::default();
// //     println!("Starting Burn on device: {:?}", device);
// //
// //     // 2. Load the JSON data you prepared with the python/rust script
// //     println!("Loading training data from disk");
// //     let json_data = std::fs::read_to_string("data/training_data.json")
// //         .expect("Failed to read JSON file. Ensure 'data/training_data.json' exists.");
// //     let mut items: Vec<MsTrainingItem> =
// //         serde_json::from_str(&json_data).expect("Failed to parse JSON into MsTrainingItems");
// //
// //     // 3. Shuffle the data before splitting!
// //     let mut rng = rand::rng();
// //     items.shuffle(&mut rng);
// //
// //     // 4. Split into Train (80%) and Validation (20%)
// //     let split_idx = (items.len() as f32 * 0.8) as usize;
// //     let (train_items, valid_items) = items.split_at(split_idx);
// //
// //     let train_dataset = InMemDataset::new(train_items.to_vec());
// //     let valid_dataset = InMemDataset::new(valid_items.to_vec());
// //
// //     println!(
// //         "Dataset split: {} training items, {} validation items",
// //         train_dataset.len(),
// //         valid_dataset.len()
// //     );
// //
// //     // 5. Build the DataLoaders
// //     // Burn automatically injects the Autodiff backend into the train batcher
// //     // and the Inner backend into the validation batcher.
// //     let train_dataloader = DataLoaderBuilder::new(MsBatcher::new())
// //         .batch_size(64)
// //         .shuffle(42) // Provides a seed for shuffling batches every epoch
// //         .num_workers(4) // Parallelize the binning logic
// //         .build(train_dataset);
// //
// //     let valid_dataloader = DataLoaderBuilder::new(MsBatcher::new())
// //         .batch_size(64)
// //         .build(valid_dataset);
// //
// //     // 6. Launch the Training Loop!
// //     println!("Training model");
// //     train_model::<MyAutodiffBackend>(device, train_dataloader, valid_dataloader);
// //
// //     println!("Training complete. Weights are saved in /tmp/ms_isotope_detector");
// // }
//
// // // Example usage of the formula generator
// // let generator = FormulaGenerator::default().with_tolerance(0.1);
//
// // // Generate formulas for glucose mass (C6H12O6)
// // let glucose_mass = 180.063;
// // let formulas = generator.generate_possible_formulas(glucose_mass);
//
// // println!("\n=== Formula Generation ===");
// // println!("Possible formulas for mass {:.3} Da:", glucose_mass);
// // println!("Found {} formulas", formulas.len());
// // for (i, formula) in formulas.iter().take(5).enumerate() {
// //     println!(
// //         "{}. {} (mass: {:.4} Da)",
// //         i + 1,
// //         formula,
// //         formula.monoisotopic_mass()
// //     );
// // }
//
// // // Example usage of spectrum comparison
// // println!("\n=== Spectrum Comparison ===");
//
// // // Create some example spectra
// // let peaks1 = vec![
// //     (100.0, 1000.0),
// //     (150.0, 500.0),
// //     (200.0, 800.0),
// //     (250.0, 300.0),
// // ];
//
// // let peaks2 = vec![
// //     (100.2, 950.0), // Very similar to first spectrum
// //     (150.1, 480.0),
// //     (200.3, 750.0),
// //     (250.1, 280.0),
// // ];
//
// // let peaks3 = vec![
// //     (300.0, 1000.0), // Completely different spectrum
// //     (350.0, 600.0),
// //     (400.0, 400.0),
// // ];
//
// // let spectrum1 = MassSpectralRecord::from_tuples(&peaks1, 1);
// // let spectrum2 = MassSpectralRecord::from_tuples(&peaks2, 1);
// // let spectrum3 = MassSpectralRecord::from_tuples(&peaks3, 1);
//
// // // Fast comparison using binning (optimized for speed)
// // let fast_config = BinnedSpectrumConfig::fast();
//
// // let similarity_1_2 =
// //     SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &fast_config).unwrap();
// // let similarity_1_3 =
// //     SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum3, &fast_config).unwrap();
//
// // println!(
// //     "Fast cosine similarity (spectrum1 vs spectrum2): {:.4}",
// //     similarity_1_2
// // );
// // println!(
// //     "Fast cosine similarity (spectrum1 vs spectrum3): {:.4}",
// //     similarity_1_3
// // );
//
// // // Precise comparison using peak matching
// // let precise_similarity_1_2 =
// //     SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum2, 0.5);
// // let precise_similarity_1_3 =
// //     SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum3, 0.5);
//
// // println!(
// //     "Precise peak similarity (spectrum1 vs spectrum2): {:.4}",
// //     precise_similarity_1_2
// // );
// // println!(
// //     "Precise peak similarity (spectrum1 vs spectrum3): {:.4}",
// //     precise_similarity_1_3
// // );
//
// // // Demonstrate different speed/precision configs
// // println!("\n=== Speed vs Precision Comparison ===");
//
// // let precise_config = BinnedSpectrumConfig::precise();
// // let similarity_precise =
// //     SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &precise_config)
// //         .unwrap();
//
// // println!("Fast config similarity: {:.6}", similarity_1_2);
// // println!("Precise config similarity: {:.6}", similarity_precise);
// // println!("Fast config bin width: {} Da", fast_config.bin_width);
// // println!("Precise config bin width: {} Da", precise_config.bin_width);
//
// // // Test ultra-precise mode to show the trade-off
// // println!("\n=== Ultra-Precise vs Fast Trade-off ===");
// // let ultra_precise_config = BinnedSpectrumConfig {
// //     bin_width: 0.1,
// //     intensity_threshold: 0.001,
// //     max_peaks: 0,
// //     mass_tolerance: 0.01,
// //     bin_offset: None, // Auto-center
// // };
//
// // let similarity_ultra =
// //     SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &ultra_precise_config)
// //         .unwrap();
//
// // println!(
// //     "Ultra-precise (0.1 Da bins) similarity: {:.6}",
// //     similarity_ultra
// // );
// // println!(
// //     "Standard precise (0.5 Da bins) similarity: {:.6}",
// //     similarity_precise
// // );
// // println!("Fast (2.0 Da bins) similarity: {:.6}", similarity_1_2);
// // println!("\nNote: Ultra-precise mode shows lower similarity due to");
// // println!("small m/z shifts (0.1-0.3 Da) putting peaks in different bins.");
//
// // // Demonstrate bin centering
// // println!("\n=== Bin Centering Demonstration ===");
// // let test_spectrum = MassSpectralRecord::from_tuples(&[(100.0, 1000.0), (150.0, 500.0)], 1);
//
// // // Default centering (bins centered at integer values)
// // let default_config = BinnedSpectrumConfig::default(); // 1.0 Da bins, offset = 0.0
// // let binned_default = test_spectrum.to_binned(&default_config);
//
// // // Custom offset (bins offset by 0.5, so centers are at X.0)
// // let offset_config = BinnedSpectrumConfig::default().with_bin_offset(0.5);
// // let _binned_offset = test_spectrum.to_binned(&offset_config);
//
// // println!("Peak at 100.0 Da (single peak):");
// // println!(
// //     "  Weighted bin center: {:.3} Da",
// //     binned_default.bin_center(0)
// // );
//
// // println!("Peak at 150.0 Da (single peak):");
// // // Find the bin with intensity for 150 Da peak
// // let mut bin_150 = 0;
// // for (i, &intensity) in binned_default.intensities().iter().enumerate() {
// //     if intensity > 0.0 && binned_default.bin_center(i) > 140.0 {
// //         bin_150 = i;
// //         break;
// //     }
// // }
// // println!(
// //     "  Weighted bin center: {:.3} Da",
// //     binned_default.bin_center(bin_150)
// // );
//
// // // Demonstrate weighted averaging with multiple peaks in same bin
// // println!("\n--- Weighted Averaging Demo ---");
// // let mixed_peaks = vec![(100.2, 300.0), (100.3, 700.0)]; // Two peaks in same 1 Da bin
// // let mixed_spectrum = MassSpectralRecord::from_tuples(&mixed_peaks, 1);
// // let binned_mixed = mixed_spectrum.to_binned(&default_config);
//
// // // Find the bin with the mixed peaks
// // let mut mixed_bin = 0;
// // for (i, &intensity) in binned_mixed.intensities().iter().enumerate() {
// //     if intensity > 0.0 {
// //         mixed_bin = i;
// //         break;
// //     }
// // }
//
// // let weighted_center = binned_mixed.bin_center(mixed_bin);
// // let expected_weighted = (100.2 * 300.0 + 100.3 * 700.0) / (300.0 + 700.0);
// // println!("  Weighted bin center: {:.3} Da", weighted_center);
// // println!("  Expected average: {:.3} Da", expected_weighted);
// // println!(
// //     "  Total intensity: {:.0}",
// //     binned_mixed.intensities()[mixed_bin]
// // );
//
// // println!("\nWeighted bin centers provide more accurate representation of");
// // println!("where the actual signal is located within each bin!");
//
// // let generate_pattern = |formula: &str| {
// //     let formula = MolecularFormula::from_str(formula).unwrap();
//
// //     formula.isotopic_pattern(3, 1)
// // };
//
// // let mut pattern1 = generate_pattern("C6H6O");
// // let mut pattern2 = generate_pattern("C3H3O");
// // let pattern3 = generate_pattern("C1H3");
// // let pattern4 = generate_pattern("C7H9O");
// // let pattern5 = generate_pattern("C2H4O");
// // pattern2.extend_from_slice(&pattern1);
// // pattern2.extend_from_slice(&pattern3);
// // pattern2.extend_from_slice(&pattern5);
// // pattern1.extend_from_slice(&pattern4);
// // pattern1.extend_from_slice(&pattern5);
//
// // let mut spectrum1 = MassSpectrum::from_tuples(&pattern1, 1);
// // let mut spectrum2 = MassSpectrum::from_tuples(&pattern2, 1);
//
// // println!("spectrum1: {:?}", spectrum1.peaks);
// // println!("spectrum2: {:?}", spectrum2.peaks);
//
// // let operation = Normalize {};
//
// // operation.operate_single(&mut spectrum1);
// // operation.operate_single(&mut spectrum2);
//
// // let plotter = SpectrumPlotter::new().match_tolerance(0.05);
//
// // plotter.plot_single(&spectrum1, "plotted_spectrum.png");
// // plotter.plot_mirror(&spectrum1, &spectrum2, "plotted_spectrum_comparison.png");
