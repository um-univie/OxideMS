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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prefix = "tests/".to_string();
    let folder_name = "PXD023047";
    // let folder_name = "PXD021518";
    let path = prefix + folder_name;

    let mut spectral_experiment = SpectralExperiment::proteomics();
    spectral_experiment.engine.exact = true;

    //spectral_experiment.ingest_files_parallel(path.as_str());
    spectral_experiment.ingest_files(path.as_str());
    spectral_experiment.prepare_clustering();

    let clusters = spectral_experiment.cluster();

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

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Kernel Density Estimation (Bandwidth = {})", bandwidth),
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
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

    let deconvolved_peaks = deconvolve_gaussian(
        &gaussian_points,
        kde_gaussian.bandwidth,
        10,   // Extract up to 10 distinct clusters
        0.05, // 5% noise threshold
    );

    let target_scale = 300.0 * (kde_gaussian.bandwidth / (x_max - x_min));
    println!("Target scale: {}", target_scale);

    let cwt_peaks = detect_peaks_cwt(&epan_points, target_scale as f64);

    println!("CWT Detected {} peaks:", cwt_peaks.len());
    for peak_x in cwt_peaks {
        println!("Peak found at: {:.5}", peak_x);
    }

    chart
        .draw_series(LineSeries::new(gaussian_points, &BLUE.mix(0.8)))?
        .label("Gaussian Kernel")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(epan_points, &RED.mix(0.8)))?
        .label("Epanechnikov Kernel")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(data.iter().map(|&x| {
            PathElement::new(vec![(x, 0.0), (x, 0.02)], &BLACK)
        }))?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Plot successfully saved to 'kde_output.png'");

    Ok(())
}
