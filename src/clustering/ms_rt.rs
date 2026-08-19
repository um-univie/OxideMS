use crate::{MSNSpectrum, clustering::cluster::SpectrumMetadata};
use nohash_hasher::IntMap;
use plotters::prelude::*;
use std::{
    collections::BTreeMap,
    fs::Metadata,
    hash::{Hash, Hasher},
};

#[derive(Default)]
pub struct BinStats {
    total_purity: f64,
    count: usize,
}

pub fn bin_results(purity_results: &[(usize, f64)]) -> BTreeMap<usize, BinStats> {
    let mut bins = BTreeMap::new();

    for &(size, purity) in purity_results {
        // Find the upper power of 2 for the bin (1, 2, 4, 8, 16...)
        let bin_key = if size == 1 {
            1
        } else {
            size.next_power_of_two()
        };

        let stats = bins.entry(bin_key).or_insert(BinStats::default());
        stats.total_purity += purity;
        stats.count += 1;
    }
    bins
}

pub fn plot_msrt_benchmarks(
    purity_data: &BTreeMap<usize, BinStats>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let bin_keys: Vec<usize> = purity_data.keys().cloned().collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("Cluster Purity by Size Bin", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (0..bin_keys.len()).into_segmented(), // Keep discrete for the bars
            0.0..1.05f64,
        )?;

    chart
        .configure_mesh()
        .y_desc("Average Purity")
        .x_desc("Cluster Size Range")
        .x_label_formatter(&|idx| {
            match idx {
                SegmentValue::CenterOf(i) => {
                    // Safely check if the index exists in our bin_keys
                    if let Some(&upper) = bin_keys.get(*i) {
                        let lower = if *i == 0 { 1 } else { bin_keys[*i - 1] + 1 };
                        if upper == lower {
                            format!("{}", upper)
                        } else {
                            format!("{}-{}", lower, upper)
                        }
                    } else {
                        "".to_string() // Return empty for out-of-bounds ticks
                    }
                }
                _ => "".to_string(),
            }
        })
        .draw()?;

    chart.draw_series(bin_keys.iter().enumerate().map(|(idx, &bin)| {
        let stats = &purity_data[&bin];
        let val = stats.total_purity / stats.count as f64;
        // Use Rectangle instead of Histogram helper for more style control
        Rectangle::new(
            [
                (SegmentValue::Exact(idx), 0.0),
                (SegmentValue::Exact(idx + 1), val),
            ],
            BLUE.mix(0.4).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

pub fn plot_cluster_distribution(
    cluster_sizes: &mut [usize],
    total_spectra: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    cluster_sizes.sort_by(|a, b| b.cmp(a)); // Descending sort

    let mut cumulative = 0;
    let threshold = total_spectra as f32 * 0.1;
    let mut n10_cluster_size = 0;
    let curve_points: Vec<(usize, f64)> = cluster_sizes
        .iter()
        .enumerate()
        .map(|(idx, &size)| {
            cumulative += size;
            if n10_cluster_size == 0 && cumulative as f32 >= threshold {
                n10_cluster_size = size;
            }
            (idx, cumulative as f64 / total_spectra as f64)
        })
        .collect();

    println!("N10 cluster size: {}", n10_cluster_size);

    let mut chart = ChartBuilder::on(&root)
        .caption("Cumulative Spectra Coverage", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..cluster_sizes.len(), 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(curve_points, &RED))?;

    // Draw a horizontal line at 0.1 (10%) to visualize N10
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0, 0.1), (cluster_sizes.len(), 0.1)],
        &BLACK,
    )))?;

    Ok(())
}

pub fn plot_cluster_size_distribution(
    cluster_sizes: &[usize],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if cluster_sizes.is_empty() {
        return Ok(());
    }

    let max_size = *cluster_sizes.iter().max().unwrap_or(&1);

    // 1. Determine how many powers of 2 we need
    // e.g., if max_size is 100, log2 is ~6.6, so we need 7-8 bins.
    let num_bins = (max_size as f64).log2().ceil() as usize + 1;
    let mut bins = vec![0usize; num_bins];

    // bin_edges will be [1.0, 2.0, 4.0, 8.0, ...]
    let bin_edges: Vec<f64> = (0..=num_bins).map(|i| 2f64.powi(i as i32)).collect();

    // 2. Assign sizes to bins using leading_zeros for speed
    // or just checking the log2 value
    for &size in cluster_sizes {
        if size == 0 {
            continue;
        }
        let bin_idx = (size as f64).log2().floor() as usize;
        if bin_idx < num_bins {
            bins[bin_idx] += 1;
        }
    }

    let max_freq = *bins.iter().max().unwrap_or(&1);

    // 3. Build the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Cluster Size Distribution (Log2 Bins)", ("sans-serif", 35))
        .margin(25)
        .x_label_area_size(50)
        .y_label_area_size(60)
        // We use log_scale with base 2 for the X axis
        .build_cartesian_2d((1.0..(bin_edges[num_bins])).log_scale(), 0..max_freq)?;

    chart
        .configure_mesh()
        .x_desc("Cluster Size (Power of 2)")
        .y_desc("Frequency")
        // Custom formatter to show clean integers on the log axis
        .x_label_formatter(&|x| format!("{:.0}", x))
        .draw()?;

    // 4. Draw the bars
    chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
        let x0 = bin_edges[i];
        let x1 = bin_edges[i + 1];

        Rectangle::new([(x0, 0), (x1, count)], CYAN.mix(0.5).filled())
    }))?;

    root.present()?;
    Ok(())
}

pub fn calculate_average_purity(cluster_results: &[(usize, f64)]) -> f64 {
    cluster_results
        .iter()
        .map(|result| result.0 as f64 * result.1)
        .sum::<f64>()
        / cluster_results
            .iter()
            .map(|result| result.0 as f64)
            .sum::<f64>()
}
fn are_from_same_file(boundaries: &[(String, usize)], id1: usize, id2: usize) -> bool {
    get_file_index(boundaries, id1) == get_file_index(boundaries, id2)
}

fn get_index_if_same_file(boundaries: &[(String, usize)], id1: usize, id2: usize) -> Option<usize> {
    let index1 = get_file_index(boundaries, id1);
    let index2 = get_file_index(boundaries, id2);
    if index1 == index2 { Some(index1) } else { None }
}

fn get_file_index(boundaries: &[(String, usize)], id: usize) -> usize {
    match boundaries.binary_search_by_key(&id, |&(_, end_id)| end_id) {
        Ok(index) => index + 1,
        Err(index) => index,
    }
}
pub fn plot_multiple_cluster_distributions(
    datasets: &[Vec<usize>],
    parameters: &[String],
    total_spectra: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_clusters = datasets.iter().map(|d| d.len()).max().unwrap_or(0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Parameter Influence on Spectra Coverage",
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..max_clusters, 0.0..1.05)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Number of Clusters (Ranked by Size)")
        .y_desc("Cumulative Fraction of Total Spectra")
        .draw()?;

    // Define how many points we want to actually render
    let max_render_points = 1000;

    for (idx, (sizes, label)) in datasets.iter().zip(parameters.iter()).enumerate() {
        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort_by(|a, b| b.cmp(a));

        let n = sorted_sizes.len();
        // Calculate the step size. If n < 1000, step is 1 (no sampling).
        let step = (n / max_render_points).max(1);

        let mut cumulative = 0;
        let mut curve_points: Vec<(usize, f64)> = Vec::with_capacity(max_render_points + 1);

        for (rank, &size) in sorted_sizes.iter().enumerate() {
            cumulative += size;

            // Only add the point to the plot if it's on our step interval
            // OR if it's the very last point (to ensure the curve reaches the end).
            if rank % step == 0 || rank == n - 1 {
                curve_points.push((rank, cumulative as f64 / total_spectra as f64));
            }
        }

        let mut hasher = std::hash::DefaultHasher::new();
        label.hash(&mut hasher);
        let hash = hasher.finish();
        let color = Palette99::pick(hash as usize).mix(0.8);

        chart
            .draw_series(LineSeries::new(curve_points, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.filled()));
    }

    // Reference line (N10)
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0, 0.1), (max_clusters, 0.1)],
        &RED.mix(0.5),
    )))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::LowerRight)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_purity_vs_n10(
    average_purities: &[f64],
    n10: &[usize],
    labels: &[String],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // 1. Prepare and Sort Data
    // We zip them together so we can sort by purity (X-axis)
    let mut data: Vec<(f64, usize, String)> = average_purities
        .iter()
        .zip(n10.iter())
        .zip(labels.iter())
        .map(|((&p, &n), l)| (p, n, l.clone()))
        .collect();

    // Sort by purity to ensure the line flows from lowest to highest
    // total_cmp: average_purities can be NaN (e.g. a 0/0 ratio from an empty
    // cluster) -- a diagnostic plot shouldn't panic the run over that.
    data.sort_by(|a, b| a.0.total_cmp(&b.0));

    let max_n10 = *n10.iter().max().unwrap_or(&100) as f64 * 1.1;
    let max_purity = 1.00;
    let min_purity = average_purities
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&0.0)
        - 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption("Purity vs. N10 Path Analysis", ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(min_purity..max_purity, 0..max_n10 as usize)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Average Purity")
        .y_desc("N10 (Cluster size at 10% coverage)")
        .draw()?;

    // 2. Draw the Black Path connecting all points
    let path_coords: Vec<(f64, usize)> = data.iter().map(|&(x, y, _)| (x, y)).collect();
    chart
        .draw_series(LineSeries::new(path_coords, &BLACK))?
        .label("Trend Path")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    // 3. Draw each point with a unique color and add to legend
    for (idx, (x, y, label)) in data.into_iter().enumerate() {
        let mut hasher = std::hash::DefaultHasher::new();
        label.hash(&mut hasher);
        let hash = hasher.finish();
        let color = Palette99::pick(hash as usize).mix(0.8);
        let point_color = color.filled();

        chart
            .draw_series(std::iter::once(
                EmptyElement::at((x, y)) + Circle::new((0, 0), 5, point_color),
            ))?
            .label(label)
            .legend(move |(x, y)| Circle::new((x + 10, y), 5, point_color));
    }

    // 4. Configure and Draw the Legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    Ok(())
}
