use crate::HashedSpectrumConfig;
use crate::clustering::cluster::ClusterQualityMetrics;
use crate::spectrum::{MSNSpectrum, MassSpectrum, Peak};
use plotters::prelude::*;
use plotters::*;
use std::f64::consts::PI;

use std::error::Error;
use std::path::Path;
use std::sync::Once;

static INIT_FONTS: Once = Once::new();

/// Ensures that standard font families ("sans-serif", "serif", "monospace") are registered
/// with plotters using an embedded pure-Rust font, completely avoiding any external
/// system dependencies like fontconfig, pkg-config, or C libraries.
pub fn ensure_fonts_registered() {
    INIT_FONTS.call_once(|| {
        const FONT_DATA: &[u8] = include_bytes!("assets/Tuffy.ttf");
        let _ = plotters::style::register_font(
            "sans-serif",
            plotters::style::FontStyle::Normal,
            FONT_DATA,
        );
        let _ = plotters::style::register_font(
            "serif",
            plotters::style::FontStyle::Normal,
            FONT_DATA,
        );
        let _ = plotters::style::register_font(
            "monospace",
            plotters::style::FontStyle::Normal,
            FONT_DATA,
        );
    });
}

// --- Plot Layout & Scaling Constants ---
const BASE_RESOLUTION: f32 = 800.0;
const BASE_WIDTH_FOR_STATS: f32 = 1024.0;

const BASE_TITLE_SIZE: f32 = 40.0;
const MIN_TITLE_SIZE: f32 = 16.0;

const BASE_AXIS_LABEL_SIZE: f32 = 40.0;
const MIN_AXIS_LABEL_SIZE: f32 = 20.0;

const BASE_MARGIN: f32 = 40.0;
const MIN_MARGIN: f32 = 20.0;

const BASE_X_LABEL_AREA: f32 = 80.0;
const MIN_X_LABEL_AREA: f32 = 40.0;

const BASE_Y_LABEL_AREA: f32 = 120.0;
const MIN_Y_LABEL_AREA: f32 = 60.0;

const MIN_LINE_WIDTH: f32 = 1.0;

// --- Statistics Panel Constants ---
const BASE_STATS_PANEL_WIDTH: f32 = 200.0;
const MIN_STATS_PANEL_WIDTH: f32 = 120.0;

const BASE_STATS_TITLE_SIZE: f32 = 20.0;
const MIN_STATS_TITLE_SIZE: f32 = 12.0;

const BASE_STATS_TEXT_SIZE: f32 = 16.0;
const MIN_STATS_TEXT_SIZE: f32 = 10.0;

const BASE_STATS_START_X: f32 = 10.0;
const MIN_STATS_START_X: f32 = 5.0;

const BASE_STATS_START_Y: f32 = 30.0;
const MIN_STATS_START_Y: f32 = 15.0;

const BASE_STATS_STEP_Y: f32 = 30.0;
const MIN_STATS_STEP_Y: f32 = 20.0;

const DEFAULT_HASHED_CONFIG: HashedSpectrumConfig = HashedSpectrumConfig {
    bin_width: 1.0,            // 1 Da bins for speed
    intensity_threshold: 0.01, // 1% threshold
    max_peaks: 0,              // Keep all peaks
    mass_tolerance: 0.1,       // 0.1 Da tolerance
    hash_size: 4096,
};

/// Helper function to cleanly map any generic plotters error to a Box<dyn Error>
/// This solves the `error[E0282]: type annotations needed` compiler issue.
fn map_err_to_box<E: std::fmt::Display>(err: E) -> Box<dyn Error> {
    Box::from(err.to_string())
}

/// Configurable plotter for mass spectra
pub struct SpectrumPlotter {
    width: u32,
    height: u32,
    title: Option<String>,
    show_statistics: bool,
    normalize: bool,
    match_tolerance: Option<f64>,
    line_width: u32,
}

impl Default for SpectrumPlotter {
    fn default() -> Self {
        ensure_fonts_registered();
        Self {
            width: 1024,
            height: 768,
            line_width: 4,
            normalize: true,
            title: None,
            show_statistics: false,
            match_tolerance: None,
        }
    }
}

impl SpectrumPlotter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output image dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set an optional title for the plot
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Toggle whether to show statistics on the right side of the plot
    pub fn show_statistics(mut self, show: bool) -> Self {
        self.show_statistics = show;
        self
    }

    /// Set the m/z tolerance for highlighting matching peaks in mirror plots
    pub fn match_tolerance(mut self, tolerance: f64) -> Self {
        self.match_tolerance = Some(tolerance);
        self
    }

    /// Plots a single mass spectrum to a file (Supports .png and .svg)
    pub fn plot_single(
        &self,
        spectrum: &MassSpectrum,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let peaks = spectrum.peaks();
        if peaks.is_empty() {
            return Ok(());
        }

        let max_mz = peaks.iter().map(|p| p.mz).fold(0f64, f64::max);
        let max_intensity = peaks.iter().map(|p| p.intensity).fold(0f32, f32::max);

        let ext = Path::new(output_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("png")
            .to_lowercase();

        if ext == "svg" {
            let root = SVGBackend::new(output_path, (self.width, self.height)).into_drawing_area();
            self.draw_single_internal(spectrum, max_mz, max_intensity, root)
        } else {
            let root =
                BitMapBackend::new(output_path, (self.width, self.height)).into_drawing_area();
            self.draw_single_internal(spectrum, max_mz, max_intensity, root)
        }
    }

    /// Plots two mass spectra mirrored against each other (Supports .png and .svg)
    pub fn plot_mirror(
        &self,
        spec_top: &MassSpectrum,
        spec_bottom: &MassSpectrum,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let peaks_top = spec_top.peaks();
        let peaks_bottom = spec_bottom.peaks();

        let max_mz_top = peaks_top.iter().map(|p| p.mz).fold(0f64, f64::max);
        let max_mz_bot = peaks_bottom.iter().map(|p| p.mz).fold(0f64, f64::max);
        let max_mz = max_mz_top.max(max_mz_bot);

        let max_int_top = peaks_top.iter().map(|p| p.intensity).fold(0f32, f32::max);
        let max_int_bot = peaks_bottom
            .iter()
            .map(|p| p.intensity)
            .fold(0f32, f32::max);

        let ext = Path::new(output_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("png")
            .to_lowercase();

        if ext == "svg" {
            let root = SVGBackend::new(output_path, (self.width, self.height)).into_drawing_area();
            self.draw_mirror_internal(
                spec_top,
                spec_bottom,
                max_mz,
                max_int_top,
                max_int_bot,
                root,
            )
        } else {
            let root =
                BitMapBackend::new(output_path, (self.width, self.height)).into_drawing_area();
            self.draw_mirror_internal(
                spec_top,
                spec_bottom,
                max_mz,
                max_int_top,
                max_int_bot,
                root,
            )
        }
    }

    fn draw_single_internal<B: DrawingBackend>(
        &self,
        spectrum: &MassSpectrum,
        max_mz: f64,
        max_intensity: f32,
        root: DrawingArea<B, plotters::coord::Shift>,
    ) -> Result<(), Box<dyn Error>>
    where
        B::ErrorType: std::fmt::Display,
    {
        let scale = (self.width.min(self.height) as f32) / BASE_RESOLUTION;

        let title_size = (BASE_TITLE_SIZE * scale).max(MIN_TITLE_SIZE) as u32;
        let axis_label_size = (BASE_AXIS_LABEL_SIZE * scale).max(MIN_AXIS_LABEL_SIZE) as u32;
        let margin = (BASE_MARGIN * scale).max(MIN_MARGIN) as u32;
        let x_label_area = (BASE_X_LABEL_AREA * scale).max(MIN_X_LABEL_AREA) as u32;
        let y_label_area = (BASE_Y_LABEL_AREA * scale).max(MIN_Y_LABEL_AREA) as u32;

        let dynamic_line_width = (self.line_width as f32 * scale).max(MIN_LINE_WIDTH) as u32;

        let stats_panel_width = (BASE_STATS_PANEL_WIDTH
            * (self.width as f32 / BASE_WIDTH_FOR_STATS))
            .max(MIN_STATS_PANEL_WIDTH) as u32;
        let stats_title_size = (BASE_STATS_TITLE_SIZE * scale).max(MIN_STATS_TITLE_SIZE) as u32;
        let stats_text_size = (BASE_STATS_TEXT_SIZE * scale).max(MIN_STATS_TEXT_SIZE) as u32;
        root.fill(&WHITE).map_err(map_err_to_box)?;
        root.fill(&WHITE).map_err(map_err_to_box)?;

        let (chart_area, stats_area) = if self.show_statistics {
            let (left, right) = root.split_horizontally(self.width - stats_panel_width);
            (left, Some(right))
        } else {
            (root, None)
        };

        let caption = self.title.as_deref().unwrap_or("Mass Spectrum");

        let mut chart = ChartBuilder::on(&chart_area)
            .caption(caption, ("sans-serif", title_size).into_font())
            .margin(margin)
            .x_label_area_size(x_label_area)
            .y_label_area_size(y_label_area)
            .build_cartesian_2d(0f64..(max_mz * 1.05), 0f32..(max_intensity * 1.05))
            .map_err(map_err_to_box)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("m/z")
            .y_desc("Intensity")
            .label_style(("sans-serif", axis_label_size).into_font())
            .draw()
            .map_err(map_err_to_box)?;

        for peak in &spectrum.peaks {
            chart
                .draw_series(LineSeries::new(
                    vec![(peak.mz, 0.0), (peak.mz, peak.intensity)],
                    BLACK.stroke_width(dynamic_line_width),
                ))
                .map_err(map_err_to_box)?;
        }

        if let Some(area) = stats_area {
            let title_style = ("sans-serif", stats_title_size, FontStyle::Bold)
                .into_font()
                .color(&BLACK);
            let text_style = ("sans-serif", stats_text_size).into_font().color(&BLACK);
            let start_x = (BASE_STATS_START_X * scale).max(MIN_STATS_START_X) as i32;
            let start_y = (BASE_STATS_START_Y * scale).max(MIN_STATS_START_Y) as i32;
            let step_y = (BASE_STATS_STEP_Y * scale).max(MIN_STATS_STEP_Y) as i32;

            // FIXED: Using `Text::new()` elements instead of `.draw_text()`
            area.draw(&Text::new("Statistics", (start_x, start_y), title_style))
                .map_err(map_err_to_box)?;
            area.draw(&Text::new(
                format!("Total Peaks: {}", spectrum.peaks.len()),
                (start_x, start_y + step_y),
                text_style.clone(),
            ))
            .map_err(map_err_to_box)?;
            area.draw(&Text::new(
                format!("Max m/z: {:.2}", max_mz),
                (start_x, start_y + step_y * 2),
                text_style.clone(),
            ))
            .map_err(map_err_to_box)?;
            area.draw(&Text::new(
                format!("Max Intensity: {:.0}", max_intensity),
                (start_x, start_y + step_y * 3),
                text_style,
            ))
            .map_err(map_err_to_box)?;
        }

        chart_area.present().map_err(map_err_to_box)?;
        Ok(())
    }

    fn draw_mirror_internal<B: DrawingBackend>(
        &self,
        spectrum_top: &MassSpectrum,
        spectrum_bottom: &MassSpectrum,
        max_mz: f64,
        max_int_top: f32,
        max_int_bot: f32,
        // FIXED: Updated `Shift` path
        root: DrawingArea<B, plotters::coord::Shift>,
    ) -> Result<(), Box<dyn Error>>
    where
        B::ErrorType: std::fmt::Display,
    {
        let scale = (self.width.min(self.height) as f32) / BASE_RESOLUTION;

        let title_size = (BASE_TITLE_SIZE * scale).max(MIN_TITLE_SIZE) as u32;
        let axis_label_size = (BASE_AXIS_LABEL_SIZE * scale).max(MIN_AXIS_LABEL_SIZE) as u32;
        let margin = (BASE_MARGIN * scale).max(MIN_MARGIN) as u32;
        let x_label_area = (BASE_X_LABEL_AREA * scale).max(MIN_X_LABEL_AREA) as u32;
        let y_label_area = (BASE_Y_LABEL_AREA * scale).max(MIN_Y_LABEL_AREA) as u32;

        let dynamic_line_width = (self.line_width as f32 * scale).max(MIN_LINE_WIDTH) as u32;

        let stats_panel_width = (BASE_STATS_PANEL_WIDTH
            * (self.width as f32 / BASE_WIDTH_FOR_STATS))
            .max(MIN_STATS_PANEL_WIDTH) as u32;
        let stats_title_size = (BASE_STATS_TITLE_SIZE * scale).max(MIN_STATS_TITLE_SIZE) as u32;
        let stats_text_size = (BASE_STATS_TEXT_SIZE * scale).max(MIN_STATS_TEXT_SIZE) as u32;
        root.fill(&WHITE).map_err(map_err_to_box)?;

        let (chart_area, stats_area) = if self.show_statistics {
            let (left, right) = root.split_horizontally(self.width - stats_panel_width);
            (left, Some(right))
        } else {
            (root, None)
        };

        let caption = self.title.as_deref().unwrap_or("Mirror Plot");

        let mut chart = ChartBuilder::on(&chart_area)
            .caption(caption, ("sans-serif", title_size).into_font())
            .margin(margin)
            .x_label_area_size(x_label_area)
            .y_label_area_size(y_label_area)
            .build_cartesian_2d(
                0f64..(max_mz * 1.05),
                (-max_int_bot * 1.05)..(max_int_top * 1.05),
            )
            .map_err(map_err_to_box)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("m/z")
            .y_desc("Intensity")
            .label_style(("sans-serif", axis_label_size).into_font())
            .y_label_formatter(&|y| format!("{:.1}", y.abs()))
            .draw()
            .map_err(map_err_to_box)?;

        let is_match = |mz: f64, other_peaks: &[Peak], tol: f64| -> bool {
            other_peaks.iter().any(|p| (p.mz - mz).abs() <= tol)
        };

        let mut matches_count = 0;

        for peak in spectrum_top.peaks() {
            let color = match self.match_tolerance {
                Some(tol) if is_match(peak.mz, &spectrum_bottom.peaks, tol) => {
                    matches_count += 1;
                    RED.stroke_width(dynamic_line_width)
                }
                _ => BLACK.stroke_width(dynamic_line_width),
            };
            chart
                .draw_series(LineSeries::new(
                    vec![(peak.mz, 0.0), (peak.mz, peak.intensity)],
                    color,
                ))
                .map_err(map_err_to_box)?;
        }

        for peak in spectrum_bottom.peaks() {
            let color = match self.match_tolerance {
                Some(tol) if is_match(peak.mz, &spectrum_top.peaks, tol) => {
                    RED.stroke_width(dynamic_line_width)
                }
                _ => BLACK.stroke_width(dynamic_line_width),
            };
            chart
                .draw_series(LineSeries::new(
                    vec![(peak.mz, 0.0), (peak.mz, -peak.intensity)],
                    color,
                ))
                .map_err(map_err_to_box)?;
        }

        chart
            .draw_series(LineSeries::new(
                vec![(0.0, 0.0), (max_mz * 1.05, 0.0)],
                BLACK.stroke_width(dynamic_line_width.max(1)),
            ))
            .map_err(map_err_to_box)?;

        if let Some(area) = stats_area {
            let title_style = ("sans-serif", stats_title_size, FontStyle::Bold)
                .into_font()
                .color(&BLACK);
            let text_style = ("sans-serif", stats_text_size).into_font().color(&BLACK);

            let start_x = (10.0 * scale).max(5.0) as i32;
            let start_y = (30.0 * scale).max(15.0) as i32;
            let step_y = (30.0 * scale).max(20.0) as i32;

            // FIXED: Using `Text::new()` elements
            area.draw(&Text::new("Statistics", (start_x, start_y), title_style))
                .map_err(map_err_to_box)?;
            area.draw(&Text::new(
                format!("Top Peaks: {}", spectrum_top.peaks().len()),
                (start_x, start_y + step_y),
                text_style.clone(),
            ))
            .map_err(map_err_to_box)?;
            area.draw(&Text::new(
                format!("Bottom Peaks: {}", spectrum_bottom.peaks().len()),
                (start_x, start_y + step_y * 2),
                text_style.clone(),
            ))
            .map_err(map_err_to_box)?;

            if self.match_tolerance.is_some() {
                let match_style = ("sans-serif", stats_text_size).into_font().color(&RED);
                area.draw(&Text::new(
                    format!("Matches: {}", matches_count),
                    (start_x, start_y + step_y * 3),
                    match_style,
                ))
                .map_err(map_err_to_box)?;
            }

            let similarity = spectrum_bottom
                .hashed_cosine_similarity(spectrum_top, &DEFAULT_HASHED_CONFIG)
                .unwrap_or(0.0);
            area.draw(&Text::new(
                format!("Cosine sim: {:.3}", similarity),
                (start_x, start_y + step_y * 4),
                text_style,
            ))
            .map_err(map_err_to_box)?;
        }

        chart_area.present().map_err(map_err_to_box)?;
        Ok(())
    }
}

pub struct SpectrumAnalyzer<'a> {
    data: &'a Vec<MSNSpectrum>,
}

impl<'a> SpectrumAnalyzer<'a> {
    pub fn new(data: &'a Vec<MSNSpectrum>) -> Self {
        ensure_fonts_registered();
        Self { data }
    }

    // Helper to define consistent styles
    const CAPTION_FONT: (&'static str, u32) = ("sans-serif", 35);
    const LABEL_FONT: (&'static str, u32) = ("sans-serif", 22);
    const TICK_FONT: (&'static str, u32) = ("sans-serif", 15);

    /// 1. M/Z vs Retention Time (Scatter Plot)
    pub fn plot_mz_rt(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let (min_rt, max_rt) = self
            .data
            .iter()
            .map(|s| s.retention_time_in_seconds)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), val| {
                (min.min(val), max.max(val))
            });

        let (min_mz, max_mz) = self
            .data
            .iter()
            .map(|s| s.precursor_mz)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
                (min.min(val), max.max(val))
            });

        let mut chart = ChartBuilder::on(&root)
            .caption("Precursor M/Z vs Retention Time", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60) // Increased for labels
            .y_label_area_size(80) // Increased for labels
            .build_cartesian_2d(min_rt..max_rt, min_mz..max_mz)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Retention Time (s)")
            .y_desc("Precursor M/Z (Da)")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        chart.draw_series(self.data.iter().map(|s| {
            Circle::new(
                (s.retention_time_in_seconds, s.precursor_mz),
                3,
                BLUE.mix(0.6).filled(),
            )
        }))?;

        root.present()?;
        Ok(())
    }

    // --- Helper Functions for Statistical Analysis ---

    fn calculate_standard_deviation(data: &[f64], mean: f64) -> f64 {
        let variance = data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (data.len() - 1) as f64;
        variance.sqrt()
    }

    fn calculate_silverman_bandwidth(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std_dev = Self::calculate_standard_deviation(data, mean);

        let mut sorted_data = data.to_vec();
        sorted_data.sort_unstable_by(|a, b| a.total_cmp(b));

        let q1 = sorted_data[data.len() / 4];
        let q3 = sorted_data[data.len() * 3 / 4];
        let iqr = q3 - q1;

        // Protection against division by zero or extreme uniformity
        let min_dispersion = f64::min(std_dev, iqr / 1.34);
        let a = if min_dispersion > 0.0 {
            min_dispersion
        } else {
            std_dev
        };

        0.9 * a * n.powf(-0.2)
    }

    // --- Kernel Functions ---

    fn gaussian_kernel(u: f64) -> f64 {
        (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
    }

    fn epanechnikov_kernel(u: f64) -> f64 {
        if u.abs() <= 1.0 {
            0.75 * (1.0 - u * u)
        } else {
            0.0
        }
    }

    // --- Your Plot Function ---

    pub fn plot_mz_kde(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mzs: Vec<f64> = self.data.iter().map(|s| s.precursor_mz).collect();
        if mzs.len() < 2 {
            return Ok(()); // Need at least 2 points for density
        }

        // mzs.len() >= 2 checked above; total_cmp since precursor m/z from a
        // real file can be NaN.
        let min_mz = *mzs
            .iter()
            .min_by(|a, b| a.total_cmp(b))
            .expect("mzs checked len >= 2 above");
        let max_mz = *mzs
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .expect("mzs checked len >= 2 above");

        // 1. Calculate dynamic bandwidth
        let bandwidth = Self::calculate_silverman_bandwidth(&mzs);
        let n_f64 = mzs.len() as f64;

        let steps = 400;
        let mut max_density = 0.0;

        // 2. Select your kernel (you can swap this to epanechnikov_kernel)
        let kernel = Self::epanechnikov_kernel;

        let kde_points: Vec<(f64, f64)> = (0..steps)
            .map(|i| {
                let x = min_mz + (max_mz - min_mz) * (i as f64 / steps as f64);

                // 3. Apply the general KDE formula
                let density = mzs
                    .iter()
                    .map(|&m| {
                        let u = (x - m) / bandwidth;
                        kernel(u)
                    })
                    .sum::<f64>()
                    / (n_f64 * bandwidth);

                if density > max_density {
                    max_density = density;
                }
                (x, density)
            })
            .collect();

        let mut chart = ChartBuilder::on(&root)
            .caption("Precursor M/Z Density Distribution", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(min_mz..max_mz, 0.0..(max_density * 1.1))?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("M/Z")
            .y_desc("Density")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        chart.draw_series(
            AreaSeries::new(kde_points, 0.0, CYAN.mix(0.3).filled())
                .border_style(CYAN.stroke_width(2)),
        )?;

        root.present()?;
        Ok(())
    }

    pub fn plot_peak_counts(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let counts: Vec<u64> = self
            .data
            .iter()
            .map(|s| s.spectrum.peaks.len() as u64)
            .collect();
        let max_val = *counts.iter().max().unwrap_or(&1);

        // Determine bin size (e.g., 20 bins)
        let bin_count = 20;
        let bin_size = (max_val / bin_count).max(1);

        // Pre-calculate frequencies to find the real Y-axis limit
        let mut bins = vec![0u32; (bin_count + 1) as usize];
        for &c in &counts {
            let idx = (c / bin_size) as usize;
            if idx < bins.len() {
                bins[idx] += 1;
            }
        }
        let max_freq = *bins.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Distribution of Peak Counts", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            // Adjust Y range to max_freq with some padding
            .build_cartesian_2d(0..max_val, 0..(max_freq + (max_freq / 10)))?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Number of Peaks")
            .y_desc("Frequency (Number of Spectra)")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(MAGENTA.mix(0.5).filled())
                .data(counts.into_iter().map(|c| (c, 1))),
        )?;

        Ok(())
    }
    /// 3. Feature Map (RT vs M/Z with Intensity Color)
    pub fn plot_feature_map(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Assuming your MSNSpectrum has an 'intensity' field
        // If not, we can use the sum of peak intensities as a proxy
        let data_points: Vec<(f32, f64, f32)> = self
            .data
            .iter()
            .map(|s| {
                let intensity = s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>();
                (s.retention_time_in_seconds, s.precursor_mz, intensity)
            })
            .collect();

        let max_intensity = data_points.iter().map(|p| p.2).fold(0.0, f32::max);
        let min_rt = data_points
            .iter()
            .map(|p| p.0)
            .fold(f32::INFINITY, f32::min);
        let max_rt = data_points
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_mz = data_points
            .iter()
            .map(|p| p.1)
            .fold(f64::INFINITY, f64::min);
        let max_mz = data_points
            .iter()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("MS Feature Map (Intensity Heatmap)", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(min_rt..max_rt, min_mz..max_mz)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Retention Time (s)")
            .y_desc("Precursor M/Z")
            .draw()?;

        chart.draw_series(data_points.iter().map(|(rt, mz, intensity)| {
            // Map intensity to a color (Higher = Redder/Darker)
            let ratio = (intensity / max_intensity) as f64;
            let color = HSLColor(0.6 * (1.0 - ratio), 0.8, 0.5); // Blue to Red scale

            Circle::new((*rt, *mz), 3, color.filled())
        }))?;

        Ok(())
    }
    /// 4. Total Ion Chromatogram (TIC)
    pub fn plot_tic(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut tic_data: Vec<(f32, f32)> = self
            .data
            .iter()
            .map(|s| {
                let total_intensity = s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>();
                (s.retention_time_in_seconds, total_intensity)
            })
            .collect();

        // Sort by RT to ensure the line draws correctly
        tic_data.sort_by(|a, b| a.0.total_cmp(&b.0));

        let max_rt = tic_data.last().map(|d| d.0).unwrap_or(1.0);
        let max_int = tic_data.iter().map(|d| d.1).fold(0.0, f32::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Total Ion Chromatogram (TIC)", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(90)
            .build_cartesian_2d(0.0..max_rt, 0.0..max_int)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Time (s)")
            .y_desc("Total Intensity")
            .draw()?;

        chart.draw_series(LineSeries::new(tic_data, &BLUE.mix(0.8)))?;

        Ok(())
    }
    pub fn plot_feature_map_log(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // 1. Collect data and apply Log10 to intensities
        // We add 1.0 to avoid log(0) which is undefined/negative infinity
        let data_points: Vec<(f32, f64, f32)> = self
            .data
            .iter()
            .map(|s| {
                let raw_intensity = s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>();
                (
                    s.retention_time_in_seconds,
                    s.precursor_mz,
                    (raw_intensity + 1.0).log10(),
                )
            })
            .collect();

        let max_log_int = data_points.iter().map(|p| p.2).fold(0.0, f32::max);
        let min_log_int = data_points
            .iter()
            .map(|p| p.2)
            .fold(f32::INFINITY, f32::min);

        let min_rt = data_points
            .iter()
            .map(|p| p.0)
            .fold(f32::INFINITY, f32::min);
        let max_rt = data_points
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_mz = data_points
            .iter()
            .map(|p| p.1)
            .fold(f64::INFINITY, f64::min);
        let max_mz = data_points
            .iter()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("MS Feature Map (Log10 Intensity)", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(min_rt..max_rt, min_mz..max_mz)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Retention Time (s)")
            .y_desc("Precursor M/Z")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        // 2. Draw points with a multi-stage color gradient
        chart.draw_series(data_points.iter().map(|(rt, mz, log_int)| {
            // Calculate normalized intensity (0.0 to 1.0)
            let norm = (log_int - min_log_int) / (max_log_int - min_log_int);

            // Custom Color Gradient: Blue (low) -> Green -> Yellow -> Red (high)
            let color = if norm < 0.25 {
                BLUE.mix(norm as f64 * 4.0)
            } else if norm < 0.5 {
                CYAN.mix((norm - 0.25) as f64 * 4.0)
            } else if norm < 0.75 {
                YELLOW.mix((norm - 0.5) as f64 * 4.0)
            } else {
                RED.mix((norm - 0.75) as f64 * 4.0)
            };

            Circle::new((*rt, *mz), 2, color.filled())
        }))?;

        Ok(())
    }
    pub fn plot_kendrick_mass_defect(
        &self,
        out_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let km_factor = 14.0 / 14.01565; // Standard CH2 scale
        let points: Vec<(f64, f64, f64)> = self
            .data
            .iter()
            .map(|s| {
                let km = s.precursor_mz * km_factor;
                let kmd = km.ceil() - km;
                let intensity = s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>() as f64;
                (s.precursor_mz, kmd, (intensity + 1.0).log10())
            })
            .collect();

        let max_mz = points.iter().map(|p| p.0).fold(0.0, f64::max);
        let max_log_int = points.iter().map(|p| p.2).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Kendrick Mass Defect (CH2 Scale)", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0.0..max_mz, 0.0..1.0)?; // KMD is always between 0 and 1

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Observed M/Z")
            .y_desc("Kendrick Mass Defect")
            .draw()?;

        chart.draw_series(points.iter().map(|(mz, kmd, log_int)| {
            let norm = log_int / max_log_int;
            Circle::new((*mz, *kmd), 2, RED.mix(norm).filled())
        }))?;

        Ok(())
    }
    pub fn plot_3d_landscape(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // 1. Dynamically find bounds
        let mut min_rt = f32::INFINITY;
        let mut max_rt = f32::NEG_INFINITY;
        let mut min_mz = f64::INFINITY;
        let mut max_mz = f64::NEG_INFINITY;
        let mut min_log_int = f32::INFINITY;
        let mut max_log_int = f32::NEG_INFINITY;

        let points: Vec<(f32, f32, f64)> = self
            .data
            .iter()
            .map(|s| {
                let log_int =
                    (s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>() + 1.0).log10();
                let rt = s.retention_time_in_seconds;
                let mz = s.precursor_mz;

                if rt < min_rt {
                    min_rt = rt;
                }
                if rt > max_rt {
                    max_rt = rt;
                }
                if mz < min_mz {
                    min_mz = mz;
                }
                if mz > max_mz {
                    max_mz = mz;
                }
                if log_int < min_log_int {
                    min_log_int = log_int;
                }
                if log_int > max_log_int {
                    max_log_int = log_int;
                }

                (rt, log_int, mz)
            })
            .collect();

        // 2. Build 3D Chart
        let mut chart = ChartBuilder::on(&root)
            .caption("3D Spectral Landscape", Self::CAPTION_FONT)
            .margin(20)
            .build_cartesian_3d(
                min_rt..max_rt,
                min_log_int..(max_log_int * 1.1),
                min_mz..max_mz,
            )?;

        chart.with_projection(|mut pb| {
            pb.yaw = 0.5;
            pb.pitch = 0.3;
            pb.scale = 0.7;
            pb.into_matrix()
        });

        chart
            .configure_axes()
            // .x_desc("RT (s)")
            // .y_desc("Log10 Intensity")
            // .z_desc("M/Z")
            .draw()?;

        chart.draw_series(
            points
                .into_iter()
                .map(|(rt, log_int, mz)| Circle::new((rt, log_int, mz), 2, BLUE.mix(0.5).filled())),
        )?;

        root.present()?;
        Ok(())
    }
    pub fn plot_intensity_distribution(
        &self,
        out_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut all_log_intensities: Vec<f32> = self
            .data
            .iter()
            .flat_map(|s| s.spectrum.peaks.iter().map(|p| (p.intensity + 1.0).log10()))
            .collect();

        if all_log_intensities.is_empty() {
            return Ok(());
        }
        all_log_intensities.sort_by(|a, b| a.total_cmp(b));

        // Create the Quartiles object required by the new API
        let quartiles = Quartiles::new(&all_log_intensities);
        let max_log = *all_log_intensities.last().unwrap_or(&1.0);

        let mut chart = ChartBuilder::on(&root)
            .caption("Intensity Dynamic Range", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(0..2, 0.0..(max_log * 1.1))?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_desc("Log10 Intensity")
            .draw()?;

        // Correct usage of new_vertical
        chart.draw_series(std::iter::once(
            Boxplot::new_vertical(1, &quartiles).style(BLUE.mix(0.6).stroke_width(2)),
        ))?;

        root.present()?;
        Ok(())
    }
    pub fn plot_charge_distribution(
        &self,
        out_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // 1. Count occurrences of each charge
        let mut charge_counts = nohash_hasher::IntMap::default();
        for s in self.data {
            // Assuming your MSNSpectrum struct has a precursor_charge field
            *charge_counts.entry(s.charge()).or_insert(0u32) += 1;
        }

        let max_count = *charge_counts.values().max().unwrap_or(&1) as i32;
        let max_charge = *charge_counts.keys().max().unwrap_or(&1) as i32;

        let mut chart = ChartBuilder::on(&root)
            .caption("Precursor Charge State Distribution", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0.0..max_charge as f32 + 1.0,
                0.0..(max_count + (max_count / 10)) as f32,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Charge State (z)")
            .y_desc("Number of Spectra")
            .axis_desc_style(Self::LABEL_FONT)
            .draw()?;

        chart.draw_series(charge_counts.iter().map(|(&charge, &count)| {
            Rectangle::new(
                [
                    (charge as f32 - 0.5, 0.0),
                    (charge as f32 + 0.5, count as f32),
                ],
                BLUE.mix(0.6).filled(),
            )
        }))?;

        root.present()?;
        Ok(())
    }
    pub fn plot_mz_rt_by_charge(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let min_rt = self
            .data
            .iter()
            .map(|s| s.retention_time_in_seconds)
            .fold(f32::INFINITY, f32::min);
        let max_rt = self
            .data
            .iter()
            .map(|s| s.retention_time_in_seconds)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_mz = self
            .data
            .iter()
            .map(|s| s.precursor_mz)
            .fold(f64::INFINITY, f64::min);
        let max_mz = self
            .data
            .iter()
            .map(|s| s.precursor_mz)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("M/Z vs RT (Colored by Charge)", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(min_rt..max_rt, min_mz..max_mz)?;

        chart.configure_mesh().disable_mesh().draw()?;

        chart.draw_series(self.data.iter().map(|s| {
            // Map common charges to distinct colors
            let color = match s.charge() {
                1 => BLUE.mix(0.7),
                2 => GREEN.mix(0.7),
                3 => RED.mix(0.7),
                4 => MAGENTA.mix(0.7),
                _ => CYAN.mix(0.7),
            };

            Circle::new(
                (s.retention_time_in_seconds, s.precursor_mz),
                3,
                color.filled(),
            )
        }))?;

        root.present()?;
        Ok(())
    }
    pub fn plot_neutral_mass_rt(&self, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(out_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let proton_mass = 1.007276;

        // 1. Calculate neutral masses and find dynamic bounds
        let mut min_rt = f32::INFINITY;
        let mut max_rt = f32::NEG_INFINITY;
        let mut min_mass = f64::INFINITY;
        let mut max_mass = f64::NEG_INFINITY;
        let mut max_log_int = 0.0f32;

        let data_points: Vec<(f32, f64, f32)> = self
            .data
            .iter()
            .filter(|s| s.charge() > 0) // Ignore unknown charges
            .map(|s| {
                let z = s.charge() as f64;
                let neutral_mass = (s.precursor_mz * z) - (z * proton_mass);
                let log_int =
                    (s.spectrum.peaks.iter().map(|p| p.intensity).sum::<f32>() + 1.0).log10();

                // Track bounds
                if s.retention_time_in_seconds < min_rt {
                    min_rt = s.retention_time_in_seconds;
                }
                if s.retention_time_in_seconds > max_rt {
                    max_rt = s.retention_time_in_seconds;
                }
                if neutral_mass < min_mass {
                    min_mass = neutral_mass;
                }
                if neutral_mass > max_mass {
                    max_mass = neutral_mass;
                }
                if log_int > max_log_int {
                    max_log_int = log_int;
                }

                (s.retention_time_in_seconds, neutral_mass, log_int)
            })
            .collect();

        // 2. Build the Chart
        let mut chart = ChartBuilder::on(&root)
            .caption("Neutral Mass vs Retention Time", Self::CAPTION_FONT)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(90)
            .build_cartesian_2d(min_rt..max_rt, min_mass..(max_mass * 1.05))?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Retention Time (s)")
            .y_desc("Neutral Mass (Da)")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        // 3. Draw with intensity-based color scaling
        chart.draw_series(data_points.iter().map(|(rt, mass, log_int)| {
            let norm = (log_int / max_log_int) as f64;

            let color = if norm < 0.25 {
                BLUE.mix(norm as f64 * 4.0)
            } else if norm < 0.5 {
                CYAN.mix((norm - 0.25) as f64 * 4.0)
            } else if norm < 0.75 {
                YELLOW.mix((norm - 0.5) as f64 * 4.0)
            } else {
                RED.mix((norm - 0.75) as f64 * 4.0)
            };

            Circle::new((*rt, *mass), 2, color.filled())
        }))?;

        root.present()?;
        Ok(())
    }
}

/// Plots distributions of per-cluster internal-validation quality metrics
/// (see [`ClusterQualityMetrics`]).
pub struct ClusterQualityAnalyzer<'a> {
    data: &'a [ClusterQualityMetrics],
}

impl<'a> ClusterQualityAnalyzer<'a> {
    pub fn new(data: &'a [ClusterQualityMetrics]) -> Self {
        Self { data }
    }

    const CAPTION_FONT: (&'static str, u32) = ("sans-serif", 26);
    const LABEL_FONT: (&'static str, u32) = ("sans-serif", 18);
    const TICK_FONT: (&'static str, u32) = ("sans-serif", 14);

    /// Buckets `values` (assumed within `[min, max]`) into `bins` equal-width
    /// buckets, returning `(bucket_start, bucket_end, count)` triples.
    fn float_histogram(values: &[f64], bins: usize, min: f64, max: f64) -> Vec<(f64, f64, u32)> {
        let width = (max - min) / bins as f64;
        let mut counts = vec![0u32; bins];
        for &v in values {
            let idx = (((v - min) / width) as usize).min(bins - 1);
            counts[idx] += 1;
        }
        counts
            .into_iter()
            .enumerate()
            .map(|(i, c)| (min + i as f64 * width, min + (i + 1) as f64 * width, c))
            .collect()
    }

    fn draw_float_histogram(
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        caption: &str,
        x_desc: &str,
        values: &[f64],
        color: RGBColor,
    ) -> Result<(), Box<dyn Error>> {
        let bins = Self::float_histogram(values, 25, 0.0, 1.0);
        let max_freq = bins.iter().map(|b| b.2).max().unwrap_or(1);

        let mut chart = ChartBuilder::on(area)
            .caption(caption, Self::CAPTION_FONT)
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0.0..1.0, 0u32..(max_freq + max_freq / 10 + 1))?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc(x_desc)
            .y_desc("# clusters")
            .axis_desc_style(Self::LABEL_FONT)
            .label_style(Self::TICK_FONT)
            .draw()?;

        chart.draw_series(
            bins.iter()
                .map(|&(x0, x1, c)| Rectangle::new([(x0, 0), (x1, c)], color.mix(0.6).filled())),
        )?;
        Ok(())
    }

    /// Plots a 2x2 overview: avg-cohesion distribution, min-cohesion (worst-case)
    /// distribution, cluster size distribution, and a scatter of size vs avg cohesion.
    pub fn plot_quality_overview(&self, out_path: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(out_path, (1400, 1000)).into_drawing_area();
        root.fill(&WHITE)?;

        if self.data.is_empty() {
            return Ok(());
        }

        let panels = root.split_evenly((2, 2));

        let avg_vals: Vec<f64> = self.data.iter().map(|m| m.avg_cohesion).collect();
        let min_vals: Vec<f64> = self.data.iter().map(|m| m.min_cohesion).collect();
        let sizes: Vec<u64> = self.data.iter().map(|m| m.size as u64).collect();

        Self::draw_float_histogram(
            &panels[0],
            "Avg. Cohesion (mean similarity to consensus)",
            "Avg cohesion",
            &avg_vals,
            CYAN,
        )?;

        Self::draw_float_histogram(
            &panels[1],
            "Min. Cohesion (worst-case similarity to consensus)",
            "Min cohesion",
            &min_vals,
            MAGENTA,
        )?;

        // Panel 3: cluster size histogram
        {
            let max_val = *sizes.iter().max().unwrap_or(&1);
            let bin_count = 20u64;
            let bin_size = (max_val / bin_count).max(1);

            let mut bins = vec![0u32; (bin_count + 1) as usize];
            for &s in &sizes {
                let idx = (s / bin_size) as usize;
                if idx < bins.len() {
                    bins[idx] += 1;
                }
            }
            let max_freq = *bins.iter().max().unwrap_or(&1);

            let mut chart = ChartBuilder::on(&panels[2])
                .caption("Cluster Size Distribution", Self::CAPTION_FONT)
                .margin(15)
                .x_label_area_size(50)
                .y_label_area_size(60)
                .build_cartesian_2d(0..max_val, 0..(max_freq + (max_freq / 10) + 1))?;

            chart
                .configure_mesh()
                .disable_mesh()
                .x_desc("Cluster size (# spectra)")
                .y_desc("# clusters")
                .axis_desc_style(Self::LABEL_FONT)
                .label_style(Self::TICK_FONT)
                .draw()?;

            chart.draw_series(
                Histogram::vertical(&chart)
                    .style(GREEN.mix(0.6).filled())
                    .data(sizes.iter().map(|&s| (s, 1))),
            )?;
        }

        // Panel 4: scatter of cluster size vs avg cohesion
        {
            let max_size = *sizes.iter().max().unwrap_or(&1);

            let mut chart = ChartBuilder::on(&panels[3])
                .caption("Cluster Size vs Avg Cohesion", Self::CAPTION_FONT)
                .margin(15)
                .x_label_area_size(50)
                .y_label_area_size(60)
                .build_cartesian_2d(0u64..(max_size + 1), 0.0..1.0)?;

            chart
                .configure_mesh()
                .disable_mesh()
                .x_desc("Cluster size (# spectra)")
                .y_desc("Avg cohesion")
                .axis_desc_style(Self::LABEL_FONT)
                .label_style(Self::TICK_FONT)
                .draw()?;

            chart.draw_series(
                self.data
                    .iter()
                    .map(|m| Circle::new((m.size as u64, m.avg_cohesion), 3, BLUE.mix(0.4).filled())),
            )?;
        }

        root.present()?;
        Ok(())
    }
}
