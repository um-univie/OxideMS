use plotters::prelude::*;
use std::f64::consts::PI;

/// Generates a Ricker (Mexican Hat) wavelet.
/// `points` is the number of points in the wavelet :qwindowwindow.
/// `width` controls the scaling of the wavelet.
fn ricker_wavelet(points: usize, width: f64) -> Vec<f64> {
    let mut wavelet = Vec::with_capacity(points);
    let a = width;
    let center = (points as f64 - 1.0) / 2.0;

    for i in 0..points {
        let t = (i as f64 - center) / a;
        // Formula: (1 - t^2) * exp(-t^2 / 2)
        // Scaled by 2 / (sqrt(3 * a) * pi^0.25) for energy normalization
        let norm = 2.0 / ((3.0 * a).sqrt() * PI.powf(0.25));
        let value = norm * (1.0 - t * t) * (-0.5 * t * t).exp();
        wavelet.push(value);
    }
    wavelet
}

/// Convolves a signal with a wavelet.
fn convolve_1d(signal: &[f64], wavelet: &[f64]) -> Vec<f64> {
    let sig_len = signal.len();
    let wav_len = wavelet.len();
    let pad = wav_len / 2;
    
    let mut output = vec![0.0; sig_len];

    for i in 0..sig_len {
        let mut sum = 0.0;
        for j in 0..wav_len {
            // Handle edge conditions by clamping to the nearest valid index
            let sig_idx = (i as isize + j as isize - pad as isize)
                .clamp(0, (sig_len - 1) as isize) as usize;
            
            sum += signal[sig_idx] * wavelet[j];
        }
        output[i] = sum;
    }
    output
}

/// Detects peaks using Continuous Wavelet Transform at a specific scale.
pub fn detect_peaks_cwt(
    signal_curve: &[(f64, f64)], 
    scale: f64
) -> Vec<f64> {
    // 1. Extract just the Y-values (density)
    let y_vals: Vec<f64> = signal_curve.iter().map(|&(_, y)| y).collect();
    
    // 2. Generate the Ricker wavelet. 
    // The window size should be roughly 10x the scale to capture the "tails" of the hat.
    let window_size = (scale * 10.0) as usize | 1; // Ensure it's an odd number
    let wavelet = ricker_wavelet(window_size, scale);
    
    // 3. Convolve the signal with the wavelet
    let cwt_output = convolve_1d(&y_vals, &wavelet);
    
    // 4. Find local maxima in the convolved output
    let mut detected_peaks = Vec::new();
    
    for i in 1..cwt_output.len() - 1 {
        let prev = cwt_output[i - 1];
        let curr = cwt_output[i];
        let next = cwt_output[i + 1];
        
        // A peak must be a local maximum AND have a positive correlation score
        // (A negative correlation means it matched an inverted shape/valley)
        if curr > prev && curr > next && curr > 0.0 {
            // Add the original X-coordinate of the peak
            detected_peaks.push(signal_curve[i].0);
        }
    }
    
    detected_peaks
}

pub type KernelFunc = fn(f64) -> f64;

pub fn gaussian_kernel(u: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * u * u).exp()
}

pub fn epanechnikov_kernel(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}
#[derive(Debug)]
pub struct CriticalPoints {
    /// The x-coordinates of the local maxima (the center of your clusters)
    pub peaks: Vec<f64>,
    /// The x-coordinates of the local minima (the boundaries separating clusters)
    pub boundaries: Vec<f64>, 
}

pub struct Kde<'a> {
    pub data: &'a [f64],
    pub bandwidth: f64,
    pub kernel: KernelFunc,
}

impl<'a> Kde<'a> {
    pub fn new(data: &'a [f64], bandwidth: f64, kernel: KernelFunc) -> Self {
        Self { data, bandwidth, kernel }
    }

    pub fn estimate(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let mut sum = 0.0;
        for &xi in self.data {
            let u = (x - xi) / self.bandwidth;
            sum += (self.kernel)(u);
        }
        sum / (n * self.bandwidth)
    }

    pub fn estimate_finite_support(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let lower_bound = x - self.bandwidth;
        let upper_bound = x + self.bandwidth;
        
        let start_idx = self.data.partition_point(|&val| val < lower_bound);
        let end_idx = start_idx + self.data[start_idx..].partition_point(|&val| val <= upper_bound);
        
        let mut sum = 0.0;
        for &xi in &self.data[start_idx..end_idx] {
            let u = (x - xi) / self.bandwidth;
            sum += (self.kernel)(u);
        }
        sum / (n * self.bandwidth)
    }

/// Helper: Calculate the value at a specific percentile for a **sorted** slice.
/// Uses linear interpolation between data points.
fn percentile_sorted(&self, p: f64) -> f64 {
    let n = self.data.len() - 1;
    let index = p * (n as f64);
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let weight = index - (lower as f64);

    if lower == upper {
        self.data[lower]
    } else {
        self.data[lower] * (1.0 - weight) + self.data[upper] * weight
    }
}

/// Calculates the optimal bandwidth using Silverman's Rule of Thumb.
/// Data must be sorted!
pub fn estimate_bandwidth_silverman(&mut self) {
    let n = self.data.len() as f64;
    
    // Fallback for extremely small self.datasets
    if n < 2.0 { self.bandwidth =  1.0; }

    // 1. Calculate Mean
    let mean = self.data.iter().sum::<f64>() / n;

    // 2. Calculate Sample Standard Deviation
    let variance = self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    // 3. Calculate Interquartile Range (IQR)
    let q1 = self.percentile_sorted(0.25);
    let q3 = self.percentile_sorted(0.75);
    let iqr = q3 - q1;

    // 4. Determine the minimum spread
    let min_spread = std_dev.min(iqr / 1.34);
    
    // Edge case: If all self.data points are nearly identical, spread is 0. 
    // Fallback to std_dev, and if that's also 0, use a hardcoded small minimum.
    let spread = if min_spread > 0.0 { min_spread } else { std_dev };
    let final_spread = if spread > 0.0 { spread } else { 1e-6 };

    // 5. Apply Silverman's formula: 0.9 * min(std, IQR/1.34) * n^(-1/5)
    self.bandwidth = 0.9 * final_spread * n.powf(-0.2);
}
/// Calculates the optimal bandwidth using Scott's Rule.
/// Faster, but slightly more prone to over-smoothing if outliers exist.
pub fn estimate_bandwidth_scott(&mut self) {
    let n = self.data.len() as f64;
    
    if n < 2.0 { self.bandwidth = 1.0; }

    let mean = self.data.iter().sum::<f64>() / n;
    let variance = self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let final_std_dev = if std_dev > 0.0 { std_dev } else { 1e-6 };

// Scott's formula: 1.06 * std * n^(-1/5)

    self.bandwidth =     1.06 * final_std_dev * n.powf(-0.2);
}
/// Scans the KDE curve over a specified range to find local minima and maxima.
pub fn detect_ridges(
    &self,
    steps: usize
) -> CriticalPoints {
    let mut peaks = Vec::new();
    let mut boundaries = Vec::new();
    let x_min = self.data[0];
    let x_max = *self.data.iter().last().unwrap();
    
    let step_size = (x_max - x_min) / (steps as f64);
    
    // Pre-calculate the curve over the grid
    let mut curve = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let x = x_min + (i as f64) * step_size;
        let y = self.estimate_finite_support(x); // Use estimate() if using Gaussian
        curve.push((x, y));
    }
    
    // Iterate through the interior points to find local extrema
    for i in 1..curve.len() - 1 {
        let y_prev = curve[i - 1].1;
        let y_curr = curve[i].1;
        let y_next = curve[i + 1].1;
        
        let x_curr = curve[i].0;
        
        // A peak occurs when the current point is strictly higher than its neighbors
        if y_curr > y_prev && y_curr > y_next {
            peaks.push(x_curr);
        }
        // A valley (boundary) occurs when the current point is strictly lower than its neighbors
        else if y_curr < y_prev && y_curr < y_next {
            boundaries.push(x_curr);
        }
    }
    
    CriticalPoints { peaks, boundaries }
}
}



#[derive(Debug)]
pub struct DeconvolvedPeak {
    pub x: f64,        // The center of the detected cluster
    pub height: f64,   // The density height
}

/// Performs greedy sequential deconvolution on a 2D curve.
/// - `curve`: The pre-calculated (x, y) points of your KDE.
/// - `bandwidth`: The same bandwidth used to generate the KDE.
/// - `max_peaks`: A safety limit to prevent infinite loops (e.g., 5 or 10).
/// - `noise_threshold`: Ignore peaks with a density lower than this (e.g., 0.05).
pub fn deconvolve_gaussian(
    curve: &[(f64, f64)],
    bandwidth: f64,
    max_peaks: usize,
    noise_threshold: f64,
) -> Vec<DeconvolvedPeak> {
    
    // Create a mutable working copy of the Y values (the residual signal)
    let xs: Vec<f64> = curve.iter().map(|&(x, _)| x).collect();
    let mut residual: Vec<f64> = curve.iter().map(|&(_, y)| y).collect();
    
    let mut found_peaks = Vec::new();

    for _ in 0..max_peaks {
        // 1. Find the index of the absolute maximum in the current residual curve
        let (max_idx, &max_y) = residual
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // 2. If the highest remaining peak is just noise, we are done
        if max_y < noise_threshold {
            break;
        }

        let peak_x = xs[max_idx];
        found_peaks.push(DeconvolvedPeak { x: peak_x, height: max_y });

        // 3. Subtract a pure Gaussian kernel centered at `peak_x` from the residual
        for i in 0..residual.len() {
            let current_x = xs[i];
            let u = (current_x - peak_x) / bandwidth;
            
            // Generate the un-normalized Gaussian shape. 
            // It equals exactly 1.0 at u=0, so multiplying by max_y perfectly matches the peak height.
            let kernel_shape = (-0.5 * u * u).exp(); 
            let subtraction_value = max_y * kernel_shape;
            
            residual[i] -= subtraction_value;
            
            // Prevent numerical underflow from pushing density below zero
            if residual[i] < 0.0 {
                residual[i] = 0.0;
            }
        }
    }

    // Sort the peaks from left to right on the X-axis before returning
    found_peaks.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    
    found_peaks
}

