use crate::feature_extraction::config::FeatureFinderConfig;
use crate::feature_extraction::roi::MassTrace;

/// A resolved chromatographic peak along retention time.
#[derive(Debug, Clone)]
pub struct ChromatographicPeak {
    /// Weighted mean m/z across the chromatographic peak points.
    pub mean_mz: f64,
    /// Exact m/z at the chromatographic apex.
    pub apex_mz: f64,
    /// Retention time at apex (seconds).
    pub apex_rt: f32,
    /// Apex intensity.
    pub apex_intensity: f32,
    /// Start retention time (seconds).
    pub rt_start: f32,
    /// End retention time (seconds).
    pub rt_end: f32,
    /// Integrated peak area (trapezoidal integration).
    pub area: f32,
    /// Full Width at Half Maximum (seconds).
    pub fwhm: f32,
    /// Estimated Signal-to-Noise Ratio.
    pub snr: f32,
    /// Best-matching CWT wavelet scale (seconds).
    pub optimal_scale: f32,
    /// Retention times for points within this peak.
    pub peak_rts: Vec<f32>,
    /// Intensities for points within this peak.
    pub peak_intensities: Vec<f32>,
}

/// Continuous Wavelet Transform (CWT) peak resolver using Mexican Hat (Ricker) wavelets.
pub struct CwtPeakResolver<'a> {
    config: &'a FeatureFinderConfig,
}

impl<'a> CwtPeakResolver<'a> {
    pub fn new(config: &'a FeatureFinderConfig) -> Self {
        Self { config }
    }

    /// Resolves one or more chromatographic peaks from an Extracted Ion Chromatogram (MassTrace).
    pub fn resolve_peaks(&self, trace: &MassTrace) -> Vec<ChromatographicPeak> {
        let n = trace.len();
        if n < self.config.min_trace_length {
            return Vec::new();
        }

        // Calculate average scan interval dt (in seconds)
        let total_time = trace.rts.last().unwrap() - trace.rts.first().unwrap();
        if total_time <= 0.0 {
            return Vec::new();
        }
        let dt = total_time / (n - 1) as f32;
        if dt <= 1e-4 {
            return Vec::new();
        }

        // Precompute noise floor (median/mean of lower quartile)
        let noise_level = estimate_noise_level(&trace.intensities);

        // Pre-generate wavelet kernels for each scale
        let mut scale_matrices: Vec<(f32, Vec<f32>)> = Vec::new();
        for &scale in &self.config.wavelet_scales {
            let sigma_pts = (scale / dt).max(1.0);
            let kernel = generate_ricker_kernel(sigma_pts);
            let conv = convolve_symmetric(&trace.intensities, &kernel);
            scale_matrices.push((scale, conv));
        }

        // Find scale-space local maxima
        // Candidate: (scale_idx, time_idx, wavelet_coef)
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for (scale_idx, (_, conv)) in scale_matrices.iter().enumerate() {
            for i in 1..(n - 1) {
                let val = conv[i];
                if val > 0.0 && val >= conv[i - 1] && val >= conv[i + 1] {
                    // Check local raw intensity peak
                    if trace.intensities[i] >= self.config.min_peak_height * 0.5 {
                        candidates.push((scale_idx, i, val));
                    }
                }
            }
        }

        if candidates.is_empty() {
            // Fallback: simple local maximum search if CWT has no positive peaks
            return self.fallback_local_maxima(trace, noise_level);
        }

        // Cluster candidate apexes along time within 1.5 * dt
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2)); // highest wavelet coefficient first
        let mut taken = vec![false; n];
        let mut peaks = Vec::new();

        for &(scale_idx, apex_idx, _) in &candidates {
            if taken[apex_idx] {
                continue;
            }

            let opt_scale = scale_matrices[scale_idx].0;
            let conv = &scale_matrices[scale_idx].1;

            // Find boundaries using zero-crossing or valley search in wavelet domain
            let mut left_idx = apex_idx;
            while left_idx > 0 {
                if conv[left_idx - 1] <= 0.0 || conv[left_idx - 1] > conv[left_idx] {
                    break;
                }
                left_idx -= 1;
            }

            let mut right_idx = apex_idx;
            while right_idx < n - 1 {
                if conv[right_idx + 1] <= 0.0 || conv[right_idx + 1] > conv[right_idx] {
                    break;
                }
                right_idx += 1;
            }

            // Ensure minimum peak width of at least 2 points
            if right_idx <= left_idx {
                if left_idx > 0 {
                    left_idx -= 1;
                }
                if right_idx < n - 1 {
                    right_idx += 1;
                }
            }

            let apex_int = trace.intensities[apex_idx];
            if apex_int < self.config.min_peak_height {
                continue;
            }

            let snr = if noise_level > 1e-4 {
                apex_int / noise_level
            } else {
                100.0
            };

            if snr < self.config.snr_threshold {
                continue;
            }

            // Mark region as taken
            for idx in left_idx..=right_idx {
                taken[idx] = true;
            }

            let peak_rts = trace.rts[left_idx..=right_idx].to_vec();
            let peak_ints = trace.intensities[left_idx..=right_idx].to_vec();
            let peak_mzs = trace.mzs[left_idx..=right_idx].to_vec();

            // Calculate peak area via trapezoidal rule
            let area = integrate_trapezoid(&peak_rts, &peak_ints);

            // Calculate FWHM
            let fwhm = calculate_fwhm(&peak_rts, &peak_ints, apex_int);

            // Weighted m/z across peak
            let mut sum_mz_w = 0.0;
            let mut sum_w = 0.0;
            for (&m, &int) in peak_mzs.iter().zip(&peak_ints) {
                sum_mz_w += m * int as f64;
                sum_w += int as f64;
            }
            let mean_mz = if sum_w > 0.0 {
                sum_mz_w / sum_w
            } else {
                trace.mzs[apex_idx]
            };

            peaks.push(ChromatographicPeak {
                mean_mz,
                apex_mz: trace.mzs[apex_idx],
                apex_rt: trace.rts[apex_idx],
                apex_intensity: apex_int,
                rt_start: trace.rts[left_idx],
                rt_end: trace.rts[right_idx],
                area,
                fwhm,
                snr,
                optimal_scale: opt_scale,
                peak_rts,
                peak_intensities: peak_ints,
            });
        }

        // Re-sort peaks by retention time
        peaks.sort_by(|a, b| a.apex_rt.total_cmp(&b.apex_rt));
        peaks
    }

    /// Simple fallback local maximum peak picker if CWT does not find distinct wavelets.
    fn fallback_local_maxima(&self, trace: &MassTrace, noise_level: f32) -> Vec<ChromatographicPeak> {
        let n = trace.len();
        if n < 3 {
            return Vec::new();
        }

        let mut apex_idx = 0;
        let mut max_int = trace.intensities[0];
        for (i, &int) in trace.intensities.iter().enumerate() {
            if int > max_int {
                max_int = int;
                apex_idx = i;
            }
        }

        if max_int < self.config.min_peak_height {
            return Vec::new();
        }

        let snr = if noise_level > 1e-4 {
            max_int / noise_level
        } else {
            100.0
        };

        if snr < self.config.snr_threshold {
            return Vec::new();
        }

        let area = integrate_trapezoid(&trace.rts, &trace.intensities);
        let fwhm = calculate_fwhm(&trace.rts, &trace.intensities, max_int);

        vec![ChromatographicPeak {
            mean_mz: trace.mean_mz,
            apex_mz: trace.mzs[apex_idx],
            apex_rt: trace.rts[apex_idx],
            apex_intensity: max_int,
            rt_start: *trace.rts.first().unwrap(),
            rt_end: *trace.rts.last().unwrap(),
            area,
            fwhm,
            snr,
            optimal_scale: 0.0,
            peak_rts: trace.rts.clone(),
            peak_intensities: trace.intensities.clone(),
        }]
    }
}

/// Generates a normalized 1D discrete Mexican Hat (Ricker) wavelet kernel.
///
/// psi(t) = 2 / (sqrt(3) * pi^(1/4)) * (1 - t^2) * exp(-t^2 / 2)
fn generate_ricker_kernel(sigma: f32) -> Vec<f32> {
    let radius = ((3.5 * sigma).ceil() as isize).max(2);
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);

    let norm_factor = 2.0 / (3.0f32.sqrt() * std::f32::consts::PI.powf(0.25) * sigma.sqrt());

    for i in -radius..=radius {
        let t = i as f32 / sigma;
        let val = norm_factor * (1.0 - t * t) * (-0.5 * t * t).exp();
        kernel.push(val);
    }

    kernel
}

/// 1D discrete convolution with symmetric boundary padding.
fn convolve_symmetric(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let k_len = kernel.len();
    let radius = k_len / 2;
    let mut out = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for (k_idx, &k_val) in kernel.iter().enumerate() {
            let offset = k_idx as isize - radius as isize;
            let sig_idx = i as isize + offset;

            // Symmetric boundary mirroring
            let reflected_idx = if sig_idx < 0 {
                (-sig_idx) as usize
            } else if sig_idx >= n as isize {
                (2 * n as isize - 2 - sig_idx).max(0) as usize
            } else {
                sig_idx as usize
            };

            let safe_idx = reflected_idx.min(n - 1);
            sum += signal[safe_idx] * k_val;
        }
        out[i] = sum;
    }

    out
}

/// Estimates background noise level from lowest 25th percentile of intensities.
fn estimate_noise_level(intensities: &[f32]) -> f32 {
    if intensities.is_empty() {
        return 1.0;
    }
    let mut sorted = intensities.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q25_idx = (sorted.len() / 4).max(1);
    let sum: f32 = sorted[..q25_idx].iter().sum();
    (sum / q25_idx as f32).max(1.0)
}

/// Computes peak area via Trapezoidal numerical integration.
fn integrate_trapezoid(x: &[f32], y: &[f32]) -> f32 {
    if x.len() < 2 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..(x.len() - 1) {
        let dx = x[i + 1] - x[i];
        let avg_y = 0.5 * (y[i] + y[i + 1]);
        area += dx * avg_y;
    }
    area.max(0.0)
}

/// Computes Full Width at Half Maximum (FWHM) in seconds.
fn calculate_fwhm(rts: &[f32], intensities: &[f32], apex_int: f32) -> f32 {
    if rts.len() < 2 || apex_int <= 0.0 {
        return 0.0;
    }
    let half_max = 0.5 * apex_int;

    // Find first point exceeding half_max from left
    let mut left_rt = rts[0];
    for i in 0..(rts.len() - 1) {
        if intensities[i] <= half_max && intensities[i + 1] >= half_max {
            let frac = (half_max - intensities[i]) / (intensities[i + 1] - intensities[i]).max(1e-6);
            left_rt = rts[i] + frac * (rts[i + 1] - rts[i]);
            break;
        }
    }

    // Find last point exceeding half_max from right
    let mut right_rt = *rts.last().unwrap();
    for i in (1..rts.len()).rev() {
        if intensities[i] <= half_max && intensities[i - 1] >= half_max {
            let frac = (half_max - intensities[i]) / (intensities[i - 1] - intensities[i]).max(1e-6);
            right_rt = rts[i] + frac * (rts[i - 1] - rts[i]);
            break;
        }
    }

    (right_rt - left_rt).max(0.0)
}
