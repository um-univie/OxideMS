use crate::feature_extraction::config::{FeatureFinderConfig, IsotopeModel};
use crate::feature_extraction::cwt::ChromatographicPeak;

/// IUPAC carbon-13 minus carbon-12 neutron mass difference.
pub const NEUTRON_C13_DIFF: f64 = 1.003355;

/// An individual isotope peak within a resolved feature.
#[derive(Debug, Clone)]
pub struct IsotopePeak {
    /// 0 for M0 (monoisotopic), 1 for M+1, 2 for M+2, etc.
    pub isotope_index: usize,
    pub mz: f64,
    pub apex_intensity: f32,
    pub area: f32,
}

/// A fully deconvoluted and deisotoped chromatographic feature.
#[derive(Debug, Clone)]
pub struct DeisotopedFeature {
    /// Monoisotopic m/z (intensity-weighted from M0).
    pub monoisotopic_mz: f64,
    /// Exact apex m/z of M0.
    pub apex_mz: f64,
    /// Charge state z (e.g. 1, 2).
    pub charge: i8,
    /// Retention time at apex (seconds).
    pub apex_rt: f32,
    /// Peak start RT (seconds).
    pub rt_start: f32,
    /// Peak end RT (seconds).
    pub rt_end: f32,
    /// Integrated area of the monoisotopic peak (M0).
    pub monoisotopic_area: f32,
    /// Total integrated area of all assembled isotopic peaks (M0 + M1 + ...).
    pub total_area: f32,
    /// Full Width at Half Maximum (seconds).
    pub fwhm: f32,
    /// Estimated Signal-to-Noise Ratio.
    pub snr: f32,
    /// Pearson correlation between M0 and isotope satellites.
    pub isotope_correlation: f32,
    /// Assembled isotopic envelope peaks.
    pub isotopes: Vec<IsotopePeak>,
}

/// Deisotoper and charge state deconvolution engine.
pub struct FeatureDeisotoper<'a> {
    config: &'a FeatureFinderConfig,
}

impl<'a> FeatureDeisotoper<'a> {
    pub fn new(config: &'a FeatureFinderConfig) -> Self {
        Self { config }
    }

    /// Deisotopes and deconvolutes a slice of resolved chromatographic peaks into features.
    pub fn deisotope(&self, peaks: &[ChromatographicPeak]) -> Vec<DeisotopedFeature> {
        if peaks.is_empty() {
            return Vec::new();
        }

        // Sort peaks by retention time apex
        let mut sorted_peaks = peaks.to_vec();
        sorted_peaks.sort_by(|a, b| a.apex_rt.total_cmp(&b.apex_rt));

        let n = sorted_peaks.len();
        let mut consumed = vec![false; n];
        let mut features = Vec::new();

        // Spatial index: bucket peaks by RT window to avoid O(N^2) scans
        let max_rt_diff = self.config.max_isotope_rt_diff_seconds;

        for i in 0..n {
            if consumed[i] {
                continue;
            }

            let m0 = &sorted_peaks[i];

            // Find all co-eluting candidates within [apex_rt - max_rt_diff, apex_rt + max_rt_diff]
            let mut candidate_indices = Vec::new();
            let mut j = i + 1;
            while j < n && (sorted_peaks[j].apex_rt - m0.apex_rt) <= max_rt_diff {
                if !consumed[j] {
                    candidate_indices.push(j);
                }
                j += 1;
            }

            // Test candidate charge states
            let mut best_feature: Option<(DeisotopedFeature, Vec<usize>)> = None;
            let mut best_score = -1.0;

            for z in self.config.charge_range() {
                if z <= 0 {
                    continue;
                }

                let delta_mz = NEUTRON_C13_DIFF / z as f64;
                let mut matched_satellites = Vec::new();
                let mut matched_indices = Vec::new();
                let mut current_expected_mz = m0.mean_mz + delta_mz;
                let mut current_iso_idx = 1;

                while current_iso_idx <= 4 {
                    // Search for matching satellite peak
                    let mut best_sat: Option<(usize, f32)> = None;
                    let mut min_mz_err = f64::INFINITY;

                    for &cand_idx in &candidate_indices {
                        if matched_indices.contains(&cand_idx) {
                            continue;
                        }
                        let cand = &sorted_peaks[cand_idx];
                        if self.config.isotope_tolerance.is_within(current_expected_mz, cand.mean_mz) {
                            let mz_err = (cand.mean_mz - current_expected_mz).abs();
                            // Check elution shape correlation
                            let corr = pearson_correlation(&m0.peak_rts, &m0.peak_intensities, &cand.peak_rts, &cand.peak_intensities);
                            if corr >= self.config.min_peak_correlation && mz_err < min_mz_err {
                                min_mz_err = mz_err;
                                best_sat = Some((cand_idx, corr));
                            }
                        }
                    }

                    if let Some((sat_idx, corr)) = best_sat {
                        matched_satellites.push((sat_idx, current_iso_idx, corr));
                        matched_indices.push(sat_idx);
                        current_expected_mz += delta_mz;
                        current_iso_idx += 1;
                    } else {
                        // Break isotope chain on first missing isotope
                        break;
                    }
                }

                // If at least M+1 was found, validate envelope score
                if !matched_satellites.is_empty() {
                    let m1_peak = &sorted_peaks[matched_satellites[0].0];
                    let obs_ratio = m1_peak.apex_intensity / m0.apex_intensity.max(1.0);

                    let is_valid = match self.config.isotope_model {
                        IsotopeModel::SmallMolecule => {
                            // Expected carbon count ~ monoisotopic_mass / 22.0
                            let mass = m0.mean_mz * z as f64;
                            let expected_c = (mass / 22.0).clamp(1.0, 150.0);
                            let expected_ratio = expected_c as f32 * 0.0108;
                            // Generous bounds for metabolomics heteroatoms
                            obs_ratio >= expected_ratio * 0.1 && obs_ratio <= expected_ratio * 2.5 + 0.2
                        }
                        IsotopeModel::Averagine => {
                            let mass = m0.mean_mz * z as f64;
                            let expected_c = (mass / 111.1 * 4.9384).clamp(1.0, 500.0);
                            let expected_ratio = expected_c as f32 * 0.0108;
                            obs_ratio >= expected_ratio * 0.2 && obs_ratio <= expected_ratio * 2.0 + 0.1
                        }
                        IsotopeModel::Lipid => {
                            let mass = m0.mean_mz * z as f64;
                            let expected_c = (mass / 16.0).clamp(10.0, 120.0);
                            let expected_ratio = expected_c as f32 * 0.0108;
                            obs_ratio >= expected_ratio * 0.3 && obs_ratio <= expected_ratio * 2.2 + 0.1
                        }
                        IsotopeModel::Permissive => true,
                    };

                    if is_valid {
                        let avg_corr: f32 = matched_satellites.iter().map(|s| s.2).sum::<f32>() / matched_satellites.len() as f32;
                        let score = avg_corr + matched_satellites.len() as f32;

                        if score > best_score {
                            best_score = score;

                            let mut isotopes = Vec::new();
                            isotopes.push(IsotopePeak {
                                isotope_index: 0,
                                mz: m0.mean_mz,
                                apex_intensity: m0.apex_intensity,
                                area: m0.area,
                            });

                            let mut total_area = m0.area;
                            for &(sat_idx, iso_idx, _) in &matched_satellites {
                                let p = &sorted_peaks[sat_idx];
                                total_area += p.area;
                                isotopes.push(IsotopePeak {
                                    isotope_index: iso_idx,
                                    mz: p.mean_mz,
                                    apex_intensity: p.apex_intensity,
                                    area: p.area,
                                });
                            }

                            let feat = DeisotopedFeature {
                                monoisotopic_mz: m0.mean_mz,
                                apex_mz: m0.apex_mz,
                                charge: z,
                                apex_rt: m0.apex_rt,
                                rt_start: m0.rt_start,
                                rt_end: m0.rt_end,
                                monoisotopic_area: m0.area,
                                total_area,
                                fwhm: m0.fwhm,
                                snr: m0.snr,
                                isotope_correlation: avg_corr,
                                isotopes,
                            };

                            best_feature = Some((feat, matched_indices));
                        }
                    }
                }
            }

            if let Some((feat, matched_indices)) = best_feature {
                consumed[i] = true;
                for idx in matched_indices {
                    consumed[idx] = true;
                }
                features.push(feat);
            } else {
                // Feature with charge 1 and no observed isotope satellites
                consumed[i] = true;
                features.push(DeisotopedFeature {
                    monoisotopic_mz: m0.mean_mz,
                    apex_mz: m0.apex_mz,
                    charge: 1,
                    apex_rt: m0.apex_rt,
                    rt_start: m0.rt_start,
                    rt_end: m0.rt_end,
                    monoisotopic_area: m0.area,
                    total_area: m0.area,
                    fwhm: m0.fwhm,
                    snr: m0.snr,
                    isotope_correlation: 1.0,
                    isotopes: vec![IsotopePeak {
                        isotope_index: 0,
                        mz: m0.mean_mz,
                        apex_intensity: m0.apex_intensity,
                        area: m0.area,
                    }],
                });
            }
        }

        // Sort resulting features by retention time apex
        features.sort_by(|a, b| a.apex_rt.total_cmp(&b.apex_rt));
        features
    }
}

/// Computes Pearson correlation coefficient between two chromatographic peak profiles.
fn pearson_correlation(rts1: &[f32], ints1: &[f32], rts2: &[f32], ints2: &[f32]) -> f32 {
    if rts1.is_empty() || rts2.is_empty() {
        return 0.0;
    }

    let start_rt = rts1[0].max(rts2[0]);
    let end_rt = rts1.last().unwrap().min(*rts2.last().unwrap());

    if start_rt >= end_rt {
        return 0.0;
    }

    // Resample onto 10 uniform query points within the overlapping RT range
    const SAMPLES: usize = 10;
    let step = (end_rt - start_rt) / (SAMPLES - 1) as f32;

    let mut y1 = [0.0f32; SAMPLES];
    let mut y2 = [0.0f32; SAMPLES];

    for k in 0..SAMPLES {
        let t = start_rt + k as f32 * step;
        y1[k] = interpolate_intensity(rts1, ints1, t);
        y2[k] = interpolate_intensity(rts2, ints2, t);
    }

    // Mean
    let mean1: f32 = y1.iter().sum::<f32>() / SAMPLES as f32;
    let mean2: f32 = y2.iter().sum::<f32>() / SAMPLES as f32;

    let mut num = 0.0;
    let mut den1 = 0.0;
    let mut den2 = 0.0;

    for k in 0..SAMPLES {
        let d1 = y1[k] - mean1;
        let d2 = y2[k] - mean2;
        num += d1 * d2;
        den1 += d1 * d1;
        den2 += d2 * d2;
    }

    let denom = (den1 * den2).sqrt();
    if denom > 1e-6 {
        (num / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Linear interpolation of intensity at time t.
fn interpolate_intensity(rts: &[f32], ints: &[f32], t: f32) -> f32 {
    if t <= rts[0] {
        return ints[0];
    }
    if t >= *rts.last().unwrap() {
        return *ints.last().unwrap();
    }

    // Binary search for interval
    let idx = match rts.binary_search_by(|r| r.total_cmp(&t)) {
        Ok(i) => return ints[i],
        Err(i) => i.saturating_sub(1),
    };

    if idx + 1 < rts.len() {
        let frac = (t - rts[idx]) / (rts[idx + 1] - rts[idx]).max(1e-6);
        ints[idx] + frac * (ints[idx + 1] - ints[idx])
    } else {
        ints[idx]
    }
}
