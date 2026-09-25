use crate::workflow::MassTolerance;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

/// Theoretical isotope distribution model for isotopic envelope validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsotopeModel {
    /// Averagine model: standard peptide composition C_4.9384 H_7.7583 N_1.3577 O_1.4773 S_0.0417.
    Averagine,
    /// Small molecule / metabolomics empirical envelope model (carbon fraction ~ mass / 22.0).
    SmallMolecule,
    /// Lipid model (higher carbon/hydrogen ratio, lower nitrogen/oxygen).
    Lipid,
    /// Permissive: minimal check on isotope spacing without strict intensity ratios.
    Permissive,
}

/// Configuration knobs for the feature extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFinderConfig {
    /// Mass tolerance for ROI extraction and mass trace tracking.
    pub mass_tolerance: MassTolerance,

    /// Minimum intensity for an individual peak to be tracked or retained.
    pub min_peak_height: f32,

    /// Signal-to-noise ratio threshold for chromatographic peaks.
    pub snr_threshold: f32,

    /// Minimum number of consecutive/active scans required to form a valid mass trace.
    pub min_trace_length: usize,

    /// Maximum number of consecutive missing scans allowed before terminating a mass trace.
    pub max_missing_scans: usize,

    /// Maximum time gap (in seconds) between adjacent points in an active mass trace.
    pub max_rt_gap_seconds: f32,

    /// Expected chromatographic peak width range (min_fwhm, max_fwhm) in seconds.
    pub fwhm_range_seconds: (f32, f32),

    /// Wavelet scales (in seconds) for Continuous Wavelet Transform (CWT) peak detection.
    pub wavelet_scales: Vec<f32>,

    /// Minimum Pearson correlation coefficient for co-eluting isotope peak shapes (0.0 to 1.0).
    pub min_peak_correlation: f32,

    /// Maximum retention time difference (in seconds) between isotope peak apexes.
    pub max_isotope_rt_diff_seconds: f32,

    /// Minimum admissible charge state for deisotoping.
    pub min_charge: i8,

    /// Maximum admissible charge state for deisotoping.
    pub max_charge: i8,

    /// Mass tolerance for matching isotope satellite peaks (M+1, M+2...).
    pub isotope_tolerance: MassTolerance,

    /// Theoretical isotope envelope model.
    pub isotope_model: IsotopeModel,

    /// Mass bin partition size (in Daltons) for Rayon parallel trace extraction.
    pub mz_partition_size: f64,

    /// Overlap margin between mass partitions (in Daltons) to prevent edge splitting.
    pub mz_partition_overlap: f64,
}

impl Default for FeatureFinderConfig {
    fn default() -> Self {
        Self::metabolomics_qtof()
    }
}

impl FeatureFinderConfig {
    /// Returns the charge range as an inclusive range.
    pub fn charge_range(&self) -> RangeInclusive<i8> {
        self.min_charge..=self.max_charge
    }

    /// Preset optimized for Orbitrap instruments (e.g. Q Exactive, Orbitrap Exploris) in metabolomics.
    ///
    /// Characterized by high mass accuracy (3-5 ppm), clean background, and moderate scan rates (5-15 Hz).
    pub fn metabolomics_orbitrap() -> Self {
        Self {
            mass_tolerance: MassTolerance::Ppm(5.0),
            min_peak_height: 5_000.0,
            snr_threshold: 3.0,
            min_trace_length: 5,
            max_missing_scans: 1,
            max_rt_gap_seconds: 3.0,
            fwhm_range_seconds: (3.0, 25.0),
            wavelet_scales: vec![2.0, 4.0, 8.0, 16.0],
            min_peak_correlation: 0.80,
            max_isotope_rt_diff_seconds: 2.0,
            min_charge: 1,
            max_charge: 2,
            isotope_tolerance: MassTolerance::Ppm(5.0),
            isotope_model: IsotopeModel::SmallMolecule,
            mz_partition_size: 50.0,
            mz_partition_overlap: 0.1,
        }
    }

    /// Preset optimized for Q-TOF instruments (e.g. Bruker impact / timsTOF, Agilent 6500, Sciex ZenoTOF) in metabolomics.
    ///
    /// Characterized by moderate-to-high mass accuracy (8-12 ppm), fast scan rates (20-60 Hz),
    /// and higher point density across chromatographic peaks.
    pub fn metabolomics_qtof() -> Self {
        Self {
            mass_tolerance: MassTolerance::Ppm(10.0),
            min_peak_height: 500.0,
            snr_threshold: 3.0,
            min_trace_length: 6,
            max_missing_scans: 2,
            max_rt_gap_seconds: 1.5,
            fwhm_range_seconds: (2.0, 20.0),
            wavelet_scales: vec![2.0, 4.0, 8.0, 14.0, 20.0],
            min_peak_correlation: 0.75,
            max_isotope_rt_diff_seconds: 1.5,
            min_charge: 1,
            max_charge: 2,
            isotope_tolerance: MassTolerance::Ppm(10.0),
            isotope_model: IsotopeModel::SmallMolecule,
            mz_partition_size: 50.0,
            mz_partition_overlap: 0.15,
        }
    }

    /// Preset optimized for Orbitrap Astral instruments.
    ///
    /// Characterized by ultra-fast scan rates, narrow chromatographic peaks (short gradients / micro-flow),
    /// and high mass accuracy.
    pub fn astral() -> Self {
        Self {
            mass_tolerance: MassTolerance::Ppm(4.0),
            min_peak_height: 1_000.0,
            snr_threshold: 3.0,
            min_trace_length: 6,
            max_missing_scans: 2,
            max_rt_gap_seconds: 0.8,
            fwhm_range_seconds: (0.5, 6.0),
            wavelet_scales: vec![0.5, 1.0, 2.0, 4.0],
            min_peak_correlation: 0.80,
            max_isotope_rt_diff_seconds: 0.6,
            min_charge: 1,
            max_charge: 4,
            isotope_tolerance: MassTolerance::Ppm(4.0),
            isotope_model: IsotopeModel::SmallMolecule,
            mz_partition_size: 50.0,
            mz_partition_overlap: 0.1,
        }
    }

    /// Preset optimized for Ultra-High-Resolution FT-ICR MS.
    ///
    /// Characterized by sub-ppm mass accuracy and slower scan cycles.
    pub fn fticr() -> Self {
        Self {
            mass_tolerance: MassTolerance::Ppm(1.5),
            min_peak_height: 10_000.0,
            snr_threshold: 4.0,
            min_trace_length: 4,
            max_missing_scans: 1,
            max_rt_gap_seconds: 4.0,
            fwhm_range_seconds: (5.0, 30.0),
            wavelet_scales: vec![4.0, 8.0, 16.0, 24.0],
            min_peak_correlation: 0.85,
            max_isotope_rt_diff_seconds: 3.0,
            min_charge: 1,
            max_charge: 2,
            isotope_tolerance: MassTolerance::Ppm(1.5),
            isotope_model: IsotopeModel::SmallMolecule,
            mz_partition_size: 25.0,
            mz_partition_overlap: 0.05,
        }
    }

    /// Preset for bottom-up proteomics (Orbitrap).
    ///
    /// Enables higher charge states (up to +6) and the Averagine peptide isotope model.
    pub fn proteomics_orbitrap() -> Self {
        Self {
            mass_tolerance: MassTolerance::Ppm(5.0),
            min_peak_height: 5_000.0,
            snr_threshold: 3.0,
            min_trace_length: 5,
            max_missing_scans: 1,
            max_rt_gap_seconds: 3.0,
            fwhm_range_seconds: (4.0, 25.0),
            wavelet_scales: vec![2.0, 4.0, 8.0, 16.0],
            min_peak_correlation: 0.80,
            max_isotope_rt_diff_seconds: 2.0,
            min_charge: 2,
            max_charge: 6,
            isotope_tolerance: MassTolerance::Ppm(5.0),
            isotope_model: IsotopeModel::Averagine,
            mz_partition_size: 50.0,
            mz_partition_overlap: 0.1,
        }
    }

    /// Low-resolution / nominal mass preset (e.g. single/triple quadrupole, ion trap).
    pub fn low_res() -> Self {
        Self {
            mass_tolerance: MassTolerance::Da(0.3),
            min_peak_height: 100.0,
            snr_threshold: 3.0,
            min_trace_length: 5,
            max_missing_scans: 1,
            max_rt_gap_seconds: 3.0,
            fwhm_range_seconds: (4.0, 30.0),
            wavelet_scales: vec![2.0, 5.0, 10.0, 20.0],
            min_peak_correlation: 0.70,
            max_isotope_rt_diff_seconds: 2.5,
            min_charge: 1,
            max_charge: 2,
            isotope_tolerance: MassTolerance::Da(0.3),
            isotope_model: IsotopeModel::Permissive,
            mz_partition_size: 100.0,
            mz_partition_overlap: 0.5,
        }
    }
}
