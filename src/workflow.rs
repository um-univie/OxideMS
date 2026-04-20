use crate::spectrum::{MassSpectralRecord, Peak};
use crate::struct_with_new;
use chemistry_consts::NEUTRON_MASS;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum MassTolerance {
    Da(f64),
    Ppm(f64),
}

impl MassTolerance {
    /// Returns the absolute tolerance in Daltons for a given m/z value.
    pub fn tolerance_at(&self, mz: f64) -> f64 {
        match self {
            MassTolerance::Da(da) => *da,
            MassTolerance::Ppm(ppm) => (ppm * mz) / 1_000_000.0,
        }
    }

    /// Checks if the difference between two m/z values is within the tolerance.
    /// This implementation uses the max m/z of the pair to be conservative.
    pub fn is_within(&self, mz1: f64, mz2: f64) -> bool {
        let diff = (mz1 - mz2).abs();
        let max_mz = mz1.max(mz2);
        diff <= self.tolerance_at(max_mz)
    }
}

// ---------------------------------------------------------
// 1. Define the traits
// ---------------------------------------------------------

/// The core trait that the WorkFlowManager actually uses.
pub trait Operation: Send + Sync {
    fn operate(&self, spectra: &mut [MassSpectralRecord], use_parallel: bool);
}

/// A convenience trait for operations that only need to look at one spectrum.
pub trait SingleOperation: Send + Sync {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord);
}

// Automatically implement `Operation` for anything that implements `SingleOperation`.
impl<T: SingleOperation + Sync> Operation for T {
    fn operate(&self, spectra: &mut [MassSpectralRecord], use_parallel: bool) {
        if use_parallel {
            spectra
                .par_iter_mut()
                .for_each(|spectrum| self.operate_single(spectrum));
        } else {
            spectra
                .iter_mut()
                .for_each(|spectrum| self.operate_single(spectrum));
        }
    }
}

pub struct WorkFlowManager {
    // Rust requires trait objects to be boxed so their size is known at compile time.
    workflow: Vec<Box<dyn Operation>>,
    pub use_parallel: bool,
}

impl Default for WorkFlowManager {
    fn default() -> Self {
        Self {
            workflow: Vec::new(),
            use_parallel: true,
        }
    }
}

impl WorkFlowManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_operation(&mut self, op: Box<dyn Operation>) {
        self.workflow.push(op);
    }

    pub fn execute(&self, spectra: &mut [MassSpectralRecord]) {
        // The manager now just passes the whole slice to every operation.
        // The traits figure out internally whether to loop or handle the whole slice.
        for operation in self.workflow.iter() {
            operation.operate(spectra, self.use_parallel);
        }
    }
}

pub struct Normalize {}

impl SingleOperation for Normalize {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let max = spectrum
            .peaks_mut()
            .iter()
            .max_by(|peak1, peak2| {
                peak1
                    .intensity
                    .partial_cmp(&peak2.intensity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&Peak::default())
            .intensity;

        // Avoid division by zero in empty spectra
        if max > 0.0 {
            spectrum
                .peaks_mut()
                .iter_mut()
                .for_each(|peak| peak.intensity /= max);
        }
    }
}

pub struct AlignSpectra {}

impl Operation for AlignSpectra {
    fn operate(&self, spectra: &mut [MassSpectralRecord], use_parallel: bool) {
        // Cross-spectrum logic goes here
        println!("Aligning {} spectra together!", spectra.len());
    }
}

struct_with_new!(
    pub struct KeepTopNPeaks {
        number_of_peaks: usize,
    }
);

impl SingleOperation for KeepTopNPeaks {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        keep_top_n_peaks(&mut spectrum.peaks_mut(), self.number_of_peaks)
    }
}
fn keep_top_n_peaks(vec: &mut Vec<Peak>, n: usize) {
    if n == 0 {
        vec.clear();
    }
    if n >= vec.len() {
        return;
    }

    // Partition the vector. We compare `b.1` to `a.1` for descending order.
    // `total_cmp` provides the strict std::cmp::Ordering required.
    vec.select_nth_unstable_by(n - 1, |a, b| b.intensity.total_cmp(&a.intensity));

    // Discard everything after the top `n` elements
    vec.truncate(n);
    // Sort vec for other operations
    vec.sort_by(|a, b| a.mz.total_cmp(&b.mz));
}

struct_with_new!(
    pub struct FilterMZ {
        min_mz: f64,
        max_mz: f64,
    }
);

impl SingleOperation for FilterMZ {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        spectrum
            .peaks_mut()
            .retain(|peak| peak.mz > self.min_mz && peak.mz < self.max_mz)
    }
}

struct_with_new!(
    pub struct FilterIntensity {
        min_intensity: f32,
        max_intensity: f32,
    }
);

impl SingleOperation for FilterIntensity {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        spectrum.peaks_mut().retain(|peak| {
            peak.intensity > self.min_intensity && peak.intensity < self.max_intensity
        })
    }
}

struct_with_new!(
    pub struct FilterIntensityRelative {
        min_relative_intensity: f32,
    }
);

impl SingleOperation for FilterIntensityRelative {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let max_value = spectrum
            .peaks_mut()
            .iter()
            .max_by(|peak1, peak2| peak1.intensity.total_cmp(&peak2.intensity))
            .copied()
            .unwrap_or(Peak::default());
        let threshhold = self.min_relative_intensity * max_value.intensity;
        spectrum
            .peaks_mut()
            .retain(|peak| peak.intensity > threshhold)
    }
}

struct_with_new!(
    pub struct MinMZRangeFilter {
        min_mz_range: f64,
    }
);

impl SingleOperation for MinMZRangeFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        // let min = spectrum
        //     .peaks_mut()
        //     .iter()
        //     .min_by(|a, b| a.mz.total_cmp(&b.mz))
        //     .copied()
        //     .unwrap_or(Peak::default());
        let min = spectrum.peaks().first().map(|next| next.mz).unwrap_or(0.0);
        // let max = spectrum
        //     .peaks_mut()
        //     .iter()
        //     .max_by(|a, b| a.mz.total_cmp(&b.mz))
        //     .copied()
        //     .unwrap_or(Peak::default());
        let max = spectrum.peaks().last().map(|next| next.mz).unwrap_or(0.0);

        if max - min < self.min_mz_range {
            spectrum.peaks_mut().clear()
        }
    }
}

pub struct SortPeaks {}

impl SingleOperation for SortPeaks {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        spectrum
            .peaks_mut()
            .sort_by(|peak1, peak2| peak1.mz.total_cmp(&peak2.mz));
    }
}

struct_with_new!(
    pub struct CentroidPeaks {
        pub mz_tolerance: f64,
    }
);

impl SingleOperation for CentroidPeaks {
    // TODO Optimize this
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks_mut().is_empty() {
            return;
        }

        // 1. Sort the peaks by m/z to ensure nearby peaks are adjacent in the slice.
        // Not necessary anymore as we ensure sorting on MassSpectralRecord creation
        // spectrum
        //     .peaks_mut()
        //     .sort_unstable_by(|a, b| a.mz.total_cmp(&b.mz));

        // Pre-allocate the new vector to avoid reallocations.
        // It will be at most the size of the original.
        let mut merged_peaks = Vec::with_capacity(spectrum.peaks_mut().len());

        // Track the current grouping
        let mut group_start_mz = spectrum.peaks_mut()[0].mz;
        let mut current_mz_intensity_product = 0.0;
        let mut current_intensity_sum = 0.0;

        for peak in spectrum.peaks_mut() {
            // Check if the current peak is within the tolerance window of the group's start
            if peak.mz - group_start_mz <= self.mz_tolerance {
                current_mz_intensity_product += peak.mz * peak.intensity as f64;
                current_intensity_sum += peak.intensity;
            } else {
                // The peak is outside the tolerance. Finalize the previous group.
                if current_intensity_sum > 0.0 {
                    merged_peaks.push(Peak {
                        mz: current_mz_intensity_product / current_intensity_sum as f64,
                        intensity: current_intensity_sum,
                    });
                }

                // Start a new group with the current peak
                group_start_mz = peak.mz;
                current_mz_intensity_product = peak.mz * peak.intensity as f64;
                current_intensity_sum = peak.intensity;
            }
        }

        if current_intensity_sum > 0.0 {
            merged_peaks.push(Peak {
                mz: current_mz_intensity_product / current_intensity_sum as f64,
                intensity: current_intensity_sum,
            });
        }

        // Overwrite the old spectrum with the cleanly merged peaks
        *spectrum.peaks_mut() = merged_peaks;
    }
}

struct_with_new!(
    pub struct Deisotope {
        pub max_charge: u32, // The maximum charge state to check for (e.g., 3 or 4 for typical proteomics)
        pub tolerance: f64,  // The m/z tolerance for matching an isotope (e.g., 0.05)
        pub max_isotopes: u32, // How many isotopes deep to look (e.g., 5 to catch up to M+5)
    }
);

impl SingleOperation for Deisotope {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks_mut().is_empty() {
            return;
        }

        // 1. Ensure peaks are sorted by m/z for the sweep algorithm
        // Not necessary anymore as we ensure sorting on MassSpectralRecord creation
        // spectrum
        //     .peaks_mut()
        //     .sort_unstable_by(|a, b| a.mz.total_cmp(&b.mz));

        // Create a mask to track which peaks are isotopes so we can remove them later
        let mut is_isotope = vec![false; spectrum.peaks_mut().len()];

        for i in 0..spectrum.peaks_mut().len() {
            // Skip if this peak was already claimed as an isotope by a lighter peak
            if is_isotope[i] {
                continue;
            }

            // Test for isotopic envelopes across all expected charge states
            for z in 1..=self.max_charge {
                let mut current_mz = spectrum.peaks_mut()[i].mz;
                let step = NEUTRON_MASS / (z as f64);

                // Chain look-aheads (find M+1, then use M+1 to find M+2, etc.)
                for _ in 0..self.max_isotopes {
                    let expected_mz = current_mz + step;
                    let mut found_in_this_step = false;

                    // Look ahead in the spectrum
                    for j in (i + 1)..spectrum.peaks_mut().len() {
                        if is_isotope[j] {
                            continue;
                        }
                        let target_mz = spectrum.peaks_mut()[j].mz;

                        // Because the array is sorted, if we exceed the window, we can stop looking
                        if target_mz > expected_mz + self.tolerance {
                            break;
                        }

                        // If it falls within the tolerance window, we found an isotope!
                        if (target_mz - expected_mz).abs() <= self.tolerance {
                            is_isotope[j] = true;

                            // Collapse the intensity into the parent monoisotopic peak
                            let extra_intensity = spectrum.peaks_mut()[j].intensity;
                            spectrum.peaks_mut()[i].intensity += extra_intensity;

                            current_mz = target_mz; // Update current_mz to look for the next in the chain
                            found_in_this_step = true;
                            break;
                        }
                    }

                    // If we didn't find an M+1, there's no mathematical sense in looking for M+2
                    if !found_in_this_step {
                        break;
                    }
                }
            }
        }

        // 2. Finally, filter the spectrum, keeping only the monoisotopic peaks
        let mut iter = is_isotope.into_iter();
        spectrum.peaks_mut().retain(|_| !iter.next().unwrap());
    }
}

struct_with_new!(
    pub struct ReplacePrecursor {}
);

impl SingleOperation for ReplacePrecursor {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let max_mz = spectrum.peaks().last().map(|last| last.mz).unwrap_or(0.0);
        spectrum.insert_metadata("PEPMASS", max_mz.to_string().as_str());
    }
}

struct_with_new!(
    pub struct MinNumberOfPeaks {
        min_number_of_peaks: usize,
    }
);

impl SingleOperation for MinNumberOfPeaks {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().len() < self.min_number_of_peaks {
            *spectrum.peaks_mut() = Vec::new();
        }
    }
}

struct_with_new!(
    pub struct RemovePrecursor {
        pub mode: MassTolerance,
        pub isotopes: i32,
    }
);

impl SingleOperation for RemovePrecursor {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        // 1. Extract Precursor MZ
        let precursor_mz = spectrum
            .get_metadata("PEPMASS")
            .and_then(|mz_str| mz_str.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok());

        // 2. Extract Precursor Charge (defaulting to 1 if not found or unparseable)
        let precursor_charge = spectrum.get_charge();

        let Some(mz_val) = precursor_mz else {
            return;
        };

        // 3. Calculate target m/z values to remove (matching Python logic)
        let adduct_mass = 1.007825;
        let c_mass_diff = 1.003355;
        let neutral_mass = (mz_val - adduct_mass) * (precursor_charge as f64);

        let mut remove_mzs = Vec::new();
        for charge in (1..=precursor_charge).rev() {
            for iso in 0..=self.isotopes {
                let mz =
                    (neutral_mass + (iso as f64 * c_mass_diff)) / (charge as f64) + adduct_mass;
                remove_mzs.push(mz);
            }
        }

        // 4. Filter peaks
        spectrum.peaks_mut().retain(|peak| {
            // Check if this peak matches ANY of our calculated removal m/zs
            !remove_mzs.iter().any(|&target_mz| match self.mode {
                MassTolerance::Da(tolerance) => (peak.mz - target_mz).abs() <= tolerance,
                MassTolerance::Ppm(tolerance) => {
                    ((peak.mz - target_mz).abs() / target_mz) * 1e6 <= tolerance
                }
            })
        });
    }
}

struct_with_new!(
    pub struct AbsoluteIntensityFilter {
        min_base_peak_intensity: f32,
    }
);

impl SingleOperation for AbsoluteIntensityFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let max_intensity = spectrum
            .peaks()
            .iter()
            .map(|p| p.intensity)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        if max_intensity < self.min_base_peak_intensity {
            spectrum.peaks_mut().clear();
        }
    }
}

struct_with_new!(
    pub struct LocalSNRFilter {
        snr_threshold: f32,
    }
);

impl SingleOperation for LocalSNRFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() {
            return;
        }

        // Copy intensities to find the noise floor
        let mut intensities: Vec<f32> = spectrum.peaks().iter().map(|p| p.intensity).collect();
        intensities.sort_by(|a, b| a.total_cmp(b));

        // Estimate noise floor as the median of the bottom 50% of peaks
        let mid = intensities.len() / 4;
        let noise_floor = intensities[mid.max(0)];

        let min_allowed = noise_floor * self.snr_threshold;

        spectrum.peaks_mut().retain(|p| p.intensity >= min_allowed);
    }
}

struct_with_new!(
    pub struct PeakDensityFilter {
        max_peaks_per_100_da: usize,
    }
);

impl SingleOperation for PeakDensityFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let peaks = spectrum.peaks();
        if peaks.len() < self.max_peaks_per_100_da {
            return;
        }

        let mut trash = false;
        // Sliding window check
        for i in 0..peaks.len() {
            let window_start = peaks[i].mz;
            let window_end = window_start + 100.0;

            // Count how many peaks fall into this 100 Da window
            let count = peaks[i..].iter().take_while(|p| p.mz <= window_end).count();

            if count > self.max_peaks_per_100_da {
                trash = true;
                break;
            }
        }

        if trash {
            spectrum.peaks_mut().clear();
        }
    }
}

struct_with_new!(
    pub struct SquareRootTransformIntensity {}
);

impl SingleOperation for SquareRootTransformIntensity {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() {
            return;
        }

        spectrum
            .peaks_mut()
            .iter_mut()
            .for_each(|p| p.intensity = p.intensity.sqrt());
    }
}
