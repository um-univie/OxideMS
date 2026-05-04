//! Composable spectrum-processing pipeline.
//!
//! Build a [`WorkFlowManager`], push [`Operation`]s onto it in order, then
//! call [`WorkFlowManager::execute`]. Most operations implement the
//! [`SingleOperation`] convenience trait and receive parallel/sequential
//! dispatch over all spectra automatically.
//!
//! # Quick start
//!
//! ```
//! use oxidems_lib::MassSpectralRecord;
//! use oxidems_lib::workflow::{WorkFlowManager, Normalize, FilterMZ, KeepTopNPeaks};
//!
//! let mut spectra = vec![
//!     MassSpectralRecord::from_tuples(&[(50.0f32, 100.0), (200.0, 800.0), (600.0, 400.0)]),
//! ];
//!
//! let mut wfm = WorkFlowManager::new();
//! wfm.use_parallel = false;
//! wfm.add_operation(FilterMZ::new(100.0, 500.0)); // drop outside (100, 500) Da
//! wfm.add_operation(Box::new(Normalize {}));        // scale max intensity to 1.0
//! wfm.execute(&mut spectra);
//!
//! assert_eq!(spectra[0].peaks().len(), 1);          // only the 200 Da peak survived
//! assert!((spectra[0].peaks()[0].intensity - 1.0).abs() < 1e-6);
//! ```

use crate::clustering::cluster::blacklist::{NEG_CONTAMINANTS, POS_CONTAMINANTS};
use crate::spectrum::{MassSpectralRecord, Peak};
use crate::struct_with_new;
use chemistry_consts::ISOTOPES;
use chemistry_consts::NEUTRON_MASS_AMU as NEUTRON_MASS;
use std::any::TypeId;
// C13-C12 monoisotopic mass difference: the expected m/z spacing between isotope peaks
// for organic molecules at charge state z=1. Derived from IUPAC isotope masses.
const C13_MASS_DIFF: f64 = ISOTOPES[6][1].unwrap().mass - ISOTOPES[6][0].unwrap().mass;
use rayon::prelude::*;

/// Expresses a peak-matching tolerance in either absolute (Da) or relative (ppm) units.
#[derive(Clone, Copy, Debug)]
pub enum MassTolerance {
    /// Absolute tolerance in Daltons — constant regardless of m/z.
    Da(f64),
    /// Parts-per-million tolerance — scales linearly with the m/z being measured.
    Ppm(f64),
}

impl MassTolerance {
    /// Returns the absolute tolerance in Daltons at a given m/z value.
    ///
    /// For `Da` tolerances this is a constant; for `Ppm` it grows linearly with m/z.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::workflow::MassTolerance;
    ///
    /// let da = MassTolerance::Da(0.02);
    /// assert_eq!(da.tolerance_at(500.0), 0.02);
    /// assert_eq!(da.tolerance_at(1000.0), 0.02); // unchanged
    ///
    /// let ppm = MassTolerance::Ppm(10.0);
    /// assert!((ppm.tolerance_at(500.0) - 0.005).abs() < 1e-10);
    /// assert!((ppm.tolerance_at(1000.0) - 0.01).abs() < 1e-10);
    /// ```
    pub fn tolerance_at(&self, mz: f64) -> f64 {
        match self {
            MassTolerance::Da(da) => *da,
            MassTolerance::Ppm(ppm) => (ppm * mz) / 1_000_000.0,
        }
    }

    /// Returns `true` if the two m/z values are within tolerance of each other.
    ///
    /// The tolerance is evaluated at the larger of the two m/z values to be conservative.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::workflow::MassTolerance;
    ///
    /// let tol = MassTolerance::Da(0.05);
    /// assert!(tol.is_within(200.00, 200.03));
    /// assert!(!tol.is_within(200.00, 200.10));
    /// ```
    pub fn is_within(&self, mz1: f64, mz2: f64) -> bool {
        let diff = (mz1 - mz2).abs();
        let max_mz = mz1.max(mz2);
        diff <= self.tolerance_at(max_mz)
    }
}

/// Core trait for pipeline steps that operate over a slice of spectra.
///
/// Implement this directly when an operation needs to see or compare multiple
/// spectra at once (e.g., cross-spectrum alignment). For per-spectrum work,
/// implement [`SingleOperation`] instead — the blanket impl provides `Operation`
/// automatically, including rayon parallelism.
pub trait Operation: Send + Sync {
    fn operate(&self, spectra: &mut [MassSpectralRecord], use_parallel: bool);
}

/// Convenience trait for operations that transform one spectrum at a time.
///
/// Implementing this is sufficient for the vast majority of operations.
/// The blanket `impl<T: SingleOperation + Sync> Operation for T` handles
/// the parallel/sequential fan-out over all spectra automatically.
pub trait SingleOperation: Send + Sync {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord);
}

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

/// Runs a sequence of [`Operation`]s over a collection of spectra in insertion order.
///
/// Set [`WorkFlowManager::use_parallel`] to `false` for single-threaded execution,
/// which is useful in tests or when the caller already parallelises at a higher level.
///
/// # Examples
///
/// ```
/// use oxidems_lib::MassSpectralRecord;
/// use oxidems_lib::workflow::{WorkFlowManager, Normalize, KeepTopNPeaks};
///
/// let mut spectra = vec![
///     MassSpectralRecord::from_tuples(&[(100.0f32, 200.0), (200.0, 1000.0), (300.0, 50.0)]),
/// ];
///
/// let mut wfm = WorkFlowManager::new();
/// wfm.use_parallel = false;
/// wfm.add_operation(Box::new(Normalize {}));
/// wfm.add_operation(KeepTopNPeaks::new(2));
/// wfm.execute(&mut spectra);
///
/// assert_eq!(spectra[0].peaks().len(), 2);
/// let max_intensity = spectra[0].peaks().iter().map(|p| p.intensity).fold(f32::NEG_INFINITY, f32::max);
/// assert!((max_intensity - 1.0).abs() < 1e-6);
/// ```
pub struct WorkFlowManager {
    workflow: Vec<(TypeId, Box<dyn Operation>)>,
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

    /// Appends an operation to the end of the pipeline.
    ///
    /// The concrete type `T` is captured so that [`replace_operation`] and
    /// [`remove_operation`] can locate the step later by type.
    pub fn add_operation<T: Operation + 'static>(&mut self, op: Box<T>) {
        self.workflow.push((TypeId::of::<T>(), op));
    }

    /// Replaces the step of type `T` with `op`, preserving its position.
    ///
    /// Returns `true` if found and replaced, `false` if `T` is not in the pipeline.
    pub fn replace_operation<T: Operation + 'static>(&mut self, op: Box<T>) -> bool {
        let target = TypeId::of::<T>();
        if let Some((_, slot)) = self.workflow.iter_mut().find(|(id, _)| *id == target) {
            *slot = op;
            true
        } else {
            false
        }
    }

    /// Removes the step of type `T` from the pipeline.
    ///
    /// Returns `true` if found and removed, `false` if `T` is not present.
    pub fn remove_operation<T: Operation + 'static>(&mut self) -> bool {
        let target = TypeId::of::<T>();
        if let Some(i) = self.workflow.iter().position(|(id, _)| *id == target) {
            self.workflow.remove(i);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the pipeline contains a step of type `T`.
    pub fn has_operation<T: Operation + 'static>(&self) -> bool {
        let target = TypeId::of::<T>();
        self.workflow.iter().any(|(id, _)| *id == target)
    }

    /// Iterates over all operations in pipeline order.
    pub fn operations(&self) -> impl Iterator<Item = &dyn Operation> {
        self.workflow.iter().map(|(_, op)| op.as_ref())
    }

    /// Runs every operation in the pipeline over `spectra` in order.
    pub fn execute(&self, spectra: &mut [MassSpectralRecord]) {
        for (_, op) in &self.workflow {
            op.operate(spectra, self.use_parallel);
        }
    }
}

/// Scales all peak intensities so the most intense peak equals 1.0.
///
/// Spectra with no peaks or a zero-intensity base peak are left unchanged.
///
/// # Examples
///
/// ```
/// use oxidems_lib::MassSpectralRecord;
/// use oxidems_lib::workflow::{SingleOperation, Normalize};
///
/// let mut s = MassSpectralRecord::from_tuples(&[(100.0f32, 200.0), (200.0, 800.0)]);
/// Normalize {}.operate_single(&mut s);
///
/// assert!((s.peaks()[0].intensity - 0.25).abs() < 1e-6);
/// assert!((s.peaks()[1].intensity - 1.0).abs() < 1e-6);
/// ```
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

        if max > 0.0 {
            spectrum
                .peaks_mut()
                .iter_mut()
                .for_each(|peak| peak.intensity /= max);
        }
    }
}


struct_with_new!(
    /// Retains only the `n` most intense peaks, then re-sorts the survivors by m/z.
    ///
    /// Uses a partial sort (`select_nth_unstable_by`) so the cost is O(n) for
    /// the partition step rather than O(n log n). Passing `n = 0` clears the spectrum.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, KeepTopNPeaks};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 50.0), (200.0, 900.0), (300.0, 300.0), (400.0, 100.0),
    /// ]);
    /// KeepTopNPeaks::new(2).operate_single(&mut s);
    ///
    /// // Top-2 by intensity are the 200 and 300 Da peaks, re-sorted by m/z.
    /// assert_eq!(s.peaks().len(), 2);
    /// assert!((s.peaks()[0].mz - 200.0).abs() < 1e-9);
    /// assert!((s.peaks()[1].mz - 300.0).abs() < 1e-9);
    /// ```
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

    vec.truncate(n);
    vec.sort_by(|a, b| a.mz.total_cmp(&b.mz));
}

struct_with_new!(
    /// Removes all peaks outside the exclusive m/z range `(min_mz, max_mz)`.
    ///
    /// Peaks at exactly `min_mz` or `max_mz` are also removed (strict inequality).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, FilterMZ};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (50.0f32, 100.0), (200.0, 800.0), (600.0, 400.0),
    /// ]);
    /// FilterMZ::new(100.0, 500.0).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 1);
    /// assert!((s.peaks()[0].mz - 200.0).abs() < 1e-9);
    /// ```
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
    /// Removes peaks whose absolute intensity falls outside `(min_intensity, max_intensity)`.
    ///
    /// Both bounds are exclusive. Use [`FilterIntensityRelative`] when you want a
    /// threshold expressed as a fraction of the base peak.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, FilterIntensity};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 5.0), (200.0, 50.0), (300.0, 500.0),
    /// ]);
    /// FilterIntensity::new(10.0, 200.0).operate_single(&mut s);
    ///
    /// // Only the 50.0 intensity peak survives (5 ≤ 10 and 500 ≥ 200 are both rejected).
    /// assert_eq!(s.peaks().len(), 1);
    /// assert!((s.peaks()[0].intensity - 50.0).abs() < 1e-4);
    /// ```
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
    /// Removes peaks below `min_relative_intensity * base_peak_intensity`.
    ///
    /// Normalise first (e.g. with [`Normalize`]) if you want a 0–1 threshold; otherwise
    /// the fraction is applied directly to the raw base-peak intensity.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, FilterIntensityRelative};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 10.0), (200.0, 100.0), (300.0, 1000.0),
    /// ]);
    /// // Keep only peaks above 5% of the base peak (1000 * 0.05 = 50).
    /// FilterIntensityRelative::new(0.05).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 2); // 10 is dropped, 100 and 1000 survive
    /// ```
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
    /// Clears the spectrum if the m/z span from the first to last peak is below `min_mz_range`.
    ///
    /// Peaks must be sorted by m/z (which is guaranteed by all standard constructors).
    /// Useful for discarding truncated or degenerate spectra before downstream processing.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, MinMZRangeFilter};
    ///
    /// let mut narrow = MassSpectralRecord::from_tuples(&[(200.0f32, 100.0), (205.0, 200.0)]);
    /// MinMZRangeFilter::new(50.0).operate_single(&mut narrow);
    /// assert!(narrow.peaks().is_empty()); // 205 - 200 = 5 Da < 50 Da threshold
    ///
    /// let mut wide = MassSpectralRecord::from_tuples(&[(100.0f32, 100.0), (600.0, 200.0)]);
    /// MinMZRangeFilter::new(50.0).operate_single(&mut wide);
    /// assert_eq!(wide.peaks().len(), 2); // 600 - 100 = 500 Da ≥ 50 Da threshold
    /// ```
    pub struct MinMZRangeFilter {
        min_mz_range: f64,
    }
);

impl SingleOperation for MinMZRangeFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let min = spectrum.peaks().first().map(|next| next.mz).unwrap_or(0.0);
        let max = spectrum.peaks().last().map(|next| next.mz).unwrap_or(0.0);

        if max - min < self.min_mz_range {
            spectrum.peaks_mut().clear()
        }
    }
}

/// Sorts peaks by m/z in ascending order.
///
/// All standard constructors (`new`, `from_tuples`) already produce sorted peaks,
/// so this is mainly useful after manual peak manipulation.
pub struct SortPeaks {}

impl SingleOperation for SortPeaks {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        spectrum
            .peaks_mut()
            .sort_by(|peak1, peak2| peak1.mz.total_cmp(&peak2.mz));
    }
}

struct_with_new!(
    /// Merges adjacent peaks within `mz_tolerance` Da into a single intensity-weighted centroid.
    ///
    /// Peaks must be sorted by m/z before calling this operation. The algorithm is a single
    /// linear pass: it groups consecutive peaks whose m/z is within `mz_tolerance` of the
    /// group's start, then emits one peak per group with mz = Σ(mz·intensity) / Σintensity.
    ///
    /// Note: because the window is anchored to the group start (not a sliding window), very
    /// wide natural clusters can span more than `mz_tolerance` and still be split.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, CentroidPeaks};
    ///
    /// // Two peaks within 0.05 Da of each other should merge; the third is separate.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 400.0), (100.02, 200.0), (200.0, 800.0),
    /// ]);
    /// CentroidPeaks::new(0.05).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 2);
    /// // Merged mz: (100.0*400 + 100.02*200) / 600 ≈ 100.007
    /// assert!((s.peaks()[0].mz - 100.007).abs() < 0.01);
    /// assert!((s.peaks()[0].intensity - 600.0).abs() < 1.0);
    /// ```
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

        let mut merged_peaks = Vec::with_capacity(spectrum.peaks_mut().len());
        let mut group_start_mz = spectrum.peaks_mut()[0].mz;
        let mut current_mz_intensity_product = 0.0;
        let mut current_intensity_sum = 0.0;

        for peak in spectrum.peaks_mut() {
            if peak.mz - group_start_mz <= self.mz_tolerance {
                current_mz_intensity_product += peak.mz * peak.intensity as f64;
                current_intensity_sum += peak.intensity;
            } else {
                if current_intensity_sum > 0.0 {
                    merged_peaks.push(Peak {
                        mz: current_mz_intensity_product / current_intensity_sum as f64,
                        intensity: current_intensity_sum,
                    });
                }

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

        *spectrum.peaks_mut() = merged_peaks;
    }
}

struct_with_new!(
    /// Collapses isotope peaks into their monoisotopic peak, adding their intensity upward.
    ///
    /// For each non-isotope peak, the algorithm searches for M+1, M+2, … M+`max_isotopes`
    /// satellites at the expected spacing of `C13_MASS_DIFF / z` Da (~1.003355 / z) for each
    /// charge state up to `max_charge`. Peaks within `tolerance` Da of an expected position are
    /// marked and their intensities are summed into the monoisotopic peak.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::{MassSpectralRecord, Peak};
    /// use oxidems_lib::workflow::{SingleOperation, Deisotope};
    ///
    /// // Monoisotopic peak at 500.0 Da; M+1 isotope at ~501.003 Da (C13-C12 spacing, z=1).
    /// let peaks = vec![Peak::new(500.0, 1000.0), Peak::new(501.003, 300.0)];
    /// let mut s = MassSpectralRecord::new(peaks, 1, 0);
    /// Deisotope::new(1, 0.02, 3).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 1);
    /// assert!((s.peaks()[0].intensity - 1300.0).abs() < 1.0);
    /// ```
    pub struct Deisotope {
        pub max_charge: u32,
        pub tolerance: f64,
        pub max_isotopes: u32,
    }
);

impl SingleOperation for Deisotope {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks_mut().is_empty() {
            return;
        }

        let mut is_isotope = vec![false; spectrum.peaks_mut().len()];

        for i in 0..spectrum.peaks_mut().len() {
            if is_isotope[i] {
                continue;
            }

            for z in 1..=self.max_charge {
                let mut current_mz = spectrum.peaks_mut()[i].mz;
                let step = C13_MASS_DIFF / (z as f64);

                for _ in 0..self.max_isotopes {
                    let expected_mz = current_mz + step;
                    let mut found_in_this_step = false;

                    for j in (i + 1)..spectrum.peaks_mut().len() {
                        if is_isotope[j] {
                            continue;
                        }
                        let target_mz = spectrum.peaks_mut()[j].mz;

                        // Array is sorted, so exceeding the window means we can stop
                        if target_mz > expected_mz + self.tolerance {
                            break;
                        }

                        if (target_mz - expected_mz).abs() <= self.tolerance {
                            is_isotope[j] = true;
                            let extra_intensity = spectrum.peaks_mut()[j].intensity;
                            spectrum.peaks_mut()[i].intensity += extra_intensity;
                            current_mz = target_mz;
                            found_in_this_step = true;
                            break;
                        }
                    }

                    // No M+1 means no M+2 either
                    if !found_in_this_step {
                        break;
                    }
                }
            }
        }

        let mut iter = is_isotope.into_iter();
        spectrum.peaks_mut().retain(|_| !iter.next().unwrap());
    }
}

struct_with_new!(
    /// Sets the `PEPMASS` metadata field to the m/z of the highest-m/z peak in the spectrum.
    ///
    /// Useful when a precursor mass is missing or needs to be inferred from the fragment
    /// ion envelope. Peaks must be sorted (highest m/z last).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, ReplacePrecursor};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[(100.0f32, 500.0), (450.0, 300.0)]);
    /// ReplacePrecursor::new().operate_single(&mut s);
    ///
    /// assert_eq!(s.get_metadata("PEPMASS"), Some("450"));
    /// ```
    pub struct ReplacePrecursor {}
);

impl SingleOperation for ReplacePrecursor {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let max_mz = spectrum.peaks().last().map(|last| last.mz).unwrap_or(0.0);
        spectrum.insert_metadata("PEPMASS", max_mz.to_string().as_str());
    }
}

struct_with_new!(
    /// Clears the peak list if fewer than `min_number_of_peaks` peaks are present.
    ///
    /// Apply after other filters to discard spectra that have become too sparse to be
    /// informative.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, MinNumberOfPeaks};
    ///
    /// let mut sparse = MassSpectralRecord::from_tuples(&[(100.0f32, 500.0), (200.0, 300.0)]);
    /// MinNumberOfPeaks::new(5).operate_single(&mut sparse);
    /// assert!(sparse.peaks().is_empty()); // only 2 peaks, below threshold of 5
    ///
    /// let mut rich = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 100.0), (200.0, 200.0), (300.0, 300.0),
    ///     (400.0, 400.0), (500.0, 500.0),
    /// ]);
    /// MinNumberOfPeaks::new(5).operate_single(&mut rich);
    /// assert_eq!(rich.peaks().len(), 5); // exactly at threshold, not cleared
    /// ```
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
    /// Removes peaks within `mode` tolerance of the precursor and its charge-state isotopes.
    ///
    /// Reads `PEPMASS` from the spectrum's metadata and `charge` from the spectrum itself.
    /// For each charge state from 1 to the precursor charge, and for each isotope from 0 to
    /// `isotopes`, it computes the expected m/z and removes any peak within `mode` tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::{MassSpectralRecord, Peak};
    /// use oxidems_lib::workflow::{SingleOperation, RemovePrecursor, MassTolerance};
    ///
    /// // Precursor at ~200.5 Da, charge +2.
    /// let peaks = vec![
    ///     Peak::new(100.0, 500.0),
    ///     Peak::new(200.5, 800.0),  // precursor
    ///     Peak::new(300.0, 400.0),
    /// ];
    /// let mut s = MassSpectralRecord::new(peaks, 2, 0);
    /// s.insert_metadata("PEPMASS", "200.5");
    /// RemovePrecursor::new(MassTolerance::Da(0.05), 0).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 2); // precursor peak removed
    /// ```
    pub struct RemovePrecursor {
        pub mode: MassTolerance,
        pub isotopes: i32,
    }
);

impl SingleOperation for RemovePrecursor {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let precursor_mz = spectrum
            .get_metadata("PEPMASS")
            .and_then(|mz_str| mz_str.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok());

        let precursor_charge = spectrum.get_charge();

        let Some(mz_val) = precursor_mz else {
            return;
        };

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

        spectrum.peaks_mut().retain(|peak| {
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
    /// Clears the spectrum if its base-peak (most intense) intensity is below the threshold.
    ///
    /// Unlike [`FilterIntensity`], which removes individual peaks, this operation discards
    /// the entire spectrum when the signal is too weak to be useful.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, AbsoluteIntensityFilter};
    ///
    /// let mut weak = MassSpectralRecord::from_tuples(&[(100.0f32, 50.0), (200.0, 80.0)]);
    /// AbsoluteIntensityFilter::new(100.0).operate_single(&mut weak);
    /// assert!(weak.peaks().is_empty()); // base peak 80 < threshold 100
    ///
    /// let mut strong = MassSpectralRecord::from_tuples(&[(100.0f32, 50.0), (200.0, 500.0)]);
    /// AbsoluteIntensityFilter::new(100.0).operate_single(&mut strong);
    /// assert_eq!(strong.peaks().len(), 2); // base peak 500 ≥ threshold 100
    /// ```
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
    /// Removes peaks below `snr_threshold` × the estimated noise floor.
    ///
    /// The noise floor is estimated as the intensity at the 25th percentile of all peaks
    /// (i.e. `sorted_intensities[n/4]`). Peaks below `noise_floor * snr_threshold` are
    /// discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, LocalSNRFilter};
    ///
    /// // Intensities sorted: [10, 20, 1000, 2000]. n=4, percentile index=1 → noise=20.
    /// // Threshold = 20 * 5.0 = 100. Peaks with intensity ≥ 100 survive.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 10.0), (200.0, 20.0), (300.0, 1000.0), (400.0, 2000.0),
    /// ]);
    /// LocalSNRFilter::new(5.0).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 2);
    /// ```
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
    /// Discards the entire spectrum if any 100 Da sliding window contains more than
    /// `max_peaks_per_100_da` peaks.
    ///
    /// This catches "chimeric" or highly contaminated spectra where peaks are so densely
    /// packed that meaningful interpretation is unlikely.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, PeakDensityFilter};
    ///
    /// // 5 peaks within a 40 Da window — exceeds a limit of 3 per 100 Da.
    /// let mut dense = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 100.0), (110.0, 100.0), (120.0, 100.0),
    ///     (130.0, 100.0), (140.0, 100.0),
    /// ]);
    /// PeakDensityFilter::new(3).operate_single(&mut dense);
    /// assert!(dense.peaks().is_empty());
    ///
    /// // 3 peaks spread across 400 Da — within the limit.
    /// let mut sparse = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 100.0), (300.0, 100.0), (500.0, 100.0),
    /// ]);
    /// PeakDensityFilter::new(3).operate_single(&mut sparse);
    /// assert_eq!(sparse.peaks().len(), 3);
    /// ```
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
    /// Applies a square-root transform to all peak intensities.
    ///
    /// Compresses the dynamic range of the spectrum, which can improve the performance of
    /// cosine-similarity-based spectral matching by down-weighting dominant peaks.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, SquareRootTransformIntensity};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[(100.0f32, 100.0), (200.0, 400.0)]);
    /// SquareRootTransformIntensity::new().operate_single(&mut s);
    ///
    /// assert!((s.peaks()[0].intensity - 10.0).abs() < 1e-4);
    /// assert!((s.peaks()[1].intensity - 20.0).abs() < 1e-4);
    /// ```
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

struct_with_new!(
    /// Applies a natural-log transform `ln(1 + x)` to all peak intensities.
    ///
    /// Compresses dynamic range more aggressively than square-root while avoiding
    /// `ln(0)` singularities. Useful when intensity distributions are highly skewed.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, LogTransformIntensity};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[(100.0f32, 0.0), (200.0, (std::f32::consts::E - 1.0))]);
    /// LogTransformIntensity::new().operate_single(&mut s);
    ///
    /// assert!((s.peaks()[0].intensity - 0.0).abs() < 1e-4);
    /// assert!((s.peaks()[1].intensity - 1.0).abs() < 1e-4);
    /// ```
    pub struct LogTransformIntensity {}
);

impl SingleOperation for LogTransformIntensity {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() {
            return;
        }

        spectrum
            .peaks_mut()
            .iter_mut()
            .for_each(|p| p.intensity = (1.0 + p.intensity).ln());
    }
}

struct_with_new!(
    pub struct RemoveCommonContaminants {
        tolerance: MassTolerance,
    }
);

impl SingleOperation for RemoveCommonContaminants {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() {
            return;
        }

        let common_contaminants = if spectrum.get_charge() < 0 {
            NEG_CONTAMINANTS.as_slice()
        } else if spectrum.get_charge() > 0 {
            POS_CONTAMINANTS.as_slice()
        } else {
            &[]
        };

        if common_contaminants.is_empty() {
            return;
        }

        let mut contaminant_pointer = 0;
        let mut spectrum_pointer = 0;

        let peaks = spectrum.peaks_mut();

        for read_pointer in 0..peaks.len() {
            let mz = peaks[read_pointer].mz;

            let (lower_bound, upper_bound) = match self.tolerance {
                MassTolerance::Da(da) => (mz - da, mz + da),
                MassTolerance::Ppm(ppm) => {
                    let delta = mz * (ppm / 1_000_000.0);
                    (mz - delta, mz + delta)
                }
            };

            while contaminant_pointer < common_contaminants.len()
                && common_contaminants[contaminant_pointer] < lower_bound
            {
                contaminant_pointer += 1;
            }

            let mut is_contaminant = false;
            let mut search_pointer = contaminant_pointer;

            // Check if any contaminants fall within the tolerance window
            while search_pointer < common_contaminants.len() {
                let c_mz = common_contaminants[search_pointer];

                if c_mz <= upper_bound {
                    is_contaminant = true;
                    break;
                } else {
                    break;
                }
            }

            if !is_contaminant {
                peaks.swap(spectrum_pointer, read_pointer);
                spectrum_pointer += 1;
            }
        }

        peaks.truncate(spectrum_pointer);
        peaks.sort_by(|peak1, peak2| peak1.mz.total_cmp(&peak2.mz));
    }
}
