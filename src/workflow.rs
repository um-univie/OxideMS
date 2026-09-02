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
//! wfm.add_operation(Normalize {});                  // scale max intensity to 1.0
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
const C13_MASS_DIFF: f64 = ISOTOPES[6][1].expect("C13 isotope entry").mass
    - ISOTOPES[6][0].expect("C12 isotope entry").mass;
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
/// wfm.add_operation(Normalize {});
/// wfm.add_operation(KeepTopNPeaks::new(2));
/// wfm.execute(&mut spectra);
///
/// assert_eq!(spectra[0].peaks().len(), 2);
/// let max_intensity = spectra[0].peaks().iter().map(|p| p.intensity).fold(f32::NEG_INFINITY, f32::max);
/// assert!((max_intensity - 1.0).abs() < 1e-6);
/// ```
pub struct WorkFlowManager {
    workflow: Vec<(TypeId, &'static str, Box<dyn Operation>)>,
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

/// Short, human-readable name for an operation type (e.g. `"FilterMZ"`), used to label
/// stages in a [`FilterReport`].
fn stage_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

impl WorkFlowManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an operation to the end of the pipeline.
    ///
    /// The concrete type `T` is captured so that [`replace_operation`] and
    /// [`remove_operation`] can locate the step later by type. Takes `op` by
    /// value and boxes it internally — pass a plain `SomeOperation { .. }` or
    /// `SomeOperation::new(..)`, no manual `Box::new` needed.
    pub fn add_operation<T: Operation + 'static>(&mut self, op: T) {
        self.workflow
            .push((TypeId::of::<T>(), stage_name::<T>(), Box::new(op)));
    }

    /// Replaces the step of type `T` with `op`, preserving its position.
    ///
    /// Returns `true` if found and replaced, `false` if `T` is not in the pipeline.
    pub fn replace_operation<T: Operation + 'static>(&mut self, op: T) -> bool {
        let target = TypeId::of::<T>();
        if let Some(entry) = self.workflow.iter_mut().find(|(id, _, _)| *id == target) {
            entry.2 = Box::new(op);
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
        if let Some(i) = self.workflow.iter().position(|(id, _, _)| *id == target) {
            self.workflow.remove(i);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the pipeline contains a step of type `T`.
    pub fn has_operation<T: Operation + 'static>(&self) -> bool {
        let target = TypeId::of::<T>();
        self.workflow.iter().any(|(id, _, _)| *id == target)
    }

    /// Iterates over all operations in pipeline order.
    pub fn operations(&self) -> impl Iterator<Item = &dyn Operation> {
        self.workflow.iter().map(|(_, _, op)| op.as_ref())
    }

    /// Runs every operation in the pipeline over `spectra` in order.
    pub fn execute(&self, spectra: &mut [MassSpectralRecord]) {
        for (_, _, op) in &self.workflow {
            op.operate(spectra, self.use_parallel);
        }
    }

    /// Like [`execute`](Self::execute), but also tracks how many spectra are newly left
    /// with zero peaks after each stage, attributing the drop to that stage's name.
    ///
    /// A spectrum that already had zero peaks before a stage runs is not re-counted at
    /// a later stage, so each spectrum is attributed to the single stage that first
    /// emptied it.
    pub fn execute_with_report(&self, spectra: &mut [MassSpectralRecord]) -> FilterReport {
        let mut report = FilterReport::new();
        for (_, name, op) in &self.workflow {
            let before_empty = spectra.iter().filter(|s| s.peaks().is_empty()).count();
            op.operate(spectra, self.use_parallel);
            let after_empty = spectra.iter().filter(|s| s.peaks().is_empty()).count();
            report.record(name, after_empty.saturating_sub(before_empty));
        }
        report
    }
}

/// Aggregated counts of how many spectra were dropped at each pipeline stage.
///
/// Built by [`WorkFlowManager::execute_with_report`]; [`SpectralExperiment`] additionally
/// records its post-workflow charge filter under the `"unknown_charge"` stage name.
/// Stages are kept in first-drop order, and a stage is only present if it dropped at
/// least one spectrum.
///
/// [`SpectralExperiment`]: crate::clustering::cluster::experiment::SpectralExperiment
#[derive(Default, Clone, Debug)]
pub struct FilterReport {
    stage_counts: Vec<(String, usize)>,
}

impl FilterReport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `count` drops attributed to `stage`, merging into an existing entry if the
    /// stage already appears in the report. A `count` of zero is a no-op.
    pub fn record(&mut self, stage: &str, count: usize) {
        if count == 0 {
            return;
        }
        if let Some(entry) = self.stage_counts.iter_mut().find(|(name, _)| name == stage) {
            entry.1 += count;
        } else {
            self.stage_counts.push((stage.to_string(), count));
        }
    }

    /// Merges another report's counts into this one, e.g. combining per-file reports
    /// produced by parallel ingestion.
    pub fn merge(&mut self, other: FilterReport) {
        for (stage, count) in other.stage_counts {
            self.record(&stage, count);
        }
    }

    /// Iterates over `(stage_name, dropped_count)` pairs in first-drop order.
    pub fn stages(&self) -> impl Iterator<Item = (&str, usize)> {
        self.stage_counts
            .iter()
            .map(|(name, count)| (name.as_str(), *count))
    }

    /// Total number of spectra dropped across all stages.
    pub fn total_dropped(&self) -> usize {
        self.stage_counts.iter().map(|(_, c)| c).sum()
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

/// Searches `peaks[anchor_idx + 1..]` for the best-explained isotope envelope anchored at
/// `peaks[anchor_idx]`, trying every charge state `1..=max_charge` independently and keeping
/// only the single charge state whose M+1, M+2, … run matches the most consecutive satellites
/// at the expected `C13_MASS_DIFF / z` spacing (ties favor the lower charge, since that's the
/// more common case for small-molecule fragments).
///
/// A candidate satellite is only accepted if its intensity does not exceed the intensity of
/// the peak immediately before it in the run (anchor, then M+1, then M+2, …). Natural isotope
/// abundance means a real M+n peak is essentially never more intense than M+(n-1) in this mass
/// range — without this check, an unrelated, more intense peak that merely happens to sit near
/// isotope spacing below a *weaker* peak gets absorbed as if the weak peak were its monoisotopic
/// anchor, deleting real signal (observed in practice: a minor H-loss fragment sitting ~1 Da
/// below a spectrum's actual base peak wrongly "explained away" the base peak as its M+1
/// isotope). Requiring non-increasing intensity along the run rules that out while still
/// allowing genuine envelopes, whose intensities fall off with each successive isotope.
///
/// Returns `(charge, satellite_indices, summed_satellite_intensity)`, or `None` if no charge
/// state matched even one satellite. Candidate satellites already claimed by another envelope
/// (`is_claimed[j]`) are skipped. Shared by [`Deisotope`] (which claims the returned satellites
/// permanently) and [`AssignPrecursorCharge`] (which only reads off the charge).
fn best_isotope_run(
    peaks: &[Peak],
    anchor_idx: usize,
    max_charge: u32,
    max_isotopes: u32,
    tolerance: f64,
    is_claimed: &[bool],
) -> Option<(u32, Vec<usize>, f32)> {
    let mut best: Option<(u32, Vec<usize>, f32)> = None;

    for z in 1..=max_charge {
        let step = C13_MASS_DIFF / (z as f64);
        let mut current_mz = peaks[anchor_idx].mz;
        let mut current_intensity = peaks[anchor_idx].intensity;
        let mut satellites = Vec::new();
        let mut total_extra = 0.0f32;

        for _ in 0..max_isotopes {
            let expected_mz = current_mz + step;
            let mut found = None;

            for j in (anchor_idx + 1)..peaks.len() {
                if is_claimed[j] || satellites.contains(&j) {
                    continue;
                }
                let target_mz = peaks[j].mz;

                // Array is sorted, so exceeding the window means we can stop.
                if target_mz > expected_mz + tolerance {
                    break;
                }

                if (target_mz - expected_mz).abs() <= tolerance
                    && peaks[j].intensity <= current_intensity
                {
                    found = Some(j);
                    break;
                }
            }

            match found {
                Some(j) => {
                    total_extra += peaks[j].intensity;
                    current_mz = peaks[j].mz;
                    current_intensity = peaks[j].intensity;
                    satellites.push(j);
                }
                // No M+1 means no M+2 either.
                None => break,
            }
        }

        if !satellites.is_empty() {
            let is_better = best
                .as_ref()
                .map(|(_, best_satellites, _)| satellites.len() > best_satellites.len())
                .unwrap_or(true);
            if is_better {
                best = Some((z, satellites, total_extra));
            }
        }
    }

    best
}

struct_with_new!(
    /// Collapses isotope peaks into their monoisotopic peak, adding their intensity upward.
    ///
    /// For each non-isotope peak, [`best_isotope_run`] tries every charge state `1..=max_charge`
    /// and keeps only the single charge state that explains the most consecutive M+1, M+2, …
    /// `max_isotopes` satellites at the expected `C13_MASS_DIFF / z` spacing (~1.003355 / z Da) —
    /// a peak's isotopes are never merged from more than one charge hypothesis at once. Matched
    /// satellites within `tolerance` Da are marked and their intensities summed into the
    /// monoisotopic peak.
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
        if spectrum.peaks().is_empty() {
            return;
        }

        let n = spectrum.peaks().len();
        let mut is_isotope = vec![false; n];

        for i in 0..n {
            if is_isotope[i] {
                continue;
            }

            let best = best_isotope_run(
                spectrum.peaks(),
                i,
                self.max_charge,
                self.max_isotopes,
                self.tolerance,
                &is_isotope,
            );

            if let Some((_, satellites, total_extra)) = best {
                for &j in &satellites {
                    is_isotope[j] = true;
                }
                spectrum.peaks_mut()[i].intensity += total_extra;
            }
        }

        // SAFETY: is_isotope has exactly one entry per peak (built as
        // vec![false; n] above) and peaks_mut()'s length hasn't changed
        // since, so this zips 1:1 with retain's iteration -- iter can't run
        // dry before retain does. Hot path: one call per peak per spectrum.
        let mut iter = is_isotope.into_iter();
        spectrum
            .peaks_mut()
            .retain(|_| !unsafe { iter.next().unwrap_unchecked() });
    }
}

struct_with_new!(
    /// Determines precursor charge from the isotope pattern of a surviving, unfragmented
    /// precursor peak in the MS2/MS3 spectrum, and writes it via
    /// [`MassSpectralRecord::set_charge`].
    ///
    /// Only applicable when the precursor ion itself still appears in the fragment list —
    /// common with soft/incomplete fragmentation, wide isolation windows, or DIA-style
    /// acquisition. This does **not** deduce the charge of individual fragment ions (assumed
    /// singly charged, the norm for small-molecule MS2); it only refines the spectrum's own
    /// `charge` field, and only when a real isotope envelope is found near `PEPMASS`.
    ///
    /// Algorithm: locate the peak closest to `PEPMASS` within `precursor_window` Da, then run
    /// [`best_isotope_run`] from that anchor — the same scoring [`Deisotope`] uses — trying
    /// every charge `1..=max_charge` and keeping the one that explains the most consecutive
    /// isotopes (ties favor the lower charge). Requires at least `min_isotopes` matched
    /// satellites to accept; a lone M+1 match is too weak to trust on its own. Leaves the
    /// spectrum untouched if there's no `PEPMASS`, no anchor peak within `precursor_window`,
    /// or too few isotopes matched.
    ///
    /// By default (`overwrite_existing = false`) a spectrum with an already-nonzero charge
    /// (e.g. parsed from a `CHARGE` field) is left alone; set it to `true` to let a detected
    /// pattern override existing metadata. Charge sign follows the spectrum's `IONMODE`
    /// metadata when present (`NEGATIVE` → `-z`, anything else → `+z`), since isotope spacing
    /// alone can't distinguish sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::{MassSpectralRecord, Peak};
    /// use oxidems_lib::workflow::{SingleOperation, AssignPrecursorCharge};
    ///
    /// // Precursor at 400.0 Da survives fragmentation with its +2 isotope envelope intact:
    /// // M+1 at 400.0 + 1.003355/2 ≈ 400.502, M+2 at ≈ 401.003.
    /// let peaks = vec![
    ///     Peak::new(150.0, 900.0),    // an ordinary singly-charged fragment
    ///     Peak::new(400.0, 500.0),    // surviving precursor, M+0
    ///     Peak::new(400.502, 150.0),  // M+1
    ///     Peak::new(401.003, 40.0),   // M+2
    /// ];
    /// let mut s = MassSpectralRecord::new(peaks, 0, 0); // charge unknown in source metadata
    /// s.insert_metadata("PEPMASS", "400.0");
    ///
    /// AssignPrecursorCharge::new(3, 3, 0.01, 0.05, 2, false).operate_single(&mut s);
    ///
    /// assert_eq!(s.get_charge(), 2);
    /// ```
    pub struct AssignPrecursorCharge {
        pub max_charge: u32,
        pub max_isotopes: u32,
        pub tolerance: f64,
        pub precursor_window: f64,
        pub min_isotopes: u32,
        pub overwrite_existing: bool,
    }
);

impl SingleOperation for AssignPrecursorCharge {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if !self.overwrite_existing && spectrum.get_charge() != 0 {
            return;
        }

        let precursor_mz = spectrum
            .get_metadata("PEPMASS")
            .and_then(|mz_str| mz_str.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok());

        let Some(precursor_mz) = precursor_mz else {
            return;
        };

        let anchor_idx = spectrum
            .peaks()
            .iter()
            .enumerate()
            .filter(|(_, peak)| (peak.mz - precursor_mz).abs() <= self.precursor_window)
            .min_by(|(_, a), (_, b)| {
                (a.mz - precursor_mz)
                    .abs()
                    .total_cmp(&(b.mz - precursor_mz).abs())
            })
            .map(|(idx, _)| idx);

        let Some(anchor_idx) = anchor_idx else {
            return;
        };

        let is_claimed = vec![false; spectrum.peaks().len()];
        let best = best_isotope_run(
            spectrum.peaks(),
            anchor_idx,
            self.max_charge,
            self.max_isotopes,
            self.tolerance,
            &is_claimed,
        );

        let Some((charge, satellites, _)) = best else {
            return;
        };
        if satellites.len() < self.min_isotopes as usize {
            return;
        }

        let negative = spectrum
            .get_metadata("IONMODE")
            .map(|mode| mode.eq_ignore_ascii_case("NEGATIVE"))
            .unwrap_or(false);

        spectrum.set_charge(if negative { -(charge as i8) } else { charge as i8 });
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

struct_with_new!(
    /// Scales peak intensities so they sum to 1.0 (total ion current normalization).
    ///
    /// Unlike [`Normalize`], which pins the base peak to 1.0, this preserves the overall
    /// signal shape: two spectra with the same relative peak pattern but different absolute
    /// magnitude become numerically comparable. Spectra with no peaks or zero total
    /// intensity are left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, TICNormalize};
    ///
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 100.0), (200.0, 300.0), (300.0, 600.0),
    /// ]);
    /// TICNormalize::new().operate_single(&mut s);
    ///
    /// let total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// assert!((s.peaks()[2].intensity - 0.6).abs() < 1e-6); // 600 / 1000
    /// ```
    pub struct TICNormalize {}
);

impl SingleOperation for TICNormalize {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        let total: f32 = spectrum.peaks().iter().map(|p| p.intensity).sum();
        if total > 0.0 {
            spectrum
                .peaks_mut()
                .iter_mut()
                .for_each(|peak| peak.intensity /= total);
        }
    }
}

struct_with_new!(
    /// Normalizes intensities independently within each fixed-width m/z window.
    ///
    /// The m/z axis is partitioned into non-overlapping `window_size`-Da windows anchored at
    /// 0 (window `k` covers `[k * window_size, (k+1) * window_size)`); peaks in each window
    /// are scaled so the window's most intense peak equals 1.0. A single global [`Normalize`]
    /// would wash out weaker regions of the spectrum under one dominant peak; this keeps each
    /// region's internal shape visible. Peaks must be sorted by m/z (guaranteed by all
    /// standard constructors).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, WindowedNormalize};
    ///
    /// // Window 1 [100,200): weak signal. Window 2 [200,300): strong signal.
    /// // A global Normalize would flatten window 1 to near-zero; this doesn't.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (110.0f32, 10.0), (150.0, 5.0), (250.0, 1000.0),
    /// ]);
    /// WindowedNormalize::new(100.0).operate_single(&mut s);
    ///
    /// assert!((s.peaks()[0].intensity - 1.0).abs() < 1e-6); // 10 / 10, base peak of window 1
    /// assert!((s.peaks()[1].intensity - 0.5).abs() < 1e-6); // 5 / 10
    /// assert!((s.peaks()[2].intensity - 1.0).abs() < 1e-6); // 1000 / 1000, base peak of window 2
    /// ```
    pub struct WindowedNormalize {
        pub window_size: f64,
    }
);

impl SingleOperation for WindowedNormalize {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() || self.window_size <= 0.0 {
            return;
        }

        // Single forward pass to find each window's max (windows are visited in order since
        // peaks are sorted by m/z, so consecutive same-window peaks are contiguous).
        let mut window_max: Vec<(i64, f32)> = Vec::new();
        for peak in spectrum.peaks() {
            let window = (peak.mz / self.window_size).floor() as i64;
            match window_max.last_mut() {
                Some((w, max)) if *w == window => *max = max.max(peak.intensity),
                _ => window_max.push((window, peak.intensity)),
            }
        }

        let mut idx = 0usize;
        for peak in spectrum.peaks_mut() {
            let window = (peak.mz / self.window_size).floor() as i64;
            while window_max[idx].0 != window {
                idx += 1;
            }
            let max = window_max[idx].1;
            if max > 0.0 {
                peak.intensity /= max;
            }
        }
    }
}

struct_with_new!(
    /// Removes peaks matching a neutral loss from the precursor (e.g. loss of water or
    /// ammonia).
    ///
    /// For each mass in `losses` (a neutral, uncharged Da value), the expected fragment m/z is
    /// `precursor_mz - loss / charge` — the fragment is assumed to retain the precursor's
    /// charge state, which holds for the vast majority of in-source neutral losses. Reads
    /// `PEPMASS` from metadata and charge via [`MassSpectralRecord::get_charge`]; spectra
    /// missing `PEPMASS`, or with charge 0, are left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::{MassSpectralRecord, Peak};
    /// use oxidems_lib::workflow::{SingleOperation, NeutralLossFilter, MassTolerance};
    ///
    /// // Precursor at 300.0 Da, charge +1. Loss of water (18.0106 Da) puts a fragment at
    /// // 281.9894 Da.
    /// let peaks = vec![Peak::new(150.0, 400.0), Peak::new(281.9894, 800.0)];
    /// let mut s = MassSpectralRecord::new(peaks, 1, 0);
    /// s.insert_metadata("PEPMASS", "300.0");
    /// NeutralLossFilter::new(vec![18.0106], MassTolerance::Da(0.01)).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 1);
    /// assert!((s.peaks()[0].mz - 150.0).abs() < 1e-9);
    /// ```
    pub struct NeutralLossFilter {
        pub losses: Vec<f64>,
        pub tolerance: MassTolerance,
    }
);

impl SingleOperation for NeutralLossFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if self.losses.is_empty() {
            return;
        }

        let precursor_mz = spectrum
            .get_metadata("PEPMASS")
            .and_then(|mz_str| mz_str.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok());

        let Some(precursor_mz) = precursor_mz else {
            return;
        };

        let charge = spectrum.get_charge();
        if charge == 0 {
            return;
        }
        let charge = charge.unsigned_abs() as f64;

        let target_mzs: Vec<f64> = self
            .losses
            .iter()
            .map(|loss| precursor_mz - loss / charge)
            .collect();

        spectrum.peaks_mut().retain(|peak| {
            !target_mzs
                .iter()
                .any(|&target_mz| self.tolerance.is_within(peak.mz, target_mz))
        });
    }
}

struct_with_new!(
    /// Resamples peaks onto a fixed-width m/z grid, summing intensity within each bin.
    ///
    /// Bin `k` covers `[min_mz + k*bin_width, min_mz + (k+1)*bin_width)`. Peaks sharing a bin
    /// are merged into one, with intensity = the sum of the contributing peaks' intensities
    /// and m/z = their intensity-weighted centroid. Peaks outside `[min_mz, max_mz)` are
    /// dropped. Output peaks are emitted in ascending m/z order, matching the sorted-peaks
    /// invariant the rest of the pipeline relies on.
    ///
    /// This is the same fixed-grid resampling as
    /// [`BinSpectra`](crate::machine_learning::feature_extraction::BinSpectra)'s ML feature
    /// matrix, exposed instead as an in-place spectrum transform — useful earlier in a
    /// pipeline, e.g. to put variable-resolution spectra on a common grid before
    /// [`cosine_similarity`](crate::spectrum::MassSpectralRecord::cosine_similarity).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, BinPeaks};
    ///
    /// // Two peaks land in the same 0.1 Da bin and merge into an intensity-weighted centroid;
    /// // a third peak lands in its own bin further away.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (100.02f32, 10.0), (100.05, 20.0), (101.5, 5.0),
    /// ]);
    /// BinPeaks::new(100.0, 102.0, 0.1).operate_single(&mut s);
    ///
    /// assert_eq!(s.peaks().len(), 2);
    /// // Merged bin: (100.02*10 + 100.05*20) / 30 ≈ 100.04
    /// assert!((s.peaks()[0].mz - 100.04).abs() < 0.01);
    /// assert!((s.peaks()[0].intensity - 30.0).abs() < 1e-4);
    /// assert!((s.peaks()[1].mz - 101.5).abs() < 1e-9);
    /// assert!((s.peaks()[1].intensity - 5.0).abs() < 1e-4);
    /// ```
    pub struct BinPeaks {
        pub min_mz: f64,
        pub max_mz: f64,
        pub bin_width: f64,
    }
);

impl SingleOperation for BinPeaks {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if self.bin_width <= 0.0 || self.max_mz <= self.min_mz {
            spectrum.peaks_mut().clear();
            return;
        }

        let num_bins = ((self.max_mz - self.min_mz) / self.bin_width).ceil() as usize;
        let mut mz_intensity_product = vec![0.0f64; num_bins];
        let mut intensity_sum = vec![0.0f32; num_bins];

        for peak in spectrum.peaks() {
            if peak.mz < self.min_mz || peak.mz >= self.max_mz {
                continue;
            }
            let bin = (((peak.mz - self.min_mz) / self.bin_width) as usize).min(num_bins - 1);
            mz_intensity_product[bin] += peak.mz * peak.intensity as f64;
            intensity_sum[bin] += peak.intensity;
        }

        *spectrum.peaks_mut() = (0..num_bins)
            .filter(|&i| intensity_sum[i] > 0.0)
            .map(|i| Peak {
                mz: mz_intensity_product[i] / intensity_sum[i] as f64,
                intensity: intensity_sum[i],
            })
            .collect();
    }
}

struct_with_new!(
    /// Corrects systematic m/z drift by matching peaks against known reference masses
    /// (lock masses / internal standards) and shifting the whole spectrum by the observed
    /// ppm error.
    ///
    /// For each mass in `reference_mzs`, looks for the nearest peak within `search_window` Da.
    /// Matched pairs contribute a signed ppm error `(observed - expected) / expected * 1e6`;
    /// the spectrum is then scaled by the *mean* of those errors, correcting instrument drift
    /// before downstream m/z-based comparisons (clustering, library matching). Requires at
    /// least `min_references` matches to apply a correction — spectra where the reference
    /// compound wasn't present/fragmented are left unchanged rather than guessing.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, MzRecalibrate};
    ///
    /// // A known reference compound at 200.0 Da is observed 10 ppm high across the board,
    /// // and so is every other peak in the (drift-affected) spectrum.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (200.002f32, 500.0), (350.0035, 300.0), // both shifted by the same +10 ppm
    /// ]);
    /// MzRecalibrate::new(vec![200.0], 0.01, 1).operate_single(&mut s);
    ///
    /// assert!((s.peaks()[0].mz - 200.0).abs() < 1e-4);
    /// assert!((s.peaks()[1].mz - 350.0).abs() < 1e-3);
    /// ```
    pub struct MzRecalibrate {
        pub reference_mzs: Vec<f64>,
        pub search_window: f64,
        pub min_references: usize,
    }
);

impl SingleOperation for MzRecalibrate {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if self.reference_mzs.is_empty() || spectrum.peaks().is_empty() {
            return;
        }

        let ppm_errors: Vec<f64> = self
            .reference_mzs
            .iter()
            .filter_map(|&reference_mz| {
                spectrum
                    .peaks()
                    .iter()
                    .filter(|peak| (peak.mz - reference_mz).abs() <= self.search_window)
                    .min_by(|a, b| {
                        (a.mz - reference_mz)
                            .abs()
                            .total_cmp(&(b.mz - reference_mz).abs())
                    })
                    .map(|nearest| (nearest.mz - reference_mz) / reference_mz * 1_000_000.0)
            })
            .collect();

        if ppm_errors.len() < self.min_references {
            return;
        }

        let mean_ppm_error = ppm_errors.iter().sum::<f64>() / ppm_errors.len() as f64;
        let correction_factor = 1.0 - mean_ppm_error / 1_000_000.0;

        spectrum
            .peaks_mut()
            .iter_mut()
            .for_each(|peak| peak.mz *= correction_factor);
    }
}

struct_with_new!(
    /// Estimates and subtracts a local intensity background, one fixed-width m/z window at a
    /// time.
    ///
    /// Distinct from [`LocalSNRFilter`], which *gates* peaks below a single spectrum-wide noise
    /// floor: this estimates a background level independently in each `window_size`-Da window
    /// (same windowing scheme as [`WindowedNormalize`]/[`BinPeaks`]) and subtracts it from
    /// every peak in that window, floored at 0 — a local correction rather than a global
    /// yes/no gate. The background estimate is the `percentile`-th percentile of intensities
    /// within the window (nearest-rank method). Windows with fewer than 2 peaks are left
    /// untouched — not enough local signal to estimate a background from. Zeroed-out peaks are
    /// not removed; chain with [`FilterIntensity`] or [`AbsoluteIntensityFilter`] to drop them.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, BaselineSubtract};
    ///
    /// // One 100 Da window with 3 peaks: intensities [50, 60, 500]. The 50th percentile
    /// // (nearest-rank on the sorted 3-element list) picks the middle value, 60, as the
    /// // local baseline.
    /// let mut s = MassSpectralRecord::from_tuples(&[
    ///     (110.0f32, 50.0), (120.0, 60.0), (150.0, 500.0),
    /// ]);
    /// BaselineSubtract::new(100.0, 50.0).operate_single(&mut s);
    ///
    /// assert!((s.peaks()[0].intensity - 0.0).abs() < 1e-4);   // 50 - 60, floored at 0
    /// assert!((s.peaks()[1].intensity - 0.0).abs() < 1e-4);   // 60 - 60
    /// assert!((s.peaks()[2].intensity - 440.0).abs() < 1e-4); // 500 - 60
    /// ```
    pub struct BaselineSubtract {
        pub window_size: f64,
        pub percentile: f32,
    }
);

impl SingleOperation for BaselineSubtract {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() || self.window_size <= 0.0 {
            return;
        }

        // Group peak indices into (window, start, end) runs. Windows are contiguous since
        // peaks are sorted by m/z.
        let mut windows: Vec<(i64, usize, usize)> = Vec::new();
        for (idx, peak) in spectrum.peaks().iter().enumerate() {
            let window = (peak.mz / self.window_size).floor() as i64;
            match windows.last_mut() {
                Some((w, _, end)) if *w == window => *end = idx + 1,
                _ => windows.push((window, idx, idx + 1)),
            }
        }

        let mut baselines = vec![0.0f32; spectrum.peaks().len()];
        for (_, start, end) in windows {
            if end - start < 2 {
                continue;
            }
            let mut intensities: Vec<f32> =
                spectrum.peaks()[start..end].iter().map(|p| p.intensity).collect();
            intensities.sort_by(|a, b| a.total_cmp(b));
            let rank = ((self.percentile / 100.0) * (intensities.len() - 1) as f32).round() as usize;
            let baseline = intensities[rank.min(intensities.len() - 1)];
            baselines[start..end].fill(baseline);
        }

        spectrum
            .peaks_mut()
            .iter_mut()
            .zip(baselines)
            .for_each(|(peak, baseline)| peak.intensity = (peak.intensity - baseline).max(0.0));
    }
}

/// Shannon entropy of the intensity-normalized peak distribution: `-Σ p_i * ln(p_i)` where
/// `p_i = intensity_i / Σintensity`. Ranges from 0 (all signal in one peak) up to `ln(n)` for
/// `n` equal-intensity peaks. Shared by [`SpectralEntropyFilter`] and [`EntropyWeightTransform`].
fn shannon_entropy(peaks: &[Peak]) -> f32 {
    let total: f32 = peaks.iter().map(|p| p.intensity).sum();
    if total <= 0.0 {
        return 0.0;
    }
    -peaks
        .iter()
        .map(|peak| {
            let p_i = peak.intensity / total;
            if p_i > 0.0 { p_i * p_i.ln() } else { 0.0 }
        })
        .sum::<f32>()
}

struct_with_new!(
    /// Clears the spectrum if its Shannon entropy falls below `min_entropy`.
    ///
    /// Low entropy means the spectrum's signal is concentrated in a few dominant peaks —
    /// often too little information to identify a compound, or a sign of poor
    /// fragmentation/noise dominance. Follows the entropy-based quality reasoning from
    /// Li et al. 2021 (*Nat. Methods*, "Spectral entropy outperforms MS/MS dot product
    /// similarity for small-molecule compound identification"), who found spectral entropy a
    /// more informative single-number quality/complexity signal than raw peak count.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, SpectralEntropyFilter};
    ///
    /// // One dominant peak (99% of signal) => low entropy, gets cleared.
    /// let mut dominated = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 1.0), (200.0, 990.0), (300.0, 9.0),
    /// ]);
    /// SpectralEntropyFilter::new(1.0).operate_single(&mut dominated);
    /// assert!(dominated.peaks().is_empty());
    ///
    /// // Four roughly equal peaks => high entropy (ln(4) ≈ 1.386), survives.
    /// let mut balanced = MassSpectralRecord::from_tuples(&[
    ///     (100.0f32, 250.0), (200.0, 250.0), (300.0, 250.0), (400.0, 250.0),
    /// ]);
    /// SpectralEntropyFilter::new(1.0).operate_single(&mut balanced);
    /// assert_eq!(balanced.peaks().len(), 4);
    /// ```
    pub struct SpectralEntropyFilter {
        pub min_entropy: f32,
    }
);

impl SingleOperation for SpectralEntropyFilter {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if shannon_entropy(spectrum.peaks()) < self.min_entropy {
            spectrum.peaks_mut().clear();
        }
    }
}

struct_with_new!(
    /// Reweights peak intensities so low-entropy (dominant-peak) spectra get compressed toward
    /// `weight_threshold`'s entropy level, following the entropy-weighting concept from
    /// Li et al. 2021 (*Nat. Methods*): dot-product/cosine similarity performs poorly when a
    /// couple of peaks overwhelm the rest, so weighting them down before comparison helps.
    /// Similar in spirit to [`SquareRootTransformIntensity`], but the compression exponent
    /// adapts to each spectrum's own entropy instead of being a fixed constant.
    ///
    /// If the spectrum's [`shannon_entropy`] `S` is below `weight_threshold` `T`, every
    /// intensity is raised to `weight = 0.25 + 0.75 * (S / T)` — so `weight` ranges from 0.25
    /// at `S = 0` up to 1.0 (no change) at `S = T` — and the result is renormalized back to the
    /// spectrum's original total intensity. Spectra already at or above `T` are left untouched.
    /// `weight_threshold = 3.0` matches the default threshold used in the reference paper; the
    /// weighting formula itself is this crate's own parameterization of that idea for
    /// arbitrary thresholds, not a verbatim reproduction of the paper's constants.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidems_lib::MassSpectralRecord;
    /// use oxidems_lib::workflow::{SingleOperation, EntropyWeightTransform};
    ///
    /// // A dominant peak next to a much weaker one: low entropy, so intensities get
    /// // compressed (raised to a fractional power) — the weak peak's relative share grows.
    /// let mut s = MassSpectralRecord::from_tuples(&[(100.0f32, 10.0), (200.0, 990.0)]);
    /// let total_before: f32 = s.peaks().iter().map(|p| p.intensity).sum();
    /// let weak_share_before = s.peaks()[0].intensity / total_before;
    ///
    /// EntropyWeightTransform::new(3.0).operate_single(&mut s);
    ///
    /// let total_after: f32 = s.peaks().iter().map(|p| p.intensity).sum();
    /// assert!((total_after - total_before).abs() < 1e-2); // total intensity conserved
    ///
    /// let weak_share_after = s.peaks()[0].intensity / total_after;
    /// assert!(weak_share_after > weak_share_before); // compression grows the weak peak's share
    /// ```
    pub struct EntropyWeightTransform {
        pub weight_threshold: f32,
    }
);

impl SingleOperation for EntropyWeightTransform {
    fn operate_single(&self, spectrum: &mut MassSpectralRecord) {
        if spectrum.peaks().is_empty() || self.weight_threshold <= 0.0 {
            return;
        }

        let entropy = shannon_entropy(spectrum.peaks());
        if entropy >= self.weight_threshold {
            return;
        }

        let original_total: f32 = spectrum.peaks().iter().map(|p| p.intensity).sum();
        if original_total <= 0.0 {
            return;
        }

        let weight = 0.25 + 0.75 * (entropy / self.weight_threshold);
        spectrum
            .peaks_mut()
            .iter_mut()
            .for_each(|peak| peak.intensity = peak.intensity.powf(weight));

        let new_total: f32 = spectrum.peaks().iter().map(|p| p.intensity).sum();
        if new_total > 0.0 {
            let rescale = original_total / new_total;
            spectrum
                .peaks_mut()
                .iter_mut()
                .for_each(|peak| peak.intensity *= rescale);
        }
    }
}

#[cfg(test)]
mod new_ops_real_data_tests {
    //! Exercises TICNormalize, WindowedNormalize, NeutralLossFilter, BinPeaks, MzRecalibrate,
    //! BaselineSubtract, and EntropyWeightTransform against
    //! `test_files/test1.mgf` — two real MS3 spectra (Glaucine and Tetramethylkaempferol,
    //! from MassIVE dataset MSV000094528) — rather than synthetic peak lists.

    use super::*;
    use crate::io::mgf::{MgfParser, MgfParserConfig};

    fn load_test1() -> Vec<MassSpectralRecord> {
        let mut config = MgfParserConfig::default();
        config.use_parallel_parsing = false;
        let mut parser = MgfParser::new(config);
        parser.parse_file("test_files/test1.mgf").unwrap()
    }

    #[test]
    fn assign_precursor_charge_leaves_known_charge_alone_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[0].clone(); // Glaucine, CHARGE=1+ already parsed
        assert_eq!(s.get_charge(), 1);

        AssignPrecursorCharge::new(3, 3, 0.01, 0.05, 2, false).operate_single(&mut s);

        // overwrite_existing = false: an already-known charge must not be touched.
        assert_eq!(s.get_charge(), 1);
    }

    #[test]
    fn assign_precursor_charge_is_noop_without_a_surviving_precursor_peak_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[0].clone();
        // PEPMASS=356.18563, but this is an MS3 fragment spectrum: every peak is well below
        // the precursor m/z (max peak 310.12 Da), so no isotope envelope survives to anchor
        // on. Forcing overwrite still can't invent a charge with nothing to detect.
        AssignPrecursorCharge::new(3, 3, 0.01, 0.05, 1, true).operate_single(&mut s);

        assert_eq!(s.get_charge(), 1); // untouched: no peak within precursor_window of PEPMASS
    }

    #[test]
    fn tic_normalize_preserves_ratios_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[1].clone(); // Tetramethylkaempferol, 7 peaks
        let original_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        let original_base_peak = s
            .peaks()
            .iter()
            .map(|p| p.intensity)
            .fold(0.0f32, f32::max);

        TICNormalize::new().operate_single(&mut s);

        let new_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        assert!((new_total - 1.0).abs() < 1e-5);

        let new_base_peak = s.peaks().iter().map(|p| p.intensity).fold(0.0f32, f32::max);
        assert!((new_base_peak - original_base_peak / original_total).abs() < 1e-6);
    }

    #[test]
    fn windowed_normalize_scales_each_window_independently_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[1].clone();
        // Peaks span 254.09-313.07 Da; a 50 Da window splits them into [250,300) and [300,350).
        WindowedNormalize::new(50.0).operate_single(&mut s);

        let base_in_low_window = s
            .peaks()
            .iter()
            .filter(|p| p.mz < 300.0)
            .map(|p| p.intensity)
            .fold(0.0f32, f32::max);
        let base_in_high_window = s
            .peaks()
            .iter()
            .filter(|p| p.mz >= 300.0)
            .map(|p| p.intensity)
            .fold(0.0f32, f32::max);

        // Each window's own base peak hits 1.0, independent of the other window's magnitude.
        assert!((base_in_low_window - 1.0).abs() < 1e-6);
        assert!((base_in_high_window - 1.0).abs() < 1e-6);
    }

    #[test]
    fn neutral_loss_filter_removes_only_the_matching_fragment_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[0].clone(); // Glaucine, PEPMASS=356.18563, charge 1+

        let precursor_mz: f64 = s.get_metadata("PEPMASS").unwrap().parse().unwrap();
        let base_peak_mz = s.peaks().iter().max_by(|a, b| a.intensity.total_cmp(&b.intensity)).unwrap().mz;
        let observed_loss = precursor_mz - base_peak_mz;

        let peaks_before = s.peaks().len();
        NeutralLossFilter::new(vec![observed_loss], MassTolerance::Da(0.01)).operate_single(&mut s);

        // Only the peak at exactly `observed_loss` from the precursor is dropped; the other
        // three real fragment peaks (whose gaps to the precursor differ by more than the
        // 0.01 Da tolerance) survive.
        assert_eq!(s.peaks().len(), peaks_before - 1);
        assert!(s.peaks().iter().all(|p| (p.mz - base_peak_mz).abs() > 1e-9));
    }

    #[test]
    fn bin_peaks_conserves_intensity_and_merges_close_peaks_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[1].clone(); // 7 peaks, 254.09-313.07 Da
        let original_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        let original_count = s.peaks().len();

        // 5 Da bins are wide enough that the three peaks clustered at 309-313 Da collapse
        // into a single bin, while the well-separated 254 and 282 Da peaks stay distinct.
        BinPeaks::new(250.0, 320.0, 5.0).operate_single(&mut s);

        assert!(s.peaks().len() < original_count);
        let new_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        assert!((new_total - original_total).abs() < 1e-3);

        // Peaks stay sorted and each centroid lands inside its own bin.
        for w in s.peaks().windows(2) {
            assert!(w[0].mz < w[1].mz);
        }
    }

    #[test]
    fn mz_recalibrate_removes_a_synthetic_drift_from_real_peak_positions() {
        let spectra = load_test1();
        let mut s = spectra[1].clone(); // 7 real peaks, 254.09-313.07 Da
        let original_mzs: Vec<f64> = s.peaks().iter().map(|p| p.mz).collect();
        let base_peak_mz = s
            .peaks()
            .iter()
            .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
            .unwrap()
            .mz;

        // Simulate +15 ppm instrument drift across every real peak.
        for peak in s.peaks_mut() {
            peak.mz *= 1.0 + 15.0 / 1_000_000.0;
        }

        // Recalibrate using the (drifted) base peak as the only reference, anchored at its
        // known-true real m/z.
        MzRecalibrate::new(vec![base_peak_mz], 0.1, 1).operate_single(&mut s);

        for (peak, &original_mz) in s.peaks().iter().zip(&original_mzs) {
            assert!((peak.mz - original_mz).abs() < 1e-6);
        }
    }

    #[test]
    fn baseline_subtract_never_increases_total_intensity_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[1].clone(); // 7 peaks, several close together at 309-313 Da
        let original_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        let original_count = s.peaks().len();

        BaselineSubtract::new(10.0, 25.0).operate_single(&mut s);

        assert_eq!(s.peaks().len(), original_count); // subtracts, never drops peaks
        let new_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        assert!(new_total <= original_total + 1e-3);
        assert!(s.peaks().iter().all(|p| p.intensity >= 0.0));
    }

    #[test]
    fn entropy_weight_transform_conserves_intensity_on_real_spectrum() {
        let spectra = load_test1();
        let mut s = spectra[0].clone(); // Glaucine: one dominant base peak among 4, low entropy
        let original_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        let entropy_before = shannon_entropy(s.peaks());

        // A generous threshold guarantees this real, dominated spectrum actually gets
        // reweighted rather than left untouched.
        EntropyWeightTransform::new(10.0).operate_single(&mut s);

        let new_total: f32 = s.peaks().iter().map(|p| p.intensity).sum();
        assert!((new_total - original_total).abs() < 1e-2);

        // Compression raises the post-transform spectrum's own entropy toward the threshold.
        let entropy_after = shannon_entropy(s.peaks());
        assert!(entropy_after > entropy_before);
    }
}

#[cfg(test)]
mod deisotope_charge_contamination_regression_test {
    //! Regression test for a bug in the original Deisotope: its charge-state loop never broke
    //! after a successful match, so a single monoisotopic candidate could absorb isotope
    //! satellites from *multiple, mutually incompatible* charge hypotheses at once (chemically
    //! impossible — an ion has one charge). Fixed by `best_isotope_run` picking a single
    //! best-scoring charge state per monoisotopic peak.

    use super::*;

    #[test]
    fn only_the_best_scoring_charge_state_is_merged() {
        // Base peak at 500.0 Da. A real M+1 isotope at the z=1 spacing (501.0034 Da) coexists
        // with a decoy peak that just happens to sit exactly at the z=2 M+1 spacing
        // (500.5017 Da) from the same base peak, but is unrelated (e.g. a genuine fragment
        // ion). max_isotopes = 1 caps the search at a single satellite per charge, so this
        // is a genuine tie (both z=1 and z=2 explain exactly 1 satellite) rather than z=2
        // legitimately winning by also reaching the M+2 isotope-train harmonic that any
        // charge state re-syncs to at z steps (which would make it the objectively better
        // explanation, not a bug). The old code tried z=1 *and* z=2 for the same base peak
        // without ever stopping, so it merged both into one bloated peak regardless. The fix
        // must pick a single charge hypothesis (z=1: found first, ties favor the lower charge)
        // and leave the decoy alone.
        let peaks = vec![
            Peak::new(500.0, 1000.0),
            Peak::new(500.5017, 50.0), // decoy: exact z=2 M+1 spacing, not a real isotope
            Peak::new(501.0034, 300.0), // real M+1 isotope at z=1 spacing
        ];
        let mut s = MassSpectralRecord::new(peaks, 1, 0);

        Deisotope::new(2, 0.005, 1).operate_single(&mut s);

        // Old buggy behavior: 1 peak left, intensity 1000 + 300 + 50 = 1350 (both merged).
        // Fixed behavior: the z=1 hypothesis wins the tie (evaluated first), so only the true
        // isotope is absorbed; the unrelated decoy survives untouched.
        assert_eq!(s.peaks().len(), 2);
        let merged = s.peaks().iter().find(|p| (p.mz - 500.0).abs() < 1e-6).unwrap();
        assert!((merged.intensity - 1300.0).abs() < 1e-3);
        let decoy = s.peaks().iter().find(|p| (p.mz - 500.5017).abs() < 1e-6).unwrap();
        assert!((decoy.intensity - 50.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod deisotope_intensity_inversion_regression_test {
    //! Regression test for a real false positive found on experimental data (steroid EAD
    //! spectrum, Stanozolol at 8 eV): a minor H-loss fragment sitting ~1.007 Da below the
    //! spectrum's actual base peak/precursor is close enough to the C13 isotope spacing
    //! (1.0034 Da) to be misread, at realistic tolerances, as the *monoisotopic* anchor of
    //! an "isotope" whose real M+1 satellite is the base peak itself — 20x more intense than
    //! the supposed monoisotopic peak. `best_isotope_run` now refuses a satellite match whose
    //! intensity exceeds the intensity of the peak before it in the run, so the dominant peak
    //! survives untouched.

    use super::*;

    #[test]
    fn weaker_peak_cannot_absorb_a_much_stronger_one_as_its_isotope() {
        let peaks = vec![
            Peak::new(328.2506, 51.0),  // minor H-loss fragment, NOT a monoisotopic anchor
            Peak::new(329.2572, 1000.0), // real base peak / precursor, 1.0066 Da higher
        ];
        let mut s = MassSpectralRecord::new(peaks, 1, 0);

        Deisotope::new(1, 0.01, 5).operate_single(&mut s);

        // Both peaks must survive: 329.2572 is far too intense to be an isotope of 328.2506.
        assert_eq!(s.peaks().len(), 2);
        let base = s.peaks().iter().find(|p| (p.mz - 329.2572).abs() < 1e-6).unwrap();
        assert!((base.intensity - 1000.0).abs() < 1e-3);
        let minor = s.peaks().iter().find(|p| (p.mz - 328.2506).abs() < 1e-6).unwrap();
        assert!((minor.intensity - 51.0).abs() < 1e-3);
    }

    #[test]
    fn genuine_descending_envelope_still_merges() {
        // A real envelope where intensity legitimately falls off with each isotope must still
        // be collapsed as before.
        let peaks = vec![
            Peak::new(500.0, 1000.0),
            Peak::new(501.0034, 300.0),
            Peak::new(502.0067, 50.0),
        ];
        let mut s = MassSpectralRecord::new(peaks, 1, 0);

        Deisotope::new(1, 0.005, 5).operate_single(&mut s);

        assert_eq!(s.peaks().len(), 1);
        let merged = s.peaks().iter().find(|p| (p.mz - 500.0).abs() < 1e-6).unwrap();
        assert!((merged.intensity - 1350.0).abs() < 1e-3);
    }
}
