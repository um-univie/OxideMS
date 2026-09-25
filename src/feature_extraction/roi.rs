use crate::feature_extraction::config::FeatureFinderConfig;
use crate::feature_extraction::raw_data::Ms1Run;
use rayon::prelude::*;

/// A continuous chromatographic mass trace (Extracted Ion Chromatogram slice)
/// extracted across retention time for a specific m/z envelope.
#[derive(Debug, Clone)]
pub struct MassTrace {
    /// Intensity-weighted centroid m/z across all points in the trace.
    pub mean_mz: f64,
    /// Minimum m/z observed in this trace.
    pub min_mz: f64,
    /// Maximum m/z observed in this trace.
    pub max_mz: f64,
    /// Scan indices corresponding to detected points.
    pub scan_indices: Vec<u32>,
    /// Retention times (in seconds) for each detected point.
    pub rts: Vec<f32>,
    /// Measured m/z for each detected point.
    pub mzs: Vec<f64>,
    /// Intensity for each detected point.
    pub intensities: Vec<f32>,
    /// Maximum intensity along this trace.
    pub apex_intensity: f32,
    /// Retention time corresponding to apex intensity.
    pub apex_rt: f32,
}

impl MassTrace {
    pub fn new(scan_idx: u32, rt: f32, mz: f64, intensity: f32) -> Self {
        Self {
            mean_mz: mz,
            min_mz: mz,
            max_mz: mz,
            scan_indices: vec![scan_idx],
            rts: vec![rt],
            mzs: vec![mz],
            intensities: vec![intensity],
            apex_intensity: intensity,
            apex_rt: rt,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rts.is_empty()
    }

    /// Retention time span `(rt_start, rt_end)`.
    pub fn rt_span(&self) -> (f32, f32) {
        if self.rts.is_empty() {
            (0.0, 0.0)
        } else {
            (*self.rts.first().unwrap(), *self.rts.last().unwrap())
        }
    }
}

/// Internal active trace state during the streaming sweep.
struct ActiveTrace {
    trace: MassTrace,
    last_seen_scan: usize,
    last_seen_rt: f32,
    sum_mz_intensity: f64,
    sum_intensity: f64,
    updated_this_scan: bool,
}

/// High-performance parallel mass trace extractor.
pub struct MassTraceExtractor<'a> {
    config: &'a FeatureFinderConfig,
}

impl<'a> MassTraceExtractor<'a> {
    pub fn new(config: &'a FeatureFinderConfig) -> Self {
        Self { config }
    }

    /// Extracts mass traces in parallel across the entire MS1 run.
    pub fn extract_traces(&self, run: &Ms1Run) -> Vec<MassTrace> {
        if run.num_scans() == 0 || run.total_peaks() == 0 {
            return Vec::new();
        }
        let (min_mz, max_mz) = run.mz_range();

        let partition_size = self.config.mz_partition_size;
        let overlap = self.config.mz_partition_overlap;

        // Generate mass intervals aligned to partition grid
        let mut partitions = Vec::new();
        let start_grid = (min_mz / partition_size).floor() * partition_size;
        let mut current_mz = start_grid;
        while current_mz <= max_mz {
            let p_start = current_mz;
            let p_end = current_mz + partition_size;
            // Overlapped query window
            let win_start = (p_start - overlap).max(0.0);
            let win_end = p_end + overlap;
            partitions.push((p_start, p_end, win_start, win_end));
            current_mz += partition_size;
        }

        // Parallel processing of mass partitions with Rayon
        let all_traces: Vec<Vec<MassTrace>> = partitions
            .par_iter()
            .map(|&(p_start, p_end, win_start, win_end)| {
                self.extract_traces_in_window(run, p_start, p_end, win_start, win_end)
            })
            .collect();

        // Flatten results
        let total_count: usize = all_traces.iter().map(|t| t.len()).sum();
        let mut result = Vec::with_capacity(total_count);
        for traces in all_traces {
            result.extend(traces);
        }

        // Sort by retention time apex, then mean m/z
        result.sort_by(|a, b| {
            a.apex_rt
                .total_cmp(&b.apex_rt)
                .then_with(|| a.mean_mz.total_cmp(&b.mean_mz))
        });

        result
    }

    /// Extracts traces for a single mass window, keeping only traces whose
    /// `mean_mz` falls into `[p_start, p_end)` to prevent duplicates.
    fn extract_traces_in_window(
        &self,
        run: &Ms1Run,
        p_start: f64,
        p_end: f64,
        win_start: f64,
        win_end: f64,
    ) -> Vec<MassTrace> {
        let mut active: Vec<ActiveTrace> = Vec::new();
        let mut completed: Vec<MassTrace> = Vec::new();
        let num_scans = run.num_scans();

        for scan_idx in 0..num_scans {
            let rt = run.scan_rt(scan_idx);
            let (mzs, ints) = run.scan_peaks(scan_idx);

            // Find slice of peaks within [win_start, win_end] using binary search
            let start_idx = match mzs.binary_search_by(|m| m.total_cmp(&win_start)) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
            let end_idx = match mzs.binary_search_by(|m| m.total_cmp(&win_end)) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            };

            let win_mzs = if start_idx < mzs.len() {
                &mzs[start_idx..end_idx.min(mzs.len())]
            } else {
                &[]
            };
            let win_ints = if start_idx < ints.len() {
                &ints[start_idx..end_idx.min(ints.len())]
            } else {
                &[]
            };

            for act in &mut active {
                act.updated_this_scan = false;
            }

            // Match each peak to existing active traces
            for (&mz, &intensity) in win_mzs.iter().zip(win_ints) {
                if intensity < self.config.min_peak_height * 0.3 {
                    continue;
                }

                // Find closest active trace within mass tolerance
                let mut best_idx = None;
                let mut min_diff = f64::INFINITY;

                for (idx, act) in active.iter().enumerate() {
                    if act.updated_this_scan {
                        continue;
                    }
                    if self.config.mass_tolerance.is_within(act.trace.mean_mz, mz) {
                        let diff = (act.trace.mean_mz - mz).abs();
                        if diff < min_diff {
                            min_diff = diff;
                            best_idx = Some(idx);
                        }
                    }
                }

                if let Some(idx) = best_idx {
                    let act = &mut active[idx];
                    act.trace.scan_indices.push(scan_idx as u32);
                    act.trace.rts.push(rt);
                    act.trace.mzs.push(mz);
                    act.trace.intensities.push(intensity);
                    if mz < act.trace.min_mz {
                        act.trace.min_mz = mz;
                    }
                    if mz > act.trace.max_mz {
                        act.trace.max_mz = mz;
                    }
                    if intensity > act.trace.apex_intensity {
                        act.trace.apex_intensity = intensity;
                        act.trace.apex_rt = rt;
                    }

                    act.sum_mz_intensity += mz * intensity as f64;
                    act.sum_intensity += intensity as f64;
                    act.trace.mean_mz = act.sum_mz_intensity / act.sum_intensity;
                    act.last_seen_scan = scan_idx;
                    act.last_seen_rt = rt;
                    act.updated_this_scan = true;
                } else {
                    // Create new active trace
                    let mut trace = MassTrace::new(scan_idx as u32, rt, mz, intensity);
                    active.push(ActiveTrace {
                        trace,
                        last_seen_scan: scan_idx,
                        last_seen_rt: rt,
                        sum_mz_intensity: mz * intensity as f64,
                        sum_intensity: intensity as f64,
                        updated_this_scan: true,
                    });
                }
            }

            // Evict expired traces
            let max_missing = self.config.max_missing_scans;
            let max_rt_gap = self.config.max_rt_gap_seconds;

            let mut i = 0;
            while i < active.len() {
                let expired = (scan_idx - active[i].last_seen_scan > max_missing)
                    || ((rt - active[i].last_seen_rt) > max_rt_gap);

                if expired {
                    let act = active.swap_remove(i);
                    // Filter by partition ownership, length, and apex intensity
                    if act.trace.mean_mz >= p_start
                        && act.trace.mean_mz < p_end
                        && act.trace.len() >= self.config.min_trace_length
                        && act.trace.apex_intensity >= self.config.min_peak_height
                    {
                        completed.push(act.trace);
                    }
                } else {
                    i += 1;
                }
            }
        }

        // Final eviction at EOF
        for act in active {
            if act.trace.mean_mz >= p_start
                && act.trace.mean_mz < p_end
                && act.trace.len() >= self.config.min_trace_length
                && act.trace.apex_intensity >= self.config.min_peak_height
            {
                completed.push(act.trace);
            }
        }

        completed
    }
}
