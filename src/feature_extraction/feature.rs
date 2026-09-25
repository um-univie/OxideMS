use crate::feature_extraction::deisotoping::DeisotopedFeature;
use crate::spectrum::MassSpectralRecord;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// A fully characterized and quantified LC-MS feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    /// Unique feature identifier within the run.
    pub id: usize,
    /// Monoisotopic m/z (intensity-weighted from M0).
    pub monoisotopic_mz: f64,
    /// Exact apex m/z.
    pub apex_mz: f64,
    /// Assigned charge state.
    pub charge: i8,
    /// Retention time at apex (seconds).
    pub apex_rt: f32,
    /// Peak start retention time (seconds).
    pub rt_start: f32,
    /// Peak end retention time (seconds).
    pub rt_end: f32,
    /// Integrated area of monoisotopic peak (M0).
    pub monoisotopic_area: f32,
    /// Total integrated area across all isotope peaks.
    pub total_area: f32,
    /// Peak apex intensity.
    pub apex_intensity: f32,
    /// Full Width at Half Maximum (seconds).
    pub fwhm: f32,
    /// Estimated Signal-to-Noise Ratio.
    pub snr: f32,
    /// Number of detected isotopic satellite peaks (including M0).
    pub isotope_count: usize,
    /// Linked MS2 spectrum IDs matching this precursor within the elution window.
    pub linked_ms2_ids: Vec<usize>,
}

impl From<(usize, DeisotopedFeature)> for Feature {
    fn from((id, feat): (usize, DeisotopedFeature)) -> Self {
        let apex_intensity = feat
            .isotopes
            .first()
            .map(|i| i.apex_intensity)
            .unwrap_or(0.0);

        Self {
            id,
            monoisotopic_mz: feat.monoisotopic_mz,
            apex_mz: feat.apex_mz,
            charge: feat.charge,
            apex_rt: feat.apex_rt,
            rt_start: feat.rt_start,
            rt_end: feat.rt_end,
            monoisotopic_area: feat.monoisotopic_area,
            total_area: feat.total_area,
            apex_intensity,
            fwhm: feat.fwhm,
            snr: feat.snr,
            isotope_count: feat.isotopes.len(),
            linked_ms2_ids: Vec::new(),
        }
    }
}

/// Selected-ion assignment; ambiguity is not silently resolved by intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ms2FeatureMatch {
    pub scan_id: usize,
    pub feature_id: usize,
    pub isotope_offset: usize,
    pub mass_error_ppm: f64,
    pub ambiguous: bool,
}

/// A collection of quantified features across an LC-MS run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureTable {
    pub features: Vec<Feature>,
    pub total_ms1_scans: usize,
    pub total_ms2_scans: usize,
    pub source_file: Option<String>,
}

impl FeatureTable {
    pub fn new(features: Vec<Feature>) -> Self {
        Self {
            features,
            total_ms1_scans: 0,
            total_ms2_scans: 0,
            source_file: None,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.features.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Links MS2 fragmentation records to their parent MS1 features based on precursor m/z
    /// and chromatographic elution bounds.
    pub fn link_ms2_records(
        &mut self,
        records: &[MassSpectralRecord],
        isolation_tolerance_da: f64,
    ) {
        self.link_ms2_records_with_tolerance(records, crate::workflow::MassTolerance::Da(isolation_tolerance_da));
    }

    /// Match selected ions using mass accuracy, not the isolation-window width.
    /// Returns all plausible assignments; multiple matches remain explicitly ambiguous.
    pub fn link_ms2_records_with_tolerance(
        &mut self,
        records: &[MassSpectralRecord],
        tolerance: crate::workflow::MassTolerance,
    ) -> Vec<Ms2FeatureMatch> {
        let precursors: Vec<_> = records.iter().filter(|r| r.ms_level == 2).filter_map(|r| {
            let mz = r.get_metadata("PEPMASS")?.split_whitespace().next()?.parse().ok()?;
            let rt = r.get_metadata("RTINSECONDS")?.parse().ok()?;
            Some(crate::feature_extraction::raw_data::Ms2PrecursorInfo {
                scan_id: r.id, precursor_mz: mz, rt, charge: r.get_charge(),
            })
        }).collect();
        let matches = self.link_ms2_precursors_with_tolerance(&precursors, tolerance);
        self.total_ms2_scans = records.iter().filter(|r| r.ms_level == 2).count();
        matches
    }

    /// Compatibility entry point with an absolute selected-ion tolerance.
    pub fn link_ms2_precursors(
        &mut self,
        precursors: &[crate::feature_extraction::raw_data::Ms2PrecursorInfo],
        tolerance_da: f64,
    ) {
        self.link_ms2_precursors_with_tolerance(precursors, crate::workflow::MassTolerance::Da(tolerance_da));
    }

    /// Indexed matching to monoisotopic ions and observed isotope offsets.
    /// Feature charges currently encode magnitude; polarity stays in scan metadata.
    pub fn link_ms2_precursors_with_tolerance(
        &mut self,
        precursors: &[crate::feature_extraction::raw_data::Ms2PrecursorInfo],
        tolerance: crate::workflow::MassTolerance,
    ) -> Vec<Ms2FeatureMatch> {
        self.total_ms2_scans = precursors.len();
        let mut index = Vec::new();
        for (i, f) in self.features.iter_mut().enumerate() {
            f.linked_ms2_ids.clear();
            if !f.monoisotopic_mz.is_finite() { continue; }
            for isotope in 0..f.isotope_count.max(1) {
                let mz = f.monoisotopic_mz + isotope as f64 * 1.003355 / f.charge.unsigned_abs().max(1) as f64;
                index.push((mz, i, isotope));
            }
        }
        index.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut matches = Vec::new();
        for ms2 in precursors {
            let mz = ms2.precursor_mz;
            let tol = tolerance.tolerance_at(mz);
            if !mz.is_finite() || mz <= 0.0 || !ms2.rt.is_finite() || !tol.is_finite() || tol < 0.0 { continue; }
            let start = index.partition_point(|entry| entry.0 < mz - tol);
            let end = index.partition_point(|entry| entry.0 <= mz + tol);
            let first = matches.len();
            let mut seen = std::collections::HashSet::new();
            for &(target, i, isotope) in &index[start..end] {
                let feat = &mut self.features[i];
                if ms2.charge != 0 && feat.charge != 0 && ms2.charge.unsigned_abs() != feat.charge.unsigned_abs() { continue; }
                if ms2.rt < feat.rt_start - 1.0 || ms2.rt > feat.rt_end + 1.0 || !seen.insert(i) { continue; }
                feat.linked_ms2_ids.push(ms2.scan_id);
                matches.push(Ms2FeatureMatch { scan_id: ms2.scan_id, feature_id: feat.id,
                    isotope_offset: isotope, mass_error_ppm: (mz - target) / target * 1e6,
                    ambiguous: false });
            }
            let ambiguous = matches.len() - first > 1;
            for m in &mut matches[first..] { m.ambiguous = ambiguous; }
        }
        matches
    }

    /// Writes feature table to standard CSV format.
    pub fn to_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(
            writer,
            "id,mz,rt,rt_start,rt_end,charge,area,monoisotopic_area,apex_intensity,fwhm,snr,isotopes,ms2_count,ms2_ids"
        )?;

        for f in &self.features {
            let ms2_ids_str = f
                .linked_ms2_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(";");

            writeln!(
                writer,
                "{},{:.5},{:.2},{:.2},{:.2},{},{:.1},{:.1},{:.1},{:.2},{:.1},{},{},\"{}\"",
                f.id,
                f.monoisotopic_mz,
                f.apex_rt,
                f.rt_start,
                f.rt_end,
                f.charge,
                f.total_area,
                f.monoisotopic_area,
                f.apex_intensity,
                f.fwhm,
                f.snr,
                f.isotope_count,
                f.linked_ms2_ids.len(),
                ms2_ids_str
            )?;
        }

        Ok(())
    }

    /// Saves feature table to a CSV file.
    pub fn write_to_file(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.to_csv(&mut writer)
    }

    /// Prints a concise summary of the feature finding results.
    pub fn print_summary(&self) {
        println!("=========================================================================");
        println!("                     OXIDEMS FEATURE TABLE SUMMARY                       ");
        println!("=========================================================================");
        println!("Total Features Found:   {}", self.len());
        println!("Total MS1 Scans:        {}", self.total_ms1_scans);
        println!("Total MS2 Scans:        {}", self.total_ms2_scans);

        let linked_count = self.features.iter().filter(|f| !f.linked_ms2_ids.is_empty()).count();
        let link_rate = if !self.features.is_empty() {
            (linked_count as f64 / self.features.len() as f64) * 100.0
        } else {
            0.0
        };
        println!("Features with MS2:      {} ({:.1}%)", linked_count, link_rate);

        let mut charges = [0usize; 7];
        for f in &self.features {
            let c = f.charge.clamp(1, 6) as usize;
            charges[c] += 1;
        }
        println!(
            "Charge Distribution:    z=1: {}, z=2: {}, z=3: {}, z=4+: {}",
            charges[1], charges[2], charges[3], charges[4] + charges[5] + charges[6]
        );

        if let Some(first) = self.features.first() {
            let min_rt = self.features.iter().map(|f| f.apex_rt).fold(f32::INFINITY, f32::min);
            let max_rt = self.features.iter().map(|f| f.apex_rt).fold(f32::NEG_INFINITY, f32::max);
            println!("Retention Time Range:   {:.1} s - {:.1} s ({:.1} min)", min_rt, max_rt, (max_rt - min_rt) / 60.0);
        }
        println!("=========================================================================");
    }
}
