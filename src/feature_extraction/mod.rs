pub mod config;
pub mod cwt;
pub mod deisotoping;
pub mod feature;
pub mod raw_data;
pub mod roi;

pub use config::{FeatureFinderConfig, IsotopeModel};
pub use cwt::{ChromatographicPeak, CwtPeakResolver};
pub use deisotoping::{DeisotopedFeature, FeatureDeisotoper, IsotopePeak};
pub use feature::{Feature, FeatureTable, Ms2FeatureMatch};
pub use raw_data::Ms1Run;
pub use roi::{MassTrace, MassTraceExtractor};

use crate::io::mzml::{MzmlParser, MzmlParserConfig};
use crate::spectrum::MassSpectralRecord;
use rayon::prelude::*;
use std::fs::File;
use std::io;
use std::time::Instant;

/// End-to-end LC-MS feature finding engine.
pub struct FeatureFinder {
    pub config: FeatureFinderConfig,
}

impl Default for FeatureFinder {
    fn default() -> Self {
        Self::new(FeatureFinderConfig::default())
    }
}

impl FeatureFinder {
    pub fn new(config: FeatureFinderConfig) -> Self {
        Self { config }
    }

    /// Finds features directly from a pre-loaded `Ms1Run`.
    pub fn find_features_from_run(&self, run: &Ms1Run) -> FeatureTable {
        let t0 = Instant::now();

        // 1. Parallel Mass Trace / ROI Extraction
        let trace_extractor = MassTraceExtractor::new(&self.config);
        let traces = trace_extractor.extract_traces(run);

        // 2. Parallel CWT Chromatographic Peak Resolving
        let resolver = CwtPeakResolver::new(&self.config);
        let resolved_peaks: Vec<ChromatographicPeak> = traces
            .par_iter()
            .flat_map(|trace| resolver.resolve_peaks(trace))
            .collect();

        // 3. Deisotoping and Charge State Deconvolution
        let deisotoper = FeatureDeisotoper::new(&self.config);
        let deisotoped_features = deisotoper.deisotope(&resolved_peaks);

        // 4. Assemble Feature Table
        let features: Vec<Feature> = deisotoped_features
            .into_iter()
            .enumerate()
            .map(Feature::from)
            .collect();

        let mut table = FeatureTable::new(features);
        table.total_ms1_scans = run.num_scans();

        table
    }

    /// Finds features from a slice of `MassSpectralRecord`s (MS1 + MS2),
    /// automatically linking MS2 precursor scans to the detected MS1 features.
    pub fn find_features_from_records(&self, records: &[MassSpectralRecord]) -> FeatureTable {
        let run = Ms1Run::from_records(records);
        let mut table = self.find_features_from_run(&run);

        // Assign selected ions at instrument mass accuracy.
        table.link_ms2_records_with_tolerance(records, self.config.mass_tolerance);
        table
    }

    /// Runs the complete feature finding pipeline on an mzML file using selective high-performance ingestion.
    pub fn find_features_from_file(&self, path: &str) -> Result<FeatureTable, io::Error> {
        let (run, ms2_info) = Ms1Run::from_mzml_file_direct(path)?;
        let mut table = self.find_features_from_run(&run);
        table.link_ms2_precursors_with_tolerance(&ms2_info, self.config.mass_tolerance);
        table.source_file = Some(path.to_string());
        Ok(table)
    }
}
