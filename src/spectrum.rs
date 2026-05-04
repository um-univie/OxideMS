use crate::{impl_fluent_builder_vanilla, workflow::MassTolerance};
use chemistry_consts::MONOISOTOPIC_MASSES;
use compact_str::CompactString;
use rkyv::{Archive, Deserialize, Serialize};
use similarity::{cosine_similarity, euclidean_distance};
use std::hash::{DefaultHasher, Hash, Hasher};
use xxhash_rust::xxh32::xxh32;

const HYDROGEN_MASS: f64 = MONOISOTOPIC_MASSES[1];

use std::ops::Deref;

#[derive(Archive, Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct Peak {
    pub mz: f64,
    pub intensity: f32,
}

impl Peak {
    pub fn new(mz: f64, intensity: f32) -> Self {
        Peak { mz, intensity }
    }
}

/// Represents a mass spectrum as a collection of peaks
/// The peaks are expected to be sorted! If the struct is created without the provided methods this responsibility falls on the user!
#[derive(Debug, Clone)]
pub struct MassSpectralRecord {
    pub metadata: Vec<(CompactString, CompactString)>,
    spectrum: MassSpectrum,
    pub id: usize,
    /// Byte offset in the source file where this spectrum's block starts.
    /// Only meaningful when parsed from a file; zero otherwise.
    pub source_byte_start: u64,
    /// Byte offset in the source file where this spectrum's block ends.
    /// Only meaningful when parsed from a file; zero otherwise.
    pub source_byte_end: u64,
}

/// Represents a mass spectrum as a collection of peaks
/// The peaks are expected to be sorted! If the struct is created without the provided methods this responsibility falls on the user!
#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct MassSpectrum {
    pub peaks: Vec<Peak>,
    pub charge: i8,
}

pub trait ToF32 {
    fn to_f32(self) -> f32;
}

// Implement it for the types you care about
impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl ToF32 for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

pub trait ToF64 {
    fn to_f64(self) -> f64;
}

// Implement it for the types you care about
impl ToF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl ToF64 for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MassSpectrum {
    pub fn peaks(&self) -> &[Peak] {
        self.peaks.as_slice()
    }

    pub fn from_tuples<T: ToF32 + ToF64 + Copy>(peaks: &[(T, T)]) -> Self {
        let mut peaks = peaks
            .iter()
            .map(|&(mz, intensity)| Peak {
                mz: mz.to_f64(),
                intensity: intensity.to_f32(),
            })
            .collect::<Vec<Peak>>();

        peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        MassSpectrum { peaks, charge: 0 }
    }

    pub fn to_hashed(&self, config: &HashedSpectrumConfig) -> HashedSpectrum {
        let mut features = vec![0.0; config.hash_size];

        for peak in self.peaks() {
            if peak.intensity >= config.intensity_threshold {
                let bin_index = (peak.mz / config.bin_width).round() as isize;
                let bytes = bin_index.to_le_bytes();
                let hash_val = xxh32(&bytes, 0) as usize;
                let index = (hash_val % config.hash_size) as usize;

                // Use highest bit for sign to prevent bias/accumulation from hash collisions
                let sign = if (hash_val as u64 >> usize::BITS - 1) == 1 {
                    -1.0
                } else {
                    1.0
                };

                features[index] += peak.intensity * sign;
            }
        }

        HashedSpectrum { features }
    }

    pub fn hashed_cosine_similarity(
        &self,
        other: &Self,
        config: &HashedSpectrumConfig,
    ) -> Option<f64> {
        self.to_hashed(config)
            .cosine_similarity(&other.to_hashed(config))
    }
}

trait SpectralHash {
    fn hash() {}
    fn hash_into_buffer(&self, config: &HashedSpectrumConfig, buffer: &mut [f32]) {
        if buffer.len() != config.hash_size {
            // TODO add error handling here
            panic!()
        }

        for peak in self.peaks() {
            if peak.intensity >= config.intensity_threshold {
                let bin_index = (peak.mz / config.bin_width).round() as isize;
                let bytes = bin_index.to_le_bytes();
                let hash_val = xxh32(&bytes, 0) as u64;
                let index = (hash_val % config.hash_size as u64) as usize;

                // Use highest bit for sign to prevent bias/accumulation from hash collisions
                let sign = if (hash_val >> 63) == 1 { -1.0 } else { 1.0 };

                buffer[index] += peak.intensity * sign;
            }
        }
    }
    fn peaks(&self) -> &[Peak];
}

impl ArchivedMassSpectrum {
    pub fn hash_into_buffer(&self, config: &HashedSpectrumConfig, buffer: &mut [f32]) {
        if buffer.len() != config.hash_size {
            // TODO add error handling here
            panic!()
        }

        for peak in self.peaks.iter() {
            let bin_index = (peak.mz / config.bin_width).round() as isize;
            let bytes = bin_index.to_le_bytes();
            let hash_val = xxh32(&bytes, 0) as usize;
            let index = (hash_val % config.hash_size) as usize;

            // Use highest bit for sign to prevent bias/accumulation from hash collisions
            let sign = if (hash_val >> usize::BITS - 1) == 1 {
                -1.0
            } else {
                1.0
            };

            buffer[index] += peak.intensity * sign;
        }
    }
}

impl SpectralHash for MassSpectrum {
    fn peaks(&self) -> &[Peak] {
        &self.peaks
    }
}

impl SpectralHash for MSNSpectrum {
    fn peaks(&self) -> &[Peak] {
        &self.spectrum.peaks
    }
}

/// Represents a mass spectrum as a collection of peaks
#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct MSNSpectrum {
    pub precursor_mz: f64,
    pub retention_time_in_seconds: f32,
    pub id: u64,
    pub spectrum: MassSpectrum,
}

impl MSNSpectrum {
    pub fn hash_into_buffer(&self, config: &HashedSpectrumConfig, buffer: &mut [f32]) {
        self.spectrum.hash_into_buffer(config, buffer);
    }
    pub fn charge(&self) -> i8 {
        self.spectrum.charge
    }
    pub fn get_neutral_mass(&self) -> f64 {
        let z = self.charge() as f64;

        (self.precursor_mz * z.abs().max(1.0)) - (HYDROGEN_MASS * z)
    }
}

#[derive(Debug, Clone)]
pub struct HashedSpectrum {
    /// Fixed-length vector of hashed intensities, downcast for speed
    pub features: Vec<f32>,
}

impl HashedSpectrum {
    pub fn cosine_similarity(&self, other: &Self) -> Option<f64> {
        cosine_similarity(self, other)
    }
}

#[derive(Debug, Clone)]
pub struct MSNSpectralRecord {
    pub precursor_mz: f32,
    pub spectrum: MassSpectralRecord,
}

#[derive(Debug, Clone)]
pub struct HashedMSNSpectrum {
    pub spectrum: HashedSpectrum,
    pub precursor_mz: f32,
}

impl Deref for HashedSpectrum {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.features
    }
}

/// Represents a binned/digitized spectrum for fast comparison
#[derive(Debug, Clone)]
pub struct BinnedSpectrum {
    /// Binned intensities where index represents m/z bin
    intensities: Vec<f32>,
    /// Actual bin centers (weighted average m/z of peaks in each bin)
    bin_centers: Vec<f64>,
    /// Bin width in m/z units
    bin_width: f64,
    /// Starting m/z value (left edge of range)
    min_mz: f64,
    /// Maximum m/z value (right edge of range)
    max_mz: f64,
}

/// Configuration for spectrum preprocessing
#[derive(Debug, Clone)]
pub struct BinnedSpectrumConfig {
    /// Bin width for digitization (smaller = more precision, larger = faster)
    pub bin_width: f64,
    /// Intensity threshold for noise filtering
    pub intensity_threshold: f32,
    /// Maximum number of peaks to keep (0 = keep all)
    pub max_peaks: usize,
    /// Mass tolerance for peak matching
    pub mass_tolerance: f64,
    /// Bin offset - bins are centered at offset + n * bin_width (default: bin_width/2 for centering)
    pub bin_offset: Option<f64>,
}

// In practice, this looks extremely clean:
// let config = default_config.hash_size(2048).intensity_threshold(0.5);

impl Default for BinnedSpectrumConfig {
    fn default() -> Self {
        Self {
            bin_width: 1.0,            // 1 Da bins for speed
            intensity_threshold: 0.01, // 1% threshold
            max_peaks: 0,              // Keep all peaks
            mass_tolerance: 0.1,       // 0.1 Da tolerance
            bin_offset: None,          // Auto-center bins at integer values (0.0)
        }
    }
}

impl BinnedSpectrumConfig {
    /// Create config optimized for speed (larger bins, more filtering)
    pub fn fast() -> Self {
        Self {
            bin_width: 2.0,
            intensity_threshold: 0.05,
            max_peaks: 100,
            mass_tolerance: 0.5,
            bin_offset: None, // Auto-center
        }
    }

    /// Create config optimized for precision but still practical for similarity
    /// Note: For cosine similarity, bins should be wide enough to account for small m/z shifts
    pub fn precise() -> Self {
        Self {
            bin_width: 0.5, // Changed from 0.1 to 0.5 for better peak alignment
            intensity_threshold: 0.001,
            max_peaks: 0,
            mass_tolerance: 0.01,
            bin_offset: None, // Auto-center
        }
    }

    /// Create config for ultra-high precision (but may have alignment issues)
    pub fn ultra_precise() -> Self {
        Self {
            bin_width: 0.1,
            intensity_threshold: 0.001,
            max_peaks: 0,
            mass_tolerance: 0.01,
            bin_offset: None, // Auto-center
        }
    }

    /// Set custom bin offset
    pub fn with_bin_offset(mut self, offset: f64) -> Self {
        self.bin_offset = Some(offset);
        self
    }
}

pub struct SpectrumHasher {
    pub min_mz: f64,
    pub max_mz: f64,
    pub tolerance: MassTolerance,
    pub hash_len: usize,
    pub lookup_table: Vec<(usize, f32)>,

    // Pre-computed fields to avoid doing logs/divisions inside the inner loop
    bin_denominator: f64,
    min_bin: f64,
}

impl SpectrumHasher {
    /// Helper to compute the absolute mathematical bin for a given m/z
    fn calculate_abs_bin(mz: f64, tolerance: &MassTolerance, denominator: f64) -> f64 {
        match tolerance {
            MassTolerance::Da(_) => mz / denominator,
            MassTolerance::Ppm(_) => mz.ln() / denominator,
        }
    }

    pub fn new(min_mz: f64, max_mz: f64, tolerance: MassTolerance, hash_len: usize) -> Self {
        // Pre-compute the denominator. For Da it's just the size; for PPM it's the log base.
        let bin_denominator = match tolerance {
            MassTolerance::Da(da) => da,
            MassTolerance::Ppm(ppm) => (1.0 + ppm / 1_000_000.0).ln(),
        };

        // Determine the absolute bin boundaries to calculate how many buckets we need
        let min_bin = Self::calculate_abs_bin(min_mz, &tolerance, bin_denominator);
        let max_bin = Self::calculate_abs_bin(max_mz, &tolerance, bin_denominator);

        let num_bins = (max_bin - min_bin).ceil() as usize;
        let mut lookup_table = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            let bytes = i.to_le_bytes();
            let hash_val = xxh32(&bytes, 0);

            // Use the hash to determine the bucket
            let target_idx = (hash_val as u64 % hash_len as u64) as usize;

            // Signed hashing trick: Use a bit from the hash to decide polarity
            let sign = if (hash_val & 0x8000_0000) == 0 {
                1.0
            } else {
                -1.0
            };

            lookup_table.push((target_idx, sign));
        }

        Self {
            min_mz,
            max_mz,
            tolerance,
            hash_len,
            lookup_table,
            bin_denominator,
            min_bin,
        }
    }

    pub fn recalculate_lookup_table(&mut self) {
        // Recalculate bin denominator and min_bin just in case bounds/tolerance changed
        self.bin_denominator = match self.tolerance {
            MassTolerance::Da(da) => da,
            MassTolerance::Ppm(ppm) => (1.0 + ppm / 1_000_000.0).ln(),
        };

        self.min_bin = Self::calculate_abs_bin(self.min_mz, &self.tolerance, self.bin_denominator);
        let max_bin = Self::calculate_abs_bin(self.max_mz, &self.tolerance, self.bin_denominator);

        let num_bins = (max_bin - self.min_bin).ceil() as usize;

        // Resize and fill
        self.lookup_table.resize(num_bins, (0, 1.0));

        // Precompute the hash index and sign for every bin
        for i in 0..num_bins {
            let bytes = i.to_le_bytes();
            let hash_val = xxh32(&bytes, 0);

            let target_idx = (hash_val as u64 % self.hash_len as u64) as usize;

            let sign = if (hash_val & 0x8000_0000) == 0 {
                1.0
            } else {
                -1.0
            };

            self.lookup_table[i] = (target_idx, sign);
        }
    }

    pub fn into_vector_native(&self, peaks: &[Peak], normalize_l2: bool, buffer: &mut [f32]) {
        let is_ppm = matches!(self.tolerance, MassTolerance::Ppm(_));

        for peak in peaks {
            let mz = peak.mz;
            let abs_bin = if is_ppm {
                mz.ln() / self.bin_denominator
            } else {
                mz / self.bin_denominator
            };
            let bin_idx = ((abs_bin - self.min_bin).floor() as usize)
                .min(self.lookup_table.len().saturating_sub(1));
            let (target_idx, sign) = self.lookup_table[bin_idx];
            buffer[target_idx] += peak.intensity * sign;
        }

        if normalize_l2 {
            let sq_sum: f32 = buffer.iter().map(|&x| x * x).sum();
            if sq_sum > 0.0 {
                let inv_norm = 1.0 / sq_sum.sqrt();
                for val in buffer.iter_mut() {
                    *val *= inv_norm;
                }
            }
        }
    }

    pub fn into_vector(&self, peaks: &[ArchivedPeak], normalize_l2: bool, buffer: &mut [f32]) {
        let is_ppm = matches!(self.tolerance, MassTolerance::Ppm(_));

        for peak in peaks {
            let mz = peak.mz.to_native();
            let abs_bin = if is_ppm {
                mz.ln() / self.bin_denominator
            } else {
                mz / self.bin_denominator
            };
            let bin_idx = ((abs_bin - self.min_bin).floor() as usize)
                .min(self.lookup_table.len().saturating_sub(1));
            let (target_idx, sign) = self.lookup_table[bin_idx];
            buffer[target_idx] += f32::from(peak.intensity) * sign;
        }

        if normalize_l2 {
            let sq_sum: f32 = buffer.iter().map(|&x| x * x).sum();
            if sq_sum > 0.0 {
                let inv_norm = 1.0 / sq_sum.sqrt();
                for val in buffer.iter_mut() {
                    *val *= inv_norm;
                }
            }
        }
    }
}
/// Configuration for spectrum preprocessing
#[derive(Debug, Clone)]
pub struct HashedSpectrumConfig {
    pub hash_size: usize,
    /// Intensity threshold for noise filtering
    pub intensity_threshold: f32,
    /// Maximum number of peaks to keep (0 = keep all)
    pub max_peaks: usize,
    /// Mass tolerance for peak matching
    pub mass_tolerance: f64,
    /// Bin width for digitization (smaller = more precision, larger = faster)
    pub bin_width: f64,
}

impl_fluent_builder_vanilla! {
    impl HashedSpectrumConfig {
        hash_size: usize,
        intensity_threshold: f32,
        max_peaks: usize,
        mass_tolerance: f64,
        bin_width: f64,
    }
}

impl Default for HashedSpectrumConfig {
    fn default() -> Self {
        Self {
            bin_width: 0.05,           // 1 Da bins for speed
            intensity_threshold: 0.01, // 1% threshold
            max_peaks: 0,              // Keep all peaks
            mass_tolerance: 0.1,       // 0.1 Da tolerance
            hash_size: 4096,
        }
    }
}
impl MassSpectralRecord {
    /// Create a new mass spectrum from peaks
    ///
    /// # Arguments
    /// * `peaks` - Vector of Peak(s)
    pub fn new(mut peaks: Vec<Peak>, charge: i8, id: usize) -> Self {
        peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

        Self {
            id,
            spectrum: MassSpectrum { peaks, charge },
            metadata: Vec::new(),
            source_byte_start: 0,
            source_byte_end: 0,
        }
    }

    pub fn get_charge(&self) -> i8 {
        self.spectrum.charge
    }

    pub fn from_tuples(peaks: &[(f32, f32)]) -> Self {
        let spectrum = MassSpectrum::from_tuples(peaks);
        MassSpectralRecord {
            id: 0,
            spectrum,
            metadata: Vec::new(),
            source_byte_start: 0,
            source_byte_end: 0,
        }
    }

    /// Get peaks as slice
    pub fn peaks(&self) -> &[Peak] {
        &self.spectrum.peaks
    }

    /// Get peaks as slice
    pub fn peaks_mut(&mut self) -> &mut Vec<Peak> {
        &mut self.spectrum.peaks
    }

    /// Get mutable access to metadata
    pub fn metadata_mut(&mut self) -> &mut Vec<(CompactString, CompactString)> {
        &mut self.metadata
    }

    /// Get metadata
    pub fn metadata(&self) -> &[(CompactString, CompactString)] {
        &self.metadata
    }

    // Helper method to make querying the Vec feel like a HashMap
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    pub fn insert_metadata(&mut self, key: &str, value: &str) {
        if let Some(idx) = self
            .metadata
            .iter()
            .position(|(k, _)| k.eq_ignore_ascii_case(&key))
        {
            self.metadata[idx].1 = CompactString::new(value);
        } else {
            self.metadata
                .push((CompactString::new(key), CompactString::new(value)))
        }
    }

    /// Convert to binned spectrum for fast comparison
    ///
    /// # Arguments
    /// * `config` - Configuration for binning
    ///
    /// # Examples
    /// ```
    /// use ms_tools::{MassSpectrum, BinnedSpectrumConfig};
    ///
    /// let peaks = vec![(100.0, 1000.0), (200.0, 500.0)];
    /// let spectrum = MassSpectrum::new(peaks);
    /// let config = BinnedSpectrumConfig::fast();
    /// let binned = spectrum.to_binned(&config);
    /// ```
    pub fn to_binned(&self, config: &BinnedSpectrumConfig) -> BinnedSpectrum {
        if self.spectrum.peaks.is_empty() {
            return BinnedSpectrum {
                intensities: vec![],
                bin_centers: vec![],
                bin_width: config.bin_width,
                min_mz: 0.0,
                max_mz: 0.0,
            };
        }

        let data_min_mz = self.peaks().first().unwrap().mz;
        let data_max_mz = self.peaks().last().unwrap().mz;
        let bin_offset = config.bin_offset.unwrap_or(0.0);

        let first_nominal_center =
            ((data_min_mz - bin_offset) / config.bin_width).round() * config.bin_width + bin_offset;
        let last_nominal_center =
            ((data_max_mz - bin_offset) / config.bin_width).round() * config.bin_width + bin_offset;

        let adjusted_first = if first_nominal_center > data_min_mz {
            first_nominal_center - config.bin_width
        } else {
            first_nominal_center
        };
        let adjusted_last = if last_nominal_center < data_max_mz {
            last_nominal_center + config.bin_width
        } else {
            last_nominal_center
        };

        let num_bins = ((adjusted_last - adjusted_first) / config.bin_width).round() as usize + 1;
        let mut intensities = vec![0.0; num_bins];
        let mut weighted_mz_sums = vec![0.0; num_bins];
        let min_mz = adjusted_first - config.bin_width / 2.0;
        let max_mz = adjusted_last + config.bin_width / 2.0;

        for peak in self.peaks().iter() {
            if peak.intensity >= config.intensity_threshold {
                let bin_index = ((peak.mz - adjusted_first) / config.bin_width).round() as isize;
                if bin_index >= 0 && (bin_index as usize) < num_bins {
                    let idx = bin_index as usize;
                    intensities[idx] += peak.intensity;
                    weighted_mz_sums[idx] += peak.mz * peak.intensity as f64;
                }
            }
        }

        let mut bin_centers = Vec::with_capacity(num_bins);
        for i in 0..num_bins {
            if intensities[i] > 0.0 {
                bin_centers.push(weighted_mz_sums[i] / intensities[i] as f64);
            } else {
                bin_centers.push(adjusted_first + i as f64 * config.bin_width);
            }
        }

        BinnedSpectrum {
            intensities,
            bin_centers,
            bin_width: config.bin_width,
            min_mz,
            max_mz,
        }
    }

    /// Preprocess spectrum (normalize, filter, etc.)
    pub fn preprocess(&mut self, config: &BinnedSpectrumConfig) {
        // Filter by intensity threshold
        if config.intensity_threshold > 0.0 {
            let max_intensity = self
                .peaks()
                .iter()
                .map(|peak| peak.intensity)
                .fold(0.0f32, f32::max);

            let threshold = max_intensity * config.intensity_threshold;
            self.spectrum
                .peaks
                .retain(|peak| peak.intensity >= threshold);
        }

        // Limit number of peaks (keep highest intensity)
        if config.max_peaks > 0 && self.peaks().len() > config.max_peaks {
            // Sort by intensity (descending), then take top N
            self.spectrum
                .peaks
                .sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
            self.spectrum.peaks.truncate(config.max_peaks);
            // Sort back by m/z
            self.spectrum
                .peaks
                .sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        }

        // Normalize intensities to [0, 1]
        let max_intensity = self
            .spectrum
            .peaks
            .iter()
            .map(|peak| peak.intensity)
            .fold(0.0f32, f32::max);

        if max_intensity > 0.0 {
            for peak in &mut self.spectrum.peaks {
                peak.intensity /= max_intensity;
            }
        }
    }

    pub fn to_mass_spectrum(&self) -> MassSpectrum {
        self.spectrum.clone()
    }

    pub fn to_msn(&self) -> MSNSpectrum {
        let precursor_mz = self
            .get_metadata("PEPMASS")
            .map(|mz_str| mz_str.split(" ").next().unwrap().parse::<f64>())
            .unwrap_or(Ok(0.0));
        let rt_in_seconds = self
            .get_metadata("RTINSECONDS")
            .map(|rt_str| rt_str.parse::<f32>())
            .unwrap_or(Ok(0.0));
        MSNSpectrum {
            retention_time_in_seconds: rt_in_seconds.unwrap_or(0.0),
            precursor_mz: precursor_mz.unwrap_or(0.0),
            id: self.id as u64,
            spectrum: self.to_mass_spectrum(),
        }
    }

    pub fn to_hashed_msn(&self, config: &HashedSpectrumConfig) -> HashedMSNSpectrum {
        let Some(precursor_mz) = self
            .get_metadata("PEPMASS")
            .map(|mz_str| mz_str.parse::<f32>())
        else {
            return HashedMSNSpectrum {
                spectrum: self.to_hashed(config),
                precursor_mz: 0.0,
            };
        };

        HashedMSNSpectrum {
            spectrum: self.to_hashed(config),
            precursor_mz: precursor_mz.unwrap_or(0.0),
        }
    }
    pub fn to_hashed(&self, config: &HashedSpectrumConfig) -> HashedSpectrum {
        self.spectrum.to_hashed(config)
    }

    pub fn hashed_cosine_similarity(
        &self,
        other: &Self,
        config: &HashedSpectrumConfig,
    ) -> Option<f64> {
        self.to_hashed(config)
            .cosine_similarity(&other.to_hashed(config))
    }
}

impl BinnedSpectrum {
    /// Get intensities as slice
    pub fn intensities(&self) -> &[f32] {
        &self.intensities
    }

    /// Get bin configuration (bin_width, min_mz, max_mz)
    pub fn bin_info(&self) -> (f64, f64, f64) {
        (self.bin_width, self.min_mz, self.max_mz)
    }

    /// Get bin center for a given bin index (weighted average of peaks in the bin)
    pub fn bin_center(&self, bin_index: usize) -> f64 {
        if bin_index < self.bin_centers.len() {
            self.bin_centers[bin_index]
        } else {
            // Fallback for out-of-bounds access
            self.min_mz + (bin_index as f64 + 0.5) * self.bin_width
        }
    }

    /// Get all bin centers
    pub fn bin_centers(&self) -> &[f64] {
        &self.bin_centers
    }

    /// Calculate cosine similarity with another binned spectrum
    ///
    /// # Arguments
    /// * `other` - Another binned spectrum to compare with
    ///
    /// # Returns
    /// Cosine similarity score (0.0 to 1.0), or None if incompatible
    ///
    /// # Examples
    /// ```
    /// use ms_tools::{MassSpectrum, BinnedSpectrumConfig};
    ///
    /// let peaks1 = vec![(100.0, 1000.0), (200.0, 500.0)];
    /// let peaks2 = vec![(100.0, 1000.0), (200.0, 500.0)];
    /// let spectrum1 = MassSpectrum::new(peaks1);
    /// let spectrum2 = MassSpectrum::new(peaks2);
    /// let config = BinnedSpectrumConfig::default();
    ///
    /// let binned1 = spectrum1.to_binned(&config);
    /// let binned2 = spectrum2.to_binned(&config);
    ///
    /// let similarity = binned1.cosine_similarity(&binned2).unwrap();
    /// assert!((similarity - 1.0).abs() < 1e-10);
    /// ```
    pub fn cosine_similarity(&self, other: &BinnedSpectrum) -> Option<f32> {
        // Align spectra to same m/z range
        let aligned = self.align_with(other);

        // Handle empty alignment (no overlap)
        if aligned.0.is_empty() || aligned.1.is_empty() {
            return Some(0.0);
        }

        cosine_similarity(&aligned.0, &aligned.1).map(|x| x as f32)
    }

    /// Calculate euclidean distance with another binned spectrum
    pub fn euclidean_distance(&self, other: &BinnedSpectrum) -> Option<f32> {
        let aligned = self.align_with(other);

        // Handle empty alignment (no overlap)
        if aligned.0.is_empty() || aligned.1.is_empty() {
            return Some(f32::INFINITY);
        }

        euclidean_distance(&aligned.0, &aligned.1)
    }

    /// Align two spectra to the same m/z range for comparison using weighted bin centers
    pub fn align_with(&self, other: &BinnedSpectrum) -> (Vec<f32>, Vec<f32>) {
        // Find the complete range covering both spectra
        let min_mz = self.min_mz.min(other.min_mz);
        let max_mz = self.max_mz.max(other.max_mz);

        if min_mz >= max_mz {
            // No overlap (shouldn't happen with min/max, but safety check)
            return (vec![], vec![]);
        }

        // Use the larger bin width for alignment to avoid over-sampling
        let bin_width = self.bin_width.max(other.bin_width);

        // Calculate aligned range
        let aligned_first = (min_mz / bin_width).floor() * bin_width;
        let aligned_last = (max_mz / bin_width).ceil() * bin_width;
        let num_bins = ((aligned_last - aligned_first) / bin_width).round() as usize + 1;

        let mut aligned_self = vec![0.0; num_bins];
        let mut aligned_other = vec![0.0; num_bins];

        // Map self intensities using actual weighted bin centers
        for (i, &intensity) in self.intensities.iter().enumerate() {
            if intensity > 0.0 {
                // Only process non-zero intensities
                let actual_bin_center = self.bin_center(i);
                let aligned_bin_index =
                    ((actual_bin_center - aligned_first) / bin_width).round() as usize;
                if aligned_bin_index < num_bins {
                    aligned_self[aligned_bin_index] += intensity;
                }
            }
        }

        // Map other intensities using actual weighted bin centers
        for (i, &intensity) in other.intensities.iter().enumerate() {
            if intensity > 0.0 {
                // Only process non-zero intensities
                let actual_bin_center = other.bin_center(i);
                let aligned_bin_index =
                    ((actual_bin_center - aligned_first) / bin_width).round() as usize;
                if aligned_bin_index < num_bins {
                    aligned_other[aligned_bin_index] += intensity;
                }
            }
        }

        (aligned_self, aligned_other)
    }
}

/// Spectral comparison utilities
pub struct SpectrumComparison;

impl SpectrumComparison {
    /// Fast comparison using binned spectra
    ///
    /// # Arguments
    /// * `spec1` - First spectrum
    /// * `spec2` - Second spectrum
    /// * `config` - Configuration for binning
    ///
    /// # Returns
    /// Cosine similarity score (always returns Some value, 0.0 if no overlap)
    ///
    /// # Examples
    /// ```
    /// use ms_tools::{MassSpectrum, SpectrumComparison, BinnedSpectrumConfig};
    ///
    /// let peaks1 = vec![(100.0, 1000.0), (200.0, 500.0)];
    /// let peaks2 = vec![(100.1, 950.0), (200.1, 480.0)];
    /// let spectrum1 = MassSpectrum::new(peaks1);
    /// let spectrum2 = MassSpectrum::new(peaks2);
    ///
    /// let similarity = SpectrumComparison::fast_cosine_similarity(
    ///     &spectrum1, &spectrum2, &BinnedSpectrumConfig::fast()
    /// ).unwrap();
    /// ```
    pub fn fast_cosine_similarity(
        spec1: &MassSpectralRecord,
        spec2: &MassSpectralRecord,
        config: &BinnedSpectrumConfig,
    ) -> Option<f32> {
        let binned1 = spec1.to_binned(config);
        let binned2 = spec2.to_binned(config);

        // Always return Some value
        binned1.cosine_similarity(&binned2).or(Some(0.0))
    }

    /// Precise peak-to-peak matching similarity
    ///
    /// This method matches peaks within tolerance and calculates similarity
    /// based on matching peak intensities. More precise but slower than binned method.
    pub fn precise_peak_similarity(
        spec1: &MassSpectralRecord,
        spec2: &MassSpectralRecord,
        tolerance: f64,
    ) -> f32 {
        let peaks1 = spec1.peaks();
        let peaks2 = spec2.peaks();

        if peaks1.is_empty() && peaks2.is_empty() {
            return 1.0;
        }

        if peaks1.is_empty() || peaks2.is_empty() {
            return 0.0;
        }

        let mut matched_intensity1 = 0.0;
        let mut matched_intensity2 = 0.0;
        let total_intensity1: f32 = peaks1.iter().map(|peak| peak.intensity).sum();

        let total_intensity2: f32 = peaks2.iter().map(|peak| peak.intensity).sum();

        // Find matching peaks
        for peak1 in peaks1 {
            for peak2 in peaks2 {
                if (peak1.mz - peak2.mz).abs() <= tolerance {
                    matched_intensity1 += peak1.intensity;
                    matched_intensity2 += peak2.intensity;
                    break; // Only match each peak once
                }
            }
        }

        // Calculate similarity as average of matched fractions
        let fraction1 = if total_intensity1 > 0.0 {
            matched_intensity1 / total_intensity1
        } else {
            0.0
        };
        let fraction2 = if total_intensity2 > 0.0 {
            matched_intensity2 / total_intensity2
        } else {
            0.0
        };

        (fraction1 + fraction2) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_creation() {
        let peaks = vec![(100.0, 1000.0), (200.0, 500.0), (300.0, 200.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        assert_eq!(spectrum.peaks().len(), 3);

        // Should be sorted by m/z
        assert_eq!(spectrum.peaks()[0].mz, 100.0);
        assert_eq!(spectrum.peaks()[1].mz, 200.0);
        assert_eq!(spectrum.peaks()[2].mz, 300.0);
    }

    #[test]
    fn test_binned_spectrum() {
        let peaks = vec![(100.0, 1000.0), (101.5, 500.0), (200.0, 800.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        let config = BinnedSpectrumConfig::default();

        let binned = spectrum.to_binned(&config);
        assert!(!binned.intensities().is_empty());

        let (bin_width, _min_mz, _max_mz) = binned.bin_info();
        assert_eq!(bin_width, 1.0);
        // First bin center should be exactly 100.0 for the peak at 100.0 (weighted average)
        assert_eq!(binned.bin_center(0), 100.0);

        // Test that peaks at round values (100.0, 200.0) are properly represented
        // Find the bins with actual peaks
        let mut peak_100_bin = None;
        let mut peak_200_bin = None;

        for (i, &intensity) in binned.intensities().iter().enumerate() {
            if intensity > 0.0 {
                let center = binned.bin_center(i);
                if (center - 100.0).abs() < 1.0 {
                    peak_100_bin = Some(i);
                } else if (center - 200.0).abs() < 1.0 {
                    peak_200_bin = Some(i);
                }
            }
        }

        // Check that weighted bin centers are close to actual peak positions
        if let Some(bin_idx) = peak_100_bin {
            let weighted_center = binned.bin_center(bin_idx);
            assert!(
                (weighted_center - 100.0).abs() < 0.1,
                "Weighted center {} should be close to 100.0",
                weighted_center
            );
        }

        if let Some(bin_idx) = peak_200_bin {
            let weighted_center = binned.bin_center(bin_idx);
            assert!(
                (weighted_center - 200.0).abs() < 0.1,
                "Weighted center {} should be close to 200.0",
                weighted_center
            );
        }
    }

    #[test]
    fn test_weighted_bin_centering() {
        // Test that bin centers are calculated as weighted averages

        // Test 1: Single peak - weighted center should equal peak position
        let peaks = vec![(100.0, 1000.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        let config = BinnedSpectrumConfig::default();
        let binned = spectrum.to_binned(&config);

        // Find which bin has the intensity
        let mut peak_bin_index = 0;
        for (i, &intensity) in binned.intensities().iter().enumerate() {
            if intensity > 0.0 {
                peak_bin_index = i;
                break;
            }
        }

        let weighted_center = binned.bin_center(peak_bin_index);
        assert!(
            (weighted_center - 100.0).abs() < 0.001,
            "Weighted center {} should equal peak position 100.0",
            weighted_center
        );

        // Test 2: Multiple peaks in same bin - should be weighted average
        let peaks_multi = vec![(100.2, 500.0), (100.4, 1000.0)]; // Two peaks in same 1 Da bin
        let spectrum_multi = MassSpectralRecord::from_tuples(&peaks_multi);
        let binned_multi = spectrum_multi.to_binned(&config);

        // Find the bin with intensity
        let mut peak_bin_multi = 0;
        for (i, &intensity) in binned_multi.intensities().iter().enumerate() {
            if intensity > 0.0 {
                peak_bin_multi = i;
                break;
            }
        }

        let weighted_center_multi = binned_multi.bin_center(peak_bin_multi);
        // Expected weighted average: (100.2 * 500 + 100.8 * 1000) / (500 + 1000) = 151900 / 1500 = 100.6
        let expected_weighted = (100.2 * 500.0 + 100.4 * 1000.0) / (500.0 + 1000.0);

        assert!(
            (weighted_center_multi - expected_weighted).abs() < 0.001,
            "Weighted center {} should equal calculated average {}",
            weighted_center_multi,
            expected_weighted
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let peaks = vec![(100.0, 1000.0), (200.0, 500.0)];
        let spectrum1 = MassSpectralRecord::from_tuples(&peaks.clone());
        let spectrum2 = MassSpectralRecord::from_tuples(&peaks);
        let config = BinnedSpectrumConfig::default();

        let similarity =
            SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &config).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_different() {
        // Test with overlapping but different spectra
        let peaks1 = vec![(100.0, 1000.0), (200.0, 500.0)];
        let peaks2 = vec![(100.5, 800.0), (200.3, 400.0)]; // Peaks within same bins
        let spectrum1 = MassSpectralRecord::from_tuples(&peaks1);
        let spectrum2 = MassSpectralRecord::from_tuples(&peaks2);
        let config = BinnedSpectrumConfig::default();

        let similarity =
            SpectrumComparison::fast_cosine_similarity(&spectrum1, &spectrum2, &config).unwrap();
        assert!(similarity < 1.0);
        assert!(similarity > 0.0); // Should have some similarity due to overlap

        // Test with completely non-overlapping spectra
        let peaks3 = vec![(100.0, 1000.0), (200.0, 500.0)];
        let peaks4 = vec![(300.0, 1000.0), (400.0, 500.0)]; // Completely different m/z range
        let spectrum3 = MassSpectralRecord::from_tuples(&peaks3);
        let spectrum4 = MassSpectralRecord::from_tuples(&peaks4);

        let similarity_no_overlap =
            SpectrumComparison::fast_cosine_similarity(&spectrum3, &spectrum4, &config).unwrap();
        assert_eq!(similarity_no_overlap, 0.0); // No overlap should give 0 similarity
    }

    #[test]
    fn test_precise_peak_similarity() {
        let peaks1 = vec![(100.0, 1000.0), (200.0, 500.0)];
        let peaks2 = vec![(100.05, 950.0), (200.03, 480.0)]; // Very close matches
        let spectrum1 = MassSpectralRecord::from_tuples(&peaks1);
        let spectrum2 = MassSpectralRecord::from_tuples(&peaks2);

        let similarity = SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum2, 0.1);
        assert!(similarity > 0.8); // Should be high similarity

        let similarity_strict =
            SpectrumComparison::precise_peak_similarity(&spectrum1, &spectrum2, 0.01);
        assert!(similarity_strict < similarity); // Should be lower with strict tolerance
    }

    #[test]
    fn test_spectrum_preprocessing() {
        let peaks = vec![
            (100.0, 1000.0),
            (200.0, 500.0),
            (300.0, 50.0),
            (400.0, 20.0),
        ];
        let mut spectrum = MassSpectralRecord::from_tuples(&peaks);

        let config = BinnedSpectrumConfig {
            intensity_threshold: 0.1, // 10% threshold
            max_peaks: 2,
            ..Default::default()
        };

        spectrum.preprocess(&config);

        // Should have filtered out low intensity peaks and limited to 2 peaks
        assert!(spectrum.peaks().len() <= 2);

        // Intensities should be normalized
        let max_intensity = spectrum
            .peaks()
            .iter()
            .map(|peak| peak.intensity)
            .fold(0.0f32, f32::max);
        assert!((max_intensity - 1.0).abs() < 1e-10);
    }

    // Helper function to create a config with a specific hash size
    fn test_config() -> HashedSpectrumConfig {
        let mut config = HashedSpectrumConfig::default();
        config.hash_size = 1024; // Small enough for fast tests, large enough to avoid frequent collisions
        config.bin_width = 1.0;
        config.intensity_threshold = 0.01;
        config
    }

    #[test]
    fn test_hashed_spectrum_dimensions() {
        let peaks = vec![(100.0, 1000.0), (200.0, 500.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        let config = test_config();

        let hashed = spectrum.to_hashed(&config);

        // Ensure the output vector perfectly matches the requested hash_size
        assert_eq!(hashed.features.len(), 1024);
    }

    #[test]
    fn test_hashed_deterministic_output() {
        let peaks = vec![(100.0, 1000.0), (200.0, 500.0), (300.0, 200.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        let config = test_config();

        let hashed1 = spectrum.to_hashed(&config);
        let hashed2 = spectrum.to_hashed(&config);

        // The same peaks should always map to the exact same feature vector
        assert_eq!(hashed1.features, hashed2.features);
    }

    #[test]
    fn test_hashed_cosine_similarity_identical() {
        let peaks = vec![(100.0, 1000.0), (200.0, 500.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);
        let config = test_config();

        let hashed = spectrum.to_hashed(&config);

        // Similarity with itself should be exactly 1.0
        let similarity = cosine_similarity(&hashed.features, &hashed.features).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hashed_cosine_similarity_different() {
        let config = test_config();

        // Spectrum A: Low mass range
        let spec_a = MassSpectralRecord::from_tuples(&[(100.0, 1000.0), (101.0, 500.0)]);
        let hash_a = spec_a.to_hashed(&config);

        // Spectrum B: High mass range (unlikely to collide with A in a 1024-space)
        let spec_b = MassSpectralRecord::from_tuples(&[(800.0, 1000.0), (801.0, 500.0)]);
        let hash_b = spec_b.to_hashed(&config);

        let similarity = cosine_similarity(&hash_a, &hash_b).unwrap();

        // Should be extremely close to 0.0 since the peaks are completely different
        // and shouldn't collide
        assert!(similarity < 0.01);
    }

    #[test]
    fn test_hashed_intensity_thresholding() {
        let mut config = test_config();
        config.intensity_threshold = 100.0; // High threshold

        // Only the first peak should survive the threshold
        let peaks = vec![(100.0, 1000.0), (200.0, 50.0), (300.0, 10.0)];
        let spectrum = MassSpectralRecord::from_tuples(&peaks);

        let hashed = spectrum.to_hashed(&config);

        // Count non-zero elements in the hashed features
        let non_zero_count = hashed.features.iter().filter(|&&x| x != 0.0).count();

        // Because of the threshold, only 1 peak should have been hashed
        assert_eq!(non_zero_count, 1);
    }

    #[test]
    fn test_hashed_mismatched_sizes() {
        let spectrum = MassSpectralRecord::from_tuples(&[(100.0, 1000.0)]);

        let mut config1 = test_config();
        config1.hash_size = 1024;
        let hashed1 = spectrum.to_hashed(&config1);

        let mut config2 = test_config();
        config2.hash_size = 2048;
        let hashed2 = spectrum.to_hashed(&config2);

        // Comparing two vectors of different sizes should cleanly return None
        let similarity = cosine_similarity(&hashed1, &hashed2);

        assert!(similarity.is_none());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct MassSpectrumSoA {
    pub mz_bins: Vec<u32>,
    pub intensities: Vec<f32>,
    pub charge: i8,
}

impl MassSpectrumSoA {
    pub fn l2_normalize(&mut self) {
        // 1. Calculate the sum of the squares of the intensities
        let sum_of_squares: f32 = self.intensities.iter().map(|&x| x * x).sum();

        // 2. Prevent division by zero or NaN errors if the spectrum is completely empty
        if sum_of_squares <= f32::EPSILON {
            return;
        }

        // 3. Calculate the magnitude (L2 norm)
        let l2_norm = sum_of_squares.sqrt();

        // 4. Pre-compute the reciprocal
        // Multiplication is significantly faster on modern CPUs than division.
        let reciprocal = 1.0 / l2_norm;

        // 5. Apply the normalization
        for intensity in self.intensities.iter_mut() {
            *intensity *= reciprocal;
        }
    }
}

impl MassSpectrum {
    pub fn to_soa(&self, tolerance: MassTolerance) -> MassSpectrumSoA {
        let capacity = self.peaks.len();
        let mut mz_bins = Vec::with_capacity(capacity);
        let mut intensities = Vec::with_capacity(capacity);

        let ppm_log_base = match tolerance {
            MassTolerance::Ppm(ppm) => (1.0 + ppm / 1_000_000.0).ln(),
            _ => 0.0,
        };

        for peak in self.peaks.iter() {
            let bin = match tolerance {
                MassTolerance::Da(da) => (peak.mz / da).round() as u32,
                MassTolerance::Ppm(_) => (peak.mz.ln() / ppm_log_base).round() as u32,
            };

            if let Some(&last_bin) = mz_bins.last() {
                if last_bin == bin {
                    let last_idx = intensities.len() - 1;
                    if peak.intensity > intensities[last_idx] {
                        intensities[last_idx] = peak.intensity;
                    }
                    continue;
                }
            }

            mz_bins.push(bin);
            intensities.push(peak.intensity);
        }

        MassSpectrumSoA {
            mz_bins,
            intensities,
            charge: self.charge,
        }
    }
}

impl ArchivedMassSpectrum {
    /// Converts the Array of Structs (AoS) to a Struct of Arrays (SoA)
    /// while discretizing the m/z values into integer bins.
    /// Assumes `self.peaks` is strictly sorted by m/z.
    pub fn to_soa(&self, tolerance: MassTolerance) -> MassSpectrumSoA {
        // Pre-allocate assuming no collisions to prevent reallocations
        let capacity = self.peaks.len();
        let mut mz_bins = Vec::with_capacity(capacity);
        let mut intensities = Vec::with_capacity(capacity);

        // Pre-compute the denominator for PPM to avoid doing logarithms inside the loop.
        // The formula is: Bin = ln(mz) / ln(1 + ppm / 1,000,000)
        let ppm_log_base = match tolerance {
            MassTolerance::Ppm(ppm) => (1.0 + ppm / 1_000_000.0).ln(),
            _ => 0.0, // Unused in Da strategy
        };

        for peak in self.peaks.iter() {
            // 1. Calculate the integer bin
            let bin = match tolerance {
                MassTolerance::Da(da) => (peak.mz / da).round() as u32,
                MassTolerance::Ppm(_) => (peak.mz.to_native().ln() / ppm_log_base).round() as u32,
            };

            // 2. Handle Intra-Spectrum Collisions (Deduplication)
            // If this peak maps to the exact same bin as the previous peak,
            // we keep the maximum intensity and do not push a new bin.
            if let Some(&last_bin) = mz_bins.last() {
                if last_bin == bin {
                    let last_idx = intensities.len() - 1;
                    if peak.intensity > intensities[last_idx] {
                        intensities[last_idx] = peak.intensity.to_native();
                    }
                    continue; // Skip the push
                }
            }

            // 3. Push unique bins
            mz_bins.push(bin);
            intensities.push(peak.intensity.to_native());
        }

        MassSpectrumSoA {
            mz_bins,
            intensities,
            charge: self.charge,
        }
    }
}

/// Represents a mass spectrum as a collection of peaks
#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct FlatMSNSpectrum {
    pub id: u64,
    pub mz: Vec<f64>,
    pub intensities: Vec<f32>,
}
