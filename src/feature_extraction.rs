use crate::Peak;
use crate::spectrum::MassSpectralRecord;
use crate::struct_with_new;
use ndarray::Array2;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use xxhash_rust::xxh32::xxh32;

/// Stage 2: Transforms preprocessed spectra into a new data structure (e.g., for ML).
pub trait FeatureExtractor<Output> {
    fn extract(&self, spectra: &[MassSpectralRecord]) -> Output;
}

struct_with_new!(
    pub struct FeatureHashing {
        pub n_features: usize,
        pub bin_width: f64,
        pub intensity_threshold: f32,
    }
);
impl FeatureExtractor<Array2<f32>> for FeatureHashing {
    fn extract(&self, spectra: &[MassSpectralRecord]) -> Array2<f32> {
        let num_spectra = spectra.len();
        let total_size = num_spectra * self.n_features;

        // 1. Pre-allocate the single, large flat vector for the entire dataset
        let mut flat_features = vec![0.0; total_size];

        // 2. Process spectra in parallel, giving each thread a safe, mutable slice
        flat_features
            .par_chunks_mut(self.n_features)
            .zip(spectra.par_iter())
            .for_each(|(feature_chunk, spectrum)| {
                for peak in spectrum.peaks() {
                    if peak.intensity >= self.intensity_threshold {
                        // A. Convert m/z to a discrete integer bin
                        let bin_index = (peak.mz / self.bin_width).round() as isize;

                        let bytes = bin_index.to_le_bytes();
                        let hash_val = xxh32(&bytes, 0) as u64;

                        // C. Map to the fixed-size array
                        let index = (hash_val % self.n_features as u64) as usize;

                        // D. The Sign Trick (Weinberger Hashing)
                        // Uses the highest bit to determine if we add or subtract to prevent bias
                        let sign = if (hash_val >> 63) == 1 { -1.0 } else { 1.0 };

                        // E. Accumulate intensity safely
                        feature_chunk[index] += peak.intensity * sign;
                    }
                }
            });

        // 3. Wrap the flat vector into a 2D ndarray (Zero-cost operation)
        Array2::from_shape_vec((num_spectra, self.n_features), flat_features)
            .expect("Shape mismatch: Total size mathematically guarantees a perfect fit.")
    }
}

struct_with_new!(
    pub struct BinSpectra {
        bin_size: f64,
        min_mz: f64,
        max_mz: f64,
    }
);

impl FeatureExtractor<Array2<f32>> for BinSpectra {
    fn extract(&self, spectra: &[MassSpectralRecord]) -> Array2<f32> {
        let num_spectra = spectra.len();
        let num_bins = ((self.max_mz - self.min_mz) / self.bin_size).ceil() as usize;
        let total_size = num_spectra * num_bins;

        // 1. Pre-allocate the flat vector
        let mut flat_features = vec![0.0; total_size];

        // 2. Safely populate it in parallel using Rayon
        flat_features
            .par_chunks_mut(num_bins)
            .zip(spectra.par_iter())
            .for_each(|(feature_chunk, spectrum)| {
                for peak in spectrum.peaks() {
                    if peak.mz >= self.min_mz && peak.mz < self.max_mz {
                        let bin_index = ((peak.mz - self.min_mz) / self.bin_size).floor() as usize;
                        if bin_index < num_bins {
                            feature_chunk[bin_index] += peak.intensity;
                        }
                    }
                }
            });

        // 3. Consume the Vec and wrap it in a 2D Array
        // The shape is (rows, columns) -> (number of spectra, number of bins per spectrum)
        Array2::from_shape_vec((num_spectra, num_bins), flat_features)
            .expect("Shape mismatch: Total size mathematically guarantees a perfect fit.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_bin_spectra_extraction() {
        // Setup mock spectra
        let spectra = vec![
            MassSpectralRecord::from_tuples(&[(100.05, 10.0), (100.15, 20.0), (100.95, 5.0)]), // Spectrum 0
            MassSpectralRecord::from_tuples(&[(100.02, 15.0), (101.50, 30.0)]), // Spectrum 1
        ];

        // Bin size 0.1, from 100.0 to 102.0 -> expects exactly 20 bins.
        let binner = BinSpectra::new(0.1, 100.0, 102.0);
        let result = binner.extract(&spectra);

        // Assert shape is (2 spectra, 20 features)
        assert_eq!(result.shape(), &[2, 20]);

        // Check Spectrum 0
        // 100.05 falls into bin 0
        assert_eq!(result[[0, 0]], 10.0);
        // 100.15 falls into bin 1
        assert_eq!(result[[0, 1]], 20.0);
        // 100.95 falls into bin 9
        assert_eq!(result[[0, 9]], 5.0);

        // Check Spectrum 1
        // 100.02 falls into bin 0
        assert_eq!(result[[1, 0]], 15.0);
        // 101.50 falls into bin 15
        assert_eq!(result[[1, 15]], 30.0);

        // Ensure out-of-bounds peaks don't crash and are ignored (empty bins remain 0)
        assert_eq!(result[[1, 19]], 0.0);
    }

    #[test]
    fn test_feature_hashing_extraction() {
        // Setup mock spectra
        let spectra = vec![MassSpectralRecord::from_tuples(&[
            (500.12, 100.0),
            (600.45, 50.0),
        ])];

        // 1024 features, multiplying mz by 100.0 for precision
        let hasher = FeatureHashing::new(1024, 0.5, 100.0);
        let result = hasher.extract(&spectra);

        // Assert shape is (1 spectrum, 1024 features)
        assert_eq!(result.shape(), &[1, 1024]);

        // In hashing, we don't know the exact index without computing the hash,
        // but we DO know that mass is conserved. The sum of the dense array
        // must equal the sum of the peak intensities.
        let total_intensity: f32 = result.sum();
        assert_eq!(total_intensity, 150.0);
    }
}
