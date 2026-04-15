use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData}; // <-- Notice the `Int` import!
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct Peak {
    pub mz: f32,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassSpectrum {
    pub peaks: Vec<Peak>,
    pub charge: i8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MsTrainingItem {
    pub spectrum: MassSpectrum,
    pub center_mz: f32,
    pub label: i32,
}

#[derive(Clone, Debug)]
pub struct MsBatch<B: Backend> {
    pub spectra: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>, // <-- CRITICAL: Targets must be Int for classification!
}

pub struct MsBatcher;

impl MsBatcher {
    // No more generics here! When you call this elsewhere, just use `MsBatcher::new()`
    pub fn new() -> Self {
        Self
    }
}

impl<B: Backend> Batcher<B, MsTrainingItem, MsBatch<B>> for MsBatcher {
    fn batch(&self, items: Vec<MsTrainingItem>, device: &B::Device) -> MsBatch<B> {
        let batch_size = items.len();
        let num_bins = 40; // Used directly, no `self.`
        let window_width = 4.0;

        let mut all_bins = Vec::with_capacity(batch_size * num_bins);
        let mut targets = Vec::with_capacity(batch_size);

        for item in items {
            let mut bins = vec![0.0_f32; num_bins];
            let window_start = item.center_mz - 1.5;

            for peak in item.spectrum.peaks {
                let relative_mz = peak.mz - window_start;
                let bin_idx = ((relative_mz / window_width) * num_bins as f32) as usize;

                if bin_idx < num_bins {
                    bins[bin_idx] += peak.intensity;
                }
            }
            all_bins.extend(bins);
            targets.push(item.label);
        }

        let spectra_data = TensorData::new(all_bins, [batch_size, 1, num_bins]);
        // Use the passed `device` directly
        let spectra_tensor = Tensor::<B, 3>::from_data(spectra_data, device);

        let targets_data = TensorData::new(targets, [batch_size]);
        // Build this specifically as an Int tensor
        let targets_tensor = Tensor::<B, 1, Int>::from_data(targets_data, device);

        MsBatch {
            spectra: spectra_tensor,
            targets: targets_tensor,
        }
    }
}
