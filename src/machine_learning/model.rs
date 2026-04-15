use crate::machine_learning::dataloader::*;
use burn::data::dataloader::DataLoader;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu, conv::Conv1d, conv::Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use std::sync::Arc;

pub fn train_model<B: AutodiffBackend>(
    device: B::Device,
    // FIX: Expect the Arc-wrapped trait objects!
    train_dataloader: Arc<dyn DataLoader<B, MsBatch<B>>>,
    valid_dataloader: Arc<dyn DataLoader<B::InnerBackend, MsBatch<B::InnerBackend>>>,
) {
    let optimizer = burn::optim::AdamConfig::new().init();
    let model = IsotopeDetector::new(&device);

    let training = SupervisedTraining::new(
        "/tmp/ms_isotope_detector",
        train_dataloader,
        valid_dataloader,
    )
    .num_epochs(10)
    .metric_train_numeric(AccuracyMetric::new())
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(AccuracyMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(burn::record::CompactRecorder::new());

    let _trained_model = training.launch(Learner::new(model, optimizer, 1e-4));
}

#[derive(Module, Debug)]
pub struct IsotopeDetector<B: Backend> {
    conv: Conv1d<B>,
    fc: Linear<B>,
    activation: Relu,
}

impl<B: Backend> IsotopeDetector<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(1, 8, 3).init(device); // Input: intensity windows
        let fc = LinearConfig::new(8 * 38, 3).init(device);

        Self {
            conv,
            fc,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.conv.forward(input);
        let x = self.activation.forward(x);
        let x = x.flatten(1, 2);
        self.fc.forward(x)
    }
}
impl<B: AutodiffBackend> TrainStep for IsotopeDetector<B> {
    type Input = MsBatch<B>;
    type Output = ClassificationOutput<B>;

    // FIX: Use explicit return types instead of Self::Output to avoid E0223 ambiguity
    fn step(&self, batch: MsBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let predictions = self.forward(batch.spectra);

        let loss = CrossEntropyLossConfig::new()
            .init(&predictions.device())
            .forward(predictions.clone(), batch.targets.clone());

        let output = ClassificationOutput::new(loss.clone(), predictions, batch.targets);
        TrainOutput::new(self, loss.backward(), output)
    }
}
// FIX: Implement InferenceStep instead of ValidStep
impl<B: Backend> InferenceStep for IsotopeDetector<B> {
    type Input = MsBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MsBatch<B>) -> ClassificationOutput<B> {
        let predictions = self.forward(batch.spectra);

        let loss = CrossEntropyLossConfig::new()
            .init(&predictions.device())
            .forward(predictions.clone(), batch.targets.clone());

        ClassificationOutput::new(loss, predictions, batch.targets)
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::dataloader::{MassSpectrum, MsTrainingItem, Peak};
    use rand::prelude::IndexedRandom;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_ml() {
        let input_path = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp";
        let output_path = "data/training_data.json";

        let file = File::open(input_path).expect("Could not open MSP file");
        let reader = BufReader::new(file);

        let mut training_items: Vec<MsTrainingItem> = Vec::new();
        let mut current_peaks: Vec<Peak> = Vec::new();
        let mut charge = 1;

        println!("Parsing MSP and extracting labeled windows...");

        for line in reader.lines() {
            let line = line.unwrap();
            let line = line.trim();

            if line.is_empty() {
                if !current_peaks.is_empty() {
                    // THE FIX: Find the highest intensity peak in the spectrum (Base Peak)
                    // We will treat this fragment as our ground-truth Monoisotopic peak.
                    let base_peak = current_peaks
                        .iter()
                        .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap())
                        .copied()
                        .unwrap();

                    // 1. Label the Monoisotopic Peak
                    let mono_window = extract_window(&current_peaks, base_peak.mz);
                    if !mono_window.is_empty() {
                        // Safety check
                        training_items.push(MsTrainingItem {
                            spectrum: MassSpectrum {
                                peaks: mono_window,
                                charge,
                            },
                            center_mz: base_peak.mz,
                            label: 1,
                        });
                    }

                    // 2. Label the +1 Isotope of the Base Peak
                    let isotope_mz = base_peak.mz + (1.0033 / charge as f32);
                    let iso_window = extract_window(&current_peaks, isotope_mz);
                    // Only push if an isotope peak actually exists!
                    if !iso_window.is_empty() {
                        training_items.push(MsTrainingItem {
                            spectrum: MassSpectrum {
                                peaks: iso_window,
                                charge,
                            },
                            center_mz: isotope_mz, // MUST be centered on the isotope mz
                            label: 2,
                        });
                    }

                    // 3. Label Noise (pick a random peak far away from the base peak)
                    let mut rng = rand::rng();
                    if let Some(random_peak) = current_peaks.choose(&mut rng) {
                        if (random_peak.mz - base_peak.mz).abs() > 2.0 {
                            let noise_window = extract_window(&current_peaks, random_peak.mz);
                            if !noise_window.is_empty() {
                                training_items.push(MsTrainingItem {
                                    spectrum: MassSpectrum {
                                        peaks: noise_window,
                                        charge,
                                    },
                                    center_mz: random_peak.mz, // MUST be centered on the random peak
                                    label: 0,
                                });
                            }
                        }
                    }
                }

                current_peaks.clear();
                charge = 1;
            } else if line.starts_with("Precursor_type:") {
                if line.contains("-") {
                    charge = -1;
                } else {
                    charge = 1;
                }
            } else if line.chars().next().unwrap_or('A').is_ascii_digit() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let (Ok(mz), Ok(intensity)) =
                        (parts[0].parse::<f32>(), parts[1].parse::<f32>())
                    {
                        current_peaks.push(Peak { mz, intensity });
                    }
                }
            }
        }

        let out_file = File::create(output_path).unwrap();
        serde_json::to_writer(out_file, &training_items).unwrap();
        println!(
            "Saved {} training items to {}",
            training_items.len(),
            output_path
        );
    }

    // Look 1.5 Da left, and 2.5 Da right (total 4.0 Da window)
    fn extract_window(peaks: &[Peak], center_mz: f32) -> Vec<Peak> {
        peaks
            .iter()
            .filter(|p| p.mz >= center_mz - 1.5 && p.mz <= center_mz + 2.5)
            .cloned()
            .collect()
    }
}
