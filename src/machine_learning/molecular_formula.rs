use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::{
    dataloader::{DataLoader, batcher::Batcher},
    dataset::Dataset,
};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use chemistry_consts::MONOISOTOPIC_MASSES;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

const ELEMENT_MASSES: [f32; 6] = [
    MONOISOTOPIC_MASSES[6] as f32,  // C
    MONOISOTOPIC_MASSES[1] as f32,  // H
    MONOISOTOPIC_MASSES[7] as f32,  // N
    MONOISOTOPIC_MASSES[8] as f32,  // O
    MONOISOTOPIC_MASSES[15] as f32, // P
    MONOISOTOPIC_MASSES[16] as f32, // S
];

const ELEMENT_ATOMIC_NUMBERS: [u8; 6] = [1, 6, 7, 8, 15, 16];

const FEATURE_SIZE: usize = 4;
const HIDDEN_SIZE: usize = 128;
const MASS_LOSS_SCALING: f32 = 0.1;
const LEARNING_RATE: f64 = 1E-3;
const NUM_EPOCHS: usize = 200;

pub fn train_formula_predictor<B: AutodiffBackend>(
    device: B::Device,
    train_dataloader: Arc<dyn DataLoader<B, FormulaBatch<B>>>,
    valid_dataloader: Arc<dyn DataLoader<B::InnerBackend, FormulaBatch<B::InnerBackend>>>,
) {
    // 1. Initialize the optimizer
    let optimizer = burn::optim::AdamConfig::new().init();

    // 2. Initialize the model
    let model = FormulaPredictor::new(&device);

    // 3. Set up the SupervisedTraining pipeline
    let training = SupervisedTraining::new(
        "/tmp/formula_predictor", // Your artifact directory
        train_dataloader,
        valid_dataloader,
    )
    .num_epochs(NUM_EPOCHS) // You can easily bump this up for longer training
    .metric_train_numeric(LossMetric::new()) // Track MSE Loss for training
    .metric_valid_numeric(LossMetric::new()) // Track MSE Loss for validation
    .with_file_checkpointer(burn::record::CompactRecorder::new());

    // 4. Launch the training
    let trained_model = training.launch(Learner::new(model, optimizer, LEARNING_RATE));
    let record = trained_model.model.into_record();
    burn::record::CompactRecorder::new()
        .record(record, format!("{}/model", "/tmp/formula_predictor").into())
        .expect("Failed to save final model");

    println!("Training completed! Checkpoints saved to /tmp/formula_predictor");
}
#[derive(Config, Debug)]
pub struct FormulaTrainingConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    pub optimizer: AdamConfig,
}
pub struct FormulaDataset {
    items: Vec<FormulaTrainingItem>,
}

impl FormulaDataset {
    /// Loads the generated JSON dataset from the specified file path.
    pub fn new(file_path: &str) -> Self {
        let file = File::open(file_path)
            .unwrap_or_else(|_| panic!("Failed to open dataset file at {}", file_path));

        let reader = BufReader::new(file);

        // Parse the JSON array directly into our Vector of training items
        let items: Vec<FormulaTrainingItem> = serde_json::from_reader(reader)
            .expect("Failed to parse JSON into Vec<FormulaTrainingItem>. Make sure the file format matches.");

        Self { items }
    }
}

// Implement the Burn Dataset trait
impl Dataset<FormulaTrainingItem> for FormulaDataset {
    fn get(&self, index: usize) -> Option<FormulaTrainingItem> {
        // Return a clone of the item at the given index, if it exists
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
#[derive(Module, Debug)]
pub struct FormulaPredictor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> FormulaPredictor<B> {
    pub fn new(device: &B::Device) -> Self {
        // Input: 3 features (Scaled Mono Mass, M+1 Rel Int, M+2 Rel Int)
        // Hidden layers: 64 nodes (expand the feature space to learn non-linear patterns)
        // Output: 6 features (Counts of C, H, N, O, P, S)
        Self {
            fc1: LinearConfig::new(FEATURE_SIZE, HIDDEN_SIZE).init(device), // Changed from 6 to 3!
            fc2: LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE).init(device),
            fc3: LinearConfig::new(HIDDEN_SIZE, 6).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.fc1.forward(input));
        let x = self.activation.forward(self.fc2.forward(x));
        self.fc3.forward(self.activation.forward(x))
    }
}

// --- The Training Batch Struct ---
#[derive(Clone, Debug)]
pub struct FormulaBatch<B: Backend> {
    pub peaks: Tensor<B, 2>,      // Shape: [batch_size, 3]
    pub true_atoms: Tensor<B, 2>, // Shape: [batch_size, 6]
    pub true_mass: Tensor<B, 1>,  // Shape: [batch_size]
}

// --- The Physics-Informed Training Step ---
impl<B: AutodiffBackend> TrainStep for FormulaPredictor<B> {
    type Input = FormulaBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let predicted_atoms = self.forward(batch.peaks);
        // 1. Pure Atom Loss: The model only cares about getting the counts right
        // Switch to L1 (Mean Absolute Error) to force sparsity (pushing small decimals to 0)
        let atom_loss = predicted_atoms
            .clone()
            .sub(batch.true_atoms.clone())
            .abs()
            .mean();

        let output = RegressionOutput::new(atom_loss.clone(), predicted_atoms, batch.true_atoms);

        // Pass atom_loss.backward() directly
        TrainOutput::new(self, atom_loss.backward(), output)
    }
}

// --- The Validation / Inference Step (Required by Burn) ---
impl<B: Backend> InferenceStep for FormulaPredictor<B> {
    type Input = FormulaBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let predicted_atoms = self.forward(batch.peaks);

        let atom_loss = predicted_atoms
            .clone()
            .sub(batch.true_atoms.clone())
            .abs()
            .mean();

        // Just return the atom_loss
        RegressionOutput::new(atom_loss, predicted_atoms, batch.true_atoms)
    }
}

// --- Our Input Data Structures ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Peak {
    pub mz: f32,
    pub intensity: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormulaTrainingItem {
    pub peaks: Vec<Peak>,
    pub true_atoms: [f32; 6], // [C, H, N, O, P, S] counts
    pub exact_mass: f32,
}

pub struct FormulaBatcher;

impl FormulaBatcher {
    pub fn new() -> Self {
        Self
    }
}

impl<B: Backend> Batcher<B, FormulaTrainingItem, FormulaBatch<B>> for FormulaBatcher {
    fn batch(&self, items: Vec<FormulaTrainingItem>, device: &B::Device) -> FormulaBatch<B> {
        let batch_size = items.len();

        // Pre-allocate our flattened vectors for 3 features
        let mut all_peaks = Vec::with_capacity(batch_size * 3);
        let mut all_atoms = Vec::with_capacity(batch_size * 6);
        let mut all_masses = Vec::with_capacity(batch_size);

        for mut item in items {
            // 1. Sort peaks by m/z to ensure [Mono, M+1, M+2] order
            item.peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));

            // 2. Extract the Monoisotopic Intensity (for scaling)
            let mono_int = item
                .peaks
                .get(0)
                .map(|p| p.intensity)
                .unwrap_or(1.0)
                .max(1e-6); // Prevent div by zero
            // 3. Build the 4 features
            let mut features = [0.0_f32; 4];

            // Feature 0: Nominal Mass (Roughly determines the total size)
            features[0] = item.exact_mass.round() / 1000.0;

            // Feature 1: Mass Defect (Crucial for identifying specific elements!)
            features[1] = item.exact_mass - item.exact_mass.round();

            // Feature 2: M+1 Scaled (Multiply by 100 so it's not tiny)
            if let Some(p1) = item.peaks.get(1) {
                features[2] = (p1.intensity / mono_int) * 100.0;
            }

            // Feature 3: M+2 Scaled
            if let Some(p2) = item.peaks.get(2) {
                features[3] = (p2.intensity / mono_int) * 100.0;
            }

            all_peaks.extend_from_slice(&features);
            all_atoms.extend_from_slice(&item.true_atoms);
            all_masses.push(item.exact_mass);
        }

        // --- Create Tensors ---

        // Input features: [batch_size, 3 features]
        let peaks_data = TensorData::new(all_peaks, [batch_size, 4]);
        let peaks_tensor = Tensor::<B, 2>::from_data(peaks_data, device);

        // Target Atoms: [batch_size, 6 elements]
        let atoms_data = TensorData::new(all_atoms, [batch_size, 6]);
        let atoms_tensor = Tensor::<B, 2>::from_data(atoms_data, device);

        // Target Mass: [batch_size]
        let mass_data = TensorData::new(all_masses, [batch_size]);
        let mass_tensor = Tensor::<B, 1>::from_data(mass_data, device);

        FormulaBatch {
            peaks: peaks_tensor,
            true_atoms: atoms_tensor,
            true_mass: mass_tensor,
        }
    }
}

pub fn build_dataloader<B: burn::tensor::backend::Backend>(
    dataset_path: &str,
    batch_size: usize,
) -> Arc<dyn burn::data::dataloader::DataLoader<B, FormulaBatch<B>>> {
    // 1. Load the dataset into memory
    let dataset = FormulaDataset::new(dataset_path);

    // 2. Instantiate your custom batcher
    let batcher = FormulaBatcher::new();

    // 3. Build the DataLoader
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42) // Seed for reproducible shuffling
        .num_workers(4) // Parallelize data loading
        .build(dataset);

    dataloader
}
// Assuming FormulaPredictor and Peak are in scope

/// Predicts the molecular formula [C, H, N, O, P, S] for a given set of peaks.
pub fn predict_formula<B: burn::tensor::backend::Backend>(
    model_dir: &str, // e.g., "/tmp/formula_predictor"
    mut peaks: Vec<Peak>,
    exact_mass: f32,
    device: &B::Device,
) -> [f32; 6] {
    let record_path = format!("{}/model", model_dir);
    let record = CompactRecorder::new()
        .load(record_path.into(), device)
        .expect("Failed to load model weights. Ensure the model was saved correctly.");

    let model = FormulaPredictor::new(device).load_record(record);

    peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
    let mono_int = peaks[0].intensity;
    // 3. Build the 4 features
    let mut features = [0.0_f32; 4];

    // Feature 0: Nominal Mass (Roughly determines the total size)
    features[0] = exact_mass.round() / 1000.0;

    // Feature 1: Mass Defect (Crucial for identifying specific elements!)
    features[1] = exact_mass - exact_mass.round();

    // Feature 2: M+1 Scaled (Multiply by 100 so it's not tiny)
    if let Some(p1) = peaks.get(1) {
        features[2] = (p1.intensity / mono_int) * 100.0;
    }

    // Feature 3: M+2 Scaled
    if let Some(p2) = peaks.get(2) {
        features[3] = (p2.intensity / mono_int) * 100.0;
    }

    // 3. Create a 2D Tensor with a batch size of 1: Shape [1, 3]
    let input_data = TensorData::new(features.to_vec(), [1, FEATURE_SIZE]);
    let input_tensor = Tensor::<B, 2>::from_data(input_data, device);

    // 4. Run the forward pass!
    // We don't need gradients for inference, so this works nicely with any standard backend.
    let predicted_tensor = model.forward(input_tensor);

    // 5. Extract the data back into standard Rust arrays
    let output_data = predicted_tensor.into_data();
    // FormulaPredictor's output layer is fixed at f32; not caller-input-dependent.
    let predicted_values = output_data
        .as_slice::<f32>()
        .expect("FormulaPredictor output tensor is always f32");

    // Copy the 6 predicted floats [C, H, N, O, P, S] into a fixed array
    let mut predicted_atoms = [0.0; 6];
    predicted_atoms.copy_from_slice(&predicted_values[0..6]);

    // Print a human-readable format, rounding to the nearest integer
    println!("Predicted Formula Counts:");
    println!("C: {}", predicted_atoms[0].round() as i32);
    println!("H: {}", predicted_atoms[1].round() as i32);
    println!("N: {}", predicted_atoms[2].round() as i32);
    println!("O: {}", predicted_atoms[3].round() as i32);
    println!("P: {}", predicted_atoms[4].round() as i32);
    println!("S: {}", predicted_atoms[5].round() as i32);

    predicted_atoms
}

#[cfg(test)]
pub mod sum_formula_tests {
    use std::str::FromStr;

    // Import your custom structs from the outer module
    use crate::machine_learning::molecular_formula::{FormulaTrainingItem, Peak};
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    use molecules::io::{MolecularGraph, MoleculeCore, MoleculeGraph, ToSMILES};
    use std::collections::HashMap;

    use super::*; // Import your Predictor, Batcher, Peaks, etc.
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray, Wgpu};
    #[test]
    fn generate_sum_formula_data() {
        let file = File::open("test_files/smiles_tree_permutations.txt").unwrap();
        let reader = BufReader::new(file);

        // HashMap to filter duplicates. Key: Formula string, Value: Training Item
        let mut unique_formulas: HashMap<String, FormulaTrainingItem> = HashMap::new();

        for line in reader.lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }

            let mol_str = line.split_ascii_whitespace().next().unwrap();

            // Handle parsing gracefully in case of invalid SMILES
            let mol = match MolecularGraph::from_smiles_single(mol_str) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let formula = mol.molecular_formula();
            let formula_string = formula.to_string();

            // 1. Filter duplicates using the HashMap
            if unique_formulas.contains_key(&formula_string) {
                continue;
            }

            if formula
                .elements()
                .keys()
                .any(|key| !ELEMENT_ATOMIC_NUMBERS.contains(&key))
            {
                continue;
            }

            // 2. Extract Data
            // This returns a vec<(f64,f64)> (mz,intensity) of length 3 and resolution 1
            let pattern = formula.isotopic_pattern(3, 1);
            if pattern
                .iter()
                .any(|(_, intensity)| intensity.is_infinite() || intensity.is_nan())
            {
                println!("Found Error for smiles {}", mol_str);
                println!("Pattern {:?}", pattern);
            }
            let monoisotopic_mass = formula.monoisotopic_mass();

            // 3. Convert pattern tuples to Peak structs
            let peaks: Vec<Peak> = pattern
                .into_iter()
                .map(|(mz, intensity)| Peak {
                    mz: mz as f32,
                    intensity: intensity as f32,
                })
                .collect();

            // 4. Parse atom counts for [C, H, N, O, P, S]
            let true_atoms = parse_formula_to_array(&formula_string);

            // 5. Construct the training item and insert it into the HashMap
            let item = FormulaTrainingItem {
                peaks,
                true_atoms,
                exact_mass: monoisotopic_mass as f32,
            };

            unique_formulas.insert(formula_string, item);
        }

        // Extract just the unique training items
        let dataset: Vec<FormulaTrainingItem> = unique_formulas.into_values().collect();
        println!("Generated {} unique formula training items.", dataset.len());

        // Save the dataset to a JSON file so the DataLoader can consume it
        // Note: You will need `serde_json` in your Cargo.toml dependencies
        let output_file = File::create("test_files/formula_training_data.json")
            .expect("Failed to create output file");
        serde_json::to_writer_pretty(output_file, &dataset).expect("Failed to write JSON dataset");
    }

    /// Helper function to parse a standard chemical formula string (e.g., "C6H12O6")
    /// into the exactly ordered [C, H, N, O, P, S] array.
    ///
    /// Note: If your `molecules` crate has a built-in method to get element counts
    /// (like `formula.get_count(Element::C)`), you can replace this string parser with that!
    fn parse_formula_to_array(formula: &str) -> [f32; 6] {
        let mut counts = [0.0; 6];
        let target_elements = ["C", "H", "N", "O", "P", "S"];

        let mut chars = formula.chars().peekable();

        while let Some(c) = chars.next() {
            if c.is_ascii_uppercase() {
                let mut elem = c.to_string();

                // Check for two-letter elements (e.g., "Cl", "Na")
                if let Some(&next_c) = chars.peek() {
                    if next_c.is_ascii_lowercase() {
                        elem.push(chars.next().unwrap());
                    }
                }

                // Read the number following the element
                let mut count_str = String::new();
                while let Some(&next_c) = chars.peek() {
                    if next_c.is_ascii_digit() {
                        count_str.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                let count = if count_str.is_empty() {
                    1.0
                } else {
                    count_str.parse::<f32>().unwrap_or(1.0)
                };

                // If the element is one of our target ones, slot it into the array
                if let Some(idx) = target_elements.iter().position(|&e| e == elem) {
                    counts[idx] = count;
                }
            }
        }

        counts
    }

    #[test]
    fn run_full_training_and_inference() {
        // 1. Setup the Backend and Device (CPU for testing)
        type Backend = NdArray;
        type MyAutodiffBackend = Autodiff<Backend>;
        let device = NdArrayDevice::Cpu;

        let data_path = "test_files/formula_training_data.json";
        let artifact_dir = "/tmp/formula_predictor";

        // IMPORTANT: Ensure the data generation test has been run first!
        assert!(
            std::path::Path::new(data_path).exists(),
            "Training data not found! Please run `generate_sum_formula_data` first."
        );

        println!("=== Starting Model Training ===");

        // 2. Build Dataloaders
        // For testing purposes, we'll use the same dataset for both train and valid.
        // In a real scenario, split these into two separate JSON files.
        let train_dl = build_dataloader::<MyAutodiffBackend>(data_path, 32);
        let valid_dl = build_dataloader::<Backend>(data_path, 32);

        // 3. Train the Model
        // This will save the trained weights to `/tmp/formula_predictor/model`
        train_formula_predictor::<MyAutodiffBackend>(device.clone(), train_dl, valid_dl);

        println!("\n=== Starting Inference Tests ===");

        // 4. Define some unseen test SMILES
        // Aspirin, Benzene, and Acetic Acid
        let test_smiles = vec![
            "O=C(C)Oc1ccccc1C(=O)O", // C9H8O4
            "c1ccccc1",              // C6H6
            "CC(=O)O",               // C2H4O2
        ];

        for smiles in test_smiles {
            // Convert SMILES to our 3-peak feature format
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            let formula = mol.molecular_formula();
            let monoisotopic_mass = mol.monoisotopic_mass();
            let pattern = formula.isotopic_pattern(3, 1);

            let peaks: Vec<Peak> = pattern
                .into_iter()
                .map(|(mz, intensity)| Peak {
                    mz: mz as f32,
                    intensity: intensity as f32,
                })
                .collect();

            println!("\n--- Testing SMILES: {} ---", smiles);
            println!("True Formula: {}", formula.to_string());

            // 5. Run Inference!
            // Notice we use the inner `Backend` here, as inference doesn't need Autodiff
            let _predicted_atoms =
                predict_formula::<Backend>(artifact_dir, peaks, monoisotopic_mass as f32, &device);
        }
    }
}
