//! Builder tool for creating pre-indexed binary `.oxdb` databases from MoNA and PubChem.
//!
//! Indexes molecular structures, pre-computes 2,214-bit bit-packed fingerprints,
//! and writes them to high-speed binary checkpoint files for sub-millisecond loading.

use oxidems_lib::sirius::fingerprint::{
    DatabaseScope, FingerprintPredictor, FingerprintSearcher, SampleContext, StructureDatabase,
};
use std::path::Path;
use std::time::Instant;

/// A file in the directory named by OXIDEMS_MOLECULES_DIR (PubChem CID-SMILES,
/// COCONUT4MetFrag.csv, HMDB4.csv).
fn molecules_file(name: &str) -> String {
    let dir = std::env::var("OXIDEMS_MOLECULES_DIR").unwrap_or_else(|_| {
        eprintln!("set OXIDEMS_MOLECULES_DIR to the directory holding {name}");
        std::process::exit(2)
    });
    std::path::Path::new(&dir).join(name).to_string_lossy().into_owned()
}

mod support;

fn main() {
    println!("=========================================================================");
    println!("       OxideMS Pre-Indexed Binary Database (.oxdb) Builder Tool          ");
    println!("=========================================================================");

    // Stored fingerprints only mean something in the layout they were computed in, and
    // the Morgan vocabulary is part of that layout. Take it from the model that will
    // query these databases rather than defaulting to plain folding.
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/sirius_fingerprints.oxfp".to_string());
    support::require("build_preindexed_database", &[(&model_path, support::MODEL)]);
    let predictor = FingerprintPredictor::load_from_file(&model_path)
        .expect("Trained model required to fix the fingerprint layout");
    let layout = FingerprintSearcher::for_predictor(&predictor).dictionary().clone();
    println!(
        "\nFingerprint layout from {}: {} dedicated Morgan vocabulary slots.",
        model_path,
        layout.morgan_vocabulary.ids().len()
    );

    // 1. Build & Save Comprehensive BioMetabolites Database (MoNA + COCONUT + HMDB)
    let bio_oxdb_path = "models/biometabolites.oxdb";
    let mut bio_db =
        StructureDatabase::new(DatabaseScope::BioMetabolites).with_dictionary(layout.clone());

    let mona_msp = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp";
    println!("\n[1/3] Ingesting MoNA Reference Standards from {}...", mona_msp);
    if Path::new(mona_msp).exists() {
        let t_start = Instant::now();
        let count = bio_db.ingest_mona_msp(mona_msp, usize::MAX).expect("Failed to ingest MoNA");
        println!(
            "  Indexed {} MoNA structures in {:.2}s (Total Pool: {} across {} formulas)",
            count,
            t_start.elapsed().as_secs_f64(),
            bio_db.total_structures,
            bio_db.entries_by_formula.len()
        );
    }

    let coconut_csv_owned = molecules_file("COCONUT4MetFrag.csv");
    let coconut_csv = coconut_csv_owned.as_str();
    println!("\n[2/3] Ingesting COCONUT Natural Products from {} (412k compounds)...", coconut_csv);
    if Path::new(coconut_csv).exists() {
        let t_start = Instant::now();
        let count = bio_db
            .ingest_structure_csv(coconut_csv, usize::MAX, SampleContext::Plant)
            .expect("Failed to ingest COCONUT");
        println!(
            "  Indexed {} COCONUT structures in {:.2}s (Total Pool: {} across {} formulas)",
            count,
            t_start.elapsed().as_secs_f64(),
            bio_db.total_structures,
            bio_db.entries_by_formula.len()
        );
    }

    let hmdb_csv_owned = molecules_file("HMDB4.csv");
    let hmdb_csv = hmdb_csv_owned.as_str();
    println!("\n[3/3] Ingesting HMDB Human Metabolites from {} (114k compounds)...", hmdb_csv);
    if Path::new(hmdb_csv).exists() {
        let t_start = Instant::now();
        let count = bio_db
            .ingest_structure_csv(hmdb_csv, usize::MAX, SampleContext::Human)
            .expect("Failed to ingest HMDB");
        println!(
            "  Indexed {} HMDB structures in {:.2}s (Total Pool: {} across {} formulas)",
            count,
            t_start.elapsed().as_secs_f64(),
            bio_db.total_structures,
            bio_db.entries_by_formula.len()
        );
    }

    println!("\nSerializing Comprehensive BioMetabolites to compact binary file: {} ...", bio_oxdb_path);
    let t_save = Instant::now();
    bio_db.save_to_file(bio_oxdb_path).expect("Failed to save bio .oxdb");
    let file_size_mb = std::fs::metadata(bio_oxdb_path)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    println!(
        "  Saved {} ({:.2} MB) in {:.2} ms",
        bio_oxdb_path,
        file_size_mb,
        t_save.elapsed().as_secs_f64() * 1000.0
    );

    // Test fast loading
    println!("  Testing binary reload speed...");
    let t_load = Instant::now();
    let loaded_db = StructureDatabase::load_from_file(bio_oxdb_path).expect("Failed to load .oxdb");
    println!(
        "  Loaded {} biological structures across {} formulas in {:.2} ms! (Instant RAM Lookups ready)",
        loaded_db.total_structures,
        loaded_db.entries_by_formula.len(),
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    // 2. Build & Save PubChem Candidate Database
    let pubchem_path_owned = molecules_file("CID-SMILES");
    let pubchem_path = pubchem_path_owned.as_str();
    let pubchem_oxdb_path = "models/pubchem_subset.oxdb";

    println!("\n[2/2] Building PubChem Candidate Database from {} (100,000 molecules with Rayon)...", pubchem_path);
    if Path::new(pubchem_path).exists() {
        let t_start = Instant::now();
        let mut pc_db =
            StructureDatabase::new(DatabaseScope::PubChemBio).with_dictionary(layout.clone());
        let count = pc_db
            .ingest_smiles_tsv(pubchem_path, 100_000, SampleContext::AllChemicals)
            .expect("Failed to ingest PubChem");
        println!(
            "  Indexed {} PubChem structures across {} unique formulas in {:.2}s",
            count,
            pc_db.entries_by_formula.len(),
            t_start.elapsed().as_secs_f64()
        );

        println!("  Serializing to compact binary file: {} ...", pubchem_oxdb_path);
        let t_save = Instant::now();
        pc_db.save_to_file(pubchem_oxdb_path).expect("Failed to save PubChem .oxdb");
        let file_size_mb = std::fs::metadata(pubchem_oxdb_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);
        println!(
            "  Saved {} ({:.2} MB) in {:.2} ms",
            pubchem_oxdb_path,
            file_size_mb,
            t_save.elapsed().as_secs_f64() * 1000.0
        );

        // Test fast loading
        println!("  Testing PubChem binary reload speed...");
        let t_load = Instant::now();
        let loaded_pc_db =
            StructureDatabase::load_from_file(pubchem_oxdb_path).expect("Failed to load PubChem .oxdb");
        println!(
            "  Loaded {} structures across {} formulas in {:.2} ms! (Sub-millisecond access ready)",
            loaded_pc_db.total_structures,
            loaded_pc_db.entries_by_formula.len(),
            t_load.elapsed().as_secs_f64() * 1000.0
        );
    } else {
        println!("  File {} not found, skipping.", pubchem_path);
    }

    println!("\n=========================================================================");
    println!("Pre-indexing complete! All databases are ready for instant mmap/binary loading.");
    println!("=========================================================================");
}
