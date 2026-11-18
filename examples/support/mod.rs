//! Input checks shared by the examples (a module, not an example itself).
//!
//! The examples are benchmarks and research tools that read public datasets or
//! models trained by other examples. None of these files are in the repository,
//! so a missing input ends with a message naming it, not with a panic.
#![allow(dead_code)]

use std::fmt::Display;
use std::path::Path;

pub const GNPS_EVAL: &str = "100 GNPS library spectra with known formulas and structures, an evaluation subset of the GNPS library (https://external.gnps2.org/gnpslibrary)";
pub const MONA: &str =
    "MoNA LC-MS/MS positive-mode export in MSP format, from https://massbank.us/downloads";
pub const MSNLIB: &str = "a large annotated MS/MS library MGF (the original runs used 600,566 MSnLib spectra, doi:10.5281/zenodo.21105617)";
pub const MODEL: &str =
    "fingerprint model; train it with `cargo run --release --example train_fingerprint_model`";
pub const STRUCTURE_DB: &str =
    "structure database; build it with `cargo run --release --example build_preindexed_database`";
pub const ASTRAL: &str = "directory of Astral metabolomics MGF runs";
pub const PXD000561: &str =
    "directory of PXD000561 (PRIDE) pilot MGF files with overall_spectrum_annotations.csv";

/// Print `message` and exit with status 2, without a panic backtrace.
pub fn fail(message: impl Display) -> ! {
    eprintln!("{message}");
    std::process::exit(2)
}

/// Exit listing every input that does not exist; each entry is (path, what it is).
pub fn require(example: &str, inputs: &[(&str, &str)]) {
    let missing: Vec<_> = inputs
        .iter()
        .filter(|(p, _)| !Path::new(p).exists())
        .collect();
    if missing.is_empty() {
        return;
    }
    let mut message = format!("{example} needs inputs that are not in the repository:\n");
    for (path, what) in missing {
        message += &format!("  {path}\n      {what}\n");
    }
    fail(message.trim_end())
}

/// Positional argument `n` (1-based), or `default` when it is absent.
pub fn arg_or(n: usize, default: &str) -> String {
    std::env::args()
        .nth(n)
        .unwrap_or_else(|| default.to_string())
}
