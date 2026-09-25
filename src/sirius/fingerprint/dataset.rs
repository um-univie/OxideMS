//! Dataset partitioning and cross-library disjointness for fingerprint training.
//!
//! A fingerprint model is only as trustworthy as the evidence that it has not seen its
//! own test data. Two kinds of leakage matter here, and they need different defences:
//!
//! - **Within a library.** The same compound is measured many times at different
//!   collision energies, and isomers of it are measured too. Splitting spectra at
//!   random puts near-copies of a test spectrum in the training set. The unit of
//!   splitting therefore has to be the molecular formula, not the spectrum: every
//!   spectrum of every isomer of a formula lands in the same partition.
//!
//! - **Across libraries.** MoNA and GNPS both aggregate public submissions, so the
//!   "independent" library shares both compounds and, in places, literally the same
//!   deposited spectra. Excluding by name or by raw SMILES string misses these,
//!   because the same molecule is written differently by different depositors.
//!
//! The module answers both with content-derived keys, and with a [`DatasetManifest`]
//! that a training run writes and a later benchmark filters against — so disjointness
//! is checked against what was actually trained on, not against a re-derivation of it
//! that may not match.
//!
//! # Determinism
//!
//! [`assign_split`] hashes the group key rather than shuffling a list, so a formula
//! lands in the same partition regardless of how many samples were collected, what
//! order the parallel extractor produced them in, or which run this is. Splits that
//! move between runs cannot be compared between runs.

use crate::spectrum::Peak;
use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Which partition a group of spectra belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// Fitted on.
    Train,
    /// Used for early stopping, checkpoint selection, and per-bit reliability.
    Validation,
    /// Touched only to report final numbers.
    Test,
}

/// Fraction of groups held out of training.
#[derive(Debug, Clone, Copy)]
pub struct SplitRatios {
    /// Fraction assigned to [`Split::Validation`].
    pub validation: f32,
    /// Fraction assigned to [`Split::Test`].
    pub test: f32,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            validation: 0.10,
            test: 0.10,
        }
    }
}

/// Assigns a group of spectra to a partition, deterministically and by content.
///
/// `group_key` must be the same for every spectrum that has to stay together — the
/// molecular formula, so that isomers and replicate measurements cannot straddle the
/// boundary. `seed` selects an independent partitioning of the same data.
pub fn assign_split(group_key: &str, ratios: SplitRatios, seed: u64) -> Split {
    // Position of this group in [0, 1), stable forever: a fixed hash of the key rather
    // than its index in a shuffled list, which would move whenever the dataset does.
    let h = fnv1a64(group_key.as_bytes()) ^ fnv1a64(&seed.to_le_bytes());
    let pos = (h >> 11) as f64 / (1u64 << 53) as f64;

    let v = ratios.validation.clamp(0.0, 1.0) as f64;
    let t = ratios.test.clamp(0.0, 1.0) as f64;

    if pos < v {
        Split::Validation
    } else if pos < v + t {
        Split::Test
    } else {
        Split::Train
    }
}

/// A structure identity that survives being written by a different depositor.
///
/// Canonicalizes the SMILES, drops stereochemistry, and keeps only the largest
/// fragment. The result is roughly the first block of an InChIKey: the 2D skeleton.
///
/// Each of those steps deliberately makes the key *coarser*, because its job is to
/// exclude, and a key that is too fine silently lets a training compound back into a
/// held-out benchmark. Two stereoisomers, or a free base and its hydrochloride salt,
/// are treated as the same structure here.
///
/// Returns `None` for SMILES that cannot be turned into a key.
///
/// Library exports contain malformed and chemically impossible annotations, and the
/// canonicalizer panics on some of them rather than returning an error. A whole scan
/// must not die on one bad record, so the panic is caught and reported as `None` like
/// any other parse failure.
pub fn structure_key(smiles: &str) -> Option<String> {
    // Stereochemistry is removed *before* canonicalization, not after. Canonical atom
    // ordering is stereo-aware, so canonicalizing first and then deleting the markers
    // leaves two spellings of one molecule in different atom orders, and therefore
    // different keys for the same structure. Fused polycyclics show it plainly: two
    // database entries for testosterone differing only in `@` produced different keys
    // under the old order, and so were counted as distinct retrieval candidates.
    let flat: String = smiles
        .chars()
        .filter(|c| !matches!(c, '@' | '/' | '\\'))
        .collect();

    let canonical = std::panic::catch_unwind(|| canonicalize_and_sort_smiles(&flat, true))
        .ok()?
        .ok()?;

    // Salts and counter-ions are separate fragments; the annotated compound is the
    // largest one.
    let key = canonical
        .split('.')
        .max_by_key(|frag| frag.len())
        .unwrap_or(&canonical)
        .to_string();

    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// A content key identifying one measured spectrum.
///
/// Derived from the precursor and the peak m/z list rounded to a millidalton, so the
/// same deposited spectrum appearing in two libraries produces the same key.
/// Intensities are excluded because libraries renormalize them.
///
/// This catches spectra that were literally re-deposited. It does *not* catch a
/// re-processed copy whose peak list was filtered differently — [`structure_key`]
/// exclusion is what covers that case, and the two are meant to be used together.
pub fn spectrum_key(precursor_mz: f64, peaks: &[Peak]) -> u64 {
    let mut mzs: Vec<i64> = peaks.iter().map(|p| (p.mz * 1000.0).round() as i64).collect();
    mzs.sort_unstable();
    mzs.dedup();

    let mut h = fnv1a64(&((precursor_mz * 1000.0).round() as i64).to_le_bytes());
    for mz in mzs {
        h ^= fnv1a64(&mz.to_le_bytes());
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The structures and spectra a training run consumed.
///
/// Written next to the model weights so that a later benchmark can prove it is
/// disjoint from training without having to reconstruct the training set — which it
/// generally cannot do, since that would mean reproducing every filtering decision the
/// training run made.
///
/// It records *all three* partitions. Validation and test spectra are held out of
/// fitting, but they still steered early stopping and reliability measurement, so an
/// independent benchmark must avoid them too.
#[derive(Debug, Default, Clone)]
pub struct DatasetManifest {
    /// [`structure_key`] of every compound in the dataset.
    pub structure_keys: HashSet<String>,
    /// [`spectrum_key`] of every spectrum in the dataset.
    pub spectrum_keys: HashSet<u64>,
}

/// First line of a manifest file, with its format version.
const MANIFEST_HEADER: &str = "OXFP-MANIFEST 1";

impl DatasetManifest {
    /// Creates an empty manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one spectrum and its annotated structure.
    ///
    /// A SMILES that will not parse contributes only its spectrum key; that is worth
    /// keeping, since an unparseable annotation is exactly the case where the
    /// structure-level check cannot help.
    pub fn insert(&mut self, smiles: &str, precursor_mz: f64, peaks: &[Peak]) {
        if let Some(key) = structure_key(smiles) {
            self.structure_keys.insert(key);
        }
        self.spectrum_keys.insert(spectrum_key(precursor_mz, peaks));
    }

    /// Whether this dataset contains the given compound, under [`structure_key`].
    pub fn contains_structure(&self, smiles: &str) -> bool {
        structure_key(smiles).is_some_and(|k| self.structure_keys.contains(&k))
    }

    /// Whether this dataset contains the given spectrum, under [`spectrum_key`].
    pub fn contains_spectrum(&self, precursor_mz: f64, peaks: &[Peak]) -> bool {
        self.spectrum_keys.contains(&spectrum_key(precursor_mz, peaks))
    }

    /// Whether a candidate benchmark record overlaps this dataset in either sense.
    pub fn overlaps(&self, smiles: &str, precursor_mz: f64, peaks: &[Peak]) -> bool {
        self.contains_structure(smiles) || self.contains_spectrum(precursor_mz, peaks)
    }

    /// Writes the manifest as text, sorted so that two runs over the same data produce
    /// identical files.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "{MANIFEST_HEADER}")?;

        let mut structures: Vec<&String> = self.structure_keys.iter().collect();
        structures.sort();
        writeln!(w, "structures {}", structures.len())?;
        for key in structures {
            writeln!(w, "{key}")?;
        }

        let mut spectra: Vec<&u64> = self.spectrum_keys.iter().collect();
        spectra.sort();
        writeln!(w, "spectra {}", spectra.len())?;
        for key in spectra {
            writeln!(w, "{key:016x}")?;
        }
        Ok(())
    }

    /// Reads a manifest written by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        let mut lines = reader.lines();

        let header = lines.next().transpose()?.unwrap_or_default();
        if header.trim() != MANIFEST_HEADER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a manifest file: expected {MANIFEST_HEADER:?}, found {header:?}"),
            ));
        }

        let mut manifest = Self::new();
        let mut remaining = 0usize;
        let mut in_spectra = false;

        for line in lines {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(n) = line.strip_prefix("structures ") {
                remaining = n.trim().parse().unwrap_or(0);
                in_spectra = false;
                continue;
            }
            if let Some(n) = line.strip_prefix("spectra ") {
                remaining = n.trim().parse().unwrap_or(0);
                in_spectra = true;
                continue;
            }

            if remaining == 0 {
                continue;
            }
            remaining -= 1;

            if in_spectra {
                if let Ok(k) = u64::from_str_radix(line, 16) {
                    manifest.spectrum_keys.insert(k);
                }
            } else {
                manifest.structure_keys.insert(line.to_string());
            }
        }

        Ok(manifest)
    }
}

/// FNV-1a, spelled out rather than taken from a dependency.
///
/// Split assignment has to give the same answer years from now, so the hash cannot be
/// something a dependency bump is free to change.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaks(mzs: &[f64]) -> Vec<Peak> {
        mzs.iter()
            .map(|&mz| Peak {
                mz,
                intensity: 1.0,
            })
            .collect()
    }

    #[test]
    fn test_split_is_stable_and_group_wise() {
        let ratios = SplitRatios::default();
        // The same formula must land in the same partition every time it is asked
        // about, which is what keeps isomers from straddling the boundary.
        for key in ["C6H12O6", "C13H10Cl2FN5O3S", "C10H12O4"] {
            let first = assign_split(key, ratios, 42);
            for _ in 0..10 {
                assert_eq!(assign_split(key, ratios, 42), first, "{key} moved");
            }
        }
    }

    #[test]
    fn test_split_respects_ratios() {
        let ratios = SplitRatios {
            validation: 0.10,
            test: 0.10,
        };
        let n = 20_000;
        let mut counts = (0usize, 0usize, 0usize);
        for i in 0..n {
            match assign_split(&format!("C{i}H{}O2", i % 40), ratios, 42) {
                Split::Train => counts.0 += 1,
                Split::Validation => counts.1 += 1,
                Split::Test => counts.2 += 1,
            }
        }
        let frac = |c: usize| c as f64 / n as f64;
        assert!((frac(counts.1) - 0.10).abs() < 0.02, "validation {}", frac(counts.1));
        assert!((frac(counts.2) - 0.10).abs() < 0.02, "test {}", frac(counts.2));
        assert!((frac(counts.0) - 0.80).abs() < 0.03, "train {}", frac(counts.0));
    }

    #[test]
    fn test_different_seeds_give_different_partitions() {
        let ratios = SplitRatios::default();
        let keys: Vec<String> = (0..500).map(|i| format!("C{i}H{i}O")).collect();
        let moved = keys
            .iter()
            .filter(|k| assign_split(k, ratios, 1) != assign_split(k, ratios, 2))
            .count();
        assert!(moved > 0, "seed had no effect on the partitioning");
    }

    #[test]
    fn test_structure_key_ignores_writing_style() {
        // The same molecule, written by two depositors.
        let a = structure_key("C1=CC=CC=C1O").expect("phenol");
        let b = structure_key("Oc1ccccc1").expect("phenol, aromatic form");
        assert_eq!(a, b, "kekulized and aromatic SMILES must agree");

        // Stereochemistry is deliberately dropped: an enantiomer is not independent
        // evidence that the model generalizes.
        let l = structure_key("C[C@H](N)C(=O)O").expect("L-alanine");
        let d = structure_key("C[C@@H](N)C(=O)O").expect("D-alanine");
        assert_eq!(l, d, "stereoisomers must share a key");

        // And a salt resolves to its parent compound.
        let free = structure_key("CNC").expect("dimethylamine");
        let salt = structure_key("CNC.Cl").expect("dimethylamine hydrochloride");
        assert_eq!(free, salt, "counter-ion must not create a new structure");

        // Genuinely different molecules must not collide.
        assert_ne!(a, structure_key("CCO").expect("ethanol"));
    }

    #[test]
    fn test_structure_key_merges_fused_ring_stereo_spellings() {
        // Real database entries the steroid benchmark surfaced as distinct candidates
        // carrying bit-identical fingerprints: one molecule, written with and without
        // stereochemistry. Canonical atom ordering depends on stereochemistry, so
        // stripping the markers afterwards left these in different orders.
        for (with_stereo, without) in [
            (
                "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
                "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
            ),
            (
                "[H][C@@]12CCC(=O)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CC[C@@]2([H])CC(=O)CC[C@]12C",
                "O=C1CCC2(C)C(C1)CCC3C4CCC(=O)C4(C)CCC32",
            ),
        ] {
            let a = structure_key(with_stereo).expect("parses");
            let b = structure_key(without).expect("parses");
            assert_eq!(a, b, "{with_stereo} and {without} are the same 2D structure");
        }
    }

    #[test]
    fn test_structure_key_survives_unparseable_input() {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        // Junk yields no key.
        for bad in ["", "not a smiles", "["] {
            assert!(structure_key(bad).is_none(), "expected None for {bad:?}");
        }

        // These are the ones that matter: input the parser accepts leniently, and
        // input that makes it panic outright. Either way the call must return, because
        // one bad record among a hundred thousand may not end the scan. What comes
        // back is the parser's business, not this function's.
        for odd in ["C(C)(C)(C)(C)(C)C", "C1CC", "[C@](F)(Cl)Br", "c1ccc1"] {
            let _ = structure_key(odd);
        }

        std::panic::set_hook(previous);
    }

    #[test]
    fn test_spectrum_key_identifies_the_same_measurement() {
        let p = peaks(&[105.0699, 107.0851, 135.0800]);
        // Peak order and intensity scaling differ between exports of one spectrum.
        let reordered = peaks(&[135.0800, 105.0699, 107.0851]);
        let rescaled: Vec<Peak> = p
            .iter()
            .map(|q| Peak {
                mz: q.mz,
                intensity: q.intensity * 1000.0,
            })
            .collect();

        assert_eq!(spectrum_key(323.258, &p), spectrum_key(323.258, &reordered));
        assert_eq!(spectrum_key(323.258, &p), spectrum_key(323.258, &rescaled));

        // A different precursor, or a different peak, is a different spectrum.
        assert_ne!(spectrum_key(323.258, &p), spectrum_key(324.258, &p));
        assert_ne!(
            spectrum_key(323.258, &p),
            spectrum_key(323.258, &peaks(&[105.0699, 107.0851, 136.0800]))
        );
    }

    #[test]
    fn test_manifest_round_trip_and_exclusion() {
        let p = peaks(&[105.0699, 107.0851, 135.0800]);
        let mut manifest = DatasetManifest::new();
        manifest.insert("Oc1ccccc1", 95.05, &p);

        // Same compound written differently, and a different compound.
        assert!(manifest.contains_structure("C1=CC=CC=C1O"));
        assert!(!manifest.contains_structure("CCO"));
        assert!(manifest.contains_spectrum(95.05, &p));
        assert!(!manifest.contains_spectrum(96.05, &p));

        // A benchmark record overlaps if either check fires.
        assert!(manifest.overlaps("CCO", 95.05, &p), "spectrum reuse must be caught");
        assert!(
            manifest.overlaps("C1=CC=CC=C1O", 500.0, &peaks(&[1.0])),
            "structure reuse must be caught"
        );
        assert!(!manifest.overlaps("CCO", 500.0, &peaks(&[1.0])));

        let path = std::env::temp_dir().join("oxidems_manifest_roundtrip.txt");
        manifest.save(&path).unwrap();
        let loaded = DatasetManifest::load(&path).unwrap();
        assert_eq!(loaded.structure_keys, manifest.structure_keys);
        assert_eq!(loaded.spectrum_keys, manifest.spectrum_keys);
        std::fs::remove_file(&path).ok();
    }
}
