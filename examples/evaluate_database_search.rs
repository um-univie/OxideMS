//! Evaluate Molecular Fingerprint Database Structure Identification.
//!
//! Demonstrates the full CSI:FingerID query pipeline on authentic metabolites:
//! 1. Ingests a database of chemical structures (SMILES) and indexes them by molecular formula.
//! 2. For an unknown experimental MS/MS spectrum, solves the MAP fragmentation tree.
//! 3. Predicts the 2,214-bit molecular fingerprint vector via the trained neural network (`.oxfp`).
//! 4. Queries the structural database for all isomer candidates with matching molecular formula.
//! 5. Scores and ranks candidate structures using ModPlatt Bayesian log-likelihood scoring.
//! 6. Evaluates retrieval accuracy (Rank #1, Top-3, Top-5, and predicted-vs-true Tanimoto).

use oxidems_lib::sirius::fingerprint::{
    BinaryFingerprint, DatasetManifest, FingerprintDictionary, FingerprintPredictor,
    CandidateScore, FingerprintSearcher, MACCS_BITS, PredictedFingerprint, SampleContext,
    StructureDatabase, apply_platt, Split, SplitRatios, StructureCandidate,
    TreeFeatureExtractor, assign_split, spectrum_key,
    structure_key, TOTAL_BITS,
};
use oxidems_lib::sirius::{
    Adduct, ChemicalRules, Formula, FragmentationTreeBuilder, SiriusClusteringWorkflow,
    TreeSolverConfig,
};
use oxidane::molecular_formula::MolecularFormula;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::molecular_graph::MolecularGraph;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use rayon::prelude::*;
use std::str::FromStr;
use std::time::Instant;

struct BenchmarkEntry {
    name: String,
    smiles: String,
    formula: Formula,
    precursor_mz: f64,
    charge: i8,
    peaks: Vec<Peak>,
}

/// Ingests the MoNA library.
///
/// Every structure goes into the candidate database — a real search ranks a query
/// against everything known, including compounds the model was trained on. Only the
/// *queries* are restricted, to the partition named by `query_split`, so that no
/// spectrum being scored was fitted on or used to pick the checkpoint. `split_seed`
/// must match the one the model was trained with, or the partitions do not line up.
fn parse_mona_records(
    path: &Path,
    layout: &FingerprintDictionary,
    max_total_records: usize,
    max_query_records: usize,
    query_split: Option<Split>,
    split_seed: u64,
) -> (Vec<BenchmarkEntry>, HashMap<Formula, Vec<(String, String, String, BinaryFingerprint)>>) {
    let file = File::open(path).expect("Could not open MSP file");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut benchmark_spectra = Vec::new();
    let mut db_by_formula: HashMap<Formula, Vec<(String, String, String, BinaryFingerprint)>> = HashMap::new();
    let mut seen_smiles = std::collections::HashSet::new();

    let mut current_name = String::new();
    let mut current_formula = String::new();
    let mut current_smiles = String::new();
    let mut current_mz = 0.0f64;
    let mut current_charge = 1i8;
    let mut current_peaks = Vec::new();
    let mut total_parsed = 0usize;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        if line.is_empty() {
            if !current_smiles.is_empty() && !current_formula.is_empty() && !current_peaks.is_empty() && current_mz > 0.0 {
                total_parsed += 1;
                if let Ok(formula) = Formula::from_str(&current_formula) {
                    if let Some(fp_bits) = layout.ground_truth_fingerprint(&current_smiles) {
                        let fp = BinaryFingerprint::from_bits(fp_bits);

                        // One entry per distinct structure, not per distinct SMILES
                        // string. The same molecule is deposited under several
                        // spellings — stereoisomers, salts, tautomer choices — and
                        // MACCS and folded Morgan see none of those differences, so
                        // string-level dedup fills the candidate list with copies that
                        // are unrankable by construction.
                        let key = structure_key(&current_smiles)
                            .unwrap_or_else(|| current_smiles.clone());
                        if seen_smiles.insert(key.clone()) {
                            db_by_formula.entry(formula).or_default().push((
                                current_name.clone(),
                                current_smiles.clone(),
                                key,
                                fp,
                            ));
                        }

                        // Store as query benchmark spectrum
                        let in_query_split = query_split.is_none_or(|want| {
                            assign_split(&current_formula, SplitRatios::default(), split_seed) == want
                        });
                        if in_query_split && benchmark_spectra.len() < max_query_records && current_peaks.len() >= 4 && current_peaks.len() <= 120 {
                            benchmark_spectra.push(BenchmarkEntry {
                                name: current_name.clone(),
                                smiles: current_smiles.clone(),
                                formula,
                                precursor_mz: current_mz,
                                charge: current_charge,
                                peaks: current_peaks.clone(),
                            });
                        }
                    }
                }

                if total_parsed >= max_total_records {
                    break;
                }
            }
            current_mz = 0.0;
            current_charge = 1;
            current_name.clear();
            current_formula.clear();
            current_smiles.clear();
            current_peaks.clear();
            continue;
        }

        if let Some(val) = line.strip_prefix("Name:") {
            current_name = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("Formula:") {
            current_formula = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("PrecursorMZ:") {
            current_mz = val.trim().parse::<f64>().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("Ion_mode:") {
            if val.trim().starts_with('N') || val.trim().starts_with('-') {
                current_charge = -1;
            } else {
                current_charge = 1;
            }
        } else if line.starts_with("Comments:") {
            if let Some(idx) = line.find("SMILES=") {
                let rest = &line[idx + 7..];
                let end = rest.find('"').or_else(|| rest.find(' ')).unwrap_or(rest.len());
                current_smiles = rest[..end].trim().to_string();
            }
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz), Ok(intensity)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if mz > 0.0 && intensity > 0.0 {
                        current_peaks.push(Peak { mz, intensity });
                    }
                }
            }
        }
    }

    (benchmark_spectra, db_by_formula)
}

/// Ingests annotated MGF spectra as an independent retrieval benchmark.
///
/// GNPS is a separate library from the MoNA export the model is trained on, so it
/// tests the model on chemistry it was not fitted to — but the two libraries
/// aggregate overlapping public sources, so compound-level overlap still has to be
/// excluded explicitly (see `--exclude-smiles-msp`).
///
/// Unlike MoNA, GNPS records carry no formula field, so the formula is derived from
/// the annotated SMILES.
fn parse_gnps_records(
    path: &Path,
    layout: &FingerprintDictionary,
    max_total_records: usize,
    max_query_records: usize,
    exclude: &DatasetManifest,
) -> (Vec<BenchmarkEntry>, HashMap<Formula, Vec<(String, String, String, BinaryFingerprint)>>) {
    let file = File::open(path).expect("Could not open GNPS MGF file");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut benchmark_spectra = Vec::new();
    let mut db_by_formula: HashMap<Formula, Vec<(String, String, String, BinaryFingerprint)>> = HashMap::new();
    let mut seen_smiles = std::collections::HashSet::new();

    let mut name = String::new();
    let mut smiles = String::new();
    let mut mz = 0.0f64;
    let mut charge = 1i8;
    let mut quality = 0u8;
    let mut graded = false;
    let mut peaks: Vec<Peak> = Vec::new();
    let mut total_parsed = 0usize;
    let mut excluded_structure = 0usize;
    let mut excluded_spectrum = 0usize;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        if line == "BEGIN IONS" {
            name.clear();
            smiles.clear();
            mz = 0.0;
            charge = 1;
            quality = 0;
            graded = false;
            peaks.clear();
            continue;
        }

        if line == "END IONS" {
            // LIBRARYQUALITY 1 is the GNPS gold tier; lower tiers carry unreviewed
            // structure annotations, which would be scored as ground truth here. Files
            // that do not grade their entries at all — an instrument export, say —
            // carry no such field, and rejecting them for its absence would reject the
            // whole file.
            let usable = (quality == 1 || !graded)
                && !smiles.is_empty()
                && smiles != "N/A"
                && mz > 0.0
                && peaks.len() >= 4
                && peaks.len() <= 120;

            if usable {
                // Structure first: it is the broader of the two checks, and knowing
                // which one fired says whether the libraries share chemistry or share
                // literal deposits.
                if exclude.contains_structure(&smiles) {
                    excluded_structure += 1;
                } else if exclude.contains_spectrum(mz, &peaks) {
                    excluded_spectrum += 1;
                } else if let Some(formula) = formula_from_smiles(&smiles) {
                    if let Some(fp_bits) = layout.ground_truth_fingerprint(&smiles) {
                        total_parsed += 1;
                        let fp = BinaryFingerprint::from_bits(fp_bits);

                        let key =
                            structure_key(&smiles).unwrap_or_else(|| smiles.clone());
                        if seen_smiles.insert(key.clone()) {
                            db_by_formula.entry(formula).or_default().push((
                                name.clone(),
                                smiles.clone(),
                                key,
                                fp,
                            ));
                        }

                        if benchmark_spectra.len() < max_query_records {
                            benchmark_spectra.push(BenchmarkEntry {
                                name: name.clone(),
                                smiles: smiles.clone(),
                                formula,
                                precursor_mz: mz,
                                charge,
                                peaks: peaks.clone(),
                            });
                        }
                    }
                }
            }

            if total_parsed >= max_total_records {
                break;
            }
            continue;
        }

        // Field names differ between library exports and instrument software: GNPS
        // writes NAME and PEPMASS, MZmine writes COMPOUND_NAME and PRECURSOR_MZ.
        if let Some(val) = line
            .strip_prefix("NAME=")
            .or_else(|| line.strip_prefix("COMPOUND_NAME="))
        {
            name = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("SMILES=") {
            smiles = val.trim().to_string();
        } else if let Some(val) = line
            .strip_prefix("PEPMASS=")
            .or_else(|| line.strip_prefix("PRECURSOR_MZ="))
        {
            mz = val.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("LIBRARYQUALITY=") {
            quality = val.trim().parse().unwrap_or(0);
            graded = true;
        } else if let Some(val) = line.strip_prefix("IONMODE=") {
            charge = if val.trim().starts_with('N') { -1 } else { 1 };
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(p_mz), Ok(inten)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if p_mz > 0.0 && inten > 0.0 {
                        peaks.push(Peak { mz: p_mz, intensity: inten });
                    }
                }
            }
        }
    }

    if !exclude.structure_keys.is_empty() || !exclude.spectrum_keys.is_empty() {
        println!(
            "  Excluded {excluded_structure} spectra of structures the model was trained on, \
             and {excluded_spectrum} further spectra that are the same measurement re-deposited."
        );
    }

    (benchmark_spectra, db_by_formula)
}

/// Derives the neutral molecular formula of a SMILES string.
///
/// SMILES leaves hydrogens implicit, so they have to be materialized before counting
/// atoms or every formula comes out hydrogen-free.
fn formula_from_smiles(smiles: &str) -> Option<Formula> {
    let mut mol = MolecularGraph::from_smiles_single(smiles).ok()?;
    mol.add_hydrogens();
    Formula::from_str(&MolecularFormula::from_molecule(&mol).to_string()).ok()
}

/// Scans a whole spectral library and records what is in it, without building trees or
/// fingerprints.
///
/// This is how an exclusion set for a *different* model is built. A model whose own
/// training set was never recorded — anything trained before manifests existed — can
/// only be benchmarked fairly on compounds that appear nowhere in the library it was
/// drawn from, and that requires scanning the library in full rather than the prefix
/// any one training run happened to read.
fn build_manifest(path: &Path) -> DatasetManifest {
    let is_mgf = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mgf"));
    let file = File::open(path).expect("Could not open spectra file");
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    // Collected first, canonicalized after: structure_key parses and canonicalizes a
    // molecular graph per record, which dominates the cost at library scale.
    let mut records: Vec<(String, f64, Vec<Peak>)> = Vec::new();
    let mut smiles = String::new();
    let mut mz = 0.0f64;
    let mut peaks: Vec<Peak> = Vec::new();

    let mut flush = |smiles: &mut String, mz: &mut f64, peaks: &mut Vec<Peak>, out: &mut Vec<_>| {
        if !smiles.is_empty() && smiles != "N/A" && !peaks.is_empty() {
            out.push((std::mem::take(smiles), *mz, std::mem::take(peaks)));
        } else {
            smiles.clear();
            peaks.clear();
        }
        *mz = 0.0;
    };

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        // MSP separates records with a blank line; MGF brackets them.
        if (is_mgf && line == "END IONS") || (!is_mgf && line.is_empty()) {
            flush(&mut smiles, &mut mz, &mut peaks, &mut records);
            continue;
        }
        if is_mgf && line == "BEGIN IONS" {
            smiles.clear();
            peaks.clear();
            mz = 0.0;
            continue;
        }

        if let Some(val) = line.strip_prefix("SMILES=") {
            smiles = val.trim().to_string();
        } else if let Some(val) = line
            .strip_prefix("PEPMASS=")
            .or_else(|| line.strip_prefix("PRECURSOR_MZ="))
        {
            mz = val.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("PrecursorMZ:") {
            mz = val.trim().parse().unwrap_or(0.0);
        } else if line.starts_with("Comments:") {
            if let Some(idx) = line.find("SMILES=") {
                let rest = &line[idx + 7..];
                let end = rest.find('"').or_else(|| rest.find(' ')).unwrap_or(rest.len());
                smiles = rest[..end].trim().to_string();
            }
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(p_mz), Ok(inten)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if p_mz > 0.0 && inten > 0.0 {
                        peaks.push(Peak { mz: p_mz, intensity: inten });
                    }
                }
            }
        }
    }
    flush(&mut smiles, &mut mz, &mut peaks, &mut records);

    // Library exports carry malformed annotations, and the canonicalizer panics on
    // some of them. structure_key catches that and returns None; silencing the hook
    // for the duration keeps a hundred thousand records from printing backtraces.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let keyed: Vec<(Option<String>, u64)> = records
        .par_iter()
        .map(|(sm, mz, pk)| (structure_key(sm), spectrum_key(*mz, pk)))
        .collect();
    std::panic::set_hook(previous_hook);

    let unreadable = keyed.iter().filter(|(s, _)| s.is_none()).count();

    let mut manifest = DatasetManifest::new();
    for (structure, spectrum) in keyed {
        if let Some(k) = structure {
            manifest.structure_keys.insert(k);
        }
        manifest.spectrum_keys.insert(spectrum);
    }
    println!(
        "  Scanned {} records from {}: {} distinct structures, {} distinct spectra ({} annotations unreadable).",
        records.len(),
        path.display(),
        manifest.structure_keys.len(),
        manifest.spectrum_keys.len(),
        unreadable
    );
    manifest
}

/// One query's outcome, with the covariates a failure could plausibly depend on.
struct DiagRow {
    name: String,
    formula: String,
    precursor_mz: f64,
    n_peaks: usize,
    tree_nodes: usize,
    explained: f64,
    n_isomers: usize,
    rank: usize,
    /// Rank the true structure would get from a *perfect* prediction, i.e. ranking
    /// candidates by fingerprint similarity to the truth. It is 1 unless some other
    /// isomer has an identical fingerprint, so it measures what the 2,214-bit
    /// representation can distinguish at all, independently of the model.
    oracle_rank: usize,
    /// Candidates whose fingerprint is bit-identical to the true structure's.
    indistinguishable: usize,
    /// Predicted-vs-true Tanimoto for this query.
    pred_tanimoto: f64,
    /// Fingerprint similarity between the truth and whatever won. 1.0 when the
    /// truth won.
    winner_tanimoto: f64,
    /// True structure's score minus the winner's. Zero when the truth won,
    /// negative otherwise; a small gap is a near miss.
    score_margin: f64,
    true_bits: usize,
    /// Bits set in the fingerprint of whatever the scorer ranked first. Compared with
    /// `true_bits`, this is how a scorer's preference for large or small candidates
    /// shows up.
    winner_bits: usize,
    pred_bits: usize,
    /// Tanimoto of the thresholded prediction against the truth, at each of
    /// [`THRESHOLDS`]. Retrieval scores on probabilities, but the fingerprint the
    /// pipeline reports is thresholded, and 0.5 is only a default — whether it is the
    /// right default is measurable.
    tanimoto_at: Vec<f64>,
    /// Tanimoto restricted to the MACCS block and to the Morgan block, at 0.5. The two
    /// halves are different prediction problems — 166 curated keys against 2,048 folded
    /// hash buckets — and an average over both hides which one is failing.
    maccs_tanimoto: f64,
    morgan_tanimoto: f64,
}

/// Thresholds swept when reporting fingerprint quality.
const THRESHOLDS: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Tanimoto between a thresholded probability vector and a binary fingerprint, over a
/// slice of the bit range.
fn tanimoto_over(probs: &[f32], truth: &[bool], range: std::ops::Range<usize>, threshold: f32) -> f64 {
    let (mut inter, mut union) = (0usize, 0usize);
    for i in range {
        let p = probs[i] >= threshold;
        let t = truth[i];
        if p || t {
            union += 1;
        }
        if p && t {
            inter += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// Prints how good the predicted fingerprints are, before asking what retrieval did
/// with them.
///
/// Retrieval accuracy tracks fingerprint Tanimoto almost linearly, so a retrieval
/// failure is usually a prediction failure wearing a different hat. This block asks
/// which part of the prediction is weak: the threshold, the calibration, or one of the
/// two fingerprint blocks.
fn report_fingerprint_quality(rows: &[DiagRow], pos: (f64, usize), neg: (f64, usize)) {
    if rows.is_empty() {
        return;
    }
    let n = rows.len() as f64;

    println!("\n-------------------------------------------------------------------------");
    println!("                        Fingerprint Quality                              ");
    println!("-------------------------------------------------------------------------");

    println!("\n  Mean Tanimoto by binarization threshold:");
    let mut best = (0.0f64, 0.5f32);
    for (i, &t) in THRESHOLDS.iter().enumerate() {
        let mean: f64 = rows.iter().map(|r| r.tanimoto_at[i]).sum::<f64>() / n;
        if mean > best.0 {
            best = (mean, t);
        }
        let marker = if (t - 0.5).abs() < 1e-6 { "  <- current default" } else { "" };
        println!("    {t:.1} : {mean:.4}{marker}");
    }
    println!("    Best threshold {:.1} at Tanimoto {:.4}", best.1, best.0);

    let maccs: f64 = rows.iter().map(|r| r.maccs_tanimoto).sum::<f64>() / n;
    let morgan: f64 = rows.iter().map(|r| r.morgan_tanimoto).sum::<f64>() / n;
    println!("\n  By fingerprint block (threshold 0.5):");
    println!("    MACCS  [0, {MACCS_BITS}) : {maccs:.4}");
    println!("    Morgan [{MACCS_BITS}, {TOTAL_BITS}) : {morgan:.4}");

    let true_bits: f64 = rows.iter().map(|r| r.true_bits as f64).sum::<f64>() / n;
    let pred_bits: f64 = rows.iter().map(|r| r.pred_bits as f64).sum::<f64>() / n;
    println!("\n  Bits set per fingerprint: truth {true_bits:.1}, prediction {pred_bits:.1} ({:+.1}%)",
        (pred_bits / true_bits - 1.0) * 100.0);

    let mean_pos = pos.0 / pos.1.max(1) as f64;
    let mean_neg = neg.0 / neg.1.max(1) as f64;
    println!("\n  Calibration: mean probability on bits that are set {mean_pos:.4}, on bits that are not {mean_neg:.4} (separation {:.4})",
        mean_pos - mean_neg);
    println!("-------------------------------------------------------------------------");
}

/// Prints where the retrieval fails, rather than only how often.
///
/// Aggregate accuracy says a model is wrong; it does not say whether it was wrong
/// because the prediction was poor, because the spectrum was uninformative, or because
/// the candidates were indistinguishable no matter what was predicted. Each block
/// below separates one of those.
fn report_failure_modes(rows: &[DiagRow]) {
    if rows.is_empty() {
        return;
    }
    let n = rows.len() as f64;
    let hit = |r: &DiagRow| r.rank == 1;

    println!("\n-------------------------------------------------------------------------");
    println!("                          Failure Mode Breakdown                          ");
    println!("-------------------------------------------------------------------------");

    // 1. What the representation can distinguish at all.
    //
    // A perfect predictor still cannot separate candidates whose fingerprints are
    // bit-identical to the truth; it can only land on the right one by chance, with
    // probability 1/(1+twins). Scoring those ties as automatic losses instead — the
    // earlier definition — reported a ceiling *below* the accuracy actually realized
    // whenever ties were common, which is not a bound at all.
    let blocked = rows.iter().filter(|r| r.indistinguishable > 0).count();
    let ceiling: f64 = rows
        .iter()
        .map(|r| 1.0 / (1 + r.indistinguishable) as f64)
        .sum::<f64>()
        / n;
    let realized = rows.iter().filter(|r| hit(r)).count() as f64 / n;
    let within_ties = rows
        .iter()
        .filter(|r| r.rank <= 1 + r.indistinguishable)
        .count() as f64
        / n;
    let twins: Vec<usize> = rows.iter().map(|r| r.indistinguishable).collect();
    let median_twins = {
        let mut t = twins.clone();
        t.sort_unstable();
        t[t.len() / 2]
    };
    println!("\n  Representational ceiling (perfect prediction, ties broken at random):");
    println!("    Queries with an identical-fingerprint twin among the candidates: {blocked} ({:.1}%)", blocked as f64 / n * 100.0);
    println!("    Twins per query: median {median_twins}, max {}", twins.iter().max().copied().unwrap_or(0));
    println!("    Best achievable Top-1 with this 2,214-bit representation: {:.1}%", ceiling * 100.0);
    println!("    Realized Top-1:                                          {:.1}%", realized * 100.0);
    println!("    Gap attributable to prediction quality:                  {:.1} points", (ceiling - realized) * 100.0);
    println!("    Truth beat every distinguishable candidate:              {:.1}%", within_ties * 100.0);

    // 2. How badly it misses when it misses.
    let mut ranks: Vec<usize> = rows.iter().map(|r| r.rank).collect();
    ranks.sort_unstable();
    let median = ranks[ranks.len() / 2];
    let misses: Vec<&DiagRow> = rows.iter().filter(|r| !hit(r)).collect();
    println!("\n  Rank distribution: median {median}, worst {}", ranks[ranks.len() - 1]);
    if !misses.is_empty() {
        let m = misses.len() as f64;
        let rank2 = misses.iter().filter(|r| r.rank == 2).count();
        let near = misses.iter().filter(|r| r.score_margin > -5.0).count();
        let mean_winner_tan: f64 = misses.iter().map(|r| r.winner_tanimoto).sum::<f64>() / m;
        println!("    Of {} failures: {} land at rank 2 ({:.0}%), {} lose by a ModPlatt margin under 5 ({:.0}%)",
            misses.len(), rank2, rank2 as f64 / m * 100.0, near, near as f64 / m * 100.0);
        println!("    Mean fingerprint similarity between the truth and the winning decoy: {mean_winner_tan:.3}");
    }

    // 3. Which queries fail.
    binned(rows, "Structural isomers in the candidate set", &[2.0, 3.0, 5.0, 10.0, 20.0], |r| r.n_isomers as f64);
    binned(rows, "Predicted-vs-true fingerprint Tanimoto", &[0.2, 0.3, 0.4, 0.5], |r| r.pred_tanimoto);
    binned(rows, "Precursor m/z", &[250.0, 400.0, 600.0, 800.0], |r| r.precursor_mz);
    binned(rows, "Fragmentation tree nodes", &[3.0, 5.0, 8.0, 12.0], |r| r.tree_nodes as f64);
    binned(rows, "Fraction of spectral intensity explained by the tree", &[0.5, 0.7, 0.85, 0.95], |r| r.explained);
    println!("-------------------------------------------------------------------------");
}

/// Top-1 accuracy within buckets of one covariate, so a trend is visible rather than
/// an average that hides it.
fn binned(rows: &[DiagRow], label: &str, edges: &[f64], key: impl Fn(&DiagRow) -> f64) {
    println!("\n  Top-1 by {label}:");
    let mut lo = f64::NEG_INFINITY;
    for (i, &hi) in edges.iter().chain(std::iter::once(&f64::INFINITY)).enumerate() {
        let bucket: Vec<&DiagRow> = rows.iter().filter(|r| key(r) >= lo && key(r) < hi).collect();
        let range = if i == 0 {
            format!("< {:.6}", edges[0]).replace(".000000", "")
        } else if hi.is_infinite() {
            format!(">= {lo}")
        } else {
            format!("{lo} - {hi}")
        };
        if bucket.is_empty() {
            println!("    {range:>16} : (none)");
        } else {
            let acc = bucket.iter().filter(|r| r.rank == 1).count() as f64 / bucket.len() as f64;
            println!("    {range:>16} : {:5.1}%  (n = {})", acc * 100.0, bucket.len());
        }
        lo = hi;
    }
}

mod support;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "models/sirius_fingerprints.oxfp".to_string();
    let mut spectra_path = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp".to_string();
    // Written by the training run next to the weights. Naming it here is what makes a
    // GNPS benchmark independent rather than merely differently-sourced.
    let mut exclude_manifest: Option<String> = None;
    let mut build_manifest_out: Option<String> = None;
    let mut diagnose: Option<String> = None;
    let mut use_platt = false;
    // Matches SiriusWorkflowConfig's default, which is what training uses.
    let mut max_peaks = 30usize;
    // Matches FingerprintSearcher's default.
    let mut score_mode = "modplatt".to_string();
    let mut structure_db: Option<String> = None;
    // Which MoNA partition to draw queries from, and the seed that defines the
    // partitions. Both must match the training run.
    let mut query_split = "test".to_string();
    let mut split_seed = 42u64;
    // Which of the model's recorded statistics to score with. Priors and reliability
    // weights are separable corrections, so the ablation has to be able to enable them
    // one at a time — lumping them hides which one is doing the work.
    // Default matches what FingerprintSearcher::for_predictor installs in production.
    let mut stats = "priors".to_string();
    let mut weight_floor = 0.0f32;
    let mut eval_queries = 100usize;
    let mut max_records = 5000usize;
    let mut max_queries = 300usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            // full | priors | weights | none
            "--stats" => {
                stats = args[i + 1].clone();
                i += 2;
            }
            // Shrink the reliability weights toward 1.0: w' = floor + (1 - floor) * w.
            // A floor of 1.0 is the unweighted score; 0.0 applies the raw MCC.
            "--weight-floor" => {
                weight_floor = args[i + 1].parse().unwrap_or(weight_floor);
                i += 2;
            }
            "--spectra" => {
                spectra_path = args[i + 1].clone();
                i += 2;
            }
            // Scan --spectra in full, write a manifest, and stop.
            "--build-manifest" => {
                build_manifest_out = Some(args[i + 1].clone());
                i += 2;
            }
            // Write a per-query diagnostic table and print the failure breakdown.
            // Apply the model's recorded Platt calibration. Off by default: it was
            // measured and it lost accuracy.
            "--platt" => {
                use_platt = true;
                i += 1;
            }
            // modplatt | normalized | tanimoto | continuous
            "--score" => {
                score_mode = args[i + 1].clone();
                i += 2;
            }
            "--max-peaks" => {
                max_peaks = args[i + 1].parse().unwrap_or(max_peaks);
                i += 2;
            }
            "--diagnose" => {
                diagnose = Some(args[i + 1].clone());
                i += 2;
            }
            // Draw candidates from a pre-indexed .oxdb instead of from the spectra file,
            // which is what a real identification does.
            "--structure-db" => {
                structure_db = Some(args[i + 1].clone());
                i += 2;
            }
            "--exclude-manifest" => {
                exclude_manifest = Some(args[i + 1].clone());
                i += 2;
            }
            // test | validation | train | all
            "--query-split" => {
                query_split = args[i + 1].clone();
                i += 2;
            }
            "--split-seed" => {
                split_seed = args[i + 1].parse().unwrap_or(split_seed);
                i += 2;
            }
            // Retrieval accuracy on 100 queries carries roughly a 5-point standard
            // error, so anything smaller than a few hundred cannot separate the
            // configurations this benchmark exists to compare.
            "--eval-queries" => {
                eval_queries = args[i + 1].parse().unwrap_or(eval_queries);
                i += 2;
            }
            "--max-records" => {
                max_records = args[i + 1].parse().unwrap_or(max_records);
                i += 2;
            }
            "--max-queries" => {
                max_queries = args[i + 1].parse().unwrap_or(max_queries);
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("=========================================================================");
    println!("        OxideMS CSI:FingerID Database Structure Query Benchmark          ");
    println!("=========================================================================\n");

    if let Some(out) = &build_manifest_out {
        let manifest = build_manifest(Path::new(&spectra_path));
        manifest.save(out).expect("Failed to write manifest");
        println!("Wrote manifest to {out}");
        return;
    }

    // 1. Load trained neural network weights
    println!("Loading trained fingerprint model from: {}...", model_path);
    support::require("evaluate_database_search", &[(&model_path, support::MODEL), (&spectra_path, "query spectra (--spectra; default the MoNA MSP export, https://mona.mass.ms/downloads)")]);
    let mut model_file = File::open(&model_path).expect("Could not open trained model file. Run train_fingerprint_model first.");
    let predictor = FingerprintPredictor::load_weights(&mut model_file).expect("Failed to parse model weights");
    println!("  Model successfully loaded! Parameters: {}\n", predictor.num_parameters());

    // 2. Build local structural database and select test benchmark spectra
    // Built before ingestion: the searcher's dictionary states the fingerprint
    // layout, and candidate fingerprints have to be computed in that same layout.
    // Start from the model's own layout — the Morgan vocabulary is not a scoring
    // option, it is what the bits mean — and only then override the scoring statistics
    // the ablation switches. Building a bare dictionary here instead silently scored
    // vocabulary-space predictions against folded-space candidates.
    let mut dict = FingerprintSearcher::for_predictor(&predictor).dictionary().clone();
    let use_priors = matches!(stats.as_str(), "full" | "priors");
    let use_weights = matches!(stats.as_str(), "full" | "weights");
    if !use_priors {
        dict.set_priors(vec![0.5; TOTAL_BITS]);
    }
    if use_weights && predictor.bit_weights().len() == TOTAL_BITS {
        let shrunk = predictor
            .bit_weights()
            .iter()
            .map(|&w| weight_floor + (1.0 - weight_floor) * w)
            .collect();
        dict.set_weights(shrunk);
    }
    println!("  Ranking candidates by {score_mode}.");
    println!(
        "  Morgan layout: {} dedicated vocabulary slots.",
        dict.morgan_vocabulary.ids().len()
    );
    println!(
        "  Scoring --stats {}: priors {}, weights {} ({} of {} bits live).\n",
        stats,
        if dict.priors.iter().all(|&p| p == 0.5) { "uniform" } else { "from model" },
        if dict.weights.iter().all(|&w| w == 1.0) { "uniform" } else { "from model" },
        dict.weights.iter().filter(|&&w| w > weight_floor).count(),
        TOTAL_BITS
    );
    // Pinned to ModPlatt so `search` always yields the published score, and every other
    // mode is an explicit re-rank of the same candidates below. Leaving it on the
    // library default silently made `--score modplatt` run whatever that default is.
    let searcher = FingerprintSearcher::new(dict).with_scoring(CandidateScore::ModPlatt);

    // Peak preprocessing is a property of the pipeline, not of this benchmark: the
    // model was fitted on trees built this way, so scoring has to build them the same
    // way. `max_peaks_per_tree` is the one knob, and it is shared.
    let mut workflow = SiriusClusteringWorkflow::default();
    workflow.config.max_peaks_per_tree = max_peaks;
    println!("  Peak preprocessing: shared with training, max_peaks_per_tree = {max_peaks}.");

    let external_db = structure_db.as_ref().map(|path| {
        let db = StructureDatabase::load_from_file(path).expect("Failed to load .oxdb");
        assert_eq!(
            db.dictionary.morgan_vocabulary,
            searcher.dictionary().morgan_vocabulary,
            "structure database and model disagree on the Morgan vocabulary; re-index it \
             with build_preindexed_database against this model"
        );
        println!(
            "  Candidates from {}: {} structures across {} formulas.",
            path,
            db.total_structures,
            db.entries_by_formula.len()
        );
        db
    });

    println!("Building structural candidate database from {}...", spectra_path);
    let t_db = Instant::now();

    let exclude = match &exclude_manifest {
        Some(p) => {
            let m = DatasetManifest::load(Path::new(p)).expect("Failed to read dataset manifest");
            println!(
                "  Holding out everything in {}: {} structures, {} spectra.",
                p,
                m.structure_keys.len(),
                m.spectrum_keys.len()
            );
            m
        }
        None => DatasetManifest::new(),
    };

    let wanted_split = match query_split.as_str() {
        "all" => None,
        "train" => Some(Split::Train),
        "validation" => Some(Split::Validation),
        "test" => Some(Split::Test),
        other => panic!("--query-split must be test, validation, train, or all (got {other:?})"),
    };

    // GNPS ships as MGF and carries no formula field; MoNA ships as MSP and does.
    let is_mgf = spectra_path.to_ascii_lowercase().ends_with(".mgf");
    let (query_spectra, db_by_formula) = if is_mgf {
        // A separate library has no partition of its own: every record in it is
        // held-out by construction, once the training overlap is removed.
        parse_gnps_records(
            Path::new(&spectra_path),
            searcher.dictionary(),
            max_records,
            max_queries,
            &exclude,
        )
    } else {
        assert!(
            exclude_manifest.is_none(),
            "--exclude-manifest applies to an independent library; within MoNA use --query-split instead"
        );
        println!("  Drawing queries from the {query_split} partition (split seed {split_seed}).");
        parse_mona_records(
            Path::new(&spectra_path),
            searcher.dictionary(),
            max_records,
            max_queries,
            wanted_split,
            split_seed,
        )
    };
    let total_structures: usize = db_by_formula.values().map(|v| v.len()).sum();
    println!(
        "  Indexed {} unique molecular structures across {} distinct formulas in {:.2}s\n",
        total_structures,
        db_by_formula.len(),
        t_db.elapsed().as_secs_f64()
    );

    // Score with the model's own statistics, not this database's. ModPlatt's πᵢ is the
    // prior of the training set the predictor was fitted on: it exists to cancel the
    // bias in p̂ᵢ that the training distribution put there. Substituting the frequency
    // of bit i in the candidate database corrects for something else entirely, and
    // would leave this benchmark measuring a scorer the production workflow never runs.
    let extractor = TreeFeatureExtractor::new();

    let mut diag: Vec<DiagRow> = Vec::new();
    let mut calib_pos = (0.0f64, 0usize);
    let mut calib_neg = (0.0f64, 0usize);

    let mut total_queries = 0;
    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut total_candidates_evaluated = 0;
    let mut total_tanimoto = 0.0f64;

    println!("---------------------------------------------------------------------------------------------------------");
    println!(
        "{:<4} | {:<32} | {:<12} | {:<5} | {:<6} | {:<8} | {:<8}",
        "#", "Compound Name", "Formula", "Isomers", "Rank", "Tanimoto", "Score"
    );
    println!("---------------------------------------------------------------------------------------------------------");

    // Queries whose true structure is not in the candidate database at all. Under a
    // real database this is a failure mode in its own right — no scorer can retrieve a
    // structure that was never indexed — and lumping it in with ranking errors would
    // misattribute it.
    let mut absent_from_db = 0usize;

    for query in &query_spectra {
        // Find structural candidates with matching formula
        let from_external = external_db.as_ref().map(|db| {
            // One entry per distinct structure, the same rule the in-file pool uses.
            // A pre-indexed database holds many spellings of one molecule, and counting
            // them as separate candidates both inflates the isomer count and penalizes
            // the scorer for failing to choose between identical things.
            let mut seen = std::collections::HashSet::new();
            db.get_candidates_for_context(&query.formula, SampleContext::AllBio)
                .into_iter()
                .filter_map(|s| {
                    let key = structure_key(&s.smiles).unwrap_or_else(|| s.smiles.clone());
                    seen.insert(key.clone())
                        .then(|| (s.name, s.smiles, key, s.fingerprint))
                })
                .collect::<Vec<_>>()
        });
        let candidate_list = match from_external.as_ref() {
            Some(list) if list.len() >= 2 => list,
            Some(_) => continue,
            None => match db_by_formula.get(&query.formula) {
                Some(list) if list.len() >= 2 => list,
                _ => continue,
            },
        };

        // Determine adduct
        let adduct = if query.charge < 0 {
            Adduct::deprotonated()
        } else {
            Adduct::protonated()
        };

        // Solve MAP fragmentation tree
        let config = TreeSolverConfig {
            tolerance: MassTolerance::Ppm(20.0),
            adduct,
            ..Default::default()
        };
        let builder = FragmentationTreeBuilder::new(config);
        // The same preprocessing training used. Taking the top 25 peaks by raw
        // intensity instead — no noise floor, no deisotoping, no precursor filter, and
        // intensity ranking rather than m/z-stratified selection — fed the model trees
        // built from a differently cleaned spectrum than it was fitted on.
        let sorted_peaks = match workflow.preprocess_peaks_for_tree(
            &query.peaks,
            query.precursor_mz,
            query.charge,
        ) {
            Some(p) => p,
            None => continue,
        };
        let tree = builder.build_tree(&query.formula, query.precursor_mz, &sorted_peaks);

        if tree.nodes.len() < 2 {
            continue;
        }

        // Extract tree features & predict fingerprint probabilities
        let features = extractor.extract(&tree);
        let mut predicted_fp = predictor.predict(&features.features);
        if use_platt {
            apply_platt(&mut predicted_fp.probabilities, predictor.bit_platt());
        }

        // Convert candidate structures to search format
        let search_candidates: Vec<StructureCandidate> = candidate_list
            .iter()
            .enumerate()
            .map(|(_i, (name, smiles, key, fp))| StructureCandidate {
                // The structure key, not a running index: matching the query against a
                // candidate has to compare structures, and the query's own SMILES
                // spelling need not be the one the database kept.
                id: key.clone(),
                smiles: smiles.clone(),
                name: name.clone(),
                fingerprint: fp.clone(),
            })
            .collect();

        // Run ModPlatt database search & ranking
        let mut scored_hits = searcher.search(&predicted_fp, search_candidates, 0);

        // ModPlatt is the published scorer, but it is not self-evidently the best one:
        // it weighs every bit by the same prior-corrected log-likelihood, where a
        // similarity coefficient weighs by agreement. Re-ranking the same candidates
        // under each is the only way to tell which the fingerprints actually support.
        if score_mode != "modplatt" {
            let modplatt_table = searcher.prepare(&predicted_fp);
            for hit in scored_hits.iter_mut() {
                hit.score = match score_mode.as_str() {
                    "tanimoto" => hit.tanimoto,
                    "continuous" => FingerprintSearcher::continuous_tanimoto_similarity(
                        &predicted_fp,
                        &hit.candidate.fingerprint,
                    ),
                    "normalized" => modplatt_table.score_size_normalized(&hit.candidate.fingerprint),
                    other => panic!("--score must be modplatt, normalized, tanimoto, or continuous (got {other:?})"),
                };
            }
            scored_hits.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, hit) in scored_hits.iter_mut().enumerate() {
                hit.rank = i + 1;
            }
        }

        // Find true structure rank
        let query_key = structure_key(&query.smiles).unwrap_or_else(|| query.smiles.clone());
        let true_hit_pos = scored_hits.iter().position(|hit| hit.candidate.id == query_key);

        let rank = match true_hit_pos {
            Some(pos) => pos + 1,
            None => {
                absent_from_db += 1;
                continue;
            }
        };

        let top_hit = &scored_hits[0];
        let true_hit = &scored_hits[true_hit_pos.unwrap()];

        total_queries += 1;
        total_candidates_evaluated += scored_hits.len();
        total_tanimoto += true_hit.tanimoto;

        if rank == 1 {
            top1_correct += 1;
        }
        if rank <= 3 {
            top3_correct += 1;
        }
        if rank <= 5 {
            top5_correct += 1;
        }

        if diagnose.is_some() {
            let true_fp = &true_hit.candidate.fingerprint;

            // A perfect predictor ranks by similarity to the truth, which puts the
            // truth first unless another isomer carries the very same bits.
            let indistinguishable = scored_hits
                .iter()
                .filter(|h| h.candidate.id != query_key && h.candidate.fingerprint.bits == true_fp.bits)
                .count();

            diag.push(DiagRow {
                name: query.name.clone(),
                formula: query.formula.to_string(),
                precursor_mz: query.precursor_mz,
                n_peaks: query.peaks.len(),
                tree_nodes: tree.nodes.len(),
                explained: tree.explained_intensity as f64,
                n_isomers: scored_hits.len(),
                rank,
                oracle_rank: 1 + indistinguishable,
                indistinguishable,
                pred_tanimoto: true_hit.tanimoto,
                winner_tanimoto: true_fp.tanimoto(&top_hit.candidate.fingerprint),
                score_margin: true_hit.score - top_hit.score,
                true_bits: true_fp.popcount(),
                winner_bits: top_hit.candidate.fingerprint.popcount(),
                pred_bits: predicted_fp.count_above(0.5),
                tanimoto_at: THRESHOLDS
                    .iter()
                    .map(|&t| tanimoto_over(&predicted_fp.probabilities, &true_fp.bits, 0..TOTAL_BITS, t))
                    .collect(),
                maccs_tanimoto: tanimoto_over(&predicted_fp.probabilities, &true_fp.bits, 0..MACCS_BITS, 0.5),
                morgan_tanimoto: tanimoto_over(&predicted_fp.probabilities, &true_fp.bits, MACCS_BITS..TOTAL_BITS, 0.5),
            });

            // Calibration, accumulated globally: the mean probability the model assigns
            // to bits that are actually set, against the mean it assigns to bits that
            // are not. A predictor that separates them well but sits on the wrong side
            // of 0.5 is miscalibrated, not uninformed, and that is a cheaper problem.
            for i in 0..TOTAL_BITS {
                let p = predicted_fp.probabilities[i] as f64;
                if true_fp.bits[i] {
                    calib_pos.0 += p;
                    calib_pos.1 += 1;
                } else {
                    calib_neg.0 += p;
                    calib_neg.1 += 1;
                }
            }
        }

        // Print sample results (first 25 queries)
        if total_queries <= 25 {
            let rank_str = if rank == 1 {
                format!("\x1b[32mRank #{}\x1b[0m", rank)
            } else if rank <= 3 {
                format!("\x1b[33mRank #{}\x1b[0m", rank)
            } else {
                format!("Rank #{}", rank)
            };

            let display_name = if query.name.len() > 30 {
                format!("{}...", &query.name[..27])
            } else {
                query.name.clone()
            };

            println!(
                "{:<4} | {:<32} | {:<12} | {:<7} | {:<6} | {:<8.4} | {:<8.1}",
                total_queries,
                display_name,
                query.formula.to_string(),
                scored_hits.len(),
                rank_str,
                true_hit.tanimoto,
                true_hit.score
            );
        }

        if total_queries >= eval_queries {
            break;
        }
    }

    let n = total_queries.max(1) as f64;
    println!("---------------------------------------------------------------------------------------------------------");
    println!("                                   Database Retrieval Summary                                            ");
    println!("---------------------------------------------------------------------------------------------------------");
    println!("  Total Queries Evaluated:               {}", total_queries);
    if absent_from_db > 0 {
        println!(
            "  Queries dropped, true structure not indexed: {} ({:.1}% of those with candidates)",
            absent_from_db,
            absent_from_db as f64 / (absent_from_db + total_queries).max(1) as f64 * 100.0
        );
    }
    println!("  Avg. Structural Isomers per Query:     {:.1}", total_candidates_evaluated as f64 / n);
    println!("  Top-1 Identification Accuracy:         {:.1}% ({}/{})", (top1_correct as f64 / n) * 100.0, top1_correct, total_queries);
    println!("  Top-3 Identification Accuracy:         {:.1}% ({}/{})", (top3_correct as f64 / n) * 100.0, top3_correct, total_queries);
    println!("  Top-5 Identification Accuracy:         {:.1}% ({}/{})", (top5_correct as f64 / n) * 100.0, top5_correct, total_queries);

    if let Some(csv_path) = &diagnose {
        report_fingerprint_quality(&diag, calib_pos, calib_neg);
        report_failure_modes(&diag);

        let mut w = BufWriter::new(File::create(csv_path).expect("Could not create diagnostics file"));
        writeln!(
            w,
            "name,formula,precursor_mz,n_peaks,tree_nodes,explained_intensity,n_isomers,rank,oracle_rank,indistinguishable,pred_tanimoto,winner_tanimoto,score_margin,true_bits,winner_bits,pred_bits"
        )
        .unwrap();
        for r in &diag {
            writeln!(
                w,
                "{:?},{},{:.4},{},{},{:.4},{},{},{},{},{:.4},{:.4},{:.2},{},{},{}",
                r.name, r.formula, r.precursor_mz, r.n_peaks, r.tree_nodes, r.explained,
                r.n_isomers, r.rank, r.oracle_rank, r.indistinguishable, r.pred_tanimoto,
                r.winner_tanimoto, r.score_margin, r.true_bits, r.winner_bits, r.pred_bits
            )
            .unwrap();
        }
        println!("\n  Per-query diagnostics written to {csv_path}");
    }
    println!("  Mean Predicted-vs-True Tanimoto:       {:.4}", total_tanimoto / n);
    println!("---------------------------------------------------------------------------------------------------------\n");
}
