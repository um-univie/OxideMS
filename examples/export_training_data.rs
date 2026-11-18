//! Export MoNA Reference Dataset for PyTorch Architectural Exploration.
//!
//! Generates:
//! 1. `data/fingerprint_dataset.json` (Structured JSON containing features, graph nodes/edges, and fingerprints).

use oxidems_lib::sirius::fingerprint::{
    FingerprintDictionary,
    BinaryFingerprint, TOTAL_BITS, TreeFeatureExtractor,
};
use oxidems_lib::sirius::{Adduct, ChemicalRules, Formula, FragmentationTreeBuilder, SolverKind, TreeSolverConfig};
use oxidems_lib::sirius::tree::FragmentationTree;
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
use xxhash_rust::xxh32::xxh32;

#[derive(Serialize)]
struct ExportRecord {
    id: usize,
    name: String,
    /// First block of the InChIKey — the structure identity used to group the
    /// train/test split. Grouping on anything finer leaks: MoNA carries a mean of
    /// 7.6 spectra per structure, so a record-level split trains and tests on the
    /// same molecule.
    inchikey14: String,
    smiles: String,
    formula: String,
    precursor_mz: f64,
    charge: i8,
    features_1555: Vec<f32>,
    features_4123: Vec<f32>,
    graph_nodes: Vec<Vec<f32>>,
    graph_edges: Vec<(usize, usize)>,
    graph_edge_feats: Vec<Vec<f32>>,
    fingerprint: Vec<u8>, // 0 or 1
    /// Which split the source dataset assigns this record to. MoNA has none and
    /// says `mona`; MassSpecGym ships `train` / `val` / `test`, and its folds are
    /// MCES-10 disjoint, which is the whole point of training against them.
    fold: String,
    adduct: String,
}

struct RawSample {
    name: String,
    inchikey14: String,
    formula: Formula,
    smiles: String,
    precursor_mz: f64,
    charge: i8,
    peaks: Vec<Peak>,
    adduct: Adduct,
    fold: String,
}

/// MassSpecGym's distribution TSV (`data/MassSpecGym.tsv`): one spectrum per line,
/// peaks as two comma-separated lists, and a `fold` column whose splits are disjoint
/// at MCES 10. Training against those folds is the only way our retrieval number is
/// comparable to that leaderboard — see round 52.2, where 97 % of the CASMI
/// evaluation turned out to sit within MCES 10 of our training set.
fn load_massspecgym(
    path: &Path,
    max_records: usize,
    folds: &std::collections::HashSet<String>,
    excluded: &std::collections::HashSet<String>,
) -> Vec<RawSample> {
    let file = File::open(path).unwrap_or_else(|e| support::fail(format!("cannot open {}: {e}\n  MassSpecGym TSV, https://huggingface.co/datasets/roman-bushuiev/MassSpecGym", path.display())));
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut lines = reader.lines();
    let header = lines.next().expect("empty TSV").expect("unreadable header");
    let cols: Vec<&str> = header.trim_end().split('\t').collect();
    let idx = |name: &str| -> usize {
        cols.iter().position(|c| *c == name)
            .unwrap_or_else(|| panic!("MassSpecGym TSV has no `{name}` column"))
    };
    let (c_id, c_mz, c_int, c_smiles, c_key, c_formula, c_pmz, c_add, c_fold) = (
        idx("identifier"), idx("mzs"), idx("intensities"), idx("smiles"), idx("inchikey"),
        idx("formula"), idx("precursor_mz"), idx("adduct"), idx("fold"),
    );

    let mut samples = Vec::new();
    let mut skipped_adduct = 0usize;
    for line in lines {
        if samples.len() >= max_records {
            break;
        }
        let Ok(line) = line else { continue };
        let f: Vec<&str> = line.trim_end().split('\t').collect();
        if f.len() <= c_fold {
            continue;
        }
        if !folds.is_empty() && !folds.contains(f[c_fold]) {
            continue;
        }
        let key14: String = f[c_key].chars().take(14).collect();
        if excluded.contains(&key14) {
            continue;
        }
        let Ok(formula) = Formula::from_str(f[c_formula]) else { continue };
        let Ok(precursor_mz) = f[c_pmz].parse::<f64>() else { continue };
        // An unparseable adduct is skipped rather than assumed to be protonation:
        // MassSpecGym is 16 % [M+Na]+, and calling those [M+H]+ shifts every
        // fragment mass by 21.98 Da.
        let Ok(adduct) = Adduct::from_str(f[c_add]) else {
            skipped_adduct += 1;
            continue;
        };
        let mzs = f[c_mz].split(',').filter_map(|v| v.parse::<f64>().ok());
        let ints = f[c_int].split(',').filter_map(|v| v.parse::<f32>().ok());
        let peaks: Vec<Peak> = mzs.zip(ints).map(|(mz, intensity)| Peak { mz, intensity }).collect();
        if peaks.len() < 2 {
            continue;
        }
        samples.push(RawSample {
            name: f[c_id].to_string(),
            inchikey14: key14,
            formula,
            smiles: f[c_smiles].to_string(),
            precursor_mz,
            charge: if adduct.name.starts_with("[M-") { -1 } else { 1 },
            peaks,
            adduct,
            fold: f[c_fold].to_string(),
        });
    }
    if skipped_adduct > 0 {
        eprintln!("   skipped {skipped_adduct} records whose adduct we cannot model");
    }
    samples
}

fn load_mona(
    path: &Path,
    max_records: usize,
    excluded: &std::collections::HashSet<String>,
) -> Vec<RawSample> {
    let file = File::open(path).unwrap_or_else(|e| support::fail(format!("cannot open {}: {e}\n  {} (or pass --tsv for MassSpecGym)", path.display(), support::MONA)));
    let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut samples = Vec::new();
    let mut cur_name = String::new();
    let mut cur_inchikey = String::new();
    let mut cur_formula = String::new();
    let mut cur_smiles = String::new();
    let mut cur_mz = 0.0f64;
    let mut cur_charge = 1i8;
    let mut cur_peaks = Vec::new();

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();

        if line.is_empty() {
            // The CASMI blind set is entirely contained in MoNA — all 123 challenge
            // structures are present — so training on MoNA without this exclusion
            // leaks 100 % of the benchmark. Excluded by InChIKey first block, which
            // is the structure, not the spectrum.
            let excluded_here = !cur_inchikey.is_empty() && excluded.contains(&cur_inchikey);
            if !excluded_here
                && !cur_smiles.is_empty()
                && !cur_formula.is_empty()
                && cur_peaks.len() >= 3
                && cur_mz > 0.0
            {
                if let Ok(formula) = Formula::from_str(&cur_formula) {
                    if (50.0..=800.0).contains(&formula.exact_mass()) {
                        samples.push(RawSample {
                            name: std::mem::take(&mut cur_name),
                            inchikey14: std::mem::take(&mut cur_inchikey),
                            formula,
                            smiles: std::mem::take(&mut cur_smiles),
                            precursor_mz: cur_mz,
                            charge: cur_charge,
                            peaks: std::mem::take(&mut cur_peaks),
                            adduct: if cur_charge < 0 { Adduct::deprotonated() } else { Adduct::protonated() },
                            fold: "mona".to_string(),
                        });
                        if samples.len() >= max_records {
                            break;
                        }
                    }
                }
            }
            cur_mz = 0.0;
            cur_charge = 1;
            cur_name.clear();
            cur_inchikey.clear();
            cur_formula.clear();
            cur_smiles.clear();
            cur_peaks.clear();
            continue;
        }

        if let Some(val) = line.strip_prefix("Name:") {
            cur_name = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("InChIKey:") {
            cur_inchikey = val.trim().chars().take(14).collect();
        } else if let Some(val) = line.strip_prefix("Formula:") {
            cur_formula = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("PrecursorMZ:") {
            cur_mz = val.trim().parse::<f64>().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("Ion_mode:") {
            if val.trim().starts_with('N') || val.trim().starts_with('-') {
                cur_charge = -1;
            } else {
                cur_charge = 1;
            }
        } else if line.starts_with("Comments:") {
            if let Some(idx) = line.find("SMILES=") {
                let rest = &line[idx + 7..];
                let end = rest.find('"').or_else(|| rest.find(' ')).unwrap_or(rest.len());
                cur_smiles = rest[..end].trim().to_string();
            }
        } else if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz), Ok(intensity)) = (parts[0].parse::<f64>(), parts[1].parse::<f32>()) {
                    if mz > 0.0 && intensity > 0.0 {
                        cur_peaks.push(Peak { mz, intensity });
                    }
                }
            }
        }
    }
    samples
}

fn extract_4123(tree: &FragmentationTree, charge: i8) -> Vec<f32> {
    const ADV_FRAG_BUCKETS: usize = 1024;
    const ADV_LOSS_BUCKETS: usize = 512;
    const ADV_ROOT_LOSS_BUCKETS: usize = 1024;
    const ADV_TREE_STATS: usize = 27;
    const ADV_TOTAL_FEATURES: usize = ADV_FRAG_BUCKETS * 2 + ADV_LOSS_BUCKETS * 2 + ADV_ROOT_LOSS_BUCKETS * 2 + ADV_TREE_STATS;

    let mut feats = vec![0.0f32; ADV_TOTAL_FEATURES];
    let off_frag_bin = 0;
    let off_frag_int = ADV_FRAG_BUCKETS;
    let off_loss_bin = ADV_FRAG_BUCKETS * 2;
    let off_loss_int = ADV_FRAG_BUCKETS * 2 + ADV_LOSS_BUCKETS;
    let off_root_bin = ADV_FRAG_BUCKETS * 2 + ADV_LOSS_BUCKETS * 2;
    let off_root_int = ADV_FRAG_BUCKETS * 2 + ADV_LOSS_BUCKETS * 2 + ADV_ROOT_LOSS_BUCKETS;
    let off_stats = ADV_FRAG_BUCKETS * 2 + ADV_LOSS_BUCKETS * 2 + ADV_ROOT_LOSS_BUCKETS * 2;

    let root_formula = &tree.precursor_formula;

    for node in &tree.nodes {
        let bucket = hash_formula_bytes(&node.formula, 0x4652_4147, ADV_FRAG_BUCKETS);
        feats[off_frag_bin + bucket] = 1.0;
        feats[off_frag_int + bucket] += node.relative_intensity.sqrt();
    }

    for edge in &tree.edges {
        let bucket = hash_formula_bytes(&edge.loss_formula, 0x4C4F_5353, ADV_LOSS_BUCKETS);
        feats[off_loss_bin + bucket] = 1.0;
        let child_int = tree.nodes[edge.child_id].relative_intensity;
        feats[off_loss_int + bucket] += child_int.sqrt();
    }

    for node in tree.nodes.iter().skip(1) {
        if let Some(rl) = root_formula.checked_sub(&node.formula) {
            let bucket = hash_formula_bytes(&rl, 0x524F_4F54, ADV_ROOT_LOSS_BUCKETS);
            feats[off_root_bin + bucket] = 1.0;
            feats[off_root_int + bucket] += node.relative_intensity.sqrt();
        }
    }

    let c = root_formula.c().max(1) as f32;
    let stats = [
        (root_formula.exact_mass() / 1000.0) as f32,
        (root_formula.exact_mass() - root_formula.exact_mass().round()) as f32,
        root_formula.dbe() as f32 / 20.0,
        root_formula.h() as f32 / c,
        root_formula.n() as f32 / c,
        root_formula.o() as f32 / c,
        root_formula.p() as f32 / c,
        root_formula.s() as f32 / c,
        root_formula.f() as f32 / c,
        root_formula.cl() as f32 / c,
        root_formula.br() as f32 / c,
        root_formula.i() as f32 / c,
        root_formula.si() as f32 / c,
        root_formula.b() as f32 / c,
        tree.nodes.len() as f32 / 50.0,
        tree.edges.len() as f32 / 50.0,
        tree.explained_intensity,
        tree.explained_peaks as f32 / tree.total_peaks.max(1) as f32,
        if charge > 0 { 1.0f32 } else { 0.0f32 },
        if charge < 0 { 1.0f32 } else { 0.0f32 },
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    feats[off_stats..off_stats + stats.len()].copy_from_slice(&stats);

    feats
}

fn hash_formula_bytes(formula: &Formula, seed: u32, n_buckets: usize) -> usize {
    let mut bytes = [0u8; 12 * 2];
    for i in 0..12 {
        let val = formula.counts[i] as u16;
        let le = val.to_le_bytes();
        bytes[i * 2] = le[0];
        bytes[i * 2 + 1] = le[1];
    }
    let hash = xxh32(&bytes, seed);
    (hash as usize) % n_buckets
}

mod support;

fn main() {
    // export_training_data [--limit N] [--out PATH] [--exclude FILE] [--graph-only]
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let msp_path = "test_files/MoNA-export-LC-MS-MS_Positive_Mode.msp";
    // `--tsv` reads MassSpecGym's distribution file instead of MoNA, and `--folds`
    // picks which of its splits to export (default `train,val` -- exporting `test`
    // into a training pack is the mistake this benchmark exists to prevent).
    let tsv_path = arg("--tsv");
    let folds: std::collections::HashSet<String> = arg("--folds")
        .unwrap_or_else(|| "train,val".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let out_path = arg("--out").unwrap_or_else(|| "data/fingerprint_dataset.json".to_string());
    let out_path = out_path.as_str();
    // The old hard-coded 3500 read the first 3.4 % of a 102 577-spectrum file and
    // yielded 627 unique structures. MoNA holds 12 124.
    let limit: usize = arg("--limit").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let graph_only = args.iter().any(|a| a == "--graph-only");
    // One peak budget for the whole pipeline. This was a hard-coded 25 here while the
    // CASMI harness used 30, so the model was evaluated on a tree-size distribution it
    // had never been trained on. Both now read the library default.
    let max_peaks: usize = arg("--max-peaks")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| oxidems_lib::sirius::SiriusWorkflowConfig::default().max_peaks_per_tree);
    // `SolverKind::Exact` falls back to Critical-Path above `EXACT_COLOR_CAP = 14`
    // distinct peaks — i.e. on most spectra at any useful peak budget — so trees
    // exported under it were greedy while the formula pipeline's own results are
    // measured with the exact optimum. The ILP reaches the same optimum with no
    // cap and is now the library default; the export follows it. Trees exported
    // before 2026-09-07 were built with the capped solver and are not comparable.
    let solver = match arg("--solver").as_deref() {
        None | Some("ilp") => SolverKind::Ilp,
        Some("exact") => SolverKind::Exact,
        Some("dagdp") => SolverKind::DagDp,
        Some("critical") | Some("critical-path") => SolverKind::CriticalPath,
        Some(other) => panic!("--solver must be exact|ilp|dagdp|critical (got {other})"),
    };
    // The exclusion list is InChIKey14 prefixes, one per line — NOT the SMILES
    // `OXFP-MANIFEST` that `build_casmi_exclusion` writes for the Rust
    // `--exclude-manifest` path. Truncating a SMILES to 14 characters yields a
    // string that matches no InChIKey, so passing the wrong file used to exclude
    // nothing while printing a reassuring count. Refuse it instead.
    let excluded: std::collections::HashSet<String> = match arg("--exclude") {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read exclusion file {path}: {e}"));
            if text.starts_with("OXFP-MANIFEST") {
                panic!(
                    "{path} is a SMILES OXFP-MANIFEST, not an InChIKey14 list. \
                     Truncating SMILES to 14 characters excludes nothing. \
                     Convert it first."
                );
            }
            let keys: std::collections::HashSet<String> = text
                .lines()
                .map(|l| l.trim().chars().take(14).collect::<String>())
                .filter(|l| !l.is_empty())
                .collect();
            // An InChIKey block is 14 uppercase letters. Anything else is not one.
            let bad = keys
                .iter()
                .filter(|k| k.len() != 14 || !k.bytes().all(|b| b.is_ascii_uppercase()))
                .count();
            if bad * 10 > keys.len() {
                panic!(
                    "{path}: {bad} of {} entries are not 14-character InChIKey blocks. \
                     This is not an InChIKey14 exclusion list.",
                    keys.len()
                );
            }
            keys
        }
        None => Default::default(),
    };
    if excluded.is_empty() && tsv_path.is_none() {
        eprintln!("WARNING: no --exclude file given. MoNA contains every CASMI structure.");
    } else {
        println!("Excluding {} structures by InChIKey14.", excluded.len());
    }

    println!("=========================================================================");
    println!("      OxideMS MoNA Dataset Exporter for PyTorch ML Exploration           ");
    println!("=========================================================================");

    let raw_samples = match &tsv_path {
        Some(p) => {
            println!("1. Parsing MassSpecGym records from {p} (folds {:?})...", folds);
            load_massspecgym(Path::new(p), limit, &folds, &excluded)
        }
        None => {
            println!("1. Parsing MSP records from {}...", msp_path);
            load_mona(Path::new(msp_path), limit, &excluded)
        }
    };
    println!("   Parsed {} candidate records.\n", raw_samples.len());

    println!("2. Solving MAP Fragmentation Trees & Ground-Truth Fingerprints (Rayon)...");
    let t_solve = Instant::now();

    // Exported ground truth is only meaningful with its layout stated.
    let layout = FingerprintDictionary::new();

    let records: Vec<ExportRecord> = raw_samples
        .par_iter()
        .enumerate()
        .filter_map(|(id, rec)| {
            let fp = layout.ground_truth_fingerprint(&rec.smiles)?;
            let adduct = rec.adduct.clone();
            let config = TreeSolverConfig {
                tolerance: MassTolerance::Ppm(20.0),
                adduct,
                solver,
                ..Default::default()
            };
            let builder = FragmentationTreeBuilder::new(config);
            let mut sorted_peaks = rec.peaks.clone();
            sorted_peaks.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
            sorted_peaks.truncate(max_peaks);
            let tree = builder.build_tree(&rec.formula, rec.precursor_mz, &sorted_peaks);
            if tree.nodes.len() < 2 {
                return None;
            }

            // The two dense feature blocks are 6 700 floats per record and dominate
            // the file — they are what a graph model is meant to replace, so
            // `--graph-only` drops them.
            let (feats_1555, feats_4123) = if graph_only {
                (Vec::new(), Vec::new())
            } else {
                (TreeFeatureExtractor::new().extract(&tree).features, extract_4123(&tree, rec.charge))
            };

            let (graph_nodes, graph_edges, graph_edge_feats) =
                oxidems_lib::sirius::fingerprint::tree_graph(&tree, rec.charge);

            let fp_u8: Vec<u8> = fp.iter().map(|&b| if b { 1 } else { 0 }).collect();

            Some(ExportRecord {
                id,
                inchikey14: rec.inchikey14.clone(),
                name: rec.name.clone(),
                smiles: rec.smiles.clone(),
                formula: rec.formula.to_string(),
                precursor_mz: rec.precursor_mz,
                charge: rec.charge,
                features_1555: feats_1555,
                features_4123: feats_4123,
                graph_nodes,
                graph_edges,
                graph_edge_feats,
                fingerprint: fp_u8,
                fold: rec.fold.clone(),
                adduct: rec.adduct.name.clone(),
            })
        })
        .collect();

    println!("   Solved {} trees in {:.2}s\n", records.len(), t_solve.elapsed().as_secs_f64());

    println!("3. Serializing dataset to {}...", out_path);
    if let Some(parent) = Path::new(out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = File::create(out_path).expect("Failed to create output file");
    let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);
    // JSON Lines: 89 k records will not fit comfortably in one array, and a
    // streaming reader is what the training script wants anyway.
    if out_path.ends_with(".jsonl") {
        for r in &records {
            serde_json::to_writer(&mut writer, r).expect("Failed to write JSONL");
            writeln!(writer).expect("Failed to write JSONL");
        }
    } else {
        serde_json::to_writer(&mut writer, &records).expect("Failed to write JSON");
    }
    writer.flush().expect("Failed to flush JSON");

    println!("=========================================================================");
    println!("  Export Complete: {} samples saved to {}", records.len(), out_path);
    println!("=========================================================================\n");
}
