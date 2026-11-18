//! Build an MS1-bearing formula-ID benchmark from MSnLib.
//!
//! Every library export we benchmark on (MoNA, MassBank, GNPS) is MS2-only, so the
//! isotope pattern — the one piece of evidence that separates C+Cl from O+P, which
//! is 52 % of our remaining losses to SIRIUS — is simply absent. On CASMI, where
//! MS1 *is* present, it is worth +8.8 points (89.2 % -> 98.0 %). This recovers it
//! at scale.
//!
//! MSnLib (MassIVE MSV000094528) ships structure-annotated MSn trees as mgf, but
//! its export keeps only `FEATURE_MS1_HEIGHT`, a scalar. The raw mzML has the
//! survey scans. This joins the two on source file + retention time and writes the
//! `MS1PEAKS=` mgf that `eval_tree_scoring` already reads.
//!
//! Usage: build_msnlib_ms1 <msnlib.mgf> <mzml_dir> <out.mgf>

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::io::mzml::{MzmlParser, MzmlParserConfig};
use oxidems_lib::spectrum::Peak;
use std::collections::HashMap;
use std::io::Write;

/// Widest window that can hold a monoisotopic peak plus its M+1..M+4 satellites,
/// with room for a doubly-charged envelope's half-spacings.
const ISO_LO: f64 = 1.6;
const ISO_HI: f64 = 6.5;

mod support;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        support::fail("usage: build_msnlib_ms1 <MSnLib .mgf> <directory of MSnLib .mzML runs> <output .mgf>\n  MSnLib: MassIVE MSV000094528, doi:10.5281/zenodo.21105617");
    }
    let (mgf_path, mzml_dir, out_path) = (&a[1], &a[2], &a[3]);

    let mut parser = MgfParser::default();
    let records = parser.parse_file(mgf_path).expect("parse msnlib mgf");
    eprintln!("parsed {} msnlib records", records.len());

    // Group the library records by the raw file they came from, so each mzML is
    // parsed exactly once.
    let mut by_file: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        let Some(fid) = r.get_metadata("FEATURE_ID") else { continue };
        // "…_id_MSn_positive.mzML msn trees:5"
        let Some(name) = fid.split_whitespace().next() else { continue };
        by_file.entry(name.to_string()).or_default().push(i);
    }
    eprintln!("{} distinct source files referenced", by_file.len());

    // One spectrum per structure: the richest MS2, so the benchmark is a set of
    // distinct compounds rather than a set of replicate acquisitions.
    let mut best: HashMap<String, (usize, String)> = HashMap::new();
    let (mut written, mut no_mzml, mut no_ms1) = (0usize, 0usize, 0usize);

    for (fname, idxs) in &by_file {
        let path = format!("{mzml_dir}/{fname}");
        if !std::path::Path::new(&path).is_file() {
            no_mzml += idxs.len();
            continue;
        }
        let mut mp = MzmlParser::new(MzmlParserConfig { parse_metadata: true, ..Default::default() });
        let Ok(scans) = mp.parse_file(&path) else { no_mzml += idxs.len(); continue };
        // MS1 survey scans with their retention times, in time order.
        let mut ms1: Vec<(f64, &[Peak])> = scans
            .iter()
            .filter(|s| s.ms_level == 1)
            .filter_map(|s| {
                s.get_metadata("RTINSECONDS")
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|rt| (rt, s.peaks()))
            })
            .collect();
        ms1.sort_by(|x, y| x.0.total_cmp(&y.0));
        if ms1.is_empty() {
            no_ms1 += idxs.len();
            continue;
        }

        for &i in idxs {
            let r = &records[i];
            // MSnLib stores every MSn level as its own record. Only the merged
            // pseudo-MS2 has the *precursor* in PEPMASS; an MS3+ record's PEPMASS
            // is a fragment, which no survey scan will contain. Those are 90 % of
            // the file and skipping them is not a loss.
            if r.get_metadata("MSLEVEL").map(str::trim) != Some("2") {
                continue;
            }
            let (Some(tf), Some(smi), Some(pep)) = (
                r.get_metadata("FORMULA"),
                r.get_metadata("SMILES"),
                r.get_metadata("PEPMASS").and_then(|s| s.split_whitespace().next()),
            ) else { continue };
            let Ok(pepmass) = pep.parse::<f64>() else { continue };
            let rt: f64 = r.get_metadata("RTINSECONDS").and_then(|v| v.parse().ok()).unwrap_or(0.0);

            // Survey scan nearest in retention time to the feature.
            let j = ms1.partition_point(|(t, _)| *t < rt).min(ms1.len() - 1);
            let cand = [j.saturating_sub(1), j];
            let nearest = cand.iter().copied()
                .min_by(|&x, &y| (ms1[x].0 - rt).abs().total_cmp(&(ms1[y].0 - rt).abs()))
                .unwrap();
            let env: Vec<&Peak> = ms1[nearest].1.iter()
                .filter(|p| p.mz > pepmass - ISO_LO && p.mz < pepmass + ISO_HI)
                .collect();
            // An envelope without the monoisotopic peak itself is not usable.
            if !env.iter().any(|p| (p.mz - pepmass).abs() < 0.02) {
                no_ms1 += 1;
                continue;
            }

            let ms1s: Vec<String> =
                env.iter().map(|p| format!("{:.6}:{:.1}", p.mz, p.intensity)).collect();
            let mut block = String::new();
            block.push_str("BEGIN IONS\n");
            block.push_str(&format!("TITLE={}\n", r.get_metadata("NAME").unwrap_or("?").trim()));
            block.push_str(&format!("TRUEFORMULA={}\n", tf.trim()));
            block.push_str(&format!("SMILES={}\n", smi.trim()));
            block.push_str(&format!("PEPMASS={pepmass}\n"));
            block.push_str("CHARGE=1\n");
            block.push_str(&format!("IONMODE={}\n", r.get_metadata("IONMODE").unwrap_or("Positive").trim()));
            // The measured adduct. ~45 % of real GNPS spectra are not [M+H]+, and a
            // harness that assumes protonation turns every one of those into a
            // wrong neutral mass. Carrying it forward lets the benchmark say
            // whether that matters.
            block.push_str(&format!("ADDUCT={}\n", r.get_metadata("ADDUCT").unwrap_or("[M+H]+").trim()));
            block.push_str(&format!("MS1PEAKS={}\n", ms1s.join(" ")));
            for p in r.peaks() {
                block.push_str(&format!("{:.6} {:.4}\n", p.mz, p.intensity));
            }
            block.push_str("END IONS\n");
            // Keyed by (structure, adduct): ~45 % of real spectra are not [M+H]+,
            // and deduping on structure alone keeps only the protonated form, which
            // is exactly the population a protonation-assuming harness looks good on.
            let key = format!("{}|{}", smi.trim(), r.get_metadata("ADDUCT").unwrap_or("[M+H]+").trim());
            let n = r.peaks().len();
            if best.get(&key).map(|(m, _)| n > *m).unwrap_or(true) {
                best.insert(key, (n, block));
            }
        }
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create out"));
    // Sorted, so the file is byte-identical across runs. A HashMap's iteration
    // order is not stable, and an mgf whose record order changes between builds
    // silently invalidates every `idx`-keyed join against it.
    let mut keys: Vec<&String> = best.keys().collect();
    keys.sort();
    for k in keys {
        let (_, block) = &best[k];
        out.write_all(block.as_bytes()).unwrap();
        written += 1;
    }
    eprintln!("wrote {written} structures with MS1; skipped {no_mzml} (mzML absent), {no_ms1} (no usable envelope)");
}
