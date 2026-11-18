//! Sanity check requested after GNPS's ~0.50 purity ceiling looked
//! suspicious: for every InChIKey with 2+ spectra in the ground truth,
//! compute the EXACT (binned, non-hashed) pairwise cosine similarity
//! between all its member spectra on the raw peak data (precursor removed,
//! nothing else stripped -- no top-n-peaks cap, no relative-intensity
//! floor, no intensity transform), and report the distribution. This
//! answers: even in the best case, using the tool's own real similarity
//! metric with no approximation shortcuts, how often would two spectra of
//! the SAME compound actually clear the similarity threshold needed to
//! cluster together at all?
//!
//! epsilon in this codebase is a cosine-DISTANCE threshold: two spectra
//! connect if cosine_similarity >= 1.0 - epsilon (engine.rs: `threshold =
//! 1.0 - self.epsilon`). So eps=0.1 requires similarity >= 0.9, eps=0.3
//! requires >= 0.7, eps=0.5 requires >= 0.5.
//!
//! Large InChIKey groups are capped (random sample) for tractability --
//! the whole point is a distribution, not exhaustive enumeration of a
//! 25,456-member group's 324M pairs.

use oxidems_lib::io::mgf::MgfParser;
use oxidems_lib::spectrum::BinnedSpectrumConfig;
use oxidems_lib::workflow::{RemovePrecursor, MassTolerance, WorkFlowManager};
use std::collections::HashMap;
use std::fs;

const MAX_GROUP_SIZE: usize = 50;
const BIN_WIDTH: f64 = 0.05; // Da, matches the fragment tolerance used elsewhere in this paper
const EPS_VALUES: [f64; 6] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5];

fn simple_rng(seed: &mut u64) -> u64 {
    // xorshift64 -- good enough for a reproducible sample, no external dep needed
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

mod support;

fn main() {
    let gnps_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dataset_manifests/metabolomics/gnps/mgf".to_string());
    let ground_truth_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "dataset_manifests/metabolomics/gnps/ground_truth.csv".to_string());

    support::require("gnps_identity_similarity", &[(&gnps_dir, "directory of GNPS library MGF files (first argument)"), (&ground_truth_path, "ground-truth CSV mapping spectra to identities (second argument)")]);
    eprintln!("Loading ground truth from {ground_truth_path}");
    let gt_content = fs::read_to_string(&ground_truth_path).expect("read ground truth");
    let mut identity_of: HashMap<String, String> = HashMap::new();
    for line in gt_content.lines().skip(1) {
        if let Some((usi, inchikey)) = line.split_once(',') {
            identity_of.insert(usi.to_string(), inchikey.to_string());
        }
    }
    eprintln!("{} ground-truth USIs loaded", identity_of.len());

    let mut workflow = WorkFlowManager::default();
    workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));

    let mut paths: Vec<String> = fs::read_dir(&gnps_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            let s = p.to_str()?.to_string();
            if s.ends_with(".mgf") { Some(s) } else { None }
        })
        .collect();
    paths.sort();
    eprintln!("{} GNPS mgf files found", paths.len());

    // inchikey -> list of (binned intensities as sparse (bin_idx, intensity) after
    // alignment isn't needed here since we keep MassSpectralRecord and bin per-pair
    // lazily via to_binned(); store raw peaks instead, grouped by identity.
    use oxidems_lib::MassSpectralRecord;
    let mut groups: HashMap<String, Vec<MassSpectralRecord>> = HashMap::new();

    for path in &paths {
        let basename = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let mut parser = MgfParser::default();
        let mut spectra = parser.parse_file(path).expect("parse mgf");
        workflow.execute(&mut spectra);
        for s in spectra {
            let scan = s
                .get_metadata("SCANS")
                .and_then(|v| v.parse::<usize>().ok());
            let Some(scan) = scan else { continue };
            let identifier = format!("mzspec:USI000000:{basename}:scan:{scan}");
            if let Some(inchikey) = identity_of.get(&identifier) {
                groups.entry(inchikey.clone()).or_default().push(s);
            }
        }
    }
    eprintln!("{} distinct InChIKeys matched to at least one spectrum", groups.len());

    groups.retain(|_, v| v.len() >= 2);
    eprintln!("{} InChIKeys have 2+ spectra (usable for pairwise comparison)", groups.len());

    let config = BinnedSpectrumConfig {
        bin_width: BIN_WIDTH,
        ..Default::default()
    };

    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut all_similarities: Vec<f32> = Vec::new();
    let mut groups_with_any_hit = [0usize; EPS_VALUES.len()];
    let mut groups_processed = 0usize;
    let mut groups_capped = 0usize;
    let mut total_pairs = 0u64;

    for (_inchikey, spectra) in groups.iter_mut() {
        let members: &mut Vec<MassSpectralRecord> = spectra;
        if members.len() > MAX_GROUP_SIZE {
            groups_capped += 1;
            // Fisher-Yates partial shuffle down to MAX_GROUP_SIZE via xorshift
            let n = members.len();
            for i in 0..MAX_GROUP_SIZE {
                let j = i + (simple_rng(&mut seed) as usize % (n - i));
                members.swap(i, j);
            }
            members.truncate(MAX_GROUP_SIZE);
        }

        let binned: Vec<_> = members.iter().map(|s| s.to_binned(&config)).collect();
        let mut group_max_sim = 0.0f32;
        for i in 0..binned.len() {
            for j in (i + 1)..binned.len() {
                if let Some(sim) = binned[i].cosine_similarity(&binned[j]) {
                    all_similarities.push(sim);
                    total_pairs += 1;
                    if sim > group_max_sim {
                        group_max_sim = sim;
                    }
                }
            }
        }
        for (idx, eps) in EPS_VALUES.iter().enumerate() {
            if group_max_sim >= (1.0 - *eps) as f32 {
                groups_with_any_hit[idx] += 1;
            }
        }
        groups_processed += 1;
        if groups_processed % 20000 == 0 {
            eprintln!("  processed {groups_processed} groups, {total_pairs} pairs so far...");
        }
    }

    eprintln!(
        "{groups_processed} groups processed ({groups_capped} capped to {MAX_GROUP_SIZE}), {total_pairs} total pairs"
    );

    all_similarities.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = all_similarities.len();
    if n == 0 {
        eprintln!("No pairs computed -- check paths/ground truth matching.");
        return;
    }
    let mean: f64 = all_similarities.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let median = all_similarities[n / 2];
    println!("=== Pairwise same-InChIKey cosine similarity (n={n} pairs) ===");
    println!("mean={mean:.4} median={median:.4}");
    println!(
        "percentiles: p10={:.4} p25={:.4} p50={:.4} p75={:.4} p90={:.4}",
        all_similarities[n / 10],
        all_similarities[n / 4],
        all_similarities[n / 2],
        all_similarities[3 * n / 4],
        all_similarities[9 * n / 10],
    );

    println!("\n=== Fraction of pairs clearing the similarity threshold at each epsilon ===");
    for eps in EPS_VALUES {
        let threshold = (1.0 - eps) as f32;
        let count = all_similarities.partition_point(|&x| x < threshold);
        let above = n - count;
        println!(
            "eps={eps:.2} (sim>={threshold:.2}): {above}/{n} pairs ({:.2}%)",
            100.0 * above as f64 / n as f64
        );
    }

    println!("\n=== Fraction of multi-spectrum InChIKeys with AT LEAST ONE qualifying pair (best case) ===");
    for (idx, eps) in EPS_VALUES.iter().enumerate() {
        println!(
            "eps={eps:.2}: {}/{groups_processed} groups ({:.2}%) have >=1 pair clearing the threshold",
            groups_with_any_hit[idx],
            100.0 * groups_with_any_hit[idx] as f64 / groups_processed as f64
        );
    }
}
