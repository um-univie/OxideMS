//! Which labelled pairs does a clustering get wrong, and why?
//!
//! Runs the engine at several (epsilon, neighbour-overlap) settings, loads a
//! Falcon assignment table for the same files, and for each breaks the two
//! kinds of error down:
//!
//! wrong merges -- labelled spectra of different peptides in one cluster
//!   * how concentrated they are in a few large clusters
//!   * whether the two spectra are themselves similar (hashed cosine and
//!     collision-free binned cosine at or above the run's threshold) or were
//!     only joined through a chain of other spectra
//!   * whether the two peptides are sequence relatives
//!   * whether Falcon merges the same pair
//!
//! missed merges -- labelled spectra of one peptide not in one cluster
//!   * different charge, precursor outside tolerance (neither can ever link)
//!   * within reach and similar enough, yet split
//!   * within reach but below the threshold
//!
//! Peptides or clusters with more than `SAMPLE` labelled spectra are
//! subsampled and their pairs re-weighted.
//!
//! `cargo run --release --example cluster_error_audit -- <data_dir> <labels.csv> <falcon.csv> <falcon_eps>`

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::clustering::similarities::branching_cosine_sim;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

const SAMPLE: usize = 300;

struct Run {
    name: String,
    threshold: f32,
    /// Cluster per spectrum id; -1 when unclustered.
    cluster_of: Vec<i64>,
}

struct Labelled {
    /// Spectrum ids of labelled spectra that survived preprocessing.
    ids: Vec<usize>,
    peptide: Vec<u32>,
    charge: Vec<i8>,
    precursor_mz: Vec<f64>,
    hashed: Vec<f32>,
    dim: usize,
    bins: Vec<(Vec<u32>, Vec<f32>)>,
}

impl Labelled {
    fn hashed_cos(&self, a: usize, b: usize) -> f32 {
        let (x, y) = (&self.hashed[a * self.dim..(a + 1) * self.dim], &self.hashed[b * self.dim..(b + 1) * self.dim]);
        x.iter().zip(y).map(|(p, q)| p * q).sum()
    }
    fn exact_cos(&self, a: usize, b: usize) -> f32 {
        branching_cosine_sim(&self.bins[a].0, &self.bins[a].1, &self.bins[b].0, &self.bins[b].1)
    }
}

fn sample(items: &[usize], rng: &mut StdRng) -> (Vec<usize>, f64) {
    if items.len() <= SAMPLE {
        return (items.to_vec(), 1.0);
    }
    let mut picked = items.to_vec();
    picked.shuffle(rng);
    picked.truncate(SAMPLE);
    let pairs = |m: usize| (m * (m - 1) / 2) as f64;
    (picked, pairs(items.len()) / pairs(SAMPLE))
}

fn related(a: &str, b: &str) -> &'static str {
    let sorted = |s: &str| {
        let mut c: Vec<u8> = s.bytes().collect();
        c.sort_unstable();
        c
    };
    if sorted(a) == sorted(b) {
        "same composition"
    } else if a.contains(b) || b.contains(a) {
        "one contains the other"
    } else if a.len() == b.len() && a.bytes().zip(b.bytes()).filter(|(x, y)| x != y).count() <= 2 {
        "1-2 substitutions"
    } else {
        "unrelated"
    }
}

fn quantiles(values: &mut [f32]) -> String {
    if values.is_empty() {
        return "-".into();
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let q = |f: f64| values[((values.len() - 1) as f64 * f) as usize];
    format!("p10 {:.2} median {:.2} p90 {:.2}", q(0.1), q(0.5), q(0.9))
}

fn audit(run: &Run, falcon: &Run, lab: &Labelled, peptides: &[String], sizes_of_all: &HashMap<i64, usize>, tolerance_at: &dyn Fn(f64) -> f64) {
    let t = run.threshold;
    let mut rng = StdRng::seed_from_u64(17);
    println!("\n================ {} (threshold {t}) ================", run.name);

    // ---- wrong merges ----
    let mut by_cluster: HashMap<i64, Vec<usize>> = HashMap::new();
    for (pos, &id) in lab.ids.iter().enumerate() {
        let c = run.cluster_of[id];
        if c >= 0 {
            by_cluster.entry(c).or_default().push(pos);
        }
    }
    let pairs = |m: usize| (m * m.saturating_sub(1) / 2) as f64;
    let mut mixed: Vec<(f64, i64)> = Vec::new();
    let (mut total_pairs, mut wrong_pairs) = (0.0, 0.0);
    for (&c, members) in &by_cluster {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &m in members {
            *counts.entry(lab.peptide[m]).or_default() += 1;
        }
        let all = pairs(members.len());
        let wrong = all - counts.values().map(|&n| pairs(n)).sum::<f64>();
        total_pairs += all;
        wrong_pairs += wrong;
        if wrong > 0.0 {
            mixed.push((wrong, c));
        }
    }
    mixed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let share = |k: usize| mixed.iter().take(k).map(|m| m.0).sum::<f64>() / wrong_pairs.max(1.0);
    println!(
        "wrong merges: {wrong_pairs:.0} of {total_pairs:.0} labelled co-clustered pairs (pair error {:.5}), in {} mixed clusters",
        wrong_pairs / total_pairs.max(1.0),
        mixed.len()
    );
    println!(
        "  concentration: top 1 cluster {:.1}%, top 10 {:.1}%, top 100 {:.1}% of wrong pairs",
        100.0 * share(1),
        100.0 * share(10),
        100.0 * share(100)
    );
    println!("  largest contributors:");
    for &(wrong, c) in mixed.iter().take(6) {
        let members = &by_cluster[&c];
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &m in members {
            *counts.entry(lab.peptide[m]).or_default() += 1;
        }
        let mut top: Vec<(usize, u32)> = counts.iter().map(|(&p, &n)| (n, p)).collect();
        top.sort_unstable_by(|a, b| b.cmp(a));
        let (lo, hi) = members.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &m| (lo.min(lab.precursor_mz[m]), hi.max(lab.precursor_mz[m])));
        let listing: Vec<String> = top.iter().take(3).map(|&(n, p)| format!("{} x{n}", peptides[p as usize])).collect();
        println!(
            "    cluster of {} spectra ({} labelled, {} peptides, z={}, m/z {lo:.4}-{hi:.4}): {wrong:.0} wrong pairs; {}",
            sizes_of_all.get(&c).copied().unwrap_or(0),
            members.len(),
            counts.len(),
            lab.charge[members[0]],
            listing.join(", ")
        );
    }

    // Each wrong pair: are the two spectra similar themselves, or chained?
    let mut class: HashMap<&'static str, f64> = HashMap::new();
    let mut relation: HashMap<&'static str, f64> = HashMap::new();
    let (mut in_falcon, mut weighted) = (0.0, 0.0);
    let (mut hashed_of_wrong, mut exact_of_wrong) = (Vec::new(), Vec::new());
    for &(_, c) in &mixed {
        let (members, weight) = sample(&by_cluster[&c], &mut rng);
        for (x, &a) in members.iter().enumerate() {
            for &b in &members[x + 1..] {
                if lab.peptide[a] == lab.peptide[b] {
                    continue;
                }
                let (h, e) = (lab.hashed_cos(a, b), lab.exact_cos(a, b));
                let name = match (h >= t, e >= t) {
                    (true, true) => "similar by both cosines",
                    (true, false) => "similar only after hashing",
                    (false, true) => "similar only before hashing",
                    (false, false) => "chained (pair itself below threshold)",
                };
                *class.entry(name).or_default() += weight;
                *relation.entry(related(&peptides[lab.peptide[a] as usize], &peptides[lab.peptide[b] as usize])).or_default() += weight;
                let (fa, fb) = (falcon.cluster_of[lab.ids[a]], falcon.cluster_of[lab.ids[b]]);
                if fa >= 0 && fa == fb {
                    in_falcon += weight;
                }
                weighted += weight;
                hashed_of_wrong.push(h);
                exact_of_wrong.push(e);
            }
        }
    }
    println!("  the wrong pair itself:");
    let mut rows: Vec<_> = class.into_iter().collect();
    rows.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    for (name, w) in rows {
        println!("    {:>5.1}%  {name}", 100.0 * w / weighted.max(1.0));
    }
    println!("    hashed cosine {}; binned cosine {}", quantiles(&mut hashed_of_wrong), quantiles(&mut exact_of_wrong));
    println!("  the two peptides:");
    let mut rows: Vec<_> = relation.into_iter().collect();
    rows.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    for (name, w) in rows {
        println!("    {:>5.1}%  {name}", 100.0 * w / weighted.max(1.0));
    }
    println!("  {:.1}% of these wrong pairs are also co-clustered by {}", 100.0 * in_falcon / weighted.max(1.0), falcon.name);

    // ---- missed merges ----
    let mut by_peptide: HashMap<u32, Vec<usize>> = HashMap::new();
    for pos in 0..lab.ids.len() {
        by_peptide.entry(lab.peptide[pos]).or_default().push(pos);
    }
    let mut keys: Vec<u32> = by_peptide.keys().copied().collect();
    keys.sort_unstable();
    let mut fate: HashMap<&'static str, f64> = HashMap::new();
    let mut versus: HashMap<&'static str, f64> = HashMap::new();
    let mut falcon_only_cos = Vec::new();
    let mut all_pairs = 0.0;
    for key in keys {
        if by_peptide[&key].len() < 2 {
            continue;
        }
        let (members, weight) = sample(&by_peptide[&key], &mut rng);
        for (x, &a) in members.iter().enumerate() {
            for &b in &members[x + 1..] {
                all_pairs += weight;
                let (ca, cb) = (run.cluster_of[lab.ids[a]], run.cluster_of[lab.ids[b]]);
                let together = ca >= 0 && ca == cb;
                let (fa, fb) = (falcon.cluster_of[lab.ids[a]], falcon.cluster_of[lab.ids[b]]);
                let falcon_together = fa >= 0 && fa == fb;
                let reachable = lab.charge[a] == lab.charge[b] && {
                    let (ma, mb) = (lab.precursor_mz[a], lab.precursor_mz[b]);
                    (ma - mb).abs() <= tolerance_at(ma.max(mb))
                };
                let name = if together {
                    "co-clustered"
                } else if lab.charge[a] != lab.charge[b] {
                    "split: different charge"
                } else if !reachable {
                    "split: precursor m/z outside tolerance"
                } else if lab.hashed_cos(a, b) >= t {
                    "split: within reach and above threshold"
                } else {
                    "split: within reach, below threshold"
                };
                *fate.entry(name).or_default() += weight;
                if reachable {
                    let name = match (together, falcon_together) {
                        (true, true) => "both",
                        (true, false) => "this run only",
                        (false, true) => "other only",
                        (false, false) => "neither",
                    };
                    *versus.entry(name).or_default() += weight;
                    if !together && falcon_together {
                        falcon_only_cos.push(lab.hashed_cos(a, b));
                    }
                }
            }
        }
    }
    println!("same-peptide labelled pairs ({all_pairs:.0}, re-weighted):");
    let mut rows: Vec<_> = fate.into_iter().collect();
    rows.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    for (name, w) in rows {
        println!("    {:>5.1}%  {name}", 100.0 * w / all_pairs.max(1.0));
    }
    let reach: f64 = versus.values().sum();
    println!("  reachable same-peptide pairs, this run vs {}:", falcon.name);
    for name in ["both", "this run only", "other only", "neither"] {
        println!("    {:>5.1}%  {name}", 100.0 * versus.get(name).copied().unwrap_or(0.0) / reach.max(1.0));
    }
    println!("    our hashed cosine of pairs only the other merges: {}", quantiles(&mut falcon_only_cos));
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("data dir");
    let labels_path = args.next().expect("labels csv");
    let falcon_path = args.next().expect("falcon csv");
    let falcon_eps: f32 = args.next().expect("falcon eps").parse().unwrap();

    let mut truth: HashMap<(String, usize), String> = HashMap::new();
    for line in BufReader::new(std::fs::File::open(&labels_path).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let mut parts = line.splitn(3, ',');
        let usi: Vec<&str> = parts.next().unwrap().split(':').collect();
        if let (Some(file), Some(Ok(scan)), Some(peptide)) = (usi.get(2), usi.get(4).map(|s| s.parse()), parts.next()) {
            truth.insert((file.to_string(), scan), peptide.replace('I', "L"));
        }
    }

    let mut experiment = SpectralExperiment::proteomics();
    experiment.ingest_files_parallel(&dir);
    let snapshot = experiment.spectra.clone();
    let n_ids = experiment.metadata.len();
    let dim = experiment.engine.hash_size();
    let fragment_tolerance = experiment.engine.hasher.tolerance;
    let precursor_tolerance = experiment.engine.mz_tolerance;

    let mut key_to_id: HashMap<(String, usize), usize> = HashMap::new();
    for (id, meta) in experiment.metadata.iter().enumerate() {
        if let (Some(file), Some(scan)) = (experiment.file_names.get(meta.file_index), meta.scan_number) {
            key_to_id.insert((file.clone(), scan), id);
        }
    }
    let mut peptide_ids: HashMap<String, u32> = HashMap::new();
    let mut peptides: Vec<String> = Vec::new();
    let mut lab = Labelled { ids: vec![], peptide: vec![], charge: vec![], precursor_mz: vec![], hashed: vec![], dim, bins: vec![] };
    let mut kept_labelled: Vec<usize> = Vec::new();
    for (slot, spectrum) in snapshot.iter().enumerate() {
        let id = spectrum.id as usize;
        let meta = &experiment.metadata[id];
        let (Some(file), Some(scan)) = (experiment.file_names.get(meta.file_index), meta.scan_number) else { continue };
        if let Some(peptide) = truth.get(&(file.clone(), scan)) {
            let next = peptides.len() as u32;
            let pid = *peptide_ids.entry(peptide.clone()).or_insert_with(|| {
                peptides.push(peptide.clone());
                next
            });
            lab.ids.push(id);
            lab.peptide.push(pid);
            lab.charge.push(spectrum.charge());
            lab.precursor_mz.push(spectrum.precursor_mz);
            kept_labelled.push(slot);
        }
    }
    lab.hashed = vec![0.0; kept_labelled.len() * dim];
    let hasher = &experiment.engine.hasher;
    lab.hashed.par_chunks_exact_mut(dim).zip(kept_labelled.par_iter()).for_each(|(buf, &slot)| {
        hasher.into_vector_native(snapshot[slot].spectrum.peaks.as_slice(), true, buf);
    });
    lab.bins = kept_labelled
        .par_iter()
        .map(|&slot| {
            let mut soa = snapshot[slot].spectrum.to_soa(fragment_tolerance);
            soa.l2_normalize();
            (soa.mz_bins, soa.intensities)
        })
        .collect();
    println!("{} labelled spectra kept, {} peptides", lab.ids.len(), peptides.len());

    // Falcon's assignments, joined by (file, scan).
    let mut falcon = Run { name: format!("Falcon eps {falcon_eps}"), threshold: 1.0 - falcon_eps, cluster_of: vec![-1; n_ids] };
    for line in BufReader::new(std::fs::File::open(&falcon_path).unwrap()).lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.starts_with("identifier") {
            continue;
        }
        let mut fields = line.split(',');
        let usi: Vec<&str> = fields.next().unwrap().split(':').collect();
        let cluster: i64 = fields.last().unwrap().parse().unwrap();
        if let Some(&id) = key_to_id.get(&(usi[2].to_string(), usi[4].parse().unwrap())) {
            falcon.cluster_of[id] = cluster;
        }
    }

    let mut runs: Vec<Run> = Vec::new();
    for (epsilon, overlap) in [(0.2f32, 0.0f32), (0.2, 0.5), (0.5, 0.0), (0.5, 0.5)] {
        experiment.spectra = snapshot.clone();
        experiment.engine.epsilon = epsilon;
        experiment.engine.min_neighbor_overlap = overlap;
        experiment.prepare_clustering();
        let clusters = experiment.cluster();
        let mut cluster_of = vec![-1i64; n_ids];
        for (c, cluster) in clusters.iter().enumerate() {
            if cluster.len() >= 2 {
                for &id in cluster {
                    cluster_of[id] = c as i64;
                }
            }
        }
        runs.push(Run { name: format!("RedshiftMS eps {epsilon} overlap {overlap}"), threshold: 1.0 - epsilon, cluster_of });
    }

    let tolerance_at = |mz: f64| precursor_tolerance.tolerance_at(mz);
    let sizes = |run: &Run| {
        let mut sizes: HashMap<i64, usize> = HashMap::new();
        for &c in &run.cluster_of {
            if c >= 0 {
                *sizes.entry(c).or_default() += 1;
            }
        }
        sizes
    };
    audit(&falcon, &runs[0], &lab, &peptides, &sizes(&falcon), &tolerance_at);
    for run in &runs {
        audit(run, &falcon, &lab, &peptides, &sizes(run), &tolerance_at);
    }
}
