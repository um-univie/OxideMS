//! Do hub spectra bridge unrelated clusters, and would mutual k-NN or CSLS
//! rescoring remove the bridges? Measured on the PXD000561 pilot with peptide
//! identifications as ground truth.
//!
//! Builds the same graph the engine clusters on -- hashed, L2-normalized
//! spectra, pairs restricted to one charge and the precursor tolerance --
//! down to a low cosine floor, then compares edge rules on it:
//!   cos      keep a pair when cosine >= t (what the engine does)
//!   mutual-k additionally require each end to be in the other's top k
//!   CSLS-k   score 2 cos(x,y) - r_k(x) - r_k(y), r_k the mean cosine to the
//!            k nearest neighbours, thresholded to keep as many pairs as cos
//!
//!   knn-k    require the pair to be in at least one end's top k
//!   snn-j    require the two ends' neighbourhoods (each including itself,
//!            at the same threshold) to overlap with Jaccard index >= j
//!
//! Every rule is scored with the revision's definitions (`reanalyze.py`):
//! clustering rate (spectra in clusters of two or more over all parsed
//! spectra), annotated spectrum purity, pair error (share of co-clustered
//! labelled pairs with different peptides) and identity completeness.
//!
//! `cargo run --release --example hubness_probe -- [data_dir [annotations.csv [cosine_floor [out.csv]]]]`

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::clustering::k_cluster_ivf::{WINDOWED_EDGES_BLOCK_SIZE, windowed_mass_edges};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};


/// (source file, scan) -> peptide with I/L merged, from USIs of the form
/// `mzspec:<pxd>:<file>:scan:<n>`.
fn load_ground_truth(path: &str) -> HashMap<(String, usize), String> {
    let reader = BufReader::new(std::fs::File::open(path).expect("annotations csv"));
    reader
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.ok()?;
            let mut parts = line.splitn(3, ',');
            let usi: Vec<&str> = parts.next()?.split(':').collect();
            let peptide = parts.next()?.replace('I', "L");
            Some(((usi.get(2)?.to_string(), usi.get(4)?.parse().ok()?), peptide))
        })
        .collect()
}

struct Graph {
    /// (i, j, cosine) with i < j, every pair at or above the cosine floor.
    edges: Vec<(u32, u32, f32)>,
    /// Per edge, j's rank among i's neighbours and i's among j's.
    ranks: Vec<(u32, u32)>,
    /// Per node, neighbours sorted by descending cosine.
    neighbours: Vec<Vec<(u32, f32)>>,
    /// Peptide id per node, or `u32::MAX` when unidentified.
    identity: Vec<u32>,
}

impl Graph {
    /// Mean cosine to the k nearest neighbours; neighbours below the floor
    /// were never stored and count as 0.
    fn local_scale(&self, k: usize) -> Vec<f32> {
        self.neighbours
            .iter()
            .map(|n| n.iter().take(k).map(|&(_, s)| s).sum::<f32>() / k as f32)
            .collect()
    }
    fn labelled(&self, i: u32, j: u32) -> Option<bool> {
        let (a, b) = (self.identity[i as usize], self.identity[j as usize]);
        (a != u32::MAX && b != u32::MAX).then_some(a == b)
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[derive(Clone)]
struct Score {
    rule: String,
    threshold: f32,
    edges: usize,
    clustering_rate: f64,
    purity: f64,
    pair_error: f64,
    identity_completeness: f64,
    largest: usize,
    most_peptides: usize,
}

/// Connected components of the kept edges -- what DBSCAN with min_pts = 2
/// returns -- scored as `reanalyze.py::score` does. Only clusters of two or
/// more spectra count as clustered; a peptide's best cluster may be a
/// singleton, which then recovers one spectrum.
fn score_clusters(
    rule: &str,
    threshold: f32,
    graph: &Graph,
    raw_total: usize,
    kept: impl Iterator<Item = (u32, u32)>,
) -> Score {
    let n = graph.identity.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut edges = 0usize;
    for (i, j) in kept {
        edges += 1;
        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
        parent[a as usize] = b;
    }
    let mut per_cluster: HashMap<u32, HashMap<u32, u64>> = HashMap::new();
    let mut sizes: HashMap<u32, usize> = HashMap::new();
    for node in 0..n as u32 {
        let root = find(&mut parent, node);
        *sizes.entry(root).or_default() += 1;
        let id = graph.identity[node as usize];
        if id != u32::MAX {
            *per_cluster.entry(root).or_default().entry(id).or_default() += 1;
        }
    }
    let clustered: usize = sizes.values().filter(|&&s| s > 1).sum();
    let largest = sizes.values().copied().max().unwrap_or(0);
    let pairs = |m: u64| m * m.saturating_sub(1) / 2;
    let (mut majority, mut members, mut pair_total, mut pair_correct) = (0u64, 0u64, 0u64, 0u64);
    let mut most_peptides = 0usize;
    let mut total: HashMap<u32, u64> = HashMap::new();
    let mut best: HashMap<u32, u64> = HashMap::new();
    for (root, counts) in &per_cluster {
        let all: u64 = counts.values().sum();
        for (&id, &count) in counts {
            *total.entry(id).or_default() += count;
            // A singleton still "recovers" its one spectrum.
            let recovered = if sizes[root] > 1 { count } else { 1 };
            let slot = best.entry(id).or_default();
            *slot = (*slot).max(recovered);
        }
        if sizes[root] > 1 {
            majority += *counts.values().max().unwrap();
            members += all;
            pair_total += pairs(all);
            pair_correct += counts.values().map(|&c| pairs(c)).sum::<u64>();
            most_peptides = most_peptides.max(counts.len());
        }
    }
    let (mut recovered, mut eligible) = (0u64, 0u64);
    for (id, &count) in &total {
        if count >= 2 {
            recovered += best[id];
            eligible += count;
        }
    }
    let score = Score {
        rule: rule.to_string(),
        threshold,
        edges,
        clustering_rate: clustered as f64 / raw_total as f64,
        purity: majority as f64 / members.max(1) as f64,
        pair_error: 1.0 - pair_correct as f64 / pair_total.max(1) as f64,
        identity_completeness: recovered as f64 / eligible.max(1) as f64,
        largest,
        most_peptides,
    };
    println!(
        "  {rule:<16} edges {edges:>10}  clustering rate {:.4}  purity {:.5}  pair error {:.5}  identity completeness {:.4}  largest {largest:>6}  most peptides in one {most_peptides:>4}",
        score.clustering_rate, score.purity, score.pair_error, score.identity_completeness,
    );
    score
}

fn skewness(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let (m2, m3) = values.iter().fold((0.0, 0.0), |(m2, m3), v| {
        let d = v - mean;
        (m2 + d * d, m3 + d * d * d)
    });
    (m3 / n) / (m2 / n).powf(1.5)
}

mod support;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "tests/PXD000561_pilot".to_string());
    let truth_path = args.next().unwrap_or_else(|| format!("{dir}/overall_spectrum_annotations.csv"));
    let floor: f32 = args.next().map_or(0.3, |a| a.parse().expect("cosine floor"));
    let out_path = args.next();
    support::require("hubness_probe", &[(&dir, support::PXD000561), (&truth_path, "per-spectrum peptide annotations CSV (second argument)")]);
    let truth = load_ground_truth(&truth_path);

    let mut experiment = SpectralExperiment::proteomics();
    experiment.ingest_files_parallel(&dir);
    let raw_total = experiment.total_spectra_ingested;
    let dim = experiment.engine.hash_size();
    let tolerance = experiment.engine.mz_tolerance;

    // Sorted by (charge, neutral mass), as the engine's batches are.
    let mut order: Vec<usize> = (0..experiment.spectra.len()).collect();
    order.sort_by(|&a, &b| {
        let (sa, sb) = (&experiment.spectra[a], &experiment.spectra[b]);
        (sa.charge(), sa.get_neutral_mass()).partial_cmp(&(sb.charge(), sb.get_neutral_mass())).unwrap()
    });
    let n = order.len();
    let mut peptide_ids: HashMap<&String, u32> = HashMap::new();
    let identity: Vec<u32> = order
        .iter()
        .map(|&s| {
            let meta = &experiment.metadata[experiment.spectra[s].id as usize];
            let key = (experiment.file_names[meta.file_index].clone(), meta.scan_number.unwrap_or(usize::MAX));
            match truth.get(&key) {
                Some(peptide) => {
                    let next = peptide_ids.len() as u32;
                    *peptide_ids.entry(peptide).or_insert(next)
                }
                None => u32::MAX,
            }
        })
        .collect();
    let identified = identity.iter().filter(|&&id| id != u32::MAX).count();
    println!(
        "{raw_total} spectra parsed, {n} after preprocessing, {identified} identified ({} peptides), dim {dim}, precursor tolerance {tolerance:?}",
        peptide_ids.len()
    );

    let mut vectors = vec![0.0f32; n * dim];
    let hasher = &experiment.engine.hasher;
    vectors.par_chunks_exact_mut(dim).zip(order.par_iter()).for_each(|(buf, &s)| {
        hasher.into_vector_native(experiment.spectra[s].spectrum.peaks.as_slice(), true, buf);
    });

    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    let mut start = 0;
    while start < n {
        let charge = experiment.spectra[order[start]].charge();
        let end = (start..n).find(|&i| experiment.spectra[order[i]].charge() != charge).unwrap_or(n);
        let masses: Vec<f64> =
            order[start..end].iter().map(|&s| experiment.spectra[s].get_neutral_mass()).collect();
        let found = windowed_mass_edges(
            &vectors[start * dim..end * dim],
            dim,
            &masses,
            tolerance,
            1.0 - floor,
            WINDOWED_EDGES_BLOCK_SIZE,
            usize::MAX,
        );
        edges.extend(found.iter().map(|e| ((start + e.source_id) as u32, (start + e.target_id) as u32, e.score)));
        start = end;
    }
    drop(vectors);
    let mut tagged: Vec<Vec<(u32, f32, u32)>> = vec![Vec::new(); n];
    for (e, &(i, j, s)) in edges.iter().enumerate() {
        tagged[i as usize].push((j, s, e as u32));
        tagged[j as usize].push((i, s, e as u32));
    }
    tagged.par_iter_mut().for_each(|list| list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)));
    let mut ranks = vec![(0u32, 0u32); edges.len()];
    for (node, list) in tagged.iter().enumerate() {
        for (rank, &(_, _, e)) in list.iter().enumerate() {
            if edges[e as usize].0 as usize == node {
                ranks[e as usize].0 = rank as u32;
            } else {
                ranks[e as usize].1 = rank as u32;
            }
        }
    }
    let neighbours: Vec<Vec<(u32, f32)>> =
        tagged.into_iter().map(|list| list.into_iter().map(|(j, s, _)| (j, s)).collect()).collect();
    let graph = Graph { edges, ranks, neighbours, identity };
    println!("{} candidate pairs with cosine >= {floor} inside the precursor window\n", graph.edges.len());

    // --- 1. Is there hubness? k-occurrence: how often a spectrum appears in
    // other spectra's top-k lists. Skewness well above 1 is the usual sign.
    println!("k-occurrence over the mass-windowed k-NN graph (spectra with at least one neighbour):");
    for k in [5usize, 10, 20] {
        let mut occurrences = vec![0u32; n];
        for list in &graph.neighbours {
            for &(j, _) in list.iter().take(k) {
                occurrences[j as usize] += 1;
            }
        }
        let active: Vec<f64> = (0..n)
            .filter(|&i| !graph.neighbours[i].is_empty())
            .map(|i| occurrences[i] as f64)
            .collect();
        let max = active.iter().cloned().fold(0.0, f64::max);
        println!(
            "  k = {k:>2}: mean {:.2}  max {max}  skewness {:.2}  (max possible in-list count is unbounded; out-degree capped at k)",
            active.iter().sum::<f64>() / active.len() as f64,
            skewness(&active)
        );
    }

    // --- 2. Do high-degree spectra carry the false edges?
    println!("\nlabelled pairs by the larger endpoint degree (degree counted at the same threshold):");
    for t in [0.95f32, 0.8, 0.6] {
        let degree: Vec<usize> =
            graph.neighbours.iter().map(|l| l.partition_point(|&(_, s)| s >= t)).collect();
        let mut sorted: Vec<usize> = degree.iter().copied().filter(|&d| d > 0).collect();
        sorted.sort_unstable();
        if sorted.is_empty() {
            continue;
        }
        let quantile = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
        let cuts = [quantile(0.5), quantile(0.9), quantile(0.99)];
        let mut buckets = [(0usize, 0usize); 4];
        for &(i, j, s) in &graph.edges {
            if s < t {
                continue;
            }
            let Some(same) = graph.labelled(i, j) else { continue };
            let d = degree[i as usize].max(degree[j as usize]);
            let bucket = cuts.iter().filter(|&&c| d > c).count();
            if same {
                buckets[bucket].0 += 1;
            } else {
                buckets[bucket].1 += 1;
            }
        }
        println!("  cos >= {t}: degree median {} p90 {} p99 {} max {}", cuts[0], cuts[1], cuts[2], sorted[sorted.len() - 1]);
        for (label, (same, different)) in ["<= median", "median..p90", "p90..p99", "> p99"].iter().zip(buckets) {
            println!(
                "    {label:<12} true {same:>8}  false {different:>6}  false share {:>6.2}%",
                100.0 * different as f64 / (same + different).max(1) as f64
            );
        }
    }

    // --- 3. Clusters under each edge rule.
    let mut scores: Vec<Score> = Vec::new();
    let thresholds: Vec<f32> = [0.98f32, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
        .into_iter()
        .filter(|&t| t >= floor)
        .collect();
    for &t in &thresholds {
        println!("\nthreshold {t}:");
        let cos_kept = graph.edges.iter().filter(|e| e.2 >= t).count();
        scores.push(score_clusters(
            "cos",
            t,
            &graph,
            raw_total,
            graph.edges.iter().filter(|e| e.2 >= t).map(|e| (e.0, e.1)),
        ));
        let ranked = || graph.edges.iter().zip(&graph.ranks).filter(move |(e, _)| e.2 >= t);
        scores.push(score_clusters(
            "cos + knn-64",
            t,
            &graph,
            raw_total,
            ranked().filter(|(_, r)| r.0 < 64 || r.1 < 64).map(|(e, _)| (e.0, e.1)),
        ));
        for k in [5u32, 10, 20, 64] {
            scores.push(score_clusters(
                &format!("cos + mutual-{k}"),
                t,
                &graph,
                raw_total,
                ranked().filter(|(_, r)| r.0 < k && r.1 < k).map(|(e, _)| (e.0, e.1)),
            ));
        }

        // Closed neighbourhoods at this threshold, sorted by id, and the
        // Jaccard overlap of the two ends of every edge.
        let closed: Vec<Vec<u32>> = graph
            .neighbours
            .par_iter()
            .enumerate()
            .map(|(i, list)| {
                let mut ids: Vec<u32> =
                    list[..list.partition_point(|&(_, s)| s >= t)].iter().map(|&(j, _)| j).collect();
                ids.push(i as u32);
                ids.sort_unstable();
                ids
            })
            .collect();
        let jaccard: Vec<f32> = graph
            .edges
            .par_iter()
            .map(|&(i, j, s)| {
                if s < t {
                    return 0.0;
                }
                let (a, b) = (&closed[i as usize], &closed[j as usize]);
                let (mut x, mut y, mut shared) = (0, 0, 0usize);
                while x < a.len() && y < b.len() {
                    match a[x].cmp(&b[y]) {
                        std::cmp::Ordering::Less => x += 1,
                        std::cmp::Ordering::Greater => y += 1,
                        std::cmp::Ordering::Equal => {
                            shared += 1;
                            x += 1;
                            y += 1;
                        }
                    }
                }
                shared as f32 / (a.len() + b.len() - shared) as f32
            })
            .collect();
        for min_jaccard in [0.05f32, 0.1, 0.2, 0.3, 0.5] {
            scores.push(score_clusters(
                &format!("cos + snn-{min_jaccard}"),
                t,
                &graph,
                raw_total,
                graph
                    .edges
                    .iter()
                    .zip(&jaccard)
                    .filter(|(e, jac)| e.2 >= t && **jac >= min_jaccard)
                    .map(|(e, _)| (e.0, e.1)),
            ));
        }

        let scale = graph.local_scale(10);
        let mut scored: Vec<(f32, u32, u32)> = graph
            .edges
            .iter()
            .map(|&(i, j, s)| (2.0 * s - scale[i as usize] - scale[j as usize], i, j))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        scores.push(score_clusters(
            "CSLS-10 (top n)",
            t,
            &graph,
            raw_total,
            scored.iter().take(cos_kept).map(|e| (e.1, e.2)),
        ));
    }

    // --- 4. Matched comparison: the highest clustering rate each rule
    // reaches while holding a purity floor, and its pair error there.
    let mut rules: Vec<String> = Vec::new();
    for score in &scores {
        if !rules.contains(&score.rule) {
            rules.push(score.rule.clone());
        }
    }
    for floor in [0.99f64, 0.995] {
        println!("\nbest clustering rate with annotated purity >= {floor}:");
        for rule in &rules {
            let best = scores
                .iter()
                .filter(|s| &s.rule == rule && s.purity >= floor)
                .max_by(|a, b| a.clustering_rate.total_cmp(&b.clustering_rate));
            match best {
                Some(s) => println!(
                    "  {rule:<16} t = {:<4}  clustering rate {:.4}  purity {:.5}  pair error {:.5}  identity completeness {:.4}  largest {}",
                    s.threshold, s.clustering_rate, s.purity, s.pair_error, s.identity_completeness, s.largest
                ),
                None => println!("  {rule:<16} not reached"),
            }
        }
    }
    if let Some(path) = out_path {
        let mut csv = String::from(
            "rule,threshold,edges,clustering_rate,purity,pair_error,identity_completeness,largest,most_peptides\n",
        );
        for s in &scores {
            csv += &format!(
                "{},{},{},{},{},{},{},{},{}\n",
                s.rule, s.threshold, s.edges, s.clustering_rate, s.purity, s.pair_error,
                s.identity_completeness, s.largest, s.most_peptides
            );
        }
        std::fs::write(&path, csv).expect("write csv");
    }
}
