//! What joins two different peptides into one cluster?
//!
//! For the clusters contributing the most wrong labelled pairs, takes the
//! majority peptide A and the largest foreign peptide B, rebuilds the
//! cluster's threshold graph over all of its members (labelled or not) and
//! finds the shortest chain of links from an A spectrum to a B spectrum.
//! Each spectrum on the chain is then annotated against both peptides: the
//! share of its intensity matching A's b/y fragment ions and B's. A spectrum
//! explained by both is chimeric (two co-isolated precursors); one explained
//! by neither is uninformative; an A-like spectrum linked straight to a
//! B-like one means the two peptides simply fragment alike.
//!
//! Also rescoring: identity completeness when "same identity" means the same
//! peptide, the same peptide and charge, or the same peptide, charge and
//! precursor mass (within tolerance) -- only the last can ever be clustered
//! together, since spectra are never linked across charge or mass.
//!
//! `cargo run --release --example bridge_audit -- <data_dir> <labels.csv> <falcon.csv> <falcon_eps>`

use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::spectrum::MSNSpectrum;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};

const PROTON: f64 = 1.007276;
const WATER: f64 = 18.010565;

fn residue(c: u8) -> f64 {
    match c {
        b'G' => 57.02146, b'A' => 71.03711, b'S' => 87.03203, b'P' => 97.05276, b'V' => 99.06841,
        b'T' => 101.04768, b'C' => 103.00919 + 57.02146, b'L' | b'I' => 113.08406, b'N' => 114.04293,
        b'D' => 115.02694, b'Q' => 128.05858, b'K' => 128.09496, b'E' => 129.04259, b'M' => 131.04049,
        b'H' => 137.05891, b'F' => 147.06841, b'R' => 156.10111, b'Y' => 163.06333, b'W' => 186.07931,
        _ => 0.0,
    }
}

/// Sorted b and y ion m/z values, singly charged, plus doubly charged when
/// the precursor carries three or more charges. `oxidized` adds 15.9949 to
/// every methionine.
fn fragment_ions(peptide: &str, charge: i8, oxidized: bool) -> Vec<f64> {
    let masses: Vec<f64> = peptide
        .bytes()
        .map(|c| residue(c) + if oxidized && c == b'M' { 15.994915 } else { 0.0 })
        .collect();
    let total: f64 = masses.iter().sum();
    let mut ions = Vec::new();
    let mut prefix = 0.0;
    for mass in &masses[..masses.len() - 1] {
        prefix += mass;
        for (neutral, _) in [(prefix, 'b'), (total - prefix + WATER, 'y')] {
            ions.push(neutral + PROTON);
            if charge >= 3 {
                ions.push((neutral + 2.0 * PROTON) / 2.0);
            }
        }
    }
    ions.sort_unstable_by(|a, b| a.total_cmp(b));
    ions
}

/// Share of the spectrum's intensity within `tolerance` Da of a b/y ion of
/// `peptide`, taking the better of the unmodified and oxidized forms.
fn explained(spectrum: &MSNSpectrum, peptide: &str, tolerance: f64) -> f32 {
    let peaks = &spectrum.spectrum.peaks;
    let total: f32 = peaks.iter().map(|p| p.intensity).sum();
    let variants: &[bool] = if peptide.contains('M') { &[false, true] } else { &[false] };
    variants
        .iter()
        .map(|&oxidized| {
            let ions = fragment_ions(peptide, spectrum.charge(), oxidized);
            let matched: f32 = peaks
                .iter()
                .filter(|p| {
                    let at = ions.partition_point(|&m| m < p.mz - tolerance);
                    at < ions.len() && ions[at] <= p.mz + tolerance
                })
                .map(|p| p.intensity)
                .sum();
            matched / total.max(f32::MIN_POSITIVE)
        })
        .fold(0.0, f32::max)
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// Shortest chain from any `sources` node to any `targets` node.
fn shortest_path(adj: &[Vec<u32>], sources: &[usize], targets: &[bool], removed: &[bool]) -> Option<Vec<usize>> {
    let mut previous = vec![u32::MAX; adj.len()];
    let mut seen = removed.to_vec();
    let mut queue = VecDeque::new();
    for &s in sources {
        if !seen[s] {
            seen[s] = true;
            queue.push_back(s);
        }
    }
    while let Some(node) = queue.pop_front() {
        if targets[node] {
            let mut path = vec![node];
            let mut at = node;
            while previous[at] != u32::MAX {
                at = previous[at] as usize;
                path.push(at);
            }
            path.reverse();
            return Some(path);
        }
        for &next in &adj[node] {
            let next = next as usize;
            if !seen[next] {
                seen[next] = true;
                previous[next] = node as u32;
                queue.push_back(next);
            }
        }
    }
    None
}

struct Run {
    name: String,
    threshold: f32,
    cluster_of: Vec<i64>,
}

mod support;

const USAGE: &str = "usage: bridge_audit <data dir> <labels.csv> <falcon clusters.csv> <falcon eps>";

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| support::fail(USAGE));
    let labels_path = args.next().unwrap_or_else(|| support::fail(USAGE));
    let falcon_path = args.next().unwrap_or_else(|| support::fail(USAGE));
    let falcon_eps: f32 = args.next().unwrap_or_else(|| support::fail(USAGE)).parse().unwrap();

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
    let fragment_tolerance = match experiment.engine.hasher.tolerance {
        oxidems_lib::workflow::MassTolerance::Da(t) => t,
        oxidems_lib::workflow::MassTolerance::Ppm(_) => 0.05,
    };
    let precursor_tolerance = experiment.engine.mz_tolerance;

    let mut slot_of = vec![u32::MAX; n_ids];
    for (slot, spectrum) in snapshot.iter().enumerate() {
        slot_of[spectrum.id as usize] = slot as u32;
    }
    let mut key_to_id: HashMap<(String, usize), usize> = HashMap::new();
    let mut peptide_ids: HashMap<String, u32> = HashMap::new();
    let mut peptides: Vec<String> = Vec::new();
    let mut identity = vec![u32::MAX; n_ids];
    for (id, meta) in experiment.metadata.iter().enumerate() {
        let (Some(file), Some(scan)) = (experiment.file_names.get(meta.file_index), meta.scan_number) else { continue };
        key_to_id.insert((file.clone(), scan), id);
        if slot_of[id] == u32::MAX {
            continue;
        }
        if let Some(peptide) = truth.get(&(file.clone(), scan)) {
            let next = peptides.len() as u32;
            identity[id] = *peptide_ids.entry(peptide.clone()).or_insert_with(|| {
                peptides.push(peptide.clone());
                next
            });
        }
    }

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
    let mut runs = vec![falcon];
    for (epsilon, overlap) in [(0.2f32, 0.0f32), (0.3, 0.5), (0.5, 0.5)] {
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

    // ---------- identity completeness under three definitions ----------
    // Mass groups: labelled spectra of one peptide and charge, sorted by
    // precursor m/z, cut wherever the gap exceeds the precursor tolerance.
    let labelled: Vec<usize> = (0..n_ids).filter(|&id| identity[id] != u32::MAX).collect();
    let spectrum_of = |id: usize| &snapshot[slot_of[id] as usize];
    let mut by_peptide_charge: HashMap<(u32, i8), Vec<usize>> = HashMap::new();
    for &id in &labelled {
        by_peptide_charge.entry((identity[id], spectrum_of(id).charge())).or_default().push(id);
    }
    let mut mass_group = vec![0u32; n_ids];
    let mut n_mass_groups = 0u32;
    for ids in by_peptide_charge.values_mut() {
        ids.sort_unstable_by(|&a, &b| spectrum_of(a).precursor_mz.total_cmp(&spectrum_of(b).precursor_mz));
        for (i, &id) in ids.iter().enumerate() {
            if i > 0 {
                let (lo, hi) = (spectrum_of(ids[i - 1]).precursor_mz, spectrum_of(id).precursor_mz);
                if hi - lo > precursor_tolerance.tolerance_at(hi) {
                    n_mass_groups += 1;
                }
            }
            mass_group[id] = n_mass_groups;
        }
        n_mass_groups += 1;
    }
    println!(
        "{} labelled spectra kept: {} peptides, {} peptide+charge groups, {} peptide+charge+mass groups",
        labelled.len(),
        peptides.len(),
        by_peptide_charge.len(),
        n_mass_groups
    );
    println!("\nidentity completeness by definition of 'same identity' (groups with >= 2 spectra):");
    println!("  {:<34} {:>10} {:>16} {:>22}", "run", "peptide", "peptide+charge", "peptide+charge+mass");
    for run in &runs {
        let mut scores = Vec::new();
        for level in 0..3 {
            let key = |id: usize| -> (u32, i8, u32) {
                match level {
                    0 => (identity[id], 0, 0),
                    1 => (identity[id], spectrum_of(id).charge(), 0),
                    _ => (identity[id], spectrum_of(id).charge(), mass_group[id]),
                }
            };
            let mut total: HashMap<(u32, i8, u32), u64> = HashMap::new();
            let mut per_cluster: HashMap<((u32, i8, u32), i64), u64> = HashMap::new();
            for &id in &labelled {
                *total.entry(key(id)).or_default() += 1;
                if run.cluster_of[id] >= 0 {
                    *per_cluster.entry((key(id), run.cluster_of[id])).or_default() += 1;
                }
            }
            let mut best: HashMap<(u32, i8, u32), u64> = HashMap::new();
            for ((k, _), &count) in &per_cluster {
                let slot = best.entry(*k).or_default();
                *slot = (*slot).max(count);
            }
            let (mut recovered, mut eligible) = (0u64, 0u64);
            for (k, &t) in &total {
                if t >= 2 {
                    eligible += t;
                    recovered += best.get(k).copied().unwrap_or(0);
                }
            }
            scores.push(recovered as f64 / eligible.max(1) as f64);
        }
        println!("  {:<34} {:>10.4} {:>16.4} {:>22.4}", run.name, scores[0], scores[1], scores[2]);
    }

    // ---------- bridges ----------
    let hasher = &experiment.engine.hasher;
    for run in [&runs[1], &runs[3]] {
        let t = run.threshold;
        println!("\n================ bridges in {} ================", run.name);
        let mut members_of: HashMap<i64, Vec<usize>> = HashMap::new();
        for id in 0..n_ids {
            if run.cluster_of[id] >= 0 {
                members_of.entry(run.cluster_of[id]).or_default().push(id);
            }
        }
        let pairs = |m: usize| (m * m.saturating_sub(1) / 2) as f64;
        let mut mixed: Vec<(f64, i64, u32, u32)> = Vec::new();
        for (&c, members) in &members_of {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for &id in members {
                if identity[id] != u32::MAX {
                    *counts.entry(identity[id]).or_default() += 1;
                }
            }
            if counts.len() < 2 {
                continue;
            }
            let all: usize = counts.values().sum();
            let wrong = pairs(all) - counts.values().map(|&n| pairs(n)).sum::<f64>();
            let mut top: Vec<(usize, u32)> = counts.iter().map(|(&p, &n)| (n, p)).collect();
            top.sort_unstable_by(|a, b| b.cmp(a));
            mixed.push((wrong, c, top[0].1, top[1].1));
        }
        mixed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let total_wrong: f64 = mixed.iter().map(|m| m.0).sum();
        let audited = mixed.len().min(200);
        let covered: f64 = mixed.iter().take(audited).map(|m| m.0).sum();
        println!(
            "{} mixed clusters; auditing the top {audited}, which hold {:.1}% of wrong pairs",
            mixed.len(),
            100.0 * covered / total_wrong.max(1.0)
        );

        let mut kinds: HashMap<&'static str, (usize, f64)> = HashMap::new();
        let mut lengths: HashMap<usize, usize> = HashMap::new();
        let mut bridge_counts: HashMap<&'static str, usize> = HashMap::new();
        let (mut own_all, mut cross_all) = (Vec::new(), Vec::new());
        // Foreign-labelled (B) spectra by what their fragments match, weighted
        // by the wrong pairs each one creates (one per A spectrum).
        let mut foreign: HashMap<&'static str, (usize, f64)> = HashMap::new();
        for (rank, &(wrong, c, pa, pb)) in mixed.iter().take(audited).enumerate() {
            let members = &members_of[&c];
            let k = members.len();
            let mut vectors = vec![0.0f32; k * dim];
            vectors.par_chunks_exact_mut(dim).zip(members.par_iter()).for_each(|(buf, &id)| {
                hasher.into_vector_native(spectrum_of(id).spectrum.peaks.as_slice(), true, buf);
            });
            let adj: Vec<Vec<u32>> = (0..k)
                .into_par_iter()
                .map(|i| {
                    let a = &vectors[i * dim..(i + 1) * dim];
                    (0..k)
                        .filter(|&j| {
                            j != i && {
                                let b = &vectors[j * dim..(j + 1) * dim];
                                a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() >= t
                            }
                        })
                        .map(|j| j as u32)
                        .collect()
                })
                .collect();
            let (a_seq, b_seq) = (&peptides[pa as usize], &peptides[pb as usize]);
            let fractions: Vec<(f32, f32)> = members
                .par_iter()
                .map(|&id| (explained(spectrum_of(id), a_seq, fragment_tolerance), explained(spectrum_of(id), b_seq, fragment_tolerance)))
                .collect();
            let sources: Vec<usize> = (0..k).filter(|&i| identity[members[i]] == pa).collect();
            let targets: Vec<bool> = (0..k).map(|i| identity[members[i]] == pb).collect();
            for (i, &id) in members.iter().enumerate() {
                if identity[id] == pa {
                    own_all.push(fractions[i].0);
                    cross_all.push(fractions[i].1);
                } else if identity[id] == pb {
                    own_all.push(fractions[i].1);
                    cross_all.push(fractions[i].0);
                }
            }

            let look_of = |fa: f32, fb: f32| -> &'static str {
                let (hi, lo) = (fa.max(fb), fa.min(fb));
                if hi < 0.15 {
                    "matches neither peptide"
                } else if lo >= 0.15 && lo >= 0.4 * hi {
                    "matches both A and B"
                } else if fa > fb {
                    "matches A only (labelled B)"
                } else {
                    "matches B only"
                }
            };
            for (i, &id) in members.iter().enumerate() {
                if identity[id] == pb {
                    let entry = foreign.entry(look_of(fractions[i].0, fractions[i].1)).or_default();
                    entry.0 += 1;
                    entry.1 += sources.len() as f64;
                }
            }

            let mut removed = vec![false; k];
            let Some(path) = shortest_path(&adj, &sources, &targets, &removed) else {
                kinds.entry("no chain found at this threshold").or_default().0 += 1;
                continue;
            };
            // Independent chains: drop the interior of each one found and look again.
            let mut n_bridges = 0usize;
            let mut current = Some(path.clone());
            while let Some(p) = current {
                n_bridges += 1;
                if p.len() <= 2 || n_bridges >= 20 {
                    break;
                }
                for &node in &p[1..p.len() - 1] {
                    removed[node] = true;
                }
                current = shortest_path(&adj, &sources, &targets, &removed);
            }
            *bridge_counts
                .entry(match n_bridges {
                    1 => "1 independent chain",
                    2..=5 => "2-5 independent chains",
                    _ => "more than 5",
                })
                .or_default() += 1;
            *lengths.entry(path.len() - 1).or_default() += 1;

            // What each spectrum on the chain looks like.
            let look = |i: usize| -> &'static str {
                let (fa, fb) = fractions[i];
                let (hi, lo) = (fa.max(fb), fa.min(fb));
                if hi < 0.15 {
                    "neither"
                } else if lo >= 0.15 && lo >= 0.4 * hi {
                    "both"
                } else if fa > fb {
                    "A"
                } else {
                    "B"
                }
            };
            let looks: Vec<&str> = path.iter().map(|&i| look(i)).collect();
            let kind = if looks.contains(&"both") {
                "a chimeric spectrum (matches both peptides) is on the chain"
            } else if looks.windows(2).any(|w| (w[0] == "A" && w[1] == "B") || (w[0] == "B" && w[1] == "A")) {
                "an A-like spectrum links directly to a B-like one"
            } else if looks.contains(&"neither") {
                "the chain passes through spectra matching neither peptide"
            } else {
                "labels and fragments disagree (chain looks like one peptide throughout)"
            };
            let entry = kinds.entry(kind).or_default();
            entry.0 += 1;
            entry.1 += wrong;

            if rank < 6 {
                println!(
                    "\n  #{} cluster of {k} spectra, {wrong:.0} wrong pairs; A = {a_seq} (x{}), B = {b_seq} (x{}); chain of {} links, {n_bridges}{} independent chain(s)",
                    rank + 1,
                    sources.len(),
                    targets.iter().filter(|&&x| x).count(),
                    path.len() - 1,
                    if n_bridges >= 20 { "+" } else { "" }
                );
                for &i in &path {
                    let id = members[i];
                    let meta = &experiment.metadata[id];
                    let label = match identity[id] {
                        u32::MAX => "unlabelled".to_string(),
                        p if p == pa => "A".to_string(),
                        p if p == pb => "B".to_string(),
                        p => peptides[p as usize].clone(),
                    };
                    println!(
                        "      {label:<12} matches A {:>4.0}%  B {:>4.0}%  looks like {:<8} degree {:>4}  m/z {:.4}  rt {:>7.1}s  file {}",
                        100.0 * fractions[i].0,
                        100.0 * fractions[i].1,
                        look(i),
                        adj[i].len(),
                        spectrum_of(id).precursor_mz,
                        meta.retention_time_in_seconds,
                        meta.file_index
                    );
                }
            }
        }
        println!(
            "\n  calibration over the audited clusters' labelled spectra: median share of intensity matching the labelled peptide {:.0}%, the other peptide {:.0}%",
            100.0 * median(&mut own_all),
            100.0 * median(&mut cross_all)
        );
        println!("  the foreign (B-labelled) spectra themselves (spectra, share of the A-B wrong pairs they create):");
        let foreign_total: f64 = foreign.values().map(|v| v.1).sum();
        let mut rows: Vec<_> = foreign.into_iter().collect();
        rows.sort_unstable_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        for (name, (n, wrong)) in rows {
            println!("    {n:>5} spectra  {:>5.1}%  {name}", 100.0 * wrong / foreign_total.max(1.0));
        }
        println!("  what joins A to B (clusters, share of audited wrong pairs):");
        let mut rows: Vec<_> = kinds.into_iter().collect();
        rows.sort_unstable_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        for (name, (n, wrong)) in rows {
            println!("    {n:>4} clusters  {:>5.1}%  {name}", 100.0 * wrong / covered.max(1.0));
        }
        let mut rows: Vec<_> = lengths.into_iter().collect();
        rows.sort_unstable();
        println!("  links on the shortest A-B chain: {}", rows.iter().map(|(l, n)| format!("{l} links x{n}")).collect::<Vec<_>>().join(", "));
        let mut rows: Vec<_> = bridge_counts.into_iter().collect();
        rows.sort_unstable();
        println!("  redundancy: {}", rows.iter().map(|(l, n)| format!("{l} x{n}")).collect::<Vec<_>>().join(", "));
    }
}
