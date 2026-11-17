//! Scores clusterings against peptide labels with the revision's metric
//! definitions (`reanalyze.py::score`) and two refinements of them.
//!
//! Reference metrics:
//!   clustering rate        spectra in clusters over all parsed spectra
//!   purity                 labelled clustered spectra carrying their
//!                          cluster's majority peptide
//!   pair error             co-clustered labelled pairs with different peptides
//!   identity completeness  share of a peptide's labelled spectra in its
//!                          largest cluster (peptides with >= 2 spectra)
//!
//! Refinement 1, identity. Spectra are only ever compared within one charge
//! state and one precursor tolerance, so completeness is also reported with
//! "same identity" meaning peptide + charge, and peptide + charge + precursor
//! mass (labelled spectra of one peptide and charge, sorted by precursor m/z
//! and cut wherever the gap exceeds the precursor tolerance).
//!
//! Refinement 2, mixed spectra. A labelled spectrum is flagged when fragment
//! ions specific to a second labelled peptide, one whose precursor lies in
//! the same isolation window, carry a substantial share of its intensity.
//! Purity and pair error are reported again without the flagged spectra, and
//! pair error once more with a pair counted as correct when one spectrum
//! carries the other's peptide as its second one. Reversed sequences of the
//! same candidates estimate how often the flag fires by chance.
//!
//! ```text
//! cargo run --release --example cluster_rules_eval -- <data_dir> <labels.csv> <out_prefix>
//!     [--csv name=clusters.csv]...  [--eps 0.2,0.3] [--overlap 0,0.5]
//!     [--window 1.0] [--share 0.15] [--ratio 0.4] [--peaks 6]
//!     [--measure jaccard|smaller] [--profile proteomics|metabolomics]
//! ```
//! Writes `<out_prefix>_identity.csv` (one row per labelled spectrum) and
//! `<out_prefix>_scores.csv` (one row per clustering).

use oxidems_lib::clustering::cluster::engine::OverlapMeasure;
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::spectrum::MSNSpectrum;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

const PROTON: f64 = 1.007276;
const WATER: f64 = 18.010565;
const UNLABELLED: u32 = u32::MAX;
/// Second peptides below this share of intensity are not recorded at all.
const RECORD_SHARE: f32 = 0.05;

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
/// the precursor carries three or more charges. Cysteine is
/// carbamidomethylated; `oxidized` adds 15.9949 to every methionine.
fn fragment_ions(peptide: &[u8], charge: i8, oxidized: bool) -> Vec<f64> {
    let masses: Vec<f64> =
        peptide.iter().map(|&c| residue(c) + if oxidized && c == b'M' { 15.994915 } else { 0.0 }).collect();
    let total: f64 = masses.iter().sum();
    let mut ions = Vec::new();
    let mut prefix = 0.0;
    for mass in &masses[..masses.len().saturating_sub(1)] {
        prefix += mass;
        for neutral in [prefix, total - prefix + WATER] {
            ions.push(neutral + PROTON);
            if charge >= 3 {
                ions.push((neutral + 2.0 * PROTON) / 2.0);
            }
        }
    }
    ions.sort_unstable_by(|a, b| a.total_cmp(b));
    ions
}

fn ion_variants(peptide: &[u8], charge: i8) -> Vec<Vec<f64>> {
    let mut variants = vec![fragment_ions(peptide, charge, false)];
    if peptide.contains(&b'M') {
        variants.push(fragment_ions(peptide, charge, true));
    }
    variants
}

fn hits(ions: &[f64], mz: f64, tolerance: f64) -> bool {
    let at = ions.partition_point(|&m| m < mz - tolerance);
    at < ions.len() && ions[at] <= mz + tolerance
}

#[derive(Default, Clone)]
struct Evidence {
    /// Share of intensity on the labelled peptide's b/y ions.
    own_share: f32,
    /// (peptide, share of intensity on its ions that are not the labelled
    /// peptide's), for candidates reaching `RECORD_SHARE`.
    others: Vec<(u32, f32)>,
    /// The same share for the best reversed candidate.
    best_decoy: f32,
    candidates: u32,
}

fn evidence(
    spectrum: &MSNSpectrum,
    own: &[u8],
    candidates: &[u32],
    peptides: &[String],
    tolerance: f64,
    min_peaks: usize,
) -> Evidence {
    let peaks = &spectrum.spectrum.peaks;
    let charge = spectrum.charge();
    let total: f32 = peaks.iter().map(|p| p.intensity).sum::<f32>().max(f32::MIN_POSITIVE);
    let (own_share, own_mask) = ion_variants(own, charge)
        .iter()
        .map(|ions| {
            let mask: Vec<bool> = peaks.iter().map(|p| hits(ions, p.mz as f64, tolerance)).collect();
            let matched: f32 = peaks.iter().zip(&mask).filter(|(_, own)| **own).map(|(p, _)| p.intensity).sum();
            (matched / total, mask)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    let foreign_share = |sequence: &[u8]| -> f32 {
        ion_variants(sequence, charge)
            .iter()
            .map(|ions| {
                let (mut matched, mut count) = (0.0f32, 0usize);
                for (p, _) in peaks.iter().zip(&own_mask).filter(|(p, own)| !**own && hits(ions, p.mz as f64, tolerance)) {
                    matched += p.intensity;
                    count += 1;
                }
                // One dominant peak landing on an ion by chance is not
                // evidence of a second peptide; a ladder of them is.
                if count >= min_peaks { matched / total } else { 0.0 }
            })
            .fold(0.0, f32::max)
    };
    let mut result = Evidence { own_share, candidates: candidates.len() as u32, ..Default::default() };
    for &candidate in candidates {
        let sequence = peptides[candidate as usize].as_bytes();
        let share = foreign_share(sequence);
        if share >= RECORD_SHARE {
            result.others.push((candidate, share));
        }
        // Reversed with the C-terminal residue kept: same mass, same
        // composition, unrelated fragment ladder.
        let mut decoy = sequence.to_vec();
        let last = decoy.len() - 1;
        decoy[..last].reverse();
        if decoy != sequence {
            result.best_decoy = result.best_decoy.max(foreign_share(&decoy));
        }
    }
    result
}

/// What a clustering is scored against.
struct Truth {
    /// Peptide per spectrum id, `UNLABELLED` if none.
    identity: Vec<u32>,
    /// Peptide + charge group and peptide + charge + mass group per spectrum
    /// id; `UNLABELLED` for unlabelled spectra and for labelled ones the
    /// preprocessing dropped.
    charge_group: Vec<u32>,
    mass_group: Vec<u32>,
    /// Second peptides a spectrum is flagged for.
    second: Vec<Vec<u32>>,
    /// Labelled spectra per peptide over the whole label file, so spectra
    /// the preprocessing dropped stay in the denominator.
    peptide_totals: Vec<u64>,
    charge_totals: Vec<u64>,
    mass_totals: Vec<u64>,
    raw_total: usize,
}

#[derive(Default)]
struct Scores {
    clusters: usize,
    largest: usize,
    clustering_rate: f64,
    purity: f64,
    pair_error: f64,
    wrong_pairs: u64,
    completeness_peptide: f64,
    completeness_charge: f64,
    completeness_mass: f64,
    purity_unflagged: f64,
    pair_error_unflagged: f64,
    pair_error_second_allowed: f64,
    flagged_clustered: u64,
}

fn pairs(n: u64) -> u64 {
    n * n.saturating_sub(1) / 2
}

fn completeness(best: &[u64], totals: &[u64]) -> f64 {
    let (mut recovered, mut eligible) = (0u64, 0u64);
    for (b, &t) in best.iter().zip(totals) {
        if t >= 2 {
            recovered += b;
            eligible += t;
        }
    }
    recovered as f64 / eligible.max(1) as f64
}

/// `cluster_of[id]` is the cluster of spectrum `id`, negative when it is in
/// none; `clustered` is how many spectra are in clusters, `clusters` and
/// `largest` describe them.
fn score(truth: &Truth, cluster_of: &[i64], clustered: usize, clusters: usize, largest: usize) -> Scores {
    let mut members: HashMap<i64, Vec<u32>> = HashMap::new();
    for (id, &cluster) in cluster_of.iter().enumerate() {
        if cluster >= 0 && truth.identity[id] != UNLABELLED {
            members.entry(cluster).or_default().push(id as u32);
        }
    }
    let mut best_peptide = vec![0u64; truth.peptide_totals.len()];
    let mut best_charge = vec![0u64; truth.charge_totals.len()];
    let mut best_mass = vec![0u64; truth.mass_totals.len()];
    let (mut majority, mut labelled, mut pair_total, mut pair_correct) = (0u64, 0u64, 0u64, 0u64);
    let (mut clean_majority, mut clean_labelled, mut clean_total, mut clean_correct) = (0u64, 0u64, 0u64, 0u64);
    let (mut allowed, mut flagged) = (0u64, 0u64);
    let mut counts: HashMap<u32, u64> = HashMap::new();
    let mut clean: HashMap<u32, u64> = HashMap::new();
    let mut groups: HashMap<u32, u64> = HashMap::new();
    let mut carries: HashMap<(u32, u32), u64> = HashMap::new();
    for ids in members.values() {
        counts.clear();
        clean.clear();
        carries.clear();
        for &id in ids {
            let peptide = truth.identity[id as usize];
            *counts.entry(peptide).or_default() += 1;
            if truth.second[id as usize].is_empty() {
                *clean.entry(peptide).or_default() += 1;
            } else {
                flagged += 1;
            }
        }
        let all: u64 = counts.values().sum();
        labelled += all;
        majority += counts.values().max().copied().unwrap_or(0);
        pair_total += pairs(all);
        for (&peptide, &count) in &counts {
            pair_correct += pairs(count);
            best_peptide[peptide as usize] = best_peptide[peptide as usize].max(count);
        }
        let clean_all: u64 = clean.values().sum();
        clean_labelled += clean_all;
        clean_majority += clean.values().max().copied().unwrap_or(0);
        clean_total += pairs(clean_all);
        clean_correct += clean.values().map(|&c| pairs(c)).sum::<u64>();

        // Pairs with different labels where one spectrum carries the other's
        // peptide as its second one; a pair that qualifies from both sides
        // is counted once.
        for &id in ids {
            let peptide = truth.identity[id as usize];
            for &second in &truth.second[id as usize] {
                allowed += counts.get(&second).copied().unwrap_or(0);
                *carries.entry((peptide, second)).or_default() += 1;
            }
        }
        for (&(a, b), &count) in &carries {
            if a < b {
                allowed -= count * carries.get(&(b, a)).copied().unwrap_or(0);
            }
        }

        for (group_of, best) in [(&truth.charge_group, &mut best_charge), (&truth.mass_group, &mut best_mass)] {
            groups.clear();
            for &id in ids {
                let group = group_of[id as usize];
                if group != UNLABELLED {
                    *groups.entry(group).or_default() += 1;
                }
            }
            for (&group, &count) in &groups {
                best[group as usize] = best[group as usize].max(count);
            }
        }
    }
    Scores {
        clusters,
        largest,
        clustering_rate: clustered as f64 / truth.raw_total as f64,
        purity: majority as f64 / labelled.max(1) as f64,
        pair_error: 1.0 - pair_correct as f64 / pair_total.max(1) as f64,
        wrong_pairs: pair_total - pair_correct,
        completeness_peptide: completeness(&best_peptide, &truth.peptide_totals),
        completeness_charge: completeness(&best_charge, &truth.charge_totals),
        completeness_mass: completeness(&best_mass, &truth.mass_totals),
        purity_unflagged: clean_majority as f64 / clean_labelled.max(1) as f64,
        pair_error_unflagged: 1.0 - clean_correct as f64 / clean_total.max(1) as f64,
        pair_error_second_allowed: 1.0 - (pair_correct + allowed) as f64 / pair_total.max(1) as f64,
        flagged_clustered: flagged,
    }
}

fn parse_list(text: &str) -> Vec<f32> {
    text.split(',').map(|v| v.parse().expect("number")).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("data dir");
    let labels_path = args.next().expect("labels csv");
    let out_prefix = args.next().expect("output prefix");
    let (mut measure, mut metabolomics) = (OverlapMeasure::Jaccard, false);
    let mut csv_runs: Vec<(String, String)> = Vec::new();
    let (mut epsilons, mut overlaps): (Vec<f32>, Vec<f32>) = (Vec::new(), vec![0.0]);
    let (mut window, mut min_share, mut min_ratio, mut min_peaks) = (1.0f64, 0.15f32, 0.4f32, 6usize);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--csv" => {
                let (name, path) = value.split_once('=').expect("--csv name=path");
                csv_runs.push((name.to_string(), path.to_string()));
            }
            "--eps" => epsilons = parse_list(&value),
            "--overlap" => overlaps = parse_list(&value),
            "--window" => window = value.parse().unwrap(),
            "--share" => min_share = value.parse().unwrap(),
            "--ratio" => min_ratio = value.parse().unwrap(),
            "--peaks" => min_peaks = value.parse().unwrap(),
            "--measure" => {
                measure = match value.as_str() {
                    "jaccard" => OverlapMeasure::Jaccard,
                    "smaller" => OverlapMeasure::Smaller,
                    other => panic!("unknown measure {other}"),
                }
            }
            // Labels are compound identifiers, not peptides: no I/L merging
            // and no search for second peptides.
            "--profile" => metabolomics = value == "metabolomics",
            other => panic!("unknown option {other}"),
        }
    }
    let mut experiment = if metabolomics { SpectralExperiment::metabolomics() } else { SpectralExperiment::proteomics() };
    experiment.engine.overlap_measure = measure;

    // (file, scan) -> (USI, peptide with I/L merged), from USIs of the form
    // `mzspec:<pxd>:<file>:scan:<n>`.
    let mut labels: HashMap<(String, usize), (String, String)> = HashMap::new();
    for line in BufReader::new(std::fs::File::open(&labels_path).expect("labels csv")).lines().skip(1) {
        let line = line.unwrap();
        let mut parts = line.splitn(3, ',');
        let usi_text = parts.next().unwrap();
        let usi: Vec<&str> = usi_text.split(':').collect();
        if let (Some(file), Some(Ok(scan)), Some(peptide)) = (usi.get(2), usi.get(4).map(|s| s.parse()), parts.next()) {
            let peptide = if metabolomics { peptide.trim().to_string() } else { peptide.replace('I', "L") };
            labels.insert((file.to_string(), scan), (usi_text.to_string(), peptide));
        }
    }

    let ingest_started = std::time::Instant::now();
    experiment.ingest_files_parallel(&dir);
    println!("ingestion {:.0}s", ingest_started.elapsed().as_secs_f64());
    // Held here and handed to the engine per run; only copied when more
    // than one run still needs it.
    let snapshot = std::mem::take(&mut experiment.spectra);
    let n_ids = experiment.metadata.len();
    let raw_total = experiment.total_spectra_ingested;
    let fragment_tolerance = match experiment.engine.hasher.tolerance {
        oxidems_lib::workflow::MassTolerance::Da(t) => t,
        oxidems_lib::workflow::MassTolerance::Ppm(_) => 0.05,
    };
    let precursor_tolerance = experiment.engine.mz_tolerance;

    let mut slot_of = vec![u32::MAX; n_ids];
    for (slot, spectrum) in snapshot.iter().enumerate() {
        slot_of[spectrum.id as usize] = slot as u32;
    }
    let spectrum_of = |id: usize| &snapshot[slot_of[id] as usize];

    // Peptide ids and totals over the whole label file.
    let mut peptide_ids: HashMap<&str, u32> = HashMap::new();
    let mut peptides: Vec<String> = Vec::new();
    let mut peptide_totals: Vec<u64> = Vec::new();
    for (_, peptide) in labels.values() {
        let id = *peptide_ids.entry(peptide.as_str()).or_insert_with(|| {
            peptides.push(peptide.clone());
            peptide_totals.push(0);
            peptides.len() as u32 - 1
        });
        peptide_totals[id as usize] += 1;
    }
    let mut key_to_id: HashMap<(String, usize), usize> = HashMap::new();
    let mut identity = vec![UNLABELLED; n_ids];
    let mut usi_of: HashMap<usize, &str> = HashMap::new();
    for (id, meta) in experiment.metadata.iter().enumerate() {
        let (Some(file), Some(scan)) = (experiment.file_names.get(meta.file_index), meta.scan_number) else { continue };
        let key = (file.clone(), scan);
        if let Some((usi, peptide)) = labels.get(&key) {
            identity[id] = peptide_ids[peptide.as_str()];
            usi_of.insert(id, usi.as_str());
        }
        if !csv_runs.is_empty() {
            key_to_id.insert(key, id);
        }
    }

    // Charge groups and mass groups over the labelled spectra that survived
    // preprocessing.
    let kept: Vec<usize> = (0..n_ids).filter(|&id| identity[id] != UNLABELLED && slot_of[id] != u32::MAX).collect();
    let mut by_peptide_charge: HashMap<(u32, i8), Vec<usize>> = HashMap::new();
    for &id in &kept {
        by_peptide_charge.entry((identity[id], spectrum_of(id).charge())).or_default().push(id);
    }
    let mut charge_group = vec![UNLABELLED; n_ids];
    let mut mass_group = vec![UNLABELLED; n_ids];
    let mut charge_totals: Vec<u64> = Vec::new();
    let mut mass_totals: Vec<u64> = Vec::new();
    // Per charge: (mean precursor m/z of a mass group, its peptide), sorted.
    let mut windows: HashMap<i8, Vec<(f64, u32)>> = HashMap::new();
    let mut keys: Vec<(u32, i8)> = by_peptide_charge.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let ids = by_peptide_charge.get_mut(&key).unwrap();
        ids.sort_unstable_by(|&a, &b| spectrum_of(a).precursor_mz.total_cmp(&spectrum_of(b).precursor_mz));
        let (mut sum, mut previous) = (0.0f64, f64::NEG_INFINITY);
        for &id in ids.iter() {
            let mz = spectrum_of(id).precursor_mz as f64;
            if mz - previous > precursor_tolerance.tolerance_at(mz as _) as f64 {
                if let Some(&count) = mass_totals.last().filter(|_| previous.is_finite()) {
                    windows.entry(key.1).or_default().push((sum / count as f64, key.0));
                }
                mass_totals.push(0);
                sum = 0.0;
            }
            *mass_totals.last_mut().unwrap() += 1;
            sum += mz;
            previous = mz;
            mass_group[id] = mass_totals.len() as u32 - 1;
            charge_group[id] = charge_totals.len() as u32;
        }
        windows.entry(key.1).or_default().push((sum / *mass_totals.last().unwrap() as f64, key.0));
        charge_totals.push(ids.len() as u64);
    }
    for list in windows.values_mut() {
        list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }
    println!(
        "{raw_total} spectra parsed, {} kept; {} labels, {} on kept spectra: {} peptides, {} peptide+charge groups, {} peptide+charge+mass groups",
        snapshot.len(),
        labels.len(),
        kept.len(),
        peptides.len(),
        charge_totals.len(),
        mass_totals.len()
    );

    // ---------- second peptides ----------
    let started = std::time::Instant::now();
    let found: Vec<Evidence> = kept
        .par_iter()
        .map(|&id| {
            let spectrum = spectrum_of(id);
            let mz = spectrum.precursor_mz as f64;
            let own = identity[id];
            if metabolomics {
                return Evidence::default();
            }
            let mut candidates: Vec<u32> = windows.get(&spectrum.charge()).map_or(Vec::new(), |list| {
                let from = list.partition_point(|&(m, _)| m < mz - window);
                list[from..].iter().take_while(|&&(m, _)| m <= mz + window).map(|&(_, p)| p).filter(|&p| p != own).collect()
            });
            candidates.sort_unstable();
            candidates.dedup();
            evidence(spectrum, peptides[own as usize].as_bytes(), &candidates, &peptides, fragment_tolerance as f64, min_peaks)
        })
        .collect();
    let passes = |share: f32, own: f32, threshold: f32| share >= threshold && share >= min_ratio * own;
    let mut own_shares: Vec<f32> = found.iter().map(|e| e.own_share).collect();
    own_shares.sort_unstable_by(|a, b| a.total_cmp(b));
    let mean_candidates = found.iter().map(|e| e.candidates as f64).sum::<f64>() / found.len().max(1) as f64;
    println!(
        "\nsecond peptides: candidates share the charge and lie within {window} m/z; {mean_candidates:.1} per spectrum on average ({:.0}s)",
        started.elapsed().as_secs_f64()
    );
    println!(
        "  intensity on the labelled peptide's b/y ions: median {:.3}, lower decile {:.3}",
        own_shares[own_shares.len() / 2],
        own_shares[own_shares.len() / 10]
    );
    println!("  flagged = a second peptide's own ions carry >= threshold of the intensity on >= {min_peaks} peaks, and >= {min_ratio} x the labelled share");
    println!("  {:>9} {:>12} {:>12} {:>24}", "threshold", "real", "reversed", "estimated false share");
    for threshold in [0.05f32, 0.10, 0.15, 0.20, 0.30] {
        let real = found.iter().filter(|e| e.others.iter().any(|&(_, s)| passes(s, e.own_share, threshold))).count();
        let reversed = found.iter().filter(|e| passes(e.best_decoy, e.own_share, threshold)).count();
        println!(
            "  {threshold:>9.2} {:>11.3}% {:>11.3}% {:>23.1}%",
            100.0 * real as f64 / found.len() as f64,
            100.0 * reversed as f64 / found.len() as f64,
            100.0 * reversed as f64 / real.max(1) as f64
        );
    }
    let mut second: Vec<Vec<u32>> = vec![Vec::new(); n_ids];
    for (&id, e) in kept.iter().zip(&found) {
        second[id] = e.others.iter().filter(|&&(_, s)| passes(s, e.own_share, min_share)).map(|&(p, _)| p).collect();
    }
    let n_flagged = second.iter().filter(|s| !s.is_empty()).count();
    println!("  using threshold {min_share}: {n_flagged} of {} labelled spectra flagged", kept.len());

    let mut identity_csv =
        String::from("USI,PeptideSequence,charge,precursor_mz,charge_group,mass_group,own_share,best_reversed_share,second_peptides\n");
    for (&id, e) in kept.iter().zip(&found) {
        let spectrum = spectrum_of(id);
        let seconds: Vec<&str> = second[id].iter().map(|&p| peptides[p as usize].as_str()).collect();
        identity_csv += &format!(
            "{},{},{},{:.5},{},{},{:.4},{:.4},{}\n",
            usi_of[&id],
            peptides[identity[id] as usize],
            spectrum.charge(),
            spectrum.precursor_mz,
            charge_group[id],
            mass_group[id],
            e.own_share,
            e.best_decoy,
            seconds.join(";")
        );
    }
    std::fs::write(format!("{out_prefix}_identity.csv"), identity_csv).expect("write identity csv");

    let truth = Truth { identity, charge_group, mass_group, second, peptide_totals, charge_totals, mass_totals, raw_total };

    // ---------- clusterings ----------
    let mut rows: Vec<(String, f32, f32, Scores)> = Vec::new();
    let report = |name: &str, epsilon: f32, overlap: f32, s: Scores, rows: &mut Vec<(String, f32, f32, Scores)>| {
        println!(
            "{name:<34} rate {:.4}  purity {:.5} (unflagged {:.5})  pair error {:.5} (unflagged {:.5}, second allowed {:.5})  completeness {:.4} / {:.4} / {:.4}  largest {}",
            s.clustering_rate, s.purity, s.purity_unflagged, s.pair_error, s.pair_error_unflagged,
            s.pair_error_second_allowed, s.completeness_peptide, s.completeness_charge, s.completeness_mass, s.largest
        );
        rows.push((name.to_string(), epsilon, overlap, s));
    };
    println!("\ncompleteness columns: peptide / peptide+charge / peptide+charge+mass");
    for (name, path) in &csv_runs {
        let mut cluster_of = vec![-1i64; n_ids];
        let mut sizes: HashMap<i64, usize> = HashMap::new();
        for line in BufReader::new(std::fs::File::open(path).expect("clusters csv")).lines() {
            let line = line.unwrap();
            if line.starts_with('#') || line.starts_with("identifier") {
                continue;
            }
            let mut fields = line.split(',');
            let usi: Vec<&str> = fields.next().unwrap().split(':').collect();
            let cluster: i64 = fields.last().unwrap().trim().parse().unwrap();
            if cluster < 0 {
                continue;
            }
            *sizes.entry(cluster).or_default() += 1;
            if let Some(&id) = key_to_id.get(&(usi[2].to_string(), usi[4].parse().unwrap())) {
                cluster_of[id] = cluster;
            }
        }
        let clustered: usize = sizes.values().sum();
        let largest = sizes.values().max().copied().unwrap_or(0);
        report(name, f32::NAN, f32::NAN, score(&truth, &cluster_of, clustered, sizes.len(), largest), &mut rows);
    }
    let mut snapshot = Some(snapshot);
    let last_run = overlaps.len() * epsilons.len();
    let mut run = 0;
    for &overlap in &overlaps {
        for &epsilon in &epsilons {
            run += 1;
            experiment.spectra =
                if run == last_run { snapshot.take().unwrap() } else { snapshot.as_ref().unwrap().clone() };
            experiment.engine.epsilon = epsilon;
            experiment.engine.min_neighbor_overlap = overlap;
            let cluster_started = std::time::Instant::now();
            experiment.prepare_clustering();
            let clusters = experiment.cluster();
            println!("clustering eps {epsilon} overlap {overlap}: {:.0}s", cluster_started.elapsed().as_secs_f64());
            let mut cluster_of = vec![-1i64; n_ids];
            let (mut clustered, mut n_clusters, mut largest) = (0usize, 0usize, 0usize);
            for (c, cluster) in clusters.iter().enumerate() {
                if cluster.len() < 2 {
                    continue;
                }
                n_clusters += 1;
                clustered += cluster.len();
                largest = largest.max(cluster.len());
                for &id in cluster {
                    cluster_of[id] = c as i64;
                }
            }
            let name = format!("engine eps {epsilon} overlap {overlap}");
            report(&name, epsilon, overlap, score(&truth, &cluster_of, clustered, n_clusters, largest), &mut rows);
        }
    }

    let mut csv = String::from(
        "run,epsilon,overlap,clusters,largest,clustering_rate,purity,pair_error,wrong_pairs,identity_completeness,identity_completeness_charge,identity_completeness_mass,purity_unflagged,pair_error_unflagged,pair_error_second_allowed,flagged_clustered\n",
    );
    for (name, epsilon, overlap, s) in &rows {
        csv += &format!(
            "{name},{epsilon},{overlap},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            s.clusters, s.largest, s.clustering_rate, s.purity, s.pair_error, s.wrong_pairs, s.completeness_peptide,
            s.completeness_charge, s.completeness_mass, s.purity_unflagged, s.pair_error_unflagged,
            s.pair_error_second_allowed, s.flagged_clustered
        );
    }
    std::fs::write(format!("{out_prefix}_scores.csv"), csv).expect("write scores csv");

    for floor in [0.99f64, 0.995] {
        println!("\nhighest clustering rate with purity >= {floor}, per overlap:");
        for &overlap in &overlaps {
            let best = rows
                .iter()
                .filter(|(_, _, o, s)| *o == overlap && s.purity >= floor)
                .max_by(|a, b| a.3.clustering_rate.total_cmp(&b.3.clustering_rate));
            match best {
                Some((name, _, _, s)) => println!(
                    "  {name:<34} rate {:.4}  pair error {:.5}  completeness {:.4} / {:.4} / {:.4}",
                    s.clustering_rate, s.pair_error, s.completeness_peptide, s.completeness_charge, s.completeness_mass
                ),
                None => println!("  overlap {overlap:<4} not reached"),
            }
        }
    }
}
