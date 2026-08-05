//! Fast, reusable MS-RT proxy purity calculator for any redshift-ms or
//! Falcon cluster-assignment CSV (identifier,precursor_charge,precursor_mz,
//! retention_time,cluster -- Falcon's noise rows, cluster=-1, are skipped).
//! Mirrors generate_fig1_comparison.py's weighted_purity_msrt exactly (same
//! per-cluster, per-file single-linkage union-find, matching
//! SpectralExperiment::calculate_cluster_purity), but in Rust instead of
//! pure Python -- the Python version takes ~20-25 minutes on PXD021518's
//! ~1.2M clustered spectra; this takes seconds. Use this instead of
//! reimplementing/waiting on the slow path every time MS-RT purity is
//! needed for a one-off check.
//!
//! Usage: msrt_purity_from_csv <cluster.csv> [mz_tol] [rt_tol]
//!   mz_tol defaults to 0.01 Da, rt_tol defaults to 65.0s (proteomics;
//!   pass 30.0 for metabolomics/Astral-scale RT tolerance).
//!
//! Prints: weighted_purity=<f64> n_clusters=<usize> n_spectra=<usize>

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

struct Row {
    file_name: String,
    mz: f64,
    rt: f32,
    cluster: i64,
}

fn parse_file_name(identifier: &str) -> &str {
    // mzspec:USI000000:<file_name>:scan:<scan_number>
    identifier.split(':').nth(2).unwrap_or("")
}

fn largest_rt_cohesive_component(mz: &[f64], rt: &[f32], mz_tol: f64, rt_tol: f32) -> usize {
    let n = mz.len();
    if n <= 1 {
        return n;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mz[a].partial_cmp(&mz[b]).unwrap());

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    for a in 0..n {
        let i = order[a];
        let (mzi, rti) = (mz[i], rt[i]);
        for &j in order.iter().skip(a + 1) {
            if mz[j] - mzi > mz_tol {
                break;
            }
            if (rti - rt[j]).abs() <= rt_tol {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for idx in 0..n {
        let r = find(&mut parent, idx);
        *sizes.entry(r).or_insert(0) += 1;
    }
    sizes.values().copied().max().unwrap_or(0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: msrt_purity_from_csv <cluster.csv> [mz_tol=0.01] [rt_tol=65.0]");
        std::process::exit(1);
    }
    let path = &args[1];
    let mz_tol: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.01);
    let rt_tol: f32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(65.0);

    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let reader = BufReader::new(file);

    let mut by_cluster: HashMap<i64, Vec<Row>> = HashMap::new();
    let mut header_seen = false;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }
        let identifier = fields[0];
        let mz: f64 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let rt: f32 = match fields[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let cluster: i64 = match fields[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if cluster == -1 {
            continue; // Falcon noise
        }
        by_cluster.entry(cluster).or_default().push(Row {
            file_name: parse_file_name(identifier).to_string(),
            mz,
            rt,
            cluster,
        });
    }

    let mut total_product = 0.0f64;
    let mut total_weight = 0usize;
    let n_clusters = by_cluster.len();
    let mut n_spectra = 0usize;

    for (_cid, rows) in by_cluster {
        let cluster_size = rows.len();
        n_spectra += cluster_size;
        let mut by_file: HashMap<&str, (Vec<f64>, Vec<f32>)> = HashMap::new();
        for r in &rows {
            let e = by_file.entry(r.file_name.as_str()).or_default();
            e.0.push(r.mz);
            e.1.push(r.rt);
        }
        let mut sum_max = 0usize;
        for (_fname, (mzs, rts)) in by_file {
            sum_max += largest_rt_cohesive_component(&mzs, &rts, mz_tol, rt_tol);
        }
        let purity = sum_max as f64 / cluster_size as f64;
        total_product += cluster_size as f64 * purity;
        total_weight += cluster_size;
        let _ = rows[0].cluster; // silence unused warning path
    }

    let weighted_purity = if total_weight > 0 {
        total_product / total_weight as f64
    } else {
        f64::NAN
    };

    println!("weighted_purity={weighted_purity:.6} n_clusters={n_clusters} n_spectra={n_spectra}");
}
