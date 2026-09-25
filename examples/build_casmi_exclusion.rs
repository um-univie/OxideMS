//! Emit a DatasetManifest of every CASMI 2016 Cat 2/3 answer structure, so
//! `train_fingerprint_model --exclude-manifest` can hold them out of training
//! and `evaluate_casmi` stays an honest held-out benchmark.
//!
//!   cargo run --release --example build_casmi_exclusion -- \
//!     scratch/casmi/raw/solutions_casmi2016_cat2and3.csv scratch/casmi/casmi2016_exclude.manifest

use oxidems_lib::sirius::fingerprint::DatasetManifest;
use std::io::{BufRead, BufReader};

fn csv_fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    for c in line.chars() {
        match c {
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sol = &args[1];
    let out = &args[2];

    let reader = BufReader::new(std::fs::File::open(sol).expect("solutions csv"));
    let mut lines = reader.lines().map_while(Result::ok);
    let header = csv_fields(&lines.next().expect("header"));
    let smiles_col = header.iter().position(|h| h == "SMILES").expect("SMILES column");

    let mut manifest = DatasetManifest::new();
    let (mut seen, mut keyed) = (0usize, 0usize);
    for line in lines {
        let f = csv_fields(&line);
        if f.len() <= smiles_col {
            continue;
        }
        let smi = f[smiles_col].trim();
        if smi.is_empty() {
            continue;
        }
        seen += 1;
        let before = manifest_structs(&manifest);
        manifest.insert(smi, 0.0, &[]);
        if manifest_structs(&manifest) > before {
            keyed += 1;
        }
    }
    manifest.save(out).expect("write manifest");
    println!("CASMI solutions rows: {seen}, structure keys written: {keyed}");
    println!("wrote {out}");
}

fn manifest_structs(m: &DatasetManifest) -> usize {
    m.structure_keys.len()
}
