//! Report what a structure database actually contains, so retrieval numbers can be
//! read against the size of the haystack they were measured in.

use oxidems_lib::sirius::fingerprint::StructureDatabase;
use std::io::Write;

mod support;

fn main() {
    // `--smiles OUT` dumps "inchikey<TAB>smiles" for every structure, so a vocabulary
    // can be counted over the search space rather than over the training spectra. The
    // two differ: a vocabulary counted on MoNA covers only 88 % of ECFP4 occurrences in
    // the candidate structures, against 95 % in its own corpus.
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--smiles").and_then(|i| args.get(i + 1)).cloned();
    if !args.iter().skip(1).any(|a| !a.starts_with("--") && Some(a) != out.as_ref()) {
        support::fail(format!("usage: db_stats <database.oxdb>... [--smiles OUT.tsv]\n  {}", support::STRUCTURE_DB));
    }
    let mut sink = out.as_ref().map(|p| {
        std::io::BufWriter::new(std::fs::File::create(p).expect("create --smiles output"))
    });

    for path in args.iter().skip(1) {
        if path.starts_with("--") || Some(path) == out.as_ref() {
            continue;
        }
        match StructureDatabase::load_from_file(path) {
            Ok(db) => {
                let mut n = 0usize;
                if let Some(w) = sink.as_mut() {
                    for entries in db.entries_by_formula.values() {
                        for e in entries {
                            let key = e.inchi_key.as_deref().unwrap_or("");
                            writeln!(w, "{}\t{}", &key[..key.len().min(14)], e.smiles)
                                .expect("write smiles");
                            n += 1;
                        }
                    }
                }
                println!("{path}: {} structures{}", db.total_structures,
                         if n > 0 { format!(", {n} written") } else { String::new() });
            }
            Err(e) => println!("{path}: failed to load: {e}"),
        }
    }
}
