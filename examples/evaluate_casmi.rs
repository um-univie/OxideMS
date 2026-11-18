//! CASMI 2016 Category 2 benchmark, run under the contest's own rules.
//!
//! The contest supplied one candidate list per challenge and scored a submission by
//! whether the candidate it ranked first matched the answer's InChIKey first block.
//! Every team ranked the same candidates, so the comparison is controlled for database
//! content — which is what makes it worth running rather than inventing a candidate set
//! of our own.
//!
//! Candidates within a challenge span many molecular formulas (13 on median), so the
//! task includes deciding the formula: one fragmentation tree is built per distinct
//! candidate formula, and each candidate is scored against the fingerprint predicted
//! from its own formula's tree.
//!
//! Usage:
//!   evaluate_casmi --model M --solutions S.csv --candidates DIR --spectra DIR [--polarity positive] [--oracle-formula]

use oxidems_lib::sirius::fingerprint::{
    BinaryFingerprint, FingerprintPredictor, FingerprintSearcher, StructureCandidate,
    TreeFeatureExtractor,
};
use oxidems_lib::sirius::{
    Adduct, Formula, FragmentationTreeBuilder, SiriusClusteringWorkflow, TreeSolverConfig,
};
use oxidems_lib::spectrum::Peak;
use oxidems_lib::workflow::MassTolerance;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

/// One row of the published solutions.
struct Solution {
    challenge: String,
    name: String,
    inchikey_block: String,
    smiles: String,
    precursor_mz: f64,
    positive: bool,
}

/// One entry of a challenge's supplied candidate list.
struct Candidate {
    smiles: String,
    inchikey_block: String,
    formula: Formula,
    name: String,
}

/// Splits a CSV line on commas outside double quotes, and strips the quotes.
fn csv_fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn read_solutions(path: &str) -> Vec<Solution> {
    let reader = BufReader::new(File::open(path).expect("solutions csv"));
    let mut lines = reader.lines().map_while(Result::ok);
    let header = csv_fields(&lines.next().expect("header"));
    let col = |name: &str| header.iter().position(|h| h == name).expect(name);
    let (c_ch, c_name, c_key, c_smiles, c_mz, c_mode) = (
        col("ChallengeName"),
        col("NAME"),
        col("INCHIKEY"),
        col("SMILES"),
        col("PRECURSOR_MZ"),
        col("ION_MODE"),
    );
    let c_max = c_ch.max(c_name).max(c_key).max(c_smiles).max(c_mz).max(c_mode);

    lines
        .filter_map(|line| {
            let f = csv_fields(&line);
            if f.len() <= c_max {
                return None;
            }
            Some(Solution {
                challenge: f[c_ch].clone(),
                name: f[c_name].clone(),
                inchikey_block: f[c_key].split('-').next().unwrap_or("").to_string(),
                smiles: f[c_smiles].clone(),
                precursor_mz: f[c_mz].trim().parse().ok()?,
                positive: f[c_mode].trim().to_uppercase().starts_with('P'),
            })
        })
        .collect()
}

fn read_candidates(path: &str) -> Vec<Candidate> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut lines = BufReader::new(file).lines().map_while(Result::ok);
    let header = match lines.next() {
        Some(h) => csv_fields(&h),
        None => return Vec::new(),
    };
    let col = |name: &str| header.iter().position(|h| h == name);
    let (c_smiles, c_key, c_formula) = match (col("SMILES"), col("InChIKey"), col("MolecularFormula")) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Vec::new(),
    };
    let c_name = col("CompoundName");

    lines
        .filter_map(|line| {
            let f = csv_fields(&line);
            if f.len() <= c_formula.max(c_key).max(c_smiles) {
                return None;
            }
            Some(Candidate {
                smiles: f[c_smiles].clone(),
                inchikey_block: f[c_key].split('-').next().unwrap_or("").to_string(),
                formula: Formula::from_str(f[c_formula].trim()).ok()?,
                name: c_name.and_then(|i| f.get(i).cloned()).unwrap_or_default(),
            })
        })
        .collect()
}

fn read_peaks(path: &str) -> Vec<Peak> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| {
            let line = line.trim();
            if !line.chars().next()?.is_ascii_digit() {
                return None;
            }
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() < 2 {
                return None;
            }
            let (mz, intensity) = (p[0].parse().ok()?, p[1].parse().ok()?);
            (mz > 0.0 && intensity > 0.0).then_some(Peak { mz, intensity })
        })
        .collect()
}

/// How the per-formula fingerprint scores and the formula's tree score are merged into
/// one ranking.
#[derive(Clone, Copy, PartialEq)]
enum Combine {
    /// Add the tree score to every fingerprint score of that formula. Two scales with no
    /// stated relationship — what this tool did originally.
    Sum,
    /// Ignore the tree entirely and rank on the fingerprint alone.
    Fingerprint,
    /// Keep only the `n` formulas whose tree explains the spectrum best, then rank the
    /// structures under them by fingerprint alone. This is the shape SIRIUS uses: the
    /// tree decides the formula, the fingerprint decides the structure.
    TopFormulas(usize),
    /// Standardize fingerprint scores within each formula before adding a tree term, so
    /// the two contributions are at least on comparable scales.
    ZScore,
}

/// Merges per-formula scores into a single ranked list.
fn combine(per_formula: &[(f64, Vec<(f64, usize)>)], mode: Combine) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = Vec::new();
    match mode {
        Combine::Sum => {
            for (tree, members) in per_formula {
                out.extend(members.iter().map(|&(s, i)| (s + tree, i)));
            }
        }
        Combine::Fingerprint => {
            for (_, members) in per_formula {
                out.extend(members.iter().copied());
            }
        }
        Combine::TopFormulas(n) => {
            let mut ranked: Vec<&(f64, Vec<(f64, usize)>)> = per_formula.iter().collect();
            ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, members) in ranked.into_iter().take(n) {
                out.extend(members.iter().copied());
            }
        }
        Combine::ZScore => {
            for (tree, members) in per_formula {
                let n = members.len() as f64;
                let mean = members.iter().map(|&(s, _)| s).sum::<f64>() / n;
                let sd = (members.iter().map(|&(s, _)| (s - mean).powi(2)).sum::<f64>() / n)
                    .sqrt()
                    .max(1e-9);
                // The tree score enters as a per-formula offset in standardized units,
                // so a better-explaining formula lifts all its structures by a
                // comparable amount rather than an arbitrary one.
                out.extend(members.iter().map(|&(s, i)| ((s - mean) / sd + tree / 10.0, i)));
            }
        }
    }
    out
}

/// What one challenge produced.
struct Outcome {
    challenge: String,
    name: String,
    candidates: usize,
    formulas: usize,
    /// Rank of the first candidate matching the answer, 1-based.
    rank: Option<usize>,
    /// Candidates that actually reached the scorer.
    scored: usize,
    /// Whether the answer was among the candidates at all.
    answer_present: bool,
    /// Whether the answer reached the scorer. An answer dropped for an unparseable
    /// structure, or because its formula produced no tree, cannot be ranked and would
    /// otherwise be indistinguishable from a ranking failure.
    answer_scored: bool,
    /// Peaks surviving preprocessing.
    peaks: usize,
    /// Formulas whose tree was too small to yield features.
    formulas_without_tree: usize,
    /// Predicted-vs-true fingerprint Tanimoto for the answer, when it was scored.
    answer_tanimoto: Option<f64>,
    /// Structural similarity between the true answer and whatever was ranked first.
    /// High means the winner is a close analogue and the fingerprint simply lacks the
    /// resolution to separate them; low means the ranking is picking unrelated things,
    /// which would point at a defect rather than at granularity.
    winner_similarity: Option<f64>,
    /// Whether the top-ranked candidate at least had the right molecular formula.
    winner_same_formula: Option<bool>,
    /// Score of the winner minus score of the answer.
    score_gap: Option<f64>,
    /// Name of the winning candidate, for eyeballing.
    winner_name: Option<String>,
}

mod support;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let model_path = get("--model", "models/sirius_fingerprints.oxfp");
    let solutions_path = get("--solutions", "");
    let candidate_dir = get("--candidates", "");
    let spectra_dir_pos = get("--spectra-positive", "");
    let spectra_dir_neg = get("--spectra-negative", "");
    let polarity = get("--polarity", "positive");
    let oracle_formula = args.iter().any(|a| a == "--oracle-formula");
    // Two choices this harness makes that are not the model's: folding the tree score
    // into the ranking, and masking predicted bits against the candidate formula.
    // Both are testable.
    let combine_mode = match get("--combine", "sum").as_str() {
        "sum" => Combine::Sum,
        "fingerprint" => Combine::Fingerprint,
        "zscore" => Combine::ZScore,
        other if other.starts_with("top") => Combine::TopFormulas(other[3..].parse().unwrap_or(1)),
        other => panic!("--combine must be sum, fingerprint, zscore, or topN (got {other:?})"),
    };
    let no_formula_mask = args.iter().any(|a| a == "--no-formula-mask");
    // Rank candidates by Tanimoto to the TRUE structure's fingerprint instead of
    // the model's prediction: the ceiling the 2214-bit representation itself
    // allows, regardless of how good the predictor is.
    let oracle_fingerprint = args.iter().any(|a| a == "--oracle-fingerprint");
    // Two hooks for evaluating an externally trained predictor (the DGAT prototype)
    // on exactly this harness, so the retrieval half of the comparison is held
    // fixed and only the fingerprint changes.
    //   --export-graphs FILE  write one JSONL line per (challenge, formula) tree
    //   --predicted-fp FILE   read those trees' predicted bit probabilities back
    let export_graphs = args
        .iter()
        .position(|a| a == "--export-graphs")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let predicted_fp_path = args
        .iter()
        .position(|a| a == "--predicted-fp")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let external_fp: std::collections::HashMap<(String, String), Vec<f32>> =
        match &predicted_fp_path {
            Some(path) => {
                let text = std::fs::read_to_string(path).expect("read --predicted-fp");
                let mut m = std::collections::HashMap::new();
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let v: serde_json::Value = serde_json::from_str(line).expect("bad JSONL");
                    let ch = v["challenge"].as_str().expect("challenge").to_string();
                    let fo = v["formula"].as_str().expect("formula").to_string();
                    let probs = v["probs"]
                        .as_array()
                        .expect("probs")
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    m.insert((ch, fo), probs);
                }
                println!("Loaded {} external fingerprint predictions.", m.len());
                m
            }
            None => Default::default(),
        };
    // Only score challenges whose answer structure is NOT in this manifest — for
    // measuring a model on compounds it genuinely never trained on.
    let require_unseen: Option<oxidems_lib::sirius::fingerprint::DatasetManifest> = args
        .iter()
        .position(|a| a == "--require-unseen")
        .and_then(|i| args.get(i + 1))
        .map(|p| {
            oxidems_lib::sirius::fingerprint::DatasetManifest::load(p)
                .expect("--require-unseen manifest")
        });

    // Optional: with `--predicted-fp` the built-in model is never consulted, and
    // with `--export-graphs` nothing is predicted at all.
    let predictor = match FingerprintPredictor::load_from_file(&model_path) {
        Ok(p) => Some(p),
        Err(e) => {
            if external_fp.is_empty() && export_graphs.is_none() {
                support::fail(format!("cannot load model {model_path}: {e}\n  {}", support::MODEL));
            }
            println!("  (no built-in model at {model_path}; using external predictions)");
            None
        }
    };
    let graph_sink = export_graphs.as_ref().map(|p| {
        std::sync::Mutex::new(std::io::BufWriter::new(
            std::fs::File::create(p).expect("create --export-graphs"),
        ))
    });
    // With no built-in model the searcher uses the default dictionary — which is
    // exactly the layout `export_training_data` built the ground-truth fingerprints
    // under, so an externally trained model's bits mean the same thing here.
    let searcher = match &predictor {
        Some(p) => FingerprintSearcher::for_predictor(p),
        None => FingerprintSearcher::new(oxidems_lib::sirius::fingerprint::FingerprintDictionary::new()),
    };
    let layout = searcher.dictionary().clone();
    // Peak budget for the trees this harness builds; must match what the fingerprint
    // model was trained on, so both take the library default.
    let max_peaks: usize = args
        .iter()
        .position(|a| a == "--max-peaks")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| oxidems_lib::sirius::SiriusWorkflowConfig::default().max_peaks_per_tree);
    let match_export_peaks = args.iter().any(|a| a == "--match-export-peaks");
    let workflow = {
        let mut cfg = oxidems_lib::sirius::SiriusWorkflowConfig::default();
        cfg.max_peaks_per_tree = max_peaks;
        SiriusClusteringWorkflow::new(cfg)
    };
    let extractor = TreeFeatureExtractor::new();

    let mut solutions: Vec<Solution> = read_solutions(&solutions_path)
        .into_iter()
        .filter(|s| match polarity.as_str() {
            "positive" => s.positive,
            "negative" => !s.positive,
            _ => true,
        })
        .collect();
    if let Some(m) = &require_unseen {
        let before = solutions.len();
        solutions.retain(|s| !m.contains_structure(&s.smiles));
        println!(
            "  --require-unseen: kept {} of {before} challenges whose answer is absent from the manifest",
            solutions.len()
        );
    }

    println!("=========================================================================");
    println!("  CASMI 2016 Category 2 — {} challenges, {polarity} mode", solutions.len());
    println!("  Model: {model_path}");
    println!("  Score combination: {}", get("--combine", "sum"));
    println!("  Candidates: supplied by the contest{}", if oracle_formula { ", restricted to the true formula" } else { "" });
    println!("=========================================================================\n");

    let outcomes: Vec<Outcome> = solutions
        .par_iter()
        .map(|sol| {
            let dir = if sol.positive { &spectra_dir_pos } else { &spectra_dir_neg };
            let peaks = read_peaks(&format!("{dir}/{}.mgf", sol.challenge));
            let mut candidates = read_candidates(&format!("{candidate_dir}/{}.csv", sol.challenge));
            let mut total_candidates = candidates.len();

            if peaks.len() < 3 || candidates.is_empty() {
                return Outcome {
                    challenge: sol.challenge.clone(),
                    name: sol.name.clone(),
                    candidates: total_candidates,
                    formulas: 0,
                    rank: None,
                    scored: 0,
                    answer_present: candidates.iter().any(|c| c.inchikey_block == sol.inchikey_block),
                    answer_scored: false,
                    peaks: peaks.len(),
                    formulas_without_tree: 0,
                    answer_tanimoto: None,
                    winner_similarity: None,
                    winner_same_formula: None,
                    score_gap: None,
                    winner_name: None,
                };
            }

            // The contest does not give the formula; the candidate list spans many.
            // Restricting to the true one isolates the ranking problem from the formula
            // problem, which is the only way to tell which of the two is costing us.
            if oracle_formula {
                if let Some(truth) = candidates
                    .iter()
                    .find(|c| c.inchikey_block == sol.inchikey_block)
                    .map(|c| c.formula)
                {
                    candidates.retain(|c| c.formula == truth);
                    total_candidates = candidates.len();
                }
            }

            let charge = if sol.positive { 1i8 } else { -1 };
            // `--match-export-peaks`: build the trees the way `export_training_data`
            // does — sort by intensity and truncate, with no adaptive noise floor.
            // The workflow's floor (3x median, or 0.2 % of base) is not applied when
            // the training trees are built, so leaving it on here makes the model
            // predict from trees systematically smaller than any it was trained on,
            // and it bites harder the wider the budget: at 30 peaks the inference
            // trees are 0.77x the training mean, at 60 peaks only 0.67x. That turns
            // a peak-budget A/B into a peak-budget-plus-mismatch A/B.
            let cleaned = if match_export_peaks {
                let mut v = peaks.clone();
                v.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
                v.truncate(max_peaks);
                if v.len() < 2 { Vec::new() } else { v }
            } else {
                workflow.preprocess_peaks_for_tree(&peaks, sol.precursor_mz, charge).unwrap_or_default()
            };
            let cleaned = match (!cleaned.is_empty()).then_some(cleaned) {
                Some(p) => p,
                None => {
                    return Outcome {
                        challenge: sol.challenge.clone(),
                        name: sol.name.clone(),
                        candidates: total_candidates,
                        formulas: 0,
                        rank: None,
                        scored: 0,
                        answer_present: candidates.iter().any(|c| c.inchikey_block == sol.inchikey_block),
                        answer_scored: false,
                        peaks: peaks.len(),
                        formulas_without_tree: 0,
                    answer_tanimoto: None,
                    winner_similarity: None,
                    winner_same_formula: None,
                    score_gap: None,
                    winner_name: None,
                    }
                }
            };

            let adduct = if sol.positive { Adduct::protonated() } else { Adduct::deprotonated() };
            let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
                tolerance: MassTolerance::Ppm(20.0),
                adduct: adduct.clone(),
                ..Default::default()
            });

            // One tree, and therefore one predicted fingerprint, per candidate formula.
            let mut by_formula: HashMap<Formula, Vec<usize>> = HashMap::new();
            for (i, c) in candidates.iter().enumerate() {
                by_formula.entry(c.formula).or_default().push(i);
            }
            let formulas = by_formula.len();

            // Kept per formula so the two kinds of evidence can be merged afterwards.
            // Adding a ModPlatt score to a tree score directly, as this did, sums two
            // quantities on unrelated scales.
            let mut per_formula: Vec<(f64, Vec<(f64, usize)>)> = Vec::new();
            let mut formulas_without_tree = 0usize;
            let mut answer_tanimoto: Option<f64> = None;
            let truth_fp = layout
                .ground_truth_fingerprint(&sol.smiles)
                .map(BinaryFingerprint::from_bits);
            for (formula, members) in &by_formula {
                let tree = builder.build_tree(formula, sol.precursor_mz, &cleaned);
                if tree.nodes.len() < 2 {
                    formulas_without_tree += 1;
                    continue;
                }
                if let Some(sink) = &graph_sink {
                    let (gn, ge, gf) =
                        oxidems_lib::sirius::fingerprint::tree_graph(&tree, 1);
                    let line = serde_json::json!({
                        "challenge": sol.challenge,
                        "formula": formula.to_string(),
                        // The tree score is the formula evidence. Exporting it lets an
                        // external ranker do what `Combine::TopFormulas` does here:
                        // let the tree choose the formula, the fingerprint the structure.
                        "tree_score": tree.total_score,
                        "graph_nodes": gn,
                        "graph_edges": ge,
                        "graph_edge_feats": gf,
                    });
                    let mut fh = sink.lock().unwrap();
                    use std::io::Write as _;
                    writeln!(fh, "{line}").expect("write --export-graphs");
                }
                let features = extractor.extract(&tree);
                // An externally supplied prediction replaces the built-in model, so
                // both are scored by identical retrieval code.
                let mut predicted = match external_fp.get(&(sol.challenge.clone(), formula.to_string())) {
                    Some(p) => oxidems_lib::sirius::fingerprint::PredictedFingerprint {
                        probabilities: p.clone(),
                    },
                    None => predictor
                        .as_ref()
                        .expect("no model and no external prediction for this tree")
                        .predict(&features.features),
                };
                let _ = no_formula_mask;
                let table = searcher.prepare(&predicted);

                let mut here: Vec<(f64, usize)> = Vec::with_capacity(members.len());
                for &i in members {
                    let bits = match layout.ground_truth_fingerprint(&candidates[i].smiles) {
                        Some(b) => b,
                        None => continue,
                    };
                    let fp = BinaryFingerprint::from_bits(bits);
                    if candidates[i].inchikey_block == sol.inchikey_block {
                        answer_tanimoto =
                            Some(FingerprintSearcher::tanimoto_similarity(&predicted, &fp, 0.5));
                    }
                    let s = if oracle_fingerprint {
                        match &truth_fp {
                            Some(t) => t.tanimoto(&fp),
                            None => continue,
                        }
                    } else {
                        table.score(&fp)
                    };
                    here.push((s, i));
                }
                if !here.is_empty() {
                    per_formula.push((tree.total_score, here));
                }
            }

            let mut scored = combine(&per_formula, combine_mode);

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let answer_pos = scored
                .iter()
                .position(|(_, i)| candidates[*i].inchikey_block == sol.inchikey_block);
            let rank = answer_pos.map(|p| p + 1);

            // What beat the answer, and how close is it?
            let (mut winner_similarity, mut winner_same_formula, mut score_gap, mut winner_name) =
                (None, None, None, None);
            if let (Some(pos), Some(&(top_score, top_i))) = (answer_pos, scored.first()) {
                if pos > 0 {
                    let answer_i = scored[pos].1;
                    let truth = layout.ground_truth_fingerprint(&candidates[answer_i].smiles);
                    let winner = layout.ground_truth_fingerprint(&candidates[top_i].smiles);
                    if let (Some(t), Some(w)) = (truth, winner) {
                        winner_similarity = Some(
                            BinaryFingerprint::from_bits(t).tanimoto(&BinaryFingerprint::from_bits(w)),
                        );
                    }
                    winner_same_formula =
                        Some(candidates[top_i].formula == candidates[answer_i].formula);
                    score_gap = Some(top_score - scored[pos].0);
                    winner_name = Some(candidates[top_i].name.clone());
                }
            }

            let _ = StructureCandidate {
                id: String::new(),
                smiles: String::new(),
                name: String::new(),
                fingerprint: BinaryFingerprint::zeros(),
            };

            Outcome {
                challenge: sol.challenge.clone(),
                name: sol.name.clone(),
                candidates: total_candidates,
                formulas,
                rank,
                scored: scored.len(),
                answer_present: candidates.iter().any(|c| c.inchikey_block == sol.inchikey_block),
                answer_scored: scored
                    .iter()
                    .any(|(_, i)| candidates[*i].inchikey_block == sol.inchikey_block),
                peaks: cleaned.len(),
                formulas_without_tree,
                answer_tanimoto,
                winner_similarity,
                winner_same_formula,
                score_gap,
                winner_name,
            }
        })
        .collect();

    let scored: Vec<&Outcome> = outcomes.iter().filter(|o| o.rank.is_some()).collect();
    let n = solutions.len() as f64;
    let hit = |k: usize| outcomes.iter().filter(|o| o.rank.is_some_and(|r| r <= k)).count();

    let mut cand_counts: Vec<usize> = outcomes.iter().map(|o| o.candidates).collect();
    cand_counts.sort_unstable();
    let mut ranks: Vec<usize> = scored.iter().filter_map(|o| o.rank).collect();
    ranks.sort_unstable();

    println!("  Challenges attempted:                  {}", solutions.len());
    println!("  Answer present in the candidate list:  {}", scored.len());
    println!(
        "  Candidates per challenge:              median {}, max {}",
        cand_counts.get(cand_counts.len() / 2).copied().unwrap_or(0),
        cand_counts.last().copied().unwrap_or(0)
    );
    println!();
    println!("  Top-1:  {:>3} ({:.1}%)", hit(1), hit(1) as f64 / n * 100.0);
    println!("  Top-3:  {:>3} ({:.1}%)", hit(3), hit(3) as f64 / n * 100.0);
    println!("  Top-5:  {:>3} ({:.1}%)", hit(5), hit(5) as f64 / n * 100.0);
    println!("  Top-10: {:>3} ({:.1}%)", hit(10), hit(10) as f64 / n * 100.0);
    if !ranks.is_empty() {
        println!(
            "  Median rank of the correct answer:     {}",
            ranks[ranks.len() / 2]
        );
    }

    // Where the misses come from. A challenge can fail because the answer was never
    // offered, because it was offered but never scored, or because it was scored and
    // ranked badly — three different problems that a single accuracy figure merges.
    let present = outcomes.iter().filter(|o| o.answer_present).count();
    let answer_scored = outcomes.iter().filter(|o| o.answer_scored).count();
    let dropped = present - answer_scored;
    let scored_total: usize = outcomes.iter().map(|o| o.scored).sum();
    let candidate_total: usize = outcomes.iter().map(|o| o.candidates).sum();
    println!("\n  Failure breakdown");
    println!("    answer never in the candidate list:  {}", solutions.len() - present);
    println!("    answer present but never scored:     {dropped}");
    println!("    answer scored but not ranked first:  {}", answer_scored - hit(1));
    println!(
        "    candidates reaching the scorer:      {scored_total} of {candidate_total} ({:.1}%)",
        scored_total as f64 / candidate_total.max(1) as f64 * 100.0
    );
    let no_tree: usize = outcomes.iter().map(|o| o.formulas_without_tree).sum();
    let all_formulas: usize = outcomes.iter().map(|o| o.formulas).sum();
    println!("    candidate formulas yielding no tree: {no_tree} of {all_formulas}");
    let thin = outcomes.iter().filter(|o| o.peaks < 5).count();
    println!("    challenges with under 5 usable peaks: {thin}");

    let tans: Vec<f64> = outcomes.iter().filter_map(|o| o.answer_tanimoto).collect();
    if !tans.is_empty() {
        println!(
            "    predicted-vs-true Tanimoto on the answer: mean {:.4}",
            tans.iter().sum::<f64>() / tans.len() as f64
        );
    }

    // What is beating the answer?
    let losses: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| o.rank.is_some_and(|r| r > 1) && o.winner_similarity.is_some())
        .collect();
    if !losses.is_empty() {
        let sims: Vec<f64> = losses.iter().filter_map(|o| o.winner_similarity).collect();
        let mean_sim = sims.iter().sum::<f64>() / sims.len() as f64;
        let close = sims.iter().filter(|&&s| s >= 0.6).count();
        let unrelated = sims.iter().filter(|&&s| s < 0.3).count();
        let same_formula = losses.iter().filter(|o| o.winner_same_formula == Some(true)).count();
        println!("\n  What outranked the answer, over {} losses", losses.len());
        println!("    structural similarity of winner to truth: mean {mean_sim:.3}");
        println!(
            "    winner is a close analogue (Tanimoto >= 0.6):  {close} ({:.0}%)",
            close as f64 / sims.len() as f64 * 100.0
        );
        println!(
            "    winner is unrelated (Tanimoto < 0.3):          {unrelated} ({:.0}%)",
            unrelated as f64 / sims.len() as f64 * 100.0
        );
        println!(
            "    winner has the right molecular formula:        {same_formula} ({:.0}%)",
            same_formula as f64 / losses.len() as f64 * 100.0
        );
        let near: Vec<f64> = losses.iter().filter_map(|o| o.score_gap).collect();
        if !near.is_empty() {
            let mut g = near.clone();
            g.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("    median score gap to the winner:               {:.2}", g[g.len() / 2]);
        }

        println!("\n  Examples of near misses (answer at rank 2-5):");
        for o in losses.iter().filter(|o| o.rank.is_some_and(|r| (2..=5).contains(&r))).take(6) {
            println!(
                "    {:<32} rank {} | winner {:<28} similarity {:.2}",
                o.name.chars().take(30).collect::<String>(),
                o.rank.unwrap_or(0),
                o.winner_name.as_deref().unwrap_or("").chars().take(26).collect::<String>(),
                o.winner_similarity.unwrap_or(0.0)
            );
        }
    }

    println!("\n  Worst-ranked answers that were scored:");
    let mut ranked: Vec<&Outcome> = outcomes.iter().filter(|o| o.rank.is_some()).collect();
    ranked.sort_by_key(|o| std::cmp::Reverse(o.rank.unwrap_or(0)));
    for o in ranked.iter().take(5) {
        println!(
            "    {:<14} {:<34} rank {:>5} of {:>5} scored ({} formulas)",
            o.challenge,
            o.name.chars().take(32).collect::<String>(),
            o.rank.unwrap_or(0),
            o.scored,
            o.formulas
        );
    }
}
