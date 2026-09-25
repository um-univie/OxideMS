//! Score SIRIUS's fragmentation tree under *our* objective.
//!
//! Rounds 24-25 established that our exact solver and SIRIUS agree on 98 % of
//! peak -> formula assignments but disagree on roughly 45 % of parent -> child
//! edges. Fixing the loss gate moved that a little. This answers what is left, by
//! taking SIRIUS's tree for the same pinned formula and evaluating it with our own
//! weights:
//!
//!   * **not representable** — some vertex or edge SIRIUS used does not exist in
//!     our graph, so the graphs still differ.
//!   * **representable, scores lower** — our objective genuinely prefers our tree.
//!     The difference is the scoring function, not the search.
//!   * **representable, scores higher** — our "exact" solver is not exact, which
//!     would be a bug.
//!
//! The mirror of our weight composition is validated before it is trusted: our own
//! tree is re-scored from its dumped nodes and edges and must reproduce the
//! `total_score` the solver reported.
//!
//! Usage:
//!   score_sirius_tree <compare_out_dir>   (needs ours/*.json and sirius_trees.json)

use oxidems_lib::sirius::sirius_scoring::{edge_score, mass_deviation_score, root_loss_score, vertex_score};
use oxidems_lib::sirius::rules::LossGate;
use oxidems_lib::sirius::{Adduct, ChemicalRules, Formula};
use std::collections::HashMap;
use std::fs;
use std::str::FromStr;

/// The solver's `sirius_sd`: `mass_sigma_ppm` in Da, floored at 1 mDa.
fn sd(mz: f64, sigma_ppm: f64) -> f64 {
    (mz * sigma_ppm * 1e-6).max(1e-3)
}

/// A fragment vertex's weight, less the per-spectrum tree-size bonus.
fn fragment_weight_without_bonus(
    f: &Formula,
    peak_mz: f64,
    precursor: &Formula,
    adduct: &Adduct,
    sigma_ppm: f64,
) -> f64 {
    let observed_neutral = adduct.ion_mz_to_neutral_mass(peak_mz);
    let md = mass_deviation_score(f.exact_mass() - observed_neutral, sd(peak_mz, sigma_ppm));
    let root_loss = precursor
        .checked_sub(f)
        .map_or(0.0, |l| root_loss_score(&l));
    vertex_score(f, f.exact_mass(), md, false) + root_loss
}

fn get<'a>(v: &'a serde_json::Value, k: &str) -> &'a serde_json::Value {
    &v[k]
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: score_sirius_tree <compare_out_dir>");
    let sigma_ppm: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);

    let sirius: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(format!("{dir}/sirius_trees.json")).unwrap())
            .unwrap();

    let mut validated = 0usize;
    let mut validation_failed = 0usize;
    let mut not_representable = 0usize;
    let mut ours_higher = 0usize;
    let mut sirius_higher = 0usize;
    let mut tied = 0usize;
    let mut missing_vertex = 0usize;
    let mut missing_edge = 0usize;
    let mut gaps: Vec<f64> = Vec::new();
    let (mut d_vertex, mut d_rootloss, mut d_edge, mut d_bonus) = (0.0, 0.0, 0.0, 0.0);
    let mut differing = 0usize;
    let mut worst: Vec<(f64, String, String)> = Vec::new();
    // What the annotator and the graph builder actually admit.
    let frag_rules = ChemicalRules::loose();
    let loss_rules = ChemicalRules { loss_gate: LossGate::Repaired, ..ChemicalRules::loose() };
    let mut rejected_fragment = 0usize;
    let mut rejected_loss = 0usize;

    let mut entries: Vec<_> = fs::read_dir(format!("{dir}/ours"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort();

    for path in entries {
        let ours: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let idx = get(&ours, "idx").as_u64().unwrap();
        let Some(sir) = sirius.get(idx.to_string()) else { continue };

        let precursor = Formula::from_str(get(&ours, "true_formula").as_str().unwrap()).unwrap();
        let precursor_mz = get(&ours, "precursor_mz").as_f64().unwrap();
        let adduct = if get(&ours, "adduct").as_str().unwrap() == "[M+H]+" {
            Adduct::protonated()
        } else {
            Adduct::deprotonated()
        };
        let name = get(&ours, "name").as_str().unwrap().to_string();
        let ppm = get(&ours, "ppm").as_f64().unwrap_or(10.0);

        // --- recover the per-spectrum tree-size bonus from one of our own nodes,
        // then check every other node agrees with it.
        let nodes = get(&ours, "nodes").as_array().unwrap();
        let edges = get(&ours, "edges").as_array().unwrap();
        let mut bonus: Option<f64> = None;
        let mut bonus_consistent = true;
        for n in nodes.iter().skip(1) {
            let f = Formula::from_str(get(n, "formula").as_str().unwrap()).unwrap();
            let mz = get(n, "mz").as_f64().unwrap();
            let implied = get(n, "score").as_f64().unwrap()
                - fragment_weight_without_bonus(&f, mz, &precursor, &adduct, sigma_ppm);
            match bonus {
                None => bonus = Some(implied),
                Some(b) if (b - implied).abs() > 1e-6 => bonus_consistent = false,
                _ => {}
            }
        }
        let Some(bonus) = bonus else { continue };

        // --- validation: re-score our own tree and reproduce total_score.
        let root_md = mass_deviation_score(
            adduct.neutral_mass_to_ion_mz(precursor.exact_mass()) - precursor_mz,
            sd(precursor_mz, sigma_ppm),
        );
        let root_w = vertex_score(&precursor, precursor.exact_mass(), root_md, true);
        let mut ours_recomputed = root_w;
        for n in nodes.iter().skip(1) {
            let f = Formula::from_str(get(n, "formula").as_str().unwrap()).unwrap();
            let mz = get(n, "mz").as_f64().unwrap();
            ours_recomputed +=
                fragment_weight_without_bonus(&f, mz, &precursor, &adduct, sigma_ppm) + bonus;
        }
        for e in edges {
            let loss = Formula::from_str(get(e, "loss").as_str().unwrap()).unwrap();
            ours_recomputed += edge_score(&loss, loss.exact_mass());
        }
        let reported = get(&ours, "total_score").as_f64().unwrap();
        if !bonus_consistent || (ours_recomputed - reported).abs() > 1e-4 {
            validation_failed += 1;
            if validation_failed <= 2 {
                eprintln!(
                    "DEBUG {name}: bonus_consistent={bonus_consistent} bonus={bonus:.4} recomputed={ours_recomputed:.4} reported={reported:.4} root_w={root_w:.4} root_score_dumped={:.4} nodes={} edges={}",
                    get(&nodes[0], "score").as_f64().unwrap(), nodes.len(), edges.len()
                );
                for n in nodes.iter().skip(1).take(4) {
                    let f = Formula::from_str(get(n, "formula").as_str().unwrap()).unwrap();
                    let mz = get(n, "mz").as_f64().unwrap();
                    eprintln!("   node {f} mz={mz:.5} dumped={:.4} mirror_no_bonus={:.4} implied_bonus={:.4}",
                        get(n, "score").as_f64().unwrap(),
                        fragment_weight_without_bonus(&f, mz, &precursor, &adduct, sigma_ppm),
                        get(n, "score").as_f64().unwrap() - fragment_weight_without_bonus(&f, mz, &precursor, &adduct, sigma_ppm));
                }
            }
            continue;
        }
        validated += 1;

        // --- score SIRIUS's tree with the same weights.
        let peaks: Vec<f64> = get(&ours, "peaks")
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p[0].as_f64().unwrap())
            .collect();
        let frags = get(sir, "fragments").as_array().unwrap();
        let losses = get(sir, "losses").as_array().unwrap();

        // Decompose both trees the same way, so a non-zero difference names the
        // scorer responsible rather than just the total.
        let mut ours_vertex = 0.0;   // vertex_score, the mass/fragment-prior terms
        let mut ours_rootloss = 0.0; // root_loss_score, cumulative loss from the precursor
        let mut ours_edge = 0.0;     // edge_score, the loss scorers
        let mut ours_bonus = 0.0;    // tree-size bonus * node count
        for n in nodes.iter().skip(1) {
            let f = Formula::from_str(get(n, "formula").as_str().unwrap()).unwrap();
            let mz = get(n, "mz").as_f64().unwrap();
            let obs = adduct.ion_mz_to_neutral_mass(mz);
            let md = mass_deviation_score(f.exact_mass() - obs, sd(mz, sigma_ppm));
            ours_vertex += vertex_score(&f, f.exact_mass(), md, false);
            ours_rootloss += precursor.checked_sub(&f).map_or(0.0, |l| root_loss_score(&l));
            ours_bonus += bonus;
        }
        for e in edges {
            let loss = Formula::from_str(get(e, "loss").as_str().unwrap()).unwrap();
            ours_edge += edge_score(&loss, loss.exact_mass());
        }
        let mut sir_vertex = 0.0;
        let mut sir_rootloss = 0.0;
        let mut sir_edge = 0.0;
        let mut sir_bonus = 0.0;

        let mut sir_score = root_w;
        let mut representable = true;
        let mut by_formula: HashMap<String, f64> = HashMap::new();
        for fr in frags {
            let fs_ = get(fr, "formula").as_str().unwrap().to_string();
            let f = match Formula::from_str(&fs_) {
                Ok(f) => f,
                Err(_) => {
                    representable = false;
                    break;
                }
            };
            by_formula.insert(fs_, get(fr, "mz").as_f64().unwrap());
            if f == precursor {
                continue; // the root, already counted
            }
            if !f.is_subformula_of(&precursor) {
                representable = false;
                missing_vertex += 1;
                break;
            }
            // The peak SIRIUS assigned, matched in our own peak list.
            let mz = get(fr, "mz").as_f64().unwrap();
            // Representable means "our graph would contain this vertex": the peak
            // must be one of ours, and the formula must explain it inside the same
            // tolerance our annotator used. Matching more loosely than that credits
            // SIRIUS for assignments we never had the option of making.
            let matched = peaks
                .iter()
                .copied()
                .min_by(|a, b| (a - mz).abs().total_cmp(&(b - mz).abs()))
                .filter(|p| (p - mz).abs() / mz * 1e6 <= ppm);
            let Some(peak_mz) = matched else {
                representable = false;
                missing_vertex += 1;
                break;
            };
            let err_ppm = (f.exact_mass() - adduct.ion_mz_to_neutral_mass(peak_mz)).abs()
                / adduct.ion_mz_to_neutral_mass(peak_mz)
                * 1e6;
            if err_ppm > ppm {
                representable = false;
                missing_vertex += 1;
                break;
            }
            // Our annotator only ever proposes formulas its chemical rules accept.
            if !frag_rules.is_valid(&f) {
                representable = false;
                rejected_fragment += 1;
                break;
            }
            let obs = adduct.ion_mz_to_neutral_mass(peak_mz);
            let md = mass_deviation_score(f.exact_mass() - obs, sd(peak_mz, sigma_ppm));
            sir_vertex += vertex_score(&f, f.exact_mass(), md, false);
            sir_rootloss += precursor.checked_sub(&f).map_or(0.0, |l| root_loss_score(&l));
            sir_bonus += bonus;
            sir_score +=
                fragment_weight_without_bonus(&f, peak_mz, &precursor, &adduct, sigma_ppm) + bonus;
        }
        if representable {
            for l in losses {
                let loss = match Formula::from_str(get(l, "loss").as_str().unwrap()) {
                    Ok(l) => l,
                    Err(_) => {
                        representable = false;
                        missing_edge += 1;
                        break;
                    }
                };
                if !loss_rules.is_valid_neutral_loss(&loss) {
                    representable = false;
                    rejected_loss += 1;
                    break;
                }
                sir_edge += edge_score(&loss, loss.exact_mass());
                sir_score += edge_score(&loss, loss.exact_mass());
            }
        }

        if !representable {
            not_representable += 1;
            continue;
        }
        let gap = reported - sir_score;
        gaps.push(gap);
        if gap.abs() > 1e-6 {
            differing += 1;
            d_vertex += ours_vertex - sir_vertex;
            d_rootloss += ours_rootloss - sir_rootloss;
            d_edge += ours_edge - sir_edge;
            d_bonus += ours_bonus - sir_bonus;
        }
        if gap > 1e-6 {
            ours_higher += 1;
            worst.push((gap, name, format!("{} vs {} nodes", nodes.len(), frags.len())));
        } else if gap < -1e-6 {
            sirius_higher += 1;
            worst.push((gap, name, format!("{} vs {} nodes", nodes.len(), frags.len())));
        } else {
            tied += 1;
        }
    }

    println!("mirror validation: {validated} reproduced our own total_score, {validation_failed} did not");
    println!("SIRIUS trees not representable in our graph: {not_representable}");
    println!("  peak/tolerance mismatch {missing_vertex}, fragment rejected by our rules {rejected_fragment}, loss rejected {rejected_loss}, unparseable edge {missing_edge}");
    println!("of the representable ones, under OUR objective:");
    println!("  our tree scores higher:    {ours_higher}");
    println!("  SIRIUS's tree scores higher: {sirius_higher}   <-- any of these means our solver is not exact");
    println!("  tied:                      {tied}");
    if !gaps.is_empty() {
        let mut g = gaps.clone();
        g.sort_by(f64::total_cmp);
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        println!("  score gap (ours - SIRIUS): mean {mean:+.3}  median {:+.3}  min {:+.3}  max {:+.3}",
                 g[g.len() / 2], g[0], g[g.len() - 1]);
    }
    if differing > 0 {
        let n = differing as f64;
        println!("\nwhere the gap comes from, averaged over the {differing} differing spectra:");
        println!("  vertex_score  (mass dev + fragment priors) {:+8.3}", d_vertex / n);
        println!("  root_loss_score (cumulative loss)          {:+8.3}", d_rootloss / n);
        println!("  edge_score    (the loss scorers)           {:+8.3}", d_edge / n);
        println!("  tree-size bonus * node count               {:+8.3}", d_bonus / n);
        println!("  ------------------------------------------ {:+8.3}",
                 (d_vertex + d_rootloss + d_edge + d_bonus) / n);
    }
    worst.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    println!("\nlargest gaps:");
    for (g, n, sz) in worst.iter().take(8) {
        println!("  {g:+9.3}  {n}  ({sz})");
    }
}
