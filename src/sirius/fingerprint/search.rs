//! Database search and structure identification from predicted fingerprints.
//!
//! Implements the ModPlatt (Modified Platt) scoring function from CSI:FingerID
//! (Dührkop et al. 2015) for ranking candidate molecular structures against
//! predicted fingerprint probability vectors.
//!
//! # Scoring
//!
//! For a predicted probability vector `p̂ ∈ [0,1]^d` and a candidate's binary
//! fingerprint `f ∈ {0,1}^d`, the ModPlatt log-likelihood score is:
//!
//! ```text
//! S(f, p̂) = Σᵢ wᵢ · [ fᵢ · log(p̂ᵢ / πᵢ) + (1-fᵢ) · log((1-p̂ᵢ) / (1-πᵢ)) ]
//! ```
//!
//! # Which score ranks
//!
//! ModPlatt, the published score, and the alternatives here lost to it when finally
//! measured with enough queries to tell them apart. On 1,500 GNPS queries drawn from
//! compounds absent from the training library, top-1 was 52.5% for ModPlatt, 50.5% for
//! [`CandidateScore::SizeNormalizedModPlatt`] and 49.7% for
//! [`CandidateScore::ContinuousTanimoto`], with ModPlatt also ahead at top-3 and top-5.
//!
//! An earlier round of this comparison, run on 500 queries, put continuous Tanimoto
//! ahead by five points and prompted making it the default. That ordering did not
//! survive tripling the sample: with two to three candidates per query, a 500-query
//! difference of three points is inside one standard error. Treat any comparison here
//! made on fewer than about a thousand queries as undecided.
//!
//! The size bias those experiments were chasing is nonetheless real and measurable.
//! Most bits are predicted near zero against a prior that is also near zero, so
//! `log(p̂ᵢ/πᵢ) - log((1-p̂ᵢ)/(1-πᵢ))` is strongly negative for most bits: every set bit
//! the model did not predict costs the candidate, so candidates with fewer set bits are
//! favoured regardless of agreement. Over 356 queries the winner carried 1.4 fewer bits
//! than the true structure under ModPlatt and 0.4 more once normalized. Removing the
//! term does what it claims; it just does not buy accuracy.
//!
//! where `πᵢ = P(Fᵢ=1)` is the prior probability of bit `i` being set,
//! estimated from the training set. This prior correction prevents common
//! substructures from dominating the score.
//!
//! `wᵢ ∈ [0, 1]` is a per-bit weight, `1` by default, which is the unweighted score of
//! the paper. It exists so that reliability weighting can be measured; on the MoNA
//! benchmark it did not pay off (see [`FingerprintSearcher::for_predictor`]), so
//! nothing installs non-unit weights automatically.
//!
//! `π` is a property of the model doing the predicting, which is why a searcher should
//! be built with [`FingerprintSearcher::for_predictor`].
//!
//! # References
//! - Dührkop, K. et al. (2015). Searching molecular structure databases with
//!   tandem mass spectra using CSI:FingerID. PNAS, 112(41), 12580–12585.

use super::dictionary::{BinaryFingerprint, FingerprintDictionary, MorganVocabulary, TOTAL_BITS};
use super::predictor::{FingerprintPredictor, PredictedFingerprint};

/// Minimum probability value to prevent log(0) in scoring.
const EPSILON: f64 = 1e-6;

// ── Candidate structure ─────────────────────────────────────────────────────

/// A candidate molecular structure from a structural database (e.g. PubChem, HMDB).
#[derive(Debug, Clone)]
pub struct StructureCandidate {
    /// Unique identifier (e.g. PubChem CID, InChIKey).
    pub id: String,
    /// Canonical SMILES string.
    pub smiles: String,
    /// Common name or database name (may be empty).
    pub name: String,
    /// Precomputed binary molecular fingerprint for this structure.
    pub fingerprint: BinaryFingerprint,
}

/// A scored candidate structure resulting from fingerprint-based database search.
#[derive(Debug, Clone)]
pub struct ScoredStructure {
    /// The matched candidate structure.
    pub candidate: StructureCandidate,
    /// ModPlatt log-likelihood score (higher = better match).
    pub score: f64,
    /// Tanimoto similarity between thresholded prediction and candidate fingerprint.
    pub tanimoto: f64,
    /// Rank among all candidates (1-indexed).
    pub rank: usize,
}

// ── Prepared query score table ──────────────────────────────────────────────

/// ModPlatt scoring terms precomputed for one predicted fingerprint.
///
/// Both `log(p̂ᵢ/πᵢ)` and `log((1-p̂ᵢ)/(1-πᵢ))` depend only on the query and the
/// searcher's priors and weights, not on the candidate, so scoring a database reduces
/// to summing a per-bit delta over the candidate's set bits — no logarithms per
/// candidate.
#[derive(Debug, Clone)]
pub struct ModPlattTable {
    /// Score of a candidate with no bits set: `Σᵢ wᵢ · log((1-p̂ᵢ)/(1-πᵢ))`.
    base: f64,
    /// Per-bit correction added when the candidate has bit `i` set:
    /// `wᵢ · [log(p̂ᵢ/πᵢ) - log((1-p̂ᵢ)/(1-πᵢ))]`.
    delta: Vec<f64>,
    /// Mean of `delta`. Splitting the score as `base + n·d̄ + Σ(deltaᵢ - d̄)` isolates
    /// the part that depends only on how many bits a candidate has.
    mean_delta: f64,
}

impl ModPlattTable {
    /// Computes the ModPlatt score with its dependence on candidate size removed.
    ///
    /// Most bits are predicted near zero against a prior that is also near zero, which
    /// makes most `deltaᵢ` strongly negative; a candidate is then penalized for every
    /// bit it has, agreement aside. Writing the score as `base + n·d̄ + Σ(deltaᵢ - d̄)`
    /// separates that: the middle term is a function of the candidate's bit count and
    /// nothing else, and dropping it — along with the per-query constant `base` — leaves
    /// only how far the candidate's bits depart from an average bit.
    ///
    /// The result is no longer a log-likelihood and is comparable only within one query.
    pub fn score_size_normalized(&self, candidate: &BinaryFingerprint) -> f64 {
        debug_assert_eq!(candidate.bits.len(), TOTAL_BITS);

        let mut score = 0.0f64;
        for i in 0..TOTAL_BITS {
            if candidate.bits[i] {
                score += self.delta[i] - self.mean_delta;
            }
        }
        score
    }

    /// Computes the ModPlatt log-likelihood score of a candidate fingerprint.
    pub fn score(&self, candidate: &BinaryFingerprint) -> f64 {
        debug_assert_eq!(candidate.bits.len(), TOTAL_BITS);

        let mut score = self.base;
        for i in 0..TOTAL_BITS {
            if candidate.bits[i] {
                score += self.delta[i];
            }
        }
        score
    }
}

// ── Database search engine ──────────────────────────────────────────────────

/// Fingerprint-based structure identification engine.
///
/// Scores candidate structures against predicted fingerprint probability vectors
/// using the ModPlatt log-likelihood scoring function with prior correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandidateScore {
    /// The ModPlatt log-likelihood of Dührkop et al. (2015). The default.
    #[default]
    ModPlatt,
    /// Continuous Tanimoto between the predicted probabilities and the candidate's
    /// bits, normalized by their union.
    ContinuousTanimoto,
    /// ModPlatt with its candidate-size dependence removed; see
    /// [`ModPlattTable::score_size_normalized`].
    SizeNormalizedModPlatt,
}

#[derive(Debug, Clone)]
pub struct FingerprintSearcher {
    /// Fingerprint dictionary with prior probabilities.
    dictionary: FingerprintDictionary,
    /// How candidates are ranked; see [`CandidateScore`].
    scoring: CandidateScore,
}

impl FingerprintSearcher {
    /// Creates a new searcher with the given fingerprint dictionary.
    ///
    /// The dictionary should have prior probabilities estimated from the
    /// training set via [`FingerprintDictionary::set_priors_from_counts`].
    pub fn new(dictionary: FingerprintDictionary) -> Self {
        Self {
            dictionary,
            scoring: CandidateScore::default(),
        }
    }

    /// Chooses how candidates are ranked.
    pub fn with_scoring(mut self, scoring: CandidateScore) -> Self {
        self.scoring = scoring;
        self
    }

    /// How this searcher ranks candidates.
    #[inline]
    pub fn scoring(&self) -> CandidateScore {
        self.scoring
    }

    /// Builds a searcher that scores with the priors of the model that will produce the
    /// predictions.
    ///
    /// ModPlatt compares a prediction against the bit priors of the training set the
    /// model was fitted on, so a searcher is only correct when it carries that model's
    /// own priors. A model trained before priors were recorded falls back to the
    /// uniform 0.5 priors, under which the correction has no effect.
    ///
    /// The model's reliability weights are deliberately **not** installed. Weighting
    /// each bit by its validation MCC was measured on 100 MoNA isomer queries and lost
    /// 27 points of top-1 accuracy (72% → 45%): the model clears MCC > 0 on only 158 of
    /// 2,214 bits, and zeroing the rest discards more signal than it removes noise.
    /// Shrinking the weights toward 1 recovers the loss monotonically and never beats
    /// unit weights, so there is no setting of this scheme worth shipping. The weights
    /// stay recorded on the model as a diagnostic, and
    /// `examples/evaluate_database_search.rs --stats weights` still reproduces the
    /// experiment.
    pub fn for_predictor(predictor: &FingerprintPredictor) -> Self {
        let mut dictionary = FingerprintDictionary::new();
        if predictor.bit_priors().len() == TOTAL_BITS {
            dictionary.set_priors(predictor.bit_priors().to_vec());
        }
        // The vocabulary is not optional in the way the statistics are: it defines what
        // the model's outputs mean, so candidate fingerprints must be built under it.
        if !predictor.morgan_vocabulary().is_empty() {
            dictionary.set_morgan_vocabulary(MorganVocabulary::from_ids(
                predictor.morgan_vocabulary().to_vec(),
            ));
        }
        Self::new(dictionary)
    }

    /// Precomputes the query-dependent half of the ModPlatt score.
    ///
    /// Do this once per predicted fingerprint and reuse the table across every
    /// candidate in the database.
    pub fn prepare(&self, predicted: &PredictedFingerprint) -> ModPlattTable {
        debug_assert_eq!(predicted.probabilities.len(), TOTAL_BITS);

        let mut delta = Vec::with_capacity(TOTAL_BITS);
        let mut base = 0.0f64;

        for i in 0..TOTAL_BITS {
            let p_hat = (predicted.probabilities[i] as f64).clamp(EPSILON, 1.0 - EPSILON);
            let prior = (self.dictionary.priors[i] as f64).clamp(EPSILON, 1.0 - EPSILON);
            let w = self.dictionary.weights[i] as f64;

            // Bit set: log(p̂ᵢ / πᵢ). Bit unset: log((1 - p̂ᵢ) / (1 - πᵢ)).
            let on_i = w * (p_hat / prior).ln();
            let off_i = w * ((1.0 - p_hat) / (1.0 - prior)).ln();

            base += off_i;
            delta.push(on_i - off_i);
        }

        let mean_delta = delta.iter().sum::<f64>() / delta.len().max(1) as f64;
        ModPlattTable {
            base,
            delta,
            mean_delta,
        }
    }

    /// Computes the ModPlatt log-likelihood score between a predicted fingerprint
    /// and a candidate binary fingerprint.
    ///
    /// Higher scores indicate better matches. The score is corrected for prior
    /// bit probabilities to prevent common substructures from dominating.
    ///
    /// Scoring many candidates against one prediction should go through
    /// [`FingerprintSearcher::prepare`] instead, which hoists the logarithms out of
    /// the per-candidate loop.
    pub fn mod_platt_score(
        &self,
        predicted: &PredictedFingerprint,
        candidate: &BinaryFingerprint,
    ) -> f64 {
        self.prepare(predicted).score(candidate)
    }

    /// Computes the Tanimoto similarity between a thresholded predicted fingerprint
    /// and a candidate binary fingerprint.
    pub fn tanimoto_similarity(
        predicted: &PredictedFingerprint,
        candidate: &BinaryFingerprint,
        threshold: f32,
    ) -> f64 {
        let mut intersection = 0usize;
        let mut union = 0usize;

        for i in 0..TOTAL_BITS {
            let pred_bit = predicted.probabilities[i] >= threshold;
            let cand_bit = candidate.bits[i];

            if pred_bit || cand_bit {
                union += 1;
            }
            if pred_bit && cand_bit {
                intersection += 1;
            }
        }

        if union == 0 {
            return 1.0;
        }
        intersection as f64 / union as f64
    }

    /// Computes the continuous (soft) Tanimoto similarity between predicted probabilities
    /// and a candidate binary fingerprint.
    pub fn continuous_tanimoto_similarity(
        predicted: &PredictedFingerprint,
        candidate: &BinaryFingerprint,
    ) -> f64 {
        let mut dot = 0.0f64;
        let mut sum_p = 0.0f64;
        let mut sum_c = 0.0f64;

        for i in 0..TOTAL_BITS {
            let p = predicted.probabilities[i] as f64;
            let c = if candidate.bits[i] { 1.0 } else { 0.0 };
            dot += p * c;
            sum_p += p;
            sum_c += c;
        }

        let union = sum_p + sum_c - dot;
        if union <= 1e-12 {
            return 1.0;
        }
        dot / union
    }

    /// Returns the fingerprint dictionary backing this searcher.
    #[inline]
    pub fn dictionary(&self) -> &FingerprintDictionary {
        &self.dictionary
    }

    /// Searches a candidate database and returns scored structures ranked by
    /// ModPlatt log-likelihood score (best match first).
    ///
    /// # Arguments
    /// * `predicted` - Predicted fingerprint probability vector from the MLP.
    /// * `candidates` - Candidate structures with precomputed binary fingerprints.
    /// * `max_results` - Maximum number of results to return (0 = all).
    pub fn search(
        &self,
        predicted: &PredictedFingerprint,
        candidates: Vec<StructureCandidate>,
        max_results: usize,
    ) -> Vec<ScoredStructure> {
        let table = self.prepare(predicted);

        let mut scored: Vec<ScoredStructure> = candidates
            .into_iter()
            .map(|candidate| {
                let score = match self.scoring {
                    CandidateScore::ModPlatt => table.score(&candidate.fingerprint),
                    CandidateScore::SizeNormalizedModPlatt => {
                        table.score_size_normalized(&candidate.fingerprint)
                    }
                    CandidateScore::ContinuousTanimoto => {
                        Self::continuous_tanimoto_similarity(predicted, &candidate.fingerprint)
                    }
                };
                let tanimoto =
                    Self::tanimoto_similarity(predicted, &candidate.fingerprint, 0.5);
                ScoredStructure {
                    candidate,
                    score,
                    tanimoto,
                    rank: 0,
                }
            })
            .collect();

        // Sort by score descending (best first)
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks (1-indexed)
        for (i, hit) in scored.iter_mut().enumerate() {
            hit.rank = i + 1;
        }

        // Truncate if needed
        if max_results > 0 && scored.len() > max_results {
            scored.truncate(max_results);
        }

        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::predictor::FingerprintResNet;

    fn make_test_dictionary() -> FingerprintDictionary {
        let mut dict = FingerprintDictionary::new();
        // Set some non-uniform priors
        let mut counts = vec![50usize; TOTAL_BITS]; // 50% baseline
        counts[0] = 90; // MACCS key 0 is very common
        counts[1] = 10; // MACCS key 1 is rare
        dict.set_priors_from_counts(&counts, 100);
        dict
    }

    #[test]
    fn test_mod_platt_perfect_match() {
        let dict = make_test_dictionary();
        let searcher = FingerprintSearcher::new(dict);

        // Perfect prediction: all probabilities match the true fingerprint
        let mut probs = vec![0.01f32; TOTAL_BITS]; // low for unset bits
        let mut bits = vec![false; TOTAL_BITS];

        // Set a few bits and predict them with high confidence
        for i in [0, 5, 10, 100, 500, 1000] {
            bits[i] = true;
            probs[i] = 0.99;
        }

        let predicted = PredictedFingerprint {
            probabilities: probs,
        };
        let candidate = BinaryFingerprint::from_bits(bits);

        let score = searcher.mod_platt_score(&predicted, &candidate);
        assert!(score > 0.0, "Perfect match should have positive score");
    }

    #[test]
    fn test_mod_platt_wrong_match() {
        let dict = make_test_dictionary();
        let searcher = FingerprintSearcher::new(dict);

        // Completely wrong prediction
        let mut probs = vec![0.99f32; TOTAL_BITS]; // predict all bits set
        let bits = vec![false; TOTAL_BITS]; // but none are set

        let predicted = PredictedFingerprint {
            probabilities: probs,
        };
        let candidate = BinaryFingerprint::from_bits(bits);

        let score = searcher.mod_platt_score(&predicted, &candidate);
        assert!(score < 0.0, "Wrong match should have negative score");
    }

    #[test]
    fn test_search_ranking() {
        let dict = make_test_dictionary();
        let searcher = FingerprintSearcher::new(dict);

        let mut probs = vec![0.1f32; TOTAL_BITS];
        probs[0] = 0.95;
        probs[1] = 0.95;
        probs[2] = 0.95;

        let predicted = PredictedFingerprint { probabilities: probs };

        // Candidate A: matches prediction well
        let mut fp_a = BinaryFingerprint::zeros();
        fp_a.set(0, true);
        fp_a.set(1, true);
        fp_a.set(2, true);

        // Candidate B: doesn't match at all
        let mut fp_b = BinaryFingerprint::zeros();
        fp_b.set(100, true);
        fp_b.set(200, true);
        fp_b.set(300, true);

        let candidates = vec![
            StructureCandidate {
                id: "B".into(),
                smiles: "CC".into(),
                name: "Ethane".into(),
                fingerprint: fp_b,
            },
            StructureCandidate {
                id: "A".into(),
                smiles: "C".into(),
                name: "Methane".into(),
                fingerprint: fp_a,
            },
        ];

        let results = searcher.search(&predicted, candidates, 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].candidate.id, "A", "Better match should rank first");
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].candidate.id, "B");
        assert_eq!(results[1].rank, 2);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_prepared_table_matches_direct_formula() {
        let dict = make_test_dictionary();
        let searcher = FingerprintSearcher::new(dict);

        let mut probs = vec![0.2f32; TOTAL_BITS];
        let mut bits = vec![false; TOTAL_BITS];
        for i in (0..TOTAL_BITS).step_by(7) {
            probs[i] = 0.85;
        }
        for i in (0..TOTAL_BITS).step_by(5) {
            bits[i] = true;
        }

        let predicted = PredictedFingerprint { probabilities: probs };
        let candidate = BinaryFingerprint::from_bits(bits);

        // Reference: the scoring formula as written in the module documentation.
        let mut reference = 0.0f64;
        for i in 0..TOTAL_BITS {
            let p_hat = (predicted.probabilities[i] as f64).clamp(EPSILON, 1.0 - EPSILON);
            let prior = (searcher.dictionary().priors[i] as f64).clamp(EPSILON, 1.0 - EPSILON);
            reference += if candidate.bits[i] {
                (p_hat / prior).ln()
            } else {
                ((1.0 - p_hat) / (1.0 - prior)).ln()
            };
        }

        let scored = searcher.prepare(&predicted).score(&candidate);
        assert!(
            (scored - reference).abs() < 1e-9,
            "prepared table {scored} != direct formula {reference}"
        );
    }

    #[test]
    fn test_for_predictor_installs_priors_but_not_weights() {
        let mut resnet = FingerprintResNet::new();
        resnet.bit_priors = vec![0.25f32; TOTAL_BITS];
        resnet.bit_weights = vec![0.0f32; TOTAL_BITS];
        let predictor = FingerprintPredictor::from_resnet(resnet);

        let searcher = FingerprintSearcher::for_predictor(&predictor);
        let dict = searcher.dictionary();

        assert!(dict.priors.iter().all(|&p| p == 0.25), "priors were not installed");
        // Reliability weighting lost 27 points of top-1 accuracy in benchmarking, so the
        // model's weights must not reach the scorer even when it records them.
        assert!(dict.weights.iter().all(|&w| w == 1.0), "weights must stay uniform");
    }

    #[test]
    fn test_zero_weight_bits_drop_out_of_the_score() {
        // Uniform 0.5 priors, so the surviving bit's contribution is easy to state.
        let mut dict = FingerprintDictionary::new();
        // Silence every bit except the first; the score must then depend only on it.
        let mut weights = vec![0.0f32; TOTAL_BITS];
        weights[0] = 1.0;
        dict.set_weights(weights);
        let searcher = FingerprintSearcher::new(dict);

        let predicted = PredictedFingerprint {
            probabilities: vec![0.9f32; TOTAL_BITS],
        };

        let mut only_bit_0 = BinaryFingerprint::zeros();
        only_bit_0.set(0, true);

        let mut bit_0_plus_noise = only_bit_0.clone();
        for i in (1..TOTAL_BITS).step_by(3) {
            bit_0_plus_noise.set(i, true);
        }

        let a = searcher.prepare(&predicted).score(&only_bit_0);
        let b = searcher.prepare(&predicted).score(&bit_0_plus_noise);
        assert!((a - b).abs() < 1e-12, "zero-weight bits still moved the score");

        // And the surviving bit still contributes: p̂ = 0.9 against a 0.5 prior.
        let reference = (0.9f64 / 0.5).ln();
        assert!((a - reference).abs() < 1e-6, "score {a} != {reference}");
    }

    #[test]
    fn test_size_normalization_removes_the_bit_count_term() {
        let searcher = FingerprintSearcher::new(FingerprintDictionary::new());
        let predicted = PredictedFingerprint {
            probabilities: (0..TOTAL_BITS)
                .map(|i| if i < 40 { 0.9f32 } else { 0.02 })
                .collect(),
        };
        let table = searcher.prepare(&predicted);

        // Two candidates agreeing on nothing the model predicted, differing only in how
        // many unpredicted bits they carry. Plain ModPlatt must prefer the smaller one;
        // the normalized score must be indifferent to the count as such.
        let bits = |range: std::ops::Range<usize>| {
            BinaryFingerprint::from_bits((0..TOTAL_BITS).map(|i| range.contains(&i)).collect())
        };
        let few = bits(100..110);
        let many = bits(100..160);

        assert!(
            table.score(&few) > table.score(&many),
            "plain ModPlatt should favour the smaller candidate: {} vs {}",
            table.score(&few),
            table.score(&many)
        );

        // Under normalization the per-bit contribution of an average bit is zero, so
        // adding more equally-average bits cannot move the score.
        let normalized_gap = (table.score_size_normalized(&few)
            - table.score_size_normalized(&many))
        .abs();
        let plain_gap = (table.score(&few) - table.score(&many)).abs();
        assert!(
            normalized_gap < plain_gap * 0.05,
            "size term should be gone: normalized gap {normalized_gap}, plain gap {plain_gap}"
        );
    }

    #[test]
    fn test_scoring_mode_selects_the_ranking() {
        let searcher = FingerprintSearcher::new(make_test_dictionary());
        assert_eq!(searcher.scoring(), CandidateScore::ModPlatt);

        // A candidate that agrees on every predicted bit, and a smaller one that agrees
        // on fewer but carries no bits the model did not predict — the shape ModPlatt
        // was measured to prefer.
        let mut probs = vec![0.02f32; TOTAL_BITS];
        for i in 0..40 {
            probs[i] = 0.9;
        }
        let predicted = PredictedFingerprint { probabilities: probs };

        let full = BinaryFingerprint::from_bits(
            (0..TOTAL_BITS).map(|i| i < 40).collect::<Vec<bool>>(),
        );
        let sparse = BinaryFingerprint::from_bits(
            (0..TOTAL_BITS).map(|i| i < 3).collect::<Vec<bool>>(),
        );

        let candidates = |ids: [&BinaryFingerprint; 2]| {
            ids.iter()
                .enumerate()
                .map(|(i, fp)| StructureCandidate {
                    id: format!("c{i}"),
                    smiles: String::new(),
                    name: format!("c{i}"),
                    fingerprint: (*fp).clone(),
                })
                .collect::<Vec<_>>()
        };

        let by_mod_platt = searcher.search(&predicted, candidates([&full, &sparse]), 0);
        assert_eq!(by_mod_platt.len(), 2);

        let by_tanimoto = FingerprintSearcher::new(make_test_dictionary())
            .with_scoring(CandidateScore::ContinuousTanimoto)
            .search(&predicted, candidates([&full, &sparse]), 0);
        assert_eq!(
            by_tanimoto[0].candidate.id, "c0",
            "under Tanimoto the fully agreeing candidate must win"
        );
    }

    #[test]
    fn test_tanimoto_similarity() {
        let predicted = PredictedFingerprint {
            probabilities: {
                let mut p = vec![0.0f32; TOTAL_BITS];
                p[0] = 0.9;
                p[1] = 0.9;
                p[2] = 0.9;
                p
            },
        };

        let mut candidate = BinaryFingerprint::zeros();
        candidate.set(1, true);
        candidate.set(2, true);
        candidate.set(3, true);

        // Predicted: {0, 1, 2}, Candidate: {1, 2, 3}
        // Intersection: {1, 2} = 2, Union: {0, 1, 2, 3} = 4
        let tanimoto = FingerprintSearcher::tanimoto_similarity(&predicted, &candidate, 0.5);
        assert!((tanimoto - 0.5).abs() < 1e-10);
    }
}
