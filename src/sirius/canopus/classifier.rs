//! CANOPUS Compound Classification from Molecular Fingerprints.
//!
//! Predicts ClassyFire chemical classes, superclasses, and subclasses directly
//! from predicted molecular fingerprint probability vectors.

use super::ontology::{
    find_node_by_chemont_id, get_children_of, ClassyFireNode, TaxonomyLevel, CLASSYFIRE_TAXONOMY,
};
use crate::sirius::fingerprint::predictor::PredictedFingerprint;

/// A single predicted ClassyFire taxonomic hit with its posterior probability and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassyFireHit {
    /// CHEMONT identifier (e.g. "CHEMONTID:0000019")
    pub chemont_id: String,
    /// Chemical class name (e.g. "Flavonoids")
    pub name: String,
    /// Taxonomy level (Kingdom, Superclass, Class, Subclass, DirectParent)
    pub level: TaxonomyLevel,
    /// Posterior probability (0.0 to 1.0)
    pub probability: f32,
    /// Definition / description of the class
    pub description: String,
}

/// Comprehensive hierarchical compound classification result for a molecule or spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundClassification {
    /// Predicted Kingdom (e.g. "Organic compounds")
    pub kingdom: Option<ClassyFireHit>,
    /// Predicted Superclass (e.g. "Phenylpropanoids and polyketides", "Lipids and lipid-like molecules")
    pub superclass: Option<ClassyFireHit>,
    /// Predicted Class (e.g. "Flavonoids", "Isoquinolines and derivatives", "Prenol lipids")
    pub class: Option<ClassyFireHit>,
    /// Predicted Subclass (e.g. "Flavones", "Flavonols", "Aporphine alkaloids")
    pub subclass: Option<ClassyFireHit>,
    /// Most specific direct parent classification term
    pub direct_parent: Option<ClassyFireHit>,
    /// All evaluated ontology terms with posterior probabilities above the cutoff
    pub all_hits: Vec<ClassyFireHit>,
}

/// High-performance CANOPUS ClassyFire chemical taxonomy classifier.
#[derive(Debug, Clone)]
pub struct CanopusClassifier {
    /// Probability threshold for reporting positive class assignments (default: 0.50)
    pub threshold: f32,
}

impl Default for CanopusClassifier {
    fn default() -> Self { Self::new() }
}

impl CanopusClassifier {
    /// Creates a new CANOPUS classifier with the default probability threshold (0.50).
    pub fn new() -> Self {
        Self { threshold: 0.50 }
    }

    /// Creates a new CANOPUS classifier with a custom reporting threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Predicts hierarchical ClassyFire compound classification from a predicted fingerprint.
    pub fn classify(&self, predicted_fp: &PredictedFingerprint) -> CompoundClassification {
        let probs = &predicted_fp.probabilities;

        // 1. Score every ontology term in the taxonomy
        let mut hit_scores: Vec<ClassyFireHit> = Vec::with_capacity(CLASSYFIRE_TAXONOMY.len());

        for node in CLASSYFIRE_TAXONOMY {
            let p_class = self.compute_node_probability(node, probs);
            if p_class >= self.threshold {
                hit_scores.push(ClassyFireHit {
                    chemont_id: node.chemont_id.to_string(),
                    name: node.name.to_string(),
                    level: node.level,
                    probability: p_class,
                    description: node.description.to_string(),
                });
            }
        }

        // Sort hits by probability descending
        hit_scores.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Resolve the most probable hierarchical path: Kingdom -> Superclass -> Class -> Subclass
        let kingdom = hit_scores
            .iter()
            .find(|h| h.level == TaxonomyLevel::Kingdom)
            .cloned();

        let superclass = hit_scores
            .iter()
            .find(|h| h.level == TaxonomyLevel::Superclass)
            .cloned();

        let class = if let Some(ref sc) = superclass {
            // Prefer child class belonging to the top superclass
            let children = get_children_of(&sc.chemont_id);
            hit_scores
                .iter()
                .find(|h| h.level == TaxonomyLevel::Class && children.iter().any(|c| c.chemont_id == h.chemont_id))
                .cloned()
                .or_else(|| hit_scores.iter().find(|h| h.level == TaxonomyLevel::Class).cloned())
        } else {
            hit_scores.iter().find(|h| h.level == TaxonomyLevel::Class).cloned()
        };

        let subclass = if let Some(ref c) = class {
            // Prefer child subclass belonging to the top class
            let children = get_children_of(&c.chemont_id);
            hit_scores
                .iter()
                .find(|h| h.level == TaxonomyLevel::Subclass && children.iter().any(|sub| sub.chemont_id == h.chemont_id))
                .cloned()
                .or_else(|| hit_scores.iter().find(|h| h.level == TaxonomyLevel::Subclass).cloned())
        } else {
            hit_scores.iter().find(|h| h.level == TaxonomyLevel::Subclass).cloned()
        };

        // Direct parent is the most specific valid level (Subclass > Class > Superclass > Kingdom)
        let direct_parent = subclass
            .clone()
            .or_else(|| class.clone())
            .or_else(|| superclass.clone())
            .or_else(|| kingdom.clone());

        CompoundClassification {
            kingdom,
            superclass,
            class,
            subclass,
            direct_parent,
            all_hits: hit_scores,
        }
    }

    /// Computes the posterior probability of a ClassyFire node given the predicted fingerprint vector.
    fn compute_node_probability(&self, node: &ClassyFireNode, probs: &[f32]) -> f32 {
        if node.key_maccs.is_empty() {
            return node.prior;
        }

        let mut log_odds_sum = 0.0f32;
        let eps = 1e-4f32;

        for &maccs_idx in node.key_maccs {
            if maccs_idx < probs.len() {
                let p_bit = probs[maccs_idx].clamp(eps, 1.0 - eps);
                // Weight support relative to neutral 0.5 baseline
                log_odds_sum += (p_bit / (1.0 - p_bit)).ln();
            }
        }

        let n = node.key_maccs.len() as f32;
        let avg_log_odds = log_odds_sum / n;

        // Prior-weighted Bayesian logistic combination
        let prior_odds = (node.prior / (1.0 - node.prior)).ln();
        let logit = 0.7 * avg_log_odds + 0.3 * prior_odds;

        1.0 / (1.0 + (-logit).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canopus_flavonoid_classification() {
        let classifier = CanopusClassifier::new();

        // Create a synthetic fingerprint with strong Flavonoid bits (MACCS 144, 145, 139, 131, 129)
        let mut probs = vec![0.05f32; 2214];
        probs[144] = 0.95;
        probs[145] = 0.92;
        probs[139] = 0.90;
        probs[131] = 0.88;
        probs[129] = 0.85;
        probs[164] = 0.98; // Organic carbon

        let predicted = PredictedFingerprint {
            probabilities: probs,
        };

        let res = classifier.classify(&predicted);
        assert!(res.superclass.is_some());
        assert_eq!(
            res.superclass.as_ref().unwrap().name,
            "Phenylpropanoids and polyketides"
        );
        assert!(res.class.is_some());
        assert_eq!(res.class.as_ref().unwrap().name, "Flavonoids");
    }

    #[test]
    fn test_canopus_lipid_classification() {
        let classifier = CanopusClassifier::new();

        // Create a synthetic fingerprint with strong fatty acyl aliphatic bits (MACCS 154, 158, 160, 141)
        let mut probs = vec![0.05f32; 2214];
        probs[154] = 0.96;
        probs[158] = 0.94;
        probs[160] = 0.95;
        probs[141] = 0.90; // Carboxylic acid / ester
        probs[164] = 0.99;

        let predicted = PredictedFingerprint {
            probabilities: probs,
        };

        let res = classifier.classify(&predicted);
        assert!(res.superclass.is_some());
        assert_eq!(
            res.superclass.as_ref().unwrap().name,
            "Lipids and lipid-like molecules"
        );
        assert!(res.class.is_some());
        assert_eq!(res.class.as_ref().unwrap().name, "Fatty Acyls");
    }
}
