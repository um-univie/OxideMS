//! Feature extraction from fragmentation trees for fingerprint prediction.
//!
//! Converts a `FragmentationTree` into a fixed-size numerical feature vector
//! suitable for neural network input. Follows the CSI:FingerID feature categories
//! (Dührkop et al. 2015) using feature hashing (Weinberger et al. 2009) with
//! square-root intensity scaling and rich heteroatom statistics.
//!
//! # Feature Categories (Total = 5,147 dimensions)
//!
//! | Offset       | Size                  | Category                         |
//! |--------------|-----------------------|----------------------------------|
//! | 0            | 1,024                 | Fragment formula binary          |
//! | 1,024        | 1,024                 | Fragment formula intensity (√I)  |
//! | 2,048        | 512                   | Neutral loss formula binary      |
//! | 2,560        | 512                   | Neutral loss formula intensity(√I)|
//! | 3,072        | 1,024                 | Root loss formula binary         |
//! | 4,096        | 1,024                 | Root loss formula intensity (√I) |
//! | 5,120        | 27                    | Tree-level scalar statistics     |
//!
//! # References
//! - Dührkop, K. et al. (2015). Searching molecular structure databases with tandem
//!   mass spectra using CSI:FingerID. PNAS, 112(41), 12580–12585.
//! - Shen, H. et al. (2014). Metabolite identification through multiple kernel
//!   learning on fragmentation trees. Bioinformatics, 30(12), i157–i164.

use super::super::element::NUM_ELEMENTS;
use super::super::formula::Formula;
use super::super::tree::FragmentationTree;
use xxhash_rust::xxh32::xxh32;

// ── Bucket sizes for hashed feature categories ──────────────────────────────

/// Number of hash buckets for fragment (node) formula features.
pub const FRAGMENT_BUCKETS: usize = 1024;

/// Number of hash buckets for neutral loss (edge) formula features.
pub const LOSS_BUCKETS: usize = 512;

/// Number of hash buckets for root-to-descendant loss formula features.
pub const ROOT_LOSS_BUCKETS: usize = 1024;

/// Number of scalar tree-level statistics.
pub const TREE_STATS: usize = 27;

/// Total feature vector dimensionality (5,147).
pub const TOTAL_FEATURES: usize =
    FRAGMENT_BUCKETS * 2 + LOSS_BUCKETS * 2 + ROOT_LOSS_BUCKETS * 2 + TREE_STATS;

// ── Hash seeds (distinct per category to avoid cross-category collisions) ───

const SEED_FRAGMENT: u32 = 0x4652_4147; // "FRAG"
const SEED_LOSS: u32 = 0x4C4F_5353;     // "LOSS"
const SEED_ROOT_LOSS: u32 = 0x524F_4F54; // "ROOT"

// ── Feature vector offsets ──────────────────────────────────────────────────

const OFF_FRAG_BIN: usize = 0;
const OFF_FRAG_INT: usize = FRAGMENT_BUCKETS;
const OFF_LOSS_BIN: usize = FRAGMENT_BUCKETS * 2;
const OFF_LOSS_INT: usize = FRAGMENT_BUCKETS * 2 + LOSS_BUCKETS;
const OFF_ROOT_BIN: usize = FRAGMENT_BUCKETS * 2 + LOSS_BUCKETS * 2;
const OFF_ROOT_INT: usize = FRAGMENT_BUCKETS * 2 + LOSS_BUCKETS * 2 + ROOT_LOSS_BUCKETS;
const OFF_STATS: usize = FRAGMENT_BUCKETS * 2 + LOSS_BUCKETS * 2 + ROOT_LOSS_BUCKETS * 2;

// ── Dedicated High-Frequency Vocabulary ─────────────────────────────────────

/// Number of dedicated slots for common diagnostic fragment ions.
pub const DEDICATED_FRAGMENTS: usize = 128;

/// Number of dedicated slots for common neutral losses.
pub const DEDICATED_LOSSES: usize = 64;

/// Number of dedicated slots for common root-to-descendant neutral losses.
pub const DEDICATED_ROOT_LOSSES: usize = 64;

const COMMON_LOSS_STRINGS: &[&str] = &[
    "H2O", "CO", "CO2", "NH3", "H2S", "HCN", "C2H4", "CH2O", "CH4O", "CH4",
    "C2H2", "H3PO4", "HPO3", "C2H4O2", "C3H6", "C4H8", "C2H4O", "C3H6O", "CH2O2", "SO2",
    "SO3", "N2", "C6H6", "C6H6O", "C7H6O2", "C6H10O5", "C5H8O4", "C2H7N", "CH5N", "HNCO",
    "C2H2O", "H2", "CH2", "C3H4", "C4H6", "C5H8", "C2H3O", "C2H5", "CH3", "C3H7",
    "C4H9", "HCl", "HBr", "HF", "C2H3N", "C3H5NO", "C4H7NO", "CH3S", "C2H6S", "C3H8S",
    "C3H6O2", "C4H8O2", "C5H10O2", "C6H12O2", "C6H12O6", "C5H10O5", "C7H8", "C8H8", "C8H10", "C9H10",
    "C2H6", "C3H8", "C4H10", "C5H12",
];

const COMMON_FRAGMENT_STRINGS: &[&str] = &[
    "C6H5", "C6H6", "C6H7", "C7H7", "C7H8", "C7H9", "C8H9", "C8H10", "C8H11", "C9H11",
    "C6H5O", "C6H6O", "C7H5O", "C7H6O", "C7H7O", "C7H5O2", "C7H7O2", "C8H7O2", "C8H9O2", "C9H9O3",
    "CH3", "C2H5", "C3H3", "C3H5", "C3H7", "C4H7", "C4H9", "C5H7", "C5H9", "C5H11",
    "C6H9", "C6H11", "C6H13", "C7H13", "C7H15", "C8H15", "C8H17", "C9H17", "C9H19", "C10H21",
    "CHO", "CH3O", "C2H3O", "C2H5O", "C3H5O", "C3H7O", "C4H7O", "C4H9O", "C5H9O", "C5H11O",
    "CH2N", "CH4N", "C2H4N", "C2H6N", "C3H6N", "C3H8N", "C4H8N", "C4H10N", "C5H10N", "C5H12N",
    "C6H12N", "C6H14N", "C7H14N", "C7H16N", "C8H16N", "C8H18N", "C9H18N", "C9H20N", "C4H6N", "C5H8N",
    "C2H4NO", "C3H6NO", "C4H8NO", "C5H10NO", "C4H10NO", "C5H12NO", "C8H10N", "C9H12N", "C8H10NO", "C9H12NO",
    "C9H10NO2", "C9H12NO2", "C9H10NO3", "C9H12NO3", "C11H11N2", "C11H13N2", "C6H8N3", "C5H8NO2", "C5H10NO2", "C6H12NO2",
    "C5H5N", "C5H6N", "C6H7N", "C7H9N", "C8H11N", "C9H7N", "C9H8N", "C8H7N", "C8H8N", "C3H4N2",
    "C4H4O", "C4H5O", "C5H5O", "C5H7O", "C4H4S", "C4H5S", "C5H5S", "C5H7S", "C3H3S", "C3H5S",
    "C5H5N5", "C5H6N5", "C4H5N3O", "C5H5N5O", "C5H6N2O2", "C4H4N2O2", "C5H8NO3", "C5H9O4", "C6H11O5", "C6H9O4",
    "H2PO4", "H4PO4", "C2H6PO4", "C3H8PO4", "C2H7NO4P", "C5H15NO4P", "HSO4", "H3SO4",
];

// ── Public types ────────────────────────────────────────────────────────────

/// Extracted feature vector from a fragmentation tree.
#[derive(Debug, Clone)]
pub struct TreeFeatures {
    /// Fixed-size feature vector of length [`TOTAL_FEATURES`].
    pub features: Vec<f32>,
}

impl TreeFeatures {
    /// Returns the total dimensionality of the feature vector.
    #[inline]
    pub fn dim(&self) -> usize {
        TOTAL_FEATURES
    }
}

/// Extracts fixed-size feature vectors from `FragmentationTree` instances.
///
/// Combines a dedicated collision-free vocabulary for high-frequency mass spectrometry
/// losses and fragments with open-vocabulary feature hashing.
#[derive(Debug, Clone)]
pub struct TreeFeatureExtractor {
    /// Number of hash buckets for fragment formula features.
    pub fragment_buckets: usize,
    /// Number of hash buckets for neutral loss formula features.
    pub loss_buckets: usize,
    /// Number of hash buckets for root loss formula features.
    pub root_loss_buckets: usize,
    fragment_vocab: std::collections::HashMap<Formula, usize>,
    loss_vocab: std::collections::HashMap<Formula, usize>,
    root_loss_vocab: std::collections::HashMap<Formula, usize>,
}

impl Default for TreeFeatureExtractor {
    fn default() -> Self {
        let mut fragment_vocab = std::collections::HashMap::new();
        for (i, &s) in COMMON_FRAGMENT_STRINGS.iter().enumerate() {
            if i < DEDICATED_FRAGMENTS {
                if let Ok(f) = s.parse::<Formula>() {
                    fragment_vocab.insert(f, i);
                }
            }
        }

        let mut loss_vocab = std::collections::HashMap::new();
        for (i, &s) in COMMON_LOSS_STRINGS.iter().enumerate() {
            if i < DEDICATED_LOSSES {
                if let Ok(f) = s.parse::<Formula>() {
                    loss_vocab.insert(f, i);
                }
            }
        }

        let mut root_loss_vocab = std::collections::HashMap::new();
        for (i, &s) in COMMON_LOSS_STRINGS.iter().enumerate() {
            if i < DEDICATED_ROOT_LOSSES {
                if let Ok(f) = s.parse::<Formula>() {
                    root_loss_vocab.insert(f, i);
                }
            }
        }

        Self {
            fragment_buckets: FRAGMENT_BUCKETS,
            loss_buckets: LOSS_BUCKETS,
            root_loss_buckets: ROOT_LOSS_BUCKETS,
            fragment_vocab,
            loss_vocab,
            root_loss_vocab,
        }
    }
}

impl TreeFeatureExtractor {
    /// Creates a new extractor with default bucket sizes (1024, 512, 1024).
    pub fn new() -> Self {
        Self::default()
    }

    /// Extracts a fixed-size feature vector from a fragmentation tree.
    pub fn extract(&self, tree: &FragmentationTree) -> TreeFeatures {
        let mut features = vec![0.0f32; TOTAL_FEATURES];

        let root_formula = &tree.precursor_formula;

        // ── Fragment formula features (from tree nodes) ─────────────────────
        for node in &tree.nodes {
            let bucket = get_formula_bucket(
                &node.formula,
                &self.fragment_vocab,
                SEED_FRAGMENT,
                DEDICATED_FRAGMENTS,
                self.fragment_buckets,
            );

            // Binary: mark presence
            features[OFF_FRAG_BIN + bucket] = 1.0;

            // Intensity-weighted with square-root scaling (√I)
            features[OFF_FRAG_INT + bucket] += node.relative_intensity.sqrt();
        }

        // ── Neutral loss features (from tree edges) ─────────────────────────
        for edge in &tree.edges {
            let bucket = get_formula_bucket(
                &edge.loss_formula,
                &self.loss_vocab,
                SEED_LOSS,
                DEDICATED_LOSSES,
                self.loss_buckets,
            );

            // Binary: mark presence
            features[OFF_LOSS_BIN + bucket] = 1.0;

            // Intensity-weighted: accumulate child node intensity with √I
            let child_intensity = tree.nodes[edge.child_id].relative_intensity;
            features[OFF_LOSS_INT + bucket] += child_intensity.sqrt();
        }

        // ── Root loss features (precursor formula minus each descendant) ────
        for node in tree.nodes.iter().skip(1) {
            if let Some(root_loss) = root_formula.checked_sub(&node.formula) {
                let bucket = get_formula_bucket(
                    &root_loss,
                    &self.root_loss_vocab,
                    SEED_ROOT_LOSS,
                    DEDICATED_ROOT_LOSSES,
                    self.root_loss_buckets,
                );

                // Binary: mark presence
                features[OFF_ROOT_BIN + bucket] = 1.0;

                // Intensity-weighted with √I
                features[OFF_ROOT_INT + bucket] += node.relative_intensity.sqrt();
            }
        }

        // ── Tree-level scalar statistics ────────────────────────────────────
        let stats = compute_tree_statistics(tree);
        features[OFF_STATS..OFF_STATS + TREE_STATS].copy_from_slice(&stats);

        TreeFeatures { features }
    }
}

/// Maps a formula to a collision-free dedicated slot or open hash bucket.
#[inline]
fn get_formula_bucket(
    formula: &Formula,
    vocab: &std::collections::HashMap<Formula, usize>,
    seed: u32,
    dedicated_count: usize,
    total_buckets: usize,
) -> usize {
    if let Some(&slot) = vocab.get(formula) {
        slot
    } else {
        let open_buckets = total_buckets.saturating_sub(dedicated_count).max(1);
        dedicated_count + hash_formula(formula, seed, open_buckets)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Hashes a molecular formula into a fixed-size bucket index.
#[inline]
fn hash_formula(formula: &Formula, seed: u32, n_buckets: usize) -> usize {
    let mut bytes = [0u8; NUM_ELEMENTS * 2];
    for i in 0..NUM_ELEMENTS {
        let val = formula.counts[i] as u16;
        let le = val.to_le_bytes();
        bytes[i * 2] = le[0];
        bytes[i * 2 + 1] = le[1];
    }
    let hash = xxh32(&bytes, seed);
    (hash as usize) % n_buckets
}

/// Computes 27 scalar statistics from a fragmentation tree.
fn compute_tree_statistics(tree: &FragmentationTree) -> [f32; TREE_STATS] {
    let root = &tree.precursor_formula;
    let exact_mass = root.exact_mass();

    // Element ratios (safe division, 0.0 if carbon count is 0)
    let c = root.c().max(1) as f32;
    let h_over_c = root.h() as f32 / c;
    let n_over_c = root.n() as f32 / c;
    let o_over_c = root.o() as f32 / c;
    let p_over_c = root.p() as f32 / c;
    let s_over_c = root.s() as f32 / c;
    let f_over_c = root.f() as f32 / c;
    let cl_over_c = root.cl() as f32 / c;
    let br_over_c = root.br() as f32 / c;
    let i_over_c = root.i() as f32 / c;
    let si_over_c = root.si() as f32 / c;
    let b_over_c = root.b() as f32 / c;

    // Tree topology
    let num_nodes = tree.nodes.len();
    let num_edges = tree.edges.len();
    let tree_depth = compute_tree_depth(tree) as f32;

    // Score statistics over edges
    let (mean_edge, max_edge, std_edge) = if !tree.edges.is_empty() {
        let scores: Vec<f64> = tree.edges.iter().map(|e| e.edge_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let variance =
            scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / scores.len() as f64;
        (mean as f32, max as f32, variance.sqrt() as f32)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Score statistics over nodes (excluding root)
    let (mean_node, max_node) = if tree.nodes.len() > 1 {
        let scores: Vec<f64> = tree.nodes.iter().skip(1).map(|n| n.node_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean as f32, max as f32)
    } else {
        (0.0, 0.0)
    };

    // Fraction of common losses
    let common_loss_frac = if !tree.edges.is_empty() {
        tree.edges.iter().filter(|e| e.is_common_loss).count() as f32 / tree.edges.len() as f32
    } else {
        0.0
    };

    [
        // Mass features [0..3]
        (exact_mass / 1000.0) as f32,                       // Scaled exact mass
        (exact_mass - exact_mass.round()) as f32,           // Mass defect
        root.dbe() as f32 / 20.0,                           // Scaled DBE
        // Element ratios [3..14]
        h_over_c,
        n_over_c,
        o_over_c,
        p_over_c,
        s_over_c,
        f_over_c,
        cl_over_c,
        br_over_c,
        i_over_c,
        si_over_c,
        b_over_c,
        // Tree topology [14..17]
        tree_depth / 10.0,                                  // Scaled depth
        num_nodes as f32 / 50.0,                            // Scaled node count
        num_edges as f32 / 50.0,                            // Scaled edge count
        // Explained spectral content [17..19]
        tree.explained_intensity,                           // Fraction explained
        tree.explained_peaks as f32 / tree.total_peaks.max(1) as f32, // Peak ratio
        // Edge score statistics [19..22]
        mean_edge / 10.0,
        max_edge / 10.0,
        std_edge / 10.0,
        // Node score statistics [22..24]
        mean_node / 10.0,
        max_node / 10.0,
        // Common loss fraction [24]
        common_loss_frac,
        // Ionization mode indicators [25..27]
        1.0, // Pos mode default (1.0)
        0.0, // Neg mode default (0.0)
    ]
}

/// Computes the maximum depth of the fragmentation tree via BFS from root.
fn compute_tree_depth(tree: &FragmentationTree) -> usize {
    if tree.nodes.is_empty() {
        return 0;
    }

    let n = tree.nodes.len();
    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((0usize, 0usize));
    visited[0] = true;
    let mut max_depth = 0;

    while let Some((node_id, d)) = queue.pop_front() {
        max_depth = max_depth.max(d);
        for edge in &tree.edges {
            if edge.parent_id == node_id && !visited[edge.child_id] {
                visited[edge.child_id] = true;
                queue.push_back((edge.child_id, d + 1));
            }
        }
    }

    max_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sirius::tree::{FragmentationTree, TreeEdge, TreeNode};

    fn make_test_tree() -> FragmentationTree {
        let root_formula = Formula::chnops(8, 10, 4, 2, 0, 0);

        FragmentationTree {
            precursor_formula: root_formula,
            precursor_mz: 195.0882,
            nodes: vec![
                TreeNode {
                    id: 0,
                    peak_index: None,
                    formula: root_formula,
                    mz: 195.0882,
                    intensity: 1000.0,
                    relative_intensity: 1.0,
                    node_score: 5.0,
                },
                TreeNode {
                    id: 1,
                    peak_index: Some(0),
                    formula: Formula::chnops(7, 7, 3, 1, 0, 0),
                    mz: 138.0662,
                    intensity: 800.0,
                    relative_intensity: 0.8,
                    node_score: 3.5,
                },
                TreeNode {
                    id: 2,
                    peak_index: Some(1),
                    formula: Formula::chnops(5, 4, 4, 1, 0, 0),
                    mz: 110.0349,
                    intensity: 500.0,
                    relative_intensity: 0.5,
                    node_score: 2.0,
                },
            ],
            edges: vec![
                TreeEdge {
                    parent_id: 0,
                    child_id: 1,
                    loss_formula: Formula::chnops(1, 3, 1, 1, 0, 0),
                    loss_exact_mass: 57.0220,
                    mass_error_da: 0.001,
                    edge_score: 4.0,
                    is_common_loss: true,
                },
                TreeEdge {
                    parent_id: 1,
                    child_id: 2,
                    loss_formula: Formula::chnops(2, 3, 0, 0, 0, 0),
                    loss_exact_mass: 28.0313,
                    mass_error_da: 0.002,
                    edge_score: 1.5,
                    is_common_loss: false,
                },
            ],
            total_score: 16.0,
            explained_intensity: 0.85,
            explained_peaks: 2,
            total_peaks: 5,
        }
    }

    #[test]
    fn test_feature_extraction_dimension() {
        let tree = make_test_tree();
        let extractor = TreeFeatureExtractor::new();
        let features = extractor.extract(&tree);
        assert_eq!(features.features.len(), TOTAL_FEATURES);
        assert_eq!(features.dim(), TOTAL_FEATURES);
    }

    #[test]
    fn test_feature_extraction_nonzero() {
        let tree = make_test_tree();
        let extractor = TreeFeatureExtractor::new();
        let features = extractor.extract(&tree);

        let frag_bin = &features.features[OFF_FRAG_BIN..OFF_FRAG_BIN + FRAGMENT_BUCKETS];
        let loss_bin = &features.features[OFF_LOSS_BIN..OFF_LOSS_BIN + LOSS_BUCKETS];
        let root_bin = &features.features[OFF_ROOT_BIN..OFF_ROOT_BIN + ROOT_LOSS_BUCKETS];
        let stats = &features.features[OFF_STATS..OFF_STATS + TREE_STATS];

        assert!(frag_bin.iter().any(|&v| v > 0.0), "Fragment binary features should be nonzero");
        assert!(loss_bin.iter().any(|&v| v > 0.0), "Loss binary features should be nonzero");
        assert!(root_bin.iter().any(|&v| v > 0.0), "Root loss binary features should be nonzero");
        assert!(stats.iter().any(|&v| v > 0.0), "Tree statistics should be nonzero");
    }

    #[test]
    fn test_feature_extraction_deterministic() {
        let tree = make_test_tree();
        let extractor = TreeFeatureExtractor::new();
        let f1 = extractor.extract(&tree);
        let f2 = extractor.extract(&tree);
        assert_eq!(f1.features, f2.features, "Feature extraction must be deterministic");
    }
}

/// The graph view of a fragmentation tree used by the graph neural models:
/// `(node features, directed edges as (parent, child), edge features)`.
///
/// Kept here rather than inline in the exporter so that training data and
/// inference build the same tensor. Two copies of this drifting apart is the
/// quiet way a model stops meaning anything.
///
/// Node features (20): 12 element counts, m/z / 1000, sqrt relative intensity,
/// mass defect, DBE / 20, node score / 10, is-root, positive mode, negative mode.
/// Edge features (16): 12 element counts of the neutral loss, loss mass / 500,
/// mass error in mDa, is-common-loss, edge score / 10.
pub fn tree_graph(
    tree: &crate::sirius::tree::FragmentationTree,
    charge: i8,
) -> (Vec<Vec<f32>>, Vec<(usize, usize)>, Vec<Vec<f32>>) {
    let mut nodes = Vec::with_capacity(tree.nodes.len());
    for (idx, node) in tree.nodes.iter().enumerate() {
        let mut nf = Vec::with_capacity(20);
        for c in 0..12 {
            nf.push(node.formula.counts[c] as f32);
        }
        nf.push((node.mz / 1000.0) as f32);
        nf.push(node.relative_intensity.sqrt());
        nf.push((node.mz - node.mz.round()) as f32);
        nf.push((node.formula.dbe() / 20.0) as f32);
        nf.push((node.node_score / 10.0) as f32);
        nf.push(if idx == 0 { 1.0 } else { 0.0 });
        nf.push(if charge > 0 { 1.0 } else { 0.0 });
        nf.push(if charge < 0 { 1.0 } else { 0.0 });
        nodes.push(nf);
    }

    let mut edges = Vec::with_capacity(tree.edges.len());
    let mut edge_feats = Vec::with_capacity(tree.edges.len());
    for edge in &tree.edges {
        edges.push((edge.parent_id, edge.child_id));
        let mut ef = Vec::with_capacity(16);
        for c in 0..12 {
            ef.push(edge.loss_formula.counts[c] as f32);
        }
        ef.push((edge.loss_exact_mass / 500.0) as f32);
        ef.push((edge.mass_error_da * 1000.0) as f32);
        ef.push(if edge.is_common_loss { 1.0 } else { 0.0 });
        ef.push((edge.edge_score / 10.0) as f32);
        edge_feats.push(ef);
    }
    (nodes, edges, edge_feats)
}
