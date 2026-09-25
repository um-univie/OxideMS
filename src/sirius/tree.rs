use super::formula::Formula;
use std::fmt;

/// A node in a fragmentation tree representing an annotated peak (or precursor root).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    pub id: usize,                  // Unique node ID in the tree (0 is root)
    pub peak_index: Option<usize>,  // None for root precursor, Some(i) for fragment peaks
    pub formula: Formula,
    pub mz: f64,
    pub intensity: f32,
    pub relative_intensity: f32,
    pub node_score: f64,
}

/// A directed edge in a fragmentation tree representing a neutral loss transition.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeEdge {
    pub parent_id: usize,
    pub child_id: usize,
    pub loss_formula: Formula,
    pub loss_exact_mass: f64,
    pub mass_error_da: f64,
    pub edge_score: f64,
    pub is_common_loss: bool,
}

/// A Maximum A Posteriori (MAP) Fragmentation Tree explaining an MS/MS spectrum
/// under a candidate precursor molecular formula.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentationTree {
    pub precursor_formula: Formula,
    pub precursor_mz: f64,
    pub nodes: Vec<TreeNode>,
    pub edges: Vec<TreeEdge>,
    /// Sum of the node and edge weights in the tree.
    pub total_score: f64,
    pub explained_intensity: f32,   // Fraction of total spectral intensity explained by the tree
    pub explained_peaks: usize,     // Number of fragment peaks explained
    pub total_peaks: usize,         // Total number of peaks in input spectrum
}

impl FragmentationTree {
    /// Returns the root node of the tree (precursor)
    pub fn root(&self) -> &TreeNode {
        &self.nodes[0]
    }

    /// Returns the children of a given node
    pub fn children_of(&self, node_id: usize) -> Vec<(&TreeNode, &TreeEdge)> {
        let mut children = Vec::new();
        for edge in &self.edges {
            if edge.parent_id == node_id {
                let child_node = &self.nodes[edge.child_id];
                children.push((child_node, edge));
            }
        }
        children
    }

    /// Formats the tree in a human-readable ASCII hierarchy
    pub fn format_tree(&self) -> String {
        let mut out = String::new();
        self.format_node_recursive(0, "", true, &mut out);
        out
    }

    fn format_node_recursive(&self, node_id: usize, prefix: &str, is_tail: bool, out: &mut String) {
        let node = &self.nodes[node_id];
        let branch = if node_id == 0 {
            ""
        } else if is_tail {
            "└── "
        } else {
            "├── "
        };

        if node_id == 0 {
            out.push_str(&format!(
                "[Root] {} (m/z {:.4}, Score: {:.2})\n",
                node.formula, node.mz, node.node_score
            ));
        } else {
            // Find edge connecting parent to this node
            let edge = self.edges.iter().find(|e| e.child_id == node_id);
            let loss_info = if let Some(e) = edge {
                let common_tag = if e.is_common_loss { " [Common Loss]" } else { "" };
                format!(" -{} ({:.4} Da){}", e.loss_formula, e.loss_exact_mass, common_tag)
            } else {
                String::new()
            };

            out.push_str(&format!(
                "{}{}{} -> {} (m/z {:.4}, Int: {:.1}%, Score: {:.2})\n",
                prefix, branch, loss_info, node.formula, node.mz, node.relative_intensity * 100.0, node.node_score
            ));
        }

        let children = self.children_of(node_id);
        let count = children.len();
        for (i, (child, _)) in children.into_iter().enumerate() {
            let next_is_tail = i == count - 1;
            let next_prefix = if node_id == 0 {
                ""
            } else if is_tail {
                &format!("{}    ", prefix)
            } else {
                &format!("{}│   ", prefix)
            };
            self.format_node_recursive(child.id, next_prefix, next_is_tail, out);
        }
    }
}

impl fmt::Display for FragmentationTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FragmentationTree(Precursor: {}, Score: {:.2}, Explained: {}/{} peaks [{:.1}% int])\n{}",
            self.precursor_formula,
            self.total_score,
            self.explained_peaks,
            self.total_peaks,
            self.explained_intensity * 100.0,
            self.format_tree()
        )
    }
}
