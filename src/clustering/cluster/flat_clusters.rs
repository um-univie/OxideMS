use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::workflow::MassTolerance;

#[derive(PartialEq, PartialOrd)]
pub(super) struct OrderedF64(pub f64);

impl Eq for OrderedF64 {}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub(super) struct Node {
    pub min: f64,
    pub max: f64,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub active: bool,
    pub start_idx: usize,
    pub end_idx: usize,
}

pub struct FlatClusters {
    pub data: Vec<usize>,
    pub offsets: Vec<usize>,
    /// Cosine of each member (aligned with `data`) to its cluster's
    /// representative, which is listed first in its cluster. Empty unless
    /// the engine picked representatives
    /// ([`ClusterEngine::representative_floor`](super::engine::ClusterEngine::representative_floor)).
    pub rep_cosine: Vec<f32>,
}

impl FlatClusters {
    fn get_cluster(&self, cluster_id: usize) -> &[usize] {
        if cluster_id >= self.offsets.len() - 1 {
            return &[];
        }
        let start = self.offsets[cluster_id];
        let end = self.offsets[cluster_id + 1];
        &self.data[start..end]
    }

    pub fn cluster_size(&self, cluster_id: usize) -> usize {
        if cluster_id >= self.offsets.len() - 1 {
            return 0;
        }
        self.offsets[cluster_id + 1] - self.offsets[cluster_id]
    }

    pub fn num_clusters(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn active_clusters_count(&self) -> usize {
        self.offsets.windows(2).filter(|w| w[1] > w[0]).count()
    }

    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    pub fn max_cluster_size(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    pub fn min_cluster_size(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .min()
            .unwrap_or(0)
    }

    pub fn avg_cluster_size(&self) -> f64 {
        let active = self.active_clusters_count();
        if active == 0 {
            return 0.0;
        }
        self.total_elements() as f64 / active as f64
    }

    pub fn iter(&self) -> ClusterIter<'_> {
        ClusterIter {
            clusters: self,
            current_id: 0,
        }
    }

    pub fn get_cluster_spans(&self, mz_values: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );
        let mut spans = vec![];

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_mzs = &mz_values[start..end];
            // Empty clusters are real (see active_clusters_count) -- a span
            // of 0.0 keeps spans[i] aligned with cluster i rather than
            // panicking or shifting indices by skipping the push.
            let span = match (cluster_mzs.first(), cluster_mzs.last()) {
                (Some(&first), Some(&last)) => last - first,
                _ => 0.0,
            };
            spans.push(span)
        }
        spans
    }

    pub fn get_cluster_spans_ppm(&self, mz_values: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );
        let mut spans = vec![];

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_mzs = &mz_values[start..end];
            // Same empty-cluster case as get_cluster_spans above.
            let span_ppm = match (
                cluster_mzs.iter().max_by(|a, b| a.total_cmp(b)),
                cluster_mzs.iter().min_by(|a, b| a.total_cmp(b)),
            ) {
                (Some(max_mass), Some(min_mass)) => {
                    1_000_000.0 * ((max_mass - min_mass) / max_mass)
                }
                _ => 0.0,
            };
            spans.push(span_ppm)
        }
        spans
    }

    pub fn refine_mz_clusters(&self, mz_values: &[f64], tolerance: MassTolerance) -> FlatClusters {
        debug_assert_eq!(
            self.data.len(),
            mz_values.len(),
            "mz_values must be perfectly parallel to self.data"
        );

        let mut new_data = Vec::with_capacity(self.data.len());
        let mut new_offsets = Vec::with_capacity(self.offsets.len());
        new_offsets.push(0);

        for i in 0..self.num_clusters() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            let cluster_ids = &self.data[start..end];
            let cluster_mzs = &mz_values[start..end];

            if cluster_ids.len() <= 1 {
                continue;
            }

            Self::refine_single_cluster(
                cluster_ids,
                cluster_mzs,
                tolerance,
                &mut new_data,
                &mut new_offsets,
            );
        }

        FlatClusters {
            data: new_data,
            offsets: new_offsets,
            rep_cosine: Vec::new(),
        }
    }

    pub(crate) fn refine_single_cluster(
        cluster_ids: &[usize],
        cluster_mzs: &[f64],
        tolerance: MassTolerance,
        new_data: &mut Vec<usize>,
        new_offsets: &mut Vec<usize>,
    ) {
        let n = cluster_ids.len();

        let mut local_indices: Vec<usize> = (0..n).collect();
        local_indices.sort_unstable_by(|&a, &b| {
            cluster_mzs[a]
                .partial_cmp(&cluster_mzs[b])
                .unwrap_or(Ordering::Equal)
        });

        let mut nodes: Vec<Node> = Vec::with_capacity(n * 2);
        for (i, &local_idx) in local_indices.iter().enumerate() {
            let val = cluster_mzs[local_idx];
            nodes.push(Node {
                min: val,
                max: val,
                left: if i > 0 { Some(i - 1) } else { None },
                right: if i < n - 1 { Some(i + 1) } else { None },
                active: true,
                start_idx: i,
                end_idx: i + 1,
            });
        }

        let mut heap = BinaryHeap::new();
        for i in 0..n - 1 {
            let dist = nodes[i + 1].max - nodes[i].min;
            heap.push(Reverse((OrderedF64(dist), i, i + 1)));
        }

        while let Some(Reverse((OrderedF64(dist), left_id, right_id))) = heap.pop() {
            if !nodes[left_id].active || !nodes[right_id].active {
                continue;
            }

            if dist > tolerance.tolerance_at(nodes[right_id].max) {
                continue;
            }

            nodes[left_id].active = false;
            nodes[right_id].active = false;

            let next_id = nodes.len();
            let new_node = Node {
                min: nodes[left_id].min,
                max: nodes[right_id].max,
                left: nodes[left_id].left,
                right: nodes[right_id].right,
                active: true,
                start_idx: nodes[left_id].start_idx,
                end_idx: nodes[right_id].end_idx,
            };

            if let Some(left_neighbor_id) = new_node.left {
                nodes[left_neighbor_id].right = Some(next_id);
                let new_dist_left = new_node.max - nodes[left_neighbor_id].min;
                heap.push(Reverse((
                    OrderedF64(new_dist_left),
                    left_neighbor_id,
                    next_id,
                )));
            }

            if let Some(right_neighbor_id) = new_node.right {
                nodes[right_neighbor_id].left = Some(next_id);
                let new_dist_right = nodes[right_neighbor_id].max - new_node.min;
                heap.push(Reverse((
                    OrderedF64(new_dist_right),
                    next_id,
                    right_neighbor_id,
                )));
            }

            nodes.push(new_node);
        }

        for node in nodes.iter().filter(|n| n.active) {
            if node.end_idx - node.start_idx <= 1 {
                continue;
            }

            for i in node.start_idx..node.end_idx {
                let original_idx = local_indices[i];
                new_data.push(cluster_ids[original_idx]);
            }
            new_offsets.push(new_data.len());
        }
    }
}

pub struct ClusterIter<'a> {
    clusters: &'a FlatClusters,
    current_id: usize,
}

impl<'a> Iterator for ClusterIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_id < self.clusters.num_clusters() {
            let members = self.clusters.get_cluster(self.current_id);
            self.current_id += 1;

            if !members.is_empty() {
                return Some(members);
            }
        }
        None
    }
}
