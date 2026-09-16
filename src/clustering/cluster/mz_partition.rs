//! Shared m/z + charge boundary-scan logic.
//!
//! [`ClusterEngine::split_on_mz_and_charge`](super::engine::ClusterEngine::split_on_mz_and_charge)
//! and the low-memory streaming ingest path (`io::mgf::MgfParser::parse_file_low_mem`
//! plus [`crate::clustering::cluster::experiment::SpectralExperiment::finalize_low_mem_partitions`])
//! both need to decide "does this spectrum belong in the same m/z/charge group
//! as its sorted neighbor", and both must agree exactly, or a spectrum could
//! end up on the wrong side of a partition boundary. This module is the one
//! place that rule lives.

use crate::workflow::MassTolerance;

/// Given a slice already sorted by `(charge, precursor_mz)`, returns the
/// boundary indices of the m/z/charge groups: consecutive elements are in
/// the same group iff they share a charge and their neutral-mass gap is
/// within `tolerance` (evaluated at the later element's neutral mass, same
/// as the historical behavior of `split_on_mz_and_charge`).
///
/// The returned `Vec` always starts with `0` and ends with `sorted.len()`,
/// so consecutive pairs `boundaries[i]..boundaries[i + 1]` are the groups.
pub(crate) fn mz_charge_boundaries<T>(
    sorted: &[T],
    charge_of: impl Fn(&T) -> i8,
    neutral_mass_of: impl Fn(&T) -> f64,
    tolerance: MassTolerance,
) -> Vec<usize> {
    let n = sorted.len();
    let mut boundaries: Vec<usize> = vec![0];
    for i in 1..n {
        let a = &sorted[i - 1];
        let b = &sorted[i];
        let same_group = if charge_of(a) != charge_of(b) {
            false
        } else {
            let m_a = neutral_mass_of(a);
            let m_b = neutral_mass_of(b);
            let gap = (m_b - m_a).abs();
            match tolerance {
                MassTolerance::Da(tol) => gap <= tol,
                MassTolerance::Ppm(tol) => gap <= m_b * (tol / 1_000_000.0),
            }
        };
        if !same_group {
            boundaries.push(i);
        }
    }
    if n > 0 {
        boundaries.push(n);
    }
    boundaries
}

/// Given a slice already sorted ascending by neutral mass, returns for each
/// element the exclusive index of the first *later* element that is out of
/// `tolerance` of it -- i.e. element `i`'s candidate partners are exactly
/// `i + 1..ends[i]`.
///
/// The tolerance is evaluated at the heavier partner's mass, matching
/// [`mz_charge_boundaries`] and
/// [`crate::clustering::k_cluster_ivf::windowed_mass_edges`], so all three
/// agree on which pairs are admissible.
///
/// Runs in O(n): the window end only ever moves forward, because the gap to
/// a fixed later element shrinks as `i` advances while the tolerance at that
/// element is unchanged.
pub(crate) fn mass_window_ends<T>(
    sorted: &[T],
    neutral_mass_of: impl Fn(&T) -> f64,
    tolerance: MassTolerance,
) -> Vec<usize> {
    let n = sorted.len();
    let mut ends = vec![0usize; n];
    let mut end = 0usize;
    for (i, slot) in ends.iter_mut().enumerate() {
        if end < i + 1 {
            end = i + 1;
        }
        let m_i = neutral_mass_of(&sorted[i]);
        while end < n {
            let m_j = neutral_mass_of(&sorted[end]);
            let max_gap = match tolerance {
                MassTolerance::Da(tol) => tol,
                MassTolerance::Ppm(tol) => m_j * (tol / 1_000_000.0),
            };
            if m_j - m_i <= max_gap {
                end += 1;
            } else {
                break;
            }
        }
        *slot = end;
    }
    ends
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ends_of(masses: &[f64], tolerance: MassTolerance) -> Vec<usize> {
        mass_window_ends(masses, |m: &f64| *m, tolerance)
    }

    /// The window must contain exactly the later elements an O(n^2) scan
    /// would admit, under the same "tolerance at the heavier partner" rule.
    fn brute_force_ends(masses: &[f64], tolerance: MassTolerance) -> Vec<usize> {
        (0..masses.len())
            .map(|i| {
                let mut end = i + 1;
                for (j, &m_j) in masses.iter().enumerate().skip(i + 1) {
                    let max_gap = match tolerance {
                        MassTolerance::Da(tol) => tol,
                        MassTolerance::Ppm(tol) => m_j * (tol / 1_000_000.0),
                    };
                    if m_j - masses[i] <= max_gap {
                        end = j + 1;
                    } else {
                        break;
                    }
                }
                end
            })
            .collect()
    }

    #[test]
    fn window_ends_match_brute_force() {
        let cases: [(Vec<f64>, MassTolerance); 4] = [
            (vec![], MassTolerance::Da(0.1)),
            (vec![1000.0], MassTolerance::Da(0.1)),
            (
                vec![1000.0, 1000.05, 1000.2, 1000.25, 1400.0],
                MassTolerance::Da(0.1),
            ),
            (
                (0..200).map(|i| 500.0 + i as f64 * 0.003).collect(),
                MassTolerance::Ppm(20.0),
            ),
        ];
        for (masses, tolerance) in cases {
            assert_eq!(
                ends_of(&masses, tolerance),
                brute_force_ends(&masses, tolerance),
                "masses {masses:?}"
            );
        }
    }

    /// Every element is at least its own successor's neighbour candidate
    /// boundary, and the ends never move backwards.
    #[test]
    fn window_ends_are_monotone_and_at_least_one_past_self() {
        let masses: Vec<f64> = (0..500).map(|i| 300.0 + (i % 7) as f64 * 0.001).collect();
        let ends = ends_of(&masses, MassTolerance::Ppm(20.0));
        for (i, &e) in ends.iter().enumerate() {
            assert!(e >= i + 1, "end {e} behind index {i}");
            if i > 0 {
                assert!(e >= ends[i - 1], "ends moved backwards at {i}");
            }
        }
    }
}
