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
