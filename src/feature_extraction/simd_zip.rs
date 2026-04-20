use std::arch::x86_64::*;

/// Finds matches between two sorted m/z arrays within a given ppm tolerance.
/// Returns a vector of (index_a, index_b) tuples representing the matches.
pub fn find_matches_simd(array_a: &[f64], array_b: &[f64], ppm_tol: f64) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { find_matches_avx2(array_a, array_b, ppm_tol) };
        }
    }

    // Fallback if AVX2 is not supported (or not running on x86_64)
    find_matches_scalar(array_a, array_b, ppm_tol)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_matches_avx2(array_a: &[f64], array_b: &[f64], ppm_tol: f64) -> Vec<(usize, usize)> {
    let mut matches = Vec::with_capacity(array_a.len()); // Pre-allocate for speed
    let b_len = array_b.len();
    if b_len == 0 {
        return matches;
    }

    let mut b_idx = 0;

    // Label the outer loop so we can skip directly to the next 'a_val'
    'a_loop: for (a_idx, &a_val) in array_a.iter().enumerate() {
        let margin = a_val * (ppm_tol / 1_000_000.0);
        let lower_bound = a_val - margin;
        let upper_bound = a_val + margin;

        // 1. Macro Sweep-Line: Fast-forward chunk-by-chunk using raw bounds
        while b_idx + 3 < b_len && *array_b.get_unchecked(b_idx + 3) < lower_bound {
            b_idx += 4;
        }
        // Handle any remaining scalar gap
        while b_idx < b_len && *array_b.get_unchecked(b_idx) < lower_bound {
            b_idx += 1;
        }

        let mut current_b = b_idx;

        // 2. Pre-computation & Broadcast
        let v_lower = _mm256_set1_pd(lower_bound);
        let v_upper = _mm256_set1_pd(upper_bound);

        // 3. Vectorized Inner Loop
        while current_b + 3 < b_len {
            // Early exit: If the first element is too big, skip the rest of 'b' for this 'a'
            if *array_b.get_unchecked(current_b) > upper_bound {
                continue 'a_loop; // Fixed: bypasses the tail loop
            }

            let v_b = _mm256_loadu_pd(array_b.as_ptr().add(current_b));

            let cmp_ge = _mm256_cmp_pd(v_b, v_lower, 13);
            let cmp_le = _mm256_cmp_pd(v_b, v_upper, 18);
            let mask_and = _mm256_and_pd(cmp_ge, cmp_le);

            let bitmask = _mm256_movemask_pd(mask_and);

            // 4. Branchless Extraction
            if bitmask > 0 {
                let mut bits = bitmask;
                while bits > 0 {
                    let i = bits.trailing_zeros();
                    matches.push((a_idx, current_b + i as usize));
                    bits &= bits - 1;
                }
            }

            // Check if any element in the chunk exceeded the upper bound
            let cmp_gt = _mm256_cmp_pd(v_b, v_upper, 14);
            if _mm256_movemask_pd(cmp_gt) > 0 {
                continue 'a_loop; // Fixed: bypasses the tail loop and prevents duplicate evaluation
            }

            current_b += 4;
        }

        // Handle the tail (the final < 4 elements of Array B)
        while current_b < b_len {
            let b_val = *array_b.get_unchecked(current_b);
            if b_val > upper_bound {
                break; // 'break' is fine here as it naturally goes to the next 'a_val'
            }
            if b_val >= lower_bound {
                matches.push((a_idx, current_b));
            }
            current_b += 1;
        }
    }

    matches
}

/// A standard scalar sweep-line used as a fallback if AVX2 is not available
fn find_matches_scalar(array_a: &[f64], array_b: &[f64], ppm_tol: f64) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut b_idx = 0;
    let b_len = array_b.len();

    for (a_idx, &a_val) in array_a.iter().enumerate() {
        let margin = a_val * (ppm_tol / 1_000_000.0);
        let lower = a_val - margin;
        let upper = a_val + margin;

        while b_idx < b_len && array_b[b_idx] < lower {
            b_idx += 1;
        }

        let mut current_b = b_idx;
        while current_b < b_len && array_b[current_b] <= upper {
            matches.push((a_idx, current_b));
            current_b += 1;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_matches() {
        let a = vec![100.0, 200.0, 300.0];
        let b = vec![50.0, 100.0, 150.0, 200.0, 300.0, 350.0];

        let matches = find_matches_simd(&a, &b, 10.0);

        // Expected matches: (A:100->B:100), (A:200->B:200), (A:300->B:300)
        assert_eq!(matches, vec![(0, 1), (1, 3), (2, 4)]);
    }

    #[test]
    fn test_ppm_tolerance_boundaries() {
        let a = vec![1000.0];
        // 10 ppm of 1000 = 0.01
        // Bounds: 999.99 to 1000.01
        let b = vec![
            999.989,  // Too low
            999.991,  // Just inside lower bound
            1000.0,   // Exact
            1000.009, // Just inside upper bound
            1000.011, // Too high
        ];

        let matches = find_matches_simd(&a, &b, 10.0);

        // Should only catch indices 1, 2, and 3
        assert_eq!(matches, vec![(0, 1), (0, 2), (0, 3)]);
    }

    #[test]
    fn test_empty_arrays() {
        let a = vec![100.0, 200.0];
        let b: Vec<f64> = vec![];
        assert_eq!(find_matches_simd(&a, &b, 10.0).len(), 0);

        let a_empty: Vec<f64> = vec![];
        let b_full = vec![100.0, 200.0];
        assert_eq!(find_matches_simd(&a_empty, &b_full, 10.0).len(), 0);
    }

    #[test]
    fn test_large_chunk_skipping() {
        // Ensures the macro-sweep line logic correctly fast-forwards
        let a = vec![500.0];
        let mut b = vec![0.0; 100]; // 100 zeroes
        b.push(500.0); // Index 100
        b.push(1000.0); // Index 101

        let matches = find_matches_simd(&a, &b, 5.0);
        assert_eq!(matches, vec![(0, 100)]);
    }

    #[test]
    fn test_isobaric_multiple_matches() {
        // Simulates two closely eluting/isobaric compounds in Array B
        let a = vec![400.0, 400.001];
        let b = vec![399.999, 400.0, 400.001, 400.002];

        // 5 ppm of 400 = 0.002.
        let matches = find_matches_simd(&a, &b, 5.0);

        // A[0] (400.0) should match B[0], B[1], B[2], B[3]
        // A[1] (400.001) should match B[0], B[1], B[2], B[3]
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(0, 1)));
        assert!(matches.contains(&(1, 2)));
        assert!(matches.contains(&(1, 3)));
    }
}
