/// 1. The Standard Branching Approach (For baseline comparison)
pub fn branching_cosine_sim(a_bins: &[u32], a_ints: &[f32], b_bins: &[u32], b_ints: &[f32]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    let len_a = a_bins.len();
    let len_b = b_bins.len();

    while i < len_a && j < len_b {
        let a_bin = a_bins[i];
        let b_bin = b_bins[j];

        if a_bin == b_bin {
            score += a_ints[i] * b_ints[j];
            i += 1;
            j += 1;
        } else if a_bin < b_bin {
            i += 1;
        } else {
            j += 1;
        }
    }
    score
}

#[inline]
fn dot_sorted(a: &[(usize, f32)], b: &[(usize, f32)]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                sum += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    sum
}
