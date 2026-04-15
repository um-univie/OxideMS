type Spectrum = Vec<(u16, f32)>;

fn to_hashed_spectrum(features: &[f32]) -> Spectrum {
    let mut spectrum = Vec::with_capacity(50);
    for (index, mass) in features.iter().enumerate() {
        if *mass > 0.0 {
            spectrum.push((index as u16, *mass))
        }
    }
    spectrum
}

#[inline]
fn sparse_cosine(a: &[(u16, f32)], b: &[(u16, f32)]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut score = 0.0;

    // Standard two-pointer intersection for sorted arrays
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            score += a[i].1 * b[j].1;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    score
}
