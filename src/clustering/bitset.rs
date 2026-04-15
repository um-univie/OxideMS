use std::fmt::Write;
pub const WORDSIZE: usize = usize::BITS as usize;

pub trait BitSetOpsMSB0 {
    /// Returns true if the bit at `index` is set.
    fn is_bit_set(&self, index: usize) -> bool;

    /// Sets the bit at `index` to 1.
    fn set_bit(&mut self, index: usize);

    /// Sets the bit at `index` to 0.
    fn unset_bit(&mut self, index: usize);

    fn population_count(&self) -> u32;

    /// Clears all bits in the bitset.
    fn clear(&mut self);

    /// Clears all bits in the bitset up to the target word
    fn clear_words(&mut self, limit: usize);

    /// Finds the index of the first set bit at or after `start_index`.
    fn next_set_bit(&self, start_index: usize) -> Option<usize>;

    /// Performs an in-place intersection (bitwise AND) with another bitset.
    /// `self = self & other`
    fn intersect_inplace(&mut self, other: &Self);

    /// Prints the bitset to stdout as a continuous string of '1's and '0's.
    /// Includes a visual separator (_) between words (every 64 bits on 64-bit systems).
    fn print_bits(&self);

    fn to_bit_string(&self) -> String;
}

// Implementation for any slice of usizes
impl BitSetOpsMSB0 for [usize] {
    fn is_bit_set(&self, index: usize) -> bool {
        let (word_idx, mask) = split_index(index);
        // Use `get` to handle out-of-bounds gracefully
        match self.get(word_idx) {
            Some(word) => (word & mask) != 0,
            None => false,
        }
    }

    fn set_bit(&mut self, index: usize) {
        let (word_idx, mask) = split_index(index);
        if let Some(word) = self.get_mut(word_idx) {
            *word |= mask;
        }
    }

    fn unset_bit(&mut self, index: usize) {
        let (word_idx, mask) = split_index(index);
        if let Some(word) = self.get_mut(word_idx) {
            *word &= !mask;
        }
    }

    fn clear(&mut self) {
        self.fill(0);
    }

    fn clear_words(&mut self, word_limit: usize) {
        if word_limit <= self.len() {
            self[..word_limit].fill(0);
        } else {
            self.fill(0);
        }
    }

    /// Finds the first set bit starting from `start_index` (scanning Left-to-Right)
    fn next_set_bit(&self, start_index: usize) -> Option<usize> {
        if start_index >= self.len() * usize::BITS as usize {
            return None;
        }

        let word_idx = start_index / (usize::BITS as usize);
        let bit_offset = start_index % (usize::BITS as usize);

        // Check partial first word
        if let Some(&word) = self.get(word_idx) {
            // Mask to keep only bits from bit_offset downwards. (!0 is all ones)
            // This effectively ignores the preceeding bits
            let mask = (!0usize) >> bit_offset;
            let masked_word = word & mask;

            if masked_word != 0 {
                // Leading zeroes returns the index of the first set bit
                return Some(
                    word_idx * usize::BITS as usize + masked_word.leading_zeros() as usize,
                );
            }
        }

        // Scan remaining words
        for (i, &word) in self.get(word_idx + 1..)?.iter().enumerate() {
            if word != 0 {
                let found_word_idx = word_idx + 1 + i;
                return Some(found_word_idx * usize::BITS as usize + word.leading_zeros() as usize);
            }
        }

        None
    }

    fn print_bits(&self) {
        println!("{}", self.to_bit_string());
    }

    /// This function counts the number of set bits (1s) in the bit set (pun intended)
    fn population_count(&self) -> u32 {
        self.iter().fold(0, |sum, subset| sum + subset.count_ones())
    }

    fn to_bit_string(&self) -> String {
        let cap = self.len() * (WORDSIZE + 1); // Words + separators
        let mut result = String::with_capacity(cap);

        for (i, &word) in self.iter().enumerate() {
            if i > 0 {
                result.push('_');
            }
            // Format as binary, padded with zeros to full width
            write!(&mut result, "{:0width$b}", word, width = WORDSIZE).unwrap();
        }
        result
    }

    fn intersect_inplace(&mut self, other: &[usize]) {
        let common_len = std::cmp::min(self.len(), other.len());

        // 1. Process common words
        for (self_word, &other_word) in self[..common_len].iter_mut().zip(&other[..common_len]) {
            *self_word &= other_word;
        }

        // // 2. Clear excess words if self is larger
        if self.len() > common_len {
            self[common_len..].fill(0);
        }
    }
}

/// Converts visual index (0 = Left/MSB) to CPU mask
#[inline(always)]
fn split_index(index: usize) -> (usize, usize) {
    let word_idx = index / (usize::BITS as usize);
    let bit_idx = index % (usize::BITS as usize);

    // MSB-0 Logic:
    // If we want index 0 (Leftmost), we want the High Bit (1 << 63).
    // If we want index 63 (Rightmost), we want the Low Bit (1 << 0).
    // Shift = (63 - index)
    let shift = (usize::BITS as usize - 1) - bit_idx;

    (word_idx, 1 << shift)
}
