use std::fmt::Write;
pub const WORDSIZE: usize = usize::BITS as usize;

pub trait BitSetOpsMSB0 {
    /// Returns true if the bit at `index` is set.
    fn is_bit_set(&self, index: usize) -> bool;

    /// Sets the bit at `index` to 1.
    fn set_bit(&mut self, index: usize);

    /// Sets the bit at `index` to 1, without bounds checking
    unsafe fn set_bit_unchecked(&mut self, index: usize);

    /// Sets the bit at `index` to 0.
    fn unset_bit(&mut self, index: usize);

    /// Sets the bit at `index` to 0, without bounds checking
    unsafe fn unset_bit_unchecked(&mut self, index: usize);

    fn population_count(&self) -> u32;

    /// Clears all bits in the bitset.
    fn clear(&mut self);

    /// Clears all bits in the bitset up to the target word
    fn clear_words(&mut self, limit: usize);

    /// Finds the index of the first set bit at or after `start_index`.
    fn next_set_bit(&self, start_index: usize) -> Option<usize>;

    ///
    unsafe fn next_set_bit_unchecked(&self, start_index: usize) -> Option<usize>;
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

    /// # Safety
    /// `index` must be strictly less than `self.len() * usize::BITS`
    unsafe fn set_bit_unchecked(&mut self, index: usize) {
        let (word_idx, mask) = split_index(index);
        *self.get_unchecked_mut(word_idx) |= mask;
    }

    fn unset_bit(&mut self, index: usize) {
        let (word_idx, mask) = split_index(index);
        if let Some(word) = self.get_mut(word_idx) {
            *word &= !mask;
        }
    }

    /// # Safety
    /// `index` must be strictly less than `self.len() * usize::BITS`
    unsafe fn unset_bit_unchecked(&mut self, index: usize) {
        let (word_idx, mask) = split_index(index);
        *self.get_unchecked_mut(word_idx) &= !mask;
    }

    fn clear(&mut self) {
        self.fill(0);
    }

    fn clear_words(&mut self, word_limit: usize) {
        let limit = word_limit.min(self.len());
        self[..limit].fill(0);
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

    /// Finds the first set bit starting from `start_index` (scanning Left-to-Right).
    ///
    /// # Safety
    /// `start_index` must be strictly less than `self.len() * usize::BITS as usize`.
    /// Calling this with an out-of-bounds index will result in Undefined Behavior.
    unsafe fn next_set_bit_unchecked(&self, start_index: usize) -> Option<usize> {
        // 1. Bitwise math instead of division/modulo (Compile-time constants)
        const SHIFT: usize = usize::BITS.trailing_zeros() as usize;
        const MASK: usize = usize::BITS as usize - 1;

        let word_idx = start_index >> SHIFT;
        let bit_offset = start_index & MASK;

        // 2. Unchecked access for the first partial word
        let word = *self.get_unchecked(word_idx);
        let mask = (!0usize) >> bit_offset;
        let masked_word = word & mask;

        if masked_word != 0 {
            return Some((word_idx << SHIFT) + masked_word.leading_zeros() as usize);
        }

        // 3. Unchecked slice creation for the remaining words.
        // Since `start_index` is guaranteed valid by the safety contract,
        // `word_idx + 1` is guaranteed to be <= `self.len()`.
        let remaining = self.get_unchecked(word_idx + 1..self.len());

        for (i, &word) in remaining.iter().enumerate() {
            if word != 0 {
                let found_word_idx = word_idx + 1 + i;
                return Some((found_word_idx << SHIFT) + word.leading_zeros() as usize);
            }
        }

        None
    }

    fn print_bits(&self) {
        println!("{}", self.to_bit_string());
    }

    /// This function counts the number of set bits (1s) in the bit set (pun intended)
    fn population_count(&self) -> u32 {
        self.iter().map(|w| w.count_ones()).sum()
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
    const SHIFT: usize = usize::BITS.trailing_zeros() as usize;
    const MASK: usize = usize::BITS as usize - 1;
    const HIGH_BIT: usize = 1 << (usize::BITS as usize - 1);

    let word_idx = index >> SHIFT;
    let bit_idx = index & MASK;

    (word_idx, HIGH_BIT >> bit_idx)
}
