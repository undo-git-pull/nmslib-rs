//! A fixed-length set of labels, one bit each, for filtered search.
//!
//! Bit `id % 64` of word `id / 64` says whether label `id` may be returned.
//! Testing a bit is one load and a shift, cheap enough for the inner search
//! loop.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// `len` labels, none set.
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// Label `i` is set where `bits[i]` is true.
    pub fn from_bools(bits: &[bool]) -> Self {
        let mut words = vec![0u64; bits.len().div_ceil(64)];
        for (word, chunk) in words.iter_mut().zip(bits.chunks(64)) {
            for (j, &b) in chunk.iter().enumerate() {
                *word |= (b as u64) << j;
            }
        }
        Self {
            words,
            len: bits.len(),
        }
    }

    /// The number of labels covered, set or not.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set(&mut self, id: u32) {
        assert!((id as usize) < self.len, "label {id} out of range");
        self.words[id as usize / 64] |= 1 << (id % 64);
    }

    /// The number of labels set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.words[id as usize / 64] >> (id % 64) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_test() {
        let bools: Vec<bool> = (0..130).map(|i| i % 3 == 0 || i == 129).collect();
        let b = Bitset::from_bools(&bools);
        assert_eq!(b.len(), 130);
        for (i, &want) in bools.iter().enumerate() {
            assert_eq!(b.contains(i as u32), want, "bit {i}");
        }
        assert_eq!(b.count(), bools.iter().filter(|&&x| x).count());
        let mut c = Bitset::new(130);
        for (i, &x) in bools.iter().enumerate() {
            if x {
                c.set(i as u32);
            }
        }
        assert_eq!(b, c);
    }
}
