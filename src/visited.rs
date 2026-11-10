//! Generation-stamped visited sets.
//!
//! Marking a node stores the current generation in its slot, and starting a
//! new search bumps the generation, so clearing costs nothing per call. The
//! array is only rewritten when the generation counter wraps.

pub struct VisitedSet {
    marks: Vec<u16>,
    generation: u16,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        Self { marks: vec![0; capacity], generation: 0 }
    }

    /// Grows to hold `capacity` nodes. New slots start unvisited.
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.marks.len() < capacity {
            self.marks.resize(capacity, 0);
        }
    }

    /// Starts a new search: every node becomes unvisited.
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    /// Marks `id` and returns whether it was unvisited before.
    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        let slot = &mut self.marks[id as usize];
        let fresh = *slot != self.generation;
        *slot = self.generation;
        fresh
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.marks[id as usize] == self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_clear() {
        let mut v = VisitedSet::new(4);
        v.clear();
        assert!(v.insert(2));
        assert!(!v.insert(2));
        assert!(v.contains(2));
        assert!(!v.contains(1));
        v.clear();
        assert!(!v.contains(2));
        assert!(v.insert(2));
    }

    #[test]
    fn survives_generation_wrap() {
        let mut v = VisitedSet::new(3);
        v.clear();
        v.insert(0);
        for _ in 0..u16::MAX as usize + 5 {
            v.clear();
            assert!(!v.contains(0));
            assert!(!v.contains(1));
        }
        assert!(v.insert(1));
        assert!(v.contains(1));
    }
}
