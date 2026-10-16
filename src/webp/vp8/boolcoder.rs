//! The VP8 boolean entropy coder (RFC 6386 section 7): an arithmetic coder over binary decisions,
//! each with an 8-bit probability that the decision is zero.

/// Reads decisions from a byte slice. Reading past the end yields zero bits, as libvpx does.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// Window of not yet consumed bits; the top part is compared against the split point.
    value: u64,
    /// True range minus one, kept in 127..=254 between calls.
    range: u32,
    /// Number of valid bits in `value` beyond the 8 being compared (negative: needs a refill).
    bits: i32,
    /// Bytes requested beyond the end of the data (a measure of truncation).
    overrun: usize,
}

impl<'a> BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = BoolDecoder { data, pos: 0, value: 0, range: 254, bits: -8, overrun: 0 };
        d.refill();
        d
    }

    fn refill(&mut self) {
        // Take up to seven bytes at once.
        if self.pos + 8 <= self.data.len() {
            let mut b = [0u8; 8];
            b[1..].copy_from_slice(&self.data[self.pos..self.pos + 7]);
            self.value = u64::from_be_bytes(b) | (self.value << 56);
            self.pos += 7;
            self.bits += 56;
        } else if self.pos < self.data.len() {
            self.value = self.data[self.pos] as u64 | (self.value << 8);
            self.pos += 1;
            self.bits += 8;
        } else {
            self.value <<= 8;
            self.bits += 8;
            self.overrun += 1;
        }
    }

    /// True when decoding needed more bytes than the partition holds.
    pub fn overran(&self) -> bool {
        self.overrun > 2
    }

    /// Decode one decision whose probability of being `false` is `prob / 256`.
    #[inline]
    pub fn get(&mut self, prob: u8) -> bool {
        if self.bits < 0 {
            self.refill();
        }
        let split = (self.range * prob as u32) >> 8;
        let pos = self.bits as u32;
        let v = (self.value >> pos) as u32;
        let bit = v > split;
        let mut range;
        if bit {
            range = self.range - split;
            self.value -= (split as u64 + 1) << pos;
        } else {
            range = split + 1;
        }
        let shift = 7 ^ (31 - range.leading_zeros());
        range <<= shift;
        self.bits -= shift as i32;
        self.range = range - 1;
        bit
    }

    /// An unsigned `n`-bit literal, most significant bit first, each bit at probability 1/2.
    pub fn literal(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.get(128) as u32;
        }
        v
    }

    pub fn flag(&mut self) -> bool {
        self.get(128)
    }

    /// An `n`-bit magnitude followed by a sign bit.
    pub fn signed(&mut self, n: u32) -> i32 {
        let v = self.literal(n) as i32;
        if self.flag() { -v } else { v }
    }

    /// Walk a binary tree stored as in the RFC (`tree[i]` > 0: next index, <= 0: negated leaf).
    pub fn tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
        let mut i = 0usize;
        loop {
            let next = tree[i + self.get(probs[i >> 1]) as usize];
            if next <= 0 {
                return (-next) as u8;
            }
            i = next as usize;
        }
    }
}

/// Writes decisions; the exact inverse of [`BoolDecoder`].
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        BoolEncoder { out: Vec::new(), range: 255, bottom: 0, bit_count: 24 }
    }

    fn carry(&mut self) {
        let mut i = self.out.len();
        while i > 0 && self.out[i - 1] == 255 {
            self.out[i - 1] = 0;
            i -= 1;
        }
        if i > 0 {
            self.out[i - 1] += 1;
        }
    }

    /// Encode `bit`, whose probability of being `false` is `prob / 256`.
    #[inline]
    pub fn put(&mut self, prob: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        if bit {
            self.bottom = self.bottom.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            if self.bottom & (1 << 31) != 0 {
                self.carry();
            }
            self.bottom <<= 1;
            self.bit_count -= 1;
            if self.bit_count == 0 {
                self.out.push((self.bottom >> 24) as u8);
                self.bottom &= (1 << 24) - 1;
                self.bit_count = 8;
            }
        }
    }

    pub fn literal(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.put(128, (value >> i) & 1 == 1);
        }
    }

    pub fn flag(&mut self, b: bool) {
        self.put(128, b);
    }

    pub fn signed(&mut self, value: i32, n: u32) {
        self.literal(value.unsigned_abs(), n);
        self.flag(value < 0);
    }

    /// Encode leaf `value` by walking `tree` (see [`BoolDecoder::tree`]).
    pub fn tree(&mut self, tree: &[i8], probs: &[u8], value: u8) {
        // Find the path to the leaf by depth-first search; trees are tiny.
        fn path(tree: &[i8], i: usize, value: u8, acc: &mut Vec<(usize, bool)>) -> bool {
            for b in 0..2 {
                let next = tree[i + b];
                acc.push((i, b == 1));
                if next <= 0 {
                    if (-next) as u8 == value {
                        return true;
                    }
                } else if path(tree, next as usize, value, acc) {
                    return true;
                }
                acc.pop();
            }
            false
        }
        let mut acc = Vec::new();
        assert!(path(tree, 0, value, &mut acc), "value not in tree");
        for (i, bit) in acc {
            self.put(probs[i >> 1], bit);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        let mut c = self.bit_count;
        let mut v = self.bottom;
        if v & (1u32.wrapping_shl((32 - c) as u32)) != 0 {
            self.carry();
        }
        v <<= c & 7;
        c >>= 3;
        while c > 0 {
            v <<= 8;
            c -= 1;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v <<= 8;
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn decisions_roundtrip_at_every_probability() {
        let mut seed = 1;
        let items: Vec<(u8, bool)> = (0..20000)
            .map(|i| {
                let prob = match i % 5 {
                    0 => 1,
                    1 => 255,
                    _ => (lcg(&mut seed) % 255 + 1) as u8,
                };
                // Make the data roughly follow its probability so long runs of likely bits occur.
                let bit = (lcg(&mut seed) % 256) as u8 >= prob;
                (prob, bit)
            })
            .collect();
        let mut e = BoolEncoder::new();
        for &(p, b) in &items {
            e.put(p, b);
        }
        let bytes = e.finish();
        let mut d = BoolDecoder::new(&bytes);
        for (i, &(p, b)) in items.iter().enumerate() {
            assert_eq!(d.get(p), b, "decision {i}");
        }
        assert!(!d.overran());
    }

    #[test]
    fn literals_signed_values_and_trees_roundtrip() {
        const TREE: [i8; 6] = [2, 4, 0, -1, -2, -3];
        let probs = [100, 150, 200];
        let mut e = BoolEncoder::new();
        for v in 0..4u8 {
            e.tree(&TREE, &probs, v);
        }
        e.literal(0b1011001, 7);
        e.signed(-37, 6);
        e.signed(21, 6);
        let bytes = e.finish();
        let mut d = BoolDecoder::new(&bytes);
        for v in 0..4u8 {
            assert_eq!(d.tree(&TREE, &probs), v);
        }
        assert_eq!(d.literal(7), 0b1011001);
        assert_eq!(d.signed(6), -37);
        assert_eq!(d.signed(6), 21);
    }

    #[test]
    fn empty_and_tiny_inputs_do_not_panic() {
        let mut d = BoolDecoder::new(&[]);
        for _ in 0..100 {
            d.get(128);
        }
        assert!(d.overran());
        let e = BoolEncoder::new().finish();
        let mut d = BoolDecoder::new(&e);
        assert!(!d.get(128));
    }
}
