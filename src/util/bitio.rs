//! LSB-first bit I/O as used by DEFLATE.

pub struct LsbReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

impl<'a> LsbReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, buf: 0, cnt: 0 }
    }

    /// Ensure at least 56 valid bits are buffered (zeros are supplied past the end).
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let w = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
            self.buf |= w << self.cnt;
            let n = (63 - self.cnt) >> 3;
            self.pos += n as usize;
            self.cnt += n * 8;
        } else {
            while self.cnt <= 56 {
                let b = self.data.get(self.pos).copied().unwrap_or(0);
                self.buf |= (b as u64) << self.cnt;
                self.pos += 1;
                self.cnt += 8;
            }
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> u64 {
        self.buf
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.cnt);
        self.buf >>= n;
        self.cnt -= n;
    }

    /// Read `n` (<= 32) bits. Caller must have refilled.
    #[inline(always)]
    pub fn take(&mut self, n: u32) -> u32 {
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        v
    }

    /// True if more bits were consumed than the input contains.
    pub fn overran(&self) -> bool {
        self.pos * 8 - self.cnt as usize > self.data.len() * 8
    }

    /// Discard bits up to the next byte boundary, then borrow `n` raw bytes.
    pub fn align_and_take(&mut self, n: usize) -> Option<&'a [u8]> {
        let logical = self.pos - (self.cnt / 8) as usize;
        if logical > self.data.len() || self.data.len() - logical < n {
            return None;
        }
        self.pos = logical + n;
        self.buf = 0;
        self.cnt = 0;
        Some(&self.data[logical..logical + n])
    }

    /// Byte position after dropping any partial byte.
    pub fn byte_pos(&self) -> usize {
        self.pos - (self.cnt / 8) as usize
    }
}

pub struct LsbWriter {
    pub out: Vec<u8>,
    buf: u64,
    cnt: u32,
}

impl LsbWriter {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self { out: Vec::with_capacity(n), buf: 0, cnt: 0 }
    }

    /// Write the low `n` (<= 32) bits of `v`.
    #[inline(always)]
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32);
        self.buf |= (v as u64) << self.cnt;
        self.cnt += n;
        if self.cnt >= 32 {
            self.out.extend_from_slice(&(self.buf as u32).to_le_bytes());
            self.buf >>= 32;
            self.cnt -= 32;
        }
    }

    pub fn align(&mut self) {
        while self.cnt > 0 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.cnt = self.cnt.saturating_sub(8);
        }
        self.buf = 0;
    }

    pub fn put_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.cnt, 0);
        self.out.extend_from_slice(bytes);
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

impl Default for LsbWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrip() {
        let mut w = LsbWriter::new();
        let items: Vec<(u32, u32)> =
            (1..=200u32).map(|i| (i.wrapping_mul(2654435761) & ((1 << (i % 17 + 1)) - 1), i % 17 + 1)).collect();
        for &(v, n) in &items {
            w.put(v, n);
        }
        let bytes = w.finish();
        let mut r = LsbReader::new(&bytes);
        for &(v, n) in &items {
            r.refill();
            assert_eq!(r.take(n), v);
        }
        assert!(!r.overran());
    }

    #[test]
    fn overrun_detected() {
        let mut r = LsbReader::new(&[0xFF]);
        r.refill();
        r.take(8);
        assert!(!r.overran());
        r.take(1);
        assert!(r.overran());
    }

    #[test]
    fn align_and_take_bytes() {
        let data = [0b101, 1, 2, 3, 4];
        let mut r = LsbReader::new(&data);
        r.refill();
        assert_eq!(r.take(3), 0b101);
        assert_eq!(r.align_and_take(2), Some(&data[1..3]));
        assert_eq!(r.align_and_take(2), Some(&data[3..5]));
        assert_eq!(r.align_and_take(1), None);
    }
}
