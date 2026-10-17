//! MSB-first bit reader for OBU headers (the AV1 `f(n)`, `su(n)`, `ns(n)`, `leb128` and `uvlc` descriptors).

use crate::error::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    pub fn bit(&mut self) -> Result<u32> {
        let byte = self.data.get(self.pos >> 3).ok_or_else(|| Error::invalid("truncated AV1 header"))?;
        let b = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// `f(n)`: an unsigned n-bit value, n <= 32.
    pub fn bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as u64;
        }
        Ok(v as u32)
    }

    /// `su(n)`: a signed value of n bits (two's complement).
    pub fn signed(&mut self, n: u32) -> Result<i32> {
        let v = self.bits(n)? as i64;
        let sign = 1i64 << (n - 1);
        Ok(if v & sign != 0 { (v - 2 * sign) as i32 } else { v as i32 })
    }

    /// `ns(n)`: a non-symmetric unsigned value in 0..n.
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let v = self.bits(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.bit()?;
        Ok((v << 1) - m + extra)
    }

    /// `uvlc()`.
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        if zeros >= 32 {
            return Ok(u32::MAX);
        }
        Ok(self.bits(zeros)? + ((1u32 << zeros) - 1))
    }

    /// `leb128()`.
    pub fn leb128(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..8 {
            let b = self.bits(8)? as u64;
            value |= (b & 0x7f) << (i * 7);
            if b & 0x80 == 0 {
                break;
            }
        }
        Ok(value)
    }

    /// Skip to the next byte boundary.
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// Whole bytes consumed so far (rounding a partial byte up).
    pub fn byte_pos(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

/// Read a `leb128` from the front of `data`: (value, bytes used).
pub fn leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    for (i, &b) in data.iter().take(8).enumerate() {
        value |= ((b & 0x7f) as u64) << (i * 7);
        if b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    Err(Error::invalid("bad leb128 in AV1 stream"))
}

/// MSB-first bit writer for headers.
#[derive(Default)]
pub struct BitWriter {
    pub data: Vec<u8>,
    cur: u8,
    n: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bit(&mut self, b: bool) {
        self.cur = (self.cur << 1) | b as u8;
        self.n += 1;
        if self.n == 8 {
            self.data.push(self.cur);
            self.cur = 0;
            self.n = 0;
        }
    }

    pub fn bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.bit((value >> i) & 1 != 0);
        }
    }

    pub fn signed(&mut self, value: i32, n: u32) {
        self.bits((value as u32) & ((1u64 << n) - 1) as u32, n);
    }

    /// `ns(n)` encoding.
    pub fn ns(&mut self, value: u32, n: u32) {
        if n <= 1 {
            return;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        if value < m {
            self.bits(value, w - 1);
        } else {
            let v = value + m;
            self.bits(v >> 1, w - 1);
            self.bit(v & 1 != 0);
        }
    }

    pub fn byte_align(&mut self) {
        while self.n != 0 {
            self.bit(false);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.data
    }
}

pub fn write_leb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ns_roundtrips() {
        for n in 1..40u32 {
            for v in 0..n {
                let mut w = BitWriter::new();
                w.ns(v, n);
                w.bits(0b101, 3);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                assert_eq!(r.ns(n).unwrap(), v, "n={n} v={v}");
                assert_eq!(r.bits(3).unwrap(), 0b101);
            }
        }
    }

    #[test]
    fn signed_and_leb128() {
        let mut w = BitWriter::new();
        w.signed(-5, 7);
        w.signed(37, 7);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.signed(7).unwrap(), -5);
        assert_eq!(r.signed(7).unwrap(), 37);
        let mut out = Vec::new();
        write_leb128(&mut out, 300);
        assert_eq!(leb128(&out).unwrap(), (300, 2));
    }
}
