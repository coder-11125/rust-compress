//! Canonical Huffman helpers shared by the DEFLATE and JPEG codecs.

use crate::error::{Error, Result};
use crate::util::bitio::LsbReader;

/// Compute length-limited Huffman code lengths for the given symbol frequencies.
///
/// Symbols with zero frequency get length 0. If exactly one symbol is used it
/// receives length 1. Lengths never exceed `max_len`.
pub fn build_lengths(freqs: &[u32], max_len: u8) -> Vec<u8> {
    let mut lengths = vec![0u8; freqs.len()];
    let mut syms: Vec<(u32, usize)> = freqs.iter().enumerate().filter(|(_, f)| **f > 0).map(|(i, f)| (*f, i)).collect();
    match syms.len() {
        0 => return lengths,
        1 => {
            lengths[syms[0].1] = 1;
            return lengths;
        }
        _ => {}
    }
    // Rarest first; among equal frequencies the highest symbol comes first so it
    // gets the longest code (JPEG relies on this to reserve the all-ones code).
    syms.sort_unstable_by_key(|&(f, i)| (f, std::cmp::Reverse(i)));
    let n = syms.len();

    // Two-queue Huffman construction over leaves sorted by frequency.
    // Nodes 0..n are leaves, n..2n-1 internal.
    let mut weight: Vec<u64> = syms.iter().map(|s| s.0 as u64).collect();
    weight.resize(2 * n - 1, 0);
    let mut parent = vec![0usize; 2 * n - 1];
    let (mut leaf, mut node, mut next) = (0usize, n, n);
    while next < 2 * n - 1 {
        let mut pick = || {
            if leaf < n && (node >= next || weight[leaf] <= weight[node]) {
                leaf += 1;
                leaf - 1
            } else {
                node += 1;
                node - 1
            }
        };
        let a = pick();
        let b = pick();
        weight[next] = weight[a] + weight[b];
        parent[a] = next;
        parent[b] = next;
        next += 1;
    }
    // Depths: root is the last node.
    let mut depth = vec![0u32; 2 * n - 1];
    for i in (0..2 * n - 2).rev() {
        depth[i] = depth[parent[i]] + 1;
    }

    // Count codes per length, clamping to max_len.
    let max = max_len as usize;
    let mut count = vec![0u32; max + 1];
    for d in depth.iter().take(n) {
        count[(*d as usize).min(max)] += 1;
    }
    // Restore the Kraft equality after clamping (miniz-style fix-up).
    let mut total: u64 = 0;
    for (l, c) in count.iter().enumerate().skip(1) {
        total += (*c as u64) << (max - l);
    }
    while total > (1u64 << max) {
        count[max] -= 1;
        for l in (1..max).rev() {
            if count[l] > 0 {
                count[l] -= 1;
                count[l + 1] += 2;
                break;
            }
        }
        total -= 1;
    }
    // Rarest symbols (front of `syms`) get the longest codes.
    let mut idx = 0;
    for l in (1..=max).rev() {
        for _ in 0..count[l] {
            lengths[syms[idx].1] = l as u8;
            idx += 1;
        }
    }
    lengths
}

/// Assign canonical (MSB-first) codes from code lengths.
pub fn canonical_codes(lengths: &[u8]) -> Vec<u16> {
    let max = lengths.iter().copied().max().unwrap_or(0) as usize;
    let mut bl_count = vec![0u32; max + 2];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next = vec![0u32; max + 2];
    let mut code = 0u32;
    for bits in 1..=max {
        code = (code + bl_count[bits - 1]) << 1;
        next[bits] = code;
    }
    lengths
        .iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                let c = next[l as usize];
                next[l as usize] += 1;
                c as u16
            }
        })
        .collect()
}

pub fn reverse_bits(code: u32, len: u32) -> u32 {
    code.reverse_bits() >> (32 - len)
}

/// Codes already bit-reversed so they can be written with an LSB-first writer.
pub fn lsb_codes(lengths: &[u8]) -> Vec<u16> {
    canonical_codes(lengths)
        .iter()
        .zip(lengths)
        .map(|(&c, &l)| if l == 0 { 0 } else { reverse_bits(c as u32, l as u32) as u16 })
        .collect()
}

const FAST_BITS: u32 = 10;

/// Table-driven decoder for DEFLATE-style (LSB-first stream, MSB-first code) Huffman codes.
pub struct Decoder {
    /// `symbol << 4 | length` for codes up to FAST_BITS long; 0 if longer.
    fast: Vec<u16>,
    count: [u16; 16],
    symbols: Vec<u16>,
}

impl Decoder {
    pub fn new(lengths: &[u8]) -> Result<Self> {
        let mut count = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err(Error::invalid("huffman code length > 15"));
            }
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left = (left << 1) - c as i32;
            if left < 0 {
                return Err(Error::invalid("oversubscribed huffman code"));
            }
        }
        let mut offs = [0u16; 17];
        for l in 1..16 {
            offs[l + 1] = offs[l] + count[l];
        }
        let mut symbols = vec![0u16; offs[16] as usize];
        for (s, &l) in lengths.iter().enumerate() {
            if l > 0 {
                symbols[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        let mut fast = vec![0u16; 1 << FAST_BITS];
        let codes = canonical_codes(lengths);
        for (s, &l) in lengths.iter().enumerate() {
            if l > 0 && (l as u32) <= FAST_BITS {
                let rev = reverse_bits(codes[s] as u32, l as u32) as usize;
                let entry = ((s as u16) << 4) | l as u16;
                let mut i = rev;
                while i < fast.len() {
                    fast[i] = entry;
                    i += 1 << l;
                }
            }
        }
        Ok(Self { fast, count, symbols })
    }

    /// Decode one symbol. Caller must have refilled at least 15 bits.
    #[inline(always)]
    pub fn decode(&self, r: &mut LsbReader) -> Result<u16> {
        let bits = r.peek();
        let e = self.fast[(bits & ((1 << FAST_BITS) - 1)) as usize];
        if e & 15 != 0 {
            r.consume((e & 15) as u32);
            return Ok(e >> 4);
        }
        self.decode_slow(r, bits)
    }

    #[cold]
    fn decode_slow(&self, r: &mut LsbReader, bits: u64) -> Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= ((bits >> (len - 1)) & 1) as i32;
            let c = self.count[len] as i32;
            if code - c < first {
                r.consume(len as u32);
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += c;
            first = (first + c) << 1;
            code <<= 1;
        }
        Err(Error::invalid("bad huffman code"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::bitio::LsbWriter;

    fn kraft(lengths: &[u8], max: u8) -> u64 {
        lengths.iter().filter(|&&l| l > 0).map(|&l| 1u64 << (max - l)).sum()
    }

    #[test]
    fn lengths_are_complete_and_limited() {
        let freqs: Vec<u32> = (0..300).map(|i| (i * i % 97) as u32 + (i % 5 == 0) as u32).collect();
        for max in [9u8, 12, 15] {
            let l = build_lengths(&freqs, max);
            assert!(l.iter().all(|&x| x <= max));
            assert_eq!(kraft(&l, max), 1u64 << max);
            for (f, &len) in freqs.iter().zip(&l) {
                assert_eq!(*f > 0, len > 0);
            }
        }
    }

    #[test]
    fn skewed_frequencies_get_clamped() {
        // Fibonacci-like frequencies produce very deep trees.
        let mut freqs = vec![1u32, 1];
        for i in 2..30 {
            let v = freqs[i - 1] + freqs[i - 2];
            freqs.push(v);
        }
        let l = build_lengths(&freqs, 15);
        assert!(l.iter().all(|&x| x <= 15));
        assert_eq!(kraft(&l, 15), 1 << 15);
    }

    #[test]
    fn single_symbol_gets_length_one() {
        let mut f = vec![0u32; 10];
        f[4] = 99;
        assert_eq!(build_lengths(&f, 15)[4], 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let freqs: Vec<u32> = (0..286).map(|i| 1 + (i * 31 % 53) as u32 * if i < 20 { 200 } else { 1 }).collect();
        let lengths = build_lengths(&freqs, 15);
        let codes = lsb_codes(&lengths);
        let msg: Vec<usize> = (0..5000).map(|i| (i * 7919) % 286).collect();
        let mut w = LsbWriter::new();
        for &s in &msg {
            w.put(codes[s] as u32, lengths[s] as u32);
        }
        let bytes = w.finish();
        let dec = Decoder::new(&lengths).unwrap();
        let mut r = LsbReader::new(&bytes);
        for &s in &msg {
            r.refill();
            assert_eq!(dec.decode(&mut r).unwrap() as usize, s);
        }
    }

    #[test]
    fn oversubscribed_rejected() {
        assert!(Decoder::new(&[1, 1, 1]).is_err());
    }
}
