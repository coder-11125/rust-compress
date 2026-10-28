//! Arithmetic entropy coding for JPEG (ITU-T T.81 annexes D, F and G).
//!
//! The QM-coder with its 113-state probability table, the block decoders used for SOF9 (sequential) and SOF10
//! (progressive) files, and an encoder that writes both. The structure follows libjpeg's `jdarith.c` and
//! `jcarith.c`: statistics bins are laid out as in T.81 Tables F.4 and F.5, so streams are interchangeable.
//!
//! Coefficient blocks are zigzag-ordered, as everywhere in this crate.

use super::encode::{CoefImage, segment, write_headers};
use rc_core::error::{Error, Result};

/// Table D.3: `(Qe, next state after an LPS with the MPS-switch flag in bit 7, next state after an MPS)`.
/// State 113 is the fixed 0.5 estimate used for sign and raw bits.
const fn s(qe: u16, lps: u8, mps: u8, switch: u8) -> (u16, u8, u8) {
    (qe, lps | (switch << 7), mps)
}

static STATES: [(u16, u8, u8); 114] = [
    s(0x5a1d, 1, 1, 1),
    s(0x2586, 14, 2, 0),
    s(0x1114, 16, 3, 0),
    s(0x080b, 18, 4, 0),
    s(0x03d8, 20, 5, 0),
    s(0x01da, 23, 6, 0),
    s(0x00e5, 25, 7, 0),
    s(0x006f, 28, 8, 0),
    s(0x0036, 30, 9, 0),
    s(0x001a, 33, 10, 0),
    s(0x000d, 35, 11, 0),
    s(0x0006, 9, 12, 0),
    s(0x0003, 10, 13, 0),
    s(0x0001, 12, 13, 0),
    s(0x5a7f, 15, 15, 1),
    s(0x3f25, 36, 16, 0),
    s(0x2cf2, 38, 17, 0),
    s(0x207c, 39, 18, 0),
    s(0x17b9, 40, 19, 0),
    s(0x1182, 42, 20, 0),
    s(0x0cef, 43, 21, 0),
    s(0x09a1, 45, 22, 0),
    s(0x072f, 46, 23, 0),
    s(0x055c, 48, 24, 0),
    s(0x0406, 49, 25, 0),
    s(0x0303, 51, 26, 0),
    s(0x0240, 52, 27, 0),
    s(0x01b1, 54, 28, 0),
    s(0x0144, 56, 29, 0),
    s(0x00f5, 57, 30, 0),
    s(0x00b7, 59, 31, 0),
    s(0x008a, 60, 32, 0),
    s(0x0068, 62, 33, 0),
    s(0x004e, 63, 34, 0),
    s(0x003b, 32, 35, 0),
    s(0x002c, 33, 9, 0),
    s(0x5ae1, 37, 37, 1),
    s(0x484c, 64, 38, 0),
    s(0x3a0d, 65, 39, 0),
    s(0x2ef1, 67, 40, 0),
    s(0x261f, 68, 41, 0),
    s(0x1f33, 69, 42, 0),
    s(0x19a8, 70, 43, 0),
    s(0x1518, 72, 44, 0),
    s(0x1177, 73, 45, 0),
    s(0x0e74, 74, 46, 0),
    s(0x0bfb, 75, 47, 0),
    s(0x09f8, 77, 48, 0),
    s(0x0861, 78, 49, 0),
    s(0x0706, 79, 50, 0),
    s(0x05cd, 48, 51, 0),
    s(0x04de, 50, 52, 0),
    s(0x040f, 50, 53, 0),
    s(0x0363, 51, 54, 0),
    s(0x02d4, 52, 55, 0),
    s(0x025c, 53, 56, 0),
    s(0x01f8, 54, 57, 0),
    s(0x01a4, 55, 58, 0),
    s(0x0160, 56, 59, 0),
    s(0x0125, 57, 60, 0),
    s(0x00f6, 58, 61, 0),
    s(0x00cb, 59, 62, 0),
    s(0x00ab, 61, 63, 0),
    s(0x008f, 61, 32, 0),
    s(0x5b12, 65, 65, 1),
    s(0x4d04, 80, 66, 0),
    s(0x412c, 81, 67, 0),
    s(0x37d8, 82, 68, 0),
    s(0x2fe8, 83, 69, 0),
    s(0x293c, 84, 70, 0),
    s(0x2379, 86, 71, 0),
    s(0x1edf, 87, 72, 0),
    s(0x1aa9, 87, 73, 0),
    s(0x174e, 72, 74, 0),
    s(0x1424, 72, 75, 0),
    s(0x119c, 74, 76, 0),
    s(0x0f6b, 74, 77, 0),
    s(0x0d51, 75, 78, 0),
    s(0x0bb6, 77, 79, 0),
    s(0x0a40, 77, 48, 0),
    s(0x5832, 80, 81, 1),
    s(0x4d1c, 88, 82, 0),
    s(0x438e, 89, 83, 0),
    s(0x3bdd, 90, 84, 0),
    s(0x34ee, 91, 85, 0),
    s(0x2eae, 92, 86, 0),
    s(0x299a, 93, 87, 0),
    s(0x2516, 86, 71, 0),
    s(0x5570, 88, 89, 1),
    s(0x4ca9, 95, 90, 0),
    s(0x44d9, 96, 91, 0),
    s(0x3e22, 97, 92, 0),
    s(0x3824, 99, 93, 0),
    s(0x32b4, 99, 94, 0),
    s(0x2e17, 93, 86, 0),
    s(0x56a8, 95, 96, 1),
    s(0x4f46, 101, 97, 0),
    s(0x47e5, 102, 98, 0),
    s(0x41cf, 103, 99, 0),
    s(0x3c3d, 104, 100, 0),
    s(0x375e, 99, 93, 0),
    s(0x5231, 105, 102, 0),
    s(0x4c0f, 106, 103, 0),
    s(0x4639, 107, 104, 0),
    s(0x415e, 103, 99, 0),
    s(0x5627, 105, 106, 1),
    s(0x50e7, 108, 107, 0),
    s(0x4b85, 109, 103, 0),
    s(0x5597, 110, 109, 0),
    s(0x504f, 111, 107, 0),
    s(0x5a10, 110, 111, 1),
    s(0x5522, 112, 109, 0),
    s(0x59eb, 112, 111, 1),
    s(0x5a1d, 113, 113, 0),
];

/// Statistics bin of the fixed 0.5 estimate.
const FIXED_BIN: u8 = 113;

/// Conditioning parameters of the DAC marker, per table: DC lower and upper bounds, AC band split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditioning {
    pub dc_l: [u8; 4],
    pub dc_u: [u8; 4],
    pub ac_k: [u8; 4],
}

impl Default for Conditioning {
    fn default() -> Self {
        Conditioning { dc_l: [0; 4], dc_u: [1; 4], ac_k: [5; 4] }
    }
}

const DC_BINS: usize = 64;
const AC_BINS: usize = 256;

// ---------------------------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------------------------

/// The QM-decoder reading from a JPEG byte stream. Marker codes end the data; after that zeros are supplied.
pub(super) struct Reader<'a> {
    data: &'a [u8],
    pub(super) pos: usize,
    marker: bool,
    c: i64,
    a: i64,
    ct: i32,
}

impl<'a> Reader<'a> {
    pub(super) fn new(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos, marker: false, c: 0, a: 0, ct: -16 }
    }

    /// Next data byte, undoing the 0xFF 0x00 stuffing. A marker is left unread (`pos` at its 0xFF) and zeros follow.
    fn next_data(&mut self) -> i64 {
        if self.marker {
            return 0;
        }
        let Some(&b) = self.data.get(self.pos) else {
            self.marker = true;
            return 0;
        };
        if b != 0xFF {
            self.pos += 1;
            return b as i64;
        }
        let mut at = self.pos + 1;
        while self.data.get(at) == Some(&0xFF) {
            at += 1;
        }
        match self.data.get(at) {
            Some(0) => {
                self.pos = at + 1;
                0xFF
            }
            Some(_) => {
                self.pos = at - 1;
                self.marker = true;
                0
            }
            None => {
                self.pos = self.data.len();
                self.marker = true;
                0
            }
        }
    }

    /// Skip to the RSTn marker that ends a restart interval, consume it and start a new arithmetic codeword.
    pub(super) fn restart(&mut self) {
        let d = self.data;
        while self.pos + 1 < d.len() {
            if d[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            match d[self.pos + 1] {
                0 | 0xFF => self.pos += 1,
                0xD0..=0xD7 => {
                    self.pos += 2;
                    break;
                }
                _ => break,
            }
        }
        self.marker = false;
        self.c = 0;
        self.a = 0;
        self.ct = -16;
    }

    fn decode(&mut self, st: &mut u8) -> u32 {
        // Renormalization and data input (D.2.6).
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let data = self.next_data();
                self.c = (self.c << 8) | data;
                self.ct += 8;
                if self.ct < 0 {
                    // Still filling the first two bytes of C.
                    self.ct += 1;
                    if self.ct == 0 {
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }
        let mut sv = *st as u32;
        let (qe, nl, nm) = STATES[(sv & 0x7F) as usize];
        let (qe, nl, nm) = (qe as i64, nl as u32, nm as u32);
        // Decode and estimation procedures (D.2.4 and D.2.5).
        let mut temp = self.a - qe;
        self.a = temp;
        temp <<= self.ct;
        if self.c >= temp {
            self.c -= temp;
            // Conditional LPS exchange.
            if self.a < qe {
                self.a = qe;
                *st = ((sv & 0x80) ^ nm) as u8;
            } else {
                self.a = qe;
                *st = ((sv & 0x80) ^ nl) as u8;
                sv ^= 0x80;
            }
        } else if self.a < 0x8000 {
            // Conditional MPS exchange.
            if self.a < qe {
                *st = ((sv & 0x80) ^ nl) as u8;
                sv ^= 0x80;
            } else {
                *st = ((sv & 0x80) ^ nm) as u8;
            }
        }
        sv >> 7
    }
}

/// Statistics and prediction state of one arithmetic-coded scan.
pub(super) struct ScanDecoder<'a> {
    pub(super) rd: Reader<'a>,
    dc: [[u8; DC_BINS]; 4],
    ac: [[u8; AC_BINS]; 4],
    fixed: u8,
    last_dc: [i32; 4],
    ctx: [usize; 4],
    cond: Conditioning,
}

fn bad() -> Error {
    Error::invalid("bad arithmetic-coded data")
}

impl<'a> ScanDecoder<'a> {
    pub(super) fn new(data: &'a [u8], pos: usize, cond: Conditioning) -> Self {
        ScanDecoder {
            rd: Reader::new(data, pos),
            dc: [[0; DC_BINS]; 4],
            ac: [[0; AC_BINS]; 4],
            fixed: FIXED_BIN,
            last_dc: [0; 4],
            ctx: [0; 4],
            cond,
        }
    }

    /// At a restart: skip the marker, clear the statistics and predictions.
    pub(super) fn restart(&mut self) {
        self.rd.restart();
        self.dc = [[0; DC_BINS]; 4];
        self.ac = [[0; AC_BINS]; 4];
        self.last_dc = [0; 4];
        self.ctx = [0; 4];
    }

    /// Decode a DC difference (F.2.4.1) for the `ci`-th component of the scan; returns the new DC value.
    fn dc_value(&mut self, ci: usize, tbl: usize) -> Result<i32> {
        let mut st = self.ctx[ci];
        if self.rd.decode(&mut self.dc[tbl][st]) == 0 {
            self.ctx[ci] = 0;
            return Ok(self.last_dc[ci]);
        }
        let sign = self.rd.decode(&mut self.dc[tbl][st + 1]);
        st += 2 + sign as usize;
        let mut m = self.rd.decode(&mut self.dc[tbl][st]) as i32;
        if m != 0 {
            st = 20;
            while self.rd.decode(&mut self.dc[tbl][st]) != 0 {
                m <<= 1;
                if m == 0x8000 {
                    return Err(bad());
                }
                st += 1;
            }
        }
        let (l, u) = (self.cond.dc_l[tbl] as u32, self.cond.dc_u[tbl] as u32);
        self.ctx[ci] = if m < ((1i32 << l) >> 1) {
            0
        } else if m > ((1i32 << u) >> 1) {
            12 + sign as usize * 4
        } else {
            4 + sign as usize * 4
        };
        let mut v = m;
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            if self.rd.decode(&mut self.dc[tbl][st]) != 0 {
                v |= m;
            }
        }
        v += 1;
        if sign != 0 {
            v = -v;
        }
        self.last_dc[ci] = self.last_dc[ci].wrapping_add(v);
        Ok(self.last_dc[ci])
    }

    /// Decode the value of a nonzero AC coefficient at position `k` (F.2.4.2).
    fn ac_value(&mut self, tbl: usize, k: usize) -> Result<i32> {
        let sign = self.rd.decode(&mut self.fixed);
        let mut st = 3 * (k - 1) + 2;
        let mut m = self.rd.decode(&mut self.ac[tbl][st]) as i32;
        if m != 0 && self.rd.decode(&mut self.ac[tbl][st]) != 0 {
            m <<= 1;
            st = if k <= self.cond.ac_k[tbl] as usize { 189 } else { 217 };
            while self.rd.decode(&mut self.ac[tbl][st]) != 0 {
                m <<= 1;
                if m == 0x8000 {
                    return Err(bad());
                }
                st += 1;
            }
        }
        let mut v = m;
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            if self.rd.decode(&mut self.ac[tbl][st]) != 0 {
                v |= m;
            }
        }
        v += 1;
        Ok(if sign != 0 { -v } else { v })
    }

    /// Spectral selection AC decoding shared by sequential blocks (1..=63) and progressive first scans.
    fn ac_band(&mut self, blk: &mut [i16], tbl: usize, ss: usize, se: usize, al: u32) -> Result<()> {
        let mut k = ss;
        while k <= se {
            let st = 3 * (k - 1);
            if self.rd.decode(&mut self.ac[tbl][st]) != 0 {
                break; // end of block
            }
            let mut st = st;
            while self.rd.decode(&mut self.ac[tbl][st + 1]) == 0 {
                st += 3;
                k += 1;
                if k > se {
                    return Err(bad());
                }
            }
            let v = self.ac_value(tbl, k)?;
            blk[k] = ((v as u32) << al) as i16;
            k += 1;
        }
        Ok(())
    }

    /// One block of a sequential scan: DC then AC.
    pub(super) fn sequential(&mut self, blk: &mut [i16], ci: usize, dc_tbl: usize, ac_tbl: usize) -> Result<()> {
        blk[0] = self.dc_value(ci, dc_tbl)? as i16;
        self.ac_band(blk, ac_tbl, 1, 63, 0)
    }

    pub(super) fn dc_first(&mut self, blk: &mut [i16], ci: usize, tbl: usize, al: u32) -> Result<()> {
        blk[0] = (self.dc_value(ci, tbl)? << al) as i16;
        Ok(())
    }

    pub(super) fn dc_refine(&mut self, blk: &mut [i16], al: u32) {
        if self.rd.decode(&mut self.fixed) != 0 {
            blk[0] |= 1 << al;
        }
    }

    pub(super) fn ac_first(&mut self, blk: &mut [i16], tbl: usize, ss: usize, se: usize, al: u32) -> Result<()> {
        self.ac_band(blk, tbl, ss, se, al)
    }

    pub(super) fn ac_refine(&mut self, blk: &mut [i16], tbl: usize, ss: usize, se: usize, al: u32) -> Result<()> {
        let p1: i16 = 1 << al;
        let m1: i16 = -1 << al;
        // End of the previous stage's block.
        let mut kex = se;
        while kex > 0 && blk[kex] == 0 {
            kex -= 1;
        }
        let mut k = ss;
        while k <= se {
            let mut st = 3 * (k - 1);
            if k > kex && self.rd.decode(&mut self.ac[tbl][st]) != 0 {
                break; // end of block
            }
            loop {
                if blk[k] != 0 {
                    // Previously nonzero: one correction bit.
                    if self.rd.decode(&mut self.ac[tbl][st + 2]) != 0 {
                        blk[k] += if blk[k] < 0 { m1 } else { p1 };
                    }
                    break;
                }
                if self.rd.decode(&mut self.ac[tbl][st + 1]) != 0 {
                    // Newly nonzero.
                    blk[k] = if self.rd.decode(&mut self.fixed) != 0 { m1 } else { p1 };
                    break;
                }
                st += 3;
                k += 1;
                if k > se {
                    return Err(bad());
                }
            }
            k += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------------------------

/// The QM-encoder with carry handling by stacked 0xFF bytes and "Pacman" termination (D.1).
struct Encoder {
    out: Vec<u8>,
    c: i64,
    a: i64,
    /// Stacked 0xFF values that a carry could still turn into zeros.
    sc: i64,
    /// Pending zero bytes that termination may discard.
    zc: i64,
    ct: i32,
    /// The most recent output byte that is not 0xFF; -1 before the first.
    buffer: i32,
}

impl Encoder {
    fn new() -> Self {
        Encoder { out: Vec::new(), c: 0, a: 0x10000, sc: 0, zc: 0, ct: 11, buffer: -1 }
    }

    fn zeros(&mut self) {
        for _ in 0..self.zc {
            self.out.push(0);
        }
        self.zc = 0;
    }

    /// Write the held byte plus one for a carry that ran into it.
    fn carry(&mut self) {
        if self.buffer >= 0 {
            self.zeros();
            self.out.push((self.buffer + 1) as u8);
            if self.buffer + 1 == 0xFF {
                self.out.push(0);
            }
        }
        self.zc += self.sc; // a carry turns the stacked 0xFF bytes into zeros
        self.sc = 0;
    }

    /// Write the held byte and the stacked 0xFF bytes, which can no longer change.
    fn settle(&mut self) {
        if self.buffer == 0 {
            self.zc += 1;
        } else if self.buffer >= 0 {
            self.zeros();
            self.out.push(self.buffer as u8);
        }
        if self.sc > 0 {
            self.zeros();
            for _ in 0..self.sc {
                self.out.extend_from_slice(&[0xFF, 0x00]);
            }
            self.sc = 0;
        }
    }

    fn encode(&mut self, st: &mut u8, val: u32) {
        let sv = *st as u32;
        let (qe, nl, nm) = STATES[(sv & 0x7F) as usize];
        let (qe, nl, nm) = (qe as i64, nl as u32, nm as u32);
        // Encode and estimation procedures (D.1.4 and D.1.5).
        self.a -= qe;
        if val != sv >> 7 {
            // Less probable symbol; the intervals are exchanged if the LPS one is larger.
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            *st = ((sv & 0x80) ^ nl) as u8;
        } else {
            if self.a >= 0x8000 {
                return;
            }
            if self.a < qe {
                self.c += self.a;
                self.a = qe;
            }
            *st = ((sv & 0x80) ^ nm) as u8;
        }
        // Renormalization and data output (D.1.6).
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                let temp = self.c >> 19;
                if temp > 0xFF {
                    self.carry();
                    // The three spacer bits of C guarantee the new byte is not 0xFF.
                    self.buffer = (temp & 0xFF) as i32;
                } else if temp == 0xFF {
                    self.sc += 1;
                } else {
                    self.settle();
                    self.buffer = (temp & 0xFF) as i32;
                }
                self.c &= 0x7FFFF;
                self.ct += 8;
            }
            if self.a >= 0x8000 {
                break;
            }
        }
    }

    /// Termination (D.1.8): flush the interval, dropping trailing zero bytes.
    fn finish(mut self) -> Vec<u8> {
        let temp = (self.a - 1 + self.c) & 0xFFFF_0000;
        self.c = if temp < self.c { temp + 0x8000 } else { temp };
        self.c <<= self.ct;
        if self.c & 0xF800_0000 != 0 {
            self.carry();
        } else {
            self.settle();
        }
        if self.c & 0x7FFF800 != 0 {
            self.zeros();
            let b = ((self.c >> 19) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0);
            }
            if self.c & 0x7F800 != 0 {
                let b = ((self.c >> 11) & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0);
                }
            }
        }
        self.out
    }
}

struct ScanEncoder<'a> {
    enc: Encoder,
    dc: [[u8; DC_BINS]; 4],
    ac: [[u8; AC_BINS]; 4],
    fixed: u8,
    last_dc: [i32; 4],
    ctx: [usize; 4],
    cond: &'a Conditioning,
}

impl<'a> ScanEncoder<'a> {
    fn new(cond: &'a Conditioning) -> Self {
        ScanEncoder {
            enc: Encoder::new(),
            dc: [[0; DC_BINS]; 4],
            ac: [[0; AC_BINS]; 4],
            fixed: FIXED_BIN,
            last_dc: [0; 4],
            ctx: [0; 4],
            cond,
        }
    }

    /// Encode a DC value as a difference from the previous one (F.1.4.1).
    fn dc_value(&mut self, ci: usize, tbl: usize, value: i32) {
        let mut st = self.ctx[ci];
        let mut v = value - self.last_dc[ci];
        if v == 0 {
            self.enc.encode(&mut self.dc[tbl][st], 0);
            self.ctx[ci] = 0;
            return;
        }
        self.last_dc[ci] = value;
        self.enc.encode(&mut self.dc[tbl][st], 1);
        let sign = (v < 0) as usize;
        self.enc.encode(&mut self.dc[tbl][st + 1], sign as u32);
        st += 2 + sign;
        v = v.abs();
        let mut m = 0i32;
        v -= 1;
        if v != 0 {
            self.enc.encode(&mut self.dc[tbl][st], 1);
            m = 1;
            let mut v2 = v;
            st = 20;
            loop {
                v2 >>= 1;
                if v2 == 0 {
                    break;
                }
                self.enc.encode(&mut self.dc[tbl][st], 1);
                m <<= 1;
                st += 1;
            }
        }
        self.enc.encode(&mut self.dc[tbl][st], 0);
        let (l, u) = (self.cond.dc_l[tbl] as u32, self.cond.dc_u[tbl] as u32);
        self.ctx[ci] = if m < ((1i32 << l) >> 1) {
            0
        } else if m > ((1i32 << u) >> 1) {
            12 + sign * 4
        } else {
            4 + sign * 4
        };
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            self.enc.encode(&mut self.dc[tbl][st], ((m & v) != 0) as u32);
        }
    }

    /// Encode the (already point-transformed) nonzero AC value `v` at position `k` (F.1.4.2).
    fn ac_value(&mut self, tbl: usize, k: usize, v: i32) {
        self.enc.encode(&mut self.fixed, (v < 0) as u32);
        let v = v.abs() - 1;
        let mut st = 3 * (k - 1) + 2;
        let mut m = 0i32;
        if v != 0 {
            self.enc.encode(&mut self.ac[tbl][st], 1);
            m = 1;
            let mut v2 = v >> 1;
            if v2 != 0 {
                self.enc.encode(&mut self.ac[tbl][st], 1);
                m <<= 1;
                st = if k <= self.cond.ac_k[tbl] as usize { 189 } else { 217 };
                loop {
                    v2 >>= 1;
                    if v2 == 0 {
                        break;
                    }
                    self.enc.encode(&mut self.ac[tbl][st], 1);
                    m <<= 1;
                    st += 1;
                }
            }
        }
        self.enc.encode(&mut self.ac[tbl][st], 0);
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            self.enc.encode(&mut self.ac[tbl][st], ((m & v) != 0) as u32);
        }
    }

    /// The coefficient after the AC point transform (division by 2^al rounding toward zero).
    fn shifted(v: i16, al: u32) -> i32 {
        let m = (v as i32).abs() >> al;
        if v < 0 { -m } else { m }
    }

    fn ac_band(&mut self, blk: &[i16], tbl: usize, ss: usize, se: usize, al: u32) {
        let mut ke = se;
        while ke > 0 && Self::shifted(blk[ke], al) == 0 {
            ke -= 1;
        }
        let mut k = ss;
        while k <= ke {
            let mut st = 3 * (k - 1);
            self.enc.encode(&mut self.ac[tbl][st], 0); // not end of block
            let v = loop {
                let v = Self::shifted(blk[k], al);
                if v != 0 {
                    break v;
                }
                self.enc.encode(&mut self.ac[tbl][st + 1], 0);
                st += 3;
                k += 1;
            };
            self.enc.encode(&mut self.ac[tbl][st + 1], 1);
            self.ac_value(tbl, k, v);
            k += 1;
        }
        if k <= se {
            self.enc.encode(&mut self.ac[tbl][3 * (k - 1)], 1);
        }
    }

    fn sequential(&mut self, blk: &[i16], ci: usize, dc_tbl: usize, ac_tbl: usize) {
        self.dc_value(ci, dc_tbl, blk[0] as i32);
        self.ac_band(blk, ac_tbl, 1, 63, 0);
    }

    fn ac_refine(&mut self, blk: &[i16], tbl: usize, ss: usize, se: usize, ah: u32, al: u32) {
        let mut ke = se;
        while ke > 0 && Self::shifted(blk[ke], al) == 0 {
            ke -= 1;
        }
        let mut kex = ke;
        while kex > 0 && Self::shifted(blk[kex], ah) == 0 {
            kex -= 1;
        }
        let mut k = ss;
        while k <= ke {
            let mut st = 3 * (k - 1);
            if k > kex {
                self.enc.encode(&mut self.ac[tbl][st], 0); // not end of block
            }
            loop {
                let v = Self::shifted(blk[k], al);
                if v != 0 {
                    if v.abs() > 1 {
                        // Previously nonzero: its next bit.
                        self.enc.encode(&mut self.ac[tbl][st + 2], (v.abs() & 1) as u32);
                    } else {
                        self.enc.encode(&mut self.ac[tbl][st + 1], 1);
                        self.enc.encode(&mut self.fixed, (v < 0) as u32);
                    }
                    break;
                }
                self.enc.encode(&mut self.ac[tbl][st + 1], 0);
                st += 3;
                k += 1;
            }
            k += 1;
        }
        if k <= se {
            self.enc.encode(&mut self.ac[tbl][3 * (k - 1)], 1);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------------------------------------------

/// How to write an arithmetic-coded JPEG.
#[derive(Clone, Debug, Default)]
pub struct ArithOptions {
    /// Progressive (SOF10) instead of sequential (SOF9).
    pub progressive: bool,
    pub conditioning: Conditioning,
    /// MCUs between restart markers (0 for none). Each interval starts a new codeword with cleared statistics.
    pub restart_interval: usize,
}

struct Scan {
    comps: Vec<usize>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

/// Successive-approximation script in the style of libjpeg's `jpeg_simple_progression`, for any component count.
fn progressive_script(ncomp: usize) -> Vec<Scan> {
    let s = |comps: &[usize], ss, se, ah, al| Scan { comps: comps.to_vec(), ss, se, ah, al };
    let all: Vec<usize> = (0..ncomp).collect();
    let mut v = vec![s(&all, 0, 0, 0, 1), s(&[0], 1, 5, 0, 2)];
    for c in 1..ncomp {
        v.push(s(&[c], 1, 63, 0, 1));
    }
    v.push(s(&[0], 6, 63, 0, 2));
    v.push(s(&[0], 1, 63, 2, 1));
    v.push(s(&all, 0, 0, 1, 0));
    for c in 1..ncomp {
        v.push(s(&[c], 1, 63, 1, 0));
    }
    v.push(s(&[0], 1, 63, 1, 0));
    v
}

/// Restart-interval bookkeeping while a scan is written.
struct Restarts {
    interval: usize,
    units: usize,
    next: u8,
}

impl Restarts {
    /// Call before each MCU: at the start of every interval but the first, close the codeword, write the marker
    /// and start afresh with cleared statistics.
    fn at_mcu<'c>(&mut self, e: &mut ScanEncoder<'c>, cond: &'c Conditioning, out: &mut Vec<u8>) {
        if self.interval > 0 && self.units > 0 && self.units.is_multiple_of(self.interval) {
            let done = std::mem::replace(e, ScanEncoder::new(cond));
            out.extend_from_slice(&done.enc.finish());
            out.extend_from_slice(&[0xFF, 0xD0 + self.next]);
            self.next = (self.next + 1) & 7;
        }
        self.units += 1;
    }
}

fn run_scan(ci: &CoefImage, sc: &Scan, cond: &Conditioning, restart: usize) -> Vec<u8> {
    let mut e = ScanEncoder::new(cond);
    let mut out = Vec::new();
    let mut marks = Restarts { interval: restart, units: 0, next: 0 };
    let (hmax, vmax) = ci.hv_max();
    let visit = |e: &mut ScanEncoder, scan_idx: usize, comp: usize, blk: &[i16]| {
        let tbl = comp.min(1);
        if sc.ss == 0 && sc.se == 63 {
            e.sequential(blk, scan_idx, tbl, tbl);
        } else if sc.ss == 0 {
            if sc.ah == 0 {
                e.dc_value(scan_idx, tbl, (blk[0] as i32) >> sc.al);
            } else {
                e.enc.encode(&mut e.fixed, ((blk[0] >> sc.al) & 1) as u32);
            }
        } else if sc.ah == 0 {
            e.ac_band(blk, tbl, sc.ss, sc.se, sc.al);
        } else {
            e.ac_refine(blk, tbl, sc.ss, sc.se, sc.ah, sc.al);
        }
    };
    if sc.comps.len() > 1 {
        let (mcus_x, mcus_y) = ci.mcus();
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                marks.at_mcu(&mut e, cond, &mut out);
                for (si, &k) in sc.comps.iter().enumerate() {
                    let c = &ci.comps[k];
                    for by in 0..c.v {
                        for bx in 0..c.h {
                            let b = (my * c.v + by) * c.blocks_w + mx * c.h + bx;
                            visit(&mut e, si, k, &c.coefs[b * 64..b * 64 + 64]);
                        }
                    }
                }
            }
        }
    } else {
        let k = sc.comps[0];
        let c = &ci.comps[k];
        // Non-interleaved scans cover only the blocks that overlap the image.
        let real_w = (ci.width * c.h).div_ceil(hmax).div_ceil(8);
        let real_h = (ci.height * c.v).div_ceil(vmax).div_ceil(8);
        for by in 0..real_h {
            for bx in 0..real_w {
                marks.at_mcu(&mut e, cond, &mut out);
                let b = by * c.blocks_w + bx;
                visit(&mut e, 0, k, &c.coefs[b * 64..b * 64 + 64]);
            }
        }
    }
    out.extend_from_slice(&e.enc.finish());
    out
}

/// Serialize as an arithmetic-coded JPEG (SOF9, or SOF10 when `opts.progressive`). The coefficients are written
/// exactly; only the entropy coding differs from a Huffman file.
pub fn write_arithmetic_with(ci: &CoefImage, opts: &ArithOptions) -> Vec<u8> {
    let cond = &opts.conditioning;
    let mut out = write_headers(ci, if opts.progressive { 0xCA } else { 0xC9 }, ci.width * ci.height / 4 + 1024);
    // Conditioning that differs from the default, for the (at most two) tables in use.
    let tables = if ci.comps.len() == 1 { 1 } else { 2 };
    let dflt = Conditioning::default();
    let mut dac = Vec::new();
    for t in 0..tables {
        if cond.dc_l[t] != dflt.dc_l[t] || cond.dc_u[t] != dflt.dc_u[t] {
            dac.extend_from_slice(&[t as u8, cond.dc_u[t] << 4 | cond.dc_l[t]]);
        }
        if cond.ac_k[t] != dflt.ac_k[t] {
            dac.extend_from_slice(&[0x10 | t as u8, cond.ac_k[t]]);
        }
    }
    if !dac.is_empty() {
        segment(&mut out, 0xCC, &dac);
    }
    if opts.restart_interval > 0 {
        segment(&mut out, 0xDD, &(opts.restart_interval.min(0xFFFF) as u16).to_be_bytes());
    }
    let script = if opts.progressive {
        progressive_script(ci.comps.len())
    } else {
        vec![Scan { comps: (0..ci.comps.len()).collect(), ss: 0, se: 63, ah: 0, al: 0 }]
    };
    for sc in &script {
        let mut sos = vec![sc.comps.len() as u8];
        for &k in &sc.comps {
            let t = k.min(1) as u8;
            sos.extend_from_slice(&[ci.comps[k].id, t << 4 | t]);
        }
        sos.extend_from_slice(&[sc.ss as u8, sc.se as u8, (sc.ah << 4 | sc.al) as u8]);
        segment(&mut out, 0xDA, &sos);
        out.extend_from_slice(&run_scan(ci, sc, cond, opts.restart_interval.min(0xFFFF)));
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

/// The smaller of the sequential and the progressive arithmetic-coded file.
pub fn write_arithmetic(ci: &CoefImage) -> Vec<u8> {
    let sequential = write_arithmetic_with(ci, &ArithOptions::default());
    let progressive = write_arithmetic_with(ci, &ArithOptions { progressive: true, ..ArithOptions::default() });
    if progressive.len() < sequential.len() { progressive } else { sequential }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_table_is_well_formed() {
        for (i, &(qe, nl, nm)) in STATES.iter().enumerate() {
            assert!(qe > 0 && qe <= 0x5b12, "state {i}");
            assert!((nl & 0x7F) < 114 && nm < 114, "state {i}");
        }
        // The first state of each ladder switches the sense of the MPS after an LPS; no other state does.
        let switching: Vec<usize> =
            STATES.iter().enumerate().filter(|(_, s)| s.1 & 0x80 != 0).map(|(i, _)| i).collect();
        assert_eq!(switching, vec![0, 14, 36, 64, 80, 88, 95, 105, 110, 112]);
        assert_eq!(STATES[FIXED_BIN as usize], (0x5a1d, 113, 113));
    }

    /// Encode a long run of decisions in assorted contexts and probabilities, then decode them again.
    fn round_trip(seed: u32, n: usize, skew: u32) {
        let mut rng = seed;
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            rng >> 8
        };
        let (mut decisions, mut ctxs) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let ctx = (next() % 40) as usize;
            // Context-dependent bias, so the estimator walks through many states.
            let p = (ctx as u32 * 97 + skew) % 1000;
            decisions.push(((next() % 1000) < p) as u32);
            ctxs.push(ctx);
        }
        let mut stats = [0u8; 64];
        let mut enc = Encoder::new();
        for (d, &c) in decisions.iter().zip(&ctxs) {
            enc.encode(&mut stats[c], *d);
        }
        let mut bytes = enc.finish();
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        let mut stats = [0u8; 64];
        let mut rd = Reader::new(&bytes, 0);
        for (i, (d, &c)) in decisions.iter().zip(&ctxs).enumerate() {
            assert_eq!(rd.decode(&mut stats[c]), *d, "decision {i} (seed {seed}, skew {skew})");
        }
    }

    /// Blocks with the coefficient ranges of 12-bit files (DC differences past 15000, AC past 9000), in the
    /// sequential and the successive-approximation procedures.
    #[test]
    fn wide_coefficients_round_trip() {
        let mut rng = 0xC0FFEEu32;
        let mut next = move |m: u32| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) % m
        };
        let blocks: Vec<[i16; 64]> = (0..400)
            .map(|b| {
                let mut blk = [0i16; 64];
                blk[0] = (next(32000) as i32 - 16000) as i16;
                for k in 1..64 {
                    // Sparse, with a mix of tiny and huge values.
                    match next(10) {
                        0 => blk[k] = (next(18000) as i32 - 9000) as i16,
                        1 | 2 => blk[k] = (next(7) as i32 - 3) as i16,
                        _ => {}
                    }
                }
                if b % 50 == 0 {
                    blk[63] = -9000;
                }
                blk
            })
            .collect();
        let cond = Conditioning::default();
        let mut e = ScanEncoder::new(&cond);
        for blk in &blocks {
            e.sequential(blk, 0, 0, 0);
        }
        let mut bytes = e.enc.finish();
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        let mut d = ScanDecoder::new(&bytes, 0, cond.clone());
        for (i, want) in blocks.iter().enumerate() {
            let mut got = [0i16; 64];
            d.sequential(&mut got, 0, 0, 0).unwrap();
            assert_eq!(&got[..], &want[..], "block {i}");
        }
        // First scans at a point transform of 2, then the refinement at 1 and 0, as a progressive file does.
        let mut e = ScanEncoder::new(&cond);
        for blk in &blocks {
            e.dc_value(0, 0, (blk[0] as i32) >> 2);
        }
        for blk in &blocks {
            e.ac_band(blk, 0, 1, 63, 2);
        }
        let mut bytes = e.enc.finish();
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        let mut d = ScanDecoder::new(&bytes, 0, cond.clone());
        let mut got = vec![[0i16; 64]; blocks.len()];
        for g in got.iter_mut() {
            d.dc_first(g, 0, 0, 2).unwrap();
        }
        for g in got.iter_mut() {
            d.ac_first(g, 0, 1, 63, 2).unwrap();
        }
        for (i, (g, want)) in got.iter().zip(&blocks).enumerate() {
            assert_eq!(g[0], (want[0] >> 2) << 2, "dc {i}");
            for k in 1..64 {
                let m = ((want[k] as i32).abs() >> 2) << 2;
                assert_eq!(g[k] as i32, if want[k] < 0 { -m } else { m }, "block {i} coefficient {k}");
            }
        }
    }

    #[test]
    fn the_qm_coder_round_trips_through_many_states() {
        for seed in 1..25 {
            for skew in [0, 13, 250, 777, 995] {
                round_trip(seed, 3000, skew);
            }
        }
        round_trip(99, 200_000, 400);
        round_trip(7, 1, 0);
        round_trip(8, 2, 500);
        // Long runs of one decision drive the estimator to its extreme states and exercise carries.
        let mut stats = [0u8; 1];
        let mut enc = Encoder::new();
        let pattern: Vec<u32> = (0..60_000).map(|i| ((i % 9973 == 0) || (i % 4001 == 7)) as u32).collect();
        for &d in &pattern {
            enc.encode(&mut stats[0], d);
        }
        let mut bytes = enc.finish();
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        let mut stats = [0u8; 1];
        let mut rd = Reader::new(&bytes, 0);
        for (i, &d) in pattern.iter().enumerate() {
            assert_eq!(rd.decode(&mut stats[0]), d, "run decision {i}");
        }
    }
}
