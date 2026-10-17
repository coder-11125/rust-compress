//! The AV1 multi-symbol arithmetic coder (spec section 8.2) and its encoder counterpart.
//!
//! Probabilities are stored in the specification's form: for a symbol alphabet of size `n` a CDF is
//! `n + 1` values `[c0, c1, ..., c(n-1) = 32768, counter]` where `c(i) = 32768 * P(symbol <= i)`.
//! The same syntax code drives decoding, encoding and bit-cost estimation through [`Coder`]: when
//! decoding, the `val` arguments are ignored and the decoded value is returned; when encoding, `val`
//! is written and returned.

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub trait Coder {
    /// Code one symbol with an adaptive CDF (`cdf.len() == nsyms + 1`).
    fn symbol(&mut self, cdf: &mut [u16], val: usize) -> usize;
    /// Code one symbol with a fixed CDF that is not adapted.
    fn symbol_fixed(&mut self, cdf: &[u16], val: usize) -> usize;
    /// Code an equiprobable bit.
    fn bool_equi(&mut self, val: bool) -> bool;
    /// Code an `n`-bit literal, most significant bit first.
    fn literal(&mut self, n: u32, val: u32) -> u32 {
        let mut v = 0;
        for i in (0..n).rev() {
            v = (v << 1) | self.bool_equi((val >> i) & 1 != 0) as u32;
        }
        v
    }
    /// True for the decoder.
    const DECODING: bool;
}

/// Adapt a CDF towards the coded symbol (spec 8.2.6, "symbol decoding process", last step).
#[inline]
pub fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    let count = cdf[n];
    let rate = 3 + (count > 15) as u32 + (count > 31) as u32 + (n.ilog2()).min(2);
    for (i, c) in cdf[..n - 1].iter_mut().enumerate() {
        if i >= symbol {
            *c += (32768 - *c) >> rate;
        } else {
            *c -= *c >> rate;
        }
    }
    cdf[n] += (count < 32) as u16;
}

pub struct SymbolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    dif: u64,
    rng: u32,
    cnt: i32,
    adapt: bool,
}

const WINDOW: i32 = 64;

impl<'a> SymbolDecoder<'a> {
    pub fn new(data: &'a [u8], adapt: bool) -> Self {
        let mut d = SymbolDecoder { data, pos: 0, dif: (1u64 << (WINDOW - 1)) - 1, rng: 0x8000, cnt: -15, adapt };
        d.refill();
        d
    }

    fn refill(&mut self) {
        let mut s = WINDOW - 9 - (self.cnt + 15);
        while s >= 0 && self.pos < self.data.len() {
            self.dif ^= (self.data[self.pos] as u64) << s;
            self.pos += 1;
            self.cnt += 8;
            s -= 8;
        }
        if self.pos >= self.data.len() {
            // Past the end the stream reads as zeros (the inverted window keeps its ones).
            self.cnt = 0x4000;
        }
    }

    #[inline]
    fn normalize(&mut self, dif: u64, rng: u32) {
        let d = 15 - rng.ilog2() as i32;
        self.cnt -= d;
        self.dif = ((dif.wrapping_add(1)) << d).wrapping_sub(1);
        self.rng = rng << d;
        if self.cnt < 0 {
            self.refill();
        }
    }

    #[inline]
    fn decode(&mut self, cdf: &[u16]) -> usize {
        let n = cdf.len() - 1;
        let c = (self.dif >> (WINDOW - 16)) as u32;
        let r = self.rng;
        let mut v = r;
        let mut sym = 0;
        let mut u;
        loop {
            u = v;
            let icdf = 32768 - cdf[sym] as u32;
            v = ((r >> 8) * (icdf >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - sym - 1) as u32;
            if c >= v {
                break;
            }
            sym += 1;
        }
        let dif = self.dif - ((v as u64) << (WINDOW - 16));
        self.normalize(dif, u - v);
        sym
    }
}

impl Coder for SymbolDecoder<'_> {
    const DECODING: bool = true;

    fn symbol(&mut self, cdf: &mut [u16], _val: usize) -> usize {
        let s = self.decode(cdf);
        if self.adapt {
            update_cdf(cdf, s);
        }
        s
    }

    fn symbol_fixed(&mut self, cdf: &[u16], _val: usize) -> usize {
        self.decode(cdf)
    }

    fn bool_equi(&mut self, _val: bool) -> bool {
        self.decode(&[16384, 32768, 0]) != 0
    }
}

pub struct SymbolEncoder {
    low: u64,
    rng: u32,
    cnt: i32,
    pre: Vec<u16>,
    adapt: bool,
}

impl SymbolEncoder {
    pub fn new(adapt: bool) -> Self {
        SymbolEncoder { low: 0, rng: 0x8000, cnt: -9, pre: Vec::new(), adapt }
    }

    fn encode(&mut self, fl: u32, fh: u32, s: usize, nsyms: usize) {
        let mut l = self.low;
        let mut r = self.rng;
        let n = (nsyms - 1) as u32;
        if fl < 32768 {
            let u = ((r >> 8) * (fl >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - (s as u32 - 1));
            let v = ((r >> 8) * (fh >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s as u32);
            l += (r - u) as u64;
            r = u - v;
        } else {
            r -= ((r >> 8) * (fh >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s as u32);
        }
        self.normalize(l, r);
    }

    fn normalize(&mut self, mut low: u64, rng: u32) {
        let mut c = self.cnt;
        let d = 15 - rng.ilog2() as i32;
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut m = (1u64 << c) - 1;
            if s >= 8 {
                self.pre.push((low >> c) as u16);
                low &= m;
                c -= 8;
                m >>= 8;
            }
            self.pre.push((low >> c) as u16);
            s = c + d - 24;
            low &= m;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// Flush and return the coded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10;
        let m = 0x3fffu64;
        let mut e = ((l + m) & !m) | (m + 1);
        s += c;
        if s > 0 {
            let mut n = (1u64 << (c + 16)) - 1;
            loop {
                self.pre.push((e >> (c + 16)) as u16);
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        let mut out = vec![0u8; self.pre.len()];
        let mut carry = 0u32;
        for i in (0..self.pre.len()).rev() {
            carry += self.pre[i] as u32;
            out[i] = carry as u8;
            carry >>= 8;
        }
        out
    }
}

impl Coder for SymbolEncoder {
    const DECODING: bool = false;

    fn symbol(&mut self, cdf: &mut [u16], val: usize) -> usize {
        let n = cdf.len() - 1;
        let fl = if val > 0 { 32768 - cdf[val - 1] as u32 } else { 32768 };
        let fh = 32768 - cdf[val] as u32;
        self.encode(fl, fh, val, n);
        if self.adapt {
            update_cdf(cdf, val);
        }
        val
    }

    fn symbol_fixed(&mut self, cdf: &[u16], val: usize) -> usize {
        let n = cdf.len() - 1;
        let fl = if val > 0 { 32768 - cdf[val - 1] as u32 } else { 32768 };
        let fh = 32768 - cdf[val] as u32;
        self.encode(fl, fh, val, n);
        val
    }

    fn bool_equi(&mut self, val: bool) -> bool {
        self.symbol_fixed(&[16384, 32768, 0], val as usize);
        val
    }
}

/// Estimates the cost of coding symbols, in 1/256 bit units, without producing output.
pub struct CostCoder {
    pub bits: u64,
    adapt: bool,
}

impl CostCoder {
    pub fn new(adapt: bool) -> Self {
        CostCoder { bits: 0, adapt }
    }
}

/// `-log2(p / 32768) * 256` for `p` in 1..=32768.
pub fn cost_table() -> &'static [u16; 129] {
    use std::sync::OnceLock;
    static T: OnceLock<[u16; 129]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u16; 129];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = (-(i as f64 / 128.0).log2() * 256.0).round() as u16;
        }
        t[0] = t[1];
        t
    })
}

/// Cost in 1/256 bits of a symbol whose probability is `p` (out of 32768).
#[inline]
pub fn prob_cost(p: u32) -> u32 {
    let p = p.clamp(1, 32768);
    let t = cost_table();
    // Linear interpolation on a 128-entry table over [0, 1] keeps the error under 1%.
    let x = p as u64 * 128;
    let i = (x >> 15) as usize;
    let f = (x & 0x7fff) as u32;
    let a = t[i.max(1)] as u32;
    let b = t[(i + 1).min(128)] as u32;
    if i == 0 {
        // Below 1/128 use the exact log.
        return (-(p as f64 / 32768.0).log2() * 256.0).round() as u32;
    }
    (a * (32768 - f) + b * f) >> 15
}

impl Coder for CostCoder {
    const DECODING: bool = false;

    fn symbol(&mut self, cdf: &mut [u16], val: usize) -> usize {
        let lo = if val > 0 { cdf[val - 1] as u32 } else { 0 };
        self.bits += prob_cost(cdf[val] as u32 - lo) as u64;
        if self.adapt {
            update_cdf(cdf, val);
        }
        val
    }

    fn symbol_fixed(&mut self, cdf: &[u16], val: usize) -> usize {
        let lo = if val > 0 { cdf[val - 1] as u32 } else { 0 };
        self.bits += prob_cost(cdf[val] as u32 - lo) as u64;
        val
    }

    fn bool_equi(&mut self, val: bool) -> bool {
        self.bits += 256;
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn random_cdf(rng: &mut Lcg, n: usize) -> Vec<u16> {
        let mut cuts: Vec<u32> = (0..n - 1).map(|_| 1 + rng.next() % 32766).collect();
        cuts.sort();
        let mut cdf: Vec<u16> = cuts.into_iter().map(|c| c as u16).collect();
        // Keep strictly increasing so every symbol has non-zero probability.
        for i in 1..cdf.len() {
            if cdf[i] <= cdf[i - 1] {
                cdf[i] = cdf[i - 1] + 1;
            }
        }
        cdf.push(32768);
        cdf.push(0);
        cdf
    }

    #[test]
    fn encoder_and_decoder_agree() {
        for seed in 0..40u64 {
            let mut rng = Lcg(seed + 1);
            let nctx = 6;
            let sizes: Vec<usize> = (0..nctx).map(|_| 2 + (rng.next() % 15) as usize).collect();
            let base: Vec<Vec<u16>> = sizes.iter().map(|&n| random_cdf(&mut rng, n)).collect();
            let count = 1 + (rng.next() % 3000) as usize;
            let ops: Vec<(usize, usize, bool)> = (0..count)
                .map(|_| {
                    let c = (rng.next() % nctx as u32) as usize;
                    let literal = rng.next() % 5 == 0;
                    let v = if literal { (rng.next() & 1) as usize } else { (rng.next() % sizes[c] as u32) as usize };
                    (c, v, literal)
                })
                .collect();
            for adapt in [true, false] {
                let mut enc = SymbolEncoder::new(adapt);
                let mut cdfs = base.clone();
                for &(c, v, lit) in &ops {
                    if lit {
                        enc.bool_equi(v != 0);
                    } else {
                        enc.symbol(&mut cdfs[c], v);
                    }
                }
                let bytes = enc.finish();
                let mut dec = SymbolDecoder::new(&bytes, adapt);
                let mut cdfs = base.clone();
                for (i, &(c, v, lit)) in ops.iter().enumerate() {
                    let got = if lit { dec.bool_equi(false) as usize } else { dec.symbol(&mut cdfs[c], 0) };
                    assert_eq!(got, v, "seed {seed} adapt {adapt} op {i}");
                }
            }
        }
    }

    #[test]
    fn cost_matches_real_size_roughly() {
        let mut rng = Lcg(7);
        let base = random_cdf(&mut rng, 8);
        let mut enc = SymbolEncoder::new(true);
        let mut cost = CostCoder::new(true);
        let (mut a, mut b) = (base.clone(), base.clone());
        for _ in 0..20000 {
            let v = (rng.next() % 8) as usize;
            enc.symbol(&mut a, v);
            cost.symbol(&mut b, v);
        }
        let real = enc.finish().len() as f64 * 8.0;
        let est = cost.bits as f64 / 256.0;
        assert!((real - est).abs() / real < 0.02, "real {real} est {est}");
    }
}
