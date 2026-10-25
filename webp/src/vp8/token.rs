//! Coefficient tokenization for the encoder. One walk over a block's levels (mirroring the
//! decoder's token reader) feeds a [`TokenSink`], which can estimate the cost in bits, gather
//! statistics for probability tuning, or write the actual bits.

use super::boolcoder::BoolEncoder;
use super::tables::*;

/// `COST[p]` = bits needed to code a `false` decision whose probability is `p / 256`.
fn cost_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|p| if p == 0 { 24.0 } else { -((p as f32) / 256.0).log2() }))
}

/// Bits to code `bit` when the probability of `false` is `prob / 256`.
#[inline]
pub fn bit_cost(prob: u8, bit: bool) -> f32 {
    let t = cost_table();
    if bit { t[256 - prob as usize % 256] } else { t[prob as usize] }
}

pub type Probs = TokenProbs;

/// Receives the binary decisions of a token stream.
pub trait TokenSink {
    /// A decision whose probability comes from the adaptive tables: (plane, band, context, node).
    fn tree(&mut self, plane: usize, band: usize, ctx: usize, node: usize, bit: bool);
    /// A decision with a fixed probability (value bits and signs).
    fn fixed(&mut self, prob: u8, bit: bool);
}

/// Position band for coefficient index `n` (index 16 is a dummy).
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Walk the tokens of one block. `levels` are in zigzag order; coding starts at `first` and `ctx`
/// is the neighbour context (0..=2).
pub fn code_block<S: TokenSink>(sink: &mut S, plane: usize, mut ctx: usize, levels: &[i16; 16], first: usize) {
    // One past the last non-zero level (0 when the block is empty).
    let eob = levels.iter().rposition(|&l| l != 0).map_or(0, |p| p + 1);
    let mut n = first;
    while n < 16 {
        let band = BANDS[n];
        if n >= eob {
            sink.tree(plane, band, ctx, 0, false);
            return;
        }
        sink.tree(plane, band, ctx, 0, true);
        // Zeros up to the next non-zero level; the context after a zero is 0.
        while levels[n] == 0 {
            sink.tree(plane, BANDS[n], ctx, 1, false);
            n += 1;
            ctx = 0;
        }
        let band = BANDS[n];
        sink.tree(plane, band, ctx, 1, true);
        let v = levels[n].unsigned_abs() as u32;
        if v == 1 {
            sink.tree(plane, band, ctx, 2, false);
            ctx = 1;
        } else {
            sink.tree(plane, band, ctx, 2, true);
            large_value(sink, plane, band, ctx, v);
            ctx = 2;
        }
        sink.fixed(128, levels[n] < 0);
        n += 1;
    }
}

fn large_value<S: TokenSink>(sink: &mut S, plane: usize, band: usize, ctx: usize, v: u32) {
    let t = |node: usize, bit: bool, s: &mut S| s.tree(plane, band, ctx, node, bit);
    if v <= 4 {
        t(3, false, sink);
        if v == 2 {
            t(4, false, sink);
        } else {
            t(4, true, sink);
            t(5, v == 4, sink);
        }
        return;
    }
    t(3, true, sink);
    if v <= 10 {
        t(6, false, sink);
        if v <= 6 {
            t(7, false, sink);
            sink.fixed(159, v == 6);
        } else {
            t(7, true, sink);
            sink.fixed(165, (v - 7) >> 1 == 1);
            sink.fixed(145, (v - 7) & 1 == 1);
        }
        return;
    }
    t(6, true, sink);
    // Categories 3..=6 cover 11..=18, 19..=34, 35..=66, 67 and up.
    let cat = if v <= 18 {
        0
    } else if v <= 34 {
        1
    } else if v <= 66 {
        2
    } else {
        3
    };
    t(8, cat >> 1 == 1, sink);
    t(9 + (cat >> 1), cat & 1 == 1, sink);
    let probs = &PROB_DCT_CAT[2 + cat];
    let extra = v - (3 + (8 << cat));
    let nbits = probs.iter().take_while(|&&p| p != 0).count();
    for (i, &p) in probs.iter().take(nbits).enumerate() {
        sink.fixed(p, (extra >> (nbits - 1 - i)) & 1 == 1);
    }
}

/// Estimates the bits a token stream costs under a set of probabilities.
pub struct CostSink<'a> {
    pub probs: &'a Probs,
    pub bits: f32,
}

impl TokenSink for CostSink<'_> {
    #[inline]
    fn tree(&mut self, plane: usize, band: usize, ctx: usize, node: usize, bit: bool) {
        self.bits += bit_cost(self.probs[plane][band][ctx][node], bit);
    }
    #[inline]
    fn fixed(&mut self, prob: u8, bit: bool) {
        self.bits += bit_cost(prob, bit);
    }
}

/// Counts decisions per adaptive probability: `[plane][band][ctx][node] = (ones, total)`.
pub struct StatSink<'a> {
    pub stats: &'a mut Stats,
}

pub type Stats = [[[[(u32, u32); 11]; 3]; 8]; 4];

pub fn new_stats() -> Box<Stats> {
    Box::new([[[[(0, 0); 11]; 3]; 8]; 4])
}

impl TokenSink for StatSink<'_> {
    #[inline]
    fn tree(&mut self, plane: usize, band: usize, ctx: usize, node: usize, bit: bool) {
        let s = &mut self.stats[plane][band][ctx][node];
        s.0 += bit as u32;
        s.1 += 1;
    }
    fn fixed(&mut self, _prob: u8, _bit: bool) {}
}

/// Writes the decisions with the final probabilities.
pub struct WriteSink<'a> {
    pub probs: &'a Probs,
    pub enc: &'a mut BoolEncoder,
}

impl TokenSink for WriteSink<'_> {
    #[inline]
    fn tree(&mut self, plane: usize, band: usize, ctx: usize, node: usize, bit: bool) {
        self.enc.put(self.probs[plane][band][ctx][node], bit);
    }
    #[inline]
    fn fixed(&mut self, prob: u8, bit: bool) {
        self.enc.put(prob, bit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp8::boolcoder::BoolDecoder;

    /// Decode a block exactly as the decoder does, from first principles, to check the tokenizer.
    fn read_block(br: &mut BoolDecoder, probs: &Probs, plane: usize, mut ctx: usize, first: usize) -> [i16; 16] {
        let mut out = [0i16; 16];
        let mut n = first;
        let p = |n: usize, ctx: usize| &probs[plane][BANDS[n]][ctx];
        let mut pr = p(n, ctx);
        while n < 16 {
            if !br.get(pr[0]) {
                return out;
            }
            while !br.get(pr[1]) {
                n += 1;
                pr = p(n, 0);
                if n == 16 {
                    return out;
                }
            }
            let next = |c: usize| p(n + 1, c);
            let v;
            if !br.get(pr[2]) {
                v = 1;
                ctx = 1;
            } else {
                v = if !br.get(pr[3]) {
                    if !br.get(pr[4]) { 2 } else { 3 + br.get(pr[5]) as i32 }
                } else if !br.get(pr[6]) {
                    if !br.get(pr[7]) {
                        5 + br.get(159) as i32
                    } else {
                        let hi = br.get(165) as i32;
                        7 + 2 * hi + br.get(145) as i32
                    }
                } else {
                    let b1 = br.get(pr[8]) as usize;
                    let b0 = br.get(pr[9 + b1]) as usize;
                    let cat = 2 * b1 + b0;
                    let mut x = 0;
                    for &q in PROB_DCT_CAT[2 + cat].iter().take_while(|&&q| q != 0) {
                        x = 2 * x + br.get(q) as i32;
                    }
                    x + 3 + (8 << cat)
                };
                ctx = 2;
            }
            out[n] = if br.get(128) { -v } else { v } as i16;
            pr = next(ctx);
            n += 1;
        }
        out
    }

    #[test]
    fn tokens_roundtrip_through_the_bool_coder() {
        let mut seed = 5u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 16
        };
        let mut blocks = Vec::new();
        for i in 0..400 {
            let mut b = [0i16; 16];
            let density = 1 + i % 5;
            for v in b.iter_mut() {
                if rnd() % 6 < density as u32 {
                    let mag = match rnd() % 10 {
                        0..=4 => 1,
                        5..=6 => 2 + rnd() % 3,
                        7 => 5 + rnd() % 6,
                        8 => 11 + rnd() % 60,
                        _ => 67 + rnd() % 1980,
                    } as i16;
                    *v = if rnd() % 2 == 0 { mag } else { -mag };
                }
            }
            blocks.push(b);
        }
        let probs = Box::new(DEFAULT_COEFF_PROBS);
        let mut enc = BoolEncoder::new();
        let mut cost = CostSink { probs: &probs, bits: 0.0 };
        for (i, b) in blocks.iter().enumerate() {
            let (plane, first, ctx) = (i % 4, i % 2, i % 3);
            code_block(&mut WriteSink { probs: &probs, enc: &mut enc }, plane, ctx, b, first);
            code_block(&mut cost, plane, ctx, b, first);
        }
        let bytes = enc.finish();
        // The cost estimate must be close to what was really written.
        let real = bytes.len() as f32 * 8.0;
        assert!((cost.bits - real).abs() < real * 0.04 + 40.0, "estimated {} bits, wrote {}", cost.bits, real);
        let mut br = BoolDecoder::new(&bytes);
        for (i, b) in blocks.iter().enumerate() {
            let (plane, first, ctx) = (i % 4, i % 2, i % 3);
            let mut want = *b;
            for v in want.iter_mut().take(first) {
                *v = 0; // levels before `first` are never coded
            }
            assert_eq!(read_block(&mut br, &probs, plane, ctx, first), want, "block {i}");
        }
    }
}
