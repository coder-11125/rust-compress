//! Trellis (rate-distortion optimized) quantization of AC coefficients.
//!
//! For each block, choose the quantized values minimizing `distortion + lambda * bits`, where the
//! bit cost of every (zero-run, size) symbol comes from Huffman code lengths measured on a first,
//! plainly quantized pass. Candidates per coefficient are the rounded value and one step toward
//! zero; zeroing is expressed by skipping a position. Dynamic programming over the candidate
//! positions finds the best run-length structure, including where to place the end-of-block.

/// Parameters for one component class (luma or chroma).
pub struct Rdo {
    /// Quantization step per zigzag position.
    q: [f32; 64],
    lambda: f32,
    /// Estimated code length in bits of each AC symbol (`run << 4 | size`).
    len: [f32; 256],
}

impl Rdo {
    /// `table` is the quantization table in natural order, `kappa` scales lambda relative to the
    /// squared median step, `sizes` are Huffman code lengths from a first pass (0 = unused symbol).
    pub fn new(table: &[u8; 64], kappa: f32, sizes: &[u8; 256]) -> Self {
        let mut q = [1f32; 64];
        for (k, &n) in super::ZIGZAG.iter().enumerate() {
            q[k] = table[n] as f32;
        }
        let mut sorted: Vec<f32> = q[1..].to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        let mut len = [0f32; 256];
        for (l, &s) in len.iter_mut().zip(sizes) {
            *l = if s == 0 { 18.0 } else { s as f32 };
        }
        Rdo { q, lambda: kappa * median * median, len }
    }

    #[inline]
    fn rate(&self, run: usize, v: u32) -> f32 {
        let size = 32 - v.leading_zeros();
        let zrl = run / 16;
        zrl as f32 * self.len[0xF0] + self.len[((run % 16) << 4) | size as usize] + size as f32
    }

    /// Quantize positions 1..64 of `a` (signed values in units of quantization steps).
    /// Position 0 (DC) is left to the caller.
    pub fn quantize_ac(&self, a: &[f32; 64], out: &mut [i16]) {
        struct Node {
            pos: usize,
            v: u32,
            cost: f32,
            prev: usize,
        }
        const START: usize = usize::MAX;
        // Cumulative distortion if everything up to k is zeroed.
        let mut zd = [0f32; 64];
        for k in 1..64 {
            let e = a[k] * self.q[k];
            zd[k] = zd[k - 1] + e * e;
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(48);
        for i in 1..64 {
            let m = a[i].abs();
            let v0 = (m + 0.5) as u32;
            if v0 == 0 {
                continue;
            }
            let lo = if v0 > 1 { v0 - 1 } else { v0 };
            for v in lo..=v0 {
                let e = (m - v as f32) * self.q[i];
                let dnz = e * e;
                // Predecessor = start of block (everything before i zeroed) ...
                let mut best = zd[i - 1] + self.lambda * self.rate(i - 1, v) + dnz;
                let mut prev = START;
                // ... or an earlier nonzero candidate.
                for (ni, n) in nodes.iter().enumerate() {
                    if n.pos >= i {
                        continue; // the other candidate value for this same position
                    }
                    let c = n.cost + (zd[i - 1] - zd[n.pos]) + self.lambda * self.rate(i - n.pos - 1, v) + dnz;
                    if c < best {
                        best = c;
                        prev = ni;
                    }
                }
                nodes.push(Node { pos: i, v, cost: best, prev });
            }
        }
        // Choose where the block ends (after the last kept coefficient comes the EOB symbol).
        let eob = self.lambda * self.len[0];
        let mut best_cost = zd[63] + eob;
        let mut best_end = START;
        for (ni, n) in nodes.iter().enumerate() {
            let c = n.cost + (zd[63] - zd[n.pos]) + if n.pos < 63 { eob } else { 0.0 };
            if c < best_cost {
                best_cost = c;
                best_end = ni;
            }
        }
        for o in out[1..64].iter_mut() {
            *o = 0;
        }
        let mut cur = best_end;
        while cur != START {
            let n = &nodes[cur];
            let v = n.v as i16;
            out[n.pos] = if a[n.pos] < 0.0 { -v } else { v };
            cur = n.prev;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_sizes() -> [u8; 256] {
        let mut s = [0u8; 256];
        for (i, v) in s.iter_mut().enumerate() {
            *v = 4 + (i >> 4).min(8) as u8 + (i & 15).min(4) as u8;
        }
        s[0] = 3;
        s[0xF0] = 8;
        s
    }

    #[test]
    fn zero_lambda_equals_plain_rounding() {
        let rdo = Rdo::new(&[10u8; 64], 0.0, &flat_sizes());
        let mut a = [0f32; 64];
        for (k, v) in a.iter_mut().enumerate() {
            *v = ((k as f32 * 7.3) % 5.0) - 2.43; // avoid exact .5 ties
        }
        let mut out = [0i16; 64];
        rdo.quantize_ac(&a, &mut out);
        for k in 1..64 {
            let want = (a[k] + 0.5 * a[k].signum()).trunc() as i16;
            assert_eq!(out[k], want, "k={k} a={}", a[k]);
        }
    }

    #[test]
    fn large_lambda_zeroes_marginal_coefficients() {
        let rdo = Rdo::new(&[10u8; 64], 4.0, &flat_sizes());
        let mut a = [0f32; 64];
        a[5] = 0.6; // barely above the rounding threshold: not worth its bits
        a[1] = 9.0; // clearly worth keeping
        let mut out = [0i16; 64];
        rdo.quantize_ac(&a, &mut out);
        assert_eq!(out[5], 0);
        assert!(out[1] >= 8, "{}", out[1]);
    }
}
