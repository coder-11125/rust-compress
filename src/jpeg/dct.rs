//! 8x8 forward and inverse DCT using the AAN factorization (float).
//!
//! Each 1-D pass is written over all eight lanes at once so the compiler can vectorize it.

pub type Block = [[f32; 8]; 8];

/// AAN per-frequency scale factors.
pub const AAN: [f32; 8] = [1.0, 1.387_039_8, 1.306_563, 1.175_875_6, 1.0, 0.785_694_95, 0.541_196_1, 0.275_899_38];

fn transpose(b: &mut Block) {
    for i in 0..8 {
        for j in i + 1..8 {
            let t = b[i][j];
            b[i][j] = b[j][i];
            b[j][i] = t;
        }
    }
}

#[inline(always)]
fn fdct_pass(d: &mut Block) {
    for j in 0..8 {
        let (d0, d1, d2, d3, d4, d5, d6, d7) = (d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j], d[6][j], d[7][j]);
        let tmp0 = d0 + d7;
        let tmp7 = d0 - d7;
        let tmp1 = d1 + d6;
        let tmp6 = d1 - d6;
        let tmp2 = d2 + d5;
        let tmp5 = d2 - d5;
        let tmp3 = d3 + d4;
        let tmp4 = d3 - d4;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        d[0][j] = tmp10 + tmp11;
        d[4][j] = tmp10 - tmp11;
        let z1 = (tmp12 + tmp13) * 0.707_106_77;
        d[2][j] = tmp13 + z1;
        d[6][j] = tmp13 - z1;

        let tmp10 = tmp4 + tmp5;
        let tmp11 = tmp5 + tmp6;
        let tmp12 = tmp6 + tmp7;
        let z5 = (tmp10 - tmp12) * 0.382_683_43;
        let z2 = 0.541_196_1 * tmp10 + z5;
        let z4 = 1.306_563 * tmp12 + z5;
        let z3 = tmp11 * 0.707_106_77;
        let z11 = tmp7 + z3;
        let z13 = tmp7 - z3;
        d[5][j] = z13 + z2;
        d[3][j] = z13 - z2;
        d[1][j] = z11 + z4;
        d[7][j] = z11 - z4;
    }
}

/// Forward DCT. Input: level-shifted samples `b[y][x]`. Output: `b[u][v]` (u = horizontal,
/// v = vertical frequency), scaled by `8 * AAN[u] * AAN[v]` relative to the orthonormal DCT.
pub fn fdct(b: &mut Block) {
    fdct_pass(b);
    transpose(b);
    fdct_pass(b);
}

#[inline(always)]
fn idct_pass(d: &mut Block) {
    for j in 0..8 {
        let tmp0 = d[0][j];
        let tmp1 = d[2][j];
        let tmp2 = d[4][j];
        let tmp3 = d[6][j];
        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;
        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * std::f32::consts::SQRT_2 - tmp13;
        let tmp0 = tmp10 + tmp13;
        let tmp3 = tmp10 - tmp13;
        let tmp1 = tmp11 + tmp12;
        let tmp2 = tmp11 - tmp12;

        let tmp4 = d[1][j];
        let tmp5 = d[3][j];
        let tmp6 = d[5][j];
        let tmp7 = d[7][j];
        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;
        let tmp7 = z11 + z13;
        let tmp11 = (z11 - z13) * std::f32::consts::SQRT_2;
        let z5 = (z10 + z12) * 1.847_759_1;
        let tmp10 = 1.082_392_2 * z12 - z5;
        let tmp12 = -2.613_126 * z10 + z5;
        let tmp6 = tmp12 - tmp7;
        let tmp5 = tmp11 - tmp6;
        let tmp4 = tmp10 + tmp5;

        d[0][j] = tmp0 + tmp7;
        d[7][j] = tmp0 - tmp7;
        d[1][j] = tmp1 + tmp6;
        d[6][j] = tmp1 - tmp6;
        d[2][j] = tmp2 + tmp5;
        d[5][j] = tmp2 - tmp5;
        d[4][j] = tmp3 + tmp4;
        d[3][j] = tmp3 - tmp4;
    }
}

/// Inverse DCT. Input `b[v][u]` must be dequantized and multiplied by `AAN[v] * AAN[u]`.
/// Output `b[y][x]` is the spatial block, still level-shifted and unrounded (already divided by 8).
pub fn idct(b: &mut Block) {
    idct_pass(b);
    transpose(b);
    idct_pass(b);
    transpose(b);
    for row in b.iter_mut() {
        for v in row.iter_mut() {
            *v *= 0.125;
        }
    }
}

/// Round to nearest integer (ties to even) without a libm call; valid for |x| < 2^22.
#[inline(always)]
pub fn round_f32(x: f32) -> i32 {
    ((x + 12_582_912.0).to_bits() as i32) - 0x4B40_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dct(b: &Block) -> Block {
        // out[u][v]: u horizontal, v vertical.
        let mut out = [[0f32; 8]; 8];
        for (u, row) in out.iter_mut().enumerate() {
            for (v, o) in row.iter_mut().enumerate() {
                let mut s = 0f64;
                for y in 0..8 {
                    for x in 0..8 {
                        s += b[y][x] as f64
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                let a = |k: usize| if k == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                *o = (0.25 * a(u) * a(v) * s) as f32;
            }
        }
        out
    }

    fn random_block(seed: u32) -> Block {
        let mut s = seed;
        let mut b = [[0f32; 8]; 8];
        for row in b.iter_mut() {
            for v in row.iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = ((s >> 24) as i32 - 128) as f32;
            }
        }
        b
    }

    #[test]
    fn fdct_matches_naive_dct() {
        for seed in 0..20 {
            let b = random_block(seed);
            let want = naive_dct(&b);
            let mut got = b;
            fdct(&mut got);
            for u in 0..8 {
                for v in 0..8 {
                    let g = got[u][v] / (8.0 * AAN[u] * AAN[v]);
                    assert!((g - want[u][v]).abs() < 1e-2, "seed {seed} ({u},{v}): {g} vs {}", want[u][v]);
                }
            }
        }
    }

    #[test]
    fn idct_inverts_dct() {
        for seed in 0..20 {
            let b = random_block(seed + 100);
            let c = naive_dct(&b); // c[u][v]
            // Build natural-order input b[v][u], premultiplied by AAN.
            let mut inp = [[0f32; 8]; 8];
            for v in 0..8 {
                for u in 0..8 {
                    inp[v][u] = c[u][v] * AAN[u] * AAN[v];
                }
            }
            idct(&mut inp);
            for y in 0..8 {
                for x in 0..8 {
                    assert!((inp[y][x] - b[y][x]).abs() < 1e-2, "seed {seed} ({y},{x})");
                }
            }
        }
    }

    #[test]
    fn rounding() {
        assert_eq!(round_f32(2.4), 2);
        assert_eq!(round_f32(2.6), 3);
        assert_eq!(round_f32(-2.6), -3);
        assert_eq!(round_f32(-0.4), 0);
    }
}
