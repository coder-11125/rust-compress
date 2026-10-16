//! VP8's 4x4 integer transforms: the inverse DCT and Walsh-Hadamard transform (normative, they
//! define the decoder's output exactly) and forward counterparts for the encoder.

// Valid streams keep every intermediate well inside 32 bits, but damaged ones may not, so the
// products are formed in 64 bits and the result wraps instead of panicking.
#[inline(always)]
fn mul1(a: i32) -> i32 {
    (((a as i64 * 20091) >> 16) as i32).wrapping_add(a)
}

#[inline(always)]
fn mul2(a: i32) -> i32 {
    ((a as i64 * 35468) >> 16) as i32
}

/// Inverse DCT of `coefs` (raster order, already dequantized), added to the 4x4 block at `dst`
/// (`stride` apart) with clamping.
pub fn idct_add(coefs: &[i32; 16], dst: &mut [u8], stride: usize) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a = coefs[i] + coefs[8 + i];
        let b = coefs[i] - coefs[8 + i];
        let c = mul2(coefs[4 + i]) - mul1(coefs[12 + i]);
        let d = mul1(coefs[4 + i]) + mul2(coefs[12 + i]);
        tmp[4 * i] = a + d;
        tmp[4 * i + 1] = b + c;
        tmp[4 * i + 2] = b - c;
        tmp[4 * i + 3] = a - d;
    }
    for i in 0..4 {
        let dc = tmp[i] + 4;
        let a = dc + tmp[8 + i];
        let b = dc - tmp[8 + i];
        let c = mul2(tmp[4 + i]) - mul1(tmp[12 + i]);
        let d = mul1(tmp[4 + i]) + mul2(tmp[12 + i]);
        let row = &mut dst[i * stride..i * stride + 4];
        for (px, v) in row.iter_mut().zip([a + d, b + c, b - c, a - d]) {
            *px = (*px as i32 + (v >> 3)).clamp(0, 255) as u8;
        }
    }
}

/// Inverse Walsh-Hadamard transform of the 16 luma DC coefficients; `out[i]` is the DC of block `i`.
pub fn iwht(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = input[i] + input[12 + i];
        let a1 = input[4 + i] + input[8 + i];
        let a2 = input[4 + i] - input[8 + i];
        let a3 = input[i] - input[12 + i];
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let dc = tmp[4 * i] + 3;
        let a0 = dc + tmp[4 * i + 3];
        let a1 = tmp[4 * i + 1] + tmp[4 * i + 2];
        let a2 = tmp[4 * i + 1] - tmp[4 * i + 2];
        let a3 = dc - tmp[4 * i + 3];
        out[4 * i] = (a0 + a1) >> 3;
        out[4 * i + 1] = (a3 + a2) >> 3;
        out[4 * i + 2] = (a0 - a1) >> 3;
        out[4 * i + 3] = (a3 - a2) >> 3;
    }
    out
}

/// Forward DCT of `src - pred` (both 4x4 blocks with their own strides). Output in raster order.
pub fn fdct(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let s = &src[i * src_stride..i * src_stride + 4];
        let p = &pred[i * pred_stride..i * pred_stride + 4];
        let d: [i32; 4] = std::array::from_fn(|k| s[k] as i32 - p[k] as i32);
        let a0 = d[0] + d[3];
        let a1 = d[1] + d[2];
        let a2 = d[1] - d[2];
        let a3 = d[0] - d[3];
        tmp[4 * i] = (a0 + a1) * 8;
        tmp[4 * i + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[4 * i + 2] = (a0 - a1) * 8;
        tmp[4 * i + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let a0 = tmp[i] + tmp[12 + i];
        let a1 = tmp[4 + i] + tmp[8 + i];
        let a2 = tmp[4 + i] - tmp[8 + i];
        let a3 = tmp[i] - tmp[12 + i];
        out[i] = (a0 + a1 + 7) >> 4;
        out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + i32::from(a3 != 0);
        out[8 + i] = (a0 - a1 + 7) >> 4;
        out[12 + i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16;
    }
    out
}

/// Forward WHT of the 16 luma DC values (`dcs[i]` = DC of block `i`).
pub fn fwht(dcs: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = dcs[4 * i] + dcs[4 * i + 2];
        let a1 = dcs[4 * i + 1] + dcs[4 * i + 3];
        let a2 = dcs[4 * i + 1] - dcs[4 * i + 3];
        let a3 = dcs[4 * i] - dcs[4 * i + 2];
        tmp[4 * i] = a0 + a1;
        tmp[4 * i + 1] = a3 + a2;
        tmp[4 * i + 2] = a3 - a2;
        tmp[4 * i + 3] = a0 - a1;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        out[i] = (a0 + a1) >> 1;
        out[4 + i] = (a3 + a2) >> 1;
        out[8 + i] = (a3 - a2) >> 1;
        out[12 + i] = (a0 - a1) >> 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(seed: u32) -> [u8; 16] {
        let mut s = seed;
        std::array::from_fn(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
    }

    #[test]
    fn dct_roundtrip_reconstructs_within_rounding() {
        for seed in 0..200 {
            let src = block(seed);
            let pred = block(seed + 1000);
            let coefs = fdct(&src, 4, &pred, 4);
            let mut out = pred;
            idct_add(&coefs, &mut out, 4);
            let worst = src.iter().zip(&out).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
            assert!(worst <= 1, "seed {seed}: error {worst}");
        }
    }

    #[test]
    fn dc_only_block_adds_a_constant() {
        let mut dst = [100u8; 16];
        let mut c = [0i32; 16];
        c[0] = 80; // (80 + 4) >> 3 = 10
        idct_add(&c, &mut dst, 4);
        assert!(dst.iter().all(|&v| v == 110));
    }

    #[test]
    fn idct_clamps() {
        let mut dst = [250u8; 16];
        let mut c = [0i32; 16];
        c[0] = 800;
        idct_add(&c, &mut dst, 4);
        assert!(dst.iter().all(|&v| v == 255));
    }

    #[test]
    fn wht_roundtrip_is_nearly_exact() {
        for seed in 0..100 {
            let b = block(seed);
            // DC values are sums over a 4x4 block, so up to ~12 bits.
            let dcs: [i32; 16] = std::array::from_fn(|i| (b[i] as i32 - 128) * 8);
            let back = iwht(&fwht(&dcs));
            for (a, b) in dcs.iter().zip(&back) {
                assert!((a - b).abs() <= 1, "seed {seed}: {a} vs {b}");
            }
        }
    }
}
