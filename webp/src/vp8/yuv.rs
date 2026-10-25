//! Colour conversion between RGB and the BT.601 limited-range YCbCr that VP8 carries, including
//! the bilinear ("fancy") chroma upsampling that WebP decoders use.

const FIX: i32 = 16;

#[inline(always)]
fn mult_hi(v: i32, coeff: i32) -> i32 {
    (v * coeff) >> 8
}

#[inline(always)]
fn clip8(v: i32) -> u8 {
    // 6 fractional bits.
    if (v & !((256 << 6) - 1)) == 0 {
        (v >> 6) as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

#[inline(always)]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let (y, u, v) = (y as i32, u as i32, v as i32);
    let r = clip8(mult_hi(y, 19077) + mult_hi(v, 26149) - 14234);
    let g = clip8(mult_hi(y, 19077) - mult_hi(u, 6419) - mult_hi(v, 13320) + 8708);
    let b = clip8(mult_hi(y, 19077) + mult_hi(u, 33050) - 17685);
    [r, g, b]
}

/// Forward conversion of one pixel (BT.601, limited range).
#[inline(always)]
pub fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
    ((16839 * r + 33059 * g + 6420 * b + (16 << FIX) + (1 << (FIX - 1))) >> FIX).clamp(0, 255) as u8
}

/// Chroma from sums of four pixels' channels (`r4` = r summed over the 2x2 block, and so on).
#[inline(always)]
pub fn rgb4_to_uv(r4: i32, g4: i32, b4: i32) -> (u8, u8) {
    // The coefficients are for one pixel; with four summed, shift by two more bits.
    let u = (-9719 * r4 - 19081 * g4 + 28800 * b4 + (128 << (FIX + 2)) + (1 << (FIX + 1))) >> (FIX + 2);
    let v = (28800 * r4 - 24116 * g4 - 4684 * b4 + (128 << (FIX + 2)) + (1 << (FIX + 1))) >> (FIX + 2);
    (u.clamp(0, 255) as u8, v.clamp(0, 255) as u8)
}

/// One output row from chroma rows `top` (weighted 3, or 1 for a bottom-role row) and `cur`.
/// `is_top_role` follows libwebp: rows 2k-1 lean towards chroma row k-1, rows 2k towards row k.
fn upsample_row(y: &[u8], top_u: &[u8], top_v: &[u8], cur_u: &[u8], cur_v: &[u8], top_role: bool, out: &mut [u8]) {
    let len = y.len();
    let last_pair = (len - 1) >> 1;
    let lane = |tu: &[u8], tv: &[u8], cu: &[u8], cv: &[u8], x: usize, diag: Diag| -> (u8, u8) {
        let f = |t: &[u8], c: &[u8]| -> u8 { diag.eval(t, c, x) };
        (f(tu, cu), f(tv, cv))
    };
    // Pixel 0.
    {
        let (u, v) = lane(top_u, top_v, cur_u, cur_v, 0, Diag::Edge { top_role, last: false });
        out[0..3].copy_from_slice(&yuv_to_rgb(y[0], u, v));
    }
    for x in 1..=last_pair {
        for odd in [true, false] {
            let px = if odd { 2 * x - 1 } else { 2 * x };
            let (u, v) = lane(top_u, top_v, cur_u, cur_v, x, Diag::Inner { top_role, odd });
            out[3 * px..3 * px + 3].copy_from_slice(&yuv_to_rgb(y[px], u, v));
        }
    }
    if len & 1 == 0 {
        let (u, v) = lane(top_u, top_v, cur_u, cur_v, last_pair, Diag::Edge { top_role, last: true });
        out[3 * (len - 1)..3 * len].copy_from_slice(&yuv_to_rgb(y[len - 1], u, v));
    }
}

#[derive(Clone, Copy)]
enum Diag {
    /// First or last pixel of the row (only one horizontal neighbour).
    Edge { top_role: bool, last: bool },
    /// Interior pixel between chroma columns x-1 and x.
    Inner { top_role: bool, odd: bool },
}

impl Diag {
    fn eval(self, t: &[u8], c: &[u8], x: usize) -> u8 {
        match self {
            Diag::Edge { top_role, last } => {
                let i = if last { x } else { 0 };
                let (near, far) = if top_role { (t[i] as i32, c[i] as i32) } else { (c[i] as i32, t[i] as i32) };
                ((3 * near + far + 2) >> 2) as u8
            }
            Diag::Inner { top_role, odd } => {
                let (tl, tt, l, uv) = (t[x - 1] as i32, t[x] as i32, c[x - 1] as i32, c[x] as i32);
                let avg = tl + tt + l + uv + 8;
                let diag12 = (avg + 2 * (tt + l)) >> 3;
                let diag03 = (avg + 2 * (tl + uv)) >> 3;
                let v = match (top_role, odd) {
                    (true, true) => (diag12 + tl) >> 1,
                    (true, false) => (diag03 + tt) >> 1,
                    (false, true) => (diag03 + l) >> 1,
                    (false, false) => (diag12 + uv) >> 1,
                };
                v as u8
            }
        }
    }
}

/// Convert planar 4:2:0 to interleaved RGB with bilinear chroma upsampling.
pub fn yuv420_to_rgb(
    width: usize,
    height: usize,
    y: &[u8],
    y_stride: usize,
    u: &[u8],
    v: &[u8],
    uv_stride: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 3];
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let crow = |plane: &'_ [u8], r: usize| -> usize {
        let _ = plane;
        r.min(ch - 1) * uv_stride
    };
    for r in 0..height {
        let yrow = &y[r * y_stride..r * y_stride + width];
        // Which chroma rows blend, and whether this row leans to the upper one.
        let (top_c, cur_c, top_role) = if r == 0 {
            (0, 0, true)
        } else if r % 2 == 1 {
            let k = r.div_ceil(2);
            (k - 1, k.min(ch - 1), true)
        } else {
            (r / 2 - 1, r / 2, false)
        };
        let (to, co) = (crow(u, top_c), crow(u, cur_c));
        upsample_row(
            yrow,
            &u[to..to + cw],
            &v[to..to + cw],
            &u[co..co + cw],
            &v[co..co + cw],
            top_role,
            &mut out[r * width * 3..(r + 1) * width * 3],
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_stays_gray() {
        for v in [0u8, 16, 100, 128, 200, 235, 255] {
            let [r, g, b] = yuv_to_rgb(v, 128, 128);
            assert!((r as i32 - g as i32).abs() <= 1 && (g as i32 - b as i32).abs() <= 1, "{v}: {r},{g},{b}");
        }
        assert_eq!(yuv_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(yuv_to_rgb(235, 128, 128), [255, 255, 255]);
    }

    #[test]
    fn forward_and_inverse_are_close() {
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (12, 200, 90), (255, 255, 255), (0, 0, 0)] {
            let y = rgb_to_y(r, g, b);
            let (u, v) = rgb4_to_uv(4 * r, 4 * g, 4 * b);
            let [r2, g2, b2] = yuv_to_rgb(y, u, v);
            for (a, c) in [(r, r2), (g, g2), (b, b2)] {
                assert!((a - c as i32).abs() <= 3, "({r},{g},{b}) -> ({r2},{g2},{b2})");
            }
        }
    }

    #[test]
    fn flat_chroma_upsamples_to_flat_colour() {
        let (w, h) = (7usize, 5usize);
        let y = vec![120u8; w * h];
        let (u, v) = (vec![90u8; 4 * 3], vec![170u8; 4 * 3]);
        let rgb = yuv420_to_rgb(w, h, &y, w, &u, &v, 4);
        let first = &rgb[..3];
        assert!(rgb.chunks_exact(3).all(|p| p == first));
    }
}
