//! Super-resolution upscaling (AV1 specification section 7.16).
//!
//! A frame coded at a reduced width (`SUPERRES_DENOM / 8` of the real one) is stretched back horizontally with
//! an 8-tap, 64-phase filter. Every row is processed independently; the position arithmetic uses 14 fractional
//! bits, with the phase taken from bits 8..14 of the position.

use super::frame::{Frame, Plane};

const SUPERRES_SCALE_BITS: u32 = 14;
const SUPERRES_SCALE_MASK: i64 = (1 << SUPERRES_SCALE_BITS) - 1;
const SUPERRES_EXTRA_BITS: u32 = 8;
const SUPERRES_FILTER_OFFSET: i64 = 3;
const FILTER_BITS: u32 = 7;

/// `Upscale_Filter` from the specification: 64 phases of 8 taps, each summing to 128.
const UPSCALE_FILTER: [[i32; 8]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// Upscales a frame decoded at reduced width back to full width.
///
/// * `src.width` is the downscaled `FrameWidth`; its planes are `MiCols * 4 >> sub` samples wide (the coded
///   area). Samples beyond the last coded column are replaced by the last column, as in the specification.
/// * `dst_width` is the real (visible) upscaled luma width, the header's `UpscaledWidth`.
/// * `denom` is `SUPERRES_DENOM` (9..=16); it only documents the ratio, since the position step is derived from
///   the two widths exactly as the specification does.
///
/// The result has `width == dst_width`, the same height and bit depth, and coded planes sized for the new
/// width (`MiCols * 4 >> sub`, with `MiCols = 2 * ((width + 7) >> 3)`). Columns past the visible width are
/// computed with the same formula and carry no meaning.
pub fn upscale_frame(src: &Frame, dst_width: usize, denom: u8) -> Frame {
    debug_assert!((9..=16).contains(&denom));
    let _ = denom;
    let max_val = (1i32 << src.bit_depth) - 1;
    let coded_luma = (2 * dst_width.div_ceil(8)) * 4;
    let mut planes = Vec::with_capacity(src.planes.len());
    for (index, sp) in src.planes.iter().enumerate() {
        let (sx, sy) = if index == 0 { (0, 0) } else { (src.sub_x as usize, src.sub_y as usize) };
        let _ = sy;
        let down_w = ((src.width + sx) >> sx) as i64;
        let up_w = ((dst_width + sx) >> sx) as i64;
        let step = ((down_w << SUPERRES_SCALE_BITS) + up_w / 2) / up_w;
        let err = up_w * step - (down_w << SUPERRES_SCALE_BITS);
        let initial = ((-((up_w - down_w) << (SUPERRES_SCALE_BITS - 1)) + up_w / 2) / up_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2)
            & SUPERRES_SCALE_MASK;
        let out_w = coded_luma >> sx;
        let last_x = sp.width as i64 - 1;
        let mut out = Plane::new(out_w, sp.height);
        for y in 0..sp.height {
            let src_row = sp.row(y);
            let dst_row = out.row_mut(y);
            for (x, d) in dst_row.iter_mut().enumerate() {
                let pos = -(1i64 << SUPERRES_SCALE_BITS) + initial + x as i64 * step;
                let px = pos >> SUPERRES_SCALE_BITS;
                let phase = ((pos & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS) as usize;
                let taps = &UPSCALE_FILTER[phase];
                let mut sum = 0i32;
                for (k, &tap) in taps.iter().enumerate() {
                    let sample_x = (px + k as i64 - SUPERRES_FILTER_OFFSET).clamp(0, last_x) as usize;
                    sum += tap * src_row[sample_x] as i32;
                }
                *d = ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, max_val) as u16;
            }
        }
        planes.push(out);
    }
    Frame { planes, bit_depth: src.bit_depth, sub_x: src.sub_x, sub_y: src.sub_y, width: dst_width, height: src.height }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
    }

    fn random_frame(rng: &mut Rng, width: usize, height: usize, bd: u8, sx: u8, sy: u8, mono: bool) -> Frame {
        let coded_w = 2 * width.div_ceil(8) * 4;
        let coded_h = height.div_ceil(4) * 4;
        let mut planes = Vec::new();
        for p in 0..if mono { 1 } else { 3 } {
            let (px, py) = if p == 0 { (0, 0) } else { (sx, sy) };
            let mut pl = Plane::new(coded_w >> px, coded_h >> py);
            for v in pl.data.iter_mut() {
                *v = (rng.next() & ((1 << bd) - 1)) as u16;
            }
            planes.push(pl);
        }
        Frame { planes, bit_depth: bd, sub_x: sx, sub_y: sy, width, height }
    }

    /// The incremental formulation used by dav1d, as an independent check of the closed-form positions.
    fn reference(src: &Plane, down_w: i32, up_w: i32, out_w: usize, bd: u8) -> Plane {
        let step = ((down_w << 14) + (up_w >> 1)) / up_w;
        let err = up_w * step - (down_w << 14);
        let mut mx0 = (-((up_w - down_w) << 13) + (up_w >> 1)) / up_w + 128 - err / 2;
        mx0 &= 0x3fff;
        let max = src.width as i32 - 1;
        let mut out = Plane::new(out_w, src.height);
        for y in 0..src.height {
            let mut mx = mx0;
            let mut sx = -1 - 3;
            for x in 0..out_w {
                let f = &UPSCALE_FILTER[(mx >> 8) as usize];
                let mut sum = 0;
                for i in 0..8 {
                    sum += f[i] * src.at((sx + i as i32).clamp(0, max) as usize, y) as i32;
                }
                out.data[y * out_w + x] = ((sum + 64) >> 7).clamp(0, (1 << bd) - 1) as u16;
                mx += step;
                sx += mx >> 14;
                mx &= 0x3fff;
            }
        }
        out
    }

    #[test]
    fn filter_rows_sum_to_128_and_mirror() {
        for (i, row) in UPSCALE_FILTER.iter().enumerate() {
            assert_eq!(row.iter().sum::<i32>(), 128, "phase {i}");
            if i > 0 {
                let mirror = UPSCALE_FILTER[64 - i];
                for k in 0..8 {
                    assert_eq!(row[k], mirror[7 - k], "phase {i} tap {k}");
                }
            }
        }
    }

    #[test]
    fn matches_incremental_reference() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut cases = 0;
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..300 {
            let denom = 9 + (rng.next() % 8) as usize;
            let up = 16 + (rng.next() % 200) as usize;
            // FrameWidth = (UpscaledWidth * 8 + denom / 2) / denom
            let down = ((up * 8 + denom / 2) / denom).max(1);
            let bd = [8u8, 10, 12][(rng.next() % 3) as usize];
            let (sx, sy) = [(1, 1), (1, 0), (0, 0)][(rng.next() % 3) as usize];
            let h = 1 + (rng.next() % 9) as usize;
            let mono = rng.next().is_multiple_of(5);
            let src = random_frame(&mut rng, down, h, bd, sx, sy, mono);
            let out = upscale_frame(&src, up, denom as u8);
            assert_eq!(out.width, up);
            for (i, p) in src.planes.iter().enumerate() {
                let s = if i == 0 { 0 } else { sx };
                let want = reference(
                    p,
                    ((down + s as usize) >> s) as i32,
                    ((up + s as usize) >> s) as i32,
                    out.planes[i].width,
                    bd,
                );
                let vis = (up + s as usize) >> s;
                for y in 0..p.height {
                    assert_eq!(out.planes[i].row(y)[..vis], want.row(y)[..vis], "case {cases} plane {i} row {y}");
                }
            }
            cases += 1;
        }
    }

    #[test]
    fn flat_picture_stays_flat() {
        let mut rng = Rng(7);
        let mut src = random_frame(&mut rng, 40, 4, 10, 1, 1, false);
        for p in src.planes.iter_mut() {
            p.data.fill(777);
        }
        let out = upscale_frame(&src, 80, 16);
        assert!(out.planes.iter().all(|p| p.data.iter().all(|&v| v == 777)));
    }

    #[test]
    fn golden_row() {
        // 8 coded luma samples upscaled 2x (denom 16): a ramp stays monotonic and endpoints clamp.
        let mut p = Plane::new(8, 1);
        p.data.copy_from_slice(&[0, 100, 200, 300, 400, 500, 600, 700]);
        let src = Frame { planes: vec![p], bit_depth: 10, sub_x: 1, sub_y: 1, width: 8, height: 1 };
        let out = upscale_frame(&src, 16, 16);
        let row = out.planes[0].row(0);
        assert_eq!(out.planes[0].width, 16);
        assert!(row[..16].windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(row[0], 0);
    }
}
