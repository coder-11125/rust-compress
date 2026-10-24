//! Intra block copy (`allow_intrabc`): a block predicts from already decoded samples of its own frame, displaced by
//! an integer vector (a *displacement vector*, stored like a motion vector in 1/8 sample units).
//!
//! This holds the displacement vector candidate stack (the single-reference case of the motion vector
//! prediction process, specification section 7.10.2), the vector syntax, and the block copy prediction
//! (7.11.3 with the current frame as the reference). The syntax is written generically over [`Coder`] so the
//! encoder uses the same code to write what it chose.

use super::cdf_inter::SUBPEL_FILTERS;
use super::consts::*;
use super::entropy::Coder;
use super::mvpred::Mv;
use super::tile::{Blk, TileCoder};

pub const INTRABC_DELAY_PIXELS: i32 = 256;
pub const INTRABC_DELAY_SB64: i32 = 4;

/// The interpolation filter index of `BILINEAR` in [`SUBPEL_FILTERS`].
const BILINEAR: usize = 3;

impl<C: Coder> TileCoder<'_, C> {
    /// `assign_mv(0)` for an intra block copy block: choose the predicted vector, then code the difference.
    /// The encoder arrives with `b.mv[0]` set to the vector it wants.
    pub(crate) fn assign_intrabc_mv(&mut self, b: &mut Blk) {
        let st = self.find_mv_stack(b, [INTRA_FRAME, NONE_FRAME], false);
        let mut pred = st.mv[0][0];
        if pred == [0, 0] {
            pred = st.mv[1][0];
        }
        if pred == [0, 0] {
            let sb4: i32 = if self.f.seq.use_128x128_superblock { 32 } else { 16 };
            if (b.mi_row as i32) - sb4 < self.mi_row_start as i32 {
                pred = [0, -(sb4 * 4 + INTRABC_DELAY_PIXELS) * 8];
            } else {
                pred = [-(sb4 * 4 * 8), 0];
            }
        }
        let want = [b.mv[0][0] - pred[0], b.mv[0][1] - pred[1]];
        let diff = self.read_mv(want);
        b.mv[0] = [pred[0] + diff[0], pred[1] + diff[1]];
        b.mv[1] = [0, 0];
    }

    /// `read_mv` with `MvCtx = MV_INTRABC_CONTEXT`: the vector difference (row, column).
    fn read_mv(&mut self, want: Mv) -> Mv {
        self.read_mv_ctx(1, want)
    }

    /// `read_mv`'s syntax for motion vector context `CTX` (0 regular, 1 intra block copy): the vector difference.
    pub(crate) fn read_mv_ctx(&mut self, ctx: usize, want: Mv) -> Mv {
        let joint_want = match (want[0] != 0, want[1] != 0) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (true, true) => 3,
        };
        let joint = self.c.symbol(&mut self.cdf.mv_joints[ctx], joint_want);
        let mut diff = [0, 0];
        if joint == 2 || joint == 3 {
            diff[0] = self.read_mv_component(ctx, 0, want[0]);
        }
        if joint == 1 || joint == 3 {
            diff[1] = self.read_mv_component(ctx, 1, want[1]);
        }
        diff
    }

    /// One vector component (1/8 sample units). With `force_integer_mv` the fraction bits are implied
    /// (`mv_fr = 3`, `mv_hp = 1`), and so is `mv_hp` without `allow_high_precision_mv`.
    fn read_mv_component(&mut self, ctx: usize, comp: usize, want: i32) -> i32 {
        let force_integer = self.f.hdr.force_integer_mv;
        let hp = self.f.hdr.allow_high_precision_mv;
        // Encoder: derive the symbols from the wanted value. With v = |want| - 1, class 0 holds v < 16 and class c
        // holds [8 << c, 16 << c); the low bits are the integer part's remainder, the fraction and the high-precision bit.
        let (want_sign, want_class, want_c0_bit, want_d, want_fr, want_hp) = if C::DECODING {
            (0, 0, 0, 0, 0, 0)
        } else {
            let v = want.unsigned_abs() as i32 - 1;
            debug_assert!(v >= 0, "zero vector component cannot be coded");
            debug_assert!(!force_integer || v & 7 == 7, "vector difference {want} is not an integer");
            debug_assert!(hp || v & 1 == 1, "vector difference {want} needs high precision");
            let class = if v < 16 { 0 } else { v.ilog2() as usize - 3 };
            let d = if class == 0 { 0 } else { (v - (8 << class)) >> 3 };
            ((want < 0) as usize, class, (v >> 3).clamp(0, 1) as usize, d, ((v >> 1) & 3) as usize, (v & 1) as usize)
        };
        let sign = self.c.symbol(&mut self.cdf.mv_sign[ctx][comp], want_sign);
        let class = self.c.symbol(&mut self.cdf.mv_class[ctx][comp], want_class);
        let mag = if class == 0 {
            let bit = self.c.symbol(&mut self.cdf.mv_class0[ctx][comp], want_c0_bit);
            let fr = if force_integer { 3 } else { self.c.symbol(&mut self.cdf.mv_class0_fp[ctx][comp][bit], want_fr) };
            let hpv = if hp { self.c.symbol(&mut self.cdf.mv_class0_hp[ctx][comp], want_hp) } else { 1 };
            ((bit << 3) | (fr << 1) | hpv) as i32 + 1
        } else {
            let mut d = 0usize;
            for i in 0..class {
                let bit = self.c.symbol(&mut self.cdf.mv_bits[ctx][comp][i], ((want_d >> i) & 1) as usize);
                d |= bit << i;
            }
            let fr = if force_integer { 3 } else { self.c.symbol(&mut self.cdf.mv_fp[ctx][comp], want_fr) };
            let hpv = if hp { self.c.symbol(&mut self.cdf.mv_hp[ctx][comp], want_hp) } else { 1 };
            (2i32 << (class + 2)) + (((d << 3) | (fr << 1) | hpv) as i32 + 1)
        };
        if sign != 0 { -mag } else { mag }
    }

    /// Predict every plane of an intra block copy block from the current frame (7.11.3 with `refIdx = -1`).
    /// `planes` is a bit mask of the planes to predict.
    pub(crate) fn predict_intrabc(&mut self, b: &Blk, planes: u8) {
        let np = if b.has_chroma { 3 } else { 1 };
        for plane in 0..np {
            if (planes >> plane) & 1 == 0 {
                continue;
            }
            let (sx, sy) = self.plane_sub(plane);
            let plane_sz = if plane == 0 { b.bsize } else { subsampled_size(b.bsize, sx as u8, sy as u8) };
            let w = NUM_4X4_WIDE[plane_sz as usize] as usize * 4;
            let h = NUM_4X4_HIGH[plane_sz as usize] as usize * 4;
            let x = (b.mi_col >> sx) * 4;
            let y = (b.mi_row >> sy) * 4;
            self.block_copy(plane, x, y, w, h, b.mv[0], sx, sy);
        }
    }

    /// The block inter prediction process for a bilinear, unscaled reference that is the current frame.
    #[allow(clippy::too_many_arguments)]
    fn block_copy(&mut self, plane: usize, x: usize, y: usize, w: usize, h: usize, mv: Mv, sx: usize, sy: usize) {
        let bd = self.f.seq.color.bit_depth as i32;
        let (round0, round1) = if bd == 12 { (5, 9) } else { (3, 11) };
        let max = (1i32 << bd) - 1;
        // Position of the block's top-left sample in the reference, in 1/16 sample units.
        let px = ((x as i32) << 4) + ((2 * mv[1]) >> sx);
        let py = ((y as i32) << 4) + ((2 * mv[0]) >> sy);
        let (ix, fx) = (px >> 4, (px & 15) as usize);
        let (iy, fy) = (py >> 4, (py & 15) as usize);
        let last_x = (((self.f.mi_cols * 4 + sx) >> sx) - 1) as i32;
        let last_y = (((self.f.mi_rows * 4 + sy) >> sy) - 1) as i32;
        let src = &self.f.planes[plane];
        // Horizontal pass over h + 7 rows, then vertical.
        let kx = &SUBPEL_FILTERS[BILINEAR][fx];
        let ky = &SUBPEL_FILTERS[BILINEAR][fy];
        let mut inter = vec![0i32; (h + 7) * w];
        for r in 0..h + 7 {
            let yy = (iy + r as i32 - 3).clamp(0, last_y) as usize;
            let row = &src.data[yy * src.stride..];
            for c in 0..w {
                let mut s = 0i32;
                for (t, &k) in kx.iter().enumerate() {
                    if k != 0 {
                        let xx = (ix + c as i32 + t as i32 - 3).clamp(0, last_x) as usize;
                        s += k as i32 * row[xx] as i32;
                    }
                }
                inter[r * w + c] = (s + (1 << (round0 - 1))) >> round0;
            }
        }
        let mut out = vec![0u16; w * h];
        for r in 0..h {
            for c in 0..w {
                let mut s = 0i32;
                for (t, &k) in ky.iter().enumerate() {
                    s += k as i32 * inter[(r + t) * w + c];
                }
                out[r * w + c] = ((s + (1 << (round1 - 1))) >> round1).clamp(0, max) as u16;
            }
        }
        let dst = &mut self.f.planes[plane];
        for r in 0..h {
            let o = (y + r) * dst.stride + x;
            dst.data[o..o + w].copy_from_slice(&out[r * w..(r + 1) * w]);
        }
    }
}
