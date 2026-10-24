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
use super::tile::{Blk, TileCoder};

pub const MV_BORDER: i32 = 128;
pub const REF_CAT_LEVEL: u32 = 640;
pub const MAX_REF_MV_STACK_SIZE: usize = 8;
pub const INTRABC_DELAY_PIXELS: i32 = 256;
pub const INTRABC_DELAY_SB64: i32 = 4;

/// Reference frame codes as stored per 4x4 block.
pub const INTRA_FRAME: i8 = 0;
pub const NONE_FRAME: i8 = -1;
/// Marks a block whose mode info has not been written yet in this frame.
pub const UNWRITTEN: i8 = -2;

/// The interpolation filter index of `BILINEAR` in [`SUBPEL_FILTERS`].
const BILINEAR: usize = 3;

/// A motion vector as (row, column) in 1/8 sample units.
pub type Mv = [i32; 2];

/// The candidate stack built by the find-MV-stack process (single reference).
#[derive(Default)]
pub struct MvStack {
    pub mv: [Mv; MAX_REF_MV_STACK_SIZE + 1],
    weight: [u32; MAX_REF_MV_STACK_SIZE + 1],
    pub num_found: usize,
    found_match: bool,
}

impl<C: Coder> TileCoder<'_, C> {
    /// `lower_mv_precision` for a frame without high precision vectors.
    fn lower_precision(&self, mv: &mut Mv) {
        if self.f.hdr.allow_high_precision_mv {
            return;
        }
        for v in mv.iter_mut() {
            if self.f.hdr.force_integer_mv {
                let a = v.abs();
                let a_int = (a + 3) >> 3;
                *v = if *v > 0 { a_int << 3 } else { -(a_int << 3) };
            } else if *v & 1 != 0 {
                *v += if *v > 0 { -1 } else { 1 };
            }
        }
    }

    /// The add-reference-motion-vector process for a single reference `ref0` (7.10.2.7 and 7.10.2.8).
    fn add_ref_mv_candidate(&self, st: &mut MvStack, ref0: i8, mv_row: usize, mv_col: usize, weight: u32) {
        let i = self.mi(mv_row, mv_col);
        if self.f.is_inter[i] == 0 {
            return;
        }
        for cand_list in 0..2 {
            if self.f.ref_frame[i][cand_list] != ref0 {
                continue;
            }
            let mut cand = self.f.mvs[i][cand_list];
            self.lower_precision(&mut cand);
            st.found_match = true;
            if let Some(idx) = (0..st.num_found).find(|&k| st.mv[k] == cand) {
                st.weight[idx] += weight;
            } else if st.num_found < MAX_REF_MV_STACK_SIZE {
                st.mv[st.num_found] = cand;
                st.weight[st.num_found] = weight;
                st.num_found += 1;
            }
        }
    }

    fn scan_row(&self, st: &mut MvStack, b: &Blk, ref0: i8, delta_row: isize) {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let end4 = bw4.min(self.f.mi_cols - b.mi_col).min(16);
        let mut delta_col = 0isize;
        let use_step16 = bw4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += (b.mi_row & 1) as isize;
            delta_col = 1 - (b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = b.mi_row as isize + delta_row;
            let mv_col = b.mi_col as isize + delta_col + i as isize;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let (r, c) = (mv_row as usize, mv_col as usize);
            let mut len = bw4.min(NUM_4X4_WIDE[self.f.mi_size[self.mi(r, c)] as usize] as usize);
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, ref0, r, c, len as u32 * 2);
            i += len;
        }
    }

    fn scan_col(&self, st: &mut MvStack, b: &Blk, ref0: i8, delta_col: isize) {
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        let end4 = bh4.min(self.f.mi_rows - b.mi_row).min(16);
        let mut delta_row = 0isize;
        let use_step16 = bh4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (b.mi_row & 1) as isize;
            delta_col += (b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = b.mi_row as isize + delta_row + i as isize;
            let mv_col = b.mi_col as isize + delta_col;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let (r, c) = (mv_row as usize, mv_col as usize);
            let mut len = bh4.min(NUM_4X4_HIGH[self.f.mi_size[self.mi(r, c)] as usize] as usize);
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, ref0, r, c, len as u32 * 2);
            i += len;
        }
    }

    fn scan_point(&self, st: &mut MvStack, b: &Blk, ref0: i8, delta_row: isize, delta_col: isize) {
        let mv_row = b.mi_row as isize + delta_row;
        let mv_col = b.mi_col as isize + delta_col;
        if self.is_inside(mv_row, mv_col) {
            let (r, c) = (mv_row as usize, mv_col as usize);
            if self.f.ref_frame[self.mi(r, c)][0] != UNWRITTEN {
                self.add_ref_mv_candidate(st, ref0, r, c, 4);
            }
        }
    }

    /// The find-MV-stack process (7.10.2) for one reference frame `ref0` and no temporal candidates.
    pub(crate) fn find_mv_stack(&self, b: &Blk, ref0: i8) -> MvStack {
        let mut st = MvStack::default();
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        // Global motion is the zero vector for intra block copy.
        self.scan_row(&mut st, b, ref0, -1);
        let mut found_above = st.found_match;
        st.found_match = false;
        self.scan_col(&mut st, b, ref0, -1);
        let mut found_left = st.found_match;
        st.found_match = false;
        if bw4.max(bh4) <= 16 {
            self.scan_point(&mut st, b, ref0, -1, bw4 as isize);
        }
        if st.found_match {
            found_above = true;
        }
        let num_nearest = st.num_found;
        for w in st.weight.iter_mut().take(num_nearest) {
            *w += REF_CAT_LEVEL;
        }
        self.scan_point(&mut st, b, ref0, -1, -1);
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        self.scan_row(&mut st, b, ref0, -3);
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        self.scan_col(&mut st, b, ref0, -3);
        if st.found_match {
            found_left = true;
        }
        st.found_match = false;
        if bh4 > 1 {
            self.scan_row(&mut st, b, ref0, -5);
        }
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        if bw4 > 1 {
            self.scan_col(&mut st, b, ref0, -5);
        }
        if st.found_match {
            found_left = true;
        }
        let _ = (found_above, found_left);
        sort_stack(&mut st, 0, num_nearest);
        let n = st.num_found;
        sort_stack(&mut st, num_nearest, n);
        // Extra search: neighbours with a different reference frame cannot contribute to intra block copy
        // (their reference is not above INTRA_FRAME), so only the global (zero) filler remains.
        for idx in st.num_found..2 {
            st.mv[idx] = [0, 0];
        }
        // Clamping.
        let (bw, bh) = (bw4 as i32 * 4, bh4 as i32 * 4);
        for idx in 0..st.num_found {
            let mv = &mut st.mv[idx];
            mv[0] = self.clamp_mv_row(b, mv[0], MV_BORDER + bh * 8);
            mv[1] = self.clamp_mv_col(b, mv[1], MV_BORDER + bw * 8);
        }
        st
    }

    fn clamp_mv_row(&self, b: &Blk, v: i32, border: i32) -> i32 {
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as i32;
        let to_top = -((b.mi_row as i32 * 4) * 8);
        let to_bottom = ((self.f.mi_rows as i32 - bh4 - b.mi_row as i32) * 4) * 8;
        v.clamp(to_top - border, to_bottom + border)
    }

    fn clamp_mv_col(&self, b: &Blk, v: i32, border: i32) -> i32 {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as i32;
        let to_left = -((b.mi_col as i32 * 4) * 8);
        let to_right = ((self.f.mi_cols as i32 - bw4 - b.mi_col as i32) * 4) * 8;
        v.clamp(to_left - border, to_right + border)
    }

    /// `assign_mv(0)` for an intra block copy block: choose the predicted vector, then code the difference.
    /// The encoder arrives with `b.mv[0]` set to the vector it wants.
    pub(crate) fn assign_intrabc_mv(&mut self, b: &mut Blk) {
        let st = self.find_mv_stack(b, INTRA_FRAME);
        let mut pred = st.mv[0];
        if pred == [0, 0] {
            pred = st.mv[1];
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
        const CTX: usize = 1;
        let joint_want = match (want[0] != 0, want[1] != 0) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (true, true) => 3,
        };
        let joint = self.c.symbol(&mut self.cdf.mv_joints[CTX], joint_want);
        let mut diff = [0, 0];
        if joint == 2 || joint == 3 {
            diff[0] = self.read_mv_component(CTX, 0, want[0]);
        }
        if joint == 1 || joint == 3 {
            diff[1] = self.read_mv_component(CTX, 1, want[1]);
        }
        diff
    }

    /// One vector component. Intra frames force integer vectors, so the fraction bits are implied
    /// (`mv_fr = 3`, `mv_hp = 1`) and only the sign, class and integer bits are coded.
    fn read_mv_component(&mut self, ctx: usize, comp: usize, want: i32) -> i32 {
        let force_integer = self.f.hdr.force_integer_mv;
        let hp = self.f.hdr.allow_high_precision_mv;
        // Encoder: derive the symbols from the wanted value (in whole samples when integer).
        let (want_sign, want_class, want_c0_bit, want_d) = if C::DECODING {
            (0, 0, 0, 0)
        } else {
            let mag = want.unsigned_abs() as i32;
            debug_assert!(force_integer && !hp && mag % 8 == 0 && mag >= 8, "unsupported vector difference {want}");
            let m = mag / 8;
            let class = if m <= 2 { 0 } else { (m - 1).ilog2() as usize };
            let d = if class == 0 { 0 } else { m - (1 << class) - 1 };
            ((want < 0) as usize, class, (m - 1).clamp(0, 1) as usize, d)
        };
        let sign = self.c.symbol(&mut self.cdf.mv_sign[ctx][comp], want_sign);
        let class = self.c.symbol(&mut self.cdf.mv_class[ctx][comp], want_class);
        let mag = if class == 0 {
            let bit = self.c.symbol(&mut self.cdf.mv_class0[ctx][comp], want_c0_bit);
            let fr = if force_integer { 3 } else { self.c.symbol(&mut self.cdf.mv_class0_fp[ctx][comp][bit], 3) };
            let hpv = if hp { self.c.symbol(&mut self.cdf.mv_class0_hp[ctx][comp], 1) } else { 1 };
            ((bit << 3) | (fr << 1) | hpv) as i32 + 1
        } else {
            let mut d = 0usize;
            for i in 0..class {
                let bit = self.c.symbol(&mut self.cdf.mv_bits[ctx][comp][i], ((want_d >> i) & 1) as usize);
                d |= bit << i;
            }
            let fr = if force_integer { 3 } else { self.c.symbol(&mut self.cdf.mv_fp[ctx][comp], 3) };
            let hpv = if hp { self.c.symbol(&mut self.cdf.mv_hp[ctx][comp], 1) } else { 1 };
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

/// The stable sort of part of the stack by descending weight (7.10.2.11).
fn sort_stack(st: &mut MvStack, start: usize, end: usize) {
    let mut end = end;
    while end > start {
        let mut new_end = start;
        for idx in start + 1..end {
            if st.weight[idx - 1] < st.weight[idx] {
                st.weight.swap(idx - 1, idx);
                st.mv.swap(idx - 1, idx);
                new_end = idx;
            }
        }
        end = new_end;
    }
}
