//! Motion vector prediction (specification section 7.10.2): the candidate stack for a block, built from the
//! motion vectors of its decoded neighbours, plus the contexts that the inter mode syntax uses.
//!
//! Temporal candidates (`use_ref_frame_mvs`) come from the motion field of [`super::motion_field`].

use super::consts::*;
use super::entropy::Coder;
use super::obu::{GM_TRANSLATION, WARPEDMODEL_PREC_BITS};
use super::tile::{Blk, TileCoder};

pub const MV_BORDER: i32 = 128;
pub const REF_CAT_LEVEL: u32 = 640;
pub const MAX_REF_MV_STACK_SIZE: usize = 8;

/// Marks a block whose mode info has not been written yet in this frame.
pub const UNWRITTEN: i8 = -2;

/// A motion vector as (row, column) in 1/8 sample units.
pub type Mv = [i32; 2];

/// The candidate stack and contexts built by the find-MV-stack process.
#[derive(Clone, Default)]
pub struct MvStack {
    /// `RefStackMv[idx][list]`.
    pub mv: [[Mv; 2]; MAX_REF_MV_STACK_SIZE + 1],
    pub weight: [u32; MAX_REF_MV_STACK_SIZE + 1],
    pub num_found: usize,
    new_mv_count: usize,
    found_match: bool,
    pub global_mvs: [Mv; 2],
    pub new_mv_context: usize,
    pub ref_mv_context: usize,
    pub zero_mv_context: usize,
    pub drl_ctx: [usize; MAX_REF_MV_STACK_SIZE + 1],
}

pub fn has_newmv(mode: u8) -> bool {
    matches!(mode, NEWMV | NEW_NEWMV | NEAR_NEWMV | NEW_NEARMV | NEAREST_NEWMV | NEW_NEARESTMV)
}

impl<C: Coder> TileCoder<'_, C> {
    /// `lower_mv_precision` (7.10.2.10).
    pub(crate) fn lower_precision(&self, mv: &mut Mv) {
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

    /// The setup-global-MV process (7.10.2.1) for reference frame `rf`.
    pub(crate) fn setup_global_mv(&self, b: &Blk, rf: i8) -> Mv {
        let mut mv = [0i32, 0];
        if rf > INTRA_FRAME {
            let hdr = &self.f.hdr;
            let ty = hdr.gm_type[rf as usize];
            let gm = &hdr.gm_params[rf as usize];
            if ty == GM_TRANSLATION {
                mv = [gm[0] >> (WARPEDMODEL_PREC_BITS - 3), gm[1] >> (WARPEDMODEL_PREC_BITS - 3)];
            } else if ty > GM_TRANSLATION {
                let (bw, bh) = (block_width(b.bsize) as i64, block_height(b.bsize) as i64);
                let x = (b.mi_col * 4) as i64 + bw / 2 - 1;
                let y = (b.mi_row * 4) as i64 + bh / 2 - 1;
                let xc = (gm[2] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * x + gm[3] as i64 * y + gm[0] as i64;
                let yc = gm[4] as i64 * x + (gm[5] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * y + gm[1] as i64;
                let r2s = |v: i64, n: u32| -> i32 {
                    let r = (v.abs() + (1 << (n - 1))) >> n;
                    (if v < 0 { -r } else { r }) as i32
                };
                if hdr.allow_high_precision_mv {
                    mv = [r2s(yc, WARPEDMODEL_PREC_BITS - 3), r2s(xc, WARPEDMODEL_PREC_BITS - 3)];
                } else {
                    mv = [r2s(yc, WARPEDMODEL_PREC_BITS - 2) * 2, r2s(xc, WARPEDMODEL_PREC_BITS - 2) * 2];
                }
            }
        }
        self.lower_precision(&mut mv);
        mv
    }

    /// The add-reference-motion-vector process (7.10.2.7 to 7.10.2.9).
    fn add_ref_mv_candidate(&self, st: &mut MvStack, refs: [i8; 2], compound: bool, r: usize, c: usize, weight: u32) {
        let i = self.mi(r, c);
        if self.f.is_inter[i] == 0 {
            return;
        }
        let cand_mode = self.f.y_mode[i];
        let cand_ref = self.f.ref_frame[i];
        if !compound {
            for list in 0..2 {
                if cand_ref[list] != refs[0] {
                    continue;
                }
                let cand_size = self.f.mi_size[i];
                let large = block_width(cand_size).min(block_height(cand_size)) >= 8;
                let mut cand = if (cand_mode == GLOBALMV || cand_mode == GLOBAL_GLOBALMV)
                    && self.f.hdr.gm_type[refs[0] as usize] > GM_TRANSLATION
                    && large
                {
                    st.global_mvs[0]
                } else {
                    self.f.mvs[i][list]
                };
                self.lower_precision(&mut cand);
                if has_newmv(cand_mode) {
                    st.new_mv_count += 1;
                }
                st.found_match = true;
                if let Some(idx) = (0..st.num_found).find(|&k| st.mv[k][0] == cand) {
                    st.weight[idx] += weight;
                } else if st.num_found < MAX_REF_MV_STACK_SIZE {
                    st.mv[st.num_found][0] = cand;
                    st.weight[st.num_found] = weight;
                    st.num_found += 1;
                }
            }
        } else if cand_ref[0] == refs[0] && cand_ref[1] == refs[1] {
            let mut cands = self.f.mvs[i];
            if cand_mode == GLOBAL_GLOBALMV {
                for l in 0..2 {
                    if self.f.hdr.gm_type[refs[l] as usize] > GM_TRANSLATION {
                        cands[l] = st.global_mvs[l];
                    }
                }
            }
            for m in cands.iter_mut() {
                self.lower_precision(m);
            }
            st.found_match = true;
            if let Some(idx) = (0..st.num_found).find(|&k| st.mv[k] == cands) {
                st.weight[idx] += weight;
            } else if st.num_found < MAX_REF_MV_STACK_SIZE {
                st.mv[st.num_found] = cands;
                st.weight[st.num_found] = weight;
                st.num_found += 1;
            }
            if has_newmv(cand_mode) {
                st.new_mv_count += 1;
            }
        }
    }

    fn scan_row(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool, delta_row: isize) {
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
            self.add_ref_mv_candidate(st, refs, compound, r, c, len as u32 * 2);
            i += len;
        }
    }

    fn scan_col(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool, delta_col: isize) {
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
            self.add_ref_mv_candidate(st, refs, compound, r, c, len as u32 * 2);
            i += len;
        }
    }

    fn scan_point(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool, delta_row: isize, delta_col: isize) {
        let mv_row = b.mi_row as isize + delta_row;
        let mv_col = b.mi_col as isize + delta_col;
        if self.is_inside(mv_row, mv_col) {
            let (r, c) = (mv_row as usize, mv_col as usize);
            if self.f.ref_frame[self.mi(r, c)][0] != UNWRITTEN {
                self.add_ref_mv_candidate(st, refs, compound, r, c, 4);
            }
        }
    }

    /// The find-MV-stack process (7.10.2) for the reference frames `refs` (`refs[1]` used when `compound`).
    pub(crate) fn find_mv_stack(&self, b: &Blk, refs: [i8; 2], compound: bool) -> MvStack {
        let mut st = MvStack::default();
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        st.global_mvs[0] = self.setup_global_mv(b, refs[0]);
        if compound {
            st.global_mvs[1] = self.setup_global_mv(b, refs[1]);
        }
        self.scan_row(&mut st, b, refs, compound, -1);
        let mut found_above = st.found_match;
        st.found_match = false;
        self.scan_col(&mut st, b, refs, compound, -1);
        let mut found_left = st.found_match;
        st.found_match = false;
        if bw4.max(bh4) <= 16 {
            self.scan_point(&mut st, b, refs, compound, -1, bw4 as isize);
        }
        if st.found_match {
            found_above = true;
        }
        let close_matches = found_above as usize + found_left as usize;
        let num_nearest = st.num_found;
        let num_new = st.new_mv_count;
        for w in st.weight.iter_mut().take(num_nearest) {
            *w += REF_CAT_LEVEL;
        }
        st.zero_mv_context = 0;
        if self.f.hdr.use_ref_frame_mvs {
            self.temporal_scan(&mut st, b, refs, compound);
        }
        self.scan_point(&mut st, b, refs, compound, -1, -1);
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        self.scan_row(&mut st, b, refs, compound, -3);
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        self.scan_col(&mut st, b, refs, compound, -3);
        if st.found_match {
            found_left = true;
        }
        st.found_match = false;
        if bh4 > 1 {
            self.scan_row(&mut st, b, refs, compound, -5);
        }
        if st.found_match {
            found_above = true;
        }
        st.found_match = false;
        if bw4 > 1 {
            self.scan_col(&mut st, b, refs, compound, -5);
        }
        if st.found_match {
            found_left = true;
        }
        let total_matches = found_above as usize + found_left as usize;
        sort_stack(&mut st, 0, num_nearest);
        let n = st.num_found;
        sort_stack(&mut st, num_nearest, n);
        if st.num_found < 2 {
            self.extra_search(&mut st, b, refs, compound);
        }
        // Context and clamping (7.10.2.14).
        for idx in 0..st.num_found {
            let mut z = 0;
            if idx + 1 < st.num_found {
                let (w0, w1) = (st.weight[idx], st.weight[idx + 1]);
                if w0 >= REF_CAT_LEVEL {
                    if w1 < REF_CAT_LEVEL {
                        z = 1;
                    }
                } else {
                    z = 2;
                }
            }
            st.drl_ctx[idx] = z;
        }
        let (bw, bh) = (bw4 as i32 * 4, bh4 as i32 * 4);
        for idx in 0..st.num_found {
            for list in 0..(1 + compound as usize) {
                let mv = &mut st.mv[idx][list];
                mv[0] = self.clamp_mv_row(b, mv[0], MV_BORDER + bh * 8);
                mv[1] = self.clamp_mv_col(b, mv[1], MV_BORDER + bw * 8);
            }
        }
        match close_matches {
            0 => {
                st.new_mv_context = total_matches.min(1);
                st.ref_mv_context = total_matches;
            }
            1 => {
                st.new_mv_context = 3 - num_new.min(1);
                st.ref_mv_context = 2 + total_matches;
            }
            _ => {
                st.new_mv_context = 5 - num_new.min(1);
                st.ref_mv_context = 5;
            }
        }
        st
    }

    /// The extra search process (7.10.2.12): partially matching neighbours, then global motion.
    fn extra_search(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool) {
        let mut ref_id: [Vec<Mv>; 2] = [Vec::new(), Vec::new()];
        let mut ref_diff: [Vec<Mv>; 2] = [Vec::new(), Vec::new()];
        let w4 = NUM_4X4_WIDE[b.bsize as usize].min(16) as usize;
        let h4 = NUM_4X4_HIGH[b.bsize as usize].min(16) as usize;
        let w4 = w4.min(self.f.mi_cols - b.mi_col);
        let h4 = h4.min(self.f.mi_rows - b.mi_row);
        let num4x4 = w4.min(h4);
        for pass in 0..2 {
            let mut idx = 0usize;
            while idx < num4x4 && st.num_found < 2 {
                let (mv_row, mv_col) = if pass == 0 {
                    (b.mi_row as isize - 1, (b.mi_col + idx) as isize)
                } else {
                    ((b.mi_row + idx) as isize, b.mi_col as isize - 1)
                };
                if !self.is_inside(mv_row, mv_col) {
                    break;
                }
                let (r, c) = (mv_row as usize, mv_col as usize);
                let i = self.mi(r, c);
                // add_extra_mv_candidate
                for cand_list in 0..2 {
                    let cand_ref = self.f.ref_frame[i][cand_list];
                    if cand_ref <= INTRA_FRAME {
                        continue;
                    }
                    let mut cand = self.f.mvs[i][cand_list];
                    if compound {
                        for list in 0..2 {
                            let mut m = cand;
                            if cand_ref == refs[list] && ref_id[list].len() < 2 {
                                ref_id[list].push(m);
                            } else if ref_diff[list].len() < 2 {
                                if self.f.hdr.ref_frame_sign_bias[cand_ref as usize]
                                    != self.f.hdr.ref_frame_sign_bias[refs[list] as usize]
                                {
                                    m = [-m[0], -m[1]];
                                }
                                ref_diff[list].push(m);
                            }
                        }
                    } else {
                        if self.f.hdr.ref_frame_sign_bias[cand_ref as usize]
                            != self.f.hdr.ref_frame_sign_bias[refs[0] as usize]
                        {
                            cand = [-cand[0], -cand[1]];
                        }
                        if !(0..st.num_found).any(|k| st.mv[k][0] == cand) {
                            st.mv[st.num_found][0] = cand;
                            st.weight[st.num_found] = 2;
                            st.num_found += 1;
                        }
                    }
                }
                idx += if pass == 0 {
                    NUM_4X4_WIDE[self.f.mi_size[i] as usize] as usize
                } else {
                    NUM_4X4_HIGH[self.f.mi_size[i] as usize] as usize
                };
            }
        }
        if compound {
            let mut combined = [[[0i32; 2]; 2]; 2]; // combined[idx][list]
            for list in 0..2 {
                let mut count = 0;
                for m in &ref_id[list] {
                    combined[count][list] = *m;
                    count += 1;
                }
                let mut k = 0;
                while k < ref_diff[list].len() && count < 2 {
                    combined[count][list] = ref_diff[list][k];
                    count += 1;
                    k += 1;
                }
                while count < 2 {
                    combined[count][list] = st.global_mvs[list];
                    count += 1;
                }
            }
            if st.num_found == 1 {
                st.mv[st.num_found] = if combined[0] == st.mv[0] { combined[1] } else { combined[0] };
                st.weight[st.num_found] = 2;
                st.num_found += 1;
            } else {
                for c in combined.iter() {
                    st.mv[st.num_found] = *c;
                    st.weight[st.num_found] = 2;
                    st.num_found += 1;
                }
            }
        } else {
            for idx in st.num_found..2 {
                st.mv[idx][0] = st.global_mvs[0];
            }
        }
    }

    pub(crate) fn clamp_mv_row(&self, b: &Blk, v: i32, border: i32) -> i32 {
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as i32;
        let to_top = -((b.mi_row as i32 * 4) * 8);
        let to_bottom = ((self.f.mi_rows as i32 - bh4 - b.mi_row as i32) * 4) * 8;
        v.clamp(to_top - border, to_bottom + border)
    }

    pub(crate) fn clamp_mv_col(&self, b: &Blk, v: i32, border: i32) -> i32 {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as i32;
        let to_left = -((b.mi_col as i32 * 4) * 8);
        let to_right = ((self.f.mi_cols as i32 - bw4 - b.mi_col as i32) * 4) * 8;
        v.clamp(to_left - border, to_right + border)
    }

    /// The temporal scan process (7.10.2.5): candidates from the projected motion field.
    fn temporal_scan(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool) {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as i32;
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as i32;
        let step_w = if bw4 >= 16 { 4 } else { 2 };
        let step_h = if bh4 >= 16 { 4 } else { 2 };
        let mut dr = 0;
        while dr < bh4.min(16) {
            let mut dc = 0;
            while dc < bw4.min(16) {
                self.add_tpl_ref_mv(st, b, refs, compound, dr, dc);
                dc += step_w;
            }
            dr += step_h;
        }
        let allow_extension = (2..16).contains(&bh4) && (2..16).contains(&bw4);
        if allow_extension {
            for (dr, dc) in [(bh4, -2), (bh4, bw4), (bh4 - 2, bw4)] {
                let row = (b.mi_row as i32 & 15) + dr;
                let col = (b.mi_col as i32 & 15) + dc;
                if (0..16).contains(&row) && (0..16).contains(&col) {
                    self.add_tpl_ref_mv(st, b, refs, compound, dr, dc);
                }
            }
        }
    }

    /// The temporal sample process (7.10.2.6).
    fn add_tpl_ref_mv(&self, st: &mut MvStack, b: &Blk, refs: [i8; 2], compound: bool, dr: i32, dc: i32) {
        let mv_row = (b.mi_row as i32 + dr) | 1;
        let mv_col = (b.mi_col as i32 + dc) | 1;
        if !self.is_inside(mv_row as isize, mv_col as isize) {
            return;
        }
        let w8 = self.f.mi_cols >> 1;
        let idx8 = (mv_row as usize >> 1) * w8 + (mv_col as usize >> 1);
        let first = dr == 0 && dc == 0;
        if first {
            st.zero_mv_context = 1;
        }
        let n = 1 + compound as usize;
        let mut cand = [[0i32; 2]; 2];
        for l in 0..n {
            let m = self.f.motion_field[refs[l] as usize][idx8];
            if m[0] == super::motion_field::INVALID_MV {
                return;
            }
            cand[l] = [m[0] as i32, m[1] as i32];
        }
        for c in cand.iter_mut().take(n) {
            self.lower_precision(c);
        }
        if first {
            let far = (0..n).any(|l| {
                (cand[l][0] - st.global_mvs[l][0]).abs() >= 16 || (cand[l][1] - st.global_mvs[l][1]).abs() >= 16
            });
            st.zero_mv_context = far as usize;
        }
        let found = (0..st.num_found).find(|&i| (0..n).all(|l| st.mv[i][l] == cand[l]));
        if let Some(i) = found {
            st.weight[i] += 2;
        } else if st.num_found < MAX_REF_MV_STACK_SIZE {
            let k = st.num_found;
            st.mv[k][..n].copy_from_slice(&cand[..n]);
            st.weight[k] = 2;
            st.num_found += 1;
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
