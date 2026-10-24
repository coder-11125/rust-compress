//! Mode info of blocks in inter frames (specification 5.11.18 to 5.11.32 and the CDF selection rules of 8.3.2):
//! segment ids, skip mode, the intra/inter choice, reference frames, prediction modes, motion vectors, motion
//! mode, compound type and interpolation filters.
//!
//! Like the rest of the tile syntax it is written once over [`Coder`]: decoding reads the elements, encoding
//! writes the values planned in the [`Blk`].

use super::consts::*;
use super::entropy::Coder;
use super::mvpred::MvStack;
use super::obu::SEG_LVL_REF_FRAME;
use super::obu::{GM_TRANSLATION, SWITCHABLE};
use super::tile::{Blk, TileCoder};
use crate::error::Result;

const SEG_LVL_GLOBALMV: usize = 7;
const UNIDIR_COMP_REFERENCE: usize = 0;
const BIDIR_COMP_REFERENCE: usize = 1;

/// The reference frames of the blocks to the left and above.
#[derive(Clone, Copy)]
pub(crate) struct Nbr {
    left: [i8; 2],
    above: [i8; 2],
    avail_l: bool,
    avail_u: bool,
}

impl Nbr {
    fn left_intra(&self) -> bool {
        self.left[0] <= INTRA_FRAME
    }
    fn above_intra(&self) -> bool {
        self.above[0] <= INTRA_FRAME
    }
    fn left_single(&self) -> bool {
        self.left[1] <= INTRA_FRAME
    }
    fn above_single(&self) -> bool {
        self.above[1] <= INTRA_FRAME
    }

    fn count_refs(&self, ft: i8) -> usize {
        let mut c = 0;
        if self.avail_u {
            c += (self.above[0] == ft) as usize + (self.above[1] == ft) as usize;
        }
        if self.avail_l {
            c += (self.left[0] == ft) as usize + (self.left[1] == ft) as usize;
        }
        c
    }

    fn ref_count_ctx(a: usize, b: usize) -> usize {
        if a < b {
            0
        } else if a == b {
            1
        } else {
            2
        }
    }

    fn ctx_pair(&self, a: &[i8], b: &[i8]) -> usize {
        Self::ref_count_ctx(a.iter().map(|&f| self.count_refs(f)).sum(), b.iter().map(|&f| self.count_refs(f)).sum())
    }
}

fn is_backward(f: i8) -> bool {
    (BWDREF_FRAME..=ALTREF_FRAME).contains(&f)
}

fn samedir(a: i8, b: i8) -> bool {
    (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME)
}

impl<C: Coder> TileCoder<'_, C> {
    fn neighbours(&self, b: &Blk) -> Nbr {
        let (r, c) = (b.mi_row, b.mi_col);
        Nbr {
            left: if b.avail_l { self.f.ref_frame[self.mi(r, c - 1)] } else { [INTRA_FRAME, NONE_FRAME] },
            above: if b.avail_u { self.f.ref_frame[self.mi(r - 1, c)] } else { [INTRA_FRAME, NONE_FRAME] },
            avail_l: b.avail_l,
            avail_u: b.avail_u,
        }
    }

    pub(crate) fn inter_frame_mode_info(&mut self, b: &mut Blk) -> Result<()> {
        let nbr = self.neighbours(b);
        b.use_intrabc = false;
        if C::DECODING {
            b.skip = false;
            b.skip_mode = false;
        }
        self.inter_segment_id(b, true);
        self.read_skip_mode(b);
        if b.skip_mode {
            b.skip = true;
        } else {
            self.read_skip(b);
        }
        if !self.f.hdr.seg.seg_id_pre_skip {
            self.inter_segment_id(b, false);
        }
        b.lossless = self.f.hdr.lossless_array[b.segment_id as usize];
        self.read_cdef(b);
        self.read_delta_qindex(b);
        self.read_delta_lf(b);
        self.read_deltas = false;
        self.read_is_inter(b, &nbr);
        if b.is_inter {
            self.inter_block_mode_info(b, &nbr)
        } else {
            self.intra_block_mode_info(b);
            Ok(())
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // Segment id, skip mode, is_inter
    // ----------------------------------------------------------------------------------------------------

    /// `get_segment_id`: the smallest previous segment id under the block.
    fn predicted_segment_id(&self, b: &Blk) -> u8 {
        let Some(prev) = self.f.prev_segment_ids.as_ref() else { return 0 };
        let (cols, rows) = (self.f.mi_cols, self.f.mi_rows);
        let x_mis = (cols - b.mi_col).min(NUM_4X4_WIDE[b.bsize as usize] as usize);
        let y_mis = (rows - b.mi_row).min(NUM_4X4_HIGH[b.bsize as usize] as usize);
        let mut seg = 7u8;
        for y in 0..y_mis {
            for x in 0..x_mis {
                seg = seg.min(prev[(b.mi_row + y) * cols + b.mi_col + x]);
            }
        }
        seg
    }

    fn set_seg_pred_ctx(&mut self, b: &Blk, v: u8) {
        for i in 0..NUM_4X4_WIDE[b.bsize as usize] as usize {
            self.above_seg_pred[b.mi_col + i] = v;
        }
        for i in 0..NUM_4X4_HIGH[b.bsize as usize] as usize {
            self.left_seg_pred[b.mi_row + i] = v;
        }
    }

    fn inter_segment_id(&mut self, b: &mut Blk, pre_skip: bool) {
        let seg = self.f.hdr.seg.clone();
        if !seg.enabled {
            b.segment_id = 0;
            return;
        }
        let predicted = self.predicted_segment_id(b);
        if !seg.update_map {
            b.segment_id = predicted;
            return;
        }
        if pre_skip && !seg.seg_id_pre_skip {
            b.segment_id = 0;
            return;
        }
        if !pre_skip && b.skip {
            self.set_seg_pred_ctx(b, 0);
            self.read_segment_id(b);
            return;
        }
        if seg.temporal_update {
            let ctx = (self.left_seg_pred[b.mi_row] + self.above_seg_pred[b.mi_col]) as usize;
            let want = (b.segment_id == predicted) as usize;
            let pred_flag = self.c.symbol(&mut self.cdf.seg_id_predicted[ctx], want);
            if pred_flag != 0 {
                b.segment_id = predicted;
            } else {
                self.read_segment_id(b);
            }
            self.set_seg_pred_ctx(b, pred_flag as u8);
        } else {
            self.read_segment_id(b);
        }
    }

    fn read_skip_mode(&mut self, b: &mut Blk) {
        let h = &self.f.hdr;
        let active = |f: usize| h.seg_feature_active(b.segment_id as usize, f);
        if active(SEG_LVL_SKIP)
            || active(SEG_LVL_REF_FRAME)
            || active(SEG_LVL_GLOBALMV)
            || !h.skip_mode_present
            || block_width(b.bsize) < 8
            || block_height(b.bsize) < 8
        {
            b.skip_mode = false;
            return;
        }
        let mut ctx = 0;
        if b.avail_u {
            ctx += self.f.skip_mode[self.mi(b.mi_row - 1, b.mi_col)] as usize;
        }
        if b.avail_l {
            ctx += self.f.skip_mode[self.mi(b.mi_row, b.mi_col - 1)] as usize;
        }
        b.skip_mode = self.c.symbol(&mut self.cdf.skip_mode[ctx], b.skip_mode as usize) != 0;
    }

    fn read_is_inter(&mut self, b: &mut Blk, n: &Nbr) {
        let h = &self.f.hdr;
        let seg = b.segment_id as usize;
        if b.skip_mode {
            b.is_inter = true;
        } else if h.seg_feature_active(seg, SEG_LVL_REF_FRAME) {
            b.is_inter = h.seg.feature_data[seg][SEG_LVL_REF_FRAME] != INTRA_FRAME as i32;
        } else if h.seg_feature_active(seg, SEG_LVL_GLOBALMV) {
            b.is_inter = true;
        } else {
            let ctx = if n.avail_u && n.avail_l {
                if n.left_intra() && n.above_intra() { 3 } else { (n.left_intra() || n.above_intra()) as usize }
            } else if n.avail_u || n.avail_l {
                2 * (if n.avail_u { n.above_intra() } else { n.left_intra() }) as usize
            } else {
                0
            };
            b.is_inter = self.c.symbol(&mut self.cdf.is_inter[ctx], b.is_inter as usize) != 0;
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // Intra block in an inter frame
    // ----------------------------------------------------------------------------------------------------

    fn intra_block_mode_info(&mut self, b: &mut Blk) {
        b.ref_frame = [INTRA_FRAME, NONE_FRAME];
        b.is_inter = false;
        let ctx = SIZE_GROUP[b.bsize as usize] as usize;
        b.y_mode = self.c.symbol(&mut self.cdf.y_mode[ctx], b.y_mode as usize) as u8;
        self.intra_mode_tail(b);
    }

    // ----------------------------------------------------------------------------------------------------
    // Inter block
    // ----------------------------------------------------------------------------------------------------

    fn inter_block_mode_info(&mut self, b: &mut Blk, n: &Nbr) -> Result<()> {
        b.pal_size_y = 0;
        b.pal_size_uv = 0;
        b.use_filter_intra = false;
        self.read_ref_frames(b, n);
        let compound = b.ref_frame[1] > INTRA_FRAME;
        let st = self.find_mv_stack(b, b.ref_frame, compound);
        let seg = b.segment_id as usize;
        let h = &self.f.hdr;
        let skip_or_global = h.seg_feature_active(seg, SEG_LVL_SKIP) || h.seg_feature_active(seg, SEG_LVL_GLOBALMV);
        if b.skip_mode {
            b.y_mode = NEAREST_NEARESTMV;
        } else if skip_or_global {
            b.y_mode = GLOBALMV;
        } else if compound {
            let ctx = COMPOUND_MODE_CTX_MAP[st.ref_mv_context >> 1][st.new_mv_context.min(4)] as usize;
            let want = b.y_mode.wrapping_sub(NEAREST_NEARESTMV) as usize;
            b.y_mode = NEAREST_NEARESTMV + self.c.symbol(&mut self.cdf.compound_mode[ctx], want) as u8;
        } else {
            let want = b.y_mode;
            let new_mv = self.c.symbol(&mut self.cdf.new_mv[st.new_mv_context], (want != NEWMV) as usize);
            if new_mv == 0 {
                b.y_mode = NEWMV;
            } else {
                let zero_mv = self.c.symbol(&mut self.cdf.zero_mv[st.zero_mv_context], (want != GLOBALMV) as usize);
                if zero_mv == 0 {
                    b.y_mode = GLOBALMV;
                } else {
                    let ref_mv = self.c.symbol(&mut self.cdf.ref_mv[st.ref_mv_context], (want == NEARMV) as usize);
                    b.y_mode = if ref_mv == 0 { NEARESTMV } else { NEARMV };
                }
            }
        }
        // Dynamic reference list.
        let want_idx = b.ref_mv_idx as usize;
        let mut ref_mv_idx = 0usize;
        if b.y_mode == NEWMV || b.y_mode == NEW_NEWMV {
            for idx in 0..2 {
                if st.num_found > idx + 1 {
                    let drl = self.c.symbol(&mut self.cdf.drl_mode[st.drl_ctx[idx]], (want_idx > idx) as usize);
                    if drl == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        } else if matches!(b.y_mode, NEARMV | NEAR_NEARMV | NEAR_NEWMV | NEW_NEARMV) {
            ref_mv_idx = 1;
            for idx in 1..3 {
                if st.num_found > idx + 1 {
                    let drl = self.c.symbol(&mut self.cdf.drl_mode[st.drl_ctx[idx]], (want_idx > idx) as usize);
                    if drl == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        }
        b.ref_mv_idx = ref_mv_idx as u8;
        self.assign_mv(b, &st, compound);
        self.read_interintra_mode(b, compound);
        self.read_motion_mode(b, n, compound)?;
        self.read_compound_type(b, n, compound);
        // Interpolation filters.
        if self.f.hdr.interpolation_filter == SWITCHABLE {
            let dual = self.f.seq.enable_dual_filter;
            for dir in 0..(if dual { 2 } else { 1 }) {
                if self.needs_interp_filter(b) {
                    let ctx = self.interp_filter_ctx(b, dir);
                    b.interp_filter[dir] =
                        self.c.symbol(&mut self.cdf.interp_filter[ctx], b.interp_filter[dir] as usize) as u8;
                } else {
                    b.interp_filter[dir] = EIGHTTAP;
                }
            }
            if !dual {
                b.interp_filter[1] = b.interp_filter[0];
            }
        } else {
            b.interp_filter = [self.f.hdr.interpolation_filter; 2];
        }
        Ok(())
    }

    fn needs_interp_filter(&self, b: &Blk) -> bool {
        let large = block_width(b.bsize).min(block_height(b.bsize)) >= 8;
        let gm = |f: i8| self.f.hdr.gm_type[f as usize];
        if b.skip_mode || b.motion_mode == LOCALWARP {
            false
        } else if large && b.y_mode == GLOBALMV {
            gm(b.ref_frame[0]) == GM_TRANSLATION
        } else if large && b.y_mode == GLOBAL_GLOBALMV {
            gm(b.ref_frame[0]) == GM_TRANSLATION || gm(b.ref_frame[1]) == GM_TRANSLATION
        } else {
            true
        }
    }

    fn interp_filter_ctx(&self, b: &Blk, dir: usize) -> usize {
        let mut ctx = ((dir & 1) * 2 + (b.ref_frame[1] > INTRA_FRAME) as usize) * 4;
        let (mut left_type, mut above_type) = (3usize, 3usize);
        if b.avail_l {
            let i = self.mi(b.mi_row, b.mi_col - 1);
            if self.f.ref_frame[i][0] == b.ref_frame[0] || self.f.ref_frame[i][1] == b.ref_frame[0] {
                left_type = self.f.interp_filter[i][dir] as usize;
            }
        }
        if b.avail_u {
            let i = self.mi(b.mi_row - 1, b.mi_col);
            if self.f.ref_frame[i][0] == b.ref_frame[0] || self.f.ref_frame[i][1] == b.ref_frame[0] {
                above_type = self.f.interp_filter[i][dir] as usize;
            }
        }
        if left_type == above_type {
            ctx += left_type;
        } else if left_type == 3 {
            ctx += above_type;
        } else if above_type == 3 {
            ctx += left_type;
        } else {
            ctx += 3;
        }
        ctx
    }

    // ----------------------------------------------------------------------------------------------------
    // Reference frames
    // ----------------------------------------------------------------------------------------------------

    fn read_ref_frames(&mut self, b: &mut Blk, n: &Nbr) {
        let h = &self.f.hdr;
        let seg = b.segment_id as usize;
        if b.skip_mode {
            b.ref_frame = [h.skip_mode_frame[0] as i8, h.skip_mode_frame[1] as i8];
            return;
        }
        if h.seg_feature_active(seg, SEG_LVL_REF_FRAME) {
            b.ref_frame = [h.seg.feature_data[seg][SEG_LVL_REF_FRAME] as i8, NONE_FRAME];
            return;
        }
        if h.seg_feature_active(seg, SEG_LVL_SKIP) || h.seg_feature_active(seg, SEG_LVL_GLOBALMV) {
            b.ref_frame = [LAST_FRAME, NONE_FRAME];
            return;
        }
        let reference_select = h.reference_select;
        let (bw4, bh4) = (NUM_4X4_WIDE[b.bsize as usize], NUM_4X4_HIGH[b.bsize as usize]);
        let want = b.ref_frame;
        let want_comp = want[1] > INTRA_FRAME;
        let comp = if reference_select && bw4.min(bh4) >= 2 {
            let ctx = self.comp_mode_ctx(n);
            self.c.symbol(&mut self.cdf.comp_mode[ctx], want_comp as usize) != 0
        } else {
            false
        };
        if comp {
            let want_uni = want[0] >= BWDREF_FRAME && want[1] >= BWDREF_FRAME
                || (want[0] < BWDREF_FRAME && want[1] < BWDREF_FRAME);
            let ctx = self.comp_ref_type_ctx(n);
            let ty = self.c.symbol(
                &mut self.cdf.comp_ref_type[ctx],
                if want_uni { UNIDIR_COMP_REFERENCE } else { BIDIR_COMP_REFERENCE },
            );
            if ty == UNIDIR_COMP_REFERENCE {
                let ctx = n.ctx_pair(
                    &[LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME],
                    &[BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME],
                );
                let bwd_alt = want[0] == BWDREF_FRAME && want[1] == ALTREF_FRAME;
                let uni = self.c.symbol(&mut self.cdf.uni_comp_ref[ctx][0], bwd_alt as usize);
                if uni != 0 {
                    b.ref_frame = [BWDREF_FRAME, ALTREF_FRAME];
                } else {
                    let ctx1 = n.ctx_pair(&[LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME]);
                    let p1 = self.c.symbol(&mut self.cdf.uni_comp_ref[ctx1][1], (want[1] != LAST2_FRAME) as usize);
                    if p1 != 0 {
                        let ctx2 = n.ctx_pair(&[LAST3_FRAME], &[GOLDEN_FRAME]);
                        let p2 = self.c.symbol(&mut self.cdf.uni_comp_ref[ctx2][2], (want[1] == GOLDEN_FRAME) as usize);
                        b.ref_frame = [LAST_FRAME, if p2 != 0 { GOLDEN_FRAME } else { LAST3_FRAME }];
                    } else {
                        b.ref_frame = [LAST_FRAME, LAST2_FRAME];
                    }
                }
            } else {
                let ctx = n.ctx_pair(&[LAST_FRAME, LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME]);
                let comp_ref = self.c.symbol(&mut self.cdf.comp_ref[ctx][0], (want[0] >= LAST3_FRAME) as usize);
                if comp_ref == 0 {
                    let ctx1 = n.ctx_pair(&[LAST_FRAME], &[LAST2_FRAME]);
                    let p1 = self.c.symbol(&mut self.cdf.comp_ref[ctx1][1], (want[0] == LAST2_FRAME) as usize);
                    b.ref_frame[0] = if p1 != 0 { LAST2_FRAME } else { LAST_FRAME };
                } else {
                    let ctx2 = n.ctx_pair(&[LAST3_FRAME], &[GOLDEN_FRAME]);
                    let p2 = self.c.symbol(&mut self.cdf.comp_ref[ctx2][2], (want[0] == GOLDEN_FRAME) as usize);
                    b.ref_frame[0] = if p2 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                }
                let ctxb = n.ctx_pair(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME]);
                let bwd = self.c.symbol(&mut self.cdf.comp_bwd_ref[ctxb][0], (want[1] == ALTREF_FRAME) as usize);
                if bwd == 0 {
                    let ctxb1 = n.ctx_pair(&[BWDREF_FRAME], &[ALTREF2_FRAME]);
                    let p1 = self.c.symbol(&mut self.cdf.comp_bwd_ref[ctxb1][1], (want[1] == ALTREF2_FRAME) as usize);
                    b.ref_frame[1] = if p1 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                } else {
                    b.ref_frame[1] = ALTREF_FRAME;
                }
            }
        } else {
            let ctx1 = n.ctx_pair(
                &[LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME],
                &[BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME],
            );
            let w = want[0];
            let p1 = self.c.symbol(&mut self.cdf.single_ref[ctx1][0], is_backward(w) as usize);
            if p1 != 0 {
                let ctx2 = n.ctx_pair(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME]);
                let p2 = self.c.symbol(&mut self.cdf.single_ref[ctx2][1], (w == ALTREF_FRAME) as usize);
                if p2 == 0 {
                    let ctx6 = n.ctx_pair(&[BWDREF_FRAME], &[ALTREF2_FRAME]);
                    let p6 = self.c.symbol(&mut self.cdf.single_ref[ctx6][5], (w == ALTREF2_FRAME) as usize);
                    b.ref_frame[0] = if p6 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                } else {
                    b.ref_frame[0] = ALTREF_FRAME;
                }
            } else {
                let ctx3 = n.ctx_pair(&[LAST_FRAME, LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME]);
                let p3 = self.c.symbol(&mut self.cdf.single_ref[ctx3][2], (w >= LAST3_FRAME) as usize);
                if p3 != 0 {
                    let ctx5 = n.ctx_pair(&[LAST3_FRAME], &[GOLDEN_FRAME]);
                    let p5 = self.c.symbol(&mut self.cdf.single_ref[ctx5][4], (w == GOLDEN_FRAME) as usize);
                    b.ref_frame[0] = if p5 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                } else {
                    let ctx4 = n.ctx_pair(&[LAST_FRAME], &[LAST2_FRAME]);
                    let p4 = self.c.symbol(&mut self.cdf.single_ref[ctx4][3], (w == LAST2_FRAME) as usize);
                    b.ref_frame[0] = if p4 != 0 { LAST2_FRAME } else { LAST_FRAME };
                }
            }
            b.ref_frame[1] = NONE_FRAME;
        }
    }

    fn comp_mode_ctx(&self, n: &Nbr) -> usize {
        if n.avail_u && n.avail_l {
            if n.above_single() && n.left_single() {
                (is_backward(n.above[0]) ^ is_backward(n.left[0])) as usize
            } else if n.above_single() {
                2 + (is_backward(n.above[0]) || n.above_intra()) as usize
            } else if n.left_single() {
                2 + (is_backward(n.left[0]) || n.left_intra()) as usize
            } else {
                4
            }
        } else if n.avail_u {
            if n.above_single() { is_backward(n.above[0]) as usize } else { 3 }
        } else if n.avail_l {
            if n.left_single() { is_backward(n.left[0]) as usize } else { 3 }
        } else {
            1
        }
    }

    fn comp_ref_type_ctx(&self, n: &Nbr) -> usize {
        let (above0, above1, left0, left1) = (n.above[0], n.above[1], n.left[0], n.left[1]);
        let above_comp = n.avail_u && !n.above_intra() && !n.above_single();
        let left_comp = n.avail_l && !n.left_intra() && !n.left_single();
        let above_uni = above_comp && samedir(above0, above1);
        let left_uni = left_comp && samedir(left0, left1);
        if n.avail_u && !n.above_intra() && n.avail_l && !n.left_intra() {
            let same = samedir(above0, left0) as usize;
            if !above_comp && !left_comp {
                1 + 2 * same
            } else if !above_comp {
                if !left_uni { 1 } else { 3 + same }
            } else if !left_comp {
                if !above_uni { 1 } else { 3 + same }
            } else if !above_uni && !left_uni {
                0
            } else if !above_uni || !left_uni {
                2
            } else {
                3 + ((above0 == BWDREF_FRAME) == (left0 == BWDREF_FRAME)) as usize
            }
        } else if n.avail_u && n.avail_l {
            if above_comp {
                1 + 2 * above_uni as usize
            } else if left_comp {
                1 + 2 * left_uni as usize
            } else {
                2
            }
        } else if above_comp {
            4 * above_uni as usize
        } else if left_comp {
            4 * left_uni as usize
        } else {
            2
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // Motion vectors
    // ----------------------------------------------------------------------------------------------------

    /// `get_mode(refList)`.
    fn comp_mode_of(y_mode: u8, ref_list: usize) -> u8 {
        if ref_list == 0 {
            match y_mode {
                m if m < NEAREST_NEARESTMV => m,
                NEW_NEWMV | NEW_NEARESTMV | NEW_NEARMV => NEWMV,
                NEAREST_NEARESTMV | NEAREST_NEWMV => NEARESTMV,
                NEAR_NEARMV | NEAR_NEWMV => NEARMV,
                _ => GLOBALMV,
            }
        } else {
            match y_mode {
                NEW_NEWMV | NEAREST_NEWMV | NEAR_NEWMV => NEWMV,
                NEAREST_NEARESTMV | NEW_NEARESTMV => NEARESTMV,
                NEAR_NEARMV | NEW_NEARMV => NEARMV,
                _ => GLOBALMV,
            }
        }
    }

    fn assign_mv(&mut self, b: &mut Blk, st: &MvStack, compound: bool) {
        let want_mv = b.mv;
        for i in 0..(1 + compound as usize) {
            let comp_mode = Self::comp_mode_of(b.y_mode, i);
            let pred = if comp_mode == GLOBALMV {
                st.global_mvs[i]
            } else {
                let mut pos = if comp_mode == NEARESTMV { 0 } else { b.ref_mv_idx as usize };
                if comp_mode == NEWMV && st.num_found <= 1 {
                    pos = 0;
                }
                st.mv[pos][i]
            };
            if comp_mode == NEWMV {
                let want = [want_mv[i][0] - pred[0], want_mv[i][1] - pred[1]];
                let diff = self.read_mv_ctx(0, want);
                b.mv[i] = [pred[0] + diff[0], pred[1] + diff[1]];
            } else {
                b.mv[i] = pred;
            }
        }
        if !compound {
            b.mv[1] = [0, 0];
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // Inter-intra, motion mode and compound type
    // ----------------------------------------------------------------------------------------------------

    fn read_interintra_mode(&mut self, b: &mut Blk, compound: bool) {
        if !b.skip_mode
            && self.f.seq.enable_interintra_compound
            && !compound
            && b.bsize >= BLOCK_8X8
            && b.bsize <= BLOCK_32X32
        {
            let ctx = SIZE_GROUP[b.bsize as usize] as usize - 1;
            b.interintra = self.c.symbol(&mut self.cdf.inter_intra[ctx], b.interintra as usize) != 0;
            if b.interintra {
                b.interintra_mode =
                    self.c.symbol(&mut self.cdf.inter_intra_mode[ctx], b.interintra_mode as usize) as u8;
                b.ref_frame[1] = INTRA_FRAME;
                b.angle_y = 0;
                b.angle_uv = 0;
                b.use_filter_intra = false;
                b.wedge_interintra =
                    self.c.symbol(&mut self.cdf.wedge_inter_intra[b.bsize as usize], b.wedge_interintra as usize) != 0;
                if b.wedge_interintra {
                    b.wedge_index =
                        self.c.symbol(&mut self.cdf.wedge_index[b.bsize as usize], b.wedge_index as usize) as u8;
                    b.wedge_sign = 0;
                }
            }
        } else {
            b.interintra = false;
        }
    }

    fn read_motion_mode(&mut self, b: &mut Blk, n: &Nbr, compound: bool) -> Result<()> {
        b.motion_mode = SIMPLE;
        let h = &self.f.hdr;
        if b.skip_mode || !h.is_motion_mode_switchable || block_width(b.bsize).min(block_height(b.bsize)) < 8 {
            return Ok(());
        }
        if !h.force_integer_mv
            && (b.y_mode == GLOBALMV || b.y_mode == GLOBAL_GLOBALMV)
            && h.gm_type[b.ref_frame[0] as usize] > GM_TRANSLATION
        {
            return Ok(());
        }
        if compound || b.ref_frame[1] == INTRA_FRAME || !self.has_overlappable_candidates(b, n) {
            return Ok(());
        }
        let h = &self.f.hdr;
        let scaled = self.is_scaled(b.ref_frame[0]);
        if h.force_integer_mv || !h.allow_warped_motion || scaled {
            // find_warp_samples() would only matter for the choice below, which these conditions settle.
            let use_obmc = self.c.symbol(&mut self.cdf.use_obmc[b.bsize as usize], (b.motion_mode == OBMC) as usize);
            b.motion_mode = if use_obmc != 0 { OBMC } else { SIMPLE };
        } else {
            let num_samples = self.find_warp_samples(b);
            if num_samples == 0 {
                let use_obmc =
                    self.c.symbol(&mut self.cdf.use_obmc[b.bsize as usize], (b.motion_mode == OBMC) as usize);
                b.motion_mode = if use_obmc != 0 { OBMC } else { SIMPLE };
            } else {
                b.motion_mode =
                    self.c.symbol(&mut self.cdf.motion_mode[b.bsize as usize], b.motion_mode as usize) as u8;
            }
        }
        Ok(())
    }

    fn read_compound_type(&mut self, b: &mut Blk, n: &Nbr, compound: bool) {
        b.comp_group_idx = 0;
        b.compound_idx = 1;
        if b.skip_mode {
            b.compound_type = COMPOUND_AVERAGE;
            return;
        }
        if compound {
            let wedge_bits = WEDGE_BITS[b.bsize as usize];
            let seq = &self.f.seq;
            let (masked, jnt) = (seq.enable_masked_compound, seq.enable_jnt_comp);
            let want = b.compound_type;
            if masked {
                let ctx = self.comp_group_idx_ctx(b, n);
                b.comp_group_idx = self.c.symbol(
                    &mut self.cdf.comp_group_idx[ctx],
                    (want == COMPOUND_WEDGE || want == COMPOUND_DIFFWTD) as usize,
                ) as u8;
            }
            if b.comp_group_idx == 0 {
                if jnt {
                    let ctx = self.compound_idx_ctx(b, n);
                    b.compound_idx =
                        self.c.symbol(&mut self.cdf.compound_idx[ctx], (want != COMPOUND_DISTANCE) as usize) as u8;
                    b.compound_type = if b.compound_idx != 0 { COMPOUND_AVERAGE } else { COMPOUND_DISTANCE };
                } else {
                    b.compound_type = COMPOUND_AVERAGE;
                }
            } else if wedge_bits == 0 {
                b.compound_type = COMPOUND_DIFFWTD;
            } else {
                b.compound_type = self.c.symbol(&mut self.cdf.compound_type[b.bsize as usize], want as usize) as u8;
            }
            if b.compound_type == COMPOUND_WEDGE {
                b.wedge_index =
                    self.c.symbol(&mut self.cdf.wedge_index[b.bsize as usize], b.wedge_index as usize) as u8;
                b.wedge_sign = self.c.literal(1, b.wedge_sign as u32) as u8;
            } else if b.compound_type == COMPOUND_DIFFWTD {
                b.mask_type = self.c.literal(1, b.mask_type as u32) as u8;
            }
        } else if b.interintra {
            b.compound_type = if b.wedge_interintra { COMPOUND_WEDGE } else { COMPOUND_INTRA };
        } else {
            b.compound_type = COMPOUND_AVERAGE;
        }
    }

    fn comp_group_idx_ctx(&self, b: &Blk, n: &Nbr) -> usize {
        let mut ctx = 0;
        if n.avail_u {
            if !n.above_single() {
                ctx += self.f.comp_group_idx[self.mi(b.mi_row - 1, b.mi_col)] as usize;
            } else if n.above[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        if n.avail_l {
            if !n.left_single() {
                ctx += self.f.comp_group_idx[self.mi(b.mi_row, b.mi_col - 1)] as usize;
            } else if n.left[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        ctx.min(5)
    }

    fn compound_idx_ctx(&self, b: &Blk, n: &Nbr) -> usize {
        let h = &self.f.hdr;
        let seq = &self.f.seq;
        let fwd =
            super::obu::FrameHeader::relative_dist(seq, h.order_hints[b.ref_frame[0] as usize], h.order_hint).abs();
        let bck =
            super::obu::FrameHeader::relative_dist(seq, h.order_hints[b.ref_frame[1] as usize], h.order_hint).abs();
        let mut ctx = if fwd == bck { 3 } else { 0 };
        if n.avail_u {
            if !n.above_single() {
                ctx += self.f.compound_idx[self.mi(b.mi_row - 1, b.mi_col)] as usize;
            } else if n.above[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        if n.avail_l {
            if !n.left_single() {
                ctx += self.f.compound_idx[self.mi(b.mi_row, b.mi_col - 1)] as usize;
            } else if n.left[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        ctx
    }

    /// `is_scaled(refFrame)`.
    pub(crate) fn is_scaled(&self, rf: i8) -> bool {
        let Some(r) = self.f.refs.as_ref().and_then(|s| s.get(rf)) else { return false };
        let h = &self.f.hdr;
        let x_scale = ((r.upscaled_width << 14) + h.frame_width / 2) / h.frame_width;
        let y_scale = ((r.frame_height << 14) + h.frame_height / 2) / h.frame_height;
        x_scale != 1 << 14 || y_scale != 1 << 14
    }

    /// `has_overlappable_candidates`.
    fn has_overlappable_candidates(&self, b: &Blk, n: &Nbr) -> bool {
        if n.avail_u {
            let w4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
            let mut x4 = b.mi_col;
            while x4 < (b.mi_col + w4).min(self.f.mi_cols) {
                let x5 = x4 | 1;
                let x5 = x5.min(self.f.mi_cols - 1);
                if self.f.ref_frame[self.mi(b.mi_row - 1, x5)][0] > INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if n.avail_l {
            let h4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
            let mut y4 = b.mi_row;
            while y4 < (b.mi_row + h4).min(self.f.mi_rows) {
                let y5 = (y4 | 1).min(self.f.mi_rows - 1);
                if self.f.ref_frame[self.mi(y5, b.mi_col - 1)][0] > INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    /// `find_warp_samples` (7.10.4): collects the neighbouring blocks' motion for the local warp estimate into
    /// `self.warp_cands` and returns `NumSamples`.
    pub(crate) fn find_warp_samples(&mut self, b: &Blk) -> usize {
        self.warp_cands.clear();
        let (mut scanned, mut num) = (0usize, 0usize);
        let w4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let h4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        let (mut do_top_left, mut do_top_right) = (true, true);
        if b.avail_u {
            let src_w = NUM_4X4_WIDE[self.f.mi_size[self.mi(b.mi_row - 1, b.mi_col)] as usize] as usize;
            if w4 <= src_w {
                let col_offset = -((b.mi_col & (src_w - 1)) as isize);
                if col_offset < 0 {
                    do_top_left = false;
                }
                if col_offset + src_w as isize > w4 as isize {
                    do_top_right = false;
                }
                self.add_sample(b, -1, 0, &mut scanned, &mut num);
            } else {
                let mut i = 0;
                while i < w4.min(self.f.mi_cols - b.mi_col) {
                    let src_w = NUM_4X4_WIDE[self.f.mi_size[self.mi(b.mi_row - 1, b.mi_col + i)] as usize] as usize;
                    let step = w4.min(src_w);
                    self.add_sample(b, -1, i as isize, &mut scanned, &mut num);
                    i += step;
                }
            }
        }
        if b.avail_l {
            let src_h = NUM_4X4_HIGH[self.f.mi_size[self.mi(b.mi_row, b.mi_col - 1)] as usize] as usize;
            if h4 <= src_h {
                let row_offset = -((b.mi_row & (src_h - 1)) as isize);
                if row_offset < 0 {
                    do_top_left = false;
                }
                self.add_sample(b, 0, -1, &mut scanned, &mut num);
            } else {
                let mut i = 0;
                while i < h4.min(self.f.mi_rows - b.mi_row) {
                    let src_h = NUM_4X4_HIGH[self.f.mi_size[self.mi(b.mi_row + i, b.mi_col - 1)] as usize] as usize;
                    let step = h4.min(src_h);
                    self.add_sample(b, i as isize, -1, &mut scanned, &mut num);
                    i += step;
                }
            }
        }
        if do_top_left {
            self.add_sample(b, -1, -1, &mut scanned, &mut num);
        }
        if do_top_right && w4.max(h4) <= 16 {
            self.add_sample(b, -1, w4 as isize, &mut scanned, &mut num);
        }
        if num == 0 && scanned > 0 {
            num = 1;
        }
        self.num_samples = num;
        num
    }

    fn add_sample(&mut self, b: &Blk, delta_row: isize, delta_col: isize, scanned: &mut usize, num: &mut usize) {
        if *scanned >= 8 {
            return;
        }
        let (mv_row, mv_col) = (b.mi_row as isize + delta_row, b.mi_col as isize + delta_col);
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        let (r, c) = (mv_row as usize, mv_col as usize);
        let i = self.mi(r, c);
        if self.f.ref_frame[i][0] == super::mvpred::UNWRITTEN
            || self.f.ref_frame[i][0] != b.ref_frame[0]
            || self.f.ref_frame[i][1] != NONE_FRAME
        {
            return;
        }
        let cand_sz = self.f.mi_size[i] as usize;
        let (cand_w4, cand_h4) = (NUM_4X4_WIDE[cand_sz] as usize, NUM_4X4_HIGH[cand_sz] as usize);
        let cand_row = r & !(cand_h4 - 1);
        let cand_col = c & !(cand_w4 - 1);
        let mid_y = (cand_row * 4 + cand_h4 * 2) as i32 - 1;
        let mid_x = (cand_col * 4 + cand_w4 * 2) as i32 - 1;
        let threshold = (block_width(b.bsize).max(block_height(b.bsize)) as i32).clamp(16, 112);
        let cmv = self.f.mvs[self.mi(cand_row, cand_col)][0];
        let valid = (cmv[0] - b.mv[0][0]).abs() + (cmv[1] - b.mv[0][1]).abs() <= threshold;
        let cand = [mid_y * 8, mid_x * 8, mid_y * 8 + cmv[0], mid_x * 8 + cmv[1]];
        *scanned += 1;
        if !valid && *scanned > 1 {
            return;
        }
        if self.warp_cands.len() <= *num {
            self.warp_cands.push(cand);
        } else {
            self.warp_cands[*num] = cand;
        }
        if valid {
            *num += 1;
        }
    }
}
