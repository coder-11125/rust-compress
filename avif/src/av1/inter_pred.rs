//! Inter prediction (specification section 7.11.3): motion compensated prediction from reference frames with
//! scaled references, compound average / distance / wedge / difference-weighted blending, inter-intra,
//! overlapped block motion compensation and local / global warped motion.

use super::cdf_inter::SUBPEL_FILTERS;
use super::consts::*;
use super::entropy::Coder;
use super::obu::{FrameHeader, GM_TRANSLATION, WARPEDMODEL_PREC_BITS};
use super::refs::RefView;
use super::tile::{Blk, TileCoder};
use super::warp_tables::{DIV_LUT, WARPED_FILTERS};
use rc_core::error::{Error, Result};
use std::sync::OnceLock;

const MAX_FRAME_DISTANCE: i32 = 31;
const QUANT_DIST_WEIGHT: [[i32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE]];
const QUANT_DIST_LOOKUP: [[i32; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];
const LS_MV_MAX: i32 = 256;
const WARPEDMODEL_TRANS_CLAMP: i64 = 1 << 23;
const WARPEDMODEL_NONDIAGAFFINE_CLAMP: i64 = 1 << 13;
const WARP_PARAM_REDUCE_BITS: u32 = 6;
const WARPEDDIFF_PREC_BITS: u32 = 10;
const WARPEDPIXEL_PREC_SHIFTS: i32 = 64;
const DIV_LUT_BITS: u32 = 8;
const DIV_LUT_PREC_BITS: u32 = 14;

const OBMC_MASK_2: [u8; 2] = [45, 64];
const OBMC_MASK_4: [u8; 4] = [39, 50, 59, 64];
const OBMC_MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
const OBMC_MASK_16: [u8; 16] = [34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64];
const OBMC_MASK_32: [u8; 32] = [
    33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62, 64, 64, 64, 64, 64,
    64, 64, 64,
];

const II_WEIGHTS_1D: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22,
    22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8,
    8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

const WEDGE_HORIZONTAL: usize = 0;
const WEDGE_VERTICAL: usize = 1;
const WEDGE_OBLIQUE27: usize = 2;
const WEDGE_OBLIQUE63: usize = 3;
const WEDGE_OBLIQUE117: usize = 4;
const WEDGE_OBLIQUE153: usize = 5;
const MASK_MASTER_SIZE: usize = 64;

const WEDGE_MASTER_OBLIQUE_ODD: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 6, 18, 37, 53, 60, 63,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_OBLIQUE_EVEN: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 11, 27, 46, 58, 62, 63,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_VERTICAL: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 21, 43, 57, 62, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// `Wedge_Codebook[shape][index] = (direction, xoff, yoff)`.
const WEDGE_CODEBOOK: [[[u8; 3]; 16]; 3] = {
    const H: u8 = 0;
    const V: u8 = 1;
    const O27: u8 = 2;
    const O63: u8 = 3;
    const O117: u8 = 4;
    const O153: u8 = 5;
    [
        [
            [O27, 4, 4],
            [O63, 4, 4],
            [O117, 4, 4],
            [O153, 4, 4],
            [H, 4, 2],
            [H, 4, 4],
            [H, 4, 6],
            [V, 4, 4],
            [O27, 4, 2],
            [O27, 4, 6],
            [O153, 4, 2],
            [O153, 4, 6],
            [O63, 2, 4],
            [O63, 6, 4],
            [O117, 2, 4],
            [O117, 6, 4],
        ],
        [
            [O27, 4, 4],
            [O63, 4, 4],
            [O117, 4, 4],
            [O153, 4, 4],
            [V, 2, 4],
            [V, 4, 4],
            [V, 6, 4],
            [H, 4, 4],
            [O27, 4, 2],
            [O27, 4, 6],
            [O153, 4, 2],
            [O153, 4, 6],
            [O63, 2, 4],
            [O63, 6, 4],
            [O117, 2, 4],
            [O117, 6, 4],
        ],
        [
            [O27, 4, 4],
            [O63, 4, 4],
            [O117, 4, 4],
            [O153, 4, 4],
            [H, 4, 2],
            [H, 4, 6],
            [V, 2, 4],
            [V, 6, 4],
            [O27, 4, 2],
            [O27, 4, 6],
            [O153, 4, 2],
            [O153, 4, 6],
            [O63, 2, 4],
            [O63, 6, 4],
            [O117, 2, 4],
            [O117, 6, 4],
        ],
    ]
};

/// `WedgeMasks[bsize][flip_sign][wedge]`: row-major `Block_Width x Block_Height` masks (empty for sizes without wedges).
fn wedge_masks() -> &'static Vec<[Vec<Vec<u8>>; 2]> {
    static MASKS: OnceLock<Vec<[Vec<Vec<u8>>; 2]>> = OnceLock::new();
    MASKS.get_or_init(|| {
        let n = MASK_MASTER_SIZE;
        let mut master = vec![vec![vec![0u8; n]; n]; 6];
        for j in 0..n {
            let mut shift = (n / 4) as i32;
            for i in (0..n).step_by(2) {
                master[WEDGE_OBLIQUE63][i][j] =
                    WEDGE_MASTER_OBLIQUE_EVEN[(j as i32 - shift).clamp(0, n as i32 - 1) as usize];
                shift -= 1;
                master[WEDGE_OBLIQUE63][i + 1][j] =
                    WEDGE_MASTER_OBLIQUE_ODD[(j as i32 - shift).clamp(0, n as i32 - 1) as usize];
                master[WEDGE_VERTICAL][i][j] = WEDGE_MASTER_VERTICAL[j];
                master[WEDGE_VERTICAL][i + 1][j] = WEDGE_MASTER_VERTICAL[j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let msk = master[WEDGE_OBLIQUE63][i][j];
                master[WEDGE_OBLIQUE27][j][i] = msk;
                master[WEDGE_OBLIQUE117][i][n - 1 - j] = 64 - msk;
                master[WEDGE_OBLIQUE153][n - 1 - j][i] = 64 - msk;
                master[WEDGE_HORIZONTAL][j][i] = master[WEDGE_VERTICAL][i][j];
            }
        }
        let mut out: Vec<[Vec<Vec<u8>>; 2]> = Vec::new();
        for bsize in 0..BLOCK_SIZES {
            let mut entry: [Vec<Vec<u8>>; 2] = [Vec::new(), Vec::new()];
            if WEDGE_BITS[bsize] > 0 {
                let (w, h) = (block_width(bsize as u8), block_height(bsize as u8));
                let shape = if h > w {
                    0
                } else if h < w {
                    1
                } else {
                    2
                };
                entry = [vec![Vec::new(); 16], vec![Vec::new(); 16]];
                for wedge in 0..16 {
                    let [dir, xo, yo] = WEDGE_CODEBOOK[shape][wedge];
                    let xoff = n / 2 - ((xo as usize * w) >> 3);
                    let yoff = n / 2 - ((yo as usize * h) >> 3);
                    let mut sum = 0usize;
                    for i in 0..w {
                        sum += master[dir as usize][yoff][xoff + i] as usize;
                    }
                    for i in 1..h {
                        sum += master[dir as usize][yoff + i][xoff] as usize;
                    }
                    let avg = (sum + (w + h - 1) / 2) / (w + h - 1);
                    let flip = (avg < 32) as usize;
                    let mut a = vec![0u8; w * h];
                    let mut b = vec![0u8; w * h];
                    for i in 0..h {
                        for j in 0..w {
                            let m = master[dir as usize][yoff + i][xoff + j];
                            a[i * w + j] = m;
                            b[i * w + j] = 64 - m;
                        }
                    }
                    entry[flip][wedge] = a;
                    entry[1 - flip][wedge] = b;
                }
            }
            out.push(entry);
        }
        out
    })
}

/// The rounding variables of the inter prediction process (InterRound0, InterRound1, InterPostRound).
#[derive(Clone, Copy)]
pub(crate) struct Rounding {
    pub r0: i32,
    pub r1: i32,
    pub post: i32,
}

impl Rounding {
    pub(crate) fn new(bit_depth: i32, is_compound: bool) -> Rounding {
        let r0 = if bit_depth == 12 { 5 } else { 3 };
        let r1 = match (is_compound, bit_depth == 12) {
            (true, _) => 7,
            (false, true) => 9,
            (false, false) => 11,
        };
        Rounding { r0, r1, post: 14 - (r0 + r1) }
    }
}

impl<C: Coder> TileCoder<'_, C> {
    /// `compute_prediction` for an inter block: predicts the planes selected by `planes` into the frame.
    pub(crate) fn predict_inter_block(&mut self, b: &Blk, planes: u8) -> Result<()> {
        if b.use_intrabc {
            self.predict_intrabc(b, planes);
            return Ok(());
        }
        self.local_warp = None;
        if std::env::var_os("AV1_TOOLS").is_some() {
            eprintln!(
                "TOOL mm={} ct={} ii={} gm={} comp={}",
                b.motion_mode,
                b.compound_type,
                b.interintra,
                matches!(b.y_mode, GLOBALMV | GLOBAL_GLOBALMV)
                    && self.f.hdr.gm_type[b.ref_frame[0].max(0) as usize] > 1,
                b.ref_frame[1] > 0
            );
        }
        if b.motion_mode == LOCALWARP {
            self.local_warp = self.warp_estimation(b);
        }
        let np = if b.has_chroma { 3 } else { 1 };
        for plane in 0..np {
            if (planes >> plane) & 1 == 0 {
                continue;
            }
            let (sx, sy) = self.plane_sub(plane);
            let plane_sz = if plane == 0 { b.bsize } else { subsampled_size(b.bsize, sx as u8, sy as u8) };
            let n4w = NUM_4X4_WIDE[plane_sz as usize] as usize;
            let n4h = NUM_4X4_HIGH[plane_sz as usize] as usize;
            let base_x = (b.mi_col >> sx) * 4;
            let base_y = (b.mi_row >> sy) * 4;
            let mut cand_row = (b.mi_row >> sy) << sy;
            let mut cand_col = (b.mi_col >> sx) << sx;
            if b.interintra {
                self.predict_interintra_base(b, plane, base_x, base_y, n4w, n4h);
            }
            let mut pred_w = block_width(b.bsize) >> sx;
            let mut pred_h = block_height(b.bsize) >> sy;
            // A chroma block covering several luma blocks takes each one's motion unless one of them is intra.
            let mut some_use_intra = false;
            for r in 0..n4h << sy {
                for c in 0..n4w << sx {
                    // Positions past the frame edge belong to the current (inter) block.
                    if cand_row + r >= self.f.mi_rows || cand_col + c >= self.f.mi_cols {
                        continue;
                    }
                    if self.motion_at(b, cand_row + r, cand_col + c).0[0] == INTRA_FRAME {
                        some_use_intra = true;
                    }
                }
            }
            if some_use_intra {
                pred_w = n4w * 4;
                pred_h = n4h * 4;
                cand_row = b.mi_row;
                cand_col = b.mi_col;
            }
            let mut r = 0;
            let mut y = 0;
            while y < n4h * 4 {
                let mut c = 0;
                let mut x = 0;
                while x < n4w * 4 {
                    self.predict_inter(b, plane, base_x + x, base_y + y, pred_w, pred_h, cand_row + r, cand_col + c)?;
                    x += pred_w;
                    c += 1;
                }
                y += pred_h;
                r += 1;
            }
        }
        Ok(())
    }

    /// Reference frames, vectors and filters of the 4x4 block at (`row`, `col`): the current block's own values
    /// (not stored yet while it is being predicted) or what was stored for an earlier block.
    fn motion_at(&self, b: &Blk, row: usize, col: usize) -> ([i8; 2], [[i32; 2]; 2], [u8; 2]) {
        let (bh4, bw4) = (NUM_4X4_HIGH[b.bsize as usize] as usize, NUM_4X4_WIDE[b.bsize as usize] as usize);
        if row >= b.mi_row && row < b.mi_row + bh4 && col >= b.mi_col && col < b.mi_col + bw4 {
            return (b.ref_frame, b.mv, b.interp_filter);
        }
        let i = self.mi(row, col);
        (self.f.ref_frame[i], self.f.mvs[i], self.f.interp_filter[i])
    }

    /// The intra prediction that an inter-intra block blends its inter prediction with (written to the frame first).
    fn predict_interintra_base(&mut self, b: &Blk, plane: usize, x: usize, y: usize, n4w: usize, n4h: usize) {
        let mode = match b.interintra_mode {
            0 => DC_PRED,
            1 => V_PRED,
            2 => H_PRED,
            _ => SMOOTH_PRED,
        };
        let (sx, sy) = self.plane_sub(plane);
        let sb_mask = if self.f.seq.use_128x128_superblock { 31 } else { 15 };
        let sub_row = (b.mi_row & sb_mask) >> sy;
        let sub_col = (b.mi_col & sb_mask) >> sx;
        let bd = &self.block_decoded[plane];
        let have_ar = bd[sub_row][sub_col + n4w + 1];
        let have_bl = bd[sub_row + n4h + 1][sub_col];
        let (have_left, have_above) =
            if plane == 0 { (b.avail_l, b.avail_u) } else { (b.avail_l_chroma, b.avail_u_chroma) };
        self.predict_intra_wh(b, plane, x, y, n4w * 4, n4h * 4, have_left, have_above, have_ar, have_bl, mode);
    }

    /// `predict_inter` for one region using the motion information of the 4x4 block (`cand_row`, `cand_col`).
    #[allow(clippy::too_many_arguments)]
    fn predict_inter(
        &mut self,
        b: &Blk,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        cand_row: usize,
        cand_col: usize,
    ) -> Result<()> {
        let (cand_refs, cand_mvs, cand_filt) = self.motion_at(b, cand_row, cand_col);
        let is_compound = cand_refs[1] > INTRA_FRAME;
        let bd = self.f.seq.color.bit_depth as i32;
        let rnd = Rounding::new(bd, is_compound);
        let max = (1i32 << bd) - 1;
        let is_global_mode = matches!(b.y_mode, GLOBALMV | GLOBAL_GLOBALMV);
        let mut preds: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        for list in 0..1 + is_compound as usize {
            let ref_frame = cand_refs[list];
            let refs = self.f.refs.clone();
            let view = refs
                .as_ref()
                .and_then(|r| r.get(ref_frame))
                .ok_or_else(|| Error::invalid("inter block references a missing frame"))?;
            let gm_valid = is_global_mode
                && self.f.hdr.gm_type[ref_frame as usize] > GM_TRANSLATION
                && setup_shear(&self.f.hdr.gm_params[ref_frame as usize]).is_some();
            let use_warp = if w < 8 || h < 8 || self.f.hdr.force_integer_mv {
                0
            } else if b.motion_mode == LOCALWARP && self.local_warp.is_some() {
                1
            } else if gm_valid && !self.is_scaled(ref_frame) {
                2
            } else {
                0
            };
            preds[list] = match use_warp {
                1 => {
                    let params = self.local_warp.unwrap();
                    self.block_warp(view, &params, plane, x, y, w, h, rnd)
                }
                2 => {
                    let params = self.f.hdr.gm_params[ref_frame as usize];
                    self.block_warp(view, &params, plane, x, y, w, h, rnd)
                }
                _ => self.block_inter(view, plane, x, y, w, h, cand_mvs[list], cand_filt, rnd),
            };
        }
        // Masks.
        let is_inter_intra = b.interintra;
        let (sx, sy) = self.plane_sub(plane);
        if b.compound_type == COMPOUND_WEDGE && plane == 0 {
            let m = &wedge_masks()[b.bsize as usize][b.wedge_sign as usize][b.wedge_index as usize];
            self.mask = m.clone();
            self.mask_stride = block_width(b.bsize);
        } else if b.compound_type == COMPOUND_INTRA {
            self.intra_variant_mask(b, w, h);
        } else if b.compound_type == COMPOUND_DIFFWTD && plane == 0 {
            let shift = (bd - 8) + rnd.post;
            self.mask = vec![0; w * h];
            self.mask_stride = w;
            for i in 0..w * h {
                let diff = (preds[0][i] - preds[1][i]).abs();
                let diff = (diff + ((1 << shift) >> 1)) >> shift;
                let m = (38 + diff / 16).clamp(0, 64);
                self.mask[i] = if b.mask_type != 0 { 64 - m } else { m } as u8;
            }
        }
        let (fwd, bck) =
            if b.compound_type == COMPOUND_DISTANCE && is_compound { self.distance_weights(cand_refs) } else { (0, 0) };
        let intra_mask = b.interintra && !b.wedge_interintra;
        let stride_mask = self.mask_stride;
        let dst = &mut self.f.planes[plane];
        for r in 0..h {
            let o = (y + r) * dst.stride + x;
            for c in 0..w {
                let i = r * w + c;
                let v = if !is_compound && !is_inter_intra {
                    preds[0][i]
                } else if b.compound_type == COMPOUND_AVERAGE {
                    (preds[0][i] + preds[1][i] + (1 << rnd.post)) >> (1 + rnd.post)
                } else if b.compound_type == COMPOUND_DISTANCE {
                    (fwd * preds[0][i] + bck * preds[1][i] + (1 << (3 + rnd.post))) >> (4 + rnd.post)
                } else {
                    let mk = &self.mask;
                    let m = if (sx == 0 && sy == 0) || intra_mask {
                        mk[r * stride_mask + c] as i32
                    } else if sx == 1 && sy == 0 {
                        (mk[r * stride_mask + 2 * c] as i32 + mk[r * stride_mask + 2 * c + 1] as i32 + 1) >> 1
                    } else {
                        (mk[2 * r * stride_mask + 2 * c] as i32
                            + mk[2 * r * stride_mask + 2 * c + 1] as i32
                            + mk[(2 * r + 1) * stride_mask + 2 * c] as i32
                            + mk[(2 * r + 1) * stride_mask + 2 * c + 1] as i32
                            + 2)
                            >> 2
                    };
                    if is_inter_intra {
                        let p0 = ((preds[0][i] + ((1 << rnd.post) >> 1)) >> rnd.post).clamp(0, max);
                        let p1 = dst.data[o + c] as i32;
                        (m * p1 + (64 - m) * p0 + 32) >> 6
                    } else {
                        (m * preds[0][i] + (64 - m) * preds[1][i] + (1 << (5 + rnd.post))) >> (6 + rnd.post)
                    }
                };
                dst.data[o + c] = v.clamp(0, max) as u16;
            }
        }
        if b.motion_mode == OBMC {
            self.overlapped_motion_compensation(b, plane, w, h, rnd)?;
        }
        Ok(())
    }

    /// The intra mode variant mask process.
    fn intra_variant_mask(&mut self, b: &Blk, w: usize, h: usize) {
        let size_scale = 128 / h.max(w);
        self.mask = vec![32; w * h];
        self.mask_stride = w;
        for i in 0..h {
            for j in 0..w {
                self.mask[i * w + j] = match b.interintra_mode {
                    1 => II_WEIGHTS_1D[i * size_scale],
                    2 => II_WEIGHTS_1D[j * size_scale],
                    3 => II_WEIGHTS_1D[i.min(j) * size_scale],
                    _ => 32,
                };
            }
        }
    }

    /// The overlapped motion compensation process (7.11.3.9).
    fn overlapped_motion_compensation(
        &mut self,
        b: &Blk,
        plane: usize,
        w: usize,
        h: usize,
        rnd: Rounding,
    ) -> Result<()> {
        let (sx, sy) = self.plane_sub(plane);
        let plane_sz = if plane == 0 { b.bsize } else { subsampled_size(b.bsize, sx as u8, sy as u8) };
        if b.avail_u && plane_sz >= BLOCK_8X8 {
            let w4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
            let mut x4 = b.mi_col;
            let y4 = b.mi_row;
            let mut count = 0;
            let limit = 4.min(NUM_4X4_WIDE[b.bsize as usize].trailing_zeros() as usize);
            while count < limit && x4 < self.f.mi_cols.min(b.mi_col + w4) {
                let (cand_row, cand_col) = (b.mi_row - 1, x4 | 1);
                let cand_sz = self.f.mi_size[self.mi(cand_row, cand_col)];
                let step4 = (NUM_4X4_WIDE[cand_sz as usize] as usize).clamp(2, 16);
                if self.f.ref_frame[self.mi(cand_row, cand_col)][0] > INTRA_FRAME {
                    count += 1;
                    let pred_w = w.min((step4 * 4) >> sx);
                    let pred_h = (h >> 1).min(32 >> sy);
                    self.predict_overlap(plane, x4, y4, cand_row, cand_col, pred_w, pred_h, 0, rnd)?;
                }
                x4 += step4;
            }
        }
        if b.avail_l {
            let h4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
            let x4 = b.mi_col;
            let mut y4 = b.mi_row;
            let mut count = 0;
            let limit = 4.min(NUM_4X4_HIGH[b.bsize as usize].trailing_zeros() as usize);
            while count < limit && y4 < self.f.mi_rows.min(b.mi_row + h4) {
                let (cand_col, cand_row) = (b.mi_col - 1, y4 | 1);
                let cand_sz = self.f.mi_size[self.mi(cand_row, cand_col)];
                let step4 = (NUM_4X4_HIGH[cand_sz as usize] as usize).clamp(2, 16);
                if self.f.ref_frame[self.mi(cand_row, cand_col)][0] > INTRA_FRAME {
                    count += 1;
                    let pred_w = (w >> 1).min(32 >> sx);
                    let pred_h = h.min((step4 * 4) >> sy);
                    self.predict_overlap(plane, x4, y4, cand_row, cand_col, pred_w, pred_h, 1, rnd)?;
                }
                y4 += step4;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_overlap(
        &mut self,
        plane: usize,
        x4: usize,
        y4: usize,
        cand_row: usize,
        cand_col: usize,
        pred_w: usize,
        pred_h: usize,
        pass: usize,
        rnd: Rounding,
    ) -> Result<()> {
        let (sx, sy) = self.plane_sub(plane);
        let ci = self.mi(cand_row, cand_col);
        let mv = self.f.mvs[ci][0];
        let rf = self.f.ref_frame[ci][0];
        let filt = self.f.interp_filter[ci];
        let (px, py) = ((x4 * 4) >> sx, (y4 * 4) >> sy);
        let refs = self.f.refs.clone();
        let view = refs
            .as_ref()
            .and_then(|r| r.get(rf))
            .ok_or_else(|| Error::invalid("inter block references a missing frame"))?;
        let obmc = self.block_inter(view, plane, px, py, pred_w, pred_h, mv, filt, rnd);
        let max = (1i32 << self.f.seq.color.bit_depth) - 1;
        let len = if pass == 0 { pred_h } else { pred_w };
        let mask: &[u8] = match len {
            2 => &OBMC_MASK_2,
            4 => &OBMC_MASK_4,
            8 => &OBMC_MASK_8,
            16 => &OBMC_MASK_16,
            _ => &OBMC_MASK_32,
        };
        let dst = &mut self.f.planes[plane];
        for i in 0..pred_h {
            for j in 0..pred_w {
                let m = if pass == 0 { mask[i] } else { mask[j] } as i32;
                let o = (py + i) * dst.stride + px + j;
                let p = obmc[i * pred_w + j].clamp(0, max);
                dst.data[o] = ((m * dst.data[o] as i32 + (64 - m) * p + 32) >> 6) as u16;
            }
        }
        Ok(())
    }

    /// The block inter prediction process (7.11.3.4) with the motion vector scaling process (7.11.3.3).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block_inter(
        &self,
        view: &RefView,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: [i32; 2],
        filt: [u8; 2],
        rnd: Rounding,
    ) -> Vec<i32> {
        let hdr = &self.f.hdr;
        let (cur_w, cur_h) = (hdr.frame_width as i64, hdr.frame_height as i64);
        let x_scale = (((view.upscaled_width as i64) << 14) + cur_w / 2) / cur_w;
        let y_scale = (((view.frame_height as i64) << 14) + cur_h / 2) / cur_h;
        let (psx, psy) = self.plane_sub(plane);
        let (sx, sy) = (psx as i32, psy as i32);
        let orig_x = ((x as i64) << 4) + ((2 * mv[1] as i64) >> sx) + 8;
        let orig_y = ((y as i64) << 4) + ((2 * mv[0] as i64) >> sy) + 8;
        let base_x = orig_x * x_scale - (8 << 14);
        let base_y = orig_y * y_scale - (8 << 14);
        let start_x = round2_signed(base_x, 14 + 4 - 10) + 32;
        let start_y = round2_signed(base_y, 14 + 4 - 10) + 32;
        let step_x = round2_signed(x_scale, 14 - 10);
        let step_y = round2_signed(y_scale, 14 - 10);
        let rp = &view.frame.planes[plane];
        let last_x = (((view.upscaled_width as i64) + sx as i64) >> sx) - 1;
        let last_y = (((view.frame_height as i64) + sy as i64) >> sy) - 1;
        let inter_h = ((((h as i64 - 1) * step_y + (1 << 10) - 1) >> 10) + 8) as usize;
        let mut fh = filt[1] as usize;
        if w <= 4 {
            fh = narrow_filter(fh);
        }
        let mut fv = filt[0] as usize;
        if h <= 4 {
            fv = narrow_filter(fv);
        }
        let mut inter = vec![0i32; inter_h * w];
        for r in 0..inter_h {
            let yy = (((start_y >> 10) + r as i64 - 3).clamp(0, last_y)) as usize;
            let row = &rp.data[yy * rp.stride..];
            for c in 0..w {
                let p = start_x + step_x * c as i64;
                let k = &SUBPEL_FILTERS[fh][((p >> 6) & 15) as usize];
                let mut s = 0i32;
                for (t, &kt) in k.iter().enumerate() {
                    let xx = (((p >> 10) + t as i64 - 3).clamp(0, last_x)) as usize;
                    s += kt as i32 * row[xx] as i32;
                }
                inter[r * w + c] = (s + (1 << (rnd.r0 - 1))) >> rnd.r0;
            }
        }
        let mut pred = vec![0i32; w * h];
        for r in 0..h {
            let p = (start_y & 1023) + step_y * r as i64;
            let k = &SUBPEL_FILTERS[fv][((p >> 6) & 15) as usize];
            let base = (p >> 10) as usize;
            for c in 0..w {
                let mut s = 0i32;
                for (t, &kt) in k.iter().enumerate() {
                    s += kt as i32 * inter[(base + t) * w + c];
                }
                pred[r * w + c] = (s + (1 << (rnd.r1 - 1))) >> rnd.r1;
            }
        }
        pred
    }

    /// The block warp process (7.11.3.5) over a whole `w` x `h` region.
    #[allow(clippy::too_many_arguments)]
    fn block_warp(
        &self,
        view: &RefView,
        params: &[i32; 6],
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        rnd: Rounding,
    ) -> Vec<i32> {
        let (psx, psy) = self.plane_sub(plane);
        let (sx, sy) = (psx as i64, psy as i64);
        let rp = &view.frame.planes[plane];
        let last_x = ((view.upscaled_width as i64 + sx) >> sx) - 1;
        let last_y = ((view.frame_height as i64 + sy) >> sy) - 1;
        let (alpha, beta, gamma, delta) = setup_shear(params).expect("warp parameters were validated");
        let mut pred = vec![0i32; w * h];
        for i8 in 0..=(h - 1) >> 3 {
            for j8 in 0..=(w - 1) >> 3 {
                let src_x = ((x + j8 * 8 + 4) as i64) << sx;
                let src_y = ((y + i8 * 8 + 4) as i64) << sy;
                let dst_x = params[2] as i64 * src_x + params[3] as i64 * src_y + params[0] as i64;
                let dst_y = params[4] as i64 * src_x + params[5] as i64 * src_y + params[1] as i64;
                let x4 = dst_x >> sx;
                let y4 = dst_y >> sy;
                let ix4 = x4 >> WARPEDMODEL_PREC_BITS;
                let sx4 = (x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
                let iy4 = y4 >> WARPEDMODEL_PREC_BITS;
                let sy4 = (y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
                let mut inter = [[0i32; 8]; 15];
                for i1 in -7i32..8 {
                    let yy = (iy4 + i1 as i64).clamp(0, last_y) as usize;
                    let row = &rp.data[yy * rp.stride..];
                    for i2 in -4i32..4 {
                        let sxv = sx4 + alpha * i2 + beta * i1;
                        let offs = (((sxv + (1 << (WARPEDDIFF_PREC_BITS - 1))) >> WARPEDDIFF_PREC_BITS)
                            + WARPEDPIXEL_PREC_SHIFTS) as usize;
                        let mut s = 0i32;
                        for i3 in 0..8i64 {
                            let xx = (ix4 + i2 as i64 - 3 + i3).clamp(0, last_x) as usize;
                            s += WARPED_FILTERS[offs][i3 as usize] as i32 * row[xx] as i32;
                        }
                        inter[(i1 + 7) as usize][(i2 + 4) as usize] = (s + (1 << (rnd.r0 - 1))) >> rnd.r0;
                    }
                }
                for i1 in -4i32..4.min(h as i32 - i8 as i32 * 8 - 4) {
                    for i2 in -4i32..4.min(w as i32 - j8 as i32 * 8 - 4) {
                        let syv = sy4 + gamma * i2 + delta * i1;
                        let offs = (((syv + (1 << (WARPEDDIFF_PREC_BITS - 1))) >> WARPEDDIFF_PREC_BITS)
                            + WARPEDPIXEL_PREC_SHIFTS) as usize;
                        let mut s = 0i32;
                        for i3 in 0..8i32 {
                            s += WARPED_FILTERS[offs][i3 as usize] as i32
                                * inter[(i1 + i3 + 4) as usize][(i2 + 4) as usize];
                        }
                        let o = ((i8 * 8) as i32 + i1 + 4) as usize * w + ((j8 * 8) as i32 + i2 + 4) as usize;
                        pred[o] = (s + (1 << (rnd.r1 - 1))) >> rnd.r1;
                    }
                }
            }
        }
        pred
    }

    /// The warp estimation process (7.11.3.8): `LocalWarpParams`, or `None` when `LocalValid` is 0.
    fn warp_estimation(&self, b: &Blk) -> Option<[i32; 6]> {
        let (mut a00, mut a01, mut a11) = (0i64, 0i64, 0i64);
        let (mut bx0, mut bx1, mut by0, mut by1) = (0i64, 0i64, 0i64, 0i64);
        let w4 = NUM_4X4_WIDE[b.bsize as usize] as i32;
        let h4 = NUM_4X4_HIGH[b.bsize as usize] as i32;
        let mid_y = b.mi_row as i32 * 4 + h4 * 2 - 1;
        let mid_x = b.mi_col as i32 * 4 + w4 * 2 - 1;
        let suy = mid_y * 8;
        let sux = mid_x * 8;
        let duy = suy + b.mv[0][0];
        let dux = sux + b.mv[0][1];
        let ls = |a: i32, b: i32| -> i64 { (((a as i64) * (b as i64)) >> 2) + (a + b) as i64 };
        for cand in self.warp_cands.iter().take(self.num_samples) {
            let sy = cand[0] - suy;
            let sx = cand[1] - sux;
            let dy = cand[2] - duy;
            let dx = cand[3] - dux;
            if (sx - dx).abs() < LS_MV_MAX && (sy - dy).abs() < LS_MV_MAX {
                a00 += ls(sx, sx) + 8;
                a01 += ls(sx, sy) + 4;
                a11 += ls(sy, sy) + 8;
                bx0 += ls(sx, dx) + 8;
                bx1 += ls(sy, dx) + 4;
                by0 += ls(sx, dy) + 4;
                by1 += ls(sy, dy) + 8;
            }
        }
        let det = a00 * a11 - a01 * a01;
        if det == 0 {
            return None;
        }
        let (mut div_shift, mut div_factor) = resolve_divisor(det);
        div_shift -= WARPEDMODEL_PREC_BITS as i32;
        if div_shift < 0 {
            div_factor <<= -div_shift;
            div_shift = 0;
        }
        let r2s = |v: i128| -> i64 {
            let p = v * div_factor as i128;
            let sh = div_shift as u32;
            let r = if sh == 0 {
                p
            } else if p >= 0 {
                (p + (1i128 << (sh - 1))) >> sh
            } else {
                -((-p + (1i128 << (sh - 1))) >> sh)
            };
            r.clamp(i64::MIN as i128, i64::MAX as i128) as i64
        };
        let nondiag = |v: i128| -> i32 {
            r2s(v).clamp(-WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1, WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1) as i32
        };
        let one = 1i64 << WARPEDMODEL_PREC_BITS;
        let diag = |v: i128| -> i32 {
            r2s(v).clamp(one - WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1, one + WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1) as i32
        };
        let (a00, a01, a11, bx0, bx1, by0, by1) =
            (a00 as i128, a01 as i128, a11 as i128, bx0 as i128, bx1 as i128, by0 as i128, by1 as i128);
        let mut p = [0i32; 6];
        p[2] = diag(a11 * bx0 - a01 * bx1);
        p[3] = nondiag(-a01 * bx0 + a00 * bx1);
        p[4] = nondiag(a11 * by0 - a01 * by1);
        p[5] = diag(-a01 * by0 + a00 * by1);
        let (mvx, mvy) = (b.mv[0][1] as i64, b.mv[0][0] as i64);
        let vx = mvx * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x as i64 * (p[2] as i64 - one) + mid_y as i64 * p[3] as i64);
        let vy = mvy * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x as i64 * p[4] as i64 + mid_y as i64 * (p[5] as i64 - one));
        p[0] = vx.clamp(-WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1) as i32;
        p[1] = vy.clamp(-WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1) as i32;
        setup_shear(&p)?;
        Some(p)
    }

    /// The distance weights process: (`FwdWeight`, `BckWeight`).
    fn distance_weights(&self, refs: [i8; 2]) -> (i32, i32) {
        let h = &self.f.hdr;
        let seq = &self.f.seq;
        let mut dist = [0i32; 2];
        for (d, &rf) in dist.iter_mut().zip(refs.iter()) {
            let hint = h.order_hints[rf as usize];
            *d = FrameHeader::relative_dist(seq, hint, h.order_hint).abs().clamp(0, MAX_FRAME_DISTANCE);
        }
        let (d0, d1) = (dist[1], dist[0]);
        let order = (d0 <= d1) as usize;
        let mut i = 3;
        if d0 != 0 && d1 != 0 {
            i = 0;
            while i < 3 {
                let c0 = QUANT_DIST_WEIGHT[i][order];
                let c1 = QUANT_DIST_WEIGHT[i][1 - order];
                if (order == 1 && d0 * c0 > d1 * c1) || (order == 0 && d0 * c0 < d1 * c1) {
                    break;
                }
                i += 1;
            }
        }
        (QUANT_DIST_LOOKUP[i][order], QUANT_DIST_LOOKUP[i][1 - order])
    }
}

/// The resolve divisor process (7.11.3.7): (`divShift`, `divFactor`).
fn resolve_divisor(d: i64) -> (i32, i64) {
    let a = d.unsigned_abs();
    let n = 63 - a.leading_zeros();
    let e = a - (1u64 << n);
    let f = if n > DIV_LUT_BITS {
        (e + (1u64 << (n - DIV_LUT_BITS - 1))) >> (n - DIV_LUT_BITS)
    } else {
        e << (DIV_LUT_BITS - n)
    };
    let factor = DIV_LUT[f as usize] as i64;
    ((n + DIV_LUT_PREC_BITS) as i32, if d < 0 { -factor } else { factor })
}

/// The setup shear process (7.11.3.6): `(alpha, beta, gamma, delta)` when the parameters are valid.
fn setup_shear(p: &[i32; 6]) -> Option<(i32, i32, i32, i32)> {
    let one = 1i64 << WARPEDMODEL_PREC_BITS;
    let alpha0 = (p[2] as i64 - one).clamp(-32768, 32767);
    let beta0 = (p[3] as i64).clamp(-32768, 32767);
    let (div_shift, div_factor) = resolve_divisor(p[2] as i64);
    let r2s = |v: i128| -> i64 {
        let sh = div_shift as u32;
        let q = v * div_factor as i128;
        (if q >= 0 { (q + (1i128 << (sh - 1))) >> sh } else { -((-q + (1i128 << (sh - 1))) >> sh) }) as i64
    };
    let v = (p[4] as i128) << WARPEDMODEL_PREC_BITS;
    let gamma0 = r2s(v).clamp(-32768, 32767);
    let w = p[3] as i128 * p[4] as i128;
    let delta0 = (p[5] as i64 - r2s(w) - one).clamp(-32768, 32767);
    let red = |x: i64| -> i32 {
        let r = WARP_PARAM_REDUCE_BITS;
        let q = if x >= 0 { (x + (1 << (r - 1))) >> r } else { -((-x + (1 << (r - 1))) >> r) };
        (q << r) as i32
    };
    let (alpha, beta, gamma, delta) = (red(alpha0), red(beta0), red(gamma0), red(delta0));
    if 4 * alpha.abs() as i64 + 7 * beta.abs() as i64 >= one || 4 * gamma.abs() as i64 + 4 * delta.abs() as i64 >= one {
        return None;
    }
    Some((alpha, beta, gamma, delta))
}

fn narrow_filter(f: usize) -> usize {
    match f as u8 {
        EIGHTTAP | EIGHTTAP_SHARP => 4,
        EIGHTTAP_SMOOTH => 5,
        _ => f,
    }
}

fn round2_signed(x: i64, n: u32) -> i64 {
    if x >= 0 { (x + (1 << (n - 1))) >> n } else { -((-x + (1 << (n - 1))) >> n) }
}
