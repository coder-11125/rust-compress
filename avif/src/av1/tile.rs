//! Tile decoding: partitions, mode info, coefficients and reconstruction of intra frames
//! (AV1 spec sections 5.11 and 7).
//!
//! The syntax is written once, generically over [`Coder`]: with a [`SymbolDecoder`] it parses a
//! stream, with a [`SymbolEncoder`] the same code writes the syntax elements held in [`Blk`] and the
//! coefficient blocks supplied by the encoder, so both sides always agree.

use super::cdf_tables::Cdfs;
use super::coef_tables::*;
use super::consts::*;
use super::entropy::Coder;
use super::frame::Plane;
use super::ipred;
use super::itx;
use super::obu::{FrameHeader, SequenceHeader};
use super::obu::{RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use rc_core::error::{Error, Result};

/// One restoration unit's parameters as read from the stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct LrUnitParams {
    /// RESTORE_NONE / RESTORE_WIENER / RESTORE_SGRPROJ
    pub kind: u8,
    pub wiener: [[i8; 3]; 2],
    pub sgr_set: u8,
    pub sgr_xqd: [i8; 2],
}

impl LrUnitParams {
    /// The parameters the encoder's restoration search chose for a unit.
    pub fn from_unit(u: &super::lr::LrUnit) -> LrUnitParams {
        use super::lr::LrType;
        LrUnitParams {
            kind: match u.kind {
                LrType::None => RESTORE_NONE,
                LrType::Wiener => RESTORE_WIENER,
                LrType::Sgrproj => RESTORE_SGRPROJ,
            },
            wiener: u.wiener,
            sgr_set: u.sgr_set,
            sgr_xqd: u.sgr_xqd,
        }
    }
}

/// Everything decoded for a frame that later stages (loop filters, output) need.
#[derive(Clone)]
pub struct FrameState {
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
    pub planes: Vec<Plane>,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub mi_size: Vec<u8>,
    pub skip: Vec<u8>,
    pub y_mode: Vec<u8>,
    pub uv_mode: Vec<u8>,
    pub seg_id: Vec<u8>,
    pub tx_size: Vec<u8>,
    pub is_inter: Vec<u8>,
    pub skip_mode: Vec<u8>,
    pub comp_group_idx: Vec<u8>,
    pub compound_idx: Vec<u8>,
    pub interp_filter: Vec<[u8; 2]>,
    /// `RefFrames` (intra block copy blocks have `INTRA_FRAME` and the vector in `mvs`), `Mvs`, `InterTxSizes` and
    /// `TxTypes` (luma, at 4x4 granularity).
    pub ref_frame: Vec<[i8; 2]>,
    pub mvs: Vec<[[i32; 2]; 2]>,
    pub inter_tx_size: Vec<u8>,
    pub tx_types: Vec<u8>,
    pub pal_size: [Vec<u8>; 2],
    pub delta_lf: Vec<[i8; 4]>,
    /// Loop-filter transform sizes per plane, at the plane's 4x4 grid (`lf_stride[plane]` wide).
    pub lf_tx: [Vec<u8>; 3],
    pub lf_stride: [usize; 3],
    /// `cdef_idx` per 64x64 luma block, -1 = not coded.
    pub cdef_idx: Vec<i8>,
    pub cdef_stride: usize,
    /// Restoration units per plane: (units_per_row, units_per_col, params).
    pub lr_units: [(usize, usize, Vec<LrUnitParams>); 3],
    /// Reference frames for inter prediction (None in intra frames) and the previous segmentation map.
    pub refs: Option<std::sync::Arc<super::refs::RefSet>>,
    pub prev_segment_ids: Option<std::sync::Arc<Vec<u8>>>,
    /// `MotionFieldMvs[ref][y8 * (mi_cols >> 1) + x8]`, empty unless `use_ref_frame_mvs`.
    pub motion_field: Vec<Vec<[i16; 2]>>,
    /// The CDFs every tile of the frame starts from.
    pub cdf_base: Box<Cdfs>,
    /// Palette colours of the block covering each position of the row above / column to the left.
    pub num_planes: usize,
    pub sub_x: usize,
    pub sub_y: usize,
}

impl FrameState {
    pub fn new(seq: &SequenceHeader, hdr: &FrameHeader) -> Result<FrameState> {
        let c = &seq.color;
        let (mi_cols, mi_rows) = (hdr.mi_cols, hdr.mi_rows);
        let num_planes = c.num_planes();
        let (sub_x, sub_y) = (c.sub_x as usize, c.sub_y as usize);
        // Blocks may overhang the picture by up to a superblock; pad the buffers to a multiple of 128.
        let aw = (mi_cols * 4).next_multiple_of(128);
        let ah = (mi_rows * 4).next_multiple_of(128);
        let mut planes = Vec::new();
        for p in 0..num_planes {
            let (w, h) = if p == 0 { (aw, ah) } else { (aw >> sub_x, ah >> sub_y) };
            planes.push(Plane::new(w, h));
        }
        let n = mi_cols * mi_rows;
        let mut lf_stride = [0usize; 3];
        let mut lf_tx: [Vec<u8>; 3] = Default::default();
        for p in 0..num_planes {
            let (sx, sy) = if p == 0 { (0, 0) } else { (sub_x, sub_y) };
            let w = (mi_cols + sx) >> sx;
            let h = (mi_rows + sy) >> sy;
            lf_stride[p] = w;
            lf_tx[p] = vec![0; w * h];
        }
        let cdef_stride = mi_cols.div_ceil(16);
        let cdef_rows = mi_rows.div_ceil(16);
        let mut lr_units: [(usize, usize, Vec<LrUnitParams>); 3] = Default::default();
        for p in 0..num_planes {
            if hdr.lr.kind[p] != RESTORE_NONE {
                let (sx, sy) = if p == 0 { (0, 0) } else { (sub_x, sub_y) };
                let unit = hdr.lr.unit_size[p];
                let rows = count_units(unit, (hdr.frame_height + sy) >> sy);
                let cols = count_units(unit, (hdr.upscaled_width + sx) >> sx);
                lr_units[p] = (cols, rows, vec![LrUnitParams::default(); rows * cols]);
            }
        }
        Ok(FrameState {
            seq: seq.clone(),
            hdr: hdr.clone(),
            planes,
            mi_cols,
            mi_rows,
            mi_size: vec![0; n],
            skip: vec![0; n],
            y_mode: vec![0; n],
            uv_mode: vec![0; n],
            seg_id: vec![0; n],
            tx_size: vec![0; n],
            is_inter: vec![0; n],
            skip_mode: vec![0; n],
            comp_group_idx: vec![0; n],
            compound_idx: vec![0; n],
            interp_filter: vec![[0; 2]; n],
            ref_frame: vec![[super::mvpred::UNWRITTEN, NONE_FRAME]; n],
            mvs: vec![[[0; 2]; 2]; n],
            inter_tx_size: vec![0; n],
            tx_types: vec![0; n],
            pal_size: [vec![0; n], vec![0; n]],
            delta_lf: vec![[0; 4]; n],
            lf_tx,
            lf_stride,
            cdef_idx: vec![-1; cdef_stride * cdef_rows],
            cdef_stride,
            lr_units,
            refs: None,
            prev_segment_ids: None,
            motion_field: Vec::new(),
            cdf_base: Cdfs::new(hdr.quant.base_q_idx as u32),
            num_planes,
            sub_x,
            sub_y,
        })
    }
}

pub fn count_units(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// Syntax elements of the block being coded.
#[derive(Clone, Debug, Default)]
pub struct Blk {
    pub mi_row: usize,
    pub mi_col: usize,
    pub bsize: u8,
    pub has_chroma: bool,
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub skip: bool,
    pub segment_id: u8,
    pub lossless: bool,
    pub use_intrabc: bool,
    /// Inter-style block: block copy or inter prediction.
    pub is_inter: bool,
    pub ref_frame: [i8; 2],
    pub skip_mode: bool,
    pub interp_filter: [u8; 2],
    pub motion_mode: u8,
    pub compound_type: u8,
    pub interintra: bool,
    pub interintra_mode: u8,
    pub wedge_interintra: bool,
    pub wedge_index: u8,
    pub wedge_sign: u8,
    pub mask_type: u8,
    pub comp_group_idx: u8,
    pub compound_idx: u8,
    pub ref_mv_idx: u8,
    /// Intra block copy displacement vector (row, column) in 1/8 sample units, in list 0.
    pub mv: [[i32; 2]; 2],
    pub y_mode: u8,
    pub uv_mode: u8,
    pub angle_y: i8,
    pub angle_uv: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub pal_size_y: u8,
    pub pal_size_uv: u8,
    pub pal_y: [u16; 8],
    pub pal_u: [u16; 8],
    pub pal_v: [u16; 8],
    pub pal_v_delta_encode: bool,
    pub tx_size: u8,
    /// Encoder inputs (ignored when decoding): the `cdef_idx` and the reduced delta-q / delta-lf values.
    pub cdef_value: u32,
    pub delta_q: i32,
    pub delta_lf_values: [i32; 4],
    /// Encoder: the coefficients chosen for this block by the search, reused instead of analysing it again.
    pub pre: Option<std::sync::Arc<Vec<StoredCoefs>>>,
}

/// Quantized coefficients of one transform block in the specification's raster order
/// (`quant[row * w + col]` over the coded, at most 32x32, area).
pub struct CoefBlock {
    pub quant: [i32; 1024],
    pub eob: usize,
    pub tx_type: u8,
}

impl Default for CoefBlock {
    fn default() -> Self {
        CoefBlock { quant: [0; 1024], eob: 0, tx_type: 0 }
    }
}

/// Quantized coefficients of one transform block in compact form, for replay by the encoder.
#[derive(Clone, Debug, Default)]
pub struct StoredCoefs {
    pub tx_type: u8,
    pub eob: usize,
    /// Non-zero coefficients as (position, value).
    pub coefs: Vec<(u16, i32)>,
    /// Encoder: hash of the residual these were chosen for (0 when unknown), to tell whether they still apply.
    pub hash: u64,
}

const SB_MI_MAX: usize = 32;

pub struct TileCoder<'a, C: Coder> {
    pub f: &'a mut FrameState,
    pub cdf: Box<Cdfs>,
    pub c: C,
    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
    above_level: [Vec<u8>; 3],
    above_dc: [Vec<u8>; 3],
    left_level: [Vec<u8>; 3],
    left_dc: [Vec<u8>; 3],
    above_pal: [Vec<[u16; 8]>; 2],
    left_pal: [Vec<[u16; 8]>; 2],
    pub(crate) above_seg_pred: Vec<u8>,
    pub(crate) left_seg_pred: Vec<u8>,
    /// Local warp candidates of the block being parsed (`CandList`).
    pub(crate) warp_cands: Vec<[i32; 4]>,
    pub(crate) trace: bool,
    pub(crate) num_samples: usize,
    pub(crate) local_warp: Option<[i32; 6]>,
    pub(crate) mask: Vec<u8>,
    pub(crate) mask_stride: usize,
    /// Set when the stream uses a tool this decoder does not implement.
    pub(crate) unsupported: Option<&'static str>,
    /// `BlockDecoded[plane][y + 1][x + 1]` for the current superblock.
    pub(crate) block_decoded: [[[bool; SB_MI_MAX + 3]; SB_MI_MAX + 3]; 3],
    pub current_q_index: i32,
    pub delta_lf: [i32; 4],
    pub read_deltas: bool,
    pub(crate) ref_sgr_xqd: [[i32; 2]; 3],
    pub(crate) ref_lr_wiener: [[[i32; 3]; 2]; 3],
    pub(crate) max_luma_w: usize,
    pub(crate) max_luma_h: usize,
    color_map_y: Vec<u8>,
    color_map_uv: Vec<u8>,
    pub cb: Box<CoefBlock>,
    /// Encoder replay queues: partition decisions, blocks and per-transform-block coefficients, consumed in the
    /// order the decoder would parse them.
    pub part_queue: std::collections::VecDeque<u8>,
    pub blk_queue: std::collections::VecDeque<Blk>,
    pub coef_queue: std::collections::VecDeque<StoredCoefs>,
    /// Encoder analysis: when set, transform blocks get their coefficients from `enc_src` instead of the bitstream.
    pub analyzing: bool,
    pub analysis_log: Vec<StoredCoefs>,
    /// Replay the queues instead of planning; when planning, decisions are appended to the logs.
    pub replay: bool,
    pub part_log: Vec<u8>,
    pub blk_log: Vec<Blk>,
    pub coef_log: Vec<StoredCoefs>,
    pub enc_src: Option<super::encode::EncSrc>,
    /// Decisions of the superblock being coded, produced by the search.
    pub part_cache: std::collections::VecDeque<u8>,
    pub blk_cache: std::collections::VecDeque<Blk>,
    /// Analysis accumulators: planes to analyse (bit mask), distortion and coefficient bits (1/256 bit).
    pub stats: bool,
    pub dbg_snap: Option<Vec<Vec<u16>>>,
    pub an_planes: u8,
    /// Whether the transform-type search tries every candidate type (off while ranking modes: DCT only).
    pub full_tx: bool,
    /// Encoder: the displacement vector of the most recently planned block copy block.
    pub last_dv: [i32; 2],
    /// Coefficients still to be applied from a block's `pre` list while analysing it.
    pub pre_coefs: std::collections::VecDeque<StoredCoefs>,
    pub an_dist: f64,
    pub an_bits: u64,
}

impl<'a, C: Coder> TileCoder<'a, C> {
    pub fn new(
        f: &'a mut FrameState,
        cdf: Box<Cdfs>,
        c: C,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> TileCoder<'a, C> {
        let mi_cols = f.mi_cols;
        let mi_rows = f.mi_rows;
        let np = f.num_planes;
        let mk = |n: usize| [vec![0u8; n + 64], vec![0u8; n + 64], vec![0u8; n + 64]];
        let q = f.hdr.quant.base_q_idx as i32;
        TileCoder {
            f,
            cdf,
            c,
            mi_row_start: rows.0,
            mi_row_end: rows.1,
            mi_col_start: cols.0,
            mi_col_end: cols.1,
            above_level: mk(mi_cols),
            above_dc: mk(mi_cols),
            left_level: mk(mi_rows),
            left_dc: mk(mi_rows),
            above_pal: [vec![[0; 8]; mi_cols + 32], vec![[0; 8]; mi_cols + 32]],
            left_pal: [vec![[0; 8]; mi_rows + 32], vec![[0; 8]; mi_rows + 32]],
            above_seg_pred: vec![0; mi_cols + 32],
            left_seg_pred: vec![0; mi_rows + 32],
            warp_cands: Vec::new(),
            trace: std::env::var_os("AV1_TRACE").is_some(),
            num_samples: 0,
            local_warp: None,
            mask: Vec::new(),
            mask_stride: 0,
            unsupported: None,
            block_decoded: [[[false; SB_MI_MAX + 3]; SB_MI_MAX + 3]; 3],
            current_q_index: q,
            delta_lf: [0; 4],
            read_deltas: false,
            ref_sgr_xqd: [[-32, 31]; 3],
            ref_lr_wiener: [[[3, -7, 15]; 2]; 3],
            max_luma_w: 0,
            max_luma_h: 0,
            color_map_y: vec![0; 64 * 64],
            color_map_uv: vec![0; 64 * 64],
            cb: Box::default(),
            part_queue: Default::default(),
            blk_queue: Default::default(),
            coef_queue: Default::default(),
            analyzing: false,
            analysis_log: Vec::new(),
            replay: false,
            part_log: Vec::new(),
            blk_log: Vec::new(),
            coef_log: Vec::new(),
            enc_src: None,
            part_cache: Default::default(),
            blk_cache: Default::default(),
            stats: std::env::var_os("AV1_STATS").is_some(),
            dbg_snap: None,
            an_planes: 7,
            full_tx: true,
            last_dv: [0, 0],
            pre_coefs: Default::default(),
            an_dist: 0.0,
            an_bits: 0,
        }
        .with_planes(np)
    }

    fn with_planes(self, _np: usize) -> Self {
        self
    }

    pub(crate) fn sb_size(&self) -> u8 {
        if self.f.seq.use_128x128_superblock { BLOCK_128X128 } else { BLOCK_64X64 }
    }

    pub(crate) fn is_inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    #[inline]
    /// Record a block's mode info in the per-position arrays that later blocks use as context.
    pub(crate) fn store_block_info(&mut self, b: &Blk) {
        let (r, c) = (b.mi_row, b.mi_col);
        let bsize = b.bsize;
        let bw4 = NUM_4X4_WIDE[bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[bsize as usize] as usize;
        let (rows, cols) = (self.f.mi_rows, self.f.mi_cols);
        let dlf = [self.delta_lf[0] as i8, self.delta_lf[1] as i8, self.delta_lf[2] as i8, self.delta_lf[3] as i8];
        for y in 0..bh4 {
            if r + y >= rows {
                break;
            }
            for x in 0..bw4 {
                if c + x >= cols {
                    break;
                }
                let i = (r + y) * cols + c + x;
                self.f.mi_size[i] = b.bsize;
                self.f.skip[i] = b.skip as u8;
                self.f.y_mode[i] = b.y_mode;
                self.f.uv_mode[i] = b.uv_mode;
                self.f.seg_id[i] = b.segment_id;
                self.f.tx_size[i] = b.tx_size;
                self.f.is_inter[i] = b.is_inter as u8;
                self.f.skip_mode[i] = b.skip_mode as u8;
                self.f.comp_group_idx[i] = b.comp_group_idx;
                self.f.compound_idx[i] = b.compound_idx;
                self.f.interp_filter[i] = b.interp_filter;
                self.f.ref_frame[i] = b.ref_frame;
                if b.is_inter {
                    self.f.mvs[i] = b.mv;
                }
                self.f.pal_size[0][i] = b.pal_size_y;
                self.f.pal_size[1][i] = b.pal_size_uv;
                self.f.delta_lf[i] = dlf;
            }
        }
    }

    pub(crate) fn mi(&self, r: usize, c: usize) -> usize {
        r * self.f.mi_cols + c
    }

    /// Decode all superblocks of the tile (spec `decode_tile`).
    pub fn decode_tile(&mut self) -> Result<()> {
        let sb4 = NUM_4X4_WIDE[self.sb_size() as usize] as usize;
        let mut r = self.mi_row_start;
        while r < self.mi_row_end {
            self.clear_left_context();
            let mut c = self.mi_col_start;
            while c < self.mi_col_end {
                self.read_deltas = self.f.hdr.delta_q_present;
                self.clear_cdef(r, c);
                self.clear_block_decoded_flags(r, c, sb4);
                self.read_lr(r, c, self.sb_size())?;
                if !C::DECODING && !self.replay && self.enc_src.is_some() {
                    self.plan_sb(r, c);
                }
                self.decode_partition(r, c, self.sb_size())?;
                c += sb4;
            }
            r += sb4;
        }
        Ok(())
    }

    pub(crate) fn clear_left_context(&mut self) {
        for p in 0..3 {
            self.left_level[p].fill(0);
            self.left_dc[p].fill(0);
        }
        for p in 0..2 {
            self.left_pal[p].fill([0; 8]);
        }
        self.left_seg_pred.fill(0);
    }

    /// Spec `clear_above_context` is implicit: the contexts start zeroed for every tile.
    pub(crate) fn clear_cdef(&mut self, r: usize, c: usize) {
        let st = self.f.cdef_stride;
        let (r64, c64) = (r >> 4, c >> 4);
        self.f.cdef_idx[r64 * st + c64] = -1;
        if self.f.seq.use_128x128_superblock {
            let rows = self.f.cdef_idx.len() / st;
            if c64 + 1 < st {
                self.f.cdef_idx[r64 * st + c64 + 1] = -1;
            }
            if r64 + 1 < rows {
                self.f.cdef_idx[(r64 + 1) * st + c64] = -1;
                if c64 + 1 < st {
                    self.f.cdef_idx[(r64 + 1) * st + c64 + 1] = -1;
                }
            }
        }
    }

    pub(crate) fn clear_block_decoded_flags(&mut self, r: usize, c: usize, sb4: usize) {
        for plane in 0..self.f.num_planes {
            let (sx, sy) = self.plane_sub(plane);
            let sb_w4 = ((self.mi_col_end - c) >> sx) as isize;
            let sb_h4 = ((self.mi_row_end - r) >> sy) as isize;
            let bd = &mut self.block_decoded[plane];
            for y in -1..=((sb4 >> sy) as isize) {
                for x in -1..=((sb4 >> sx) as isize) {
                    let v = if y < 0 && x < sb_w4 { true } else { x < 0 && y < sb_h4 };
                    bd[(y + 1) as usize][(x + 1) as usize] = v;
                }
            }
            bd[(sb4 >> sy) + 1][0] = false;
        }
    }

    pub(crate) fn plane_sub(&self, plane: usize) -> (usize, usize) {
        if plane == 0 { (0, 0) } else { (self.f.sub_x, self.f.sub_y) }
    }

    // ---------------------------------------------------------------------------------------------
    // Loop restoration syntax
    // ---------------------------------------------------------------------------------------------

    pub(crate) fn read_lr(&mut self, r: usize, c: usize, bsize: u8) -> Result<()> {
        if self.f.hdr.allow_intrabc {
            return Ok(());
        }
        let w = NUM_4X4_WIDE[bsize as usize] as usize;
        let h = NUM_4X4_HIGH[bsize as usize] as usize;
        for plane in 0..self.f.num_planes {
            if self.f.hdr.lr.kind[plane] == RESTORE_NONE {
                continue;
            }
            let (sx, sy) = self.plane_sub(plane);
            let unit_size = self.f.hdr.lr.unit_size[plane];
            let (unit_cols, unit_rows, _) = {
                let u = &self.f.lr_units[plane];
                (u.0, u.1, ())
            };
            let row_start = (r * (4 >> sy)).div_ceil(unit_size);
            let row_end = unit_rows.min(((r + h) * (4 >> sy)).div_ceil(unit_size));
            let (num, den) = if self.f.hdr.use_superres {
                ((4 >> sx) * self.f.hdr.superres_denom, unit_size * 8)
            } else {
                (4 >> sx, unit_size)
            };
            let col_start = (c * num).div_ceil(den);
            let col_end = unit_cols.min(((c + w) * num).div_ceil(den));
            for ur in row_start..row_end {
                for uc in col_start..col_end {
                    self.read_lr_unit(plane, ur, uc);
                }
            }
        }
        Ok(())
    }

    fn read_lr_unit(&mut self, plane: usize, ur: usize, uc: usize) {
        let frame_kind = self.f.hdr.lr.kind[plane];
        let cols = self.f.lr_units[plane].0;
        // The encoder writes the parameters it stored for this unit; the decoder ignores `want`.
        let want = if C::DECODING { LrUnitParams::default() } else { self.f.lr_units[plane].2[ur * cols + uc] };
        let mut p = LrUnitParams::default();
        let restoration_type = match frame_kind {
            RESTORE_WIENER => {
                let s = self.c.symbol(&mut self.cdf.lr_wiener, (want.kind == RESTORE_WIENER) as usize);
                if s != 0 { RESTORE_WIENER } else { RESTORE_NONE }
            }
            RESTORE_SGRPROJ => {
                let s = self.c.symbol(&mut self.cdf.lr_sgrproj, (want.kind == RESTORE_SGRPROJ) as usize);
                if s != 0 { RESTORE_SGRPROJ } else { RESTORE_NONE }
            }
            _ => {
                debug_assert_eq!(frame_kind, RESTORE_SWITCHABLE);
                let w = match want.kind {
                    RESTORE_WIENER => 1,
                    RESTORE_SGRPROJ => 2,
                    _ => 0,
                };
                match self.c.symbol(&mut self.cdf.lr_switchable, w) {
                    0 => RESTORE_NONE,
                    1 => RESTORE_WIENER,
                    _ => RESTORE_SGRPROJ,
                }
            }
        };
        p.kind = restoration_type;
        if restoration_type == RESTORE_WIENER {
            const MIN: [i32; 3] = [-5, -23, -17];
            const MAX: [i32; 3] = [10, 8, 46];
            const K: [u32; 3] = [1, 2, 3];
            for pass in 0..2 {
                let first = if plane > 0 { 1 } else { 0 };
                if plane > 0 {
                    p.wiener[pass][0] = 0;
                }
                for j in first..3 {
                    let v = self.code_signed_subexp_with_ref(
                        MIN[j],
                        MAX[j] + 1,
                        K[j],
                        self.ref_lr_wiener[plane][pass][j],
                        want.wiener[pass][j] as i32,
                    );
                    p.wiener[pass][j] = v as i8;
                    self.ref_lr_wiener[plane][pass][j] = v;
                }
            }
        } else if restoration_type == RESTORE_SGRPROJ {
            let set = self.c.literal(4, want.sgr_set as u32) as usize;
            p.sgr_set = set as u8;
            const MIN: [i32; 2] = [-96, -32];
            const MAX: [i32; 2] = [31, 95];
            for i in 0..2 {
                let radius = SGR_RADII[set][i];
                let v = if radius != 0 {
                    self.code_signed_subexp_with_ref(
                        MIN[i],
                        MAX[i] + 1,
                        4,
                        self.ref_sgr_xqd[plane][i],
                        want.sgr_xqd[i] as i32,
                    )
                } else if i == 1 {
                    ((1 << 7) - self.ref_sgr_xqd[plane][0]).clamp(MIN[i], MAX[i])
                } else {
                    0
                };
                p.sgr_xqd[i] = v as i8;
                self.ref_sgr_xqd[plane][i] = v;
            }
        }
        self.f.lr_units[plane].2[ur * cols + uc] = p;
    }

    fn code_signed_subexp_with_ref(&mut self, low: i32, high: i32, k: u32, r: i32, want: i32) -> i32 {
        let x = self.code_unsigned_subexp_with_ref(high - low, k, r - low, want - low);
        x + low
    }

    fn code_unsigned_subexp_with_ref(&mut self, mx: i32, k: u32, r: i32, want: i32) -> i32 {
        // Forward recentering of the wanted value (the encoder's side of `inverse_recenter`).
        let mirrored = (r << 1) > mx;
        let v_want = if C::DECODING {
            0
        } else if mirrored {
            recenter(mx - 1 - r, mx - 1 - want)
        } else {
            recenter(r, want)
        };
        let v = self.code_subexp(mx, k, v_want);
        if mirrored { mx - 1 - inverse_recenter(mx - 1 - r, v) } else { inverse_recenter(r, v) }
    }

    fn code_subexp(&mut self, num_syms: i32, k: u32, want: i32) -> i32 {
        let mut i = 0;
        let mut mk = 0;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1 << b2;
            if num_syms <= mk + 3 * a {
                let v = self.code_ns((num_syms - mk) as u32, want.wrapping_sub(mk) as u32);
                return v as i32 + mk;
            }
            let more = self.c.literal(1, (want >= mk + a) as u32) != 0;
            if more {
                i += 1;
                mk += a;
            } else {
                let v = self.c.literal(b2, want.wrapping_sub(mk) as u32);
                return v as i32 + mk;
            }
        }
    }

    fn literal_ns(&mut self, n: u32) -> u32 {
        self.code_ns(n, 0)
    }

    /// `NS(n)` coded with literal bits.
    fn code_ns(&mut self, n: u32, want: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let hi = if want < m { want } else { want.wrapping_add(m) >> 1 };
        let v = self.c.literal(w - 1, hi);
        if v < m {
            return v;
        }
        let extra = self.c.literal(1, want.wrapping_add(m) & 1);
        (v << 1) - m + extra
    }

    // ---------------------------------------------------------------------------------------------
    // Partition
    // ---------------------------------------------------------------------------------------------

    pub fn decode_partition(&mut self, r: usize, c: usize, bsize: u8) -> Result<()> {
        if r >= self.f.mi_rows || c >= self.f.mi_cols {
            return Ok(());
        }
        let avail_u = self.is_inside(r as isize - 1, c as isize);
        let avail_l = self.is_inside(r as isize, c as isize - 1);
        let num4 = NUM_4X4_WIDE[bsize as usize] as usize;
        let half = num4 >> 1;
        let quarter = half >> 1;
        let has_rows = (r + half) < self.f.mi_rows;
        let has_cols = (c + half) < self.f.mi_cols;
        let want_part =
            if !C::DECODING && bsize >= BLOCK_8X8 { self.next_partition(r, c, bsize, has_rows, has_cols) } else { 0 };
        let partition = if bsize < BLOCK_8X8 {
            PARTITION_NONE
        } else {
            let bsl = (NUM_4X4_WIDE[bsize as usize] as usize).ilog2() as usize; // 8x8 -> 1
            let above =
                avail_u && (NUM_4X4_WIDE[self.f.mi_size[self.mi(r - 1, c)] as usize] as usize).ilog2() < bsl as u32;
            let left =
                avail_l && (NUM_4X4_HIGH[self.f.mi_size[self.mi(r, c - 1)] as usize] as usize).ilog2() < bsl as u32;
            let ctx = left as usize * 2 + above as usize;
            if has_rows && has_cols {
                let cdf: &mut [u16] = match bsl {
                    1 => &mut self.cdf.partition_w8[ctx],
                    5 => &mut self.cdf.partition_w128[ctx],
                    n => &mut self.cdf.partition[(n - 2) * 4 + ctx],
                };
                self.c.symbol(cdf, want_part as usize) as u8
            } else if has_cols || has_rows {
                // Only a split (or the one partition that fits) is possible: a binary choice whose
                // probability is derived from the full partition CDF.
                let cdf: &[u16] = match bsl {
                    1 => &self.cdf.partition_w8[ctx],
                    5 => &self.cdf.partition_w128[ctx],
                    n => &self.cdf.partition[(n - 2) * 4 + ctx],
                };
                let p = |i: u8| -> u32 {
                    let hi = cdf[i as usize] as u32;
                    let lo = if i > 0 { cdf[i as usize - 1] as u32 } else { 0 };
                    hi - lo
                };
                let mut psum;
                if has_cols {
                    // split_or_horz
                    psum = p(PARTITION_VERT) + p(PARTITION_SPLIT);
                    if bsl > 1 {
                        psum += p(PARTITION_HORZ_A) + p(PARTITION_VERT_A) + p(PARTITION_VERT_B);
                        if bsl < 5 {
                            psum += p(PARTITION_VERT_4);
                        }
                    }
                } else {
                    // split_or_vert
                    psum = p(PARTITION_HORZ) + p(PARTITION_SPLIT);
                    if bsl > 1 {
                        psum += p(PARTITION_HORZ_A) + p(PARTITION_HORZ_B) + p(PARTITION_VERT_A);
                        if bsl < 5 {
                            psum += p(PARTITION_HORZ_4);
                        }
                    }
                }
                let tmp = [(32768 - psum).min(32768) as u16, 32768, 0];
                let split = self.c.symbol_fixed(&tmp, (want_part == PARTITION_SPLIT) as usize) != 0;
                if split {
                    PARTITION_SPLIT
                } else if has_cols {
                    PARTITION_HORZ
                } else {
                    PARTITION_VERT
                }
            } else {
                PARTITION_SPLIT
            }
        };
        let sub = partition_subsize(partition, bsize);
        let split_size = partition_subsize(PARTITION_SPLIT, bsize);
        match partition {
            PARTITION_NONE => self.decode_block(r, c, sub)?,
            PARTITION_HORZ => {
                self.decode_block(r, c, sub)?;
                if has_rows {
                    self.decode_block(r + half, c, sub)?;
                }
            }
            PARTITION_VERT => {
                self.decode_block(r, c, sub)?;
                if has_cols {
                    self.decode_block(r, c + half, sub)?;
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(r, c, sub)?;
                self.decode_partition(r, c + half, sub)?;
                self.decode_partition(r + half, c, sub)?;
                self.decode_partition(r + half, c + half, sub)?;
            }
            PARTITION_HORZ_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r, c + half, split_size)?;
                self.decode_block(r + half, c, sub)?;
            }
            PARTITION_HORZ_B => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r + half, c, split_size)?;
                self.decode_block(r + half, c + half, split_size)?;
            }
            PARTITION_VERT_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r + half, c, split_size)?;
                self.decode_block(r, c + half, sub)?;
            }
            PARTITION_VERT_B => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r, c + half, split_size)?;
                self.decode_block(r + half, c + half, split_size)?;
            }
            PARTITION_HORZ_4 => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r + quarter, c, sub)?;
                self.decode_block(r + quarter * 2, c, sub)?;
                if r + quarter * 3 < self.f.mi_rows {
                    self.decode_block(r + quarter * 3, c, sub)?;
                }
            }
            PARTITION_VERT_4 => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r, c + quarter, sub)?;
                self.decode_block(r, c + quarter * 2, sub)?;
                if c + quarter * 3 < self.f.mi_cols {
                    self.decode_block(r, c + quarter * 3, sub)?;
                }
            }
            _ => return Err(Error::invalid("bad AV1 partition")),
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------------
    // Block
    // ---------------------------------------------------------------------------------------------

    /// Fill the position-derived fields of a block (chroma presence and neighbour availability).
    pub(crate) fn set_block_geometry(&self, b: &mut Blk) {
        let (r, c, bsize) = (b.mi_row, b.mi_col, b.bsize);
        let bw4 = NUM_4X4_WIDE[bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[bsize as usize] as usize;
        let (ssx, ssy) = (self.f.sub_x, self.f.sub_y);
        b.has_chroma =
            !(bh4 == 1 && ssy == 1 && (r & 1) == 0 || bw4 == 1 && ssx == 1 && (c & 1) == 0) && self.f.num_planes > 1;
        b.avail_u = self.is_inside(r as isize - 1, c as isize);
        b.avail_l = self.is_inside(r as isize, c as isize - 1);
        b.avail_u_chroma = b.avail_u;
        b.avail_l_chroma = b.avail_l;
        if b.has_chroma {
            if ssy == 1 && bh4 == 1 {
                b.avail_u_chroma = self.is_inside(r as isize - 2, c as isize);
            }
            if ssx == 1 && bw4 == 1 {
                b.avail_l_chroma = self.is_inside(r as isize, c as isize - 2);
            }
        } else {
            b.avail_u_chroma = false;
            b.avail_l_chroma = false;
        }
    }

    pub(crate) fn decode_block(&mut self, r: usize, c: usize, bsize: u8) -> Result<()> {
        let mut b = if C::DECODING { Blk::default() } else { self.next_block(r, c, bsize) };
        b.mi_row = r;
        b.mi_col = c;
        b.bsize = bsize;
        self.set_block_geometry(&mut b);
        let bw4 = NUM_4X4_WIDE[bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[bsize as usize] as usize;
        if self.f.hdr.is_intra() {
            self.intra_frame_mode_info(&mut b)?;
        } else {
            self.inter_frame_mode_info(&mut b)?;
        }
        if let Some(what) = self.unsupported {
            return Err(Error::unsupported(what));
        }
        self.palette_tokens(&b);
        self.read_block_tx_size(&mut b);
        if b.skip {
            self.reset_block_context(&b);
        }
        if self.trace {
            eprintln!(
                "BLK r{} c{} sz{} inter{} mode{} ref{:?} mv{:?} skip{} tx{} filt{:?} ymode{} uv{}",
                b.mi_row,
                b.mi_col,
                b.bsize,
                b.is_inter,
                b.y_mode,
                b.ref_frame,
                b.mv,
                b.skip,
                b.tx_size,
                b.interp_filter,
                b.y_mode,
                b.uv_mode
            );
        }
        self.store_block_info(&b);
        let (rows, cols) = (self.f.mi_rows, self.f.mi_cols);
        for x in 0..bw4 {
            if c + x < cols + 32 {
                self.above_pal[0][c + x] = b.pal_y;
                self.above_pal[1][c + x] = b.pal_u;
            }
        }
        for y in 0..bh4 {
            if r + y < rows + 32 {
                self.left_pal[0][r + y] = b.pal_y;
                self.left_pal[1][r + y] = b.pal_u;
            }
        }
        if C::DECODING && self.stats {
            self.blk_log.push(b.clone());
        }
        self.residual(&b)?;
        if let Some(snap) = self.dbg_snap.take() {
            // Encoder self-check: the real pass must reproduce the analysis reconstruction.
            let np = if b.has_chroma { 3 } else { 1 };
            for plane in 0..np {
                let (sx, sy) = self.plane_sub(plane);
                let (x0, y0) = ((b.mi_col >> sx) * 4, (b.mi_row >> sy) * 4);
                let (w, h) = ((block_width(b.bsize) >> sx).max(4), (block_height(b.bsize) >> sy).max(4));
                let pl = &self.f.planes[plane];
                let mut bad = 0;
                for i in 0..h {
                    for j in 0..w {
                        if snap[plane][i * w + j] != pl.at(x0 + j, y0 + i) {
                            bad += 1;
                        }
                    }
                }
                if bad > 0 && std::env::var_os("AV1_CHECK_V").is_some() {
                    for i in 0..h {
                        let a: Vec<u16> = (0..w).map(|j| snap[plane][i * w + j]).collect();
                        let c: Vec<u16> = (0..w).map(|j| pl.at(x0 + j, y0 + i)).collect();
                        eprintln!("  at ({x0},{y0}) row {i}: analysis {a:?} real {c:?}");
                    }
                }
                if bad > 0 {
                    eprintln!(
                        "MISMATCH plane {plane} block r{} c{} bsize {} skip {} ymode {} uvmode {} tx {}: {bad} px",
                        b.mi_row, b.mi_col, b.bsize, b.skip, b.y_mode, b.uv_mode, b.tx_size
                    );
                }
            }
        }
        Ok(())
    }

    fn reset_block_context(&mut self, b: &Blk) {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        let planes = if b.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sx, sy) = self.plane_sub(plane);
            for i in (b.mi_col >> sx)..=((b.mi_col + bw4 - 1) >> sx) {
                self.above_level[plane][i] = 0;
                self.above_dc[plane][i] = 0;
            }
            for i in (b.mi_row >> sy)..=((b.mi_row + bh4 - 1) >> sy) {
                self.left_level[plane][i] = 0;
                self.left_dc[plane][i] = 0;
            }
        }
    }

    fn intra_frame_mode_info(&mut self, b: &mut Blk) -> Result<()> {
        let seg_pre_skip = self.f.hdr.seg.seg_id_pre_skip;
        // The encoder arrives with the planned skip flag; the decoder starts from false.
        if C::DECODING {
            b.skip = false;
        }
        if seg_pre_skip {
            self.intra_segment_id(b);
        }
        self.read_skip(b);
        if !seg_pre_skip {
            self.intra_segment_id(b);
        }
        b.lossless = self.f.hdr.lossless_array[b.segment_id as usize];
        self.read_cdef(b);
        self.read_delta_qindex(b);
        self.read_delta_lf(b);
        self.read_deltas = false;
        if C::DECODING {
            b.use_intrabc = false;
        }
        if self.f.hdr.allow_intrabc {
            b.use_intrabc = self.c.symbol(&mut self.cdf.intrabc, b.use_intrabc as usize) != 0;
        } else {
            b.use_intrabc = false;
        }
        if b.use_intrabc {
            b.is_inter = true;
            b.ref_frame = [INTRA_FRAME, NONE_FRAME];
            b.y_mode = DC_PRED;
            b.uv_mode = DC_PRED;
            b.angle_y = 0;
            b.angle_uv = 0;
            b.pal_size_y = 0;
            b.pal_size_uv = 0;
            b.use_filter_intra = false;
            self.assign_intrabc_mv(b);
            return Ok(());
        }
        // intra_frame_y_mode
        let above_mode = if b.avail_u { self.f.y_mode[self.mi(b.mi_row - 1, b.mi_col)] } else { DC_PRED };
        let left_mode = if b.avail_l { self.f.y_mode[self.mi(b.mi_row, b.mi_col - 1)] } else { DC_PRED };
        let (ac, lc) =
            (INTRA_MODE_CONTEXT[above_mode as usize] as usize, INTRA_MODE_CONTEXT[left_mode as usize] as usize);
        b.y_mode = self.c.symbol(&mut self.cdf.kf_y_mode[ac][lc], b.y_mode as usize) as u8;
        self.intra_mode_tail(b);
        Ok(())
    }

    /// `read_skip()`.
    pub(crate) fn read_skip(&mut self, b: &mut Blk) {
        if self.f.hdr.seg.seg_id_pre_skip && self.f.hdr.seg_feature_active(b.segment_id as usize, SEG_LVL_SKIP) {
            b.skip = true;
        } else {
            let mut ctx = 0;
            if b.avail_u {
                ctx += self.f.skip[self.mi(b.mi_row - 1, b.mi_col)] as usize;
            }
            if b.avail_l {
                ctx += self.f.skip[self.mi(b.mi_row, b.mi_col - 1)] as usize;
            }
            b.skip = self.c.symbol(&mut self.cdf.skip[ctx], b.skip as usize) != 0;
        }
    }

    /// The rest of an intra block's mode info after its luma mode: angle deltas, chroma mode, palette and filter
    /// intra (shared by intra frames and intra blocks in inter frames).
    pub(crate) fn intra_mode_tail(&mut self, b: &mut Blk) {
        // intra_angle_info_y (the encoder arrives with the planned values already set)
        if C::DECODING {
            b.angle_y = 0;
        }
        if b.bsize >= BLOCK_8X8 && is_directional(b.y_mode) {
            let s = self.c.symbol(&mut self.cdf.angle_delta[(b.y_mode - V_PRED) as usize], (b.angle_y + 3) as usize);
            b.angle_y = s as i8 - 3;
        }
        if C::DECODING || !b.has_chroma {
            b.uv_mode = DC_PRED;
            b.angle_uv = 0;
        }
        if b.has_chroma {
            let cfl_allowed = if b.lossless {
                subsampled_size(b.bsize, self.f.sub_x as u8, self.f.sub_y as u8) == BLOCK_4X4
            } else {
                block_width(b.bsize).max(block_height(b.bsize)) <= 32
            };
            let ym = b.y_mode as usize;
            b.uv_mode = if cfl_allowed {
                self.c.symbol(&mut self.cdf.uv_mode_cfl[ym], b.uv_mode as usize) as u8
            } else {
                self.c.symbol(&mut self.cdf.uv_mode[ym], b.uv_mode as usize) as u8
            };
            if b.uv_mode == UV_CFL_PRED {
                self.read_cfl_alphas(b);
            }
            if b.bsize >= BLOCK_8X8 && is_directional(b.uv_mode) {
                let s =
                    self.c.symbol(&mut self.cdf.angle_delta[(b.uv_mode - V_PRED) as usize], (b.angle_uv + 3) as usize);
                b.angle_uv = s as i8 - 3;
            }
        }
        b.pal_size_y = 0;
        b.pal_size_uv = 0;
        if b.bsize >= BLOCK_8X8
            && block_width(b.bsize) <= 64
            && block_height(b.bsize) <= 64
            && self.f.hdr.allow_screen_content_tools
        {
            self.palette_mode_info(b);
        }
        // filter_intra_mode_info
        let want_filter = b.use_filter_intra;
        b.use_filter_intra = false;
        if self.f.seq.enable_filter_intra
            && b.y_mode == DC_PRED
            && b.pal_size_y == 0
            && block_width(b.bsize).max(block_height(b.bsize)) <= 32
        {
            b.use_filter_intra =
                self.c.symbol(&mut self.cdf.use_filter_intra[b.bsize as usize], want_filter as usize) != 0;
            if b.use_filter_intra {
                b.filter_intra_mode =
                    self.c.symbol(&mut self.cdf.filter_intra_mode, b.filter_intra_mode as usize) as u8;
            }
        }
    }

    fn read_cfl_alphas(&mut self, b: &mut Blk) {
        // Encode the pair back into the joint sign symbol when writing.
        let sign_of = |a: i8| {
            if a == 0 {
                0
            } else if a < 0 {
                1
            } else {
                2
            }
        };
        let (su, sv) = (sign_of(b.cfl_alpha_u), sign_of(b.cfl_alpha_v));
        let joint = (su * 3 + sv).max(1) - 1;
        let joint = self.c.symbol(&mut self.cdf.cfl_sign, joint as usize) + 1;
        let sign_u = joint / 3;
        let sign_v = joint % 3;
        let (want_u, want_v) = (b.cfl_alpha_u.unsigned_abs() as usize, b.cfl_alpha_v.unsigned_abs() as usize);
        b.cfl_alpha_u = 0;
        b.cfl_alpha_v = 0;
        if sign_u != 0 {
            let ctx = (sign_u - 1) * 3 + sign_v;
            let a = self.c.symbol(&mut self.cdf.cfl_alpha[ctx], want_u.max(1) - 1) + 1;
            b.cfl_alpha_u = if sign_u == 1 { -(a as i8) } else { a as i8 };
        }
        if sign_v != 0 {
            let ctx = (sign_v - 1) * 3 + sign_u;
            let a = self.c.symbol(&mut self.cdf.cfl_alpha[ctx], want_v.max(1) - 1) + 1;
            b.cfl_alpha_v = if sign_v == 1 { -(a as i8) } else { a as i8 };
        }
    }

    fn intra_segment_id(&mut self, b: &mut Blk) {
        if !self.f.hdr.seg.enabled {
            b.segment_id = 0;
            return;
        }
        self.read_segment_id(b);
    }

    /// `read_segment_id()`.
    pub(crate) fn read_segment_id(&mut self, b: &mut Blk) {
        let (r, c) = (b.mi_row, b.mi_col);
        let prev_ul = if b.avail_u && b.avail_l { self.f.seg_id[self.mi(r - 1, c - 1)] as i32 } else { -1 };
        let prev_u = if b.avail_u { self.f.seg_id[self.mi(r - 1, c)] as i32 } else { -1 };
        let prev_l = if b.avail_l { self.f.seg_id[self.mi(r, c - 1)] as i32 } else { -1 };
        let pred = if prev_u == -1 {
            prev_l.max(0)
        } else if prev_l == -1 || prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        let last = self.f.hdr.seg.last_active_seg_id as i32;
        if b.skip {
            b.segment_id = pred as u8;
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let coded = interleave(b.segment_id as i32, pred, last + 1);
            let s = self.c.symbol(&mut self.cdf.segment_id[ctx], coded as usize) as i32;
            let id = neg_deinterleave(s, pred, last + 1);
            b.segment_id = id.clamp(0, last) as u8;
        }
    }

    pub(crate) fn read_cdef(&mut self, b: &Blk) {
        let hdr = &self.f.hdr;
        if b.skip || hdr.coded_lossless || !self.f.seq.enable_cdef || hdr.allow_intrabc {
            return;
        }
        let st = self.f.cdef_stride;
        let (r64, c64) = (b.mi_row >> 4, b.mi_col >> 4);
        if self.f.cdef_idx[r64 * st + c64] == -1 {
            let bits = self.f.hdr.cdef.bits as u32;
            let cur = self.c.literal(bits, b.cdef_value);
            let w4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
            let h4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
            let rows = self.f.cdef_idx.len() / st;
            for y in r64..r64 + h4.div_ceil(16) {
                for x in c64..c64 + w4.div_ceil(16) {
                    if y < rows && x < st {
                        self.f.cdef_idx[y * st + x] = cur as i8;
                    }
                }
            }
        }
    }

    /// Code the magnitude of a delta-q / delta-lf value; `which` selects the CDF (0 = delta q, 1 = delta lf,
    /// 2.. = multi delta lf).
    fn code_delta_abs(&mut self, which: usize, want: u32) -> i32 {
        let mut abs = {
            let cdf: &mut [u16] = match which {
                0 => &mut self.cdf.delta_q,
                1 => &mut self.cdf.delta_lf,
                n => &mut self.cdf.delta_lf_multi[n - 2],
            };
            self.c.symbol(cdf, want.min(3) as usize) as i32
        };
        if abs == 3 {
            let rem_bits_want = if want >= 3 { (want - 1).ilog2() } else { 1 };
            let rem_bits = self.c.literal(3, rem_bits_want - 1) + 1;
            let bits_want = want.wrapping_sub(1 + (1 << rem_bits));
            let abs_bits = self.c.literal(rem_bits, bits_want);
            abs = abs_bits as i32 + (1 << rem_bits) + 1;
        }
        abs
    }

    pub(crate) fn read_delta_qindex(&mut self, b: &Blk) {
        if b.bsize == self.sb_size() && b.skip {
            return;
        }
        if self.read_deltas {
            let abs = self.code_delta_abs(0, b.delta_q.unsigned_abs());
            if abs != 0 {
                let sign = self.c.literal(1, (b.delta_q < 0) as u32) != 0;
                let reduced = if sign { -abs } else { abs };
                self.current_q_index = (self.current_q_index + (reduced << self.f.hdr.delta_q_res)).clamp(1, 255);
            }
        }
    }

    pub(crate) fn read_delta_lf(&mut self, b: &Blk) {
        if b.bsize == self.sb_size() && b.skip {
            return;
        }
        if self.read_deltas && self.f.hdr.delta_lf_present {
            let multi = self.f.hdr.delta_lf_multi;
            let count = if multi { if self.f.num_planes > 1 { 4 } else { 2 } } else { 1 };
            for i in 0..count {
                let want = b.delta_lf_values[i];
                let abs = self.code_delta_abs(if multi { 2 + i } else { 1 }, want.unsigned_abs());
                if abs != 0 {
                    let sign = self.c.literal(1, (want < 0) as u32) != 0;
                    let reduced = if sign { -abs } else { abs };
                    self.delta_lf[i] = (self.delta_lf[i] + (reduced << self.f.hdr.delta_lf_res)).clamp(-63, 63);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Palette
    // ---------------------------------------------------------------------------------------------

    fn palette_cache(&self, b: &Blk, plane: usize) -> Vec<u16> {
        let (r, c) = (b.mi_row, b.mi_col);
        let above_n =
            if (r * 4) % 64 != 0 && b.avail_u { self.f.pal_size[plane][self.mi(r - 1, c)] as usize } else { 0 };
        let left_n = if b.avail_l { self.f.pal_size[plane][self.mi(r, c - 1)] as usize } else { 0 };
        let above = if above_n > 0 { self.above_pal[plane][c] } else { [0; 8] };
        let left = if left_n > 0 { self.left_pal[plane][r] } else { [0; 8] };
        let (mut ai, mut li) = (0, 0);
        let mut cache: Vec<u16> = Vec::new();
        while ai < above_n && li < left_n {
            let (ac, lc) = (above[ai], left[li]);
            if lc < ac {
                if cache.last() != Some(&lc) {
                    cache.push(lc);
                }
                li += 1;
            } else {
                if cache.last() != Some(&ac) {
                    cache.push(ac);
                }
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            let v = above[ai];
            ai += 1;
            if cache.last() != Some(&v) {
                cache.push(v);
            }
        }
        while li < left_n {
            let v = left[li];
            li += 1;
            if cache.last() != Some(&v) {
                cache.push(v);
            }
        }
        cache
    }

    fn palette_mode_info(&mut self, b: &mut Blk) {
        let bd = self.f.seq.color.bit_depth as u32;
        let ctx_idx = (NUM_4X4_WIDE[b.bsize as usize].ilog2() + NUM_4X4_HIGH[b.bsize as usize].ilog2()) as usize - 2;
        let (r, c) = (b.mi_row, b.mi_col);
        if b.y_mode == DC_PRED {
            let ctx = (b.avail_u && self.f.pal_size[0][self.mi(r - 1, c)] > 0) as usize
                + (b.avail_l && self.f.pal_size[0][self.mi(r, c - 1)] > 0) as usize;
            let has = self.c.symbol(&mut self.cdf.palette_y_mode[ctx_idx][ctx], (b.pal_size_y > 0) as usize) != 0;
            if has {
                let n = self.c.symbol(&mut self.cdf.palette_size_y[ctx_idx], (b.pal_size_y as usize).max(2) - 2) + 2;
                b.pal_size_y = n as u8;
                let cache = self.palette_cache(b, 0);
                let mut colors = b.pal_y;
                self.read_palette_colors(&mut colors, n, &cache, bd, true);
                b.pal_y = colors;
            }
        }
        if b.has_chroma && b.uv_mode == DC_PRED {
            let ctx = (b.pal_size_y > 0) as usize;
            let has = self.c.symbol(&mut self.cdf.palette_uv_mode[ctx], (b.pal_size_uv > 0) as usize) != 0;
            if has {
                let n = self.c.symbol(&mut self.cdf.palette_size_uv[ctx_idx], (b.pal_size_uv as usize).max(2) - 2) + 2;
                b.pal_size_uv = n as u8;
                let cache = self.palette_cache(b, 1);
                let mut colors = b.pal_u;
                self.read_palette_colors(&mut colors, n, &cache, bd, false);
                b.pal_u = colors;
                // V colours
                let maxv = 1i32 << bd;
                let delta_encode = self.c.literal(1, b.pal_v_delta_encode as u32) != 0;
                if delta_encode {
                    let min_bits = bd - 4;
                    let bits = min_bits + self.c.literal(2, 0);
                    b.pal_v[0] = self.c.literal(bd, b.pal_v[0] as u32) as u16;
                    for idx in 1..n {
                        let mut delta = self.c.literal(bits, 0) as i32;
                        if delta != 0 && self.c.literal(1, 0) != 0 {
                            delta = -delta;
                        }
                        let mut val = b.pal_v[idx - 1] as i32 + delta;
                        if val < 0 {
                            val += maxv;
                        }
                        if val >= maxv {
                            val -= maxv;
                        }
                        b.pal_v[idx] = val.clamp(0, maxv - 1) as u16;
                    }
                } else {
                    for idx in 0..n {
                        b.pal_v[idx] = self.c.literal(bd, b.pal_v[idx] as u32) as u16;
                    }
                }
            }
        }
    }

    fn read_palette_colors(&mut self, colors: &mut [u16; 8], n: usize, cache: &[u16], bd: u32, luma: bool) {
        let mut idx = 0;
        for &cv in cache {
            if idx >= n {
                break;
            }
            let use_cache = self.c.literal(1, 0) != 0;
            if use_cache {
                colors[idx] = cv;
                idx += 1;
            }
        }
        if idx < n {
            colors[idx] = self.c.literal(bd, colors[idx] as u32) as u16;
            idx += 1;
        }
        let mut palette_bits = 0;
        if idx < n {
            let min_bits = bd - 3;
            palette_bits = min_bits + self.c.literal(2, 0);
        }
        let maxv = (1i32 << bd) - 1;
        while idx < n {
            let mut delta = self.c.literal(palette_bits, 0) as i32;
            if luma {
                delta += 1;
            }
            let prev = colors[idx - 1] as i32;
            colors[idx] = (prev + delta).clamp(0, maxv) as u16;
            let range = (1i32 << bd) - colors[idx] as i32 - if luma { 1 } else { 0 };
            palette_bits = palette_bits.min(ceil_log2(range.max(0) as u32));
            idx += 1;
        }
        colors[..n].sort_unstable();
    }

    fn palette_tokens(&mut self, b: &Blk) {
        if b.pal_size_y == 0 && b.pal_size_uv == 0 {
            return;
        }
        let bw = block_width(b.bsize);
        let bh = block_height(b.bsize);
        let on_h = bh.min((self.f.mi_rows - b.mi_row) * 4);
        let on_w = bw.min((self.f.mi_cols - b.mi_col) * 4);
        if b.pal_size_y > 0 {
            let mut map = std::mem::take(&mut self.color_map_y);
            self.read_color_map(&mut map, b.pal_size_y as usize, bw, bh, on_w, on_h, true);
            self.color_map_y = map;
        }
        if b.pal_size_uv > 0 {
            let (sx, sy) = (self.f.sub_x, self.f.sub_y);
            let (mut bw, mut bh, mut on_w, mut on_h) = (bw >> sx, bh >> sy, on_w >> sx, on_h >> sy);
            if bw < 4 {
                bw += 2;
                on_w += 2;
            }
            if bh < 4 {
                bh += 2;
                on_h += 2;
            }
            let mut map = std::mem::take(&mut self.color_map_uv);
            self.read_color_map(&mut map, b.pal_size_uv as usize, bw, bh, on_w, on_h, false);
            self.color_map_uv = map;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn read_color_map(&mut self, map: &mut [u8], n: usize, bw: usize, bh: usize, on_w: usize, on_h: usize, luma: bool) {
        // The index map is stored with a fixed stride of 64.
        let stride = 64;
        debug_assert!(C::DECODING, "palette blocks are not produced by the encoder");
        let first = self.literal_ns(n as u32) as u8;
        map[0] = first;
        for i in 1..(on_h + on_w - 1) {
            let jmax = i.min(on_w - 1);
            let jmin = (i + 1).saturating_sub(on_h);
            let mut j = jmax as isize;
            while j >= jmin as isize {
                let (row, col) = (i - j as usize, j as usize);
                let (order, ctx) = palette_color_context(map, stride, row, col, n);
                let want =
                    if C::DECODING { 0 } else { order.iter().position(|&o| o == map[row * stride + col]).unwrap_or(0) };
                let sym = {
                    let cdf: &mut [u16] = match (luma, n) {
                        (true, 2) => &mut self.cdf.palette_color_y_2[ctx],
                        (true, 3) => &mut self.cdf.palette_color_y_3[ctx],
                        (true, 4) => &mut self.cdf.palette_color_y_4[ctx],
                        (true, 5) => &mut self.cdf.palette_color_y_5[ctx],
                        (true, 6) => &mut self.cdf.palette_color_y_6[ctx],
                        (true, 7) => &mut self.cdf.palette_color_y_7[ctx],
                        (true, _) => &mut self.cdf.palette_color_y_8[ctx],
                        (false, 2) => &mut self.cdf.palette_color_uv_2[ctx],
                        (false, 3) => &mut self.cdf.palette_color_uv_3[ctx],
                        (false, 4) => &mut self.cdf.palette_color_uv_4[ctx],
                        (false, 5) => &mut self.cdf.palette_color_uv_5[ctx],
                        (false, 6) => &mut self.cdf.palette_color_uv_6[ctx],
                        (false, 7) => &mut self.cdf.palette_color_uv_7[ctx],
                        (false, _) => &mut self.cdf.palette_color_uv_8[ctx],
                    };
                    self.c.symbol(cdf, want)
                };
                map[row * stride + col] = order[sym];
                j -= 1;
            }
        }
        for i in 0..on_h {
            for j in on_w..bw {
                map[i * stride + j] = map[i * stride + on_w - 1];
            }
        }
        for i in on_h..bh {
            for j in 0..bw {
                map[i * stride + j] = map[(on_h - 1) * stride + j];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Transform size
    // ---------------------------------------------------------------------------------------------

    fn read_block_tx_size(&mut self, b: &mut Blk) {
        let bw4 = NUM_4X4_WIDE[b.bsize as usize] as usize;
        let bh4 = NUM_4X4_HIGH[b.bsize as usize] as usize;
        if self.f.hdr.tx_mode_select && b.bsize > BLOCK_4X4 && b.is_inter && !b.skip && !b.lossless {
            // Variable transform sizes: a quad/binary tree of `txfm_split` flags per maximum-size transform.
            let max_tx = MAX_TX_SIZE_RECT[b.bsize as usize];
            let (tx4w, tx4h) = (TX_W[max_tx as usize] as usize / 4, TX_H[max_tx as usize] as usize / 4);
            let mut row = b.mi_row;
            while row < b.mi_row + bh4 {
                let mut col = b.mi_col;
                while col < b.mi_col + bw4 {
                    self.read_var_tx_size(b, row, col, max_tx, 0);
                    col += tx4w;
                }
                row += tx4h;
            }
        } else {
            self.read_tx_size(b, !b.skip || !b.is_inter);
            let (rows, cols) = (self.f.mi_rows, self.f.mi_cols);
            for row in b.mi_row..(b.mi_row + bh4).min(rows) {
                for col in b.mi_col..(b.mi_col + bw4).min(cols) {
                    self.f.inter_tx_size[row * cols + col] = b.tx_size;
                }
            }
        }
    }

    fn above_tx_width(&self, b: &Blk, row: usize, col: usize) -> usize {
        if row == b.mi_row {
            if !b.avail_u {
                return 64;
            }
            let i = self.mi(row - 1, col);
            if self.f.skip[i] != 0 && self.f.is_inter[i] != 0 {
                return block_width(self.f.mi_size[i]);
            }
        }
        TX_W[self.f.inter_tx_size[self.mi(row - 1, col)] as usize] as usize
    }

    fn left_tx_height(&self, b: &Blk, row: usize, col: usize) -> usize {
        if col == b.mi_col {
            if !b.avail_l {
                return 64;
            }
            let i = self.mi(row, col - 1);
            if self.f.skip[i] != 0 && self.f.is_inter[i] != 0 {
                return block_height(self.f.mi_size[i]);
            }
        }
        TX_H[self.f.inter_tx_size[self.mi(row, col - 1)] as usize] as usize
    }

    /// `read_var_tx_size`. The encoder only plans unsplit transforms for block copy blocks, so it writes
    /// `txfm_split = 0` everywhere.
    fn read_var_tx_size(&mut self, b: &mut Blk, row: usize, col: usize, tx_sz: u8, depth: u32) {
        let (rows, cols) = (self.f.mi_rows, self.f.mi_cols);
        if row >= rows || col >= cols {
            return;
        }
        let split = if tx_sz == TX_4X4 || depth == 2 {
            false
        } else {
            let above = (self.above_tx_width(b, row, col) < TX_W[tx_sz as usize] as usize) as usize;
            let left = (self.left_tx_height(b, row, col) < TX_H[tx_sz as usize] as usize) as usize;
            let size = 64.min(block_width(b.bsize).max(block_height(b.bsize)));
            let max_sz = match size {
                4 => TX_4X4,
                8 => TX_8X8,
                16 => TX_16X16,
                32 => TX_32X32,
                _ => TX_64X64,
            } as usize;
            let sqr_up = TX_SQR_UP[tx_sz as usize] as usize;
            let ctx = (sqr_up != max_sz) as usize * 3 + (4 - max_sz) * 6 + above + left;
            self.c.symbol(&mut self.cdf.txfm_split[ctx], 0) != 0
        };
        let w4 = TX_W[tx_sz as usize] as usize / 4;
        let h4 = TX_H[tx_sz as usize] as usize / 4;
        if split {
            let sub = SPLIT_TX_SIZE[tx_sz as usize];
            let (step_w, step_h) = (TX_W[sub as usize] as usize / 4, TX_H[sub as usize] as usize / 4);
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(b, row + i, col + j, sub, depth + 1);
                    j += step_w;
                }
                i += step_h;
            }
        } else {
            for i in 0..h4 {
                for j in 0..w4 {
                    if row + i < rows && col + j < cols {
                        self.f.inter_tx_size[(row + i) * cols + col + j] = tx_sz;
                    }
                }
            }
            b.tx_size = tx_sz;
        }
    }

    /// `read_tx_size(allowSelect)`.
    fn read_tx_size(&mut self, b: &mut Blk, allow_select: bool) {
        if b.lossless {
            b.tx_size = TX_4X4;
            return;
        }
        let max_rect = MAX_TX_SIZE_RECT[b.bsize as usize];
        let want = b.tx_size;
        b.tx_size = max_rect;
        if b.bsize > BLOCK_4X4 && allow_select && self.f.hdr.tx_mode_select {
            // The context compares neighbouring transform extents with the largest transform.
            let max_w = TX_W[max_rect as usize] as usize;
            let max_h = TX_H[max_rect as usize] as usize;
            let above_w = if b.avail_u {
                let i = self.mi(b.mi_row - 1, b.mi_col);
                if self.f.is_inter[i] != 0 {
                    block_width(self.f.mi_size[i])
                } else {
                    self.above_tx_width(b, b.mi_row, b.mi_col)
                }
            } else {
                0
            };
            let left_h = if b.avail_l {
                let i = self.mi(b.mi_row, b.mi_col - 1);
                if self.f.is_inter[i] != 0 {
                    block_height(self.f.mi_size[i])
                } else {
                    self.left_tx_height(b, b.mi_row, b.mi_col)
                }
            } else {
                0
            };
            let ctx = (above_w >= max_w) as usize + (left_h >= max_h) as usize;
            let mut depth = 0usize;
            let mut t = max_rect;
            while t != want && depth < 2 && !C::DECODING {
                t = SPLIT_TX_SIZE[t as usize];
                depth += 1;
            }
            let d = match MAX_TX_DEPTH[b.bsize as usize] {
                1 => self.c.symbol(&mut self.cdf.tx_size_8x8[ctx], depth),
                2 => self.c.symbol(&mut self.cdf.tx_size[0][ctx], depth),
                3 => self.c.symbol(&mut self.cdf.tx_size[1][ctx], depth),
                _ => self.c.symbol(&mut self.cdf.tx_size[2][ctx], depth),
            };
            for _ in 0..d {
                b.tx_size = SPLIT_TX_SIZE[b.tx_size as usize];
            }
        }
    }
}

/// The encoder's side of [`inverse_recenter`].
fn recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

fn ceil_log2(x: u32) -> u32 {
    if x < 2 { 0 } else { 32 - (x - 1).leading_zeros() }
}

fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            return if diff & 1 != 0 { r + ((diff + 1) >> 1) } else { r - (diff >> 1) };
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            return if diff & 1 != 0 { r + ((diff + 1) >> 1) } else { r - (diff >> 1) };
        }
        max - (diff + 1)
    }
}

/// Inverse of [`neg_deinterleave`] for the encoder.
fn interleave(id: i32, r: i32, max: i32) -> i32 {
    (0..max.max(1)).find(|&d| neg_deinterleave(d, r, max) == id).unwrap_or(0)
}

/// `get_palette_color_context`: the colour order and CDF context for position (r, c) of the map.
fn palette_color_context(map: &[u8], stride: usize, r: usize, c: usize, n: usize) -> ([u8; 8], usize) {
    let mut scores = [0u32; 8];
    let mut order = [0u8, 1, 2, 3, 4, 5, 6, 7];
    if c > 0 {
        scores[map[r * stride + c - 1] as usize] += 2;
    }
    if r > 0 && c > 0 {
        scores[map[(r - 1) * stride + c - 1] as usize] += 1;
    }
    if r > 0 {
        scores[map[(r - 1) * stride + c] as usize] += 2;
    }
    for i in 0..3 {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in i + 1..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_score = scores[max_idx];
            let max_order = order[max_idx];
            for k in (i + 1..=max_idx).rev() {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
            }
            scores[i] = max_score;
            order[i] = max_order;
        }
    }
    let mut hash = 0u32;
    for i in 0..3 {
        hash += scores[i] * PALETTE_COLOR_HASH_MULTIPLIERS[i];
    }
    (order, PALETTE_COLOR_CONTEXT[hash as usize].max(0) as usize)
}

// -------------------------------------------------------------------------------------------------
// Residual: transform blocks, coefficients and reconstruction
// -------------------------------------------------------------------------------------------------

impl<C: Coder> TileCoder<'_, C> {
    pub(crate) fn residual(&mut self, b: &Blk) -> Result<()> {
        let sb_mask = if self.f.seq.use_128x128_superblock { 31 } else { 15 };
        let _ = sb_mask;
        let bw = block_width(b.bsize);
        let bh = block_height(b.bsize);
        let width_chunks = (bw >> 6).max(1);
        let height_chunks = (bh >> 6).max(1);
        let planes = if b.has_chroma { 3 } else { 1 };
        if b.is_inter {
            self.predict_inter_block(b, if self.analyzing { self.an_planes } else { 7 })?;
        }
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                for plane in 0..planes {
                    if self.analyzing && (self.an_planes >> plane) & 1 == 0 {
                        continue;
                    }
                    if b.is_inter && !b.lossless && plane == 0 {
                        let chunk_sz = if width_chunks > 1 || height_chunks > 1 { BLOCK_64X64 } else { b.bsize };
                        self.transform_tree(
                            b,
                            (b.mi_col + (chunk_x << 4)) * 4,
                            (b.mi_row + (chunk_y << 4)) * 4,
                            block_width(chunk_sz),
                            block_height(chunk_sz),
                        );
                        continue;
                    }
                    let tx_sz = if b.lossless { TX_4X4 } else { self.get_tx_size(b, plane) };
                    let step_x = (TX_W[tx_sz as usize] >> 2) as usize;
                    let step_y = (TX_H[tx_sz as usize] >> 2) as usize;
                    let (sx, sy) = self.plane_sub(plane);
                    let plane_sz = if plane == 0 { b.bsize } else { subsampled_size(b.bsize, sx as u8, sy as u8) };
                    let n4w = NUM_4X4_WIDE[plane_sz as usize] as usize;
                    let n4h = NUM_4X4_HIGH[plane_sz as usize] as usize;
                    let base_x = (b.mi_col >> sx) * 4;
                    let base_y = (b.mi_row >> sy) * 4;
                    let max_y4 = n4h.min(16 >> sy);
                    let max_x4 = n4w.min(16 >> sx);
                    let mut y = 0;
                    while y < max_y4 {
                        let mut x = 0;
                        while x < max_x4 {
                            self.transform_block(
                                b,
                                plane,
                                base_x,
                                base_y,
                                tx_sz,
                                x + ((chunk_x << 4) >> sx),
                                y + ((chunk_y << 4) >> sy),
                            );
                            x += step_x;
                        }
                        y += step_y;
                    }
                }
            }
        }
        Ok(())
    }

    /// `transform_tree`: the luma transform blocks of an inter-style block, following `InterTxSizes`.
    fn transform_tree(&mut self, b: &Blk, start_x: usize, start_y: usize, w: usize, h: usize) {
        if start_x >= self.f.mi_cols * 4 || start_y >= self.f.mi_rows * 4 {
            return;
        }
        let luma_tx = self.f.inter_tx_size[self.mi(start_y >> 2, start_x >> 2)] as usize;
        if w <= TX_W[luma_tx] as usize && h <= TX_H[luma_tx] as usize {
            let tx_sz = (0..19u8).find(|&t| TX_W[t as usize] as usize == w && TX_H[t as usize] as usize == h);
            self.transform_block(b, 0, start_x, start_y, tx_sz.unwrap_or(TX_4X4), 0, 0);
        } else if w > h {
            self.transform_tree(b, start_x, start_y, w / 2, h);
            self.transform_tree(b, start_x + w / 2, start_y, w / 2, h);
        } else if w < h {
            self.transform_tree(b, start_x, start_y, w, h / 2);
            self.transform_tree(b, start_x, start_y + h / 2, w, h / 2);
        } else {
            self.transform_tree(b, start_x, start_y, w / 2, h / 2);
            self.transform_tree(b, start_x + w / 2, start_y, w / 2, h / 2);
            self.transform_tree(b, start_x, start_y + h / 2, w / 2, h / 2);
            self.transform_tree(b, start_x + w / 2, start_y + h / 2, w / 2, h / 2);
        }
    }

    pub(crate) fn get_tx_size(&self, b: &Blk, plane: usize) -> u8 {
        if plane == 0 {
            return b.tx_size;
        }
        let (sx, sy) = self.plane_sub(plane);
        let uv_tx = MAX_TX_SIZE_RECT[subsampled_size(b.bsize, sx as u8, sy as u8) as usize];
        if TX_W[uv_tx as usize] == 64 || TX_H[uv_tx as usize] == 64 {
            if TX_W[uv_tx as usize] == 16 {
                return TX_16X32;
            }
            if TX_H[uv_tx as usize] == 16 {
                return TX_32X16;
            }
            return TX_32X32;
        }
        uv_tx
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_block(&mut self, b: &Blk, plane: usize, base_x: usize, base_y: usize, tx_sz: u8, x: usize, y: usize) {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let (sx, sy) = self.plane_sub(plane);
        let row = (start_y << sy) >> 2;
        let col = (start_x << sx) >> 2;
        let sb_mask = if self.f.seq.use_128x128_superblock { 31 } else { 15 };
        let sub_row = row & sb_mask;
        let sub_col = col & sb_mask;
        let step_x = (TX_W[tx_sz as usize] >> 2) as usize;
        let step_y = (TX_H[tx_sz as usize] >> 2) as usize;
        let max_x = (self.f.mi_cols * 4) >> sx;
        let max_y = (self.f.mi_rows * 4) >> sy;
        if start_x >= max_x || start_y >= max_y {
            return;
        }
        // Intra prediction (or palette) of this transform block; block copy blocks were predicted whole.
        if !b.is_inter {
            let is_palette = (plane == 0 && b.pal_size_y > 0) || (plane != 0 && b.pal_size_uv > 0);
            if is_palette {
                self.predict_palette(b, plane, start_x, start_y, x, y, tx_sz);
            } else {
                let is_cfl = plane > 0 && b.uv_mode == UV_CFL_PRED;
                let mode = if plane == 0 {
                    b.y_mode
                } else if is_cfl {
                    DC_PRED
                } else {
                    b.uv_mode
                };
                let have_left = (if plane == 0 { b.avail_l } else { b.avail_l_chroma }) || x > 0;
                let have_above = (if plane == 0 { b.avail_u } else { b.avail_u_chroma }) || y > 0;
                let bd = &self.block_decoded[plane];
                let have_ar = bd[sub_row >> sy][(sub_col >> sx) + step_x + 1];
                let have_bl = bd[(sub_row >> sy) + step_y + 1][sub_col >> sx];
                self.predict_intra_block(
                    b, plane, start_x, start_y, tx_sz, have_left, have_above, have_ar, have_bl, mode,
                );
                if is_cfl {
                    self.predict_cfl(b, plane, start_x, start_y, tx_sz);
                }
            }
            if plane == 0 {
                self.max_luma_w = start_x + step_x * 4;
                self.max_luma_h = start_y + step_y * 4;
            }
        }
        if self.analyzing {
            let sc = self.analyze_tx(b, plane, start_x, start_y, tx_sz);
            self.analysis_log.push(sc);
        } else if !b.skip {
            let eob = self.coeffs(b, plane, start_x, start_y, tx_sz);
            if eob > 0 {
                self.reconstruct(b, plane, start_x, start_y, tx_sz);
            }
        }
        // Loop filter transform sizes and decoded flags.
        let stride = self.f.lf_stride[plane];
        let rows = self.f.lf_tx[plane].len() / stride;
        for i in 0..step_y {
            for j in 0..step_x {
                let (rr, cc) = ((row >> sy) + i, (col >> sx) + j);
                if rr < rows && cc < stride {
                    self.f.lf_tx[plane][rr * stride + cc] = tx_sz;
                }
                self.block_decoded[plane][(sub_row >> sy) + i + 1][(sub_col >> sx) + j + 1] = true;
            }
        }
    }

    fn filter_type(&self, b: &Blk, plane: usize) -> bool {
        let is_smooth = |r: usize, c: usize| -> bool {
            let i = r * self.f.mi_cols + c;
            let mode = if plane == 0 { self.f.y_mode[i] } else { self.f.uv_mode[i] };
            mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED
        };
        let (sx, sy) = self.plane_sub(plane);
        let mut above = false;
        let mut left = false;
        if if plane == 0 { b.avail_u } else { b.avail_u_chroma } {
            let mut r = b.mi_row - 1;
            let mut c = b.mi_col;
            if plane > 0 {
                if sx == 1 && (b.mi_col & 1) == 0 {
                    c += 1;
                }
                if sy == 1 && (b.mi_row & 1) == 1 {
                    r -= 1;
                }
            }
            above = is_smooth(r, c.min(self.f.mi_cols - 1));
        }
        if if plane == 0 { b.avail_l } else { b.avail_l_chroma } {
            let mut r = b.mi_row;
            let mut c = b.mi_col - 1;
            if plane > 0 {
                if sx == 1 && (b.mi_col & 1) == 1 {
                    c -= 1;
                }
                if sy == 1 && (b.mi_row & 1) == 0 {
                    r += 1;
                }
            }
            left = is_smooth(r.min(self.f.mi_rows - 1), c);
        }
        above || left
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_intra_block(
        &mut self,
        b: &Blk,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: u8,
        have_left: bool,
        have_above: bool,
        have_ar: bool,
        have_bl: bool,
        mode: u8,
    ) {
        let w = TX_W[tx_sz as usize] as usize;
        let h = TX_H[tx_sz as usize] as usize;
        self.predict_intra_wh(b, plane, x, y, w, h, have_left, have_above, have_ar, have_bl, mode);
    }

    /// Intra prediction of a `w` x `h` region.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_intra_wh(
        &mut self,
        b: &Blk,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        have_left: bool,
        have_above: bool,
        have_ar: bool,
        have_bl: bool,
        mode: u8,
    ) {
        let (sx, sy) = self.plane_sub(plane);
        let bit_depth = self.f.seq.color.bit_depth;
        let max_x = ((self.f.mi_cols * 4) >> sx) - 1;
        let max_y = ((self.f.mi_rows * 4) >> sy) - 1;
        let smooth_neighbor = self.filter_type(b, plane);
        let mut edges = ipred::Edges::default();
        let pl = &mut self.f.planes[plane];
        ipred::build_edges(
            &pl.data, pl.stride, x, y, w, h, have_left, have_above, have_ar, have_bl, max_x, max_y, bit_depth,
            &mut edges,
        );
        let params = ipred::IntraParams {
            mode,
            angle_delta: if plane == 0 { b.angle_y } else { b.angle_uv },
            filter_intra: if plane == 0 && b.use_filter_intra { Some(b.filter_intra_mode) } else { None },
            w,
            h,
            have_left,
            have_above,
            enable_edge_filter: self.f.seq.enable_intra_edge_filter,
            smooth_neighbor,
            bit_depth,
        };
        let stride = pl.stride;
        ipred::predict(&mut pl.data[y * stride + x..], stride, &edges, &params);
    }

    pub(crate) fn predict_cfl(&mut self, b: &Blk, plane: usize, x: usize, y: usize, tx_sz: u8) {
        let w = TX_W[tx_sz as usize] as usize;
        let h = TX_H[tx_sz as usize] as usize;
        let (sx, sy) = (self.f.sub_x, self.f.sub_y);
        let alpha = if plane == 1 { b.cfl_alpha_u } else { b.cfl_alpha_v } as i32;
        let bit_depth = self.f.seq.color.bit_depth;
        let luma_x = x << sx;
        let luma_y = y << sy;
        let avail_w = self.max_luma_w.saturating_sub(luma_x).max(1 << sx);
        let avail_h = self.max_luma_h.saturating_sub(luma_y).max(1 << sy);
        let (luma_plane, rest) = self.f.planes.split_at_mut(1);
        let luma = &luma_plane[0];
        let dst_plane = &mut rest[plane - 1];
        let stride = dst_plane.stride;
        ipred::cfl_predict(
            &mut dst_plane.data[y * stride + x..],
            stride,
            w,
            h,
            &luma.data,
            luma.stride,
            luma_x,
            luma_y,
            avail_w,
            avail_h,
            sx as u8,
            sy as u8,
            alpha,
            bit_depth,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_palette(&mut self, b: &Blk, plane: usize, x: usize, y: usize, bx: usize, by: usize, tx_sz: u8) {
        let w = TX_W[tx_sz as usize] as usize;
        let h = TX_H[tx_sz as usize] as usize;
        let palette = match plane {
            0 => b.pal_y,
            1 => b.pal_u,
            _ => b.pal_v,
        };
        let map = if plane == 0 { &self.color_map_y } else { &self.color_map_uv };
        let pl = &mut self.f.planes[plane];
        for i in 0..h {
            for j in 0..w {
                let idx = map[(by * 4 + i) * 64 + bx * 4 + j] as usize;
                pl.data[(y + i) * pl.stride + x + j] = palette[idx];
            }
        }
    }

    /// Quantizer step sizes (dc, ac) for a plane of the current block.
    pub(crate) fn quant_steps(&self, b: &Blk, plane: usize) -> (i32, i32) {
        let q = &self.f.hdr.quant;
        let qidx = self.f.hdr.qidx(false, b.segment_id as usize, self.current_q_index);
        let (ddc, dac) = match plane {
            0 => (q.delta_q_y_dc, 0),
            1 => (q.delta_q_u_dc, q.delta_q_u_ac),
            _ => (q.delta_q_v_dc, q.delta_q_v_ac),
        };
        let bd = self.f.seq.color.bit_depth;
        let (dct, act): (&[u16; 256], &[u16; 256]) = match bd {
            8 => (&DC_Q, &AC_Q),
            10 => (&DC_Q_10, &AC_Q_10),
            _ => (&DC_Q_12, &AC_Q_12),
        };
        (dct[(qidx + ddc).clamp(0, 255) as usize] as i32, act[(qidx + dac).clamp(0, 255) as usize] as i32)
    }

    pub(crate) fn tx_set_intra(&self, tx_sz: u8) -> u8 {
        // 0 = DCT only, 1 = 7-type set, 2 = 5-type set
        let sqr_up = TX_SQR_UP[tx_sz as usize];
        let sqr = TX_SQR[tx_sz as usize];
        if sqr_up >= TX_32X32 {
            0
        } else if self.f.hdr.reduced_tx_set || sqr == TX_16X16 {
            2
        } else {
            1
        }
    }

    /// `get_tx_set` for inter-style blocks: 0 = DCT only, 1 = 16 types, 2 = 12 types, 3 = IDTX and DCT.
    pub(crate) fn tx_set_inter(&self, tx_sz: u8) -> u8 {
        let sqr_up = TX_SQR_UP[tx_sz as usize];
        let sqr = TX_SQR[tx_sz as usize];
        if sqr_up > TX_32X32 {
            0
        } else if self.f.hdr.reduced_tx_set || sqr_up == TX_32X32 {
            3
        } else if sqr == TX_16X16 {
            2
        } else {
            1
        }
    }

    /// Contexts and fixed choices of one transform block's coefficient syntax.
    pub(crate) fn txb_setup(&self, b: &Blk, plane: usize, start_x: usize, start_y: usize, tx_sz: u8) -> TxbSetup {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let tw = TX_W[tx_sz as usize] as usize;
        let th = TX_H[tx_sz as usize] as usize;
        let w4 = tw >> 2;
        let h4 = th >> 2;
        let tx_sz_ctx = ((TX_SQR[tx_sz as usize] + TX_SQR_UP[tx_sz as usize] + 1) >> 1) as usize;
        let ptype = (plane > 0) as usize;
        let (sx, sy) = self.plane_sub(plane);
        let max_x4 = (self.f.mi_cols + sx) >> sx;
        let max_y4 = (self.f.mi_rows + sy) >> sy;

        // all_zero context
        let plane_sz = if plane == 0 { b.bsize } else { subsampled_size(b.bsize, sx as u8, sy as u8) };
        let (bw, bh) = (block_width(plane_sz), block_height(plane_sz));
        let all_zero_ctx = if plane == 0 {
            let mut top = 0u8;
            let mut left = 0u8;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(self.above_level[plane][x4 + k]);
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(self.left_level[plane][y4 + k]);
                }
            }
            if bw == tw && bh == th {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + (top.max(left) > 3) as usize
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    above |= self.above_level[plane][x4 + i] | self.above_dc[plane][x4 + i];
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    left |= self.left_level[plane][y4 + i] | self.left_dc[plane][y4 + i];
                }
            }
            let mut ctx = 7 + (above != 0) as usize + (left != 0) as usize;
            if bw * bh > tw * th {
                ctx += 3;
            }
            ctx
        };
        let mut dc_sign = 0i32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                match self.above_dc[plane][x4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                match self.left_dc[plane][y4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        let dc_sign_ctx = if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        };
        // Transform type signalling.
        let inter = b.is_inter;
        let set = if inter { self.tx_set_inter(tx_sz) } else { self.tx_set_intra(tx_sz) };
        let mut fixed_type = Some(0u8);
        let mut read_type = false;
        if !(b.lossless || TX_SQR_UP[tx_sz as usize] > TX_32X32) {
            if plane > 0 && inter {
                // Chroma follows the luma transform type at the matching position.
                let cols = self.f.mi_cols;
                let (cx, cy) = (b.mi_col.max(x4 << sx), b.mi_row.max(y4 << sy));
                let t = self.f.tx_types[(cy.min(self.f.mi_rows - 1)) * cols + cx.min(cols - 1)];
                fixed_type = Some(if TX_TYPE_IN_SET_INTER[set as usize][t as usize] { t } else { 0 });
            } else if plane > 0 {
                fixed_type = Some(if set == 0 { 0 } else { MODE_TO_TXFM[b.uv_mode as usize] });
            } else {
                let qidx = if self.f.hdr.seg.enabled {
                    self.f.hdr.qidx(true, b.segment_id as usize, 0)
                } else {
                    self.f.hdr.quant.base_q_idx as i32
                };
                if set > 0 && qidx > 0 {
                    read_type = true;
                    fixed_type = None;
                }
            }
        }
        let intra_dir =
            if b.use_filter_intra { FILTER_INTRA_MODE_TO_INTRA_DIR[b.filter_intra_mode as usize] } else { b.y_mode };
        TxbSetup {
            tx_sz,
            ptype,
            tx_sz_ctx,
            all_zero_ctx,
            dc_sign_ctx,
            tx_set: set,
            read_type,
            fixed_type,
            intra_dir,
            is_inter: inter,
            x4,
            y4,
            w4,
            h4,
        }
    }

    fn coeffs(&mut self, b: &Blk, plane: usize, start_x: usize, start_y: usize, tx_sz: u8) -> usize {
        let s = self.txb_setup(b, plane, start_x, start_y, tx_sz);
        if !C::DECODING {
            let sc = self.coef_queue.pop_front().unwrap_or_default();
            self.cb.eob = sc.eob;
            self.cb.tx_type = sc.tx_type;
            for &(pos, v) in &sc.coefs {
                self.cb.quant[pos as usize] = v;
            }
        }
        let out = code_txb(&mut self.c, &mut self.cdf, &s, &mut self.cb);
        if plane == 0 {
            // TxTypes: later chroma blocks of inter-style blocks look the luma type up here.
            let t = if out.eob > 0 { self.cb.tx_type } else { itx::DCT_DCT };
            self.store_tx_type(s.x4, s.y4, s.w4, s.h4, t);
        }
        for i in 0..s.w4 {
            self.above_level[plane][s.x4 + i] = out.cul_level as u8;
            self.above_dc[plane][s.x4 + i] = out.dc_category;
        }
        for i in 0..s.h4 {
            self.left_level[plane][s.y4 + i] = out.cul_level as u8;
            self.left_dc[plane][s.y4 + i] = out.dc_category;
        }
        out.eob
    }

    pub(crate) fn store_tx_type(&mut self, x4: usize, y4: usize, w4: usize, h4: usize, t: u8) {
        let (rows, cols) = (self.f.mi_rows, self.f.mi_cols);
        for j in 0..h4 {
            for i in 0..w4 {
                if y4 + j < rows && x4 + i < cols {
                    self.f.tx_types[(y4 + j) * cols + x4 + i] = t;
                }
            }
        }
    }

    /// Estimated cost, in 1/256 bit, of coding `cb` as a transform block here (contexts are not modified).
    pub(crate) fn txb_cost(
        &mut self,
        b: &Blk,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: u8,
        cb: &mut CoefBlock,
    ) -> u64 {
        let s = self.txb_setup(b, plane, start_x, start_y, tx_sz);
        let mut cost = super::entropy::CostCoder::new(false);
        code_txb(&mut cost, &mut self.cdf, &s, cb);
        cost.bits
    }

    /// Dequantize `quant` (specification raster order) for a transform block, returning the coefficient array in
    /// the layout the inverse transform expects and the number of non-zero rows and columns.
    pub(crate) fn dequant_block(
        &self,
        b: &Blk,
        plane: usize,
        tx_sz: u8,
        tx_type: u8,
        quant: &[i32],
    ) -> (Vec<i32>, usize, usize) {
        let tw = TX_W[tx_sz as usize] as usize;
        let th = TX_H[tx_sz as usize] as usize;
        let cw = tw.min(32);
        let ch = th.min(32);
        let tx_sz_ctx = ((TX_SQR[tx_sz as usize] + TX_SQR_UP[tx_sz as usize] + 1) >> 1) as i32;
        let dq_shift = (tx_sz_ctx - 2).max(0);
        let bit_depth = self.f.seq.color.bit_depth as i32;
        let (dc_q, ac_q) = self.quant_steps(b, plane);
        let limit = 1i64 << (7 + bit_depth);
        let mut coefs = vec![0i32; tw * th];
        let qm_level = self.f.hdr.seg_qm_level[plane][b.segment_id as usize];
        let qm: Option<&[u8]> = if self.f.hdr.quant.using_qmatrix && !b.lossless && tx_type < IDTX && qm_level < 15 {
            let off = super::qm_tables::QM_OFFSET[tx_sz as usize];
            Some(&super::qm_tables::QM_DATA[qm_level as usize][(plane > 0) as usize][off..off + cw * ch])
        } else {
            None
        };
        let mut nz_rows = 0;
        let mut nz_cols = 0;
        for pos in 0..cw * ch {
            let q = quant[pos];
            if q == 0 {
                continue;
            }
            let (i, j) = (pos / cw, pos % cw);
            let mut step = if pos == 0 { dc_q } else { ac_q } as i64;
            if let Some(m) = qm {
                step = (step * m[pos] as i64 + 16) >> 5;
            }
            let mut dq = ((q.unsigned_abs() as i64 * step) & 0xff_ffff) >> dq_shift;
            if q < 0 {
                dq = -dq;
            }
            coefs[i * tw + j] = dq.clamp(-limit, limit - 1) as i32;
            nz_rows = nz_rows.max(i + 1);
            nz_cols = nz_cols.max(j + 1);
        }
        (coefs, nz_rows, nz_cols)
    }

    pub(crate) fn reconstruct(&mut self, b: &Blk, plane: usize, start_x: usize, start_y: usize, tx_sz: u8) {
        let tw = TX_W[tx_sz as usize] as usize;
        let th = TX_H[tx_sz as usize] as usize;
        let (cw, ch) = (tw.min(32), th.min(32));
        let bit_depth = self.f.seq.color.bit_depth;
        let (mut coefs, nz_rows, nz_cols) = self.dequant_block(b, plane, tx_sz, self.cb.tx_type, &self.cb.quant);
        let stride = self.f.planes[plane].stride;
        let dst = &mut self.f.planes[plane].data[start_y * stride + start_x..];
        itx::inverse_transform_add(
            &mut coefs,
            tx_sz,
            self.cb.tx_type,
            b.lossless,
            bit_depth,
            nz_rows,
            nz_cols,
            dst,
            stride,
        );
        // Reset the quantized coefficients for the next block.
        for pos in 0..cw * ch {
            self.cb.quant[pos] = 0;
        }
    }
}

/// Setup of one transform block's coefficient syntax (contexts derived from the neighbourhood).
pub(crate) struct TxbSetup {
    pub tx_sz: u8,
    pub ptype: usize,
    pub tx_sz_ctx: usize,
    pub all_zero_ctx: usize,
    pub dc_sign_ctx: usize,
    /// 0 = DCT only, 1 = seven intra types, 2 = five intra types.
    pub tx_set: u8,
    pub read_type: bool,
    pub fixed_type: Option<u8>,
    pub intra_dir: u8,
    pub is_inter: bool,
    pub x4: usize,
    pub y4: usize,
    pub w4: usize,
    pub h4: usize,
}

pub(crate) struct TxbOut {
    pub eob: usize,
    pub cul_level: u32,
    pub dc_category: u8,
}

/// The coefficient syntax of one transform block (spec `coeffs()` without the context bookkeeping), generic over
/// decoding, encoding and cost estimation.
pub(crate) fn code_txb<K: Coder>(k: &mut K, cdf: &mut Cdfs, s: &TxbSetup, cb: &mut CoefBlock) -> TxbOut {
    let (tx_sz, ptype, tx_sz_ctx) = (s.tx_sz, s.ptype, s.tx_sz_ctx);
    let all_zero = k.symbol(&mut cdf.txb_skip[tx_sz_ctx][s.all_zero_ctx], (cb.eob == 0) as usize) != 0;
    let mut cul_level = 0u32;
    let mut dc_category = 0u8;
    let mut eob = 0usize;
    if all_zero {
        cb.eob = 0;
    } else {
        let tx_type = if s.read_type {
            let sqr = TX_SQR[tx_sz as usize] as usize;
            let want = cb.tx_type;
            if s.is_inter {
                let (table, cdf): (&[u8], &mut [u16]) = match s.tx_set {
                    1 => (&TX_TYPE_INTER_INV_SET1, &mut cdf.inter_tx_set1[sqr]),
                    2 => (&TX_TYPE_INTER_INV_SET2, &mut cdf.inter_tx_set2),
                    _ => (&TX_TYPE_INTER_INV_SET3, &mut cdf.inter_tx_set3[sqr]),
                };
                let want_sym = table.iter().position(|&t| t == want).unwrap_or(0);
                table[k.symbol(cdf, want_sym)]
            } else if s.tx_set == 1 {
                let want_sym = TX_TYPE_INTRA_INV_SET1.iter().position(|&t| t == want).unwrap_or(1);
                TX_TYPE_INTRA_INV_SET1[k.symbol(&mut cdf.intra_tx_set1[sqr][s.intra_dir as usize], want_sym)]
            } else {
                let want_sym = TX_TYPE_INTRA_INV_SET2.iter().position(|&t| t == want).unwrap_or(1);
                TX_TYPE_INTRA_INV_SET2[k.symbol(&mut cdf.intra_tx_set2[sqr][s.intra_dir as usize], want_sym)]
            }
        } else {
            s.fixed_type.unwrap_or(0)
        };
        cb.tx_type = tx_type;
        let class = tx_class(tx_type);
        let scan_kind = match class {
            TX_CLASS_VERT => 1,
            TX_CLASS_HORIZ => 2,
            _ => 0,
        };
        let scan: &[u16] = SCAN_TABLE[tx_sz as usize][if tx_type == IDTX { 0 } else { scan_kind }];
        let adj = ADJUSTED_TX_SIZE[tx_sz as usize] as usize;
        let bwl = TX_W_LOG2[adj] as usize;
        let cw = 1usize << bwl;
        let ch = TX_H[adj] as usize;
        let area = cw * ch;
        // eob
        let eob_multisize = (TX_W_LOG2[tx_sz as usize].min(5) + TX_H_LOG2[tx_sz as usize].min(5)) as usize - 4;
        let ectx = if class == TX_CLASS_2D { 0 } else { 1 };
        let want_eob = cb.eob;
        let (want_pt, want_extra) = eob_to_pt(want_eob);
        let pt_sym = match eob_multisize {
            0 => k.symbol(&mut cdf.eob_pt_16[ptype][ectx], want_pt - 1),
            1 => k.symbol(&mut cdf.eob_pt_32[ptype][ectx], want_pt - 1),
            2 => k.symbol(&mut cdf.eob_pt_64[ptype][ectx], want_pt - 1),
            3 => k.symbol(&mut cdf.eob_pt_128[ptype][ectx], want_pt - 1),
            4 => k.symbol(&mut cdf.eob_pt_256[ptype][ectx], want_pt - 1),
            5 => k.symbol(&mut cdf.eob_pt_512[ptype][ectx], want_pt - 1),
            _ => k.symbol(&mut cdf.eob_pt_1024[ptype][ectx], want_pt - 1),
        };
        let eob_pt = pt_sym + 1;
        eob = if eob_pt < 2 { eob_pt } else { (1 << (eob_pt - 2)) + 1 };
        let mut eob_shift = eob_pt as isize - 3;
        if eob_shift >= 0 {
            let extra_want = (want_extra >> eob_shift) & 1;
            let eob_extra = k.symbol(&mut cdf.eob_extra[tx_sz_ctx][ptype][eob_pt - 3], extra_want) != 0;
            if eob_extra {
                eob += 1 << eob_shift;
            }
            let n = (eob_pt as isize - 2).max(0);
            for i in 1..n {
                eob_shift = n - 1 - i;
                let bit = k.literal(1, ((want_extra >> eob_shift) & 1) as u32) != 0;
                if bit {
                    eob += 1 << eob_shift;
                }
            }
        }
        if eob > area {
            // Corrupt stream; clamp so the rest of the block stays in bounds.
            eob = area;
        }

        // Levels, in reverse scan order.
        let stride = 36usize;
        let mut levels = vec![0u8; stride * (ch + 4)];
        let mut quant = vec![0u32; cw * ch];
        let want_abs = |pos: usize, q: &CoefBlock| q.quant[pos].unsigned_abs();
        for c in (0..eob).rev() {
            let pos = scan[c] as usize;
            let row = pos >> bwl;
            let col = pos & (cw - 1);
            let want_level = if K::DECODING { 0 } else { want_abs(pos, cb).min(15) as usize };
            let mut level;
            if c == eob - 1 {
                let ctx = if c == 0 {
                    0
                } else if c <= area / 8 {
                    1
                } else if c <= area / 4 {
                    2
                } else {
                    3
                };
                let s = k.symbol(&mut cdf.coeff_base_eob[tx_sz_ctx][ptype][ctx], want_level.clamp(1, 3) - 1);
                level = s + 1;
            } else {
                let ctx = coeff_base_ctx(&levels, stride, tx_sz, class, row, col);
                level = k.symbol(&mut cdf.coeff_base[tx_sz_ctx][ptype][ctx], want_level.min(3));
            }
            if level > 2 {
                let ctx = coeff_br_ctx(&levels, stride, class, row, col, pos);
                let mut want_rest = want_level.saturating_sub(3);
                for _ in 0..4 {
                    let sym = k.symbol(&mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype][ctx], want_rest.min(3));
                    want_rest = want_rest.saturating_sub(3);
                    level += sym;
                    if sym < 3 {
                        break;
                    }
                }
            }
            levels[row * stride + col] = level as u8;
            quant[pos] = level as u32;
        }
        // Signs and Golomb-coded remainders, in scan order.
        for c in 0..eob {
            let pos = scan[c] as usize;
            let want = cb.quant[pos];
            let mut sign = false;
            if quant[pos] != 0 {
                if c == 0 {
                    let ctx = s.dc_sign_ctx;
                    sign = k.symbol(&mut cdf.dc_sign[ptype][ctx], (want < 0) as usize) != 0;
                } else {
                    sign = k.literal(1, (want < 0) as u32) != 0;
                }
            }
            if quant[pos] > 14 {
                // Golomb code for the amount above 14.
                let value = want.unsigned_abs();
                let x_want = value.wrapping_sub(14);
                let length_want = 32 - x_want.leading_zeros();
                let mut length = 0u32;
                loop {
                    length += 1;
                    let bit = k.literal(1, (length == length_want) as u32) != 0;
                    if bit || length >= 32 {
                        break;
                    }
                    if length > 20 {
                        break;
                    }
                }
                let mut x = 1u32;
                for i in (0..length.saturating_sub(1)).rev() {
                    let bit = k.literal(1, (x_want >> i) & 1);
                    x = (x << 1) | bit;
                }
                quant[pos] = x.wrapping_add(14);
            }
            if pos == 0 && quant[pos] > 0 {
                dc_category = if sign { 1 } else { 2 };
            }
            quant[pos] &= 0xfffff;
            cul_level = cul_level.wrapping_add(quant[pos]);
            cb.quant[pos] = if sign { -(quant[pos] as i32) } else { quant[pos] as i32 };
        }
        cul_level = cul_level.min(63);
        cb.eob = eob;
    }
    TxbOut { eob, cul_level, dc_category }
}

/// `eob` → (eob_pt, extra bits) as in the encoder's `eob` binarization.
fn eob_to_pt(eob: usize) -> (usize, usize) {
    if eob <= 2 {
        return (eob.max(1), 0);
    }
    // eob_pt such that (1 << (eob_pt - 2)) + 1 <= eob < (1 << (eob_pt - 1)) + 1
    let pt = (usize::BITS - (eob - 1).leading_zeros()) as usize + 1;
    (pt, eob - ((1 << (pt - 2)) + 1))
}

#[inline]
fn lv(levels: &[u8], stride: usize, row: usize, col: usize) -> usize {
    levels[row * stride + col] as usize
}

fn coeff_base_ctx(levels: &[u8], stride: usize, tx_sz: u8, class: u8, row: usize, col: usize) -> usize {
    let mut mag;
    match class {
        TX_CLASS_2D => {
            mag = lv(levels, stride, row, col + 1).min(3)
                + lv(levels, stride, row + 1, col).min(3)
                + lv(levels, stride, row + 1, col + 1).min(3)
                + lv(levels, stride, row, col + 2).min(3)
                + lv(levels, stride, row + 2, col).min(3);
        }
        TX_CLASS_HORIZ => {
            mag = lv(levels, stride, row, col + 1).min(3)
                + lv(levels, stride, row + 1, col).min(3)
                + lv(levels, stride, row, col + 2).min(3)
                + lv(levels, stride, row, col + 3).min(3)
                + lv(levels, stride, row, col + 4).min(3);
        }
        _ => {
            mag = lv(levels, stride, row, col + 1).min(3)
                + lv(levels, stride, row + 1, col).min(3)
                + lv(levels, stride, row + 2, col).min(3)
                + lv(levels, stride, row + 3, col).min(3)
                + lv(levels, stride, row + 4, col).min(3);
        }
    }
    let ctx = ((mag + 1) >> 1).min(4);
    mag = 0;
    let _ = mag;
    match class {
        TX_CLASS_2D => {
            if row == 0 && col == 0 {
                return 0;
            }
            ctx + COEFF_BASE_CTX_OFFSET[tx_sz as usize][row.min(4)][col.min(4)] as usize
        }
        TX_CLASS_VERT => ctx + COEFF_BASE_POS_CTX_OFFSET_1D[row.min(31)] as usize,
        _ => ctx + COEFF_BASE_POS_CTX_OFFSET_1D[col.min(31)] as usize,
    }
}

fn coeff_br_ctx(levels: &[u8], stride: usize, class: u8, row: usize, col: usize, pos: usize) -> usize {
    let mag = match class {
        TX_CLASS_2D => {
            lv(levels, stride, row, col + 1).min(15)
                + lv(levels, stride, row + 1, col).min(15)
                + lv(levels, stride, row + 1, col + 1).min(15)
        }
        TX_CLASS_HORIZ => {
            lv(levels, stride, row, col + 1).min(15)
                + lv(levels, stride, row + 1, col).min(15)
                + lv(levels, stride, row, col + 2).min(15)
        }
        _ => {
            lv(levels, stride, row, col + 1).min(15)
                + lv(levels, stride, row + 1, col).min(15)
                + lv(levels, stride, row + 2, col).min(15)
        }
    };
    let mag = ((mag + 1) >> 1).min(6);
    if pos == 0 {
        return mag;
    }
    match class {
        TX_CLASS_2D => {
            if row < 2 && col < 2 {
                return mag + 7;
            }
        }
        TX_CLASS_HORIZ => {
            if col == 0 {
                return mag + 7;
            }
        }
        _ => {
            if row == 0 {
                return mag + 7;
            }
        }
    }
    mag + 14
}
