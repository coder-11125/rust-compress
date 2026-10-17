//! AV1 intra-frame encoder: block planning on top of the shared tile coder.
//!
//! The syntax is written by [`TileCoder`] with a [`SymbolEncoder`]; this module decides what to write. Planning
//! happens inside the coder's traversal: when it reaches a partition node or a block it asks the planner
//! (methods of `TileCoder` defined here) for the decision, which also reconstructs the block so later blocks see
//! the decoder's pixels.

use super::coef_tables::SCAN_TABLE;
use super::consts::*;
use super::entropy::Coder;
use super::frame::Plane;
use super::itx;
use super::tile::{Blk, CoefBlock, FrameState, StoredCoefs, TileCoder};

/// Source pictures and encoder tuning, attached to a [`TileCoder`] that encodes.
pub struct EncSrc {
    /// Source planes padded to the frame buffer size (edge replicated).
    pub planes: Vec<Plane>,
    /// Lagrange multiplier: cost = sse + lambda * bits (bits in 1/256 units are converted by the caller).
    pub lambda: f64,
    /// Fixed block size used by the simple planner.
    pub fixed_bsize: u8,
    /// Rounding offset of the quantizer, 0..=256 (256 = round to nearest).
    pub quant_round: i64,
}

/// Pad (replicating edges) the planes of a picture to the size of the frame buffers in `fs`.
pub fn pad_source(fs: &FrameState, src: &super::frame::Frame) -> Vec<Plane> {
    let mut out = Vec::new();
    for (p, dst) in fs.planes.iter().enumerate() {
        let sp = &src.planes[p];
        let mut pl = Plane::new(dst.width, dst.height);
        for y in 0..dst.height {
            let sy = y.min(sp.height - 1);
            let row = sp.row(sy);
            let drow = pl.row_mut(y);
            for (x, d) in drow.iter_mut().enumerate() {
                *d = row[x.min(sp.width - 1)];
            }
        }
        out.push(pl);
    }
    out
}

/// Quantize forward-transform output (`coefs`, row stride = transform width) into the specification's raster
/// order, returning the number of coefficients up to the last non-zero one in scan order.
pub fn quantize(
    coefs: &[i32],
    tx_sz: u8,
    tx_type: u8,
    dc_q: i64,
    ac_q: i64,
    round: i64,
    out: &mut [i32; 1024],
) -> usize {
    let tw = TX_W[tx_sz as usize] as usize;
    let th = TX_H[tx_sz as usize] as usize;
    let (cw, ch) = (tw.min(32), th.min(32));
    let tx_sz_ctx = ((TX_SQR[tx_sz as usize] + TX_SQR_UP[tx_sz as usize] + 1) >> 1) as i32;
    let shift = (tx_sz_ctx - 2).max(0);
    let mut any = false;
    for i in 0..ch {
        for j in 0..cw {
            let pos = i * cw + j;
            let c = coefs[i * tw + j] as i64;
            let step = if pos == 0 { dc_q } else { ac_q };
            let mag = ((c.abs() << shift) * 256 + step * round) / (step * 256);
            let q = mag.min(0xfffff) as i32;
            out[pos] = if c < 0 { -q } else { q };
            any |= q != 0;
        }
    }
    if !any {
        return 0;
    }
    let class = tx_class(tx_type);
    let kind = if tx_type == IDTX {
        0
    } else {
        match class {
            TX_CLASS_VERT => 1,
            TX_CLASS_HORIZ => 2,
            _ => 0,
        }
    };
    let scan = SCAN_TABLE[tx_sz as usize][kind];
    let mut eob = 0;
    for (idx, &pos) in scan.iter().enumerate() {
        if out[pos as usize] != 0 {
            eob = idx + 1;
        }
    }
    eob
}

impl<C: Coder> TileCoder<'_, C> {
    /// Partition decision for the node at (r, c): from the superblock search, or the next logged one when replaying.
    pub(crate) fn next_partition(&mut self, _r: usize, _c: usize, _bsize: u8, _has_rows: bool, _has_cols: bool) -> u8 {
        if self.replay {
            return self.part_queue.pop_front().unwrap_or(0);
        }
        let p = self.part_cache.pop_front().unwrap_or(PARTITION_SPLIT);
        self.part_log.push(p);
        p
    }

    /// The planned block at (r, c): run the analysis to obtain its coefficients and queue them for the writer.
    pub(crate) fn next_block(&mut self, r: usize, c: usize, bsize: u8) -> Blk {
        if self.replay {
            return self.blk_queue.pop_front().unwrap_or_default();
        }
        let mut b = self.blk_cache.pop_front().unwrap_or_default();
        b.mi_row = r;
        b.mi_col = c;
        b.bsize = bsize;
        self.set_block_geometry(&mut b);
        let log = self.analyze_block(&mut b);
        b.skip = log.iter().all(|s| s.eob == 0);
        if std::env::var_os("AV1_CHECK").is_some() {
            let np = if b.has_chroma { 3 } else { 1 };
            let mut snap = Vec::new();
            for plane in 0..np {
                let (sx, sy) = self.plane_sub(plane);
                let (x0, y0) = ((b.mi_col >> sx) * 4, (b.mi_row >> sy) * 4);
                let (w, h) = ((block_width(b.bsize) >> sx).max(4), (block_height(b.bsize) >> sy).max(4));
                let pl = &self.f.planes[plane];
                let mut v = Vec::with_capacity(w * h);
                for i in 0..h {
                    for j in 0..w {
                        v.push(pl.at(x0 + j, y0 + i));
                    }
                }
                snap.push(v);
            }
            self.dbg_snap = Some(snap);
        }
        // A skipped block codes no coefficients at all, so nothing is queued for it.
        if !b.skip {
            for sc in &log {
                self.coef_queue.push_back(sc.clone());
                self.coef_log.push(sc.clone());
            }
        }
        self.blk_log.push(b.clone());
        b
    }

    /// Run the residual traversal in analysis mode: predict, transform, quantize and reconstruct every transform
    /// block of `b`, returning the coefficients in coding order.
    pub(crate) fn analyze_block(&mut self, b: &mut Blk) -> Vec<StoredCoefs> {
        let saved = self.block_decoded;
        self.analyzing = true;
        self.analysis_log.clear();
        let _ = self.residual(b);
        self.analyzing = false;
        self.block_decoded = saved;
        std::mem::take(&mut self.analysis_log)
    }

    /// Transform block analysis: pick the transform type, quantize, and reconstruct into the frame buffer
    /// (which already holds the prediction).
    pub(crate) fn analyze_tx(&mut self, b: &Blk, plane: usize, x: usize, y: usize, tx_sz: u8) -> StoredCoefs {
        let tw = TX_W[tx_sz as usize] as usize;
        let th = TX_H[tx_sz as usize] as usize;
        let bit_depth = self.f.seq.color.bit_depth;
        let (dc_q, ac_q) = self.quant_steps(b, plane);
        let round = self.enc_src.as_ref().map_or(128, |e| e.quant_round);
        // Residual = source - prediction.
        let mut residual = vec![0i32; tw * th];
        {
            let src = &self.enc_src.as_ref().expect("encoder source").planes[plane];
            let cur = &self.f.planes[plane];
            for i in 0..th {
                for j in 0..tw {
                    residual[i * tw + j] = src.at(x + j, y + i) as i32 - cur.at(x + j, y + i) as i32;
                }
            }
        }
        let tx_type = if b.lossless || TX_SQR_UP[tx_sz as usize] > TX_32X32 {
            0
        } else if plane > 0 {
            if self.tx_set_intra(tx_sz) == 0 { 0 } else { MODE_TO_TXFM[b.uv_mode as usize] }
        } else {
            0
        };
        let mut coefs = vec![0i32; tw * th];
        itx::forward_transform(&residual, tw, tx_sz, tx_type, b.lossless, bit_depth, &mut coefs);
        let mut quant = [0i32; 1024];
        let eob = quantize(&coefs, tx_sz, tx_type, dc_q as i64, ac_q as i64, round, &mut quant);
        let mut sc = StoredCoefs { tx_type, eob, coefs: Vec::new() };
        if let (Ok(dx), Ok(dy)) = (std::env::var("AV1_DBG_X"), std::env::var("AV1_DBG_Y")) {
            let (dx, dy): (usize, usize) = (dx.parse().unwrap(), dy.parse().unwrap());
            if plane == std::env::var("AV1_DBG_PLANE").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize)
                && self.analyzing
                && self.an_planes == 7
                && dx >= x
                && dx < x + tw
                && dy >= y
                && dy < y + th
            {
                eprintln!("tx at ({x},{y}) {tw}x{th} type {tx_type} eob {eob} dcq {dc_q} acq {ac_q}");
                eprintln!("residual row {}: {:?}", dy - y, &residual[(dy - y) * tw..(dy - y) * tw + tw]);
                eprintln!("coefs[0..8]: {:?} quant[0..8] {:?}", &coefs[..tw.min(8)], &quant[..tw.min(8)]);
            }
        }
        {
            // Distortion and rate of this choice, for the search.
            let mut cb = CoefBlock { eob, tx_type, quant };
            let bits = self.txb_cost(b, plane, x, y, tx_sz, &mut cb);
            self.an_bits += bits;
            let mut recon = vec![0u16; tw * th];
            for i in 0..th {
                for j in 0..tw {
                    recon[i * tw + j] = self.f.planes[plane].at(x + j, y + i);
                }
            }
            if eob > 0 {
                let (mut dq, nzr, nzc) = self.dequant_block(b, plane, tx_sz, tx_type, &quant);
                itx::inverse_transform_add(&mut dq, tx_sz, tx_type, b.lossless, bit_depth, nzr, nzc, &mut recon, tw);
            }
            let src = &self.enc_src.as_ref().expect("encoder source").planes[plane];
            let mut sse = 0f64;
            for i in 0..th {
                for j in 0..tw {
                    let d = src.at(x + j, y + i) as i64 - recon[i * tw + j] as i64;
                    sse += (d * d) as f64;
                }
            }
            self.an_dist += sse;
            if let (Ok(dx), Ok(dy)) = (std::env::var("AV1_DBG_X"), std::env::var("AV1_DBG_Y")) {
                let (dx, dy): (usize, usize) = (dx.parse().unwrap(), dy.parse().unwrap());
                if plane == std::env::var("AV1_DBG_PLANE").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize)
                    && self.analyzing
                    && self.an_planes == 7
                    && dx >= x
                    && dx < x + tw
                    && dy >= y
                    && dy < y + th
                {
                    eprintln!("  -> sse {sse} first recon row {:?}", &recon[(dy - y) * tw..(dy - y) * tw + tw]);
                    let srow: Vec<u16> = (0..tw).map(|j| src.at(x + j, dy)).collect();
                    eprintln!("     src row {:?}", srow);
                }
            }
        }
        if eob > 0 {
            let (cw, ch) = (tw.min(32), th.min(32));
            for pos in 0..cw * ch {
                if quant[pos] != 0 {
                    sc.coefs.push((pos as u16, quant[pos]));
                    self.cb.quant[pos] = quant[pos];
                }
            }
            self.cb.tx_type = tx_type;
            self.reconstruct(b, plane, x, y, tx_sz);
        }
        sc
    }
}

// -------------------------------------------------------------------------------------------------
// Frame level
// -------------------------------------------------------------------------------------------------

use super::bits::BitWriter;
use super::cdf_tables::Cdfs;
use super::entropy::SymbolEncoder;
use super::frame::Frame;
use super::obu::{self, ColorConfig, FrameHeader, OperatingPoint, SequenceHeader};
use super::obu_write;
use crate::error::Result;

#[derive(Clone, Debug)]
pub struct EncodeParams {
    pub base_q_idx: u8,
    pub fixed_bsize: u8,
    pub quant_round: i64,
    pub lambda_scale: f64,
}

impl Default for EncodeParams {
    fn default() -> Self {
        EncodeParams { base_q_idx: 100, fixed_bsize: BLOCK_16X16, quant_round: 120, lambda_scale: 0.12 }
    }
}

/// An encoded still picture.
pub struct Encoded {
    /// Sequence header OBU followed by the frame OBU (no temporal delimiter).
    pub data: Vec<u8>,
    /// The picture the decoder will produce (after the loop filters).
    pub recon: Frame,
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
}

fn level_for(width: usize, height: usize) -> u8 {
    const LIMITS: [(usize, u8); 7] =
        [(147_456, 0), (278_784, 1), (665_856, 4), (1_065_024, 5), (2_359_296, 8), (8_912_896, 12), (35_651_584, 16)];
    let area = width * height;
    LIMITS.iter().find(|(max, _)| area <= *max).map_or(31, |&(_, level)| level)
}

pub fn make_sequence_header(color: &ColorConfig, width: usize, height: usize) -> SequenceHeader {
    let bits = |n: usize| (usize::BITS - (n.max(2) - 1).leading_zeros()).max(1);
    let profile = if color.sub_x == 0 && color.sub_y == 0 && !color.mono_chrome {
        1
    } else if (color.sub_x == 1 && color.sub_y == 0) || color.bit_depth == 12 {
        2
    } else {
        0
    };
    SequenceHeader {
        profile,
        still_picture: true,
        reduced_still_picture_header: true,
        operating_points: vec![OperatingPoint {
            idc: 0,
            level: level_for(width, height),
            tier: false,
            decoder_model_present: false,
        }],
        frame_width_bits: bits(width),
        frame_height_bits: bits(height),
        max_frame_width: width as u32,
        max_frame_height: height as u32,
        use_128x128_superblock: false,
        enable_filter_intra: false,
        enable_intra_edge_filter: true,
        seq_force_screen_content_tools: obu::SELECT_SCREEN_CONTENT_TOOLS,
        seq_force_integer_mv: obu::SELECT_INTEGER_MV,
        enable_superres: false,
        enable_cdef: true,
        enable_restoration: false,
        color: color.clone(),
        film_grain_params_present: false,
        ..Default::default()
    }
}

/// Encode one picture (planes already at the sequence's bit depth and subsampling).
pub fn encode_picture(src: &Frame, color: &ColorConfig, p: &EncodeParams) -> Result<Encoded> {
    let (w, h) = (src.width, src.height);
    let seq = make_sequence_header(color, w, h);
    let mut hdr = FrameHeader {
        frame_type: obu::KEY_FRAME,
        show_frame: true,
        frame_width: w,
        frame_height: h,
        upscaled_width: w,
        render_width: w,
        render_height: h,
        superres_denom: 8,
        mi_cols: 2 * w.div_ceil(8),
        mi_rows: 2 * h.div_ceil(8),
        ..Default::default()
    };
    hdr.tile_info = obu_write::uniform_tile_info(&seq, hdr.mi_cols, hdr.mi_rows, 0, 0);
    hdr.quant.base_q_idx = p.base_q_idx;
    hdr.coded_lossless = p.base_q_idx == 0;
    hdr.all_lossless = hdr.coded_lossless;
    hdr.lossless_array = [p.base_q_idx == 0; 8];
    hdr.loop_filter.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
    hdr.cdef.damping = 3;
    hdr.tx_mode_select = true;

    let mut fs = FrameState::new(&seq, &hdr)?;
    let planes = pad_source(&fs, src);
    let cdf = Cdfs::new(p.base_q_idx as u32);
    let enc = SymbolEncoder::new(!hdr.disable_cdf_update);
    let ti = hdr.tile_info.clone();
    let (rows, cols) = ((ti.mi_row_starts[0], ti.mi_row_starts[1]), (ti.mi_col_starts[0], ti.mi_col_starts[1]));
    let tile_bytes = {
        let mut tc = TileCoder::new(&mut fs, cdf, enc, rows, cols);
        let q = ac_q_for(p.base_q_idx, color.bit_depth) / 8.0;
        tc.enc_src = Some(EncSrc {
            planes,
            lambda: p.lambda_scale * q * q,
            fixed_bsize: p.fixed_bsize,
            quant_round: p.quant_round,
        });
        tc.decode_tile()?;
        if std::env::var_os("AV1_STATS").is_some() {
            let mut sizes = [0usize; 22];
            let mut skips = 0;
            let mut modes = [0usize; 14];
            let mut txs = [0usize; 19];
            let mut uvm = [0usize; 14];
            for b in &tc.blk_log {
                sizes[b.bsize as usize] += 1;
                skips += b.skip as usize;
                modes[b.y_mode as usize] += 1;
                if b.has_chroma {
                    uvm[b.uv_mode as usize] += 1;
                }
                txs[b.tx_size as usize] += 1;
            }
            eprintln!(
                "blocks {} skip {skips}\nsizes {sizes:?}\nymodes {modes:?}\nuvmodes {uvm:?}\ntx {txs:?}",
                tc.blk_log.len()
            );
            let (mut nz, mut tot) = (0usize, 0usize);
            for c in &tc.coef_log {
                nz += c.coefs.len();
                tot += 1;
            }
            eprintln!("tx blocks {tot}, nonzero coefs {nz}");
        }
        tc.c.finish()
    };
    choose_filters(&mut fs, src, p.base_q_idx);
    let hdr = fs.hdr.clone();
    let mut hw = BitWriter::new();
    obu_write::frame_header_bits(&seq, &hdr, &mut hw);
    let mut payload = hw.finish();
    payload.extend_from_slice(&tile_bytes);
    let mut data = obu_write::write_obu(obu::OBU_SEQUENCE_HEADER, &obu_write::sequence_header_payload(&seq));
    data.extend(obu_write::write_obu(obu::OBU_FRAME, &payload));
    let recon = super::decoder::finish_frame(fs)?;
    Ok(Encoded { data, recon, seq, hdr })
}

/// Sum of squared differences over the visible area of one plane.
fn plane_sse(a: &Frame, b: &Frame, plane: usize) -> f64 {
    let (pa, pb) = (&a.planes[plane], &b.planes[plane]);
    let (sx, sy) = if plane == 0 { (0, 0) } else { (a.sub_x as usize, a.sub_y as usize) };
    let (w, h) = ((b.width + sx) >> sx, (b.height + sy) >> sy);
    let mut sse = 0f64;
    for y in 0..h {
        let (ra, rb) = (pa.row(y), pb.row(y));
        let mut acc = 0u64;
        for x in 0..w {
            let d = ra[x] as i64 - rb[x] as i64;
            acc += (d * d) as u64;
        }
        sse += acc as f64;
    }
    sse
}

/// Pick the deblocking levels and the CDEF strengths that minimise the error against the source, and store them
/// in the frame header. The entropy-coded data does not depend on them.
fn choose_filters(fs: &mut FrameState, src: &Frame, base_q_idx: u8) {
    use super::cdef::{CdefParams, cdef_frame};
    use super::deblock::deblock_frame;
    use super::decoder::{build_deblock, coded_frame};
    if fs.hdr.coded_lossless || std::env::var_os("AV1_NOFILT").is_some() {
        return;
    }
    let np = fs.num_planes;
    let mut cur0 = coded_frame(fs);
    cur0.width = fs.hdr.frame_width;
    cur0.height = fs.hdr.frame_height;
    let (params0, mi) = build_deblock(fs);
    let deblocked_with = |levels: [u8; 4]| -> Frame {
        let mut f = cur0.clone();
        let mut p = params0.clone();
        p.filter_level = levels;
        deblock_frame(&mut f, &p, &mi);
        f
    };
    // Luma level: coarse scan then refinement.
    let luma_cost = |l: u8| plane_sse(&deblocked_with([l, l, 0, 0]), src, 0);
    let mut best = (luma_cost(0), 0u8);
    for l in (4..=60).step_by(4) {
        let c = luma_cost(l);
        if c < best.0 {
            best = (c, l);
        }
    }
    for d in [-2i32, -1, 1, 2] {
        let l = (best.1 as i32 + d).clamp(0, 63) as u8;
        let c = luma_cost(l);
        if c < best.0 {
            best = (c, l);
        }
    }
    let luma_level = best.1;
    let mut levels = [luma_level, luma_level, 0, 0];
    if luma_level > 0 && np > 1 {
        for pl in 1..3 {
            let cost = |l: u8| {
                let mut lv = [luma_level, luma_level, 0, 0];
                lv[1 + pl] = l;
                plane_sse(&deblocked_with(lv), src, pl)
            };
            let mut b = (cost(0), 0u8);
            for l in (4..=60).step_by(4) {
                let c = cost(l);
                if c < b.0 {
                    b = (c, l);
                }
            }
            for d in [-2i32, -1, 1, 2] {
                let l = (b.1 as i32 + d).clamp(0, 63) as u8;
                let c = cost(l);
                if c < b.0 {
                    b = (c, l);
                }
            }
            levels[1 + pl] = b.1;
        }
    }
    fs.hdr.loop_filter.level = levels;
    let deblocked = deblocked_with(levels);

    // CDEF: one strength set for the whole frame.
    let damping = (3 + (base_q_idx >> 6)).min(6);
    let skip: Vec<bool> = fs.skip.iter().map(|&s| s != 0).collect();
    let mut out = deblocked.clone();
    let mut cdef_cost = |y: (u8, u8), uv: (u8, u8), plane: usize| -> f64 {
        let p = CdefParams { damping, bits: 0, y_pri: [y.0; 8], y_sec: [y.1; 8], uv_pri: [uv.0; 8], uv_sec: [uv.1; 8] };
        cdef_frame(&deblocked, &mut out, &p, &fs.cdef_idx, fs.cdef_stride, &skip, fs.mi_cols, fs.mi_rows);
        plane_sse(&out, src, plane)
    };
    let pris = [0u8, 1, 2, 3, 4, 6, 8, 11, 15];
    let secs = [0u8, 1, 2, 4];
    let mut yb = (f64::MAX, (0u8, 0u8));
    for &pr in &pris {
        for &sc in &secs {
            let c = cdef_cost((pr, sc), (0, 0), 0);
            if c < yb.0 {
                yb = (c, (pr, sc));
            }
        }
    }
    let mut uvb = (0.0, (0u8, 0u8));
    if np > 1 {
        uvb.0 = f64::MAX;
        for &pr in &pris {
            for &sc in &secs {
                let c = cdef_cost(yb.1, (pr, sc), 1) + cdef_cost(yb.1, (pr, sc), 2);
                if c < uvb.0 {
                    uvb = (c, (pr, sc));
                }
            }
        }
    }
    fs.hdr.cdef.damping = damping;
    fs.hdr.cdef.bits = 0;
    fs.hdr.cdef.y_pri[0] = yb.1.0;
    fs.hdr.cdef.y_sec[0] = yb.1.1;
    fs.hdr.cdef.uv_pri[0] = uvb.1.0;
    fs.hdr.cdef.uv_sec[0] = uvb.1.1;
}

fn ac_q_for(qidx: u8, bit_depth: u8) -> f64 {
    use super::coef_tables::{AC_Q, AC_Q_10, AC_Q_12};
    let t = match bit_depth {
        8 => &AC_Q,
        10 => &AC_Q_10,
        _ => &AC_Q_12,
    };
    t[qidx as usize] as f64
}
