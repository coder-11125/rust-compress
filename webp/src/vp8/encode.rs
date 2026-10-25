//! VP8 key-frame encoder.
//!
//! Every macroblock is predicted from already *reconstructed* neighbours (exactly what the decoder
//! will have), and the prediction mode with the lowest rate-distortion cost `SSE + lambda * bits` is
//! kept: whole-block luma modes against per-4x4 sub-block modes, then chroma. Token probabilities
//! are tuned from the statistics of a first pass, the loop filter strength is chosen by measuring
//! the filtered reconstruction, and the result is written as one mode partition and one token
//! partition.

use super::boolcoder::BoolEncoder;
use super::decode::{BMODE_TREE, Dequant, Frame, KF_YMODE_TREE, UV_MODE_TREE, dequant_for};
use super::loopfilter::{FilterInfo, Plane, filter_macroblock};
use super::predict::{ORIGIN, WS, fill_borders, predict_4x4, predict_block, store_block};
use super::tables::*;
use super::token::{CostSink, Probs, StatSink, Stats, TokenSink, WriteSink, bit_cost, code_block, new_stats};
use super::transform::{fdct, fwht, idct_add, iwht};
use rc_core::error::{Error, Result};

/// Encoder settings.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// 0-100, higher is better quality and larger.
    pub quality: f32,
    /// Effort 0-9.
    pub level: u32,
}

/// Source planes at their visible size (chroma planes are half size, rounded up).
pub struct Planes<'a> {
    pub width: usize,
    pub height: usize,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

const QFIX: u32 = 17;
const MAX_LEVEL: u64 = 2047;
/// The first partition's size field has 19 bits.
const MAX_FIRST_PARTITION: usize = (1 << 19) - 1;

/// Quantizer index (0-127) for a quality setting, following the usual mapping where file size
/// scales roughly with the cube of the quantizer step.
pub fn quality_to_q(quality: f32) -> i32 {
    let c = (quality as f64 / 100.0).clamp(0.0, 1.0);
    let linear = if c < 0.75 { c * (2.0 / 3.0) } else { 2.0 * c - 1.0 };
    ((127.0 * (1.0 - linear.powf(1.0 / 3.0))) as i32).clamp(0, 127)
}

#[derive(Clone, Copy)]
struct Matrix {
    q: [i32; 2],
    iq: [u64; 2],
    bias: [u64; 2],
}

impl Matrix {
    /// `bias` is the rounding offset in 1/256 of a step for the DC and AC coefficients.
    fn new(q: [i32; 2], bias: [u64; 2]) -> Matrix {
        Matrix {
            q,
            iq: [(1u64 << QFIX) / q[0] as u64, (1u64 << QFIX) / q[1] as u64],
            bias: [bias[0] << (QFIX - 8), bias[1] << (QFIX - 8)],
        }
    }
}

/// Quantize a block (`coefs` in raster order) from position `first` on. Levels come out in zigzag
/// order and `rec` receives the dequantized coefficients in raster order. Returns whether any level
/// is non-zero.
fn quantize_block(coefs: &[i32; 16], m: &Matrix, first: usize, levels: &mut [i16; 16], rec: &mut [i32; 16]) -> bool {
    let mut any = false;
    *levels = [0; 16];
    *rec = [0; 16];
    for n in first..16 {
        let j = ZIGZAG[n] as usize;
        let k = usize::from(n > 0);
        let c = coefs[j];
        let a = c.unsigned_abs() as u64;
        let level = ((a * m.iq[k] + m.bias[k]) >> QFIX).min(MAX_LEVEL) as i32;
        if level != 0 {
            let signed = if c < 0 { -level } else { level };
            levels[n] = signed as i16;
            rec[j] = signed * m.q[k];
            any = true;
        }
    }
    any
}

/// "Has coefficients" flags along a macroblock edge (same meaning as in the decoder).
#[derive(Clone, Copy, Default)]
struct Nz {
    luma: [bool; 4],
    u: [bool; 2],
    v: [bool; 2],
    dc: bool,
}

#[derive(Clone, Copy, Default)]
struct MbMeta {
    y_mode: u8,
    uv_mode: u8,
    bmodes: [u8; 16],
    /// Some block has a non-zero level.
    any: bool,
}

type Levels = [[i16; 16]; 25];
const Y2_BLOCK: usize = 24;

fn sse(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, size: usize) -> f32 {
    let mut s = 0u32;
    for y in 0..size {
        for x in 0..size {
            let d = a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32;
            s += (d * d) as u32;
        }
    }
    s as f32
}

/// Bits to code `value` by walking `tree` with `probs`.
fn tree_bits(tree: &[i8], probs: &[u8], value: u8) -> f32 {
    fn walk(tree: &[i8], probs: &[u8], i: usize, value: u8, acc: f32) -> Option<f32> {
        for b in 0..2 {
            let next = tree[i + b];
            let cost = acc + bit_cost(probs[i >> 1], b == 1);
            if next <= 0 {
                if (-next) as u8 == value {
                    return Some(cost);
                }
            } else if let Some(c) = walk(tree, probs, next as usize, value, cost) {
                return Some(c);
            }
        }
        None
    }
    walk(tree, probs, 0, value, 0.0).expect("value in tree")
}

struct Trial16 {
    ws: [u8; WS * 17],
    y2: [i16; 16],
    ac: [[i16; 16]; 16],
    j: f32,
    nz: (Nz, Nz),
}

struct Trial4 {
    ws: [u8; WS * 17],
    modes: [u8; 16],
    levels: [[i16; 16]; 16],
    j: f32,
    nz: (Nz, Nz),
}

struct TrialUv {
    ws_u: [u8; WS * 9],
    ws_v: [u8; WS * 9],
    levels: [[i16; 16]; 8],
    j: f32,
    nz: (Nz, Nz),
}

struct Enc {
    opts: Options,
    width: usize,
    height: usize,
    mbw: usize,
    mbh: usize,
    ys: usize,
    cs: usize,
    sy: Vec<u8>,
    su: Vec<u8>,
    sv: Vec<u8>,
    ry: Vec<u8>,
    ru: Vec<u8>,
    rv: Vec<u8>,
    base_q: i32,
    dq_uv_dc: i32,
    dq: Dequant,
    m_y1: Matrix,
    m_y2: Matrix,
    m_uv: Matrix,
    lambda: f32,
    lambda_uv: f32,
    bmode_cost: Box<[[[f32; 10]; 10]; 10]>,
    ymode_cost: [f32; 5],
    uvmode_cost: [f32; 4],
    mbs: Vec<MbMeta>,
    levels: Vec<Levels>,
}

fn pad_plane(src: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> Vec<u8> {
    let mut out = vec![0u8; pw * ph];
    for y in 0..ph {
        let sy = y.min(h - 1);
        let row = &src[sy * w..sy * w + w];
        let dst = &mut out[y * pw..(y + 1) * pw];
        dst[..w].copy_from_slice(row);
        dst[w..].fill(row[w - 1]);
    }
    out
}

impl Enc {
    fn new(src: &Planes, opts: Options) -> Enc {
        let (w, h) = (src.width, src.height);
        let (mbw, mbh) = (w.div_ceil(16), h.div_ceil(16));
        let (ys, cs) = (mbw * 16, mbw * 8);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let base_q = quality_to_q(opts.quality);
        let dq_uv_dc = -2;
        let dq = dequant_for(base_q, 0, 0, 0, dq_uv_dc, 0);
        let m_y1 = Matrix::new(dq.y1, [96, 110]);
        let m_y2 = Matrix::new(dq.y2, [96, 108]);
        let m_uv = Matrix::new(dq.uv, [110, 115]);
        let q = dq.y1[1] as f32;
        let mut bmode_cost = Box::new([[[0f32; 10]; 10]; 10]);
        for a in 0..10 {
            for l in 0..10 {
                for m in 0..10 {
                    bmode_cost[a][l][m] = tree_bits(&BMODE_TREE, &KF_BMODE_PROBS[a][l], m as u8);
                }
            }
        }
        Enc {
            opts,
            width: w,
            height: h,
            mbw,
            mbh,
            ys,
            cs,
            sy: pad_plane(src.y, w, h, ys, mbh * 16),
            su: pad_plane(src.u, cw, ch, cs, mbh * 8),
            sv: pad_plane(src.v, cw, ch, cs, mbh * 8),
            ry: vec![0; ys * mbh * 16],
            ru: vec![0; cs * mbh * 8],
            rv: vec![0; cs * mbh * 8],
            base_q,
            dq_uv_dc,
            dq,
            m_y1,
            m_y2,
            m_uv,
            lambda: 0.0234 * q * q,
            lambda_uv: 0.047 * (dq.uv[1] as f32).powi(2),
            bmode_cost,
            ymode_cost: std::array::from_fn(|m| tree_bits(&KF_YMODE_TREE, &KF_YMODE_PROBS, m as u8)),
            uvmode_cost: std::array::from_fn(|m| tree_bits(&UV_MODE_TREE, &KF_UV_MODE_PROBS, m as u8)),
            mbs: vec![MbMeta::default(); mbw * mbh],
            levels: vec![[[0; 16]; 25]; mbw * mbh],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trial_i16(
        &self,
        mode: u8,
        ws0: &[u8; WS * 17],
        src: &[u8],
        top: &Nz,
        left: &Nz,
        probs: &Probs,
        have: (bool, bool),
    ) -> Trial16 {
        let mut ws = *ws0;
        predict_block(&mut ws, ORIGIN, 16, mode, have.0, have.1);
        let ys = self.ys;
        let mut dcs = [0i32; 16];
        let mut ac = [[0i16; 16]; 16];
        let mut rec_ac = [[0i32; 16]; 16];
        for b in 0..16 {
            let (bx, by) = (b % 4, b / 4);
            let at = ORIGIN + by * 4 * WS + bx * 4;
            let coefs = fdct(&src[by * 4 * ys + bx * 4..], ys, &ws[at..], WS);
            dcs[b] = coefs[0];
            quantize_block(&coefs, &self.m_y1, 1, &mut ac[b], &mut rec_ac[b]);
        }
        let mut y2 = [0i16; 16];
        let mut y2_rec = [0i32; 16];
        quantize_block(&fwht(&dcs), &self.m_y2, 0, &mut y2, &mut y2_rec);
        let dcs_rec = iwht(&y2_rec);
        for b in 0..16 {
            let (bx, by) = (b % 4, b / 4);
            let at = ORIGIN + by * 4 * WS + bx * 4;
            let mut c = rec_ac[b];
            c[0] = dcs_rec[b];
            if c.iter().any(|&v| v != 0) {
                idct_add(&c, &mut ws[at..], WS);
            }
        }
        let d = sse(src, ys, &ws[ORIGIN..], WS, 16);
        let mut cost = CostSink { probs, bits: self.ymode_cost[mode as usize] };
        let (mut t, mut l) = (*top, *left);
        code_block(&mut cost, 1, usize::from(t.dc) + usize::from(l.dc), &y2, 0);
        let has_dc = y2.iter().any(|&v| v != 0);
        (t.dc, l.dc) = (has_dc, has_dc);
        for b in 0..16 {
            let (bx, by) = (b % 4, b / 4);
            code_block(&mut cost, 0, usize::from(t.luma[bx]) + usize::from(l.luma[by]), &ac[b], 1);
            let has = ac[b][1..].iter().any(|&v| v != 0);
            (t.luma[bx], l.luma[by]) = (has, has);
        }
        Trial16 { ws, y2, ac, j: d + self.lambda * cost.bits, nz: (t, l) }
    }

    #[allow(clippy::too_many_arguments)]
    fn trial_i4(
        &self,
        ws0: &[u8; WS * 17],
        src: &[u8],
        top: &Nz,
        left: &Nz,
        top_modes: &[u8; 4],
        left_modes: &[u8; 4],
        probs: &Probs,
    ) -> Trial4 {
        let mut ws = *ws0;
        // Sub-blocks on the right edge below the first row use the macroblock's above-right pixels.
        for by in 1..4 {
            let (a, b) = ws.split_at_mut(by * 4 * WS + 17);
            b[..4].copy_from_slice(&a[17..21]);
        }
        let ys = self.ys;
        let mut modes = [0u8; 16];
        let mut levels = [[0i16; 16]; 16];
        let (mut t, mut l) = (*top, *left);
        let mut total = self.ymode_cost[B_PRED as usize] * self.lambda;
        for blk in 0..16 {
            let (bx, by) = (blk % 4, blk / 4);
            let at = ORIGIN + by * 4 * WS + bx * 4;
            let src_blk = &src[by * 4 * ys + bx * 4..];
            let above = if by == 0 { top_modes[bx] } else { modes[blk - 4] } as usize;
            let lft = if bx == 0 { left_modes[by] } else { modes[blk - 1] } as usize;
            let ctx = usize::from(t.luma[bx]) + usize::from(l.luma[by]);
            let mut best: Option<(f32, u8, [i16; 16], [i32; 16])> = None;
            for mode in 0..10u8 {
                predict_4x4(&mut ws, at, mode);
                let coefs = fdct(src_blk, ys, &ws[at..], WS);
                let mut lv = [0i16; 16];
                let mut rec = [0i32; 16];
                quantize_block(&coefs, &self.m_y1, 0, &mut lv, &mut rec);
                let mut px = [0u8; 16];
                for r in 0..4 {
                    px[r * 4..r * 4 + 4].copy_from_slice(&ws[at + r * WS..at + r * WS + 4]);
                }
                idct_add(&rec, &mut px, 4);
                let d = sse(src_blk, ys, &px, 4, 4);
                let mut cost = CostSink { probs, bits: self.bmode_cost[above][lft][mode as usize] };
                code_block(&mut cost, 3, ctx, &lv, 0);
                let j = d + self.lambda * cost.bits;
                if best.as_ref().is_none_or(|b| j < b.0) {
                    best = Some((j, mode, lv, rec));
                }
            }
            let (j, mode, lv, rec) = best.expect("ten modes tried");
            // Commit the winner so the next sub-blocks predict from its reconstruction.
            predict_4x4(&mut ws, at, mode);
            if rec.iter().any(|&v| v != 0) {
                idct_add(&rec, &mut ws[at..], WS);
            }
            modes[blk] = mode;
            levels[blk] = lv;
            let has = lv.iter().any(|&v| v != 0);
            (t.luma[bx], l.luma[by]) = (has, has);
            total += j;
        }
        Trial4 { ws, modes, levels, j: total, nz: (t, l) }
    }

    #[allow(clippy::too_many_arguments)]
    fn trial_uv(
        &self,
        mode: u8,
        ws_u0: &[u8; WS * 9],
        ws_v0: &[u8; WS * 9],
        src_u: &[u8],
        src_v: &[u8],
        top: &Nz,
        left: &Nz,
        probs: &Probs,
        have: (bool, bool),
    ) -> TrialUv {
        let cs = self.cs;
        let (mut ws_u, mut ws_v) = (*ws_u0, *ws_v0);
        let mut levels = [[0i16; 16]; 8];
        let (mut t, mut l) = (*top, *left);
        let mut d = 0.0;
        let mut cost = CostSink { probs, bits: self.uvmode_cost[mode as usize] };
        for (plane, ws, src) in [(0usize, &mut ws_u, src_u), (1, &mut ws_v, src_v)] {
            predict_block(ws, ORIGIN, 8, mode, have.0, have.1);
            for b in 0..4 {
                let (bx, by) = (b % 2, b / 2);
                let at = ORIGIN + by * 4 * WS + bx * 4;
                let coefs = fdct(&src[by * 4 * cs + bx * 4..], cs, &ws[at..], WS);
                let mut rec = [0i32; 16];
                let lv = &mut levels[plane * 4 + b];
                quantize_block(&coefs, &self.m_uv, 0, lv, &mut rec);
                if rec.iter().any(|&v| v != 0) {
                    idct_add(&rec, &mut ws[at..], WS);
                }
                let (tf, lf) = if plane == 0 { (&mut t.u[bx], &mut l.u[by]) } else { (&mut t.v[bx], &mut l.v[by]) };
                code_block(&mut cost, 2, usize::from(*tf) + usize::from(*lf), lv, 0);
                let has = lv.iter().any(|&v| v != 0);
                (*tf, *lf) = (has, has);
            }
            d += sse(src, cs, &ws[ORIGIN..], WS, 8);
        }
        TrialUv { ws_u, ws_v, levels, j: d + self.lambda_uv * cost.bits, nz: (t, l) }
    }

    /// Decide every macroblock's modes and levels, reconstructing as the decoder will.
    fn analyze(&mut self, probs: &Probs) {
        let (mbw, mbh, ys, cs) = (self.mbw, self.mbh, self.ys, self.cs);
        let try_i4 = self.opts.level >= 3;
        let mut top_nz = vec![Nz::default(); mbw];
        let mut top_modes = vec![[B_DC_PRED; 4]; mbw];
        for mby in 0..mbh {
            let mut left_nz = Nz::default();
            let mut left_modes = [B_DC_PRED; 4];
            for mbx in 0..mbw {
                let have = (mby > 0, mbx > 0);
                let mut ws_y = [0u8; WS * 17];
                fill_borders(&mut ws_y, &self.ry, ys, mbx, mby, mbw, 16, true);
                let mut ws_u = [0u8; WS * 9];
                let mut ws_v = [0u8; WS * 9];
                fill_borders(&mut ws_u, &self.ru, cs, mbx, mby, mbw, 8, false);
                fill_borders(&mut ws_v, &self.rv, cs, mbx, mby, mbw, 8, false);
                let src_y = &self.sy[mby * 16 * ys + mbx * 16..];
                let src_u = &self.su[mby * 8 * cs + mbx * 8..];
                let src_v = &self.sv[mby * 8 * cs + mbx * 8..];

                // Luma: best whole-block mode, then (maybe) sub-block modes.
                let mut best16: Option<(u8, Trial16)> = None;
                for mode in [DC_PRED, TM_PRED, V_PRED, H_PRED] {
                    let t = self.trial_i16(mode, &ws_y, src_y, &top_nz[mbx], &left_nz, probs, have);
                    if best16.as_ref().is_none_or(|b| t.j < b.1.j) {
                        best16 = Some((mode, t));
                    }
                }
                let (mode16, t16) = best16.expect("four modes tried");
                let t4 = try_i4
                    .then(|| self.trial_i4(&ws_y, src_y, &top_nz[mbx], &left_nz, &top_modes[mbx], &left_modes, probs));

                let idx = mby * mbw + mbx;
                let mut meta = MbMeta::default();
                let mut lv = [[0i16; 16]; 25];
                let (nz_t, nz_l);
                match t4 {
                    Some(t4) if t4.j < t16.j => {
                        meta.y_mode = B_PRED;
                        meta.bmodes = t4.modes;
                        lv[..16].copy_from_slice(&t4.levels);
                        ws_y = t4.ws;
                        (nz_t, nz_l) = t4.nz;
                        // Sub-block modes become the neighbours' contexts.
                        top_modes[mbx] = [t4.modes[12], t4.modes[13], t4.modes[14], t4.modes[15]];
                        left_modes = [t4.modes[3], t4.modes[7], t4.modes[11], t4.modes[15]];
                    }
                    _ => {
                        meta.y_mode = mode16;
                        lv[..16].copy_from_slice(&t16.ac);
                        lv[Y2_BLOCK] = t16.y2;
                        ws_y = t16.ws;
                        (nz_t, nz_l) = t16.nz;
                        let implied = match mode16 {
                            DC_PRED => B_DC_PRED,
                            V_PRED => B_VE_PRED,
                            H_PRED => B_HE_PRED,
                            _ => B_TM_PRED,
                        };
                        top_modes[mbx] = [implied; 4];
                        left_modes = [implied; 4];
                    }
                }
                top_nz[mbx].luma = nz_t.luma;
                left_nz.luma = nz_l.luma;
                if meta.y_mode != B_PRED {
                    top_nz[mbx].dc = nz_t.dc;
                    left_nz.dc = nz_l.dc;
                }
                store_block(&ws_y, &mut self.ry, ys, mbx, mby, 16);

                // Chroma.
                let mut best_uv: Option<(u8, TrialUv)> = None;
                for mode in [DC_PRED, TM_PRED, V_PRED, H_PRED] {
                    let t = self.trial_uv(mode, &ws_u, &ws_v, src_u, src_v, &top_nz[mbx], &left_nz, probs, have);
                    if best_uv.as_ref().is_none_or(|b| t.j < b.1.j) {
                        best_uv = Some((mode, t));
                    }
                }
                let (uv_mode, tuv) = best_uv.expect("four modes tried");
                meta.uv_mode = uv_mode;
                lv[16..24].copy_from_slice(&tuv.levels);
                (top_nz[mbx].u, left_nz.u) = (tuv.nz.0.u, tuv.nz.1.u);
                (top_nz[mbx].v, left_nz.v) = (tuv.nz.0.v, tuv.nz.1.v);
                store_block(&tuv.ws_u, &mut self.ru, cs, mbx, mby, 8);
                store_block(&tuv.ws_v, &mut self.rv, cs, mbx, mby, 8);

                meta.any = lv.iter().any(|b| b.iter().any(|&v| v != 0));
                self.mbs[idx] = meta;
                self.levels[idx] = lv;
            }
        }
    }

    /// Walk every token the frame will contain, in bitstream order, with the contexts the decoder
    /// will see. `skip` says whether macroblocks without coefficients are coded with the skip flag.
    fn walk_tokens<S: TokenSink>(&self, sink: &mut S, skip: bool) {
        let mut top = vec![Nz::default(); self.mbw];
        for mby in 0..self.mbh {
            let mut left = Nz::default();
            for mbx in 0..self.mbw {
                let idx = mby * self.mbw + mbx;
                let (meta, lv) = (&self.mbs[idx], &self.levels[idx]);
                let is_i4 = meta.y_mode == B_PRED;
                let t = &mut top[mbx];
                if skip && !meta.any {
                    let dc = if is_i4 { (t.dc, left.dc) } else { (false, false) };
                    *t = Nz { dc: dc.0, ..Default::default() };
                    left = Nz { dc: dc.1, ..Default::default() };
                    continue;
                }
                let first = if is_i4 { 0 } else { 1 };
                if !is_i4 {
                    let ctx = usize::from(t.dc) + usize::from(left.dc);
                    code_block(sink, 1, ctx, &lv[Y2_BLOCK], 0);
                    let has = lv[Y2_BLOCK].iter().any(|&v| v != 0);
                    (t.dc, left.dc) = (has, has);
                }
                for b in 0..16 {
                    let (bx, by) = (b % 4, b / 4);
                    let ctx = usize::from(t.luma[bx]) + usize::from(left.luma[by]);
                    code_block(sink, if is_i4 { 3 } else { 0 }, ctx, &lv[b], first);
                    let has = lv[b][first..].iter().any(|&v| v != 0);
                    (t.luma[bx], left.luma[by]) = (has, has);
                }
                for (plane, base) in [(0usize, 16usize), (1, 20)] {
                    for b in 0..4 {
                        let (bx, by) = (b % 2, b / 2);
                        let (tf, lf) =
                            if plane == 0 { (&mut t.u[bx], &mut left.u[by]) } else { (&mut t.v[bx], &mut left.v[by]) };
                        code_block(sink, 2, usize::from(*tf) + usize::from(*lf), &lv[base + b], 0);
                        let has = lv[base + b].iter().any(|&v| v != 0);
                        (*tf, *lf) = (has, has);
                    }
                }
            }
        }
    }

    /// Filter parameters the decoder will derive for a frame-level `level` (no segments or deltas).
    fn filter_info(level: i32, is_i4: bool, any: bool) -> FilterInfo {
        if level <= 0 {
            return FilterInfo::default();
        }
        FilterInfo {
            limit: (3 * level) as u8,
            ilevel: level as u8,
            hev_thresh: if level >= 40 {
                2
            } else if level >= 15 {
                1
            } else {
                0
            },
            inner: is_i4 || any,
        }
    }

    /// Reconstruction after the loop filter at `level`.
    fn filtered(&self, level: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (mut y, mut u, mut v) = (self.ry.clone(), self.ru.clone(), self.rv.clone());
        if level > 0 {
            for mby in 0..self.mbh {
                for mbx in 0..self.mbw {
                    let m = &self.mbs[mby * self.mbw + mbx];
                    let info = Self::filter_info(level, m.y_mode == B_PRED, m.any);
                    filter_macroblock(
                        &info,
                        false,
                        mbx,
                        mby,
                        &mut Plane { data: &mut y, stride: self.ys },
                        &mut Plane { data: &mut u, stride: self.cs },
                        &mut Plane { data: &mut v, stride: self.cs },
                    );
                }
            }
        }
        (y, u, v)
    }

    /// Squared error of the visible part of a reconstruction against the source.
    fn error(&self, y: &[u8], u: &[u8], v: &[u8]) -> f64 {
        let (cw, ch) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let plane = |a: &[u8], b: &[u8], stride: usize, w: usize, h: usize| -> f64 {
            let mut s = 0u64;
            for r in 0..h {
                for c in 0..w {
                    let d = a[r * stride + c] as i64 - b[r * stride + c] as i64;
                    s += (d * d) as u64;
                }
            }
            s as f64
        };
        plane(y, &self.sy, self.ys, self.width, self.height)
            + plane(u, &self.su, self.cs, cw, ch)
            + plane(v, &self.sv, self.cs, cw, ch)
    }

    fn choose_filter_level(&self) -> i32 {
        if self.opts.level < 2 {
            // Cheap heuristic: stronger smoothing for coarser quantizers.
            return (self.dq.y1[1] / 4).min(40);
        }
        let mut best = (f64::MAX, 0);
        for level in (0..=48).step_by(4) {
            let (y, u, v) = self.filtered(level);
            let e = self.error(&y, &u, &v);
            if e < best.0 {
                best = (e, level);
            }
        }
        // Refine around the winner.
        let center = best.1;
        for level in [center - 2, center + 2] {
            if (0..=63).contains(&level) {
                let (y, u, v) = self.filtered(level);
                let e = self.error(&y, &u, &v);
                if e < best.0 {
                    best = (e, level);
                }
            }
        }
        best.1
    }
}

/// Probabilities to transmit: for each context, the new value if updating it saves bits.
type Updates = [[[[bool; 11]; 3]; 8]; 4];

fn tune_probs(stats: &Stats) -> (Box<Probs>, Box<Updates>) {
    let mut probs = Box::new(DEFAULT_COEFF_PROBS);
    let mut update = Box::new([[[[false; 11]; 3]; 8]; 4]);
    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for p in 0..11 {
                    let (ones, total) = stats[t][b][c][p];
                    if total == 0 {
                        continue;
                    }
                    let old = DEFAULT_COEFF_PROBS[t][b][c][p];
                    let new =
                        if ones == 0 { 255 } else { (255 - (ones as u64 * 255 / total as u64)).clamp(1, 255) as u8 };
                    let branch = |prob: u8| -> f32 {
                        ones as f32 * bit_cost(prob, true) + (total - ones) as f32 * bit_cost(prob, false)
                    };
                    let upd_prob = COEFF_UPDATE_PROBS[t][b][c][p];
                    let old_cost = branch(old) + bit_cost(upd_prob, false);
                    let new_cost = branch(new) + bit_cost(upd_prob, true) + 8.0;
                    if old_cost > new_cost {
                        probs[t][b][c][p] = new;
                        update[t][b][c][p] = true;
                    }
                }
            }
        }
    }
    (probs, update)
}

/// A finished frame: the VP8 bitstream and the decoder-side reconstruction (after loop filtering).
pub struct Encoded {
    pub data: Vec<u8>,
    pub recon: Frame,
}

/// Encode planar 4:2:0 `src` as a VP8 key frame.
pub fn encode_frame(src: &Planes, opts: Options) -> Result<Encoded> {
    if src.width == 0 || src.height == 0 || src.width > 16383 || src.height > 16383 {
        return Err(Error::unsupported("VP8 frames are limited to 16383x16383 pixels"));
    }
    let mut enc = Enc::new(src, opts);
    let mut probs = Box::new(DEFAULT_COEFF_PROBS);
    enc.analyze(&probs);
    let two_pass = opts.level >= 5;
    if two_pass {
        // Decide again with probabilities fitted to what the first pass produced.
        let mut stats = new_stats();
        enc.walk_tokens(&mut StatSink { stats: &mut stats }, true);
        probs = tune_probs(&stats).0;
        enc.analyze(&probs);
    }
    let total = enc.mbs.len();
    let skipped = enc.mbs.iter().filter(|m| !m.any).count();
    let skip_prob = ((total - skipped) * 255).checked_div(total).map_or(255, |p| p.clamp(1, 255) as u8);
    let use_skip = skip_prob < 250;
    let mut stats = new_stats();
    enc.walk_tokens(&mut StatSink { stats: &mut stats }, use_skip);
    let (final_probs, update) = tune_probs(&stats);
    let filter_level = enc.choose_filter_level();

    // First partition: header and per-macroblock modes.
    let mut p1 = BoolEncoder::new();
    p1.flag(false); // colour space
    p1.flag(false); // clamping type
    p1.flag(false); // no segmentation
    p1.flag(false); // normal loop filter
    p1.literal(filter_level as u32, 6);
    p1.literal(0, 3); // sharpness
    p1.flag(false); // no loop-filter deltas
    p1.literal(0, 2); // one token partition
    p1.literal(enc.base_q as u32, 7);
    p1.flag(false); // y1 dc delta
    p1.flag(false); // y2 dc delta
    p1.flag(false); // y2 ac delta
    p1.flag(enc.dq_uv_dc != 0);
    if enc.dq_uv_dc != 0 {
        p1.signed(enc.dq_uv_dc, 4);
    }
    p1.flag(false); // uv ac delta
    p1.flag(false); // do not refresh probabilities for later frames
    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for p in 0..11 {
                    let u = update[t][b][c][p];
                    p1.put(COEFF_UPDATE_PROBS[t][b][c][p], u);
                    if u {
                        p1.literal(final_probs[t][b][c][p] as u32, 8);
                    }
                }
            }
        }
    }
    p1.flag(use_skip);
    if use_skip {
        p1.literal(skip_prob as u32, 8);
    }
    let mut top_modes = vec![[B_DC_PRED; 4]; enc.mbw];
    for mby in 0..enc.mbh {
        let mut left = [B_DC_PRED; 4];
        for mbx in 0..enc.mbw {
            let m = &enc.mbs[mby * enc.mbw + mbx];
            if use_skip {
                p1.put(skip_prob, !m.any);
            }
            p1.tree(&KF_YMODE_TREE, &KF_YMODE_PROBS, m.y_mode);
            if m.y_mode == B_PRED {
                for by in 0..4 {
                    for bx in 0..4 {
                        let mode = m.bmodes[by * 4 + bx];
                        let probs = &KF_BMODE_PROBS[top_modes[mbx][bx] as usize][left[by] as usize];
                        p1.tree(&BMODE_TREE, probs, mode);
                        top_modes[mbx][bx] = mode;
                        left[by] = mode;
                    }
                }
            } else {
                let implied = match m.y_mode {
                    DC_PRED => B_DC_PRED,
                    V_PRED => B_VE_PRED,
                    H_PRED => B_HE_PRED,
                    _ => B_TM_PRED,
                };
                top_modes[mbx] = [implied; 4];
                left = [implied; 4];
            }
            p1.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, m.uv_mode);
        }
    }
    let first = p1.finish();
    if first.len() > MAX_FIRST_PARTITION {
        return Err(Error::unsupported("image too large for a single VP8 mode partition"));
    }

    // Token partition.
    let mut tokens = BoolEncoder::new();
    enc.walk_tokens(&mut WriteSink { probs: &final_probs, enc: &mut tokens }, use_skip);
    let tokens = tokens.finish();

    let mut data = Vec::with_capacity(10 + first.len() + tokens.len());
    let tag = (1u32 << 4) | ((first.len() as u32) << 5); // key frame, version 0, shown
    data.extend_from_slice(&tag.to_le_bytes()[..3]);
    data.extend_from_slice(&[0x9d, 0x01, 0x2a]);
    data.extend_from_slice(&(enc.width as u16).to_le_bytes());
    data.extend_from_slice(&(enc.height as u16).to_le_bytes());
    data.extend_from_slice(&first);
    data.extend_from_slice(&tokens);

    let (y, u, v) = enc.filtered(filter_level);
    let recon = Frame { width: enc.width, height: enc.height, y, u, v, y_stride: enc.ys, uv_stride: enc.cs };
    Ok(Encoded { data, recon })
}
