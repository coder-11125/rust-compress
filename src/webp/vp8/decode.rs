//! VP8 key-frame decoder: frame header, macroblock modes, coefficient tokens, intra prediction,
//! reconstruction and the loop filter, producing planar 4:2:0 YUV.

use super::boolcoder::BoolDecoder;
use super::loopfilter::{FilterInfo, Plane, filter_macroblock};
use super::predict::{self, ORIGIN, WS, fill_borders, store_block};
use super::tables::*;
use super::transform::{idct_add, iwht};
use crate::error::{Error, Result};
use crate::image::MAX_PIXELS;

/// A decoded frame: padded planes (whole macroblocks) plus the visible size.
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
}

pub const KF_YMODE_TREE: [i8; 8] =
    [-(B_PRED as i8), 2, 4, 6, -(DC_PRED as i8), -(V_PRED as i8), -(H_PRED as i8), -(TM_PRED as i8)];
pub const UV_MODE_TREE: [i8; 6] = [-(DC_PRED as i8), 2, -(V_PRED as i8), 4, -(H_PRED as i8), -(TM_PRED as i8)];
pub const SEGMENT_TREE: [i8; 6] = [2, 4, 0, -1, -2, -3];
pub const BMODE_TREE: [i8; 18] = [
    -(B_DC_PRED as i8),
    2,
    -(B_TM_PRED as i8),
    4,
    -(B_VE_PRED as i8),
    6,
    8,
    12,
    -(B_HE_PRED as i8),
    10,
    -(B_RD_PRED as i8),
    -(B_VR_PRED as i8),
    -(B_LD_PRED as i8),
    14,
    -(B_VL_PRED as i8),
    16,
    -(B_HD_PRED as i8),
    -(B_HU_PRED as i8),
];

/// Band of each coefficient position, with a dummy entry for position 16.
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

#[derive(Default, Clone)]
struct SegmentHeader {
    enabled: bool,
    update_map: bool,
    absolute: bool,
    quant: [i32; 4],
    filter: [i32; 4],
    tree_probs: [u8; 3],
}

#[derive(Default, Clone)]
struct FilterHeader {
    simple: bool,
    level: i32,
    sharpness: i32,
    use_delta: bool,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],
}

/// Dequantization factors `[dc, ac]` for the three kinds of block.
#[derive(Default, Clone, Copy, Debug)]
pub struct Dequant {
    pub y1: [i32; 2],
    pub y2: [i32; 2],
    pub uv: [i32; 2],
}

/// The factors for quantizer index `q` and the per-plane index deltas of the frame header.
pub fn dequant_for(q: i32, dy1_dc: i32, dy2_dc: i32, dy2_ac: i32, duv_dc: i32, duv_ac: i32) -> Dequant {
    let clip = |v: i32, max: i32| v.clamp(0, max) as usize;
    let mut d = Dequant {
        y1: [DC_QUANT[clip(q + dy1_dc, 127)], AC_QUANT[clip(q, 127)]],
        y2: [DC_QUANT[clip(q + dy2_dc, 127)] * 2, (AC_QUANT[clip(q + dy2_ac, 127)] * 101581) >> 16],
        uv: [DC_QUANT[clip(q + duv_dc, 117)], AC_QUANT[clip(q + duv_ac, 127)]],
    };
    d.y2[1] = d.y2[1].max(8);
    d
}

#[derive(Clone, Copy, Default)]
struct MbInfo {
    y_mode: u8,
    uv_mode: u8,
    segment: u8,
    skip: bool,
    bmodes: [u8; 16],
}

struct Header {
    width: usize,
    height: usize,
    segment: SegmentHeader,
    filter: FilterHeader,
    partitions: usize,
    dequant: [Dequant; 4],
    probs: Box<TokenProbs>,
    skip_prob: Option<u8>,
}

fn read_optional_delta(br: &mut BoolDecoder, bits: u32) -> i32 {
    if br.flag() { br.signed(bits) } else { 0 }
}

fn parse_segments(br: &mut BoolDecoder) -> SegmentHeader {
    let mut s = SegmentHeader { enabled: true, tree_probs: [255; 3], ..Default::default() };
    s.update_map = br.flag();
    let update_data = br.flag();
    if update_data {
        s.absolute = br.flag();
        for q in &mut s.quant {
            *q = read_optional_delta(br, 7);
        }
        for f in &mut s.filter {
            *f = read_optional_delta(br, 6);
        }
    }
    if s.update_map {
        for p in &mut s.tree_probs {
            *p = if br.flag() { br.literal(8) as u8 } else { 255 };
        }
    }
    s
}

fn parse_filter(br: &mut BoolDecoder) -> FilterHeader {
    let mut f = FilterHeader {
        simple: br.flag(),
        level: br.literal(6) as i32,
        sharpness: br.literal(3) as i32,
        ..Default::default()
    };
    f.use_delta = br.flag();
    if f.use_delta && br.flag() {
        for d in &mut f.ref_delta {
            *d = read_optional_delta(br, 6);
        }
        for d in &mut f.mode_delta {
            *d = read_optional_delta(br, 6);
        }
    }
    f
}

fn parse_dequant(br: &mut BoolDecoder, segment: &SegmentHeader) -> [Dequant; 4] {
    let base = br.literal(7) as i32;
    let dy1_dc = read_optional_delta(br, 4);
    let dy2_dc = read_optional_delta(br, 4);
    let dy2_ac = read_optional_delta(br, 4);
    let duv_dc = read_optional_delta(br, 4);
    let duv_ac = read_optional_delta(br, 4);
    std::array::from_fn(|s| {
        let q = if segment.enabled {
            if segment.absolute { segment.quant[s] } else { base + segment.quant[s] }
        } else {
            base
        };
        dequant_for(q, dy1_dc, dy2_dc, dy2_ac, duv_dc, duv_ac)
    })
}

fn parse_header(data: &[u8]) -> Result<(Header, BoolDecoder<'_>, usize)> {
    if data.len() < 10 {
        return Err(Error::invalid("VP8 frame too short"));
    }
    let tag = u32::from_le_bytes([data[0], data[1], data[2], 0]);
    if tag & 1 != 0 {
        return Err(Error::unsupported("VP8 inter frame"));
    }
    let first_size = (tag >> 5) as usize;
    if data[3..6] != [0x9d, 0x01, 0x2a] {
        return Err(Error::invalid("bad VP8 start code"));
    }
    let width = (u16::from_le_bytes([data[6], data[7]]) & 0x3FFF) as usize;
    let height = (u16::from_le_bytes([data[8], data[9]]) & 0x3FFF) as usize;
    if width == 0 || height == 0 || (width * height) as u64 > MAX_PIXELS {
        return Err(Error::invalid("bad VP8 dimensions"));
    }
    let body = &data[10..];
    if first_size > body.len() {
        return Err(Error::invalid("VP8 first partition exceeds the data"));
    }
    let mut br = BoolDecoder::new(&body[..first_size]);
    let _color_space = br.flag();
    let _clamp_type = br.flag();
    let segment = if br.flag() { parse_segments(&mut br) } else { SegmentHeader::default() };
    let filter = parse_filter(&mut br);
    let partitions = 1usize << br.literal(2);
    let dequant = parse_dequant(&mut br, &segment);
    let _refresh_entropy = br.flag();
    let mut probs = Box::new(DEFAULT_COEFF_PROBS);
    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for p in 0..11 {
                    if br.get(COEFF_UPDATE_PROBS[t][b][c][p]) {
                        probs[t][b][c][p] = br.literal(8) as u8;
                    }
                }
            }
        }
    }
    let skip_prob = if br.flag() { Some(br.literal(8) as u8) } else { None };
    if br.overran() {
        return Err(Error::invalid("truncated VP8 header"));
    }
    Ok((Header { width, height, segment, filter, partitions, dequant, probs, skip_prob }, br, 10 + first_size))
}

/// Parse every macroblock's prediction modes from the first partition.
fn parse_modes(h: &Header, br: &mut BoolDecoder, mbw: usize, mbh: usize) -> Result<Vec<MbInfo>> {
    let mut mbs = vec![MbInfo::default(); mbw * mbh];
    let mut top = vec![[B_DC_PRED; 4]; mbw];
    for mby in 0..mbh {
        let mut left = [B_DC_PRED; 4];
        for mbx in 0..mbw {
            let mut mb = MbInfo::default();
            if h.segment.enabled && h.segment.update_map {
                mb.segment = br.tree(&SEGMENT_TREE, &h.segment.tree_probs);
            }
            mb.skip = h.skip_prob.is_some_and(|p| br.get(p));
            mb.y_mode = br.tree(&KF_YMODE_TREE, &KF_YMODE_PROBS);
            if mb.y_mode == B_PRED {
                for by in 0..4 {
                    for bx in 0..4 {
                        let probs = &KF_BMODE_PROBS[top[mbx][bx] as usize][left[by] as usize];
                        let m = br.tree(&BMODE_TREE, probs);
                        mb.bmodes[by * 4 + bx] = m;
                        top[mbx][bx] = m;
                        left[by] = m;
                    }
                }
            } else {
                // Whole-macroblock modes imply a matching sub-block mode for the neighbours' contexts.
                let implied = match mb.y_mode {
                    DC_PRED => B_DC_PRED,
                    V_PRED => B_VE_PRED,
                    H_PRED => B_HE_PRED,
                    _ => B_TM_PRED,
                };
                top[mbx] = [implied; 4];
                left = [implied; 4];
            }
            mb.uv_mode = br.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS);
            mbs[mby * mbw + mbx] = mb;
        }
        if br.overran() {
            return Err(Error::invalid("truncated VP8 mode data"));
        }
    }
    Ok(mbs)
}

fn large_value(br: &mut BoolDecoder, p: &[u8; 11]) -> i32 {
    if !br.get(p[3]) {
        if !br.get(p[4]) { 2 } else { 3 + br.get(p[5]) as i32 }
    } else if !br.get(p[6]) {
        if !br.get(p[7]) {
            5 + br.get(159) as i32
        } else {
            let hi = br.get(165) as i32;
            7 + 2 * hi + br.get(145) as i32
        }
    } else {
        let bit1 = br.get(p[8]) as usize;
        let bit0 = br.get(p[9 + bit1]) as usize;
        let cat = 2 * bit1 + bit0;
        let probs = &PROB_DCT_CAT[2 + cat];
        let mut v = 0i32;
        for &pr in probs.iter().take_while(|&&pr| pr != 0) {
            v = 2 * v + br.get(pr) as i32;
        }
        v + 3 + (8 << cat)
    }
}

/// Decode one block's coefficients (dequantized, raster order) into `out`. Returns the position
/// after the last non-zero coefficient (so `> first` means the block has any).
fn get_coeffs(
    br: &mut BoolDecoder,
    probs: &[[[u8; 11]; 3]; 8],
    ctx: usize,
    dq: &[i32; 2],
    first: usize,
    out: &mut [i32],
) -> usize {
    let mut n = first;
    let mut p = &probs[BANDS[n]][ctx];
    while n < 16 {
        if !br.get(p[0]) {
            return n;
        }
        while !br.get(p[1]) {
            n += 1;
            p = &probs[BANDS[n]][0];
            if n == 16 {
                return 16;
            }
        }
        let next = &probs[BANDS[n + 1]];
        let v;
        if !br.get(p[2]) {
            v = 1;
            p = &next[1];
        } else {
            v = large_value(br, p);
            p = &next[2];
        }
        let signed = if br.get(128) { -v } else { v };
        // Coefficients are 16-bit quantities; damaged streams wrap exactly as libwebp's do.
        out[ZIGZAG[n] as usize] = (signed * dq[usize::from(n > 0)]) as i16 as i32;
        n += 1;
    }
    16
}

/// "Has coefficients" flags of the blocks along one macroblock edge.
#[derive(Clone, Copy, Default)]
struct Nz {
    luma: [bool; 4],
    u: [bool; 2],
    v: [bool; 2],
    dc: bool,
}

/// Decode the residuals of one macroblock into `coefs` (25 blocks of 16; block 24 unused after the
/// Walsh transform). Returns whether any block had coefficients.
#[allow(clippy::too_many_arguments)]
fn parse_residuals(
    br: &mut BoolDecoder,
    h: &Header,
    dq: &Dequant,
    is_i4x4: bool,
    top: &mut Nz,
    left: &mut Nz,
    coefs: &mut [i32; 400],
) -> bool {
    let probs = &h.probs;
    let mut any = false;
    let (first, luma_type) = if is_i4x4 {
        (0, 3)
    } else {
        let mut y2 = [0i32; 16];
        let ctx = usize::from(top.dc) + usize::from(left.dc);
        let nz = get_coeffs(br, &probs[1], ctx, &dq.y2, 0, &mut y2);
        top.dc = nz > 0;
        left.dc = nz > 0;
        any |= nz > 0;
        let dcs = iwht(&y2);
        for (i, dc) in dcs.iter().enumerate() {
            coefs[i * 16] = *dc as i16 as i32;
        }
        (1, 0)
    };
    for by in 0..4 {
        for bx in 0..4 {
            let ctx = usize::from(top.luma[bx]) + usize::from(left.luma[by]);
            let blk = &mut coefs[(by * 4 + bx) * 16..(by * 4 + bx) * 16 + 16];
            let nz = get_coeffs(br, &probs[luma_type], ctx, &dq.y1, first, blk);
            let has = nz > first;
            top.luma[bx] = has;
            left.luma[by] = has;
            any |= has;
        }
    }
    for (plane, base) in [(0usize, 16usize), (1, 20)] {
        for by in 0..2 {
            for bx in 0..2 {
                let (t, l) =
                    if plane == 0 { (&mut top.u[bx], &mut left.u[by]) } else { (&mut top.v[bx], &mut left.v[by]) };
                let ctx = usize::from(*t) + usize::from(*l);
                let blk = &mut coefs[(base + by * 2 + bx) * 16..(base + by * 2 + bx) * 16 + 16];
                let nz = get_coeffs(br, &probs[2], ctx, &dq.uv, 0, blk);
                *t = nz > 0;
                *l = nz > 0;
                any |= nz > 0;
            }
        }
    }
    any
}

fn add_blocks(ws: &mut [u8], coefs: &[i32; 400], first_block: usize, blocks_per_row: usize, count: usize) {
    for i in 0..count {
        let blk: &[i32; 16] = coefs[(first_block + i) * 16..(first_block + i) * 16 + 16].try_into().unwrap();
        if blk.iter().any(|&c| c != 0) {
            let at = ORIGIN + (i / blocks_per_row) * 4 * WS + (i % blocks_per_row) * 4;
            idct_add(blk, &mut ws[at..], WS);
        }
    }
}

/// Per-segment, per-mode loop filter strengths.
fn filter_strengths(h: &Header) -> [[FilterInfo; 2]; 4] {
    let mut out = [[FilterInfo::default(); 2]; 4];
    for (s, row) in out.iter_mut().enumerate() {
        let mut base = h.filter.level;
        if h.segment.enabled {
            base = h.segment.filter[s] + if h.segment.absolute { 0 } else { h.filter.level };
        }
        for (i4x4, info) in row.iter_mut().enumerate() {
            let mut level = base;
            if h.filter.use_delta {
                level += h.filter.ref_delta[0];
                if i4x4 == 1 {
                    level += h.filter.mode_delta[0];
                }
            }
            let level = level.clamp(0, 63);
            if level > 0 {
                let mut ilevel = level;
                if h.filter.sharpness > 0 {
                    ilevel >>= if h.filter.sharpness > 4 { 2 } else { 1 };
                    ilevel = ilevel.min(9 - h.filter.sharpness);
                }
                let ilevel = ilevel.max(1);
                *info = FilterInfo {
                    limit: (2 * level + ilevel) as u8,
                    ilevel: ilevel as u8,
                    hev_thresh: if level >= 40 {
                        2
                    } else if level >= 15 {
                        1
                    } else {
                        0
                    },
                    inner: i4x4 == 1,
                };
            }
        }
    }
    out
}

/// Decode a VP8 key frame (the payload of a `VP8 ` chunk).
pub fn decode_frame(data: &[u8]) -> Result<Frame> {
    let (h, mut br, pos) = parse_header(data)?;
    let (mbw, mbh) = (h.width.div_ceil(16), h.height.div_ceil(16));
    let mbs = parse_modes(&h, &mut br, mbw, mbh)?;

    // Token partitions: a table of 3-byte sizes for all but the last, then the data.
    let table = 3 * (h.partitions - 1);
    if pos + table > data.len() {
        return Err(Error::invalid("truncated VP8 partition table"));
    }
    let mut starts = Vec::with_capacity(h.partitions);
    let mut at = pos + table;
    for i in 0..h.partitions {
        let size = if i + 1 < h.partitions {
            let s = &data[pos + 3 * i..pos + 3 * i + 3];
            s[0] as usize | (s[1] as usize) << 8 | (s[2] as usize) << 16
        } else {
            data.len().saturating_sub(at)
        };
        let end = (at + size).min(data.len());
        starts.push(&data[at.min(end)..end]);
        at = end;
    }
    let mut readers: Vec<BoolDecoder> = starts.iter().map(|s| BoolDecoder::new(s)).collect();

    let (y_stride, uv_stride) = (mbw * 16, mbw * 8);
    let mut frame = Frame {
        width: h.width,
        height: h.height,
        y: vec![0; y_stride * mbh * 16],
        u: vec![0; uv_stride * mbh * 8],
        v: vec![0; uv_stride * mbh * 8],
        y_stride,
        uv_stride,
    };
    let strengths = filter_strengths(&h);
    let filtering = h.filter.level != 0;
    let mut finfo = vec![FilterInfo::default(); if filtering { mbw * mbh } else { 0 }];

    let mut top_nz = vec![Nz::default(); mbw];
    let mut coefs = [0i32; 400];
    let mut ws_y = [0u8; WS * 17];
    let mut ws_u = [0u8; WS * 9];
    let mut ws_v = [0u8; WS * 9];
    for mby in 0..mbh {
        let br_row = &mut readers[mby % h.partitions];
        let mut left_nz = Nz::default();
        for mbx in 0..mbw {
            let mb = &mbs[mby * mbw + mbx];
            let is_i4x4 = mb.y_mode == B_PRED;
            coefs.fill(0);
            let mut any = false;
            if h.skip_prob.is_some() && mb.skip {
                // Skipped macroblocks reset the neighbour flags; the Y2 flag survives for B_PRED.
                let dc = if is_i4x4 { (top_nz[mbx].dc, left_nz.dc) } else { (false, false) };
                top_nz[mbx] = Nz { dc: dc.0, ..Default::default() };
                left_nz = Nz { dc: dc.1, ..Default::default() };
            } else {
                let dq = &h.dequant[mb.segment as usize];
                any = parse_residuals(br_row, &h, dq, is_i4x4, &mut top_nz[mbx], &mut left_nz, &mut coefs);
            }

            // Luma.
            fill_borders(&mut ws_y, &frame.y, y_stride, mbx, mby, mbw, 16, true);
            if is_i4x4 {
                for by in 1..4 {
                    let (a, b) = ws_y.split_at_mut(by * 4 * WS + 17);
                    b[..4].copy_from_slice(&a[17..21]);
                }
                for blk in 0..16 {
                    let at = ORIGIN + (blk / 4) * 4 * WS + (blk % 4) * 4;
                    predict::predict_4x4(&mut ws_y, at, mb.bmodes[blk]);
                    let c: &[i32; 16] = coefs[blk * 16..blk * 16 + 16].try_into().unwrap();
                    if c.iter().any(|&v| v != 0) {
                        idct_add(c, &mut ws_y[at..], WS);
                    }
                }
            } else {
                predict::predict_block(&mut ws_y, ORIGIN, 16, mb.y_mode, mby > 0, mbx > 0);
                add_blocks(&mut ws_y, &coefs, 0, 4, 16);
            }
            store_block(&ws_y, &mut frame.y, y_stride, mbx, mby, 16);

            // Chroma.
            for (ws, plane, first) in [(&mut ws_u, &mut frame.u, 16usize), (&mut ws_v, &mut frame.v, 20)] {
                fill_borders(ws, plane, uv_stride, mbx, mby, mbw, 8, false);
                predict::predict_block(ws, ORIGIN, 8, mb.uv_mode, mby > 0, mbx > 0);
                add_blocks(ws, &coefs, first, 2, 4);
                store_block(ws, plane, uv_stride, mbx, mby, 8);
            }

            if filtering {
                let mut info = strengths[mb.segment as usize][usize::from(is_i4x4)];
                info.inner |= any;
                finfo[mby * mbw + mbx] = info;
            }
        }
        if br_row.overran() {
            return Err(Error::invalid("truncated VP8 coefficient data"));
        }
    }

    if filtering {
        for mby in 0..mbh {
            for mbx in 0..mbw {
                filter_macroblock(
                    &finfo[mby * mbw + mbx],
                    h.filter.simple,
                    mbx,
                    mby,
                    &mut Plane { data: &mut frame.y, stride: y_stride },
                    &mut Plane { data: &mut frame.u, stride: uv_stride },
                    &mut Plane { data: &mut frame.v, stride: uv_stride },
                );
            }
        }
    }
    Ok(frame)
}
