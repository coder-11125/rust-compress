//! AV1 still-picture decoder: OBU stream in, picture out.

use super::bits::BitReader;
use super::cdef::{CdefParams, cdef_frame};
use super::cdf_tables::Cdfs;
use super::deblock::{DeblockMi, DeblockParams, deblock_frame};
use super::entropy::SymbolDecoder;
use super::filmgrain::{FilmGrainParams, apply_film_grain};
use super::lr::{LrPlane, LrType, LrUnit, loop_restoration_frame};
use super::superres::upscale_frame;
use super::frame::{Frame, Plane};
use super::obu::{self, FrameHeader, SequenceHeader};
use super::tile::{FrameState, TileCoder};
use crate::error::{Error, Result};

/// A decoded picture and the headers it came with.
pub struct Decoded {
    pub frame: Frame,
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
}

/// Decode the first shown (intra) frame of a low-overhead OBU stream.
pub fn decode(data: &[u8]) -> Result<Decoded> {
    let obus = obu::split_obus(data)?;
    let mut seq: Option<SequenceHeader> = None;
    let mut hdr: Option<FrameHeader> = None;
    let mut state: Option<FrameState> = None;
    let mut tiles_done = 0usize;
    for o in &obus {
        match o.kind {
            obu::OBU_SEQUENCE_HEADER => {
                seq = Some(SequenceHeader::parse(o.payload)?);
            }
            obu::OBU_FRAME_HEADER | obu::OBU_FRAME => {
                if state.is_some() {
                    // A redundant copy of the header, or a second frame: only the first frame is decoded.
                    if o.kind == obu::OBU_FRAME_HEADER {
                        continue;
                    }
                    break;
                }
                let s = seq.as_ref().ok_or_else(|| Error::invalid("AV1 frame before sequence header"))?;
                let mut r = BitReader::new(o.payload);
                let h = FrameHeader::parse(s, &mut r)?;
                if std::env::var_os("AV1_DEBUG").is_some() {
                    eprintln!("{s:?}\n{h:?}");
                }
                check_limits(s, &h)?;
                let fs = FrameState::new(s, &h)?;
                hdr = Some(h);
                state = Some(fs);
                if o.kind == obu::OBU_FRAME {
                    r.byte_align();
                    let off = r.byte_pos();
                    decode_tile_group(
                        state.as_mut().unwrap(),
                        hdr.as_ref().unwrap(),
                        &o.payload[off..],
                        &mut tiles_done,
                    )?;
                }
            }
            obu::OBU_TILE_GROUP => {
                let (Some(fs), Some(h)) = (state.as_mut(), hdr.as_ref()) else {
                    return Err(Error::invalid("AV1 tile group before frame header"));
                };
                decode_tile_group(fs, h, o.payload, &mut tiles_done)?;
            }
            _ => {}
        }
        if let (Some(h), true) = (hdr.as_ref(), state.is_some())
            && tiles_done == h.tile_info.cols * h.tile_info.rows
        {
            break;
        }
    }
    let (Some(seq), Some(hdr), Some(fs)) = (seq, hdr, state) else {
        return Err(Error::invalid("no AV1 frame in stream"));
    };
    if tiles_done != hdr.tile_info.cols * hdr.tile_info.rows {
        return Err(Error::invalid("AV1 frame is missing tiles"));
    }
    let frame = finish_frame(fs)?;
    Ok(Decoded { frame, seq, hdr })
}

fn check_limits(seq: &SequenceHeader, h: &FrameHeader) -> Result<()> {
    let pixels = h.upscaled_width as u64 * h.frame_height as u64;
    if pixels > crate::image::MAX_PIXELS {
        return Err(Error::unsupported("AV1 picture too large"));
    }
    if seq.color.mono_chrome && h.frame_type > 3 {
        return Err(Error::invalid("bad frame type"));
    }
    Ok(())
}

fn decode_tile_group(fs: &mut FrameState, hdr: &FrameHeader, data: &[u8], done: &mut usize) -> Result<()> {
    let ti = &hdr.tile_info;
    let num_tiles = ti.cols * ti.rows;
    let mut r = BitReader::new(data);
    let mut start = 0;
    let mut end = num_tiles - 1;
    if num_tiles > 1 && r.flag()? {
        let bits = ti.cols_log2 + ti.rows_log2;
        start = r.bits(bits)? as usize;
        end = r.bits(bits)? as usize;
    }
    r.byte_align();
    let mut pos = r.byte_pos();
    if start >= num_tiles || end >= num_tiles || end < start {
        return Err(Error::invalid("bad AV1 tile group"));
    }
    for tile_num in start..=end {
        let tile_row = tile_num / ti.cols;
        let tile_col = tile_num % ti.cols;
        let size = if tile_num == end {
            data.len().checked_sub(pos).ok_or_else(|| Error::invalid("truncated AV1 tile"))?
        } else {
            let n = ti.tile_size_bytes;
            let b = data.get(pos..pos + n).ok_or_else(|| Error::invalid("truncated AV1 tile size"))?;
            pos += n;
            let mut v = 0usize;
            for (i, &x) in b.iter().enumerate() {
                v |= (x as usize) << (8 * i);
            }
            v + 1
        };
        let tile = data.get(pos..pos + size).ok_or_else(|| Error::invalid("truncated AV1 tile"))?;
        pos += size;
        let cdf = Cdfs::new(hdr.quant.base_q_idx as u32);
        let dec = SymbolDecoder::new(tile, !hdr.disable_cdf_update);
        let rows = (ti.mi_row_starts[tile_row], ti.mi_row_starts[tile_row + 1]);
        let cols = (ti.mi_col_starts[tile_col], ti.mi_col_starts[tile_col + 1]);
        let mut tc = TileCoder::new(fs, cdf, dec, rows, cols);
        tc.decode_tile()?;
        *done += 1;
    }
    Ok(())
}

/// Copy the coded area of the decoded planes into a [`Frame`] whose planes are exactly `mi_cols * 4` wide.
fn coded_frame(fs: &FrameState) -> Frame {
    let mut planes = Vec::new();
    for (p, plane) in fs.planes.iter().enumerate() {
        let (sx, sy) = if p == 0 { (0, 0) } else { (fs.sub_x, fs.sub_y) };
        let (pw, ph) = ((fs.mi_cols * 4) >> sx, (fs.mi_rows * 4) >> sy);
        let mut out = Plane::new(pw, ph);
        for y in 0..ph {
            out.row_mut(y).copy_from_slice(&plane.row(y)[..pw]);
        }
        planes.push(out);
    }
    Frame {
        planes,
        bit_depth: fs.seq.color.bit_depth,
        sub_x: fs.seq.color.sub_x,
        sub_y: fs.seq.color.sub_y,
        width: fs.hdr.frame_width,
        height: fs.hdr.frame_height,
    }
}

/// Post-filter and crop the decoded frame to its visible size.
fn finish_frame(fs: FrameState) -> Result<Frame> {
    let hdr = &fs.hdr;
    let mut cur = coded_frame(&fs);
    // Deblocking.
    let lf = &hdr.loop_filter;
    if lf.level[0] != 0 || lf.level[1] != 0 {
        let mut mi = DeblockMi::new(fs.mi_cols, fs.mi_rows, fs.sub_x, fs.sub_y, fs.num_planes);
        mi.mi_size.copy_from_slice(&fs.mi_size);
        for (d, s) in mi.skip.iter_mut().zip(&fs.skip) {
            *d = *s != 0;
        }
        mi.y_mode.copy_from_slice(&fs.y_mode);
        mi.segment_id.copy_from_slice(&fs.seg_id);
        mi.delta_lf.copy_from_slice(&fs.delta_lf);
        for p in 0..fs.num_planes {
            mi.tx_size[p].copy_from_slice(&fs.lf_tx[p]);
        }
        let mut seg_lf = [[0i8; 4]; 8];
        for (seg, row) in seg_lf.iter_mut().enumerate() {
            for (k, v) in row.iter_mut().enumerate() {
                if hdr.seg_feature_active(seg, obu::SEG_LVL_ALT_LF_Y_V + k) {
                    *v = hdr.seg.feature_data[seg][obu::SEG_LVL_ALT_LF_Y_V + k] as i8;
                }
            }
        }
        let params = DeblockParams {
            filter_level: lf.level,
            sharpness: lf.sharpness,
            delta_enabled: lf.delta_enabled,
            ref_deltas: lf.ref_deltas,
            mode_deltas: lf.mode_deltas,
            delta_lf_multi: hdr.delta_lf_multi,
            seg_lf,
        };
        deblock_frame(&mut cur, &params, &mi);
    }
    // CDEF.
    let mut filtered = cur.clone();
    if fs.seq.enable_cdef && !hdr.coded_lossless && !hdr.allow_intrabc {
        let c = &hdr.cdef;
        let params = CdefParams {
            damping: c.damping,
            bits: c.bits,
            y_pri: c.y_pri,
            y_sec: c.y_sec,
            uv_pri: c.uv_pri,
            uv_sec: c.uv_sec,
        };
        let skip: Vec<bool> = fs.skip.iter().map(|&s| s != 0).collect();
        cdef_frame(&cur, &mut filtered, &params, &fs.cdef_idx, fs.cdef_stride, &skip, fs.mi_cols, fs.mi_rows);
    }
    let (mut deblocked, mut filtered) = (cur, filtered);
    if hdr.use_superres {
        deblocked = upscale_frame(&deblocked, hdr.upscaled_width, hdr.superres_denom as u8);
        filtered = upscale_frame(&filtered, hdr.upscaled_width, hdr.superres_denom as u8);
    }
    let out = if hdr.lr.uses_lr {
        let mut lr_planes = Vec::new();
        for p in 0..fs.num_planes {
            let (cols, rows, params) = &fs.lr_units[p];
            let mut lp = LrPlane { unit_size: hdr.lr.unit_size[p].max(1), units_per_row: *cols, units_per_col: *rows, units: Vec::new() };
            for u in params {
                lp.units.push(LrUnit {
                    kind: match u.kind {
                        obu::RESTORE_WIENER => LrType::Wiener,
                        obu::RESTORE_SGRPROJ => LrType::Sgrproj,
                        _ => LrType::None,
                    },
                    wiener: u.wiener,
                    sgr_set: u.sgr_set,
                    sgr_xqd: u.sgr_xqd,
                });
            }
            if params.is_empty() {
                lp = LrPlane::new(64, 1, 1);
            }
            lr_planes.push(lp);
        }
        let mut dst = filtered.clone();
        deblocked.width = hdr.upscaled_width;
        filtered.width = hdr.upscaled_width;
        dst.width = hdr.upscaled_width;
        loop_restoration_frame(&filtered, &deblocked, &mut dst, &lr_planes);
        dst
    } else {
        filtered
    };
    let (w, h) = (hdr.upscaled_width, hdr.frame_height);
    let mut planes = Vec::new();
    for (p, plane) in out.planes.iter().enumerate() {
        let (sx, sy) = if p == 0 { (0, 0) } else { (fs.sub_x, fs.sub_y) };
        let (pw, ph) = ((w + sx) >> sx, (h + sy) >> sy);
        let mut o = Plane::new(pw, ph);
        for y in 0..ph {
            o.row_mut(y).copy_from_slice(&plane.row(y)[..pw]);
        }
        planes.push(o);
    }
    let frame = Frame {
        planes,
        bit_depth: fs.seq.color.bit_depth,
        sub_x: fs.seq.color.sub_x,
        sub_y: fs.seq.color.sub_y,
        width: w,
        height: h,
    };
    let g = &hdr.film_grain;
    if !g.apply_grain {
        return Ok(frame);
    }
    let mut p = FilmGrainParams { apply_grain: true, grain_seed: g.grain_seed, ..Default::default() };
    p.num_y_points = g.num_y_points as u8;
    p.point_y_value.copy_from_slice(&g.point_y_value[..14]);
    p.point_y_scaling.copy_from_slice(&g.point_y_scaling[..14]);
    p.chroma_scaling_from_luma = g.chroma_scaling_from_luma;
    p.num_cb_points = g.num_cb_points as u8;
    p.point_cb_value.copy_from_slice(&g.point_cb_value[..10]);
    p.point_cb_scaling.copy_from_slice(&g.point_cb_scaling[..10]);
    p.num_cr_points = g.num_cr_points as u8;
    p.point_cr_value.copy_from_slice(&g.point_cr_value[..10]);
    p.point_cr_scaling.copy_from_slice(&g.point_cr_scaling[..10]);
    p.grain_scaling_minus_8 = g.grain_scaling_minus_8;
    p.ar_coeff_lag = g.ar_coeff_lag;
    p.ar_coeffs_y_plus_128 = g.ar_coeffs_y_plus_128;
    p.ar_coeffs_cb_plus_128 = g.ar_coeffs_cb_plus_128;
    p.ar_coeffs_cr_plus_128 = g.ar_coeffs_cr_plus_128;
    p.ar_coeff_shift_minus_6 = g.ar_coeff_shift_minus_6;
    p.grain_scale_shift = g.grain_scale_shift;
    p.cb_mult = g.cb_mult;
    p.cb_luma_mult = g.cb_luma_mult;
    p.cb_offset = g.cb_offset;
    p.cr_mult = g.cr_mult;
    p.cr_luma_mult = g.cr_luma_mult;
    p.cr_offset = g.cr_offset;
    p.overlap_flag = g.overlap_flag;
    p.clip_to_restricted_range = g.clip_to_restricted_range;
    Ok(apply_film_grain(&frame, &p, fs.seq.color.matrix_coefficients == 0))
}
