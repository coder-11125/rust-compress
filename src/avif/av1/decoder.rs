//! AV1 still-picture decoder: OBU stream in, picture out.

use super::bits::BitReader;
use super::cdef::{CdefParams, cdef_frame};
use super::cdf_tables::Cdfs;
use super::consts::{INTRA_FRAME, NONE_FRAME};
use super::deblock::{DeblockMi, DeblockParams, deblock_frame};
use super::entropy::SymbolDecoder;
use super::filmgrain::{FilmGrainParams, apply_film_grain};
use super::frame::{Frame, Plane};
use super::lr::{LrPlane, LrType, LrUnit, loop_restoration_frame};
use super::obu::{self, FrameHeader, SequenceHeader};
use super::refs::{RefSet, RefSlot, RefView};
use super::superres::upscale_frame;
use super::tile::{FrameState, TileCoder};
use crate::error::{Error, Result};
use std::sync::Arc;

const REFMVS_LIMIT: i32 = (1 << 12) - 1;

/// A decoded picture and the headers it came with.
pub struct Decoded {
    pub frame: Frame,
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
}

/// Decode the first shown frame of a low-overhead OBU stream.
pub fn decode(data: &[u8]) -> Result<Decoded> {
    let mut dec = Decoder::new();
    dec.decode_temporal_unit(data)?.into_iter().next().ok_or_else(|| Error::invalid("no AV1 frame in stream"))
}

/// The frame being decoded between its header and its last tile group.
struct Cur {
    hdr: FrameHeader,
    fs: FrameState,
    tiles_done: usize,
    /// The CDFs the frame ends with (`Saved*Cdf`), when `disable_frame_end_update_cdf` is 0.
    saved_cdf: Option<Box<Cdfs>>,
}

/// A stateful AV1 decoder: the eight reference slots live across frames.
pub struct Decoder {
    seq: Option<SequenceHeader>,
    slots: [RefSlot; 8],
    cur: Option<Cur>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder { seq: None, slots: Default::default(), cur: None }
    }

    /// Decode the OBUs of one temporal unit (or any run of whole frames) and return the shown frames in order.
    pub fn decode_temporal_unit(&mut self, data: &[u8]) -> Result<Vec<Decoded>> {
        let obus = obu::split_obus(data)?;
        let mut out = Vec::new();
        for o in &obus {
            match o.kind {
                obu::OBU_SEQUENCE_HEADER => {
                    self.seq = Some(SequenceHeader::parse(o.payload)?);
                }
                obu::OBU_FRAME_HEADER | obu::OBU_FRAME => {
                    if self.cur.is_some() {
                        // A redundant copy of the header of the frame being decoded.
                        if o.kind == obu::OBU_FRAME_HEADER {
                            continue;
                        }
                        return Err(Error::invalid("AV1 frame starts before the previous one is complete"));
                    }
                    let seq = self.seq.clone().ok_or_else(|| Error::invalid("AV1 frame before sequence header"))?;
                    let mut r = BitReader::new(o.payload);
                    let ref_infos: obu::RefHeaders = std::array::from_fn(|i| self.slots[i].info.clone());
                    let h = FrameHeader::parse_with_refs(&seq, &mut r, &ref_infos)?;
                    if std::env::var_os("AV1_DEBUG").is_some() {
                        eprintln!("{seq:?}\n{h:?}");
                    }
                    if h.show_existing_frame {
                        out.push(self.show_existing(&seq, h)?);
                        continue;
                    }
                    check_limits(&seq, &h)?;
                    let fs = self.start_frame(&seq, &h)?;
                    self.cur = Some(Cur { hdr: h, fs, tiles_done: 0, saved_cdf: None });
                    if o.kind == obu::OBU_FRAME {
                        r.byte_align();
                        let off = r.byte_pos();
                        self.tile_group(&o.payload[off..])?;
                    }
                }
                obu::OBU_TILE_GROUP => {
                    if self.cur.is_none() {
                        return Err(Error::invalid("AV1 tile group before frame header"));
                    }
                    self.tile_group(o.payload)?;
                }
                _ => {}
            }
            if let Some(c) = &self.cur
                && c.tiles_done == c.hdr.tile_info.cols * c.hdr.tile_info.rows
            {
                let d = self.finish_cur()?;
                out.extend(d);
            }
        }
        if self.cur.is_some() {
            return Err(Error::invalid("AV1 frame is missing tiles"));
        }
        Ok(out)
    }

    fn tile_group(&mut self, data: &[u8]) -> Result<()> {
        let c = self.cur.as_mut().unwrap();
        decode_tile_group(&mut c.fs, &c.hdr, data, &mut c.tiles_done, &mut c.saved_cdf)
    }

    /// Set up the frame state: CDFs, previous segmentation map and the reference views.
    fn start_frame(&self, seq: &SequenceHeader, h: &FrameHeader) -> Result<FrameState> {
        let mut fs = FrameState::new(seq, h)?;
        if h.is_intra() {
            return Ok(fs);
        }
        let mut set = RefSet::default();
        for i in 0..7 {
            let slot = &self.slots[h.ref_frame_idx[i]];
            let frame =
                slot.frame.clone().ok_or_else(|| Error::invalid("inter frame refers to a missing reference frame"))?;
            if frame.bit_depth != seq.color.bit_depth
                || frame.sub_x != seq.color.sub_x
                || frame.sub_y != seq.color.sub_y
            {
                return Err(Error::unsupported("reference frame with a different format"));
            }
            set.views[i + 1] = Some(RefView {
                frame,
                upscaled_width: slot.info.upscaled_width,
                frame_width: slot.info.frame_width,
                frame_height: slot.info.frame_height,
                mi_cols: slot.mi_cols,
                mi_rows: slot.mi_rows,
                order_hint: slot.info.order_hint,
                frame_type: slot.info.frame_type,
                saved_order_hints: slot.saved_order_hints,
                mf_ref_frames: slot.mf_ref_frames.clone(),
                mf_mvs: slot.mf_mvs.clone(),
            });
        }
        if h.use_ref_frame_mvs {
            fs.motion_field = super::motion_field::estimate(seq, h, &set);
        }
        fs.refs = Some(Arc::new(set));
        if h.primary_ref_frame != obu::PRIMARY_REF_NONE {
            let slot = &self.slots[h.ref_frame_idx[h.primary_ref_frame as usize]];
            if let Some(c) = &slot.cdfs {
                let mut cdf = Box::new((**c).clone());
                cdf.load_reset();
                fs.cdf_base = cdf;
            }
            if h.seg.enabled && slot.mi_cols == h.mi_cols && slot.mi_rows == h.mi_rows {
                fs.prev_segment_ids = slot.segment_ids.clone();
            }
        }
        Ok(fs)
    }

    /// Output a frame that is already in a slot (`show_existing_frame`).
    fn show_existing(&mut self, seq: &SequenceHeader, h: FrameHeader) -> Result<Decoded> {
        let slot = self.slots[h.frame_to_show_map_idx].clone();
        let frame = slot.frame.clone().ok_or_else(|| Error::invalid("show_existing_frame of an empty slot"))?;
        if h.frame_type == obu::KEY_FRAME {
            // The reference frame loading and update processes: every slot becomes a copy of this one.
            for s in self.slots.iter_mut() {
                *s = slot.clone();
                s.showable_frame = false;
            }
        }
        let frame = add_grain(&frame, seq, &h.film_grain);
        Ok(Decoded { frame, seq: seq.clone(), hdr: h })
    }

    /// All tiles of the current frame are in: post-filter, update the reference slots and output it if shown.
    fn finish_cur(&mut self) -> Result<Option<Decoded>> {
        let Cur { hdr, fs, saved_cdf, .. } = self.cur.take().unwrap();
        let seq = fs.seq.clone();
        let recon = Arc::new(reconstruct(&fs)?);
        if hdr.refresh_frame_flags != 0 {
            let slot = self.make_slot(&hdr, &fs, recon.clone(), saved_cdf);
            for i in 0..8 {
                if (hdr.refresh_frame_flags >> i) & 1 == 1 {
                    self.slots[i] = slot.clone();
                }
            }
        }
        if !hdr.show_frame {
            return Ok(None);
        }
        let frame = add_grain(&recon, &seq, &hdr.film_grain);
        Ok(Some(Decoded { frame, seq, hdr }))
    }

    /// What the reference frame update process (7.20) saves for a frame.
    fn make_slot(&self, h: &FrameHeader, fs: &FrameState, frame: Arc<Frame>, saved_cdf: Option<Box<Cdfs>>) -> RefSlot {
        let seq = &fs.seq;
        let (cols2, rows2) = (fs.mi_cols / 2, fs.mi_rows / 2);
        let mut mf_ref = vec![NONE_FRAME; cols2 * rows2];
        let mut mf_mv = vec![[0i16; 2]; cols2 * rows2];
        if !h.is_intra() {
            for y in 0..rows2 {
                for x in 0..cols2 {
                    let i = (2 * y + 1) * fs.mi_cols + 2 * x + 1;
                    for list in 0..2 {
                        let r = fs.ref_frame[i][list];
                        if r <= INTRA_FRAME {
                            continue;
                        }
                        if FrameHeader::relative_dist(seq, h.order_hints[r as usize], h.order_hint) >= 0 {
                            continue;
                        }
                        let mv = fs.mvs[i][list];
                        if mv[0].abs() <= REFMVS_LIMIT && mv[1].abs() <= REFMVS_LIMIT {
                            mf_ref[y * cols2 + x] = r;
                            mf_mv[y * cols2 + x] = [mv[0] as i16, mv[1] as i16];
                        }
                    }
                }
            }
        }
        let seg = if h.seg.enabled && !h.seg.update_map {
            fs.prev_segment_ids.clone().unwrap_or_else(|| Arc::new(vec![0; fs.mi_cols * fs.mi_rows]))
        } else {
            Arc::new(fs.seg_id.clone())
        };
        let cdfs = saved_cdf.unwrap_or_else(|| fs.cdf_base.clone());
        RefSlot {
            info: obu::RefHeaderInfo {
                valid: true,
                frame_id: h.current_frame_id,
                upscaled_width: h.upscaled_width,
                frame_width: h.frame_width,
                frame_height: h.frame_height,
                render_width: h.render_width,
                render_height: h.render_height,
                order_hint: h.order_hint,
                frame_type: h.frame_type,
                gm_params: h.gm_params,
                lf_ref_deltas: h.loop_filter.ref_deltas,
                lf_mode_deltas: h.loop_filter.mode_deltas,
                feature_enabled: h.seg.feature_enabled,
                feature_data: h.seg.feature_data,
                film_grain: h.film_grain.clone(),
            },
            mi_cols: fs.mi_cols,
            mi_rows: fs.mi_rows,
            bit_depth: seq.color.bit_depth,
            sub_x: seq.color.sub_x,
            sub_y: seq.color.sub_y,
            showable_frame: h.showable_frame,
            saved_order_hints: h.order_hints,
            frame: Some(frame),
            mf_ref_frames: Some(Arc::new(mf_ref)),
            mf_mvs: Some(Arc::new(mf_mv)),
            segment_ids: Some(seg),
            cdfs: Some(Arc::new(*cdfs)),
        }
    }
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

fn decode_tile_group(
    fs: &mut FrameState,
    hdr: &FrameHeader,
    data: &[u8],
    done: &mut usize,
    saved_cdf: &mut Option<Box<Cdfs>>,
) -> Result<()> {
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
        let cdf = fs.cdf_base.clone();
        let dec = SymbolDecoder::new(tile, !hdr.disable_cdf_update);
        let rows = (ti.mi_row_starts[tile_row], ti.mi_row_starts[tile_row + 1]);
        let cols = (ti.mi_col_starts[tile_col], ti.mi_col_starts[tile_col + 1]);
        let mut tc = TileCoder::new(fs, cdf, dec, rows, cols);
        tc.decode_tile()?;
        if !hdr.disable_frame_end_update_cdf && tile_num == ti.context_update_tile_id {
            *saved_cdf = Some(tc.cdf.clone());
        }
        if tc.stats {
            let mut sizes = [0usize; 22];
            let (mut skips, mut modes, mut txs) = (0, [0usize; 14], [0usize; 19]);
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
        }
        *done += 1;
    }
    Ok(())
}

/// Copy the coded area of the decoded planes into a [`Frame`] whose planes are exactly `mi_cols * 4` wide.
pub(crate) fn coded_frame(fs: &FrameState) -> Frame {
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

/// Deblocking parameters and per-block data of a decoded (or encoded) frame.
pub(crate) fn build_deblock(fs: &FrameState) -> (DeblockParams, DeblockMi) {
    let hdr = &fs.hdr;
    let lf = &hdr.loop_filter;
    let mut mi = DeblockMi::new(fs.mi_cols, fs.mi_rows, fs.sub_x, fs.sub_y, fs.num_planes);
    mi.mi_size.copy_from_slice(&fs.mi_size);
    for (d, s) in mi.skip.iter_mut().zip(&fs.skip) {
        *d = *s != 0;
    }
    mi.y_mode.copy_from_slice(&fs.y_mode);
    for (d, s) in mi.ref_frame.iter_mut().zip(&fs.ref_frame) {
        *d = s[0].max(0);
    }
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
    (params, mi)
}

/// Post-filter, crop to the visible size and add film grain.
pub(crate) fn finish_frame(fs: FrameState) -> Result<Frame> {
    let frame = reconstruct(&fs)?;
    Ok(add_grain(&frame, &fs.seq, &fs.hdr.film_grain))
}

/// Post-filter the decoded frame and crop it to its visible size (what a reference slot stores).
pub(crate) fn reconstruct(fs: &FrameState) -> Result<Frame> {
    let hdr = &fs.hdr;
    let mut cur = coded_frame(fs);
    // Deblocking.
    let lf = &hdr.loop_filter;
    if lf.level[0] != 0 || lf.level[1] != 0 {
        let (params, mi) = build_deblock(fs);
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
            let mut lp = LrPlane {
                unit_size: hdr.lr.unit_size[p].max(1),
                units_per_row: *cols,
                units_per_col: *rows,
                units: Vec::new(),
            };
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
    Ok(frame)
}

/// Apply film grain to an output frame.
fn add_grain(frame: &Frame, seq: &SequenceHeader, g: &obu::FilmGrain) -> Frame {
    if !g.apply_grain {
        return frame.clone();
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
    apply_film_grain(frame, &p, seq.color.matrix_coefficients == 0)
}
