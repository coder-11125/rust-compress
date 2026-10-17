//! Serialization of OBUs: sequence header and intra frame header for still pictures
//! (the `reduced_still_picture_header` form), mirroring the parsers in [`super::obu`].

use super::bits::{BitWriter, write_leb128};
use super::obu::*;

/// Wrap a payload as an OBU with a size field.
pub fn write_obu(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![(kind << 3) | 2];
    write_leb128(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    out
}

fn trailing_bits(w: &mut BitWriter) {
    w.bit(true);
    w.byte_align();
}

/// A sequence header OBU payload for a reduced still picture header sequence.
pub fn sequence_header_payload(s: &SequenceHeader) -> Vec<u8> {
    assert!(s.reduced_still_picture_header, "only reduced still picture sequences are written");
    let mut w = BitWriter::new();
    w.bits(s.profile as u32, 3);
    w.bit(true); // still_picture
    w.bit(true); // reduced_still_picture_header
    w.bits(s.operating_points[0].level as u32, 5);
    w.bits(s.frame_width_bits - 1, 4);
    w.bits(s.frame_height_bits - 1, 4);
    w.bits(s.max_frame_width - 1, s.frame_width_bits);
    w.bits(s.max_frame_height - 1, s.frame_height_bits);
    w.bit(s.use_128x128_superblock);
    w.bit(s.enable_filter_intra);
    w.bit(s.enable_intra_edge_filter);
    w.bit(s.enable_superres);
    w.bit(s.enable_cdef);
    w.bit(s.enable_restoration);
    let c = &s.color;
    let high_bitdepth = c.bit_depth > 8;
    w.bit(high_bitdepth);
    if s.profile == 2 && high_bitdepth {
        w.bit(c.bit_depth == 12);
    }
    if s.profile != 1 {
        w.bit(c.mono_chrome);
    }
    let described = !(c.color_primaries == 2 && c.transfer_characteristics == 2 && c.matrix_coefficients == 2);
    w.bit(described);
    if described {
        w.bits(c.color_primaries as u32, 8);
        w.bits(c.transfer_characteristics as u32, 8);
        w.bits(c.matrix_coefficients as u32, 8);
    }
    if c.mono_chrome {
        w.bit(c.color_range);
    } else if c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0 {
        // sRGB identity: range and subsampling are implied; only separate_uv_delta_q follows.
        w.bit(c.separate_uv_delta_q);
    } else {
        w.bit(c.color_range);
        if s.profile == 2 && c.bit_depth == 12 {
            w.bit(c.sub_x != 0);
            if c.sub_x != 0 {
                w.bit(c.sub_y != 0);
            }
        }
        if c.sub_x != 0 && c.sub_y != 0 {
            w.bits(c.chroma_sample_position as u32, 2);
        }
        w.bit(c.separate_uv_delta_q);
    }
    if c.mono_chrome {
        // Monochrome returns before separate_uv_delta_q.
    }
    w.bit(s.film_grain_params_present);
    trailing_bits(&mut w);
    w.finish()
}

fn write_delta_q(w: &mut BitWriter, v: i32) {
    w.bit(v != 0);
    if v != 0 {
        w.signed(v, 7);
    }
}

/// The frame header bits for a key frame of a reduced still picture sequence (no tile data).
pub fn frame_header_bits(s: &SequenceHeader, h: &FrameHeader, w: &mut BitWriter) {
    assert!(s.reduced_still_picture_header && h.frame_type == KEY_FRAME);
    w.bit(h.disable_cdf_update);
    w.bit(h.allow_screen_content_tools);
    if h.allow_screen_content_tools {
        w.bit(h.force_integer_mv);
    }
    // frame_size_override_flag, order_hint, primary_ref_frame, refresh_frame_flags: implied.
    if s.enable_superres {
        w.bit(h.use_superres);
        if h.use_superres {
            w.bits((h.superres_denom - 9) as u32, 3);
        }
    }
    w.bit(false); // render_and_frame_size_different
    if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
        w.bit(h.allow_intrabc);
    }
    // disable_frame_end_update_cdf: implied 1.
    write_tile_info(s, h, w);
    // quantization_params
    let q = &h.quant;
    w.bits(q.base_q_idx as u32, 8);
    write_delta_q(w, q.delta_q_y_dc);
    if s.color.num_planes() > 1 {
        let diff_uv = q.delta_q_v_dc != q.delta_q_u_dc || q.delta_q_v_ac != q.delta_q_u_ac;
        if s.color.separate_uv_delta_q {
            w.bit(diff_uv);
        }
        write_delta_q(w, q.delta_q_u_dc);
        write_delta_q(w, q.delta_q_u_ac);
        if diff_uv {
            write_delta_q(w, q.delta_q_v_dc);
            write_delta_q(w, q.delta_q_v_ac);
        }
    }
    w.bit(q.using_qmatrix);
    if q.using_qmatrix {
        w.bits(q.qm_y as u32, 4);
        w.bits(q.qm_u as u32, 4);
        if s.color.separate_uv_delta_q {
            w.bits(q.qm_v as u32, 4);
        }
    }
    // segmentation_params
    w.bit(h.seg.enabled);
    if h.seg.enabled {
        const BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
        const SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let en = h.seg.feature_enabled[i][j];
                w.bit(en);
                if en {
                    if SIGNED[j] {
                        w.signed(h.seg.feature_data[i][j], 1 + BITS[j]);
                    } else {
                        w.bits(h.seg.feature_data[i][j] as u32, BITS[j]);
                    }
                }
            }
        }
    }
    // delta_q_params / delta_lf_params
    if q.base_q_idx > 0 {
        w.bit(h.delta_q_present);
    }
    if h.delta_q_present {
        w.bits(h.delta_q_res as u32, 2);
        if !h.allow_intrabc {
            w.bit(h.delta_lf_present);
        }
        if h.delta_lf_present {
            w.bits(h.delta_lf_res as u32, 2);
            w.bit(h.delta_lf_multi);
        }
    }
    // loop_filter_params
    if !(h.coded_lossless || h.allow_intrabc) {
        let lf = &h.loop_filter;
        w.bits(lf.level[0] as u32, 6);
        w.bits(lf.level[1] as u32, 6);
        if s.color.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            w.bits(lf.level[2] as u32, 6);
            w.bits(lf.level[3] as u32, 6);
        }
        w.bits(lf.sharpness as u32, 3);
        w.bit(lf.delta_enabled);
        if lf.delta_enabled {
            let default_ref = [1i8, 0, 0, 0, -1, 0, -1, -1];
            let update = lf.ref_deltas != default_ref || lf.mode_deltas != [0, 0];
            w.bit(update);
            if update {
                for i in 0..8 {
                    let changed = lf.ref_deltas[i] != default_ref[i];
                    w.bit(changed);
                    if changed {
                        w.signed(lf.ref_deltas[i] as i32, 7);
                    }
                }
                for i in 0..2 {
                    let changed = lf.mode_deltas[i] != 0;
                    w.bit(changed);
                    if changed {
                        w.signed(lf.mode_deltas[i] as i32, 7);
                    }
                }
            }
        }
    }
    // cdef_params
    if !(h.coded_lossless || h.allow_intrabc || !s.enable_cdef) {
        let c = &h.cdef;
        w.bits((c.damping - 3) as u32, 2);
        w.bits(c.bits as u32, 2);
        for i in 0..(1usize << c.bits) {
            w.bits(c.y_pri[i] as u32, 4);
            w.bits(if c.y_sec[i] == 4 { 3 } else { c.y_sec[i] as u32 }, 2);
            if s.color.num_planes() > 1 {
                w.bits(c.uv_pri[i] as u32, 4);
                w.bits(if c.uv_sec[i] == 4 { 3 } else { c.uv_sec[i] as u32 }, 2);
            }
        }
    }
    // lr_params
    if !(h.all_lossless || h.allow_intrabc || !s.enable_restoration) {
        // Frame restoration type -> 2-bit code (inverse of the Remap_Lr_Type table).
        let code = |k: u8| match k {
            RESTORE_NONE => 0,
            RESTORE_SWITCHABLE => 1,
            RESTORE_WIENER => 2,
            _ => 3,
        };
        let mut uses_chroma = false;
        for i in 0..s.color.num_planes() {
            w.bits(code(h.lr.kind[i]), 2);
            if i > 0 && h.lr.kind[i] != RESTORE_NONE {
                uses_chroma = true;
            }
        }
        if h.lr.uses_lr {
            // unit sizes: luma 64 << shift (shift 0..2, 128 SB: 1..2)
            let shift = (h.lr.unit_size[0] / 64).ilog2();
            if s.use_128x128_superblock {
                w.bit(shift >= 2);
            } else {
                w.bit(shift >= 1);
                if shift >= 1 {
                    w.bit(shift >= 2);
                }
            }
            if s.color.sub_x != 0 && s.color.sub_y != 0 && uses_chroma {
                w.bit(h.lr.unit_size[1] != h.lr.unit_size[0]);
            }
        }
    }
    if !h.coded_lossless {
        w.bit(h.tx_mode_select);
    }
    w.bit(h.reduced_tx_set);
}

fn write_tile_info(s: &SequenceHeader, h: &FrameHeader, w: &mut BitWriter) {
    let t = &h.tile_info;
    let sb_cols = if s.use_128x128_superblock { (h.mi_cols + 31) >> 5 } else { (h.mi_cols + 15) >> 4 };
    let sb_rows = if s.use_128x128_superblock { (h.mi_rows + 31) >> 5 } else { (h.mi_rows + 15) >> 4 };
    let sb_size = if s.use_128x128_superblock { 7 } else { 6 };
    let max_tile_width_sb = 4096 >> sb_size;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sb_size);
    let tl2 = |blk: usize, target: usize| {
        let mut k = 0;
        while (blk << k) < target {
            k += 1;
        }
        k
    };
    let min_log2_tile_cols = tl2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tl2(1, sb_cols.min(64));
    let max_log2_tile_rows = tl2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tl2(max_tile_area_sb, sb_rows * sb_cols));
    w.bit(true); // uniform_tile_spacing_flag
    let mut cl = min_log2_tile_cols;
    while cl < max_log2_tile_cols {
        if cl < t.cols_log2 {
            w.bit(true);
            cl += 1;
        } else {
            w.bit(false);
            break;
        }
    }
    let min_log2_tile_rows = min_log2_tiles.saturating_sub(cl);
    let mut rl = min_log2_tile_rows;
    while rl < max_log2_tile_rows {
        if rl < t.rows_log2 {
            w.bit(true);
            rl += 1;
        } else {
            w.bit(false);
            break;
        }
    }
    if t.cols_log2 > 0 || t.rows_log2 > 0 {
        w.bits(t.context_update_tile_id as u32, t.rows_log2 + t.cols_log2);
        w.bits((t.tile_size_bytes - 1) as u32, 2);
    }
}

/// Tile boundaries for uniform spacing with the given log2 tile counts, as the parser derives them.
pub fn uniform_tile_info(
    s: &SequenceHeader,
    mi_cols: usize,
    mi_rows: usize,
    cols_log2: u32,
    rows_log2: u32,
) -> TileInfo {
    let sb_shift = if s.use_128x128_superblock { 5 } else { 4 };
    let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
    let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
    let mut t = TileInfo { cols_log2, rows_log2, tile_size_bytes: 4, ..Default::default() };
    let tile_w = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
    let mut start = 0;
    while start < sb_cols {
        t.mi_col_starts.push(start << sb_shift);
        start += tile_w;
    }
    t.cols = t.mi_col_starts.len();
    t.mi_col_starts.push(mi_cols);
    let tile_h = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
    let mut start = 0;
    while start < sb_rows {
        t.mi_row_starts.push(start << sb_shift);
        start += tile_h;
    }
    t.rows = t.mi_row_starts.len();
    t.mi_row_starts.push(mi_rows);
    // The log2 values written are those implied by the actual tile counts.
    t.cols_log2 = (t.cols.next_power_of_two()).ilog2();
    t.rows_log2 = (t.rows.next_power_of_two()).ilog2();
    t
}
