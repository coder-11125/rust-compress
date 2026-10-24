//! OBU framing, the sequence header and the (intra-frame) frame header, after AV1 spec sections 5.3-5.9.

use super::bits::{BitReader, leb128};
use crate::error::{Error, Result};

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_FRAME: u8 = 6;
pub const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;

pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY_FRAME: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;

pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;
pub const SEG_LVL_REF_FRAME: usize = 5;
pub const SEG_LVL_MAX: usize = 8;
const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

/// One OBU: its type and payload (without header or size field).
pub struct Obu<'a> {
    pub kind: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub has_extension: bool,
    pub payload: &'a [u8],
}

/// Split a low-overhead (Section 5) OBU stream into OBUs.
pub fn split_obus(mut data: &[u8]) -> Result<Vec<Obu<'_>>> {
    let mut out = Vec::new();
    while !data.is_empty() {
        let h = data[0];
        if h & 0x80 != 0 {
            return Err(Error::invalid("AV1 OBU forbidden bit set"));
        }
        let kind = (h >> 3) & 15;
        let has_extension = h & 4 != 0;
        let has_size = h & 2 != 0;
        let mut pos = 1;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if has_extension {
            let e = *data.get(1).ok_or_else(|| Error::invalid("truncated OBU header"))?;
            temporal_id = e >> 5;
            spatial_id = (e >> 3) & 3;
            pos = 2;
        }
        let size = if has_size {
            let (v, n) = leb128(&data[pos..])?;
            pos += n;
            usize::try_from(v).map_err(|_| Error::invalid("OBU too large"))?
        } else {
            data.len() - pos
        };
        let end = pos.checked_add(size).filter(|&e| e <= data.len()).ok_or_else(|| Error::invalid("truncated OBU"))?;
        out.push(Obu { kind, temporal_id, spatial_id, has_extension, payload: &data[pos..end] });
        data = &data[end..];
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct ColorConfig {
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub sub_x: u8,
    pub sub_y: u8,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SequenceHeader {
    pub profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present: bool,
    pub equal_picture_interval: bool,
    pub decoder_model_info_present: bool,
    pub buffer_delay_length: u32,
    pub buffer_removal_time_length: u32,
    pub frame_presentation_time_length: u32,
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits: u32,
    pub frame_height_bits: u32,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length: u32,
    pub additional_frame_id_length: u32,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: ColorConfig,
    pub film_grain_params_present: bool,
}

#[derive(Clone, Debug, Default)]
pub struct OperatingPoint {
    pub idc: u32,
    pub level: u8,
    pub tier: bool,
    pub decoder_model_present: bool,
}

impl SequenceHeader {
    pub fn parse(payload: &[u8]) -> Result<SequenceHeader> {
        let mut r = BitReader::new(payload);
        let mut s = SequenceHeader { profile: r.bits(3)? as u8, ..Default::default() };
        if s.profile > 2 {
            return Err(Error::invalid("bad AV1 profile"));
        }
        s.still_picture = r.flag()?;
        s.reduced_still_picture_header = r.flag()?;
        if s.reduced_still_picture_header {
            s.operating_points.push(OperatingPoint {
                idc: 0,
                level: r.bits(5)? as u8,
                tier: false,
                decoder_model_present: false,
            });
        } else {
            s.timing_info_present = r.flag()?;
            if s.timing_info_present {
                r.bits(32)?;
                r.bits(32)?;
                s.equal_picture_interval = r.flag()?;
                if s.equal_picture_interval {
                    r.uvlc()?;
                }
                s.decoder_model_info_present = r.flag()?;
                if s.decoder_model_info_present {
                    s.buffer_delay_length = r.bits(5)? + 1;
                    r.bits(32)?;
                    s.buffer_removal_time_length = r.bits(5)? + 1;
                    s.frame_presentation_time_length = r.bits(5)? + 1;
                }
            }
            let initial_display_delay_present = r.flag()?;
            let count = r.bits(5)? + 1;
            for _ in 0..count {
                let idc = r.bits(12)?;
                let level = r.bits(5)? as u8;
                let tier = if level > 7 { r.flag()? } else { false };
                let mut decoder_model_present = false;
                if s.decoder_model_info_present {
                    decoder_model_present = r.flag()?;
                    if decoder_model_present {
                        r.bits(s.buffer_delay_length)?;
                        r.bits(s.buffer_delay_length)?;
                        r.bit()?;
                    }
                }
                if initial_display_delay_present && r.flag()? {
                    r.bits(4)?;
                }
                s.operating_points.push(OperatingPoint { idc, level, tier, decoder_model_present });
            }
        }
        s.frame_width_bits = r.bits(4)? + 1;
        s.frame_height_bits = r.bits(4)? + 1;
        s.max_frame_width = r.bits(s.frame_width_bits)? + 1;
        s.max_frame_height = r.bits(s.frame_height_bits)? + 1;
        if !s.reduced_still_picture_header {
            s.frame_id_numbers_present = r.flag()?;
        }
        if s.frame_id_numbers_present {
            s.delta_frame_id_length = r.bits(4)? + 2;
            s.additional_frame_id_length = r.bits(3)? + 1;
        }
        s.use_128x128_superblock = r.flag()?;
        s.enable_filter_intra = r.flag()?;
        s.enable_intra_edge_filter = r.flag()?;
        s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        s.seq_force_integer_mv = SELECT_INTEGER_MV;
        if !s.reduced_still_picture_header {
            s.enable_interintra_compound = r.flag()?;
            s.enable_masked_compound = r.flag()?;
            s.enable_warped_motion = r.flag()?;
            s.enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                s.enable_jnt_comp = r.flag()?;
                s.enable_ref_frame_mvs = r.flag()?;
            }
            if r.flag()? {
                s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
            } else {
                s.seq_force_screen_content_tools = r.bit()? as u8;
            }
            if s.seq_force_screen_content_tools > 0 {
                if r.flag()? {
                    s.seq_force_integer_mv = SELECT_INTEGER_MV;
                } else {
                    s.seq_force_integer_mv = r.bit()? as u8;
                }
            } else {
                s.seq_force_integer_mv = SELECT_INTEGER_MV;
            }
            if s.enable_order_hint {
                s.order_hint_bits = r.bits(3)? + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.color = parse_color_config(&mut r, s.profile)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }
}

fn parse_color_config(r: &mut BitReader, profile: u8) -> Result<ColorConfig> {
    let mut c = ColorConfig::default();
    let high_bitdepth = r.flag()?;
    c.bit_depth = if profile == 2 && high_bitdepth {
        if r.flag()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    c.mono_chrome = if profile == 1 { false } else { r.flag()? };
    if r.flag()? {
        c.color_primaries = r.bits(8)? as u8;
        c.transfer_characteristics = r.bits(8)? as u8;
        c.matrix_coefficients = r.bits(8)? as u8;
    } else {
        c.color_primaries = 2;
        c.transfer_characteristics = 2;
        c.matrix_coefficients = 2;
    }
    if c.mono_chrome {
        c.color_range = r.flag()?;
        c.sub_x = 1;
        c.sub_y = 1;
        return Ok(c);
    }
    if c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0 {
        c.color_range = true;
    } else {
        c.color_range = r.flag()?;
        match profile {
            0 => (c.sub_x, c.sub_y) = (1, 1),
            1 => (c.sub_x, c.sub_y) = (0, 0),
            _ => {
                if c.bit_depth == 12 {
                    c.sub_x = r.bit()? as u8;
                    c.sub_y = if c.sub_x != 0 { r.bit()? as u8 } else { 0 };
                } else {
                    (c.sub_x, c.sub_y) = (1, 0);
                }
            }
        }
        if c.sub_x != 0 && c.sub_y != 0 {
            c.chroma_sample_position = r.bits(2)? as u8;
        }
    }
    c.separate_uv_delta_q = r.flag()?;
    Ok(c)
}

#[derive(Clone, Debug, Default)]
pub struct TileInfo {
    pub cols: usize,
    pub rows: usize,
    pub cols_log2: u32,
    pub rows_log2: u32,
    /// In 4x4 mode-info units, `cols + 1` / `rows + 1` entries.
    pub mi_col_starts: Vec<usize>,
    pub mi_row_starts: Vec<usize>,
    pub context_update_tile_id: usize,
    pub tile_size_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct QuantParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    pub delta_q_v_dc: i32,
    pub delta_q_v_ac: i32,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Clone, Debug, Default)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

#[derive(Clone, Debug, Default)]
pub struct LoopFilterParams {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

#[derive(Clone, Debug, Default)]
pub struct CdefHeader {
    pub damping: u8,
    pub bits: u8,
    pub y_pri: [u8; 8],
    pub y_sec: [u8; 8],
    pub uv_pri: [u8; 8],
    pub uv_sec: [u8; 8],
}

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_WIENER: u8 = 1;
pub const RESTORE_SGRPROJ: u8 = 2;
pub const RESTORE_SWITCHABLE: u8 = 3;

#[derive(Clone, Debug, Default)]
pub struct LrHeader {
    pub kind: [u8; 3],
    pub unit_size: [usize; 3],
    pub uses_lr: bool,
}

#[derive(Clone, Debug, Default)]
pub struct FilmGrain {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub num_y_points: usize,
    pub point_y_value: [u8; 16],
    pub point_y_scaling: [u8; 16],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: usize,
    pub point_cb_value: [u8; 16],
    pub point_cb_scaling: [u8; 16],
    pub num_cr_points: usize,
    pub point_cr_value: [u8; 16],
    pub point_cr_scaling: [u8; 16],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

#[derive(Clone, Debug, Default)]
pub struct FrameHeader {
    pub frame_type: u8,
    pub show_frame: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub allow_high_precision_mv: bool,
    pub frame_size_override: bool,
    pub frame_width: usize,
    pub frame_height: usize,
    pub upscaled_width: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub use_superres: bool,
    pub superres_denom: usize,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub allow_intrabc: bool,
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub quant: QuantParams,
    pub seg: SegmentationParams,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefHeader,
    pub lr: LrHeader,
    pub tx_mode_select: bool,
    pub reduced_tx_set: bool,
    pub film_grain: FilmGrain,
    /// Size of the header in bytes, when parsed from an `OBU_FRAME` payload (tile data follows).
    pub header_bytes: usize,
    // --- Inter frames.
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: usize,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub current_frame_id: u32,
    pub order_hint: u32,
    /// `PRIMARY_REF_NONE` (7) or an index into `ref_frame_idx`.
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    /// Slot used by LAST_FRAME..ALTREF_FRAME (index 0..7 for refFrame - 1).
    pub ref_frame_idx: [usize; 7],
    /// `OrderHints[refFrame]` and `RefFrameSignBias[refFrame]`, indexed by reference frame 1..=7.
    pub order_hints: [u32; 8],
    pub ref_frame_sign_bias: [bool; 8],
    /// 0..=3 EIGHTTAP, SMOOTH, SHARP, BILINEAR; 4 = SWITCHABLE.
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    /// `GmType[ref]` and `gm_params[ref][..]` for ref 1..=7 (index 0 unused).
    pub gm_type: [u8; 8],
    pub gm_params: [[i32; 6]; 8],
}

pub const PRIMARY_REF_NONE: u8 = 7;
pub const SWITCHABLE: u8 = 4;
pub const GM_IDENTITY: u8 = 0;
pub const GM_TRANSLATION: u8 = 1;
pub const GM_ROTZOOM: u8 = 2;
pub const GM_AFFINE: u8 = 3;
pub const WARPEDMODEL_PREC_BITS: u32 = 16;

/// The default (identity) global motion parameters of one reference.
pub const DEFAULT_GM_PARAMS: [i32; 6] = [0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS];

/// What the frame header needs to know about a reference slot.
#[derive(Clone, Debug)]
pub struct RefHeaderInfo {
    pub valid: bool,
    pub frame_id: u32,
    pub upscaled_width: usize,
    pub frame_width: usize,
    pub frame_height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub order_hint: u32,
    pub frame_type: u8,
    pub gm_params: [[i32; 6]; 8],
    pub lf_ref_deltas: [i8; 8],
    pub lf_mode_deltas: [i8; 2],
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub film_grain: FilmGrain,
}

impl Default for RefHeaderInfo {
    fn default() -> Self {
        RefHeaderInfo {
            valid: false,
            frame_id: 0,
            upscaled_width: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            order_hint: 0,
            frame_type: KEY_FRAME,
            gm_params: [DEFAULT_GM_PARAMS; 8],
            lf_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            lf_mode_deltas: [0, 0],
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
            film_grain: FilmGrain::default(),
        }
    }
}

pub type RefHeaders = [RefHeaderInfo; 8];

impl FrameHeader {
    pub fn is_intra(&self) -> bool {
        self.frame_type == KEY_FRAME || self.frame_type == INTRA_ONLY_FRAME
    }

    pub fn seg_feature_active(&self, segment: usize, feature: usize) -> bool {
        self.seg.enabled && self.seg.feature_enabled[segment][feature]
    }

    /// `get_qidx(ignoreDeltaQ, segmentId)` of the spec.
    pub fn qidx(&self, ignore_delta_q: bool, segment: usize, current_q_index: i32) -> i32 {
        let use_delta = !ignore_delta_q && self.delta_q_present;
        if self.seg_feature_active(segment, SEG_LVL_ALT_Q) {
            let data = self.seg.feature_data[segment][SEG_LVL_ALT_Q];
            let q = if use_delta { current_q_index + data } else { self.quant.base_q_idx as i32 + data };
            q.clamp(0, 255)
        } else if use_delta {
            current_q_index
        } else {
            self.quant.base_q_idx as i32
        }
    }

    /// Parse `uncompressed_header()` of a frame that needs no reference frames (a key or intra-only frame).
    pub fn parse(seq: &SequenceHeader, r: &mut BitReader) -> Result<FrameHeader> {
        Self::parse_with_refs(seq, r, &RefHeaders::default())
    }

    /// `get_relative_dist` of order hints.
    pub fn relative_dist(seq: &SequenceHeader, a: u32, b: u32) -> i32 {
        if !seq.enable_order_hint {
            return 0;
        }
        let diff = a as i32 - b as i32;
        let m = 1i32 << (seq.order_hint_bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    /// Parse `uncompressed_header()` (specification section 5.9) given what the reference slots hold.
    pub fn parse_with_refs(seq: &SequenceHeader, r: &mut BitReader, refs: &RefHeaders) -> Result<FrameHeader> {
        let mut h = FrameHeader::default();
        let color = &seq.color;
        let id_len = seq.additional_frame_id_length + seq.delta_frame_id_length;
        let mut frame_type = KEY_FRAME;
        let mut error_resilient;
        h.primary_ref_frame = PRIMARY_REF_NONE;
        if seq.reduced_still_picture_header {
            h.show_frame = true;
            error_resilient = true;
        } else {
            if r.flag()? {
                // show_existing_frame
                h.show_existing_frame = true;
                h.frame_to_show_map_idx = r.bits(3)? as usize;
                if seq.decoder_model_info_present && !seq.equal_picture_interval {
                    r.bits(seq.frame_presentation_time_length)?;
                }
                if seq.frame_id_numbers_present {
                    r.bits(id_len)?;
                }
                let slot = &refs[h.frame_to_show_map_idx];
                h.frame_type = slot.frame_type;
                h.refresh_frame_flags = if slot.frame_type == KEY_FRAME { 0xff } else { 0 };
                if seq.film_grain_params_present {
                    h.film_grain = slot.film_grain.clone();
                }
                return Ok(h);
            }
            frame_type = r.bits(2)? as u8;
            h.show_frame = r.flag()?;
            if h.show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
                r.bits(seq.frame_presentation_time_length)?;
            }
            h.showable_frame = if h.show_frame { frame_type != KEY_FRAME } else { r.flag()? };
            error_resilient = frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && h.show_frame);
            if !error_resilient {
                error_resilient = r.flag()?;
            }
        }
        h.error_resilient_mode = error_resilient;
        h.frame_type = frame_type;
        let frame_is_intra = frame_type == KEY_FRAME || frame_type == INTRA_ONLY_FRAME;
        h.disable_cdf_update = r.flag()?;
        h.allow_screen_content_tools = if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        if h.allow_screen_content_tools {
            h.force_integer_mv =
                if seq.seq_force_integer_mv == SELECT_INTEGER_MV { r.flag()? } else { seq.seq_force_integer_mv != 0 };
        }
        if frame_is_intra {
            h.force_integer_mv = true;
        }
        if seq.frame_id_numbers_present {
            h.current_frame_id = r.bits(id_len)?;
        }
        h.frame_size_override = if frame_type == SWITCH_FRAME {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };
        if seq.order_hint_bits > 0 {
            h.order_hint = r.bits(seq.order_hint_bits)?;
        }
        if !(frame_is_intra || error_resilient) {
            h.primary_ref_frame = r.bits(3)? as u8;
        }
        if seq.decoder_model_info_present {
            let present = r.flag()?;
            if present {
                for op in &seq.operating_points {
                    if op.decoder_model_present {
                        // Frames in this crate come from single-layer streams; the temporal/spatial layer test
                        // would need the OBU extension, which only matters for layered streams.
                        let in_layer = op.idc == 0 || (op.idc & 1) != 0 && ((op.idc >> 8) & 1) != 0;
                        if in_layer {
                            r.bits(seq.buffer_removal_time_length)?;
                        }
                    }
                }
            }
        }
        let all_frames = 0xff;
        h.refresh_frame_flags = if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && h.show_frame) {
            all_frames
        } else {
            r.bits(8)? as u8
        };
        if (!frame_is_intra || h.refresh_frame_flags != all_frames) && error_resilient && seq.enable_order_hint {
            for slot in refs.iter() {
                let hint = r.bits(seq.order_hint_bits)?;
                if hint != slot.order_hint {
                    return Err(Error::unsupported("reference frames lost (order hint mismatch)"));
                }
            }
        }
        // Loaded from the primary reference frame (load_previous) or defaults (setup_past_independence).
        let mut prev = RefHeaderInfo::default();
        if frame_is_intra {
            read_frame_size(seq, &mut h, r)?;
            read_render_size(&mut h, r)?;
            if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
                h.allow_intrabc = r.flag()?;
            }
        } else {
            let mut short_signaling = false;
            if seq.enable_order_hint {
                short_signaling = r.flag()?;
                if short_signaling {
                    let last = r.bits(3)? as usize;
                    let gold = r.bits(3)? as usize;
                    h.ref_frame_idx = set_frame_refs(seq, h.order_hint, refs, last, gold);
                }
            }
            for i in 0..7 {
                if !short_signaling {
                    h.ref_frame_idx[i] = r.bits(3)? as usize;
                }
                if seq.frame_id_numbers_present {
                    r.bits(seq.delta_frame_id_length)?;
                }
            }
            if h.ref_frame_idx.iter().any(|&i| !refs[i].valid) {
                return Err(Error::invalid("inter frame refers to a missing reference frame"));
            }
            if h.frame_size_override && !error_resilient {
                // frame_size_with_refs()
                let mut found = false;
                for i in 0..7 {
                    if r.flag()? {
                        let slot = &refs[h.ref_frame_idx[i]];
                        h.upscaled_width = slot.upscaled_width;
                        h.frame_width = h.upscaled_width;
                        h.frame_height = slot.frame_height;
                        h.render_width = slot.render_width;
                        h.render_height = slot.render_height;
                        found = true;
                        break;
                    }
                }
                if found {
                    read_superres(seq, &mut h, r)?;
                    compute_image_size(&mut h);
                } else {
                    read_frame_size(seq, &mut h, r)?;
                    read_render_size(&mut h, r)?;
                }
            } else {
                read_frame_size(seq, &mut h, r)?;
                read_render_size(&mut h, r)?;
            }
            h.allow_high_precision_mv = if h.force_integer_mv { false } else { r.flag()? };
            // read_interpolation_filter()
            h.interpolation_filter = if r.flag()? { SWITCHABLE } else { r.bits(2)? as u8 };
            h.is_motion_mode_switchable = r.flag()?;
            h.use_ref_frame_mvs = if error_resilient || !seq.enable_ref_frame_mvs { false } else { r.flag()? };
            for i in 0..7 {
                let hint = refs[h.ref_frame_idx[i]].order_hint;
                h.order_hints[i + 1] = hint;
                h.ref_frame_sign_bias[i + 1] =
                    seq.enable_order_hint && Self::relative_dist(seq, hint, h.order_hint) > 0;
            }
            if h.primary_ref_frame != PRIMARY_REF_NONE {
                prev = refs[h.ref_frame_idx[h.primary_ref_frame as usize]].clone();
            }
        }
        h.disable_frame_end_update_cdf =
            if seq.reduced_still_picture_header || h.disable_cdf_update { true } else { r.flag()? };
        h.tile_info = parse_tile_info(seq, &h, r)?;
        h.quant = parse_quant(seq, r)?;
        h.seg = parse_segmentation(r, h.primary_ref_frame == PRIMARY_REF_NONE, &prev)?;
        // delta_q_params, delta_lf_params
        if h.quant.base_q_idx > 0 {
            h.delta_q_present = r.flag()?;
        }
        if h.delta_q_present {
            h.delta_q_res = r.bits(2)? as u8;
            if !h.allow_intrabc {
                h.delta_lf_present = r.flag()?;
            }
            if h.delta_lf_present {
                h.delta_lf_res = r.bits(2)? as u8;
                h.delta_lf_multi = r.flag()?;
            }
        }
        h.coded_lossless = true;
        for seg in 0..MAX_SEGMENTS {
            let q = h.qidx(true, seg, 0);
            let lossless = q == 0
                && h.quant.delta_q_y_dc == 0
                && h.quant.delta_q_u_ac == 0
                && h.quant.delta_q_u_dc == 0
                && h.quant.delta_q_v_ac == 0
                && h.quant.delta_q_v_dc == 0;
            h.lossless_array[seg] = lossless;
            if !lossless {
                h.coded_lossless = false;
            }
            if h.quant.using_qmatrix {
                if lossless {
                    h.seg_qm_level[0][seg] = 15;
                    h.seg_qm_level[1][seg] = 15;
                    h.seg_qm_level[2][seg] = 15;
                } else {
                    h.seg_qm_level[0][seg] = h.quant.qm_y;
                    h.seg_qm_level[1][seg] = h.quant.qm_u;
                    h.seg_qm_level[2][seg] = h.quant.qm_v;
                }
            }
        }
        h.all_lossless = h.coded_lossless && h.frame_width == h.upscaled_width;
        parse_loop_filter(&mut h, color, r, &prev)?;
        parse_cdef(&mut h, seq, r)?;
        parse_lr(&mut h, seq, r)?;
        h.tx_mode_select = if h.coded_lossless { false } else { r.flag()? };
        h.reference_select = if frame_is_intra { false } else { r.flag()? };
        parse_skip_mode(seq, &mut h);
        if h.skip_mode_present_allowed() {
            h.skip_mode_present = r.flag()?;
        }
        h.allow_warped_motion =
            if frame_is_intra || error_resilient || !seq.enable_warped_motion { false } else { r.flag()? };
        h.reduced_tx_set = r.flag()?;
        parse_global_motion(&mut h, r, &prev)?;
        h.film_grain = parse_film_grain(seq, &h, r, refs)?;
        Ok(h)
    }

    /// Whether `skip_mode_present` is coded (the result of the derivation in `skip_mode_params`).
    fn skip_mode_present_allowed(&self) -> bool {
        self.skip_mode_frame[0] != 0
    }
}

fn read_frame_size(seq: &SequenceHeader, h: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
    if h.frame_size_override {
        h.frame_width = r.bits(seq.frame_width_bits)? as usize + 1;
        h.frame_height = r.bits(seq.frame_height_bits)? as usize + 1;
    } else {
        h.frame_width = seq.max_frame_width as usize;
        h.frame_height = seq.max_frame_height as usize;
    }
    read_superres(seq, h, r)?;
    compute_image_size(h);
    Ok(())
}

fn read_superres(seq: &SequenceHeader, h: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
    h.superres_denom = 8;
    h.use_superres = false;
    if seq.enable_superres && r.flag()? {
        h.use_superres = true;
        h.superres_denom = r.bits(3)? as usize + 9;
    }
    h.upscaled_width = h.frame_width;
    h.frame_width = (h.upscaled_width * 8 + h.superres_denom / 2) / h.superres_denom;
    Ok(())
}

fn compute_image_size(h: &mut FrameHeader) {
    h.mi_cols = 2 * h.frame_width.div_ceil(8);
    h.mi_rows = 2 * h.frame_height.div_ceil(8);
}

fn read_render_size(h: &mut FrameHeader, r: &mut BitReader) -> Result<()> {
    if r.flag()? {
        h.render_width = r.bits(16)? as usize + 1;
        h.render_height = r.bits(16)? as usize + 1;
    } else {
        h.render_width = h.upscaled_width;
        h.render_height = h.frame_height;
    }
    Ok(())
}

/// The set-frame-refs process (7.8): derive `ref_frame_idx` from the two explicitly coded references.
fn set_frame_refs(seq: &SequenceHeader, order_hint: u32, refs: &RefHeaders, last: usize, gold: usize) -> [usize; 7] {
    let mut idx = [-1i32; 7];
    idx[0] = last as i32;
    idx[3] = gold as i32;
    let mut used = [false; 8];
    used[last] = true;
    used[gold] = true;
    let cur = 1i32 << (seq.order_hint_bits - 1);
    let shifted: Vec<i32> =
        refs.iter().map(|s| cur + FrameHeader::relative_dist(seq, s.order_hint, order_hint)).collect();
    // find_latest_backward / find_earliest_backward / find_latest_forward
    let latest_backward = |used: &[bool; 8]| {
        let mut r: Option<usize> = None;
        for i in 0..8 {
            if !used[i] && shifted[i] >= cur && r.is_none_or(|b| shifted[i] >= shifted[b]) {
                r = Some(i);
            }
        }
        r
    };
    let earliest_backward = |used: &[bool; 8]| {
        let mut r: Option<usize> = None;
        for i in 0..8 {
            if !used[i] && shifted[i] >= cur && r.is_none_or(|b| shifted[i] < shifted[b]) {
                r = Some(i);
            }
        }
        r
    };
    let latest_forward = |used: &[bool; 8]| {
        let mut r: Option<usize> = None;
        for i in 0..8 {
            if !used[i] && shifted[i] < cur && r.is_none_or(|b| shifted[i] >= shifted[b]) {
                r = Some(i);
            }
        }
        r
    };
    if let Some(r) = latest_backward(&used) {
        idx[6] = r as i32;
        used[r] = true;
    }
    if let Some(r) = earliest_backward(&used) {
        idx[4] = r as i32;
        used[r] = true;
    }
    if let Some(r) = earliest_backward(&used) {
        idx[5] = r as i32;
        used[r] = true;
    }
    // Ref_Frame_List: LAST2, LAST3, BWDREF, ALTREF2, ALTREF
    for ref_frame in [2usize, 3, 5, 6, 7] {
        if idx[ref_frame - 1] < 0
            && let Some(r) = latest_forward(&used)
        {
            idx[ref_frame - 1] = r as i32;
            used[r] = true;
        }
    }
    let mut earliest: Option<usize> = None;
    for i in 0..8 {
        if earliest.is_none_or(|b| shifted[i] < shifted[b]) {
            earliest = Some(i);
        }
    }
    let fallback = earliest.unwrap_or(0) as i32;
    let mut out = [0usize; 7];
    for i in 0..7 {
        out[i] = if idx[i] < 0 { fallback } else { idx[i] } as usize;
    }
    out
}

/// `skip_mode_params`: finds the two references skip mode uses; leaves `skip_mode_frame[0] == 0` when it is not
/// allowed.
fn parse_skip_mode(seq: &SequenceHeader, h: &mut FrameHeader) {
    h.skip_mode_frame = [0, 0];
    if h.is_intra() || !h.reference_select || !seq.enable_order_hint {
        return;
    }
    let dist = |a: u32, b: u32| FrameHeader::relative_dist(seq, a, b);
    let hints: Vec<u32> = (0..7).map(|i| h.order_hints[i + 1]).collect();
    let (mut forward, mut backward): (Option<usize>, Option<usize>) = (None, None);
    for (i, &hint) in hints.iter().enumerate() {
        if dist(hint, h.order_hint) < 0 {
            if forward.is_none_or(|f| dist(hint, hints[f]) > 0) {
                forward = Some(i);
            }
        } else if dist(hint, h.order_hint) > 0 && backward.is_none_or(|b| dist(hint, hints[b]) < 0) {
            backward = Some(i);
        }
    }
    let Some(fwd) = forward else { return };
    if let Some(bwd) = backward {
        h.skip_mode_frame = [(1 + fwd.min(bwd)) as u8, (1 + fwd.max(bwd)) as u8];
    } else {
        let mut second: Option<usize> = None;
        for (i, &hint) in hints.iter().enumerate() {
            if dist(hint, hints[fwd]) < 0 && second.is_none_or(|s| dist(hint, hints[s]) > 0) {
                second = Some(i);
            }
        }
        if let Some(sec) = second {
            h.skip_mode_frame = [(1 + fwd.min(sec)) as u8, (1 + fwd.max(sec)) as u8];
        }
    }
}

fn parse_global_motion(h: &mut FrameHeader, r: &mut BitReader, prev: &RefHeaderInfo) -> Result<()> {
    h.gm_params = [DEFAULT_GM_PARAMS; 8];
    h.gm_type = [GM_IDENTITY; 8];
    if h.is_intra() {
        return Ok(());
    }
    // PrevGmParams: the primary reference frame's, or the defaults.
    let prev_gm = &prev.gm_params;
    for rf in 1..8 {
        let ty = if r.flag()? {
            if r.flag()? {
                GM_ROTZOOM
            } else if r.flag()? {
                GM_TRANSLATION
            } else {
                GM_AFFINE
            }
        } else {
            GM_IDENTITY
        };
        h.gm_type[rf] = ty;
        if ty >= GM_ROTZOOM {
            read_global_param(h, r, prev_gm, ty, rf, 2)?;
            read_global_param(h, r, prev_gm, ty, rf, 3)?;
            if ty == GM_AFFINE {
                read_global_param(h, r, prev_gm, ty, rf, 4)?;
                read_global_param(h, r, prev_gm, ty, rf, 5)?;
            } else {
                h.gm_params[rf][4] = -h.gm_params[rf][3];
                h.gm_params[rf][5] = h.gm_params[rf][2];
            }
        }
        if ty >= GM_TRANSLATION {
            read_global_param(h, r, prev_gm, ty, rf, 0)?;
            read_global_param(h, r, prev_gm, ty, rf, 1)?;
        }
    }
    Ok(())
}

fn read_global_param(
    h: &mut FrameHeader,
    r: &mut BitReader,
    prev_gm: &[[i32; 6]; 8],
    ty: u8,
    rf: usize,
    idx: usize,
) -> Result<()> {
    let (mut abs_bits, mut prec_bits) = (12u32, 15u32);
    if idx < 2 {
        if ty == GM_TRANSLATION {
            abs_bits = 9 - !h.allow_high_precision_mv as u32;
            prec_bits = 3 - !h.allow_high_precision_mv as u32;
        } else {
            abs_bits = 12;
            prec_bits = 6;
        }
    }
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let round = if idx % 3 == 2 { 1i32 << WARPEDMODEL_PREC_BITS } else { 0 };
    let sub = if idx % 3 == 2 { 1i32 << prec_bits } else { 0 };
    let mx = 1i32 << abs_bits;
    let rr = (prev_gm[rf][idx] >> prec_diff) - sub;
    let v = decode_signed_subexp_with_ref(r, -mx, mx + 1, rr)?;
    h.gm_params[rf][idx] = (v << prec_diff) + round;
    Ok(())
}

fn decode_signed_subexp_with_ref(r: &mut BitReader, low: i32, high: i32, rr: i32) -> Result<i32> {
    let x = decode_unsigned_subexp_with_ref(r, high - low, rr - low)?;
    Ok(x + low)
}

fn decode_unsigned_subexp_with_ref(r: &mut BitReader, mx: i32, rr: i32) -> Result<i32> {
    let v = decode_subexp(r, mx)?;
    Ok(if (rr << 1) <= mx { inverse_recenter(rr, v) } else { mx - 1 - inverse_recenter(mx - 1 - rr, v) })
}

fn decode_subexp(r: &mut BitReader, num_syms: i32) -> Result<i32> {
    let (mut i, mut mk, k) = (0u32, 0i32, 3u32);
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1i32 << b2;
        if num_syms <= mk + 3 * a {
            return Ok(r.ns((num_syms - mk) as u32)? as i32 + mk);
        }
        if r.flag()? {
            i += 1;
            mk += a;
        } else {
            return Ok(r.bits(b2)? as i32 + mk);
        }
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

fn tile_log2(blk: usize, target: usize) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

fn parse_tile_info(seq: &SequenceHeader, h: &FrameHeader, r: &mut BitReader) -> Result<TileInfo> {
    let sb_cols = if seq.use_128x128_superblock { (h.mi_cols + 31) >> 5 } else { (h.mi_cols + 15) >> 4 };
    let sb_rows = if seq.use_128x128_superblock { (h.mi_rows + 31) >> 5 } else { (h.mi_rows + 15) >> 4 };
    let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = 4096 >> sb_size;
    let mut max_tile_area_sb = (4096 * 2304) >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    let mut t = TileInfo::default();
    let uniform = r.flag()?;
    if uniform {
        t.cols_log2 = min_log2_tile_cols;
        while t.cols_log2 < max_log2_tile_cols {
            if r.flag()? {
                t.cols_log2 += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1 << t.cols_log2) - 1) >> t.cols_log2;
        let mut start = 0;
        while start < sb_cols {
            t.mi_col_starts.push(start << sb_shift);
            start += tile_width_sb;
        }
        t.cols = t.mi_col_starts.len();
        t.mi_col_starts.push(h.mi_cols);
        let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.cols_log2);
        t.rows_log2 = min_log2_tile_rows;
        while t.rows_log2 < max_log2_tile_rows {
            if r.flag()? {
                t.rows_log2 += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1 << t.rows_log2) - 1) >> t.rows_log2;
        let mut start = 0;
        while start < sb_rows {
            t.mi_row_starts.push(start << sb_shift);
            start += tile_height_sb;
        }
        t.rows = t.mi_row_starts.len();
        t.mi_row_starts.push(h.mi_rows);
    } else {
        let mut widest = 0;
        let mut start = 0;
        while start < sb_cols {
            t.mi_col_starts.push(start << sb_shift);
            let max_width = (sb_cols - start).min(max_tile_width_sb);
            let size = r.ns(max_width as u32)? as usize + 1;
            widest = widest.max(size);
            start += size;
        }
        t.cols = t.mi_col_starts.len();
        t.mi_col_starts.push(h.mi_cols);
        t.cols_log2 = tile_log2(1, t.cols);
        max_tile_area_sb =
            if min_log2_tiles > 0 { (sb_rows * sb_cols) >> (min_log2_tiles + 1) } else { sb_rows * sb_cols };
        let max_tile_height_sb = (max_tile_area_sb / widest).max(1);
        let mut start = 0;
        while start < sb_rows {
            t.mi_row_starts.push(start << sb_shift);
            let max_height = (sb_rows - start).min(max_tile_height_sb);
            let size = r.ns(max_height as u32)? as usize + 1;
            start += size;
        }
        t.rows = t.mi_row_starts.len();
        t.mi_row_starts.push(h.mi_rows);
        t.rows_log2 = tile_log2(1, t.rows);
    }
    if t.cols_log2 > 0 || t.rows_log2 > 0 {
        t.context_update_tile_id = r.bits(t.rows_log2 + t.cols_log2)? as usize;
        t.tile_size_bytes = r.bits(2)? as usize + 1;
        if t.context_update_tile_id >= t.cols * t.rows {
            return Err(Error::invalid("bad context_update_tile_id"));
        }
    }
    Ok(t)
}

fn read_delta_q(r: &mut BitReader) -> Result<i32> {
    if r.flag()? { r.signed(7) } else { Ok(0) }
}

fn parse_quant(seq: &SequenceHeader, r: &mut BitReader) -> Result<QuantParams> {
    let c = &seq.color;
    let mut q = QuantParams { base_q_idx: r.bits(8)? as u8, ..Default::default() };
    q.delta_q_y_dc = read_delta_q(r)?;
    if c.num_planes() > 1 {
        let diff_uv_delta = if c.separate_uv_delta_q { r.flag()? } else { false };
        q.delta_q_u_dc = read_delta_q(r)?;
        q.delta_q_u_ac = read_delta_q(r)?;
        if diff_uv_delta {
            q.delta_q_v_dc = read_delta_q(r)?;
            q.delta_q_v_ac = read_delta_q(r)?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = r.flag()?;
    if q.using_qmatrix {
        q.qm_y = r.bits(4)? as u8;
        q.qm_u = r.bits(4)? as u8;
        q.qm_v = if !c.separate_uv_delta_q { q.qm_u } else { r.bits(4)? as u8 };
    }
    Ok(q)
}

fn parse_segmentation(r: &mut BitReader, primary_none: bool, prev: &RefHeaderInfo) -> Result<SegmentationParams> {
    let mut s = SegmentationParams { enabled: r.flag()?, ..Default::default() };
    if s.enabled {
        if primary_none {
            s.update_map = true;
            s.update_data = true;
        } else {
            s.update_map = r.flag()?;
            if s.update_map {
                s.temporal_update = r.flag()?;
            }
            s.update_data = r.flag()?;
        }
        if !s.update_data {
            // The features of the primary reference frame stay in force.
            s.feature_enabled = prev.feature_enabled;
            s.feature_data = prev.feature_data;
        }
        for i in 0..MAX_SEGMENTS {
            if !s.update_data {
                break;
            }
            for j in 0..SEG_LVL_MAX {
                let enabled = r.flag()?;
                s.feature_enabled[i][j] = enabled;
                let mut clipped = 0;
                if enabled {
                    let bits = SEGMENTATION_FEATURE_BITS[j];
                    let limit = SEGMENTATION_FEATURE_MAX[j];
                    if SEGMENTATION_FEATURE_SIGNED[j] {
                        clipped = r.signed(1 + bits)?.clamp(-limit, limit);
                    } else {
                        clipped = (r.bits(bits)? as i32).clamp(0, limit);
                    }
                }
                s.feature_data[i][j] = clipped;
            }
        }
    }
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if s.feature_enabled[i][j] {
                s.last_active_seg_id = i as u8;
                if j >= SEG_LVL_REF_FRAME {
                    s.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(s)
}

fn parse_loop_filter(h: &mut FrameHeader, color: &ColorConfig, r: &mut BitReader, prev: &RefHeaderInfo) -> Result<()> {
    let lf = &mut h.loop_filter;
    lf.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
    lf.mode_deltas = [0, 0];
    if h.coded_lossless || h.allow_intrabc {
        return Ok(());
    }
    if h.primary_ref_frame != PRIMARY_REF_NONE {
        lf.ref_deltas = prev.lf_ref_deltas;
        lf.mode_deltas = prev.lf_mode_deltas;
    }
    lf.level[0] = r.bits(6)? as u8;
    lf.level[1] = r.bits(6)? as u8;
    if color.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = r.bits(6)? as u8;
        lf.level[3] = r.bits(6)? as u8;
    }
    lf.sharpness = r.bits(3)? as u8;
    lf.delta_enabled = r.flag()?;
    if lf.delta_enabled && r.flag()? {
        for i in 0..8 {
            if r.flag()? {
                lf.ref_deltas[i] = r.signed(7)? as i8;
            }
        }
        for i in 0..2 {
            if r.flag()? {
                lf.mode_deltas[i] = r.signed(7)? as i8;
            }
        }
    }
    Ok(())
}

fn parse_cdef(h: &mut FrameHeader, seq: &SequenceHeader, r: &mut BitReader) -> Result<()> {
    let c = &mut h.cdef;
    c.damping = 3;
    if h.coded_lossless || h.allow_intrabc || !seq.enable_cdef {
        return Ok(());
    }
    c.damping = r.bits(2)? as u8 + 3;
    c.bits = r.bits(2)? as u8;
    for i in 0..(1usize << c.bits) {
        c.y_pri[i] = r.bits(4)? as u8;
        c.y_sec[i] = r.bits(2)? as u8;
        if c.y_sec[i] == 3 {
            c.y_sec[i] += 1;
        }
        if seq.color.num_planes() > 1 {
            c.uv_pri[i] = r.bits(4)? as u8;
            c.uv_sec[i] = r.bits(2)? as u8;
            if c.uv_sec[i] == 3 {
                c.uv_sec[i] += 1;
            }
        }
    }
    Ok(())
}

fn parse_lr(h: &mut FrameHeader, seq: &SequenceHeader, r: &mut BitReader) -> Result<()> {
    if h.all_lossless || h.allow_intrabc || !seq.enable_restoration {
        return Ok(());
    }
    const REMAP: [u8; 4] = [RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ];
    let planes = seq.color.num_planes();
    let mut uses_chroma_lr = false;
    for i in 0..planes {
        h.lr.kind[i] = REMAP[r.bits(2)? as usize];
        if h.lr.kind[i] != RESTORE_NONE {
            h.lr.uses_lr = true;
            if i > 0 {
                uses_chroma_lr = true;
            }
        }
    }
    if h.lr.uses_lr {
        let mut shift;
        if seq.use_128x128_superblock {
            shift = r.bits(1)? + 1;
        } else {
            shift = r.bits(1)?;
            if shift != 0 {
                shift += r.bits(1)?;
            }
        }
        h.lr.unit_size[0] = 256 >> (2 - shift);
        let uv_shift = if seq.color.sub_x != 0 && seq.color.sub_y != 0 && uses_chroma_lr { r.bits(1)? } else { 0 };
        h.lr.unit_size[1] = h.lr.unit_size[0] >> uv_shift;
        h.lr.unit_size[2] = h.lr.unit_size[0] >> uv_shift;
    }
    Ok(())
}

fn parse_film_grain(seq: &SequenceHeader, h: &FrameHeader, r: &mut BitReader, refs: &RefHeaders) -> Result<FilmGrain> {
    let mut g = FilmGrain::default();
    if !seq.film_grain_params_present || (!h.show_frame && !h.showable_frame) {
        return Ok(g);
    }
    g.apply_grain = r.flag()?;
    if !g.apply_grain {
        return Ok(g);
    }
    g.grain_seed = r.bits(16)? as u16;
    let update_grain = if h.frame_type == INTER_FRAME { r.flag()? } else { true };
    if !update_grain {
        let idx = r.bits(3)? as usize;
        let seed = g.grain_seed;
        g = refs[idx].film_grain.clone();
        g.grain_seed = seed;
        return Ok(g);
    }
    g.num_y_points = r.bits(4)? as usize;
    if g.num_y_points > 14 {
        return Err(Error::invalid("too many film grain points"));
    }
    for i in 0..g.num_y_points {
        g.point_y_value[i] = r.bits(8)? as u8;
        g.point_y_scaling[i] = r.bits(8)? as u8;
    }
    let mono = seq.color.mono_chrome;
    g.chroma_scaling_from_luma = if mono { false } else { r.flag()? };
    if mono || g.chroma_scaling_from_luma || (seq.color.sub_x == 1 && seq.color.sub_y == 1 && g.num_y_points == 0) {
        g.num_cb_points = 0;
        g.num_cr_points = 0;
    } else {
        g.num_cb_points = r.bits(4)? as usize;
        if g.num_cb_points > 10 {
            return Err(Error::invalid("too many film grain points"));
        }
        for i in 0..g.num_cb_points {
            g.point_cb_value[i] = r.bits(8)? as u8;
            g.point_cb_scaling[i] = r.bits(8)? as u8;
        }
        g.num_cr_points = r.bits(4)? as usize;
        if g.num_cr_points > 10 {
            return Err(Error::invalid("too many film grain points"));
        }
        for i in 0..g.num_cr_points {
            g.point_cr_value[i] = r.bits(8)? as u8;
            g.point_cr_scaling[i] = r.bits(8)? as u8;
        }
    }
    g.grain_scaling_minus_8 = r.bits(2)? as u8;
    g.ar_coeff_lag = r.bits(2)? as u8;
    let lag = g.ar_coeff_lag as usize;
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma;
    if g.num_y_points > 0 {
        num_pos_chroma = num_pos_luma + 1;
        for i in 0..num_pos_luma {
            g.ar_coeffs_y_plus_128[i] = r.bits(8)? as u8;
        }
    } else {
        num_pos_chroma = num_pos_luma;
    }
    if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cb_plus_128[i] = r.bits(8)? as u8;
        }
    }
    if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cr_plus_128[i] = r.bits(8)? as u8;
        }
    }
    g.ar_coeff_shift_minus_6 = r.bits(2)? as u8;
    g.grain_scale_shift = r.bits(2)? as u8;
    if g.num_cb_points > 0 {
        g.cb_mult = r.bits(8)? as u8;
        g.cb_luma_mult = r.bits(8)? as u8;
        g.cb_offset = r.bits(9)? as u16;
    }
    if g.num_cr_points > 0 {
        g.cr_mult = r.bits(8)? as u8;
        g.cr_luma_mult = r.bits(8)? as u8;
        g.cr_offset = r.bits(9)? as u16;
    }
    g.overlap_flag = r.flag()?;
    g.clip_to_restricted_range = r.flag()?;
    Ok(g)
}
