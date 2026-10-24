//! Motion field estimation (specification section 7.9): the motion vectors saved with the reference frames are
//! projected onto the current frame, giving temporal motion vector candidates for every 8x8 block.

use super::obu::{FrameHeader, INTRA_ONLY_FRAME, KEY_FRAME, SequenceHeader};
use super::refs::RefSet;

pub const INVALID_MV: i16 = i16::MIN;
const MAX_FRAME_DISTANCE: i32 = 31;
const MFMV_STACK_SIZE: i32 = 3;
const MAX_OFFSET_WIDTH: i32 = 8;
const MAX_OFFSET_HEIGHT: i32 = 0;

const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092, 1024, 963, 910, 862,
    819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// `get_mv_projection` (7.9.3).
pub fn mv_projection(mv: [i32; 2], numerator: i32, denominator: i32) -> [i32; 2] {
    let den = denominator.min(MAX_FRAME_DISTANCE);
    let num = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let mut out = [0; 2];
    for i in 0..2 {
        let v = mv[i] * num * DIV_MULT[den as usize];
        let scaled = if v >= 0 { (v + (1 << 13)) >> 14 } else { -((-v + (1 << 13)) >> 14) };
        out[i] = scaled.clamp(-(1 << 14) + 1, (1 << 14) - 1);
    }
    out
}

/// `MotionFieldMvs[ref][y8 * (MiCols >> 1) + x8]` for every reference frame (index 0 unused).
pub fn estimate(seq: &SequenceHeader, h: &FrameHeader, refs: &RefSet) -> Vec<Vec<[i16; 2]>> {
    let (w8, h8) = ((h.mi_cols >> 1) as i32, (h.mi_rows >> 1) as i32);
    let mut field = vec![vec![[INVALID_MV; 2]; (w8 * h8) as usize]; 8];
    let rel = |a: u32, b: u32| FrameHeader::relative_dist(seq, a, b);
    let last_alt_hint = refs.get(1).map_or(0, |v| v.saved_order_hints[7]);
    if last_alt_hint != h.order_hints[4] {
        project(seq, h, refs, 1, -1, &mut field);
    }
    let mut ref_stamp = MFMV_STACK_SIZE - 2;
    if rel(h.order_hints[5], h.order_hint) > 0 && project(seq, h, refs, 5, 1, &mut field) {
        ref_stamp -= 1;
    }
    if rel(h.order_hints[6], h.order_hint) > 0 && project(seq, h, refs, 6, 1, &mut field) {
        ref_stamp -= 1;
    }
    if rel(h.order_hints[7], h.order_hint) > 0 && ref_stamp >= 0 && project(seq, h, refs, 7, 1, &mut field) {
        ref_stamp -= 1;
    }
    if ref_stamp >= 0 {
        project(seq, h, refs, 2, -1, &mut field);
    }
    field
}

/// The projection process (7.9.2): returns whether the source frame could be used.
fn project(
    seq: &SequenceHeader,
    h: &FrameHeader,
    refs: &RefSet,
    src: usize,
    dst_sign: i32,
    field: &mut [Vec<[i16; 2]>],
) -> bool {
    let Some(view) = refs.get(src as i8) else { return false };
    let (w8, h8) = ((h.mi_cols >> 1) as i32, (h.mi_rows >> 1) as i32);
    if view.mi_rows != h.mi_rows
        || view.mi_cols != h.mi_cols
        || view.frame_type == INTRA_ONLY_FRAME
        || view.frame_type == KEY_FRAME
    {
        return false;
    }
    let (Some(saved_refs), Some(saved_mvs)) = (&view.mf_ref_frames, &view.mf_mvs) else { return false };
    let rel = |a: u32, b: u32| FrameHeader::relative_dist(seq, a, b);
    let ref_to_cur = rel(h.order_hints[src], h.order_hint);
    for y8 in 0..h8 {
        for x8 in 0..w8 {
            let i = (y8 * w8 + x8) as usize;
            let src_ref = saved_refs[i];
            if src_ref <= 0 {
                continue;
            }
            let ref_offset = rel(h.order_hints[src], view.saved_order_hints[src_ref as usize]);
            if ref_to_cur.abs() > MAX_FRAME_DISTANCE || ref_offset.abs() > MAX_FRAME_DISTANCE || ref_offset <= 0 {
                continue;
            }
            let mv = [saved_mvs[i][0] as i32, saved_mvs[i][1] as i32];
            let proj = mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
            let (Some(py), Some(px)) = (
                position(y8, proj[0], dst_sign, h8, MAX_OFFSET_HEIGHT),
                position(x8, proj[1], dst_sign, w8, MAX_OFFSET_WIDTH),
            ) else {
                continue;
            };
            for dst in 1..8 {
                let ref_to_dst = rel(h.order_hint, h.order_hints[dst]);
                let p = mv_projection(mv, ref_to_dst, ref_offset);
                field[dst][(py * w8 + px) as usize] = [p[0] as i16, p[1] as i16];
            }
        }
    }
    true
}

/// `project` of the get block position process (7.9.4); `None` when the position is not valid.
fn position(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> Option<i32> {
    let base8 = (v8 >> 3) << 3;
    let offset8 = if delta >= 0 { delta >> 6 } else { -((-delta) >> 6) };
    let v = v8 + dst_sign * offset8;
    if v < 0 || v >= max8 || v < base8 - max_off8 || v >= base8 + 8 + max_off8 { None } else { Some(v) }
}
