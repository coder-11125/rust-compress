//! Reference frames for inter prediction: what a decoded frame leaves behind for the frames after it.

use super::cdf_tables::Cdfs;
use super::frame::Frame;
use super::obu::RefHeaderInfo;
use std::sync::Arc;

/// Everything stored for one reference slot (specification section 7.20).
#[derive(Clone, Default)]
pub struct RefSlot {
    pub info: RefHeaderInfo,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub bit_depth: u8,
    pub sub_x: u8,
    pub sub_y: u8,
    pub showable_frame: bool,
    /// `SavedOrderHints[slot][refFrame]`.
    pub saved_order_hints: [u32; 8],
    /// The frame as output: after the loop filters and super resolution, before film grain.
    pub frame: Option<Arc<Frame>>,
    /// `SavedRefFrames` / `SavedMvs` at 8x8 granularity (only the odd 4x4 positions matter), `MiRows/2 x MiCols/2`.
    pub mf_ref_frames: Option<Arc<Vec<i8>>>,
    pub mf_mvs: Option<Arc<Vec<[i16; 2]>>>,
    pub segment_ids: Option<Arc<Vec<u8>>>,
    pub cdfs: Option<Arc<Cdfs>>,
}

/// One reference as the frame being decoded sees it.
#[derive(Clone)]
pub struct RefView {
    pub frame: Arc<Frame>,
    pub upscaled_width: usize,
    pub frame_width: usize,
    pub frame_height: usize,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub order_hint: u32,
    pub frame_type: u8,
    pub saved_order_hints: [u32; 8],
    pub mf_ref_frames: Option<Arc<Vec<i8>>>,
    pub mf_mvs: Option<Arc<Vec<[i16; 2]>>>,
}

/// The seven references (index `refFrame`, 1 = LAST .. 7 = ALTREF) of the frame being decoded.
#[derive(Clone, Default)]
pub struct RefSet {
    pub views: [Option<RefView>; 8],
}

impl RefSet {
    pub fn get(&self, ref_frame: i8) -> Option<&RefView> {
        self.views.get(ref_frame as usize).and_then(|v| v.as_ref())
    }
}
