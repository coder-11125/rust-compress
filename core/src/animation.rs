//! Animations as a list of full-canvas frames, the common form the animated decoders produce and the animated
//! AVIF encoder consumes.

use crate::image::Image;

/// Frames this long (milliseconds) or shorter are shown for [`DEFAULT_DELAY_MS`], as browsers do for GIF.
pub const MIN_DELAY_MS: u32 = 20;
pub const DEFAULT_DELAY_MS: u32 = 100;

/// Upper bound on frames times pixels held in memory at once.
pub const MAX_ANIMATION_PIXELS: u64 = 1 << 28;

#[derive(Clone, Debug)]
pub struct AnimFrame {
    /// The whole canvas as it is shown during this frame (8-bit RGB or RGBA).
    pub image: Image,
    pub duration_ms: u32,
}

#[derive(Clone, Debug)]
pub struct Animation {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<AnimFrame>,
    /// Play once (`false`) or forever (`true`). Finite repeat counts above one are played forever.
    pub loop_forever: bool,
}

impl Animation {
    pub fn has_alpha(&self) -> bool {
        self.frames.first().is_some_and(|f| f.image.color.has_alpha())
    }
}

/// A frame delay in milliseconds with the browser convention for tiny delays.
pub fn normalize_delay(ms: u32) -> u32 {
    if ms < MIN_DELAY_MS { DEFAULT_DELAY_MS } else { ms }
}
