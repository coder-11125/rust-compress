//! Sample buffers shared by the AV1 decoder, encoder and post-filters.

/// One colour plane. Samples are 16-bit whatever the bit depth (8, 10 or 12).
///
/// `width` x `height` is the *coded* area (a multiple of the 4x4 mode-info unit, so it can be larger
/// than the visible picture); `stride >= width`.
#[derive(Clone, Debug, Default)]
pub struct Plane {
    pub data: Vec<u16>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Plane {
        Plane { data: vec![0; width * height], stride: width, width, height }
    }

    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.stride + x]
    }

    #[inline(always)]
    pub fn row(&self, y: usize) -> &[u16] {
        &self.data[y * self.stride..y * self.stride + self.width]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, y: usize) -> &mut [u16] {
        let w = self.width;
        &mut self.data[y * self.stride..y * self.stride + w]
    }
}

/// A picture: one plane (monochrome) or three (Y, U, V).
#[derive(Clone, Debug)]
pub struct Frame {
    pub planes: Vec<Plane>,
    pub bit_depth: u8,
    /// Chroma subsampling shifts (1 = halved). Both 0 for 4:4:4, both 1 for 4:2:0, (1, 0) for 4:2:2.
    pub sub_x: u8,
    pub sub_y: u8,
    /// Visible size in luma samples.
    pub width: usize,
    pub height: usize,
}
