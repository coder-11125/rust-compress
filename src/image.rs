//! In-memory raster image shared by all codecs.

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorType {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl ColorType {
    pub fn channels(self) -> usize {
        match self {
            ColorType::Gray => 1,
            ColorType::GrayAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, ColorType::GrayAlpha | ColorType::Rgba)
    }

    pub fn is_gray(self) -> bool {
        matches!(self, ColorType::Gray | ColorType::GrayAlpha)
    }
}

/// Sanity limit on decoded size, to stop hostile headers from exhausting memory.
pub const MAX_PIXELS: u64 = 150_000_000;

/// Metadata carried through recompression.
///
/// The color-management part (`icc`, `png_color`) always survives, since dropping it shifts colors.
/// The descriptive part (EXIF, XMP, comments, text chunks, ...) is collected too but discarded
/// unless the user asks for `--keep-metadata`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metadata {
    /// Raw ICC profile.
    pub icc: Option<Vec<u8>>,
    /// PNG colorimetry chunks (`sRGB`, `gAMA`, `cHRM`) as (type, body).
    pub png_color: Vec<([u8; 4], Vec<u8>)>,
    /// EXIF as a bare TIFF block; converts between a PNG `eXIf` chunk and a JPEG APP1 segment.
    pub exif: Option<Vec<u8>>,
    /// XMP packet (the XML text); converts between a PNG `iTXt` chunk and a JPEG APP1 segment.
    pub xmp: Option<Vec<u8>>,
    /// JPEG-only segments (comments, Photoshop/IPTC blocks) as (marker, body).
    pub jpeg_extra: Vec<(u8, Vec<u8>)>,
    /// PNG-only chunks (text, modification time, pixel density) as (type, body).
    pub png_extra: Vec<([u8; 4], Vec<u8>)>,
}

impl Metadata {
    /// Forget everything that is not needed to display the image correctly.
    pub fn drop_descriptive(&mut self) {
        self.exif = None;
        self.xmp = None;
        self.jpeg_extra.clear();
        self.png_extra.clear();
    }

    pub fn has_descriptive(&self) -> bool {
        self.exif.is_some() || self.xmp.is_some() || !self.jpeg_extra.is_empty() || !self.png_extra.is_empty()
    }
}

/// Interleaved samples, row-major. 16-bit samples are stored big-endian.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub color: ColorType,
    /// 8 or 16.
    pub depth: u8,
    pub data: Vec<u8>,
    pub meta: Metadata,
}

impl Image {
    pub fn new(width: u32, height: u32, color: ColorType, depth: u8, data: Vec<u8>) -> Result<Self> {
        if depth != 8 && depth != 16 {
            return Err(Error::unsupported(format!("sample depth {depth}")));
        }
        let img = Image { width, height, color, depth, data, meta: Metadata::default() };
        if img.data.len() != img.expected_len() {
            return Err(Error::invalid("pixel buffer size does not match dimensions"));
        }
        Ok(img)
    }

    pub fn bytes_per_sample(&self) -> usize {
        self.depth as usize / 8
    }

    pub fn bytes_per_pixel(&self) -> usize {
        self.color.channels() * self.bytes_per_sample()
    }

    pub fn stride(&self) -> usize {
        self.width as usize * self.bytes_per_pixel()
    }

    pub fn expected_len(&self) -> usize {
        self.stride() * self.height as usize
    }

    /// Drop the low byte of 16-bit samples (rounding to nearest).
    pub fn to_depth8(&self) -> Image {
        if self.depth == 8 {
            return self.clone();
        }
        let data = self
            .data
            .chunks_exact(2)
            .map(|s| {
                let v = u16::from_be_bytes([s[0], s[1]]) as u32;
                ((v * 255 + 32767) / 65535) as u8
            })
            .collect();
        Image { depth: 8, data, ..self.clone() }
    }

    /// Composite onto an opaque background and drop the alpha channel; 8-bit output.
    pub fn flatten(&self, bg: [u8; 3]) -> Image {
        let src = self.to_depth8();
        match src.color {
            ColorType::Gray | ColorType::Rgb => src,
            ColorType::GrayAlpha => {
                let data = src.data.chunks_exact(2).map(|p| blend(p[0], p[1], bg[0])).collect();
                Image { color: ColorType::Gray, data, ..src }
            }
            ColorType::Rgba => {
                let mut data = Vec::with_capacity(src.data.len() / 4 * 3);
                for p in src.data.chunks_exact(4) {
                    data.push(blend(p[0], p[3], bg[0]));
                    data.push(blend(p[1], p[3], bg[1]));
                    data.push(blend(p[2], p[3], bg[2]));
                }
                Image { color: ColorType::Rgb, data, ..src }
            }
        }
    }

    /// Convert to 8-bit Gray or Rgb (alpha flattened onto `bg`).
    pub fn to_opaque8(&self, bg: [u8; 3]) -> Image {
        self.flatten(bg)
    }
}

#[inline]
fn blend(c: u8, a: u8, bg: u8) -> u8 {
    let a = a as u32;
    ((c as u32 * a + bg as u32 * (255 - a) + 127) / 255) as u8
}

impl Image {
    /// Apply an EXIF orientation (1..=8) so pixels are stored upright.
    pub fn oriented(self, orientation: u8) -> Image {
        if !(2..=8).contains(&orientation) {
            return self;
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let bpp = self.bytes_per_pixel();
        let (dw, dh) = if orientation >= 5 { (h, w) } else { (w, h) };
        let mut out = vec![0u8; self.data.len()];
        for sy in 0..h {
            for sx in 0..w {
                let (dx, dy) = match orientation {
                    2 => (w - 1 - sx, sy),
                    3 => (w - 1 - sx, h - 1 - sy),
                    4 => (sx, h - 1 - sy),
                    5 => (sy, sx),
                    6 => (h - 1 - sy, sx),
                    7 => (h - 1 - sy, w - 1 - sx),
                    _ => (sy, w - 1 - sx),
                };
                let s = (sy * w + sx) * bpp;
                let d = (dy * dw + dx) * bpp;
                out[d..d + bpp].copy_from_slice(&self.data[s..s + bpp]);
            }
        }
        Image { width: dw as u32, height: dh as u32, data: out, ..self }
    }
}
