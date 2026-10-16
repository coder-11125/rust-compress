//! WebP: container handling plus the lossless (VP8L) codec. Lossy (VP8) files are recognised but
//! not decoded, and animated files are refused.

pub mod cluster;
pub mod vp8l;
pub mod vp8l_enc;

use crate::error::{Error, Result};
use crate::image::{ColorType, Image, MAX_PIXELS, Metadata};

/// True if `data` starts like a WebP file (`RIFF` size `WEBP`).
pub fn is_webp(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP"
}

/// One RIFF chunk: its four-character code and payload.
struct Chunk<'a> {
    id: [u8; 4],
    body: &'a [u8],
}

fn chunks(data: &[u8]) -> Result<Vec<Chunk<'_>>> {
    if !is_webp(data) {
        return Err(Error::invalid("not a WebP file"));
    }
    let riff_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    // Trust the declared size only as far as the file goes (a truncated file is still parsed).
    let end = (riff_len + 8).min(data.len());
    let mut pos = 12;
    let mut out = Vec::new();
    while pos + 8 <= end {
        let id: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
        let len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_end = pos + 8 + len;
        if body_end > end {
            return Err(Error::invalid("truncated WebP chunk"));
        }
        out.push(Chunk { id, body: &data[pos + 8..body_end] });
        pos = body_end + (len & 1);
    }
    Ok(out)
}

/// Why a WebP file cannot be recompressed here (lossy or animated), if it cannot.
pub fn unsupported_reason(data: &[u8]) -> Option<&'static str> {
    let chunks = chunks(data).ok()?;
    let has = |id: &[u8; 4]| chunks.iter().any(|c| &c.id == id);
    let animated = has(b"ANIM")
        || has(b"ANMF")
        || chunks.iter().any(|c| &c.id == b"VP8X" && c.body.first().is_some_and(|f| f & 2 != 0));
    if animated {
        Some("animated WebP")
    } else if !has(b"VP8L") && has(b"VP8 ") {
        Some("lossy WebP")
    } else {
        None
    }
}

pub fn decode(data: &[u8]) -> Result<Image> {
    let chunks = chunks(data)?;
    let mut meta = Metadata::default();
    let mut lossless: Option<&[u8]> = None;
    let mut lossy = false;
    for c in &chunks {
        match &c.id {
            b"VP8X" => {
                if c.body.first().is_some_and(|f| f & 0x02 != 0) {
                    return Err(Error::unsupported("animated WebP"));
                }
            }
            b"ANIM" | b"ANMF" => return Err(Error::unsupported("animated WebP")),
            b"VP8L" if lossless.is_none() => lossless = Some(c.body),
            b"VP8 " => lossy = true,
            b"ICCP" => meta.icc = Some(c.body.to_vec()),
            b"EXIF" => {
                // Some writers keep the JPEG-style "Exif\0\0" prefix; the bare TIFF block is what we store.
                meta.exif = Some(c.body.strip_prefix(b"Exif\0\0").unwrap_or(c.body).to_vec());
            }
            b"XMP " => meta.xmp = Some(c.body.to_vec()),
            _ => {}
        }
    }
    let Some(body) = lossless else {
        return Err(if lossy {
            Error::unsupported("lossy WebP (VP8); only lossless WebP is supported")
        } else {
            Error::invalid("WebP file without image data")
        });
    };
    let d = vp8l::decode(body)?;
    debug_assert!((d.width * d.height) as u64 <= MAX_PIXELS);
    let (color, ch) = if d.has_alpha { (ColorType::Rgba, 4) } else { (ColorType::Rgb, 3) };
    let mut out = Vec::with_capacity(d.argb.len() * ch);
    for p in &d.argb {
        out.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
        if d.has_alpha {
            out.push((p >> 24) as u8);
        }
    }
    let mut img = Image::new(d.width as u32, d.height as u32, color, 8, out)?;
    img.meta = meta;
    Ok(img)
}

/// Largest width or height a VP8L stream can describe.
pub const MAX_DIMENSION: u32 = 16384;

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Effort 0-9.
    pub level: u32,
    /// Discard the colour of fully transparent pixels.
    pub clean_alpha: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self { level: 7, clean_alpha: true }
    }
}

fn push_chunk(out: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(id);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
}

/// Losslessly encode `img` as a WebP file. Callers must keep both dimensions within [`MAX_DIMENSION`].
pub fn encode(img: &Image, opts: &EncodeOptions) -> Vec<u8> {
    let src = img.to_depth8();
    let (w, h) = (src.width as usize, src.height as usize);
    let ch = src.color.channels();
    let mut argb = Vec::with_capacity(w * h);
    for p in src.data.chunks_exact(ch) {
        argb.push(match src.color {
            ColorType::Gray => 0xFF00_0000 | ((p[0] as u32) * 0x0001_0101),
            ColorType::GrayAlpha => ((p[1] as u32) << 24) | ((p[0] as u32) * 0x0001_0101),
            ColorType::Rgb => 0xFF00_0000 | (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32,
            ColorType::Rgba => (p[3] as u32) << 24 | (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32,
        });
    }
    let has_alpha = argb.iter().any(|p| p >> 24 != 255);
    let stream = vp8l_enc::encode(
        &argb,
        w,
        h,
        has_alpha,
        vp8l_enc::Options { level: opts.level, clean_alpha: opts.clean_alpha },
    );

    let meta = &img.meta;
    let extended = meta.icc.is_some() || meta.exif.is_some() || meta.xmp.is_some();
    let mut body = b"WEBP".to_vec();
    if extended {
        let flags = (meta.icc.is_some() as u8) << 5
            | (has_alpha as u8) << 4
            | (meta.exif.is_some() as u8) << 3
            | (meta.xmp.is_some() as u8) << 2;
        let mut vp8x = vec![flags, 0, 0, 0];
        vp8x.extend_from_slice(&(w as u32 - 1).to_le_bytes()[..3]);
        vp8x.extend_from_slice(&(h as u32 - 1).to_le_bytes()[..3]);
        push_chunk(&mut body, b"VP8X", &vp8x);
        if let Some(icc) = &meta.icc {
            push_chunk(&mut body, b"ICCP", icc);
        }
    }
    push_chunk(&mut body, b"VP8L", &stream);
    if extended {
        if let Some(exif) = &meta.exif {
            push_chunk(&mut body, b"EXIF", exif);
        }
        if let Some(xmp) = &meta.xmp {
            push_chunk(&mut body, b"XMP ", xmp);
        }
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}
