//! WebP: container handling, the lossless (VP8L) codec and the lossy (VP8) codec. [`decode`] refuses animated
//! files; [`decode_animation`] reads them.

pub mod alpha;
pub mod anim_enc;
pub mod cluster;
pub mod vp8;
pub mod vp8l;
pub mod vp8l_enc;

pub use anim_enc::{AnimOptions, encode_animation};
use rc_core::animation::Animation;
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image, MAX_PIXELS, Metadata};

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
    for c in &chunks {
        match &c.id {
            b"VP8X" if c.body.first().is_some_and(|f| f & 0x02 != 0) => {
                return Err(Error::unsupported("animated WebP"));
            }
            b"ANIM" | b"ANMF" => return Err(Error::unsupported("animated WebP")),
            _ => {}
        }
    }
    image_from_chunks(&chunks)
}

/// The picture described by the chunks of one image (a whole still file, or the payload of an animation frame).
fn image_from_chunks(chunks: &[Chunk<'_>]) -> Result<Image> {
    let mut meta = Metadata::default();
    let mut lossless: Option<&[u8]> = None;
    let mut lossy_data: Option<&[u8]> = None;
    let mut alpha_data: Option<&[u8]> = None;
    for c in chunks {
        match &c.id {
            b"VP8L" if lossless.is_none() => lossless = Some(c.body),
            b"VP8 " if lossy_data.is_none() => lossy_data = Some(c.body),
            b"ALPH" if alpha_data.is_none() => alpha_data = Some(c.body),
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
        let Some(frame_data) = lossy_data else {
            return Err(Error::invalid("WebP file without image data"));
        };
        let frame = vp8::decode::decode_frame(frame_data)?;
        let (w, h) = (frame.width, frame.height);
        let rgb = vp8::yuv::yuv420_to_rgb(w, h, &frame.y, frame.y_stride, &frame.u, &frame.v, frame.uv_stride);
        let mut img = match alpha_data {
            Some(a) => {
                let plane = alpha::decode(a, w, h)?;
                let mut out = Vec::with_capacity(w * h * 4);
                for (px, &al) in rgb.chunks_exact(3).zip(&plane) {
                    out.extend_from_slice(&[px[0], px[1], px[2], al]);
                }
                Image::new(w as u32, h as u32, ColorType::Rgba, 8, out)?
            }
            None => Image::new(w as u32, h as u32, ColorType::Rgb, 8, rgb)?,
        };
        img.meta = meta;
        return Ok(img);
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

/// Chunks packed back to back, as inside an `ANMF` payload.
fn sub_chunks(body: &[u8]) -> Result<Vec<Chunk<'_>>> {
    let mut pos = 0;
    let mut out = Vec::new();
    while pos + 8 <= body.len() {
        let id: [u8; 4] = body[pos..pos + 4].try_into().unwrap();
        let len = u32::from_le_bytes(body[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = pos.checked_add(8).and_then(|p| p.checked_add(len)).filter(|&e| e <= body.len());
        let end = end.ok_or_else(|| Error::invalid("truncated WebP frame chunk"))?;
        out.push(Chunk { id, body: &body[pos + 8..end] });
        pos = end + (len & 1);
    }
    Ok(out)
}

fn le24(b: &[u8]) -> usize {
    b[0] as usize | (b[1] as usize) << 8 | (b[2] as usize) << 16
}

/// Every frame of an animated WebP as the canvas it produces (8-bit RGBA, or RGB when nothing is transparent).
/// A still WebP gives a one-frame animation.
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    use rc_core::animation::{AnimFrame, MAX_ANIMATION_PIXELS, normalize_delay};
    let list = chunks(data)?;
    let animated = list.iter().any(|c| &c.id == b"ANMF");
    if !animated {
        let img = decode(data)?;
        return Ok(Animation {
            width: img.width,
            height: img.height,
            frames: vec![AnimFrame { image: img, duration_ms: rc_core::animation::DEFAULT_DELAY_MS }],
            loop_forever: false,
        });
    }
    let vp8x = list
        .iter()
        .find(|c| &c.id == b"VP8X" && c.body.len() >= 10)
        .ok_or_else(|| Error::invalid("animated WebP without VP8X"))?;
    let (cw, ch) = (le24(&vp8x.body[4..7]) + 1, le24(&vp8x.body[7..10]) + 1);
    if (cw as u64) * (ch as u64) > MAX_PIXELS {
        return Err(Error::invalid("bad WebP canvas size"));
    }
    let loop_forever = list
        .iter()
        .find(|c| &c.id == b"ANIM" && c.body.len() >= 6)
        .is_none_or(|c| u16::from_le_bytes([c.body[4], c.body[5]]) != 1);
    let mut canvas = vec![0u8; cw * ch * 4];
    let mut shown: Vec<(Vec<u8>, u32)> = Vec::new();
    let mut prev: Option<(bool, usize, usize, usize, usize)> = None;
    for c in list.iter().filter(|c| &c.id == b"ANMF") {
        if c.body.len() < 16 {
            return Err(Error::invalid("short ANMF chunk"));
        }
        if (shown.len() as u64 + 1) * (cw as u64) * (ch as u64) > MAX_ANIMATION_PIXELS {
            return Err(Error::unsupported("WebP animation too large"));
        }
        let b = c.body;
        let (x, y) = (le24(&b[0..3]) * 2, le24(&b[3..6]) * 2);
        let (w, h) = (le24(&b[6..9]) + 1, le24(&b[9..12]) + 1);
        let ms = le24(&b[12..15]) as u32;
        let (dispose_bg, no_blend) = (b[15] & 1 != 0, b[15] & 2 != 0);
        if x + w > cw || y + h > ch {
            return Err(Error::invalid("WebP frame outside the canvas"));
        }
        if let Some((true, px, py, pw, ph)) = prev {
            for row in py..py + ph {
                canvas[(row * cw + px) * 4..(row * cw + px + pw) * 4].fill(0);
            }
        }
        let img = image_from_chunks(&sub_chunks(&b[16..])?)?;
        if (img.width as usize, img.height as usize) != (w, h) {
            return Err(Error::invalid("WebP frame size does not match its header"));
        }
        let chs = img.color.channels();
        for row in 0..h {
            for col in 0..w {
                let p = &img.data[(row * w + col) * chs..][..chs];
                let s = if chs == 4 { [p[0], p[1], p[2], p[3]] } else { [p[0], p[1], p[2], 255] };
                let d = &mut canvas[((y + row) * cw + x + col) * 4..][..4];
                if no_blend || s[3] == 255 {
                    d.copy_from_slice(&s);
                } else if s[3] != 0 {
                    let (sa, da) = (s[3] as u32, d[3] as u32);
                    let oa = sa * 255 + da * (255 - sa);
                    for k in 0..3 {
                        d[k] = ((s[k] as u32 * sa * 255 + d[k] as u32 * da * (255 - sa) + oa / 2) / oa) as u8;
                    }
                    d[3] = ((oa + 127) / 255) as u8;
                }
            }
        }
        shown.push((canvas.clone(), normalize_delay(ms)));
        prev = Some((dispose_bg, x, y, w, h));
    }
    if shown.is_empty() {
        return Err(Error::invalid("animated WebP without frames"));
    }
    let opaque = shown.iter().all(|(f, _)| f.chunks_exact(4).all(|p| p[3] == 255));
    let mut frames = Vec::with_capacity(shown.len());
    for (rgba, ms) in shown {
        let image = if opaque {
            let rgb = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            Image::new(cw as u32, ch as u32, ColorType::Rgb, 8, rgb)?
        } else {
            Image::new(cw as u32, ch as u32, ColorType::Rgba, 8, rgba)?
        };
        frames.push(AnimFrame { image, duration_ms: ms });
    }
    Ok(Animation { width: cw as u32, height: ch as u32, frames, loop_forever })
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

    assemble(img, w, h, has_alpha, None, b"VP8L", &stream)
}

/// Wrap an image stream (`VP8L` or `VP8 `) with its optional `ALPH` chunk and the metadata chunks.
fn assemble(
    img: &Image,
    w: usize,
    h: usize,
    has_alpha: bool,
    alph: Option<&[u8]>,
    id: &[u8; 4],
    stream: &[u8],
) -> Vec<u8> {
    let meta = &img.meta;
    let extended = meta.icc.is_some() || meta.exif.is_some() || meta.xmp.is_some() || alph.is_some();
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
    if let Some(alph) = alph {
        push_chunk(&mut body, b"ALPH", alph);
    }
    push_chunk(&mut body, id, stream);
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

/// Largest width or height a VP8 frame can describe.
pub const MAX_LOSSY_DIMENSION: u32 = 16383;

/// Lossy encode `img` as a WebP file: VP8 for the colour and a lossless `ALPH` chunk for any
/// transparency. `quality` is 0-100 and `level` the effort 0-9.
pub fn encode_lossy(img: &Image, quality: u8, level: u32, clean_alpha: bool) -> Result<Vec<u8>> {
    if img.width > MAX_LOSSY_DIMENSION || img.height > MAX_LOSSY_DIMENSION {
        return Err(Error::unsupported(format!(
            "lossy WebP images are limited to {0}x{0} pixels",
            MAX_LOSSY_DIMENSION
        )));
    }
    let src = img.to_depth8();
    let (w, h) = (src.width as usize, src.height as usize);
    let ch = src.color.channels();
    let mut rgb = Vec::with_capacity(w * h * 3);
    let mut alpha = Vec::with_capacity(w * h);
    for p in src.data.chunks_exact(ch) {
        match src.color {
            ColorType::Gray => rgb.extend_from_slice(&[p[0]; 3]),
            ColorType::GrayAlpha => {
                rgb.extend_from_slice(&[p[0]; 3]);
                alpha.push(p[1]);
            }
            ColorType::Rgb => rgb.extend_from_slice(p),
            ColorType::Rgba => {
                rgb.extend_from_slice(&p[..3]);
                alpha.push(p[3]);
            }
        }
    }
    let has_alpha = alpha.iter().any(|&a| a != 255);
    if has_alpha && clean_alpha {
        smooth_hidden(&mut rgb, &alpha, w, h);
    }
    let (y, u, v) = rgb_to_yuv420(&rgb, w, h);
    let planes = vp8::encode::Planes { width: w, height: h, y: &y, u: &u, v: &v };
    let frame = vp8::encode::encode_frame(&planes, vp8::encode::Options { quality: quality as f32, level })?;
    let alph = has_alpha.then(|| encode_alpha(&alpha, w, h, level));
    Ok(assemble(img, w, h, has_alpha, alph.as_deref(), b"VP8 ", &frame.data))
}

/// Give fully transparent pixels the average colour of their visible neighbours in the same 8x8
/// block (or the previous block's colour), so they cost almost nothing to code.
fn smooth_hidden(rgb: &mut [u8], alpha: &[u8], w: usize, h: usize) {
    let mut carry = [128u8; 3];
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let (ye, xe) = ((by + 8).min(h), (bx + 8).min(w));
            let (mut sum, mut n) = ([0u32; 3], 0u32);
            for y in by..ye {
                for x in bx..xe {
                    if alpha[y * w + x] != 0 {
                        for c in 0..3 {
                            sum[c] += rgb[(y * w + x) * 3 + c] as u32;
                        }
                        n += 1;
                    }
                }
            }
            if let Some(avg) = sum.map(|c| c.checked_div(n)).into_iter().collect::<Option<Vec<_>>>() {
                carry = [avg[0] as u8, avg[1] as u8, avg[2] as u8];
            }
            for y in by..ye {
                for x in bx..xe {
                    if alpha[y * w + x] == 0 {
                        rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&carry);
                    }
                }
            }
        }
    }
}

/// BT.601 limited-range planes with 4:2:0 chroma (odd edges replicated), as libwebp computes them.
fn rgb_to_yuv420(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use vp8::yuv::{rgb_to_y, rgb4_to_uv};
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = Vec::with_capacity(w * h);
    for p in rgb.chunks_exact(3) {
        y.push(rgb_to_y(p[0] as i32, p[1] as i32, p[2] as i32));
    }
    let (mut u, mut v) = (Vec::with_capacity(cw * ch), Vec::with_capacity(cw * ch));
    for cy in 0..ch {
        for cx in 0..cw {
            let mut s = [0i32; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let (px, py) = ((cx * 2 + dx).min(w - 1), (cy * 2 + dy).min(h - 1));
                    for c in 0..3 {
                        s[c] += rgb[(py * w + px) * 3 + c] as i32;
                    }
                }
            }
            let (uu, vv) = rgb4_to_uv(s[0], s[1], s[2]);
            u.push(uu);
            v.push(vv);
        }
    }
    (y, u, v)
}

/// The `ALPH` chunk body: the smallest of the lossless-coded filtered planes (or the raw plane).
fn encode_alpha(alpha: &[u8], w: usize, h: usize, level: u32) -> Vec<u8> {
    use alpha::Filter;
    let filters: &[Filter] = if level >= 5 {
        &[Filter::None, Filter::Horizontal, Filter::Vertical, Filter::Gradient]
    } else {
        &[Filter::None, Filter::Gradient]
    };
    let mut best: Option<Vec<u8>> = None;
    for &f in filters {
        let filtered = alpha::filter(alpha, w, f);
        let stream = vp8l_enc::encode_plane(&filtered, w, h, level);
        let mut chunk = vec![1 | (f as u8) << 2];
        chunk.extend_from_slice(&stream);
        if best.as_ref().is_none_or(|b| chunk.len() < b.len()) {
            best = Some(chunk);
        }
    }
    let mut raw = vec![0u8];
    raw.extend_from_slice(alpha);
    let best = best.expect("at least one filter");
    if raw.len() < best.len() { raw } else { best }
}
