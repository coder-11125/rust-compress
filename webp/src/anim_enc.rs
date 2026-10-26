//! Animated WebP writer.
//!
//! The first frame is the whole canvas. Every later frame is coded as the rectangle that changed since the one before
//! (its corner moved to even coordinates, as `ANMF` requires). Lossless frames are tried both replacing the canvas
//! and, when every changed pixel is opaque, blended over it with the unchanged pixels transparent, which codes much
//! smaller; the smaller one is kept. Lossy frames are compared with the canvas as the decoder reconstructs it, so
//! coding errors do not pile up from frame to frame.

use super::{EncodeOptions, MAX_DIMENSION, MAX_LOSSY_DIMENSION, chunks, decode, encode, encode_lossy, push_chunk};
use rc_core::animation::Animation;
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image};

#[derive(Clone, Debug)]
pub struct AnimOptions {
    /// Effort 0-9.
    pub level: u32,
    /// Discard the colour of fully transparent pixels.
    pub clean_alpha: bool,
    /// `Some(quality)` codes the frames with the lossy codec (quality 0-100); `None` is lossless.
    pub lossy_quality: Option<u8>,
}

impl Default for AnimOptions {
    fn default() -> Self {
        AnimOptions { level: 7, clean_alpha: true, lossy_quality: None }
    }
}

type Px = [u8; 4];

fn rgba(img: &Image) -> Vec<Px> {
    let src = img.to_depth8();
    let ch = src.color.channels();
    src.data
        .chunks_exact(ch)
        .map(|p| match src.color {
            ColorType::Gray => [p[0], p[0], p[0], 255],
            ColorType::GrayAlpha => [p[0], p[0], p[0], p[1]],
            ColorType::Rgb => [p[0], p[1], p[2], 255],
            ColorType::Rgba => [p[0], p[1], p[2], p[3]],
        })
        .collect()
}

/// A rectangle with even `x` and `y`.
#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// The smallest even-cornered rectangle holding every pixel for which `changed` is true (a single pixel if none).
fn bounds(w: usize, h: usize, changed: impl Fn(usize) -> bool) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if changed(y * w + x) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 == 0 {
        (x0, y0, x1, y1) = (0, 0, 1, 1);
    }
    let (x0, y0) = (x0 & !1, y0 & !1);
    Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}

fn crop(canvas: &[Px], width: usize, r: Rect, opaque: bool, f: impl Fn(usize, Px) -> Px) -> Result<Image> {
    let mut data = Vec::with_capacity(r.w * r.h * 4);
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            let p = y * width + x;
            let v = f(p, canvas[p]);
            if opaque {
                data.extend_from_slice(&v[..3]);
            } else {
                data.extend_from_slice(&v);
            }
        }
    }
    Image::new(r.w as u32, r.h as u32, if opaque { ColorType::Rgb } else { ColorType::Rgba }, 8, data)
}

/// The `ALPH` (if any) and image chunks of a still WebP file, as sub-chunks of a frame.
fn frame_chunks(file: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for c in chunks(file)? {
        if matches!(&c.id, b"ALPH" | b"VP8 " | b"VP8L") {
            push_chunk(&mut out, &c.id, c.body);
        }
    }
    if out.is_empty() {
        return Err(Error::invalid("encoded frame without image data"));
    }
    Ok(out)
}

fn le24(v: usize) -> [u8; 3] {
    [v as u8, (v >> 8) as u8, (v >> 16) as u8]
}

struct Coded {
    rect: Rect,
    blend: bool,
    payload: Vec<u8>,
}

/// Lossy frames: tolerated difference per channel before a pixel counts as changed.
fn tolerance(quality: u8) -> i32 {
    ((100 - quality.min(100) as i32) / 6).max(1)
}

pub fn encode_animation(anim: &Animation, opts: &AnimOptions) -> Result<Vec<u8>> {
    let (w, h) = (anim.width as usize, anim.height as usize);
    let limit = if opts.lossy_quality.is_some() { MAX_LOSSY_DIMENSION } else { MAX_DIMENSION } as usize;
    if w == 0 || h == 0 || w > limit || h > limit {
        return Err(Error::unsupported(format!("WebP images are limited to {limit}x{limit} pixels")));
    }
    if anim.frames.is_empty() {
        return Err(Error::invalid("animation without frames"));
    }
    let mut canvases = Vec::with_capacity(anim.frames.len());
    for f in &anim.frames {
        if (f.image.width as usize, f.image.height as usize) != (w, h) {
            return Err(Error::invalid("animation frames differ in size"));
        }
        canvases.push(rgba(&f.image));
    }
    let has_alpha = canvases.iter().any(|c| c.iter().any(|p| p[3] != 255));
    let lossless = EncodeOptions { level: opts.level, clean_alpha: opts.clean_alpha };

    let mut body = b"WEBP".to_vec();
    let mut vp8x = vec![0x02 | (has_alpha as u8) << 4, 0, 0, 0];
    vp8x.extend_from_slice(&le24(w - 1));
    vp8x.extend_from_slice(&le24(h - 1));
    push_chunk(&mut body, b"VP8X", &vp8x);
    let mut anim_chunk = vec![0u8; 4]; // background colour: transparent black
    anim_chunk.extend_from_slice(&(if anim.loop_forever { 0u16 } else { 1u16 }).to_le_bytes());
    push_chunk(&mut body, b"ANIM", &anim_chunk);

    // What a decoder shows after the frames coded so far.
    let mut shown: Vec<Px> = vec![[0; 4]; w * h];
    for (i, cur) in canvases.iter().enumerate() {
        let coded = match opts.lossy_quality {
            None => {
                let first = i == 0;
                let same = |p: usize| !first && visually_equal(cur[p], shown[p]);
                let rect = if first { Rect { x: 0, y: 0, w, h } } else { bounds(w, h, |p| !same(p)) };
                let replace = crop(cur, w, rect, !has_alpha, |_, v| v)?;
                let mut best = Coded { rect, blend: false, payload: frame_chunks(&encode(&replace, &lossless))? };
                // Blending is exact only if every changed pixel is opaque (unchanged ones become transparent).
                let blendable = !first
                    && has_alpha
                    && (0..w * h).all(|p| {
                        let (x, y) = (p % w, p / w);
                        let inside = x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h;
                        !inside || same(p) || cur[p][3] == 255
                    });
                if blendable {
                    let over = crop(cur, w, rect, false, |p, v| if same(p) { [0; 4] } else { v })?;
                    let payload = frame_chunks(&encode(&over, &lossless))?;
                    if payload.len() < best.payload.len() {
                        best = Coded { rect, blend: true, payload };
                    }
                }
                best
            }
            Some(q) => {
                let tol = tolerance(q);
                let first = i == 0;
                let rect = if first {
                    Rect { x: 0, y: 0, w, h }
                } else {
                    bounds(w, h, |p| (0..4).any(|c| (cur[p][c] as i32 - shown[p][c] as i32).abs() > tol))
                };
                let img = crop(cur, w, rect, !has_alpha, |_, v| v)?;
                let file = encode_lossy(&img, q, opts.level, opts.clean_alpha)?;
                // Keep the decoder's own picture, so later frames are measured against what is really shown.
                let back = rgba(&decode(&file)?);
                for y in 0..rect.h {
                    for x in 0..rect.w {
                        shown[(rect.y + y) * w + rect.x + x] = back[y * rect.w + x];
                    }
                }
                Coded { rect, blend: false, payload: frame_chunks(&file)? }
            }
        };
        if opts.lossy_quality.is_none() {
            // Lossless: the decoder ends up showing the source frame (hidden colours aside).
            shown.clone_from(cur);
        }
        let r = coded.rect;
        let mut anmf = Vec::with_capacity(16 + coded.payload.len());
        anmf.extend_from_slice(&le24(r.x / 2));
        anmf.extend_from_slice(&le24(r.y / 2));
        anmf.extend_from_slice(&le24(r.w - 1));
        anmf.extend_from_slice(&le24(r.h - 1));
        anmf.extend_from_slice(&le24(anim.frames[i].duration_ms.min(0xFF_FFFF) as usize));
        anmf.push(if coded.blend { 0 } else { 2 });
        anmf.extend_from_slice(&coded.payload);
        push_chunk(&mut body, b"ANMF", &anmf);
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Same picture: equal colour, or both invisible (the colour of a transparent pixel is not shown).
fn visually_equal(a: Px, b: Px) -> bool {
    a == b || (a[3] == 0 && b[3] == 0)
}
