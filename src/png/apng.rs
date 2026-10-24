//! Animated PNG (APNG) to full-canvas frames. Each frame's pixels are decoded by the ordinary PNG decoder from a
//! rebuilt single-frame file, then composited with the frame's dispose and blend operations.

use super::SIGNATURE;
use crate::animation::{AnimFrame, Animation, MAX_ANIMATION_PIXELS, normalize_delay};
use crate::error::{Error, Result};
use crate::image::{ColorType, Image, MAX_PIXELS};
use crate::util::checksum::crc32;

struct Chunk<'a> {
    kind: [u8; 4],
    body: &'a [u8],
}

fn chunks(data: &[u8]) -> Result<Vec<Chunk<'_>>> {
    if !data.starts_with(&SIGNATURE) {
        return Err(Error::invalid("not a PNG file"));
    }
    let mut out = Vec::new();
    let mut pos = 8;
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let end = pos.checked_add(12).and_then(|p| p.checked_add(len)).filter(|&e| e <= data.len());
        let end = end.ok_or_else(|| Error::invalid("truncated PNG chunk"))?;
        out.push(Chunk {
            kind: [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]],
            body: &data[pos + 8..end - 4],
        });
        pos = end;
        if out.last().is_some_and(|c| &c.kind == b"IEND") {
            break;
        }
    }
    Ok(out)
}

fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

struct FrameCtl {
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    delay_ms: u32,
    dispose: u8,
    blend: u8,
    data: Vec<u8>,
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Every frame of an APNG as the canvas it produces. A PNG without animation gives one frame.
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    let list = chunks(data)?;
    let ihdr = list.first().filter(|c| &c.kind == b"IHDR" && c.body.len() == 13);
    let ihdr = ihdr.ok_or_else(|| Error::invalid("PNG without IHDR"))?.body;
    let (cw, ch) = (be32(&ihdr[0..4]) as usize, be32(&ihdr[4..8]) as usize);
    if cw == 0 || ch == 0 || (cw as u64) * (ch as u64) > MAX_PIXELS {
        return Err(Error::invalid("bad PNG dimensions"));
    }
    let mut plays = 1u32;
    let mut frames: Vec<FrameCtl> = Vec::new();
    let mut shared: Vec<(&[u8; 4], &[u8])> = Vec::new();
    for c in &list {
        match &c.kind {
            b"acTL" if c.body.len() >= 8 => plays = be32(&c.body[4..8]),
            b"PLTE" | b"tRNS" => shared.push((&c.kind, c.body)),
            b"fcTL" if c.body.len() >= 26 => {
                let b = c.body;
                let (num, den) = (u16::from_be_bytes([b[20], b[21]]) as u64, u16::from_be_bytes([b[22], b[23]]) as u64);
                let ms = (num * 1000).checked_div(den).unwrap_or(num * 10);
                if frames.len() >= 100_000 {
                    return Err(Error::unsupported("too many animation frames"));
                }
                frames.push(FrameCtl {
                    w: be32(&b[4..8]) as usize,
                    h: be32(&b[8..12]) as usize,
                    x: be32(&b[12..16]) as usize,
                    y: be32(&b[16..20]) as usize,
                    delay_ms: ms.min(u32::MAX as u64) as u32,
                    dispose: b[24],
                    blend: b[25],
                    data: Vec::new(),
                });
            }
            // The default image only belongs to the animation when an fcTL precedes it.
            b"IDAT" => {
                if frames.len() == 1 {
                    frames[0].data.extend_from_slice(c.body);
                }
            }
            b"fdAT" if c.body.len() > 4 => {
                if let Some(f) = frames.last_mut() {
                    f.data.extend_from_slice(&c.body[4..]);
                }
            }
            _ => {}
        }
    }
    if frames.is_empty() {
        let img = super::decode::decode(data)?;
        let rgba = to_rgba8(&img);
        let image = finish(rgba, cw, ch, true)?;
        return Ok(Animation {
            width: cw as u32,
            height: ch as u32,
            frames: vec![AnimFrame { image, duration_ms: crate::animation::DEFAULT_DELAY_MS }],
            loop_forever: false,
        });
    }
    if (frames.len() as u64) * (cw as u64) * (ch as u64) > MAX_ANIMATION_PIXELS {
        return Err(Error::unsupported("APNG animation too large"));
    }

    let mut canvas = vec![0u8; cw * ch * 4];
    let mut saved: Option<Vec<u8>> = None;
    let mut prev: Option<(u8, usize, usize, usize, usize)> = None;
    let mut shown: Vec<(Vec<u8>, u32)> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if f.w == 0 || f.h == 0 || f.x + f.w > cw || f.y + f.h > ch {
            return Err(Error::invalid("APNG frame outside the canvas"));
        }
        if let Some((dispose, x, y, w, h)) = prev {
            match dispose {
                1 => {
                    for row in y..y + h {
                        canvas[(row * cw + x) * 4..(row * cw + x + w) * 4].fill(0);
                    }
                }
                2 => {
                    if let Some(s) = saved.take() {
                        canvas = s;
                    }
                }
                _ => {}
            }
        }
        // A first frame cannot restore "the previous" canvas; it clears instead.
        let dispose = if i == 0 && f.dispose == 2 { 1 } else { f.dispose };
        if dispose == 2 {
            saved = Some(canvas.clone());
        }
        let mut png = SIGNATURE.to_vec();
        let mut h = ihdr.to_vec();
        h[0..4].copy_from_slice(&(f.w as u32).to_be_bytes());
        h[4..8].copy_from_slice(&(f.h as u32).to_be_bytes());
        push_chunk(&mut png, b"IHDR", &h);
        for (k, b) in &shared {
            push_chunk(&mut png, k, b);
        }
        push_chunk(&mut png, b"IDAT", &f.data);
        push_chunk(&mut png, b"IEND", &[]);
        let px = to_rgba8(&super::decode::decode(&png)?);
        for row in 0..f.h {
            for col in 0..f.w {
                let s = &px[(row * f.w + col) * 4..][..4];
                let d = &mut canvas[((f.y + row) * cw + f.x + col) * 4..][..4];
                if f.blend == 0 || s[3] == 255 {
                    d.copy_from_slice(s);
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
        shown.push((canvas.clone(), normalize_delay(f.delay_ms)));
        prev = Some((dispose, f.x, f.y, f.w, f.h));
    }
    let opaque = shown.iter().all(|(c, _)| c.chunks_exact(4).all(|p| p[3] == 255));
    let mut out = Vec::with_capacity(shown.len());
    for (rgba, ms) in shown {
        out.push(AnimFrame { image: finish(rgba, cw, ch, opaque)?, duration_ms: ms });
    }
    Ok(Animation { width: cw as u32, height: ch as u32, frames: out, loop_forever: plays != 1 })
}

/// 8-bit RGBA samples of any decoded PNG.
fn to_rgba8(img: &Image) -> Vec<u8> {
    let src = img.to_depth8();
    let mut out = Vec::with_capacity(src.data.len() / src.color.channels() * 4);
    for p in src.data.chunks_exact(src.color.channels()) {
        match src.color {
            ColorType::Gray => out.extend([p[0], p[0], p[0], 255]),
            ColorType::GrayAlpha => out.extend([p[0], p[0], p[0], p[1]]),
            ColorType::Rgb => out.extend([p[0], p[1], p[2], 255]),
            ColorType::Rgba => out.extend([p[0], p[1], p[2], p[3]]),
        }
    }
    out
}

fn finish(rgba: Vec<u8>, w: usize, h: usize, opaque_ok: bool) -> Result<Image> {
    if opaque_ok && rgba.chunks_exact(4).all(|p| p[3] == 255) {
        let rgb = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        Image::new(w as u32, h as u32, ColorType::Rgb, 8, rgb)
    } else {
        Image::new(w as u32, h as u32, ColorType::Rgba, 8, rgba)
    }
}
