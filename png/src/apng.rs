//! Animated PNG (APNG) to full-canvas frames. Each frame's pixels are decoded by the ordinary PNG decoder from a
//! rebuilt single-frame file, then composited with the frame's dispose and blend operations.

use super::SIGNATURE;
use super::encode::{EncodeOptions, idat_for_rows};
use rayon::prelude::*;
use rc_core::animation::{AnimFrame, Animation, MAX_ANIMATION_FRAMES, MAX_ANIMATION_PIXELS, normalize_delay};
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image, max_pixels};
use rc_core::util::checksum::crc32;

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
    if cw == 0 || ch == 0 || (cw as u64) * (ch as u64) > max_pixels() {
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
            frames: vec![AnimFrame { image, duration_ms: rc_core::animation::DEFAULT_DELAY_MS }],
            loop_forever: false,
        });
    }
    if (frames.len() as u64) * (cw as u64) * (ch as u64) > MAX_ANIMATION_PIXELS || frames.len() > MAX_ANIMATION_FRAMES {
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

/// APNG stores the delay as `delay_num / delay_den` seconds with both fields 16-bit. Millisecond delays up to
/// 65535 fit directly; longer ones (GIF can name 655.35 s) need a smaller denominator. The search below picks a
/// denominator whose numerator still fits and whose integer-millisecond decode is exact when one exists.
fn delay_ratio(ms: u32) -> (u16, u16) {
    const MAX_MS: u64 = (u16::MAX as u64) * 1000;
    let ms = u64::from(ms).min(MAX_MS);
    if ms <= u64::from(u16::MAX) {
        return (ms as u16, 1000);
    }
    for den in (1..=999u16).rev() {
        let den = u64::from(den);
        let num = (ms * den).div_ceil(1000);
        if num > u64::from(u16::MAX) {
            continue;
        }
        if num * 1000 / den == ms {
            return (num as u16, den as u16);
        }
    }
    let den = ((u64::from(u16::MAX) * 1000) / ms).max(1) as u16;
    let num = ((ms * u64::from(den) + 500) / 1000).min(u64::from(u16::MAX)) as u16;
    (num, den)
}

/// Where a frame's pixels go on the canvas and how they combine with what is already there.
#[derive(Clone, Copy)]
struct Region {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    /// APNG_BLEND_OP_OVER (true) or APNG_BLEND_OP_SOURCE (false).
    over: bool,
}

fn frame_ctl(seq: u32, r: Region, delay_ms: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(26);
    b.extend_from_slice(&seq.to_be_bytes());
    b.extend_from_slice(&(r.w as u32).to_be_bytes());
    b.extend_from_slice(&(r.h as u32).to_be_bytes());
    b.extend_from_slice(&(r.x as u32).to_be_bytes());
    b.extend_from_slice(&(r.y as u32).to_be_bytes());
    let (num, den) = delay_ratio(delay_ms);
    b.extend_from_slice(&num.to_be_bytes());
    b.extend_from_slice(&den.to_be_bytes());
    b.push(0); // APNG_DISPOSE_OP_NONE: the canvas simply carries over to the next frame.
    b.push(r.over as u8);
    b
}

/// Normalize a frame to 8-bit RGB or RGBA, whichever the animation uses for every frame.
fn normalize(img: &Image, alpha: bool) -> Image {
    let src = img.to_depth8();
    let ch = src.color.channels();
    let mut data = Vec::with_capacity(src.data.len() / ch * if alpha { 4 } else { 3 });
    for p in src.data.chunks_exact(ch) {
        let (r, g, b, a) = match src.color {
            ColorType::Gray => (p[0], p[0], p[0], 255),
            ColorType::GrayAlpha => (p[0], p[0], p[0], p[1]),
            ColorType::Rgb => (p[0], p[1], p[2], 255),
            ColorType::Rgba => (p[0], p[1], p[2], p[3]),
        };
        if alpha {
            data.extend_from_slice(&[r, g, b, a]);
        } else {
            data.extend_from_slice(&[r, g, b]);
        }
    }
    let color = if alpha { ColorType::Rgba } else { ColorType::Rgb };
    Image::new(src.width, src.height, color, 8, data).expect("normalized frame has the right size")
}

/// Rewrite the colour of fully transparent pixels so they compress well (the decoder composites them away).
fn clean_transparent(img: &mut Image) {
    let (w, h) = (img.width as usize, img.height as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            if img.data[i + 3] != 0 {
                continue;
            }
            let src = if x > 0 {
                Some(i - 4)
            } else if y > 0 {
                Some(i - w * 4)
            } else {
                None
            };
            for c in 0..3 {
                img.data[i + c] = src.map_or(0, |s| img.data[s + c]);
            }
        }
    }
}

/// Smallest rectangle holding every pixel that differs between two canvases (`None` when they are the same).
/// With `ignore_hidden`, two fully transparent pixels count as equal whatever their colour.
fn dirty_rect(
    prev: &[u8],
    cur: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    ignore_hidden: bool,
) -> Option<(usize, usize, usize, usize)> {
    let differs = |i: usize| {
        let (a, b) = (&prev[i * ch..(i + 1) * ch], &cur[i * ch..(i + 1) * ch]);
        a != b && !(ignore_hidden && ch == 4 && a[3] == 0 && b[3] == 0)
    };
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if differs(y * w + x) {
                x0 = x0.min(x);
                x1 = x1.max(x + 1);
                y0 = y0.min(y);
                y1 = y + 1;
            }
        }
    }
    (x1 > x0).then(|| (x0, y0, x1 - x0, y1 - y0))
}

/// Pixels of `r` cut out of `cur`. For OVER, pixels equal to the previous canvas become fully transparent so they
/// leave it untouched (and compress to nothing); returns `None` when some changed pixel is not opaque, since OVER
/// can then not reproduce it exactly.
fn region_pixels(prev: &[u8], cur: &[u8], w: usize, ch: usize, r: Region, ignore_hidden: bool) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(r.w * r.h * ch);
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            let i = (y * w + x) * ch;
            let (a, b) = (&prev[i..i + ch], &cur[i..i + ch]);
            if !r.over {
                out.extend_from_slice(b);
            } else if a == b || (ignore_hidden && a[3] == 0 && b[3] == 0) {
                out.extend_from_slice(&[0, 0, 0, 0]);
            } else if b[3] == 255 {
                out.extend_from_slice(b);
            } else {
                return None;
            }
        }
    }
    Some(out)
}

/// Encode an animation as an APNG. After the first frame (a full canvas), each frame stores only the rectangle that
/// changed from the previous canvas, using APNG_BLEND_OP_OVER when that is exact and smaller (unchanged pixels inside
/// the rectangle become transparent) and SOURCE otherwise. A frame identical to the one before it is a 1x1 no-op.
/// `clean_alpha` rewrites the invisible colour of fully transparent pixels.
pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let (w, h) = (anim.width, anim.height);
    if anim.frames.is_empty() {
        return Err(Error::invalid("animation without frames"));
    }
    if w == 0 || h == 0 || w > i32::MAX as u32 || h > i32::MAX as u32 {
        return Err(Error::unsupported("APNG dimensions are limited to 2147483647 pixels per side"));
    }
    if (anim.frames.len() as u64) * (w as u64) * (h as u64) > MAX_ANIMATION_PIXELS {
        return Err(Error::unsupported("APNG animation too large"));
    }
    for f in &anim.frames {
        if (f.image.width, f.image.height) != (w, h) {
            return Err(Error::invalid("animation frames differ in size"));
        }
    }
    let alpha = anim.frames.iter().any(|f| f.image.color.has_alpha());
    let color_type = if alpha { 6 } else { 2 };
    let ch = if alpha { 4 } else { 3 };

    let mut canvases = Vec::with_capacity(anim.frames.len());
    for f in &anim.frames {
        let mut img = normalize(&f.image, alpha);
        if alpha && opts.clean_alpha {
            clean_transparent(&mut img);
        }
        canvases.push(img.data);
    }
    let (wu, hu) = (w as usize, h as usize);
    let hidden_equal = alpha && opts.clean_alpha;
    let level = opts.level;
    let encode_region = |pix: &[u8], r: Region| idat_for_rows(pix, r.w * ch, ch, 8, color_type, level);
    let planned: Vec<(Region, Vec<u8>)> = (0..canvases.len())
        .into_par_iter()
        .map(|i| {
            let full = Region { x: 0, y: 0, w: wu, h: hu, over: false };
            if i == 0 {
                return (full, encode_region(&canvases[0], full));
            }
            let (prev, cur) = (&canvases[i - 1], &canvases[i]);
            let Some((x, y, rw, rh)) = dirty_rect(prev, cur, wu, hu, ch, hidden_equal) else {
                // Nothing changed: a single pixel that leaves the canvas as it is.
                let r = Region { x: 0, y: 0, w: 1, h: 1, over: alpha };
                let px = if alpha { vec![0, 0, 0, 0] } else { cur[..3].to_vec() };
                return (r, encode_region(&px, r));
            };
            let src = Region { x, y, w: rw, h: rh, over: false };
            let source = encode_region(&region_pixels(prev, cur, wu, ch, src, hidden_equal).unwrap(), src);
            if alpha {
                let over = Region { over: true, ..src };
                if let Some(px) = region_pixels(prev, cur, wu, ch, over, hidden_equal) {
                    let o = encode_region(&px, over);
                    if o.len() < source.len() {
                        return (over, o);
                    }
                }
            }
            (src, source)
        })
        .collect();

    let mut out = SIGNATURE.to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, color_type, 0, 0, 0]);
    push_chunk(&mut out, b"IHDR", &ihdr);

    let mut actl = Vec::with_capacity(8);
    actl.extend_from_slice(&(anim.frames.len() as u32).to_be_bytes());
    actl.extend_from_slice(&(if anim.loop_forever { 0u32 } else { 1u32 }).to_be_bytes());
    push_chunk(&mut out, b"acTL", &actl);

    // APNG sequence numbers count fcTL and fdAT chunks, not frames: every chunk after acTL
    // takes the next number, while IDAT has no sequence number.
    let mut seq = 0u32;
    for (i, (f, (region, idat))) in anim.frames.iter().zip(planned).enumerate() {
        push_chunk(&mut out, b"fcTL", &frame_ctl(seq, region, normalize_delay(f.duration_ms)));
        seq += 1;
        if i == 0 {
            for chunk in idat.chunks(1 << 20) {
                push_chunk(&mut out, b"IDAT", chunk);
            }
        } else {
            for chunk in idat.chunks(1 << 20) {
                let mut body = Vec::with_capacity(4 + chunk.len());
                body.extend_from_slice(&seq.to_be_bytes());
                body.extend_from_slice(chunk);
                push_chunk(&mut out, b"fdAT", &body);
                seq += 1;
            }
        }
    }
    push_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}
