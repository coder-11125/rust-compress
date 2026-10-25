//! GIF decoder: stills, and animations as full-canvas frames. GIF87a/GIF89a, global and local palettes, interlacing and
//! transparency. [`decode`] refuses animated GIFs rather than flattening them to their first frame; use
//! [`decode_animation`] for those.

use rc_core::animation::{AnimFrame, Animation, MAX_ANIMATION_PIXELS, normalize_delay};
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image, MAX_PIXELS};

pub const SIGNATURES: [&[u8]; 2] = [b"GIF87a", b"GIF89a"];

pub fn is_gif(data: &[u8]) -> bool {
    SIGNATURES.iter().any(|s| data.starts_with(s))
}

struct Cursor<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end =
            self.pos.checked_add(n).filter(|&e| e <= self.d.len()).ok_or_else(|| Error::invalid("truncated GIF"))?;
        let s = &self.d[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<usize> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]) as usize)
    }

    /// Concatenate a chain of length-prefixed sub-blocks.
    fn sub_blocks(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let n = self.u8()? as usize;
            if n == 0 {
                return Ok(out);
            }
            out.extend_from_slice(self.take(n)?);
        }
    }

    fn skip_sub_blocks(&mut self) -> Result<()> {
        loop {
            let n = self.u8()? as usize;
            if n == 0 {
                return Ok(());
            }
            self.take(n)?;
        }
    }
}

fn color_table(c: &mut Cursor, packed: u8) -> Result<Vec<[u8; 3]>> {
    let n = 2usize << (packed & 7);
    Ok(c.take(n * 3)?.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
}

/// Variable-width LZW as used by GIF. Produces at most `limit` palette indices.
fn lzw_decode(data: &[u8], min_code_size: u8, limit: usize) -> Result<Vec<u8>> {
    if !(1..=11).contains(&min_code_size) {
        return Err(Error::invalid("bad GIF LZW code size"));
    }
    let clear = 1usize << min_code_size;
    let end = clear + 1;
    // Each table entry is (prefix code, byte); strings are rebuilt backwards.
    let mut prefix = vec![0u16; 4096];
    let mut suffix = vec![0u8; 4096];
    for i in 0..clear {
        suffix[i] = i as u8;
    }
    let mut out = Vec::with_capacity(limit);
    let mut size = min_code_size as usize + 1;
    let mut next = end + 1;
    let mut prev: Option<usize> = None;
    let (mut bits, mut nbits, mut at) = (0u32, 0usize, 0usize);
    let mut stack = Vec::new();
    while out.len() < limit {
        while nbits < size {
            let Some(&b) = data.get(at) else {
                return Ok(out); // ran out of data: keep what decoded
            };
            bits |= (b as u32) << nbits;
            nbits += 8;
            at += 1;
        }
        let code = (bits & ((1 << size) - 1)) as usize;
        bits >>= size;
        nbits -= size;
        if code == clear {
            size = min_code_size as usize + 1;
            next = end + 1;
            prev = None;
            continue;
        }
        if code == end {
            break;
        }
        match prev {
            None => {
                if code >= clear {
                    return Err(Error::invalid("bad GIF LZW stream"));
                }
                out.push(code as u8);
            }
            Some(p) => {
                // Expand `code` (or, for the not-yet-defined code, prev + its own first byte).
                let known = code < next;
                if !known && code != next {
                    return Err(Error::invalid("bad GIF LZW code"));
                }
                stack.clear();
                let mut c = if known { code } else { p };
                while c >= clear {
                    stack.push(suffix[c]);
                    c = prefix[c] as usize;
                }
                stack.push(suffix[c]);
                let first_byte = *stack.last().unwrap();
                out.extend(stack.iter().rev());
                if !known {
                    out.push(first_byte);
                }
                if next < 4096 {
                    prefix[next] = p as u16;
                    suffix[next] = first_byte;
                    next += 1;
                    if next == (1 << size) && size < 12 {
                        size += 1;
                    }
                }
            }
        }
        prev = Some(code);
    }
    out.truncate(limit);
    Ok(out)
}

/// The single image of a GIF: its place on the canvas, palette and palette indices.
struct Frame {
    left: usize,
    top: usize,
    w: usize,
    h: usize,
    table: Vec<[u8; 3]>,
    idx: Vec<u8>,
}

/// Decode the first (and only) frame of a GIF.
pub fn decode(data: &[u8]) -> Result<Image> {
    if !is_gif(data) {
        return Err(Error::invalid("not a GIF file"));
    }
    let mut c = Cursor { d: data, pos: 6 };
    let (sw, sh) = (c.u16()?, c.u16()?);
    let packed = c.u8()?;
    let _background = c.u8()?;
    let _aspect = c.u8()?;
    if sw == 0 || sh == 0 || (sw as u64) * (sh as u64) > MAX_PIXELS {
        return Err(Error::invalid("bad GIF dimensions"));
    }
    let global = if packed & 0x80 != 0 { Some(color_table(&mut c, packed)?) } else { None };

    let mut transparent: Option<u8> = None;
    let mut frame: Option<Frame> = None;
    loop {
        match c.u8()? {
            0x21 => {
                let label = c.u8()?;
                if label == 0xF9 {
                    let b = c.sub_blocks()?;
                    // Transparency only applies to the frame that follows; remember it for that frame.
                    if frame.is_none() && b.len() >= 4 && b[0] & 1 != 0 {
                        transparent = Some(b[3]);
                    }
                } else {
                    c.skip_sub_blocks()?;
                }
            }
            0x2C => {
                if frame.is_some() {
                    return Err(Error::unsupported("animated GIF"));
                }
                let (left, top, w, h) = (c.u16()?, c.u16()?, c.u16()?, c.u16()?);
                let fp = c.u8()?;
                if w == 0 || h == 0 || left + w > sw || top + h > sh {
                    return Err(Error::invalid("GIF frame outside the canvas"));
                }
                let table = if fp & 0x80 != 0 { Some(color_table(&mut c, fp)?) } else { global.clone() };
                let table = table.ok_or_else(|| Error::invalid("GIF without a color table"))?;
                let min_code = c.u8()?;
                let packed_data = c.sub_blocks()?;
                let mut idx = lzw_decode(&packed_data, min_code, w * h)?;
                if idx.len() < w * h {
                    idx.resize(w * h, 0);
                }
                if fp & 0x40 != 0 {
                    idx = deinterlace(&idx, w, h);
                }
                frame = Some(Frame { left, top, w, h, table, idx });
            }
            0x3B => break,
            _ => return Err(Error::invalid("bad GIF block")),
        }
    }
    let Frame { left, top, w, h, table, idx } = frame.ok_or_else(|| Error::invalid("GIF contains no image"))?;
    let covers_canvas = (left, top, w, h) == (0, 0, sw, sh);
    let has_alpha = transparent.is_some() || !covers_canvas;
    let ch = if has_alpha { 4 } else { 3 };
    let mut out = vec![0u8; sw * sh * ch];
    for y in 0..h {
        for x in 0..w {
            let i = idx[y * w + x];
            if transparent == Some(i) {
                continue; // stays fully transparent
            }
            let rgb = table.get(i as usize).copied().unwrap_or([0, 0, 0]);
            let o = ((top + y) * sw + left + x) * ch;
            out[o..o + 3].copy_from_slice(&rgb);
            if has_alpha {
                out[o + 3] = 255;
            }
        }
    }
    Image::new(sw as u32, sh as u32, if has_alpha { ColorType::Rgba } else { ColorType::Rgb }, 8, out)
}

/// Decode every frame of a GIF into full-canvas frames, honouring disposal methods, transparency, local palettes
/// and the loop extension. A still GIF gives a one-frame animation.
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    if !is_gif(data) {
        return Err(Error::invalid("not a GIF file"));
    }
    let mut c = Cursor { d: data, pos: 6 };
    let (sw, sh) = (c.u16()?, c.u16()?);
    let packed = c.u8()?;
    let _background = c.u8()?;
    let _aspect = c.u8()?;
    if sw == 0 || sh == 0 || (sw as u64) * (sh as u64) > MAX_PIXELS {
        return Err(Error::invalid("bad GIF dimensions"));
    }
    let global = if packed & 0x80 != 0 { Some(color_table(&mut c, packed)?) } else { None };

    // RGBA canvas; shown frames are snapshots of it.
    let mut canvas = vec![0u8; sw * sh * 4];
    let mut previous: Option<Vec<u8>> = None;
    let mut frames: Vec<(Vec<u8>, u32)> = Vec::new();
    let mut gce: Option<(u8, u32, Option<u8>)> = None; // (disposal, delay ms, transparent index)
    let mut loops: Option<u16> = None;
    let mut last_disposal = (0u8, 0usize, 0usize, 0usize, 0usize);
    loop {
        match c.u8()? {
            0x21 => {
                let label = c.u8()?;
                if label == 0xF9 {
                    let b = c.sub_blocks()?;
                    if b.len() >= 4 {
                        let delay = u16::from_le_bytes([b[1], b[2]]) as u32 * 10;
                        gce = Some(((b[0] >> 2) & 7, delay, (b[0] & 1 != 0).then_some(b[3])));
                    }
                } else if label == 0xFF {
                    // Application extension: the loop count lives in a NETSCAPE2.0 / ANIMEXTS1.0 sub-block.
                    let n = c.u8()? as usize;
                    let id = c.take(n)?.to_vec();
                    let body = c.sub_blocks()?;
                    if (id.starts_with(b"NETSCAPE2.0") || id.starts_with(b"ANIMEXTS1.0"))
                        && body.len() >= 3
                        && body[0] == 1
                    {
                        loops = Some(u16::from_le_bytes([body[1], body[2]]));
                    }
                } else {
                    c.skip_sub_blocks()?;
                }
            }
            0x2C => {
                let (left, top, w, h) = (c.u16()?, c.u16()?, c.u16()?, c.u16()?);
                let fp = c.u8()?;
                if w == 0 || h == 0 || left + w > sw || top + h > sh {
                    return Err(Error::invalid("GIF frame outside the canvas"));
                }
                let table = if fp & 0x80 != 0 { Some(color_table(&mut c, fp)?) } else { global.clone() };
                let table = table.ok_or_else(|| Error::invalid("GIF without a color table"))?;
                let min_code = c.u8()?;
                let packed_data = c.sub_blocks()?;
                let mut idx = lzw_decode(&packed_data, min_code, w * h)?;
                idx.resize(w * h, 0);
                if fp & 0x40 != 0 {
                    idx = deinterlace(&idx, w, h);
                }
                if (frames.len() as u64 + 1) * (sw * sh) as u64 > MAX_ANIMATION_PIXELS {
                    return Err(Error::unsupported("GIF animation too large"));
                }
                // Dispose of the previous frame first.
                let (disposal, dl, dt, dw, dh) = last_disposal;
                match disposal {
                    2 => {
                        for y in dt..dt + dh {
                            canvas[(y * sw + dl) * 4..(y * sw + dl + dw) * 4].fill(0);
                        }
                    }
                    3 => {
                        if let Some(p) = previous.take() {
                            canvas = p;
                        }
                    }
                    _ => {}
                }
                let (disposal, delay, transparent) = gce.take().unwrap_or((0, 0, None));
                if disposal == 3 {
                    previous = Some(canvas.clone());
                }
                for y in 0..h {
                    for x in 0..w {
                        let i = idx[y * w + x];
                        if transparent == Some(i) {
                            continue;
                        }
                        let rgb = table.get(i as usize).copied().unwrap_or([0, 0, 0]);
                        let o = ((top + y) * sw + left + x) * 4;
                        canvas[o..o + 3].copy_from_slice(&rgb);
                        canvas[o + 3] = 255;
                    }
                }
                frames.push((canvas.clone(), normalize_delay(delay)));
                last_disposal = (disposal, left, top, w, h);
            }
            0x3B => break,
            _ => return Err(Error::invalid("bad GIF block")),
        }
    }
    if frames.is_empty() {
        return Err(Error::invalid("GIF contains no image"));
    }
    let opaque = frames.iter().all(|(f, _)| f.chunks_exact(4).all(|p| p[3] == 255));
    let mut out = Vec::with_capacity(frames.len());
    for (rgba, ms) in frames {
        let image = if opaque {
            let rgb = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            Image::new(sw as u32, sh as u32, ColorType::Rgb, 8, rgb)?
        } else {
            Image::new(sw as u32, sh as u32, ColorType::Rgba, 8, rgba)?
        };
        out.push(AnimFrame { image, duration_ms: ms });
    }
    // No loop extension means play once; any repeat count (0 meaning forever) is played forever.
    let loop_forever = loops.is_some();
    Ok(Animation { width: sw as u32, height: sh as u32, frames: out, loop_forever })
}

/// Rows of an interlaced frame arrive in four passes; put them back in order.
fn deinterlace(idx: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; idx.len()];
    let mut src = 0;
    for (start, step) in [(0, 8), (4, 8), (2, 4), (1, 2)] {
        let mut y = start;
        while y < h {
            out[y * w..(y + 1) * w].copy_from_slice(&idx[src * w..(src + 1) * w]);
            src += 1;
            y += step;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lzw_decodes_the_specification_example() {
        // The canonical example from the GIF specification: 10x10 image, min code size 2.
        let packed = [
            0x8C, 0x2D, 0x99, 0x87, 0x2A, 0x1C, 0xDC, 0x33, 0xA0, 0x02, 0x75, 0xEC, 0x95, 0xFA, 0xA8, 0xDE, 0x60, 0x8C,
            0x04, 0x91, 0x4C, 0x01,
        ];
        let px = lzw_decode(&packed, 2, 100).unwrap();
        assert_eq!(px.len(), 100);
        assert_eq!(&px[..10], &[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode(b"GIF89a").is_err());
        assert!(decode(b"GIF89a\x01\x00\x01\x00\x00\x00\x00,").is_err());
        // Arbitrary bytes must not panic, whatever they decode to.
        let _ = lzw_decode(&[0xFF; 64], 8, 1000);
        let _ = lzw_decode(&(0..=255).collect::<Vec<u8>>(), 2, 5000);
    }

    #[test]
    fn deinterlace_restores_row_order() {
        let (w, h) = (1usize, 10usize);
        // Row order delivered by the four passes for h = 10: 0,8, 4, 2,6, 1,3,5,7,9.
        let order = [0, 8, 4, 2, 6, 1, 3, 5, 7, 9];
        let stream: Vec<u8> = order.iter().map(|&r| r as u8).collect();
        let out = deinterlace(&stream, w, h);
        assert_eq!(out, (0..10).collect::<Vec<u8>>());
    }
}
