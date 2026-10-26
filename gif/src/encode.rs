//! GIF writer for animations (a still is a one-frame animation).
//!
//! All frames share one global palette. After the first frame each frame is coded as the rectangle that changed
//! since the previous one, with the pixels inside it that did not change drawn in the transparent index so the
//! previous frame shows through. A frame that makes a visible pixel transparent again cannot be drawn over its
//! predecessor, so the predecessor is disposed to the background and that frame is drawn on a blank canvas.
//! Pixels are transparent below 50% alpha (GIF has no partial transparency).

use rc_core::animation::Animation;
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image};
use rc_core::quant::{self, RgbPalette};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Palette size (2-256), including the transparent entry.
    pub colors: usize,
    /// Error diffusion strength (0-1) for stills. Animations are never dithered: the noise would change from frame
    /// to frame and defeat the frame differencing.
    pub dither: f32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions { colors: 256, dither: 0.5 }
    }
}

/// Alpha below this is transparent.
const OPAQUE_FROM: u8 = 128;

fn rgba(img: &Image) -> Vec<[u8; 4]> {
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

/// Palette entries a canvas sequence needs besides its colors: one for transparency, or for "unchanged" in frames
/// drawn over their predecessor.
fn reserved(frames: &[Vec<[u8; 4]>]) -> usize {
    let transparent = frames.iter().any(|f| f.iter().any(|p| p[3] < OPAQUE_FROM));
    (transparent || frames.len() > 1) as usize
}

/// The distinct opaque colors of all frames, or `None` as soon as there are more than `limit`.
fn distinct_colors(frames: &[Vec<[u8; 4]>], limit: usize) -> Option<Vec<[u8; 3]>> {
    let mut set: HashSet<[u8; 3]> = HashSet::new();
    for f in frames {
        for p in f.iter().filter(|p| p[3] >= OPAQUE_FROM) {
            if set.insert([p[0], p[1], p[2]]) && set.len() > limit {
                return None;
            }
        }
    }
    Some(set.into_iter().collect())
}

fn canvases(anim: &Animation) -> Result<Vec<Vec<[u8; 4]>>> {
    let (w, h) = (anim.width as usize, anim.height as usize);
    if w == 0 || h == 0 || w > 65535 || h > 65535 {
        return Err(Error::unsupported("GIF canvases are limited to 65535x65535 pixels"));
    }
    if anim.frames.is_empty() {
        return Err(Error::invalid("animation without frames"));
    }
    anim.frames
        .iter()
        .map(|f| {
            if (f.image.width as usize, f.image.height as usize) != (w, h) {
                return Err(Error::invalid("animation frames differ in size"));
            }
            Ok(rgba(&f.image))
        })
        .collect()
}

/// True if `anim` fits a `colors`-entry GIF palette without changing any color: the colors of every frame, plus the
/// transparent entry when one is needed, fit.
pub fn palette_is_exact(anim: &Animation, colors: usize) -> bool {
    let Ok(frames) = canvases(anim) else { return false };
    let limit = colors.clamp(2, 256) - reserved(&frames);
    distinct_colors(&frames, limit).is_some()
}

pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let (w, h) = (anim.width as usize, anim.height as usize);
    let mut frames = canvases(anim)?;
    let n = frames.len();
    let reserve = reserved(&frames);
    let has_t = reserve == 1;
    let limit = opts.colors.clamp(2, 256) - reserve;

    // A still that needs more colors than it may have is dithered down first (transparent pixels are left out of
    // the palette by giving them a color that is already there).
    let mut exact = distinct_colors(&frames, limit);
    if exact.is_none() && n == 1 && opts.dither > 0.0 {
        let fill = frames[0].iter().find(|p| p[3] >= OPAQUE_FROM).map_or([0, 0, 0], |p| [p[0], p[1], p[2]]);
        let mut rgb = Vec::with_capacity(w * h * 3);
        for p in &frames[0] {
            rgb.extend_from_slice(if p[3] >= OPAQUE_FROM { &p[..3] } else { &fill });
        }
        let img = Image::new(w as u32, h as u32, ColorType::Rgb, 8, rgb)?;
        let q = quant::quantize(&img, limit, opts.dither);
        for (p, o) in frames[0].iter_mut().zip(q.data.chunks_exact(3)) {
            if p[3] >= OPAQUE_FROM {
                p[..3].copy_from_slice(o);
            }
        }
        exact = distinct_colors(&frames, limit);
    }
    let sample: Vec<[u8; 3]> = match exact {
        Some(colors) => colors,
        None => {
            // Sample very large animations: the palette needs a representative set, not every pixel.
            let total = n * w * h;
            let stride = (total / (1 << 22)).max(1);
            frames
                .iter()
                .flatten()
                .filter(|p| p[3] >= OPAQUE_FROM)
                .step_by(stride)
                .map(|p| [p[0], p[1], p[2]])
                .collect()
        }
    };
    let pal = RgbPalette::new(&sample, limit);
    let k = pal.colors().len();
    let transparent = k as u8; // only meaningful when `has_t`
    let table_len = (k + reserve).next_power_of_two().max(2);
    let bits = table_len.trailing_zeros() as u8; // 1..=8

    let mut cache: HashMap<[u8; 3], u8> = HashMap::new();
    let idx: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            f.iter()
                .map(|p| {
                    if p[3] < OPAQUE_FROM {
                        transparent
                    } else {
                        let c = [p[0], p[1], p[2]];
                        *cache.entry(c).or_insert_with(|| pal.nearest(c) as u8)
                    }
                })
                .collect()
        })
        .collect();

    // Frame i is drawn on a blank canvas when it shows nothing of frame i-1 (which is then disposed to background).
    let blank_before: Vec<bool> = (0..n)
        .map(|i| i == 0 || (has_t && (0..w * h).any(|p| idx[i][p] == transparent && idx[i - 1][p] != transparent)))
        .collect();

    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h as u16).to_le_bytes());
    out.push(0x80 | 0x70 | (bits - 1));
    out.extend_from_slice(&[0, 0]);
    for i in 0..table_len {
        out.extend_from_slice(&pal.colors().get(i).copied().unwrap_or([0, 0, 0]));
    }
    if anim.loop_forever && n > 1 {
        out.extend_from_slice(&[0x21, 0xFF, 11]);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.extend_from_slice(&[3, 1, 0, 0, 0]);
    }

    let min_code = bits.max(2);
    for i in 0..n {
        let base: Option<&[u8]> = if blank_before[i] { None } else { Some(&idx[i - 1]) };
        let dispose = if i + 1 < n && blank_before[i + 1] { 2u8 } else { 1u8 };
        let cur = &idx[i];
        let differs = |p: usize| match base {
            Some(b) => cur[p] != b[p],
            None => cur[p] != transparent || !has_t,
        };
        // Bounding box of what must be drawn; a frame that is disposed to background must also cover every visible
        // pixel, since only its own rectangle is cleared.
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                if differs(p) || (dispose == 2 && has_t && cur[p] != transparent) {
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
        let (rw, rh) = (x1 - x0, y1 - y0);
        let mut pixels = Vec::with_capacity(rw * rh);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = y * w + x;
                pixels.push(match base {
                    Some(b) if has_t && cur[p] == b[p] => transparent,
                    _ => cur[p],
                });
            }
        }
        let delay = ((anim.frames[i].duration_ms + 5) / 10).clamp(2, 65535) as u16;
        out.extend_from_slice(&[0x21, 0xF9, 4, dispose << 2 | has_t as u8]);
        out.extend_from_slice(&delay.to_le_bytes());
        out.extend_from_slice(&[transparent, 0]);
        out.push(0x2C);
        for v in [x0, y0, rw, rh] {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        out.push(0);
        out.push(min_code);
        for block in lzw_encode(&pixels, min_code).chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0);
    }
    out.push(0x3B);
    Ok(out)
}

/// Packs variable-width codes least significant bit first.
struct BitSink {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitSink {
    fn put(&mut self, code: u32, size: u32) {
        self.acc |= code << self.n;
        self.n += size;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// GIF's variable-width LZW. `indices` must not be empty and every index must be below `1 << min_code`.
pub(crate) fn lzw_encode(indices: &[u8], min_code: u8) -> Vec<u8> {
    const SLOTS: usize = 1 << 13;
    const EMPTY: u32 = u32::MAX;
    let clear = 1u32 << min_code;
    let end = clear + 1;
    let first_free = end + 1;
    let mut keys = vec![EMPTY; SLOTS];
    let mut vals = vec![0u16; SLOTS];
    let mut sink = BitSink { out: Vec::with_capacity(indices.len() / 2 + 16), acc: 0, n: 0 };
    let mut size = min_code as u32 + 1;
    let mut next = first_free;
    sink.put(clear, size);
    let mut prefix = indices[0] as u32;
    for &b in &indices[1..] {
        let key = prefix << 8 | b as u32;
        let mut slot = (key.wrapping_mul(2_654_435_761) >> 19) as usize;
        let found = loop {
            if keys[slot] == key {
                break Some(vals[slot]);
            }
            if keys[slot] == EMPTY {
                break None;
            }
            slot = (slot + 1) & (SLOTS - 1);
        };
        if let Some(code) = found {
            prefix = code as u32;
            continue;
        }
        sink.put(prefix, size);
        // The decoder is one entry behind: it widens its codes once the table reaches the next power of two.
        if next >= 1 << size && size < 12 {
            size += 1;
        }
        if next < 4096 {
            keys[slot] = key;
            vals[slot] = next as u16;
            next += 1;
        } else {
            sink.put(clear, size);
            keys.fill(EMPTY);
            size = min_code as u32 + 1;
            next = first_free;
        }
        prefix = b as u32;
    }
    sink.put(prefix, size);
    if next >= 1 << size && size < 12 {
        size += 1;
    }
    sink.put(end, size);
    sink.finish()
}
