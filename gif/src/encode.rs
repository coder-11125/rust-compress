//! GIF writer for animations (a still is a one-frame animation).
//!
//! Frames normally share one global palette; when the union of frame colors does not fit but every frame does,
//! each image gets a local color table instead so the pixels stay exact. After the first frame each frame is coded
//! as the rectangle that changed since the previous one, with the pixels inside it that did not change drawn in
//! the transparent index so the previous frame shows through. A frame that makes a visible pixel transparent again
//! cannot be drawn over its predecessor, so the predecessor is disposed to the background and that frame is drawn
//! on a blank canvas. Pixels are transparent below 50% alpha (GIF has no partial transparency).

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
    /// Lossiness, 0 (exact, the default) to 200, on a scale comparable to gifsicle's `--lossy`. Pixels may be
    /// replaced by a nearby palette colour, or by "unchanged from the previous frame", when that lets the LZW coder
    /// extend a match; larger values allow bigger colour differences. Applies to the shared-palette encoder (not the
    /// per-frame local palettes used when no single palette fits).
    pub lossy: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions { colors: 256, dither: 0.5, lossy: 0 }
    }
}

/// Squared distance between two colours (plain RGB; the lossiness thresholds are tuned against it).
fn dist2(a: [u8; 3], b: [u8; 3]) -> i32 {
    let d = |x: u8, y: u8| (x as i32 - y as i32).pow(2);
    d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
}

/// For every palette entry, the other entries within the lossiness threshold, nearest first (at most 16).
fn near_colors(pal: &[[u8; 3]], lossy: u32) -> (i32, Vec<Vec<u8>>) {
    if lossy == 0 {
        return (0, Vec::new());
    }
    let t = lossy.min(200) as f32 * 0.4;
    let t2 = (t * t) as i32;
    let nb = (0..pal.len())
        .map(|c| {
            let mut v: Vec<(i32, u8)> = (0..pal.len())
                .filter(|&d| d != c)
                .map(|d| (dist2(pal[c], pal[d]), d as u8))
                .filter(|&(d2, _)| d2 <= t2)
                .collect();
            v.sort_unstable();
            v.truncate(16);
            v.into_iter().map(|(_, d)| d).collect()
        })
        .collect();
    (t2, nb)
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

/// The visible colour as one comparable key: `None` is transparent, RGB is offset by one so black is not confused.
fn opaque_key(p: [u8; 4]) -> Option<u32> {
    (p[3] >= OPAQUE_FROM).then(|| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)
}

fn key_color(k: u32) -> [u8; 3] {
    [(k >> 16) as u8, (k >> 8) as u8, k as u8]
}

/// One frame of the exact per-frame-palette encoding: the rectangle that must be drawn and its pixels as visible
/// colour keys (`0` = transparent or unchanged, otherwise `opaque_key + 1`).
struct LocalPlan {
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
    dispose: u8,
    keys: Vec<u32>,
    /// Distinct non-zero keys in the rectangle.
    colors: usize,
}

/// Plan every frame's rectangle and check that each fits a palette of `colors` entries (counting the transparent
/// entry when some pixel is left unchanged). A frame may use many colours overall: only the changed rectangle counts.
/// `None` when a frame's rectangle needs more.
fn local_plan(frames: &[Vec<[u8; 4]>], w: usize, h: usize, colors: usize) -> Option<Vec<LocalPlan>> {
    let n = frames.len();
    if n < 2 {
        return None;
    }
    let vis: Vec<Vec<u32>> =
        frames.iter().map(|f| f.iter().map(|p| opaque_key(*p).map_or(0, |k| k + 1)).collect()).collect();
    let blank_before: Vec<bool> =
        (0..n).map(|i| i == 0 || (0..w * h).any(|p| vis[i][p] == 0 && vis[i - 1][p] != 0)).collect();
    let mut plans = Vec::with_capacity(n);
    for i in 0..n {
        let cur = &vis[i];
        let base = if blank_before[i] { None } else { Some(&vis[i - 1]) };
        let dispose = if i + 1 < n && blank_before[i + 1] { 2u8 } else { 1u8 };
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let differs = match base {
                    Some(b) => cur[p] != b[p],
                    None => cur[p] != 0,
                };
                if differs || (dispose == 2 && cur[p] != 0) {
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
        let mut keys = Vec::with_capacity(rw * rh);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = y * w + x;
                keys.push(match base {
                    Some(b) if cur[p] == b[p] => 0,
                    _ => cur[p],
                });
            }
        }
        let mut set: HashSet<u32> = HashSet::new();
        for &k in &keys {
            if k != 0 {
                set.insert(k);
            }
        }
        let needs_transparent = keys.contains(&0);
        if set.len() + needs_transparent as usize > colors {
            return None;
        }
        plans.push(LocalPlan { x0, y0, rw, rh, dispose, keys, colors: set.len() });
    }
    Some(plans)
}

/// Exact multi-palette encoding for animations whose changed rectangles each fit a palette even though their union
/// does not. Returns `None` when some rectangle needs more than `colors` entries, leaving the caller to quantize.
fn encode_local_palettes(
    frames: &[Vec<[u8; 4]>],
    delays: &[u32],
    w: usize,
    h: usize,
    colors: usize,
    loop_forever: bool,
    lossy: u32,
) -> Option<Vec<u8>> {
    if lossy > 0 {
        return encode_local_lossy(frames, delays, w, h, colors.clamp(2, 256), loop_forever, lossy);
    }
    let plans = local_plan(frames, w, h, colors.clamp(2, 256))?;
    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h as u16).to_le_bytes());
    out.push(0x70); // No global colour table: every image carries its own.
    out.extend_from_slice(&[0, 0]);
    if loop_forever {
        out.extend_from_slice(&[0x21, 0xFF, 11]);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.extend_from_slice(&[3, 1, 0, 0, 0]);
    }
    for (pl, &delay_ms) in plans.iter().zip(delays) {
        // Table: the transparent entry first if some pixel is unchanged, then the rectangle's colours.
        let has_transparent = pl.keys.contains(&0);
        let mut keys_used: Vec<u32> = pl.keys.iter().copied().filter(|&k| k != 0).collect();
        keys_used.sort_unstable();
        keys_used.dedup();
        debug_assert_eq!(keys_used.len(), pl.colors);
        let first = has_transparent as usize;
        let map: HashMap<u32, u8> = keys_used.iter().enumerate().map(|(i, &k)| (k, (i + first) as u8)).collect();
        let mut table: Vec<[u8; 3]> = Vec::with_capacity(keys_used.len() + first);
        if has_transparent {
            table.push([0, 0, 0]);
        }
        table.extend(keys_used.iter().map(|&k| key_color(k - 1)));
        let pixels: Vec<u8> = pl.keys.iter().map(|&k| if k == 0 { 0 } else { map[&k] }).collect();
        let table_len = table.len().next_power_of_two().max(2);
        let bits = table_len.trailing_zeros() as u8;
        let delay = ((delay_ms + 5) / 10).clamp(2, 65535) as u16;
        out.extend_from_slice(&[
            0x21,
            0xF9,
            4,
            pl.dispose << 2 | has_transparent as u8,
            delay as u8,
            (delay >> 8) as u8,
            0,
            0,
        ]);
        out.push(0x2C);
        for v in [pl.x0, pl.y0, pl.rw, pl.rh] {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        out.push(0x80 | (bits - 1));
        for t in 0..table_len {
            out.extend_from_slice(table.get(t).copied().unwrap_or([0, 0, 0]).as_slice());
        }
        let min_code = bits.max(2);
        out.push(min_code);
        for block in lzw_encode(&pixels, min_code).chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0);
    }
    out.push(0x3B);
    Some(out)
}

/// The lossy variant of [`encode_local_palettes`]: frames are coded one after another, each against what the previous
/// one actually displays, so a substitution made in one frame is what the next frame compares with. Every pixel stays
/// within the lossiness budget of its source colour.
fn encode_local_lossy(
    frames: &[Vec<[u8; 4]>],
    delays: &[u32],
    w: usize,
    h: usize,
    colors: usize,
    loop_forever: bool,
    lossy: u32,
) -> Option<Vec<u8>> {
    let n = frames.len();
    if n < 2 {
        return None;
    }
    let vis: Vec<Vec<u32>> =
        frames.iter().map(|f| f.iter().map(|p| opaque_key(*p).map_or(0, |k| k + 1)).collect()).collect();
    let blank_before: Vec<bool> =
        (0..n).map(|i| i == 0 || (0..w * h).any(|p| vis[i][p] == 0 && vis[i - 1][p] != 0)).collect();
    let t = lossy.min(200) as f32 * 0.4;
    let t2 = (t * t) as i32;
    let color = |k: u32| key_color(k - 1);

    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h as u16).to_le_bytes());
    out.push(0x70);
    out.extend_from_slice(&[0, 0]);
    if loop_forever {
        out.extend_from_slice(&[0x21, 0xFF, 11]);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.extend_from_slice(&[3, 1, 0, 0, 0]);
    }
    // What the viewer sees after the frames written so far (0 = transparent).
    let mut shown: Vec<u32> = vec![0; w * h];
    for i in 0..n {
        let src = &vis[i];
        let mut cur = src.clone();
        let base: Option<&[u32]> = if blank_before[i] { None } else { Some(&shown) };
        if let Some(b) = base {
            for p in 0..w * h {
                if cur[p] != 0 && b[p] != 0 && cur[p] != b[p] && dist2(color(cur[p]), color(b[p])) <= t2 {
                    cur[p] = b[p];
                }
            }
        }
        let dispose = if i + 1 < n && blank_before[i + 1] { 2u8 } else { 1u8 };
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let differs = match base {
                    Some(b) => cur[p] != b[p],
                    None => cur[p] != 0,
                };
                if differs || (dispose == 2 && cur[p] != 0) {
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
        let pos = |k: usize| (y0 + k / rw) * w + x0 + k % rw;
        let keys: Vec<u32> = (0..rw * rh)
            .map(|k| {
                let p = pos(k);
                match base {
                    Some(b) if cur[p] == b[p] => 0,
                    _ => cur[p],
                }
            })
            .collect();
        let has_transparent = keys.contains(&0);
        let mut used: Vec<u32> = keys.iter().copied().filter(|&k| k != 0).collect();
        used.sort_unstable();
        used.dedup();
        if used.len() + has_transparent as usize > colors {
            return None;
        }
        let first = has_transparent as usize;
        let map: HashMap<u32, u8> = used.iter().enumerate().map(|(j, &k)| (k, (j + first) as u8)).collect();
        // Table entry -> displayed key (0 for the transparent entry).
        let mut entry_key: Vec<u32> = Vec::with_capacity(used.len() + first);
        if has_transparent {
            entry_key.push(0);
        }
        entry_key.extend(used.iter().copied());
        let table: Vec<[u8; 3]> = entry_key.iter().map(|&k| if k == 0 { [0, 0, 0] } else { color(k) }).collect();
        let mut pixels: Vec<u8> = keys.iter().map(|&k| if k == 0 { 0 } else { map[&k] }).collect();
        let (_, near) = near_colors(&table, lossy);
        let orig = pixels.clone();
        let table_len = table.len().next_power_of_two().max(2);
        let bits = table_len.trailing_zeros() as u8;
        let min_code = bits.max(2);
        let mut fill = |k: usize, cand: &mut Vec<u8>| {
            let p = pos(k);
            if src[p] == 0 {
                return;
            }
            let want = color(src[p]);
            let ok = |c: u8| (c != 0 || !has_transparent) && dist2(table[c as usize], want) <= t2;
            let v = orig[k];
            if v == 0 && has_transparent {
                // Unchanged pixel: any entry close to the source colour would do as well.
                if base.is_some() {
                    cand.extend(
                        entry_key.iter().enumerate().skip(1).filter(|&(c, _)| ok(c as u8)).map(|(c, _)| c as u8),
                    );
                    cand.sort_by_key(|&c| dist2(table[c as usize], want));
                    cand.truncate(16);
                }
            } else {
                if has_transparent
                    && let Some(b) = base
                    && b[p] != 0
                    && dist2(color(b[p]), want) <= t2
                {
                    cand.push(0);
                }
                cand.extend(near[v as usize].iter().copied().filter(|&c| ok(c)));
            }
        };
        let data = lzw_core(&mut pixels, min_code, Some(&mut fill));
        for (k, (&new, &old)) in pixels.iter().zip(&orig).enumerate() {
            if new != old {
                let p = pos(k);
                cur[p] = if new == 0 && has_transparent { base.map_or(0, |b| b[p]) } else { entry_key[new as usize] };
            }
        }
        let delay = ((delays[i] + 5) / 10).clamp(2, 65535) as u16;
        out.extend_from_slice(&[
            0x21,
            0xF9,
            4,
            dispose << 2 | has_transparent as u8,
            delay as u8,
            (delay >> 8) as u8,
            0,
            0,
        ]);
        out.push(0x2C);
        for v in [x0, y0, rw, rh] {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        out.push(0x80 | (bits - 1));
        for t in 0..table_len {
            out.extend_from_slice(table.get(t).copied().unwrap_or([0, 0, 0]).as_slice());
        }
        out.push(min_code);
        for block in data.chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0);
        shown = cur;
    }
    out.push(0x3B);
    Some(out)
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

/// True if `anim` fits `colors`-entry GIF palettes without changing any color: either one palette for every frame,
/// or (for animations) one local palette per frame.
pub fn palette_is_exact(anim: &Animation, colors: usize) -> bool {
    let Ok(frames) = canvases(anim) else { return false };
    let colors = colors.clamp(2, 256);
    let (w, h) = (anim.width as usize, anim.height as usize);
    let limit = colors - reserved(&frames);
    distinct_colors(&frames, limit).is_some() || local_plan(&frames, w, h, colors).is_some()
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
    if exact.is_none() && n > 1 {
        let delays: Vec<u32> = anim.frames.iter().map(|f| f.duration_ms).collect();
        if let Some(bytes) = encode_local_palettes(&frames, &delays, w, h, opts.colors, anim.loop_forever, opts.lossy) {
            return Ok(bytes);
        }
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
    let mut idx: Vec<Vec<u8>> = frames
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
    let pc = pal.colors();
    let (t2, near) = near_colors(pc, opts.lossy);
    let lossy = opts.lossy > 0;
    for i in 0..n {
        // The frame is taken out of `idx` so substitutions can be written back; the previous frame in `idx` already
        // holds what a viewer sees after its own substitutions, which is what this frame is compared against.
        let mut cur = std::mem::take(&mut idx[i]);
        // The frame as the palette quantized it: every substitution below stays within the lossiness budget of this,
        // so the error against the source never compounds.
        let src: Vec<u8> = if lossy { cur.clone() } else { Vec::new() };
        let base: Option<&[u8]> = if blank_before[i] { None } else { Some(&idx[i - 1]) };
        let dispose = if i + 1 < n && blank_before[i + 1] { 2u8 } else { 1u8 };
        if lossy && let Some(b) = base {
            // A pixel close enough to what is already on screen is simply left unchanged.
            for p in 0..w * h {
                let (c, o) = (cur[p], b[p]);
                if c != o && c != transparent && o != transparent && dist2(pc[c as usize], pc[o as usize]) <= t2 {
                    cur[p] = o;
                }
            }
        }
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
        let data = if lossy {
            let orig = pixels.clone();
            let pos = |k: usize| (y0 + k / rw) * w + x0 + k % rw;
            let mut fill = |k: usize, out: &mut Vec<u8>| {
                let v = orig[k];
                let p = pos(k);
                if v == transparent && has_t {
                    // Unchanged pixel: a colour near the one on screen would do as well.
                    if let Some(b) = base
                        && b[p] != transparent
                    {
                        let want = pc[src[p] as usize];
                        out.extend(near[b[p] as usize].iter().copied().filter(|&c| dist2(pc[c as usize], want) <= t2));
                    }
                } else {
                    if has_t
                        && let Some(b) = base
                        && b[p] != transparent
                        && dist2(pc[v as usize], pc[b[p] as usize]) <= t2
                    {
                        out.push(transparent);
                    }
                    let want = pc[src[p] as usize];
                    out.extend(near[v as usize].iter().copied().filter(|&c| dist2(pc[c as usize], want) <= t2));
                }
            };
            let data = lzw_core(&mut pixels, min_code, Some(&mut fill));
            for (k, (&new, &old)) in pixels.iter().zip(&orig).enumerate() {
                if new != old {
                    let p = pos(k);
                    cur[p] = if new == transparent { base.map_or(cur[p], |b| b[p]) } else { new };
                }
            }
            data
        } else {
            lzw_encode(&pixels, min_code)
        };
        for block in data.chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0);
        idx[i] = cur;
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

/// Source of replacement candidates for the lossy LZW coder: `f(k, out)` appends values for pixel `k`, best first.
pub(crate) type Subst<'a> = &'a mut dyn FnMut(usize, &mut Vec<u8>);

/// GIF's variable-width LZW. `indices` must not be empty and every index must be below `1 << min_code`.
pub(crate) fn lzw_encode(indices: &[u8], min_code: u8) -> Vec<u8> {
    lzw_core(&mut indices.to_vec(), min_code, None)
}

/// LZW coder with an optional lossy hook. When the dictionary has no entry for the current string plus the next pixel,
/// `subst(k, out)` lists replacement values for pixel `k` (best first); the first that does extend the match replaces
/// the pixel in `indices`, so on return `indices` holds what the decoder will reproduce.
pub(crate) fn lzw_core(indices: &mut [u8], min_code: u8, mut subst: Option<Subst>) -> Vec<u8> {
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
    let mut cand: Vec<u8> = Vec::new();
    sink.put(clear, size);
    let mut prefix = indices[0] as u32;
    for k in 1..indices.len() {
        let mut b = indices[k];
        let key = prefix << 8 | b as u32;
        let mut slot = (key.wrapping_mul(2_654_435_761) >> 19) as usize;
        let mut found = loop {
            if keys[slot] == key {
                break Some(vals[slot]);
            }
            if keys[slot] == EMPTY {
                break None;
            }
            slot = (slot + 1) & (SLOTS - 1);
        };
        if found.is_none()
            && let Some(f) = subst.as_mut()
        {
            cand.clear();
            f(k, &mut cand);
            for &c in &cand {
                let ck = prefix << 8 | c as u32;
                let mut s = (ck.wrapping_mul(2_654_435_761) >> 19) as usize;
                let hit = loop {
                    if keys[s] == ck {
                        break Some(vals[s]);
                    }
                    if keys[s] == EMPTY {
                        break None;
                    }
                    s = (s + 1) & (SLOTS - 1);
                };
                if hit.is_some() {
                    found = hit;
                    b = c;
                    indices[k] = c;
                    break;
                }
            }
        }
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
