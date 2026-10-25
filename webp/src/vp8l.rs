//! VP8L (WebP lossless) bitstream decoder.
//!
//! An image is a sequence of optional transforms (predictor, cross-colour, subtract-green, colour
//! indexing) followed by an LZ77 + colour-cache + prefix-coded ARGB stream. Everything here follows
//! the "WebP Lossless Bitstream Specification"; pixels are packed `0xAARRGGBB`.

use rc_core::error::{Error, Result};
use rc_core::util::bitio::LsbReader;
use rc_core::util::huffman::Decoder;

pub const SIGNATURE: u8 = 0x2F;
const NUM_LITERAL: usize = 256;
const NUM_LENGTH: usize = 24;
const NUM_DIST: usize = 40;
const MAX_CACHE_BITS: u32 = 11;
const CODE_LENGTH_ORDER: [usize; 19] = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// (dx, dy) pairs that short backward-reference distance codes 1..=120 stand for;
/// the distance is `dx + dy * width`.
pub const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

#[inline(always)]
pub fn div_round_up(a: usize, b_bits: u32) -> usize {
    (a + (1 << b_bits) - 1) >> b_bits
}

/// A decoded prefix code: a table, or a single symbol that costs no bits.
enum Code {
    Single(u16),
    Table(Decoder),
}

impl Code {
    #[inline(always)]
    fn read(&self, r: &mut LsbReader) -> Result<u16> {
        match self {
            Code::Single(s) => Ok(*s),
            Code::Table(d) => d.decode(r),
        }
    }

    fn from_lengths(lengths: &[u8]) -> Result<Code> {
        let used: Vec<usize> = lengths.iter().enumerate().filter(|(_, l)| **l > 0).map(|(s, _)| s).collect();
        match used.len() {
            0 => Err(Error::invalid("empty WebP prefix code")),
            1 => Ok(Code::Single(used[0] as u16)),
            _ => {
                // The code must be complete: a hole would make some bit patterns undecodable.
                let kraft: u64 = lengths.iter().filter(|&&l| l > 0).map(|&l| 1u64 << (15 - l)).sum();
                if kraft != 1 << 15 {
                    return Err(Error::invalid("incomplete WebP prefix code"));
                }
                Ok(Code::Table(Decoder::new(lengths)?))
            }
        }
    }
}

/// Five codes used together: green/length/cache, red, blue, alpha, distance.
struct Group {
    codes: [Code; 5],
}

fn check(r: &LsbReader) -> Result<()> {
    if r.overran() { Err(Error::invalid("truncated WebP lossless data")) } else { Ok(()) }
}

fn read_code(r: &mut LsbReader, alphabet: usize) -> Result<Code> {
    r.refill();
    let mut lengths = vec![0u8; alphabet];
    if r.take(1) == 1 {
        // Simple code: one or two symbols given directly.
        let two = r.take(1) == 1;
        let first_8 = r.take(1) == 1;
        let s0 = r.take(if first_8 { 8 } else { 1 }) as usize;
        if s0 >= alphabet {
            return Err(Error::invalid("WebP symbol outside the alphabet"));
        }
        lengths[s0] = 1;
        if two {
            let s1 = r.take(8) as usize;
            if s1 >= alphabet {
                return Err(Error::invalid("WebP symbol outside the alphabet"));
            }
            lengths[s1] = 1;
        }
    } else {
        let n = 4 + r.take(4) as usize;
        let mut cl = [0u8; 19];
        for &o in CODE_LENGTH_ORDER.iter().take(n) {
            r.refill();
            cl[o] = r.take(3) as u8;
        }
        let cl_code = Code::from_lengths(&cl)?;
        r.refill();
        let mut max_symbol = if r.take(1) == 1 {
            let nbits = 2 + 2 * r.take(3);
            r.refill();
            2 + r.take(nbits) as usize
        } else {
            alphabet
        };
        let (mut i, mut prev) = (0usize, 8u8);
        while i < alphabet && max_symbol > 0 {
            max_symbol -= 1;
            r.refill();
            let sym = cl_code.read(r)? as u8;
            if sym < 16 {
                lengths[i] = sym;
                i += 1;
                if sym != 0 {
                    prev = sym;
                }
            } else {
                let (n, v) = match sym {
                    16 => (3 + r.take(2) as usize, prev),
                    17 => (3 + r.take(3) as usize, 0),
                    _ => (11 + r.take(7) as usize, 0),
                };
                if i + n > alphabet {
                    return Err(Error::invalid("WebP code-length run overflows the alphabet"));
                }
                lengths[i..i + n].fill(v);
                i += n;
            }
            check(r)?;
        }
    }
    check(r)?;
    Code::from_lengths(&lengths)
}

/// Value of an LZ77 length/distance prefix symbol, reading its extra bits.
#[inline(always)]
fn prefix_value(r: &mut LsbReader, prefix: u32) -> u32 {
    if prefix < 4 {
        return prefix + 1;
    }
    let extra = (prefix - 2) >> 1;
    let offset = (2 + (prefix & 1)) << extra;
    offset + r.take(extra) + 1
}

/// Colour cache hash shared with the encoder.
#[inline(always)]
pub fn cache_index(argb: u32, bits: u32) -> usize {
    (0x1e35_a7bdu32.wrapping_mul(argb) >> (32 - bits)) as usize
}

/// Decode an entropy-coded ARGB image of `w` x `h`. `main` images may carry a meta prefix image.
fn read_image(r: &mut LsbReader, w: usize, h: usize, main: bool) -> Result<Vec<u32>> {
    r.refill();
    let cache_bits = if r.take(1) == 1 {
        let b = r.take(4);
        if !(1..=MAX_CACHE_BITS).contains(&b) {
            return Err(Error::invalid("bad WebP colour cache size"));
        }
        b
    } else {
        0
    };
    let (mut meta_bits, mut meta_w, mut meta): (u32, usize, Vec<u32>) = (0, 0, Vec::new());
    let mut num_groups = 1usize;
    if main && r.take(1) == 1 {
        meta_bits = r.take(3) + 2;
        meta_w = div_round_up(w, meta_bits);
        let meta_h = div_round_up(h, meta_bits);
        meta = read_image(r, meta_w, meta_h, false)?;
        for p in &mut meta {
            *p = (*p >> 8) & 0xFFFF;
            num_groups = num_groups.max(*p as usize + 1);
        }
        r.refill();
    }
    if num_groups > 1 << 12 {
        return Err(Error::invalid("too many WebP prefix groups"));
    }
    let cache_size = if cache_bits > 0 { 1usize << cache_bits } else { 0 };
    let sizes = [NUM_LITERAL + NUM_LENGTH + cache_size, NUM_LITERAL, NUM_LITERAL, NUM_LITERAL, NUM_DIST];
    let mut groups = Vec::with_capacity(num_groups);
    for _ in 0..num_groups {
        let mk = |i: usize, r: &mut LsbReader| read_code(r, sizes[i]);
        groups.push(Group { codes: [mk(0, r)?, mk(1, r)?, mk(2, r)?, mk(3, r)?, mk(4, r)?] });
    }

    let total = w.checked_mul(h).ok_or_else(|| Error::invalid("WebP dimensions overflow"))?;
    let mut out = vec![0u32; total];
    let mut cache = vec![0u32; cache_size];
    let (mut pos, mut cached) = (0usize, 0usize);
    let mut group = &groups[0];
    while pos < total {
        if !meta.is_empty() {
            let (x, y) = (pos % w, pos / w);
            group = &groups[meta[(y >> meta_bits) * meta_w + (x >> meta_bits)] as usize];
        }
        r.refill();
        let s = group.codes[0].read(r)? as usize;
        if s < NUM_LITERAL {
            let red = group.codes[1].read(r)? as u32;
            r.refill();
            let blue = group.codes[2].read(r)? as u32;
            let alpha = group.codes[3].read(r)? as u32;
            out[pos] = alpha << 24 | red << 16 | (s as u32) << 8 | blue;
            pos += 1;
        } else if s < NUM_LITERAL + NUM_LENGTH {
            let len = prefix_value(r, (s - NUM_LITERAL) as u32) as usize;
            r.refill();
            let dsym = group.codes[4].read(r)? as u32;
            r.refill();
            let code = prefix_value(r, dsym) as usize;
            let dist = if code > 120 {
                code - 120
            } else {
                let (dx, dy) = DISTANCE_MAP[code - 1];
                (dx as isize + dy as isize * w as isize).max(1) as usize
            };
            if dist > pos || len > total - pos {
                return Err(Error::invalid("WebP backward reference out of range"));
            }
            for i in 0..len {
                out[pos + i] = out[pos + i - dist];
            }
            pos += len;
        } else {
            let idx = s - NUM_LITERAL - NUM_LENGTH;
            if idx >= cache_size {
                return Err(Error::invalid("WebP colour cache index out of range"));
            }
            out[pos] = cache[idx];
            pos += 1;
        }
        if cache_bits > 0 {
            while cached < pos {
                cache[cache_index(out[cached], cache_bits)] = out[cached];
                cached += 1;
            }
        }
        check(r)?;
    }
    Ok(out)
}

// ------------------------------------------------------------------ inverse transforms

enum Transform {
    Predictor { bits: u32, data: Vec<u32> },
    CrossColor { bits: u32, data: Vec<u32> },
    SubtractGreen,
    ColorIndexing { palette: Vec<u32>, width_bits: u32 },
}

#[inline(always)]
fn add_pixels(a: u32, b: u32) -> u32 {
    let ag = (a & 0xFF00_FF00).wrapping_add(b & 0xFF00_FF00) & 0xFF00_FF00;
    let rb = (a & 0x00FF_00FF).wrapping_add(b & 0x00FF_00FF) & 0x00FF_00FF;
    ag | rb
}

#[inline(always)]
fn channels(p: u32) -> [i32; 4] {
    [(p >> 24) as i32, ((p >> 16) & 255) as i32, ((p >> 8) & 255) as i32, (p & 255) as i32]
}

#[inline(always)]
fn pack(c: [i32; 4]) -> u32 {
    (c[0] as u32) << 24 | (c[1] as u32) << 16 | (c[2] as u32) << 8 | c[3] as u32
}

#[inline(always)]
pub fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xFEFE_FEFE) >> 1).wrapping_add(a & b)
}

#[inline(always)]
fn clamp255(v: i32) -> i32 {
    v.clamp(0, 255)
}

#[inline(always)]
fn select(l: u32, t: u32, tl: u32) -> u32 {
    let (cl, ct, ctl) = (channels(l), channels(t), channels(tl));
    // |p - L| and |p - T| summed over channels, where p = L + T - TL.
    let mut pl = 0;
    let mut pt = 0;
    for i in 0..4 {
        pl += (ct[i] - ctl[i]).abs();
        pt += (cl[i] - ctl[i]).abs();
    }
    if pl < pt { l } else { t }
}

#[inline(always)]
fn clamp_add_sub_full(a: u32, b: u32, c: u32) -> u32 {
    let (ca, cb, cc) = (channels(a), channels(b), channels(c));
    pack([0, 1, 2, 3].map(|i| clamp255(ca[i] + cb[i] - cc[i])))
}

#[inline(always)]
fn clamp_add_sub_half(a: u32, b: u32) -> u32 {
    let (ca, cb) = (channels(a), channels(b));
    pack([0, 1, 2, 3].map(|i| clamp255(ca[i] + (ca[i] - cb[i]) / 2)))
}

/// Predicted value of mode `mode` (0..=13) from the left, top, top-left and top-right neighbours.
#[inline(always)]
pub fn predict(mode: u32, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        0 => 0xFF00_0000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(l, tr), t),
        6 => average2(l, tl),
        7 => average2(l, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(l, tl), average2(t, tr)),
        11 => select(l, t, tl),
        12 => clamp_add_sub_full(l, t, tl),
        13 => clamp_add_sub_half(average2(l, t), tl),
        _ => 0xFF00_0000, // modes 14 and 15 behave like 0
    }
}

fn inverse_predictor(px: &mut [u32], w: usize, h: usize, bits: u32, modes: &[u32]) {
    let tiles_w = div_round_up(w, bits);
    // First row: the first pixel predicts from opaque black, the rest from the left.
    px[0] = add_pixels(px[0], 0xFF00_0000);
    for x in 1..w {
        px[x] = add_pixels(px[x], px[x - 1]);
    }
    for y in 1..h {
        // First column predicts from the pixel above.
        let row = y * w;
        px[row] = add_pixels(px[row], px[row - w]);
        for x in 1..w {
            let i = row + x;
            let mode = (modes[(y >> bits) * tiles_w + (x >> bits)] >> 8) & 15;
            let (l, t, tl) = (px[i - 1], px[i - w], px[i - w - 1]);
            // The top-right of the last column is the first pixel of the current row.
            let tr = px[i - w + 1];
            px[i] = add_pixels(px[i], predict(mode, l, t, tr, tl));
        }
    }
}

#[inline(always)]
fn color_delta(t: i8, c: i8) -> i32 {
    (t as i32 * c as i32) >> 5
}

fn inverse_cross_color(px: &mut [u32], w: usize, h: usize, bits: u32, data: &[u32]) {
    let tiles_w = div_round_up(w, bits);
    for y in 0..h {
        for x in 0..w {
            let t = data[(y >> bits) * tiles_w + (x >> bits)];
            let (r2b, g2b, g2r) = ((t >> 16) as u8 as i8, (t >> 8) as u8 as i8, t as u8 as i8);
            let p = px[y * w + x];
            let (a, mut red, green, mut blue) =
                ((p >> 24) & 255, ((p >> 16) & 255) as i32, ((p >> 8) & 255) as i32, (p & 255) as i32);
            red = (red + color_delta(g2r, green as u8 as i8)) & 255;
            blue = (blue + color_delta(g2b, green as u8 as i8)) & 255;
            blue = (blue + color_delta(r2b, red as u8 as i8)) & 255;
            px[y * w + x] = a << 24 | (red as u32) << 16 | (green as u32) << 8 | blue as u32;
        }
    }
}

fn inverse_subtract_green(px: &mut [u32]) {
    for p in px {
        let green = (*p >> 8) & 255;
        let red = (((*p >> 16) & 255) + green) & 255;
        let blue = ((*p & 255) + green) & 255;
        *p = (*p & 0xFF00_FF00) | red << 16 | blue;
    }
}

/// Expand bundled palette indices (several pixels per stored pixel) to colours. `w` is the real width.
fn inverse_color_indexing(px: &[u32], w: usize, h: usize, palette: &[u32], width_bits: u32) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    let packed_w = div_round_up(w, width_bits);
    let bits_per = 8 >> width_bits; // bits per index: 8, 4, 2 or 1
    let per_pixel = 1usize << width_bits;
    let mask = (1u32 << bits_per) - 1;
    for y in 0..h {
        for x in 0..w {
            let src = px[y * packed_w + (x >> width_bits)];
            let green = (src >> 8) & 255;
            let idx = if width_bits == 0 { green } else { (green >> (bits_per * (x & (per_pixel - 1)) as u32)) & mask };
            out[y * w + x] = palette.get(idx as usize).copied().unwrap_or(0);
        }
    }
    out
}

/// Header facts plus pixels.
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
    pub argb: Vec<u32>,
}

pub fn decode(data: &[u8]) -> Result<Decoded> {
    if data.first() != Some(&SIGNATURE) {
        return Err(Error::invalid("bad VP8L signature"));
    }
    let mut r = LsbReader::new(&data[1..]);
    r.refill();
    let width = r.take(14) as usize + 1;
    let height = r.take(14) as usize + 1;
    let has_alpha = r.take(1) == 1;
    if r.take(3) != 0 {
        return Err(Error::unsupported("VP8L version"));
    }
    if (width as u64) * (height as u64) > rc_core::image::MAX_PIXELS {
        return Err(Error::invalid("unreasonable image dimensions"));
    }
    let argb = decode_body(&mut r, width, height)?;
    Ok(Decoded { width, height, has_alpha, argb })
}

/// Decode a bare VP8L image stream (transforms plus pixels, no signature or size header) of a
/// known size, as used for the alpha plane of lossy WebP files.
pub fn decode_headerless(data: &[u8], width: usize, height: usize) -> Result<Vec<u32>> {
    if width == 0 || height == 0 || (width as u64) * (height as u64) > rc_core::image::MAX_PIXELS {
        return Err(Error::invalid("unreasonable image dimensions"));
    }
    let mut r = LsbReader::new(data);
    r.refill();
    decode_body(&mut r, width, height)
}

fn decode_body(r: &mut LsbReader, width: usize, height: usize) -> Result<Vec<u32>> {
    let mut transforms: Vec<(Transform, usize)> = Vec::new();
    let mut xsize = width;
    let mut seen = [false; 4];
    loop {
        r.refill();
        if r.take(1) == 0 {
            break;
        }
        let ty = r.take(2) as usize;
        if std::mem::replace(&mut seen[ty], true) {
            return Err(Error::invalid("repeated WebP transform"));
        }
        match ty {
            0 | 1 => {
                let bits = r.take(3) + 2;
                let data = read_image(r, div_round_up(xsize, bits), div_round_up(height, bits), false)?;
                transforms.push((
                    if ty == 0 { Transform::Predictor { bits, data } } else { Transform::CrossColor { bits, data } },
                    xsize,
                ));
            }
            2 => transforms.push((Transform::SubtractGreen, xsize)),
            _ => {
                let n = r.take(8) as usize + 1;
                let mut palette = read_image(r, n, 1, false)?;
                for i in 1..n {
                    palette[i] = add_pixels(palette[i], palette[i - 1]);
                }
                let width_bits = match n {
                    0..=2 => 3,
                    3..=4 => 2,
                    5..=16 => 1,
                    _ => 0,
                };
                transforms.push((Transform::ColorIndexing { palette, width_bits }, xsize));
                xsize = div_round_up(xsize, width_bits);
            }
        }
    }
    let mut px = read_image(r, xsize, height, true)?;
    for (t, w) in transforms.into_iter().rev() {
        match t {
            Transform::Predictor { bits, data } => inverse_predictor(&mut px, w, height, bits, &data),
            Transform::CrossColor { bits, data } => inverse_cross_color(&mut px, w, height, bits, &data),
            Transform::SubtractGreen => inverse_subtract_green(&mut px),
            Transform::ColorIndexing { palette, width_bits } => {
                px = inverse_color_indexing(&px, w, height, &palette, width_bits);
            }
        }
    }
    Ok(px)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_map_is_consistent() {
        assert_eq!(DISTANCE_MAP.len(), 120);
        // Every entry is distinct, within the documented range, and never behind the current row.
        let mut seen = std::collections::HashSet::new();
        for &(dx, dy) in &DISTANCE_MAP {
            assert!(seen.insert((dx, dy)));
            assert!((-7..=8).contains(&dx) && (0..=7).contains(&dy));
            assert!(dy > 0 || dx > 0, "({dx},{dy}) points forward");
        }
    }

    #[test]
    fn pixel_helpers() {
        assert_eq!(add_pixels(0x01FF_80FE, 0x0201_8003), 0x0300_0001);
        assert_eq!(average2(0x0000_00FF, 0x0000_0001), 0x0000_0080);
        assert_eq!(predict(0, 1, 2, 3, 4), 0xFF00_0000);
        assert_eq!(predict(1, 7, 8, 9, 10), 7);
        assert_eq!(predict(12, 0x0000_0010, 0x0000_0020, 0, 0x0000_0008), 0x0000_0028);
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        assert!(decode(b"").is_err());
        assert!(decode(&[SIGNATURE]).is_err());
        for len in 0..64 {
            let junk: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let mut d = vec![SIGNATURE];
            d.extend(junk);
            let _ = decode(&d);
        }
    }
}
