use super::SIGNATURE;
use super::filter::{Strategy, filter_image};
use crate::deflate::zlib_compress;
use crate::image::{ColorType, Image};
use crate::util::checksum::crc32;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// DEFLATE effort, 0..=9.
    pub level: u32,
    /// Rewrite the (invisible) color of fully transparent pixels to whatever compresses best.
    pub clean_alpha: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self { level: 7, clean_alpha: true }
    }
}

/// Image after lossless representation reduction, ready for filtering.
struct Reduced {
    width: u32,
    height: u32,
    /// PNG color type code (0, 2, 3, 4, 6).
    color_type: u8,
    depth: u8,
    /// Packed scanlines, `row_bytes` each.
    rows: Vec<u8>,
    row_bytes: usize,
    /// Filter byte distance.
    bpp: usize,
    palette: Vec<[u8; 4]>,
}

struct Analysis {
    opaque: bool,
    gray: bool,
    /// Distinct RGBA colors (None if more than 256).
    colors: Option<Vec<[u8; 4]>>,
}

fn analyze(img: &Image) -> Analysis {
    let ch = img.color.channels();
    let mut opaque = true;
    let mut gray = img.color.is_gray();
    // Open-addressing set of packed RGBA colors; abort when it exceeds 256 entries.
    const SLOTS: usize = 1024;
    let mut table = [u64::MAX; SLOTS];
    let mut colors: Vec<[u8; 4]> = Vec::new();
    let mut overflow = false;
    for px in img.data.chunks_exact(ch) {
        let rgba = match img.color {
            ColorType::Gray => [px[0], px[0], px[0], 255],
            ColorType::GrayAlpha => [px[0], px[0], px[0], px[1]],
            ColorType::Rgb => [px[0], px[1], px[2], 255],
            ColorType::Rgba => [px[0], px[1], px[2], px[3]],
        };
        opaque &= rgba[3] == 255;
        gray &= rgba[0] == rgba[1] && rgba[1] == rgba[2];
        if !overflow {
            let key = u32::from_le_bytes(rgba) as u64;
            let mut slot = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 54) as usize & (SLOTS - 1);
            loop {
                if table[slot] == key {
                    break;
                }
                if table[slot] == u64::MAX {
                    table[slot] = key;
                    colors.push(rgba);
                    if colors.len() > 256 {
                        overflow = true;
                    }
                    break;
                }
                slot = (slot + 1) & (SLOTS - 1);
            }
        }
        if overflow && !opaque && !gray {
            break;
        }
    }
    Analysis { opaque, gray, colors: if overflow { None } else { Some(colors) } }
}

/// `perm[appearance index] = palette index`. Several orderings are tried (by luma, first appearance,
/// frequency, and a nearest-neighbor chain through color space) and the one whose index stream
/// compresses best is kept; neighboring pixels with similar colors then get similar indices.
fn best_palette_order(colors: &[[u8; 4]], app: &[u8], width: usize, depth: u8, level: u32) -> Vec<u8> {
    let n = colors.len();
    let invert = |order: &[usize]| -> Vec<u8> {
        let mut perm = vec![0u8; n];
        for (new, &old) in order.iter().enumerate() {
            perm[old] = new as u8;
        }
        perm
    };
    let luma = |c: &[u8; 4]| 299 * c[0] as u32 + 587 * c[1] as u32 + 114 * c[2] as u32;
    let mut by_luma: Vec<usize> = (0..n).collect();
    by_luma.sort_by_key(|&i| (colors[i][3] == 255, colors[i][3], luma(&colors[i])));
    let luma_perm = invert(&by_luma);
    if n < 3 || level < 3 {
        return luma_perm;
    }
    let mut freq = vec![0u64; n];
    for &a in app {
        freq[a as usize] += 1;
    }
    let mut by_freq: Vec<usize> = (0..n).collect();
    by_freq.sort_by_key(|&i| std::cmp::Reverse(freq[i]));
    // Greedy nearest-neighbor chain starting from the darkest, most transparent color.
    let dist =
        |a: &[u8; 4], b: &[u8; 4]| -> u32 { (0..4).map(|i| (a[i] as i32 - b[i] as i32).pow(2) as u32).sum::<u32>() };
    let mut chain = Vec::with_capacity(n);
    let mut used = vec![false; n];
    let mut cur = by_luma[0];
    used[cur] = true;
    chain.push(cur);
    for _ in 1..n {
        let next = (0..n).filter(|&i| !used[i]).min_by_key(|&i| dist(&colors[cur], &colors[i])).unwrap();
        used[next] = true;
        chain.push(next);
        cur = next;
    }
    let candidates = vec![luma_perm, invert(&(0..n).collect::<Vec<_>>()), invert(&by_freq), invert(&chain)];
    // Score on at most ~2M pixels (a band from the middle for big images).
    let total_rows = app.len() / width;
    let rows = (2_000_000 / width).clamp(1, total_rows);
    let first = (total_rows - rows) / 2;
    let sample = &app[first * width..(first + rows) * width];
    candidates
        .into_par_iter()
        .map(|perm| {
            let idx: Vec<u8> = sample.iter().map(|&a| perm[a as usize]).collect();
            let (packed, row_bytes) = if depth == 8 { (idx, width) } else { pack_bits(&idx, width, depth, |v| v) };
            let f = filter_image(&packed, row_bytes, 1, Strategy::Fixed(0));
            (zlib_compress(&f, 2).len(), perm)
        })
        .min_by_key(|(len, _)| *len)
        .map(|(_, perm)| perm)
        .unwrap()
}

/// How the hidden color of fully transparent pixels is rewritten.
#[derive(Clone, Copy)]
enum Clean {
    /// Black: collapses all transparent pixels to one palette entry.
    Zero,
    /// Copy the pixel to the left (or above at the row start): zero deltas for Sub filtering.
    Left,
    /// Copy the pixel above (or to the left in the first row): zero deltas for Up filtering.
    Up,
}

/// Rewrite the color channels of pixels whose alpha is 0 (8-bit images with alpha only).
fn clean_transparent(img: &Image, mode: Clean) -> Image {
    let ch = img.color.channels();
    let (w, h) = (img.width as usize, img.height as usize);
    let mut out = img.clone();
    let stride = w * ch;
    for y in 0..h {
        for x in 0..w {
            let i = y * stride + x * ch;
            if out.data[i + ch - 1] != 0 {
                continue;
            }
            let from_left = (x > 0).then(|| i - ch);
            let from_up = (y > 0).then(|| i - stride);
            let src = match mode {
                Clean::Zero => None,
                Clean::Left => from_left.or(from_up),
                Clean::Up => from_up.or(from_left),
            };
            for c in 0..ch - 1 {
                out.data[i + c] = src.map_or(0, |s| out.data[s + c]);
            }
        }
    }
    out
}

/// Smallest gray bit depth (1, 2, 4, 8) that represents all samples exactly.
fn gray_depth(samples: impl Iterator<Item = u8> + Clone) -> u8 {
    for (depth, scale) in [(1u8, 255u8), (2, 85), (4, 17)] {
        if samples.clone().all(|v| v % scale == 0) {
            return depth;
        }
    }
    8
}

fn pack_bits(indices: &[u8], width: usize, depth: u8, scale: impl Fn(u8) -> u8) -> (Vec<u8>, usize) {
    let height = indices.len() / width;
    let row_bytes = (width * depth as usize).div_ceil(8);
    let mut out = vec![0u8; row_bytes * height];
    let per_byte = 8 / depth as usize;
    for y in 0..height {
        for x in 0..width {
            let v = scale(indices[y * width + x]);
            let shift = 8 - depth as usize * (x % per_byte + 1);
            out[y * row_bytes + x / per_byte] |= v << shift;
        }
    }
    (out, row_bytes)
}

fn reduce(img: &Image, level: u32) -> Reduced {
    let (w, h) = (img.width, img.height);
    // 16-bit images: only reducible if every sample's two bytes are equal.
    let img8;
    let img = if img.depth == 16 {
        if img.data.chunks_exact(2).all(|s| s[0] == s[1]) {
            img8 = Image { depth: 8, data: img.data.iter().step_by(2).copied().collect(), ..img.clone() };
            &img8
        } else {
            let (ct, bpp) = match img.color {
                ColorType::Gray => (0, 2),
                ColorType::GrayAlpha => (4, 4),
                ColorType::Rgb => (2, 6),
                ColorType::Rgba => (6, 8),
            };
            return Reduced {
                width: w,
                height: h,
                color_type: ct,
                depth: 16,
                rows: img.data.clone(),
                row_bytes: img.stride(),
                bpp,
                palette: Vec::new(),
            };
        }
    } else {
        img
    };

    let an = analyze(img);
    let ch = img.color.channels();

    // Palette candidate.
    if let Some(colors) = &an.colors {
        let n = colors.len();
        let pal_depth = match n {
            0..=2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        let direct_bits = match (an.gray, an.opaque) {
            (true, true) => {
                let d = gray_depth(img.data.chunks_exact(ch).map(|p| p[0]));
                d as usize
            }
            (true, false) => 16,
            (false, true) => 24,
            (false, false) => 32,
        };
        if (pal_depth as usize) < direct_bits {
            let key = |px: &[u8]| -> u32 {
                u32::from_le_bytes(match img.color {
                    ColorType::Gray => [px[0], px[0], px[0], 255],
                    ColorType::GrayAlpha => [px[0], px[0], px[0], px[1]],
                    ColorType::Rgb => [px[0], px[1], px[2], 255],
                    ColorType::Rgba => [px[0], px[1], px[2], px[3]],
                })
            };
            // Index every pixel by first-appearance order once; any palette order is then a
            // 256-entry permutation of these indices.
            let lookup: std::collections::HashMap<u32, u8> =
                colors.iter().enumerate().map(|(i, c)| (u32::from_le_bytes(*c), i as u8)).collect();
            let app: Vec<u8> = img.data.chunks_exact(ch).map(|px| lookup[&key(px)]).collect();
            let perm = best_palette_order(colors, &app, w as usize, pal_depth, level);
            let mut palette = vec![[0u8; 4]; n];
            for (old, &new) in perm.iter().enumerate() {
                palette[new as usize] = colors[old];
            }
            let indices: Vec<u8> = app.iter().map(|&a| perm[a as usize]).collect();
            let (rows, row_bytes) =
                if pal_depth == 8 { (indices, w as usize) } else { pack_bits(&indices, w as usize, pal_depth, |v| v) };
            return Reduced { width: w, height: h, color_type: 3, depth: pal_depth, rows, row_bytes, bpp: 1, palette };
        }
    }

    match (an.gray, an.opaque) {
        (true, true) => {
            let samples = img.data.chunks_exact(ch).map(|p| p[0]);
            let depth = gray_depth(samples.clone());
            let gray: Vec<u8> = samples.collect();
            if depth == 8 {
                Reduced {
                    width: w,
                    height: h,
                    color_type: 0,
                    depth: 8,
                    rows: gray,
                    row_bytes: w as usize,
                    bpp: 1,
                    palette: vec![],
                }
            } else {
                let scale = 255 / ((1u16 << depth) - 1) as u8;
                let (rows, row_bytes) = pack_bits(&gray, w as usize, depth, |v| v / scale);
                Reduced { width: w, height: h, color_type: 0, depth, rows, row_bytes, bpp: 1, palette: vec![] }
            }
        }
        (true, false) => {
            let rows: Vec<u8> =
                img.data.chunks_exact(ch).flat_map(|p| [p[0], if ch == 2 { p[1] } else { p[3] }]).collect();
            Reduced {
                width: w,
                height: h,
                color_type: 4,
                depth: 8,
                rows,
                row_bytes: w as usize * 2,
                bpp: 2,
                palette: vec![],
            }
        }
        (false, true) => {
            let rows = if ch == 3 {
                img.data.clone()
            } else {
                img.data.chunks_exact(ch).flat_map(|p| [p[0], p[1], p[2]]).collect()
            };
            Reduced {
                width: w,
                height: h,
                color_type: 2,
                depth: 8,
                rows,
                row_bytes: w as usize * 3,
                bpp: 3,
                palette: vec![],
            }
        }
        (false, false) => Reduced {
            width: w,
            height: h,
            color_type: 6,
            depth: 8,
            rows: img.data.clone(),
            row_bytes: w as usize * 4,
            bpp: 4,
            palette: vec![],
        },
    }
}

/// Largest image (in bytes) for which the brute-force filter search is attempted.
const BRUTE_MAX_BYTES: usize = 48 << 20;

/// Pick the filter strategy by trial-compressing a sample of the image, then filter everything.
fn filter_with_best_strategy(r: &Reduced, level: u32) -> Vec<u8> {
    // Brute compresses every candidate row ~5 times; skip it for huge images.
    let brute = level >= 9 && r.rows.len() <= BRUTE_MAX_BYTES;
    let mut candidates: Vec<Strategy> = if level < 2 {
        vec![if r.depth < 8 || r.color_type == 3 { Strategy::Fixed(0) } else { Strategy::MinSum }]
    } else if r.depth < 8 || r.color_type == 3 {
        vec![Strategy::Fixed(0), Strategy::MinSum, Strategy::Entropy, Strategy::Bigrams]
    } else {
        vec![
            Strategy::MinSum,
            Strategy::Entropy,
            Strategy::Bigrams,
            Strategy::Fixed(0),
            Strategy::Fixed(1),
            Strategy::Fixed(2),
            Strategy::Fixed(3),
            Strategy::Fixed(4),
        ]
    };
    if brute {
        candidates.push(Strategy::Brute);
    }
    if candidates.len() == 1 {
        return filter_image(&r.rows, r.row_bytes, r.bpp, candidates[0]);
    }
    // Representative sample: whole image if small, otherwise a few evenly spaced bands of rows.
    let budget: usize = if level >= 9 { 8 << 20 } else { 1 << 20 };
    let trial_level = if level >= 7 { 6 } else { 2 };
    let sample: std::borrow::Cow<[u8]> = if r.rows.len() <= budget {
        std::borrow::Cow::Borrowed(&r.rows)
    } else {
        let bands = 4;
        let band_rows = (budget / bands / r.row_bytes).max(1);
        let total_rows = r.rows.len() / r.row_bytes;
        let mut v = Vec::with_capacity(budget + r.row_bytes * bands);
        for b in 0..bands {
            let start = (total_rows - band_rows.min(total_rows)) * b / (bands - 1);
            let rows = &r.rows[start * r.row_bytes..(start + band_rows.min(total_rows)) * r.row_bytes];
            v.extend_from_slice(rows);
        }
        std::borrow::Cow::Owned(v)
    };
    let best = candidates
        .par_iter()
        .map(|&s| {
            let f = filter_image(&sample, r.row_bytes, r.bpp, s);
            (zlib_compress(&f, trial_level).len(), s)
        })
        .min_by_key(|t| t.0)
        .map(|t| t.1)
        .unwrap();
    filter_image(&r.rows, r.row_bytes, r.bpp, best)
}

fn write_chunk(out: &mut Vec<u8>, ty: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(ty);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Losslessly encode `img` as a PNG with minimal color representation.
pub fn encode(img: &Image, opts: &EncodeOptions) -> Vec<u8> {
    let has_hidden = opts.clean_alpha
        && img.depth == 8
        && img.color.has_alpha()
        && img.data.chunks_exact(img.color.channels()).any(|p| p[img.color.channels() - 1] == 0);
    if !has_hidden {
        return encode_one(img, opts);
    }
    let modes: &[Clean] = match opts.level {
        0..=2 => &[Clean::Left],
        3..=6 => &[Clean::Zero, Clean::Left],
        _ => &[Clean::Zero, Clean::Left, Clean::Up],
    };
    modes
        .par_iter()
        .map(|&m| encode_one(&clean_transparent(img, m), opts))
        .min_by_key(|v| v.len())
        .expect("at least one cleaning mode")
}

fn encode_one(img: &Image, opts: &EncodeOptions) -> Vec<u8> {
    let level = opts.level.min(9);
    let r = reduce(img, level);

    let filtered = filter_with_best_strategy(&r, level);
    let idat = zlib_compress(&filtered, level);

    let mut out = Vec::with_capacity(idat.len() + 1024);
    out.extend_from_slice(&SIGNATURE);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&r.width.to_be_bytes());
    ihdr.extend_from_slice(&r.height.to_be_bytes());
    ihdr.extend_from_slice(&[r.depth, r.color_type, 0, 0, 0]);
    write_chunk(&mut out, b"IHDR", &ihdr);
    for (ty, body) in &img.meta.png_color {
        write_chunk(&mut out, ty, body);
    }
    if let Some(icc) = &img.meta.icc {
        let mut body = b"ICC Profile\0\0".to_vec();
        body.extend_from_slice(&zlib_compress(icc, 9));
        write_chunk(&mut out, b"iCCP", &body);
    }
    if r.color_type == 3 {
        let plte: Vec<u8> = r.palette.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
        write_chunk(&mut out, b"PLTE", &plte);
        if let Some(last) = r.palette.iter().rposition(|c| c[3] != 255) {
            let trns: Vec<u8> = r.palette[..=last].iter().map(|c| c[3]).collect();
            write_chunk(&mut out, b"tRNS", &trns);
        }
    }
    for chunk in idat.chunks(1 << 20) {
        write_chunk(&mut out, b"IDAT", chunk);
    }
    write_chunk(&mut out, b"IEND", &[]);
    out
}
