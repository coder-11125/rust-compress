//! VP8L (WebP lossless) encoder.
//!
//! Candidate pipelines are tried and the smallest kept: a colour-indexing transform for images with
//! at most 256 colours (with several pixels bundled per stored pixel), or subtract-green plus a
//! per-tile predictor (and optionally cross-colour) for everything else. The transformed pixels go
//! through LZ77 with plane-aware distance codes, an optional colour cache and canonical prefix codes.

use super::vp8l::{DISTANCE_MAP, SIGNATURE, cache_index, div_round_up, predict};
use crate::util::bitio::LsbWriter;
use crate::util::huffman::{build_lengths, lsb_codes};
use rayon::prelude::*;

const NUM_LITERAL: usize = 256;
const NUM_LENGTH: usize = 24;
const NUM_DIST: usize = 40;
const MAX_LEN: usize = 4096;
/// Largest distance code value the format can express.
const MAX_DIST: usize = (1 << 20) - 120;
const CODE_LENGTH_ORDER: [usize; 19] = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Effort 0-9.
    pub level: u32,
    /// Zero the colour of fully transparent pixels (they are invisible; this makes them cheap).
    pub clean_alpha: bool,
}

// ------------------------------------------------------------------ pixel helpers

#[inline(always)]
fn sub_pixels(a: u32, b: u32) -> u32 {
    let ag = (a | 0x00FF_00FF).wrapping_sub(b & 0xFF00_FF00) & 0xFF00_FF00;
    let rb = (a | 0xFF00_FF00).wrapping_sub(b & 0x00FF_00FF) & 0x00FF_00FF;
    ag | rb
}

#[inline(always)]
fn abs_residual(p: u32) -> u32 {
    (0..4).map(|i| ((p >> (8 * i)) as u8 as i8).unsigned_abs() as u32).sum()
}

// ------------------------------------------------------------------ transforms

/// Subtract the green channel from red and blue.
fn subtract_green(px: &mut [u32]) {
    for p in px {
        let g = (*p >> 8) & 255;
        let r = (((*p >> 16) & 255).wrapping_sub(g)) & 255;
        let b = ((*p & 255).wrapping_sub(g)) & 255;
        *p = (*p & 0xFF00_FF00) | r << 16 | b;
    }
}

/// Residuals of the best per-tile predictor, and the image of chosen modes (mode in green, opaque).
fn predictor_transform(src: &[u32], w: usize, h: usize, bits: u32) -> (Vec<u32>, Vec<u32>) {
    let (tw, th) = (div_round_up(w, bits), div_round_up(h, bits));
    let tile = 1usize << bits;
    let modes: Vec<u32> = (0..th)
        .into_par_iter()
        .flat_map_iter(|ty| {
            (0..tw).map(move |tx| {
                let (x0, y0) = (tx * tile, ty * tile);
                let (x1, y1) = ((x0 + tile).min(w), (y0 + tile).min(h));
                let mut best = (u64::MAX, 0u32);
                for mode in 0..14u32 {
                    let mut cost = 0u64;
                    for y in y0.max(1)..y1 {
                        for x in x0.max(1)..x1 {
                            let i = y * w + x;
                            let pred = predict(mode, src[i - 1], src[i - w], src[i - w + 1], src[i - w - 1]);
                            cost += abs_residual(sub_pixels(src[i], pred)) as u64;
                        }
                    }
                    if cost < best.0 {
                        best = (cost, mode);
                    }
                }
                best.1
            })
        })
        .collect();
    let mut res = vec![0u32; src.len()];
    res.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let pred = if y == 0 {
                if x == 0 { 0xFF00_0000 } else { src[i - 1] }
            } else if x == 0 {
                src[i - w]
            } else {
                predict(modes[(y >> bits) * tw + (x >> bits)], src[i - 1], src[i - w], src[i - w + 1], src[i - w - 1])
            };
            row[x] = sub_pixels(src[i], pred);
        }
    });
    (res, modes.into_iter().map(|m| 0xFF00_0000 | m << 8).collect())
}

/// The distinct colours in ascending order, or `None` when there are more than 256.
fn small_palette(px: &[u32]) -> Option<Vec<u32>> {
    let mut seen = std::collections::HashSet::new();
    for &p in px {
        if seen.insert(p) && seen.len() > 256 {
            return None;
        }
    }
    let mut v: Vec<u32> = seen.into_iter().collect();
    v.sort_unstable();
    Some(v)
}

fn bundle_bits(n: usize) -> u32 {
    match n {
        0..=2 => 3,
        3..=4 => 2,
        5..=16 => 1,
        _ => 0,
    }
}

/// Replace colours by palette indices, packing 2^bits indices into each stored pixel.
fn index_pixels(px: &[u32], w: usize, h: usize, palette: &[u32], bits: u32) -> Vec<u32> {
    let lookup: std::collections::HashMap<u32, u32> = palette.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
    let pw = div_round_up(w, bits);
    let per_bits = 8 >> bits;
    let mut out = vec![0xFF00_0000u32; pw * h];
    for y in 0..h {
        for x in 0..w {
            let idx = lookup[&px[y * w + x]];
            out[y * pw + (x >> bits)] |= idx << (8 + per_bits * (x & ((1 << bits) - 1)) as u32);
        }
    }
    out
}

// ------------------------------------------------------------------ LZ77

const HASH_BITS: u32 = 16;
const WINDOW: usize = 1 << 20;

/// One parse step: a literal (`len == 1`, `dist == 0`) or a copy.
#[derive(Clone, Copy)]
struct Step {
    len: u32,
    dist: u32,
}

#[inline(always)]
fn hash_pair(a: u32, b: u32) -> usize {
    ((a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA6B)) >> (32 - HASH_BITS)) as usize
}

fn common_len(px: &[u32], a: usize, b: usize, max: usize) -> usize {
    let mut l = 0;
    while l < max && px[a + l] == px[b + l] {
        l += 1;
    }
    l
}

/// Greedy-with-lazy parse into literals and copies.
fn parse(px: &[u32], xsize: usize, depth: usize, lazy: bool) -> Vec<Step> {
    let n = px.len();
    let mut head = vec![u32::MAX; 1 << HASH_BITS];
    let mut prev = vec![u32::MAX; WINDOW];
    let mut steps = Vec::with_capacity(n / 2 + 16);
    let mask = WINDOW - 1;
    let insert = |head: &mut [u32], prev: &mut [u32], i: usize| {
        if i + 1 < n {
            let h = hash_pair(px[i], px[i + 1]);
            prev[i & mask] = head[h];
            head[h] = i as u32;
        }
    };
    let find = |head: &[u32], prev: &[u32], i: usize| -> (usize, usize) {
        let max = (n - i).min(MAX_LEN);
        if max < 2 {
            return (0, 0);
        }
        let mut best = (0usize, 0usize);
        // Very cheap, very common candidates first: the pixel to the left, the one above and its neighbours.
        for d in [1, xsize, xsize + 1, xsize.saturating_sub(1)] {
            if d >= 1 && d <= i && d <= MAX_DIST {
                let l = common_len(px, i - d, i, max);
                if l > best.0 {
                    best = (l, d);
                }
            }
        }
        if i + 1 < n {
            let mut cand = head[hash_pair(px[i], px[i + 1])];
            let mut left = depth;
            while cand != u32::MAX && left > 0 {
                let c = cand as usize;
                if c >= i || i - c > MAX_DIST {
                    break;
                }
                let l = common_len(px, c, i, max);
                if l > best.0 || (l == best.0 && l > 0 && i - c < best.1) {
                    best = (l, i - c);
                    if l == max {
                        break;
                    }
                }
                let next = prev[c & mask];
                if next == u32::MAX || next as usize >= c {
                    break;
                }
                cand = next;
                left -= 1;
            }
        }
        best
    };
    let mut i = 0;
    while i < n {
        let (mut len, mut dist) = find(&head, &prev, i);
        if len < 3 {
            steps.push(Step { len: 1, dist: 0 });
            insert(&mut head, &mut prev, i);
            i += 1;
            continue;
        }
        let mut inserted_first = false;
        if lazy && len < 32 && i + 1 < n {
            // Would starting one pixel later give a clearly longer copy?
            insert(&mut head, &mut prev, i);
            inserted_first = true;
            let (l2, d2) = find(&head, &prev, i + 1);
            if l2 > len + 1 {
                steps.push(Step { len: 1, dist: 0 });
                i += 1;
                (len, dist) = (l2, d2);
                inserted_first = false;
            }
        }
        steps.push(Step { len: len as u32, dist: dist as u32 });
        for k in usize::from(inserted_first)..len {
            insert(&mut head, &mut prev, i + k);
        }
        i += len;
    }
    steps
}

// ------------------------------------------------------------------ tokens and entropy coding

#[derive(Clone, Copy)]
enum Tok {
    Literal(u32),
    Cache(u32),
    /// Copy of `len` pixels from the distance given by `code` (already in the format's code space).
    Copy {
        len: u32,
        code: u32,
    },
}

/// Distance in pixels -> distance code, using the short 2-D neighbourhood codes where possible.
fn distance_code(dist: usize, xsize: usize, plane: &[u8; 8 * 16]) -> u32 {
    let ys = dist / xsize;
    let xs = dist - ys * xsize;
    // (dx, dy) with dx in -7..=8: either the pixel is in this row band or just left of the next one.
    let try_code = |dx: isize, dy: usize| -> Option<u32> {
        if (-7..=8).contains(&dx) && dy < 8 {
            let c = plane[dy * 16 + (dx + 7) as usize];
            (c != 0).then_some(c as u32)
        } else {
            None
        }
    };
    if xs <= 8
        && let Some(c) = try_code(xs as isize, ys)
    {
        return c;
    }
    if xs + 8 > xsize
        && let Some(c) = try_code(xs as isize - xsize as isize, ys + 1)
    {
        return c;
    }
    dist as u32 + 120
}

fn plane_table() -> [u8; 8 * 16] {
    let mut t = [0u8; 8 * 16];
    for (i, &(dx, dy)) in DISTANCE_MAP.iter().enumerate() {
        t[dy as usize * 16 + (dx as isize + 7) as usize] = (i + 1) as u8;
    }
    t
}

/// Split a length/distance value into (prefix symbol, extra bit count, extra bit value).
fn prefix_encode(value: u32) -> (u32, u32, u32) {
    let v = value - 1;
    if v < 4 {
        return (v, 0, 0);
    }
    let hb = 31 - v.leading_zeros();
    let second = (v >> (hb - 1)) & 1;
    let extra = hb - 1;
    (2 * hb + second, extra, v & ((1 << extra) - 1))
}

fn to_tokens(px: &[u32], steps: &[Step], xsize: usize, cache_bits: u32, plane: &[u8; 8 * 16]) -> Vec<Tok> {
    let mut toks = Vec::with_capacity(steps.len());
    let mut cache = vec![0u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
    let mut i = 0;
    for s in steps {
        if s.len == 1 {
            let p = px[i];
            if cache_bits > 0 {
                let idx = cache_index(p, cache_bits);
                if cache[idx] == p {
                    toks.push(Tok::Cache(idx as u32));
                } else {
                    toks.push(Tok::Literal(p));
                }
                cache[idx] = p;
            } else {
                toks.push(Tok::Literal(p));
            }
            i += 1;
        } else {
            toks.push(Tok::Copy { len: s.len, code: distance_code(s.dist as usize, xsize, plane) });
            if cache_bits > 0 {
                for k in 0..s.len as usize {
                    let p = px[i + k];
                    cache[cache_index(p, cache_bits)] = p;
                }
            }
            i += s.len as usize;
        }
    }
    toks
}

/// Symbol histograms of one prefix-code group: green+length+cache, red, blue, alpha, distance.
struct Hist {
    h: [Vec<u32>; 5],
}

impl Hist {
    fn new(cache_size: usize) -> Self {
        Hist {
            h: [
                vec![0; NUM_LITERAL + NUM_LENGTH + cache_size],
                vec![0; NUM_LITERAL],
                vec![0; NUM_LITERAL],
                vec![0; NUM_LITERAL],
                vec![0; NUM_DIST],
            ],
        }
    }

    fn add(&mut self, t: Tok) {
        match t {
            Tok::Literal(p) => {
                self.h[0][((p >> 8) & 255) as usize] += 1;
                self.h[1][((p >> 16) & 255) as usize] += 1;
                self.h[2][(p & 255) as usize] += 1;
                self.h[3][(p >> 24) as usize] += 1;
            }
            Tok::Cache(i) => self.h[0][NUM_LITERAL + NUM_LENGTH + i as usize] += 1,
            Tok::Copy { len, code } => {
                self.h[0][NUM_LITERAL + prefix_encode(len).0 as usize] += 1;
                self.h[4][prefix_encode(code).0 as usize] += 1;
            }
        }
    }
}

/// Code lengths for one histogram; at least one symbol is always present.
fn code_lengths(freq: &[u32]) -> Vec<u8> {
    if freq.iter().all(|&f| f == 0) {
        let mut l = vec![0u8; freq.len()];
        l[0] = 1;
        return l;
    }
    build_lengths(freq, 15)
}

/// Extra bits that follow a length/distance prefix symbol.
fn prefix_extra_bits(p: usize) -> u64 {
    if p < 4 { 0 } else { ((p - 2) >> 1) as u64 }
}

/// Estimated size in bits of coding `h` with its own optimal codes (header cost ignored).
fn estimate_bits(h: &Hist) -> u64 {
    let mut bits = 0u64;
    for f in &h.h {
        // A code with one symbol costs nothing per symbol.
        if f.iter().filter(|&&x| x > 0).count() > 1 {
            let l = code_lengths(f);
            bits += f.iter().zip(&l).map(|(&f, &l)| f as u64 * l as u64).sum::<u64>();
        }
    }
    for s in 0..NUM_LENGTH {
        bits += h.h[0][NUM_LITERAL + s] as u64 * prefix_extra_bits(s);
    }
    for s in 0..NUM_DIST {
        bits += h.h[4][s] as u64 * prefix_extra_bits(s);
    }
    bits
}

/// Encode one run of code lengths with the code-length alphabet: (symbol, extra bits, extra value).
fn rle_code_lengths(lens: &[u8]) -> Vec<(u8, u32, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut prev_nonzero = 8u8;
    while i < lens.len() {
        let v = lens[i];
        let mut run = 1;
        while i + run < lens.len() && lens[i + run] == v {
            run += 1;
        }
        i += run;
        if v == 0 {
            while run >= 3 {
                if run >= 11 {
                    let n = run.min(138);
                    out.push((18, 7, (n - 11) as u32));
                    run -= n;
                } else {
                    let n = run.min(10);
                    out.push((17, 3, (n - 3) as u32));
                    run -= n;
                }
            }
            for _ in 0..run {
                out.push((0, 0, 0));
            }
        } else {
            if v != prev_nonzero || run < 3 {
                out.push((v, 0, 0));
                run -= 1;
                prev_nonzero = v;
            }
            while run >= 3 {
                let n = run.min(6);
                out.push((16, 2, (n - 3) as u32));
                run -= n;
            }
            for _ in 0..run {
                out.push((v, 0, 0));
            }
        }
    }
    out
}

/// A serialized prefix code ready for symbol emission.
struct Written {
    codes: Vec<u16>,
    lens: Vec<u8>,
    /// Exactly one symbol is used: symbols cost no bits.
    single: bool,
}

fn write_code(w: &mut LsbWriter, lens: &[u8]) -> Written {
    let used: Vec<usize> = lens.iter().enumerate().filter(|(_, l)| **l > 0).map(|(s, _)| s).collect();
    let codes = lsb_codes(lens);
    if used.len() == 1 && used[0] < 256 || used.len() == 2 && used[1] < 256 {
        // Simple code: up to two symbols below 256 given directly.
        w.put(1, 1);
        w.put(used.len() as u32 - 1, 1);
        let first_8 = used[0] > 1;
        w.put(first_8 as u32, 1);
        w.put(used[0] as u32, if first_8 { 8 } else { 1 });
        if used.len() == 2 {
            w.put(used[1] as u32, 8);
        }
        return Written { codes, lens: lens.to_vec(), single: used.len() == 1 };
    }
    w.put(0, 1);
    let tokens = rle_code_lengths(lens);
    let mut freq = [0u32; 19];
    for &(s, _, _) in &tokens {
        freq[s as usize] += 1;
    }
    let cl_lens = build_lengths(&freq, 7);
    let cl_codes = lsb_codes(&cl_lens);
    let cl_single = cl_lens.iter().filter(|&&l| l > 0).count() == 1;
    let n = CODE_LENGTH_ORDER.iter().rposition(|&i| cl_lens[i] > 0).map_or(4, |p| p + 1).max(4);
    w.put(n as u32 - 4, 4);
    for &o in CODE_LENGTH_ORDER.iter().take(n) {
        w.put(cl_lens[o] as u32, 3);
    }
    w.put(0, 1); // no explicit max_symbol: every alphabet entry is described
    for &(s, eb, ev) in &tokens {
        if !cl_single {
            w.put(cl_codes[s as usize] as u32, cl_lens[s as usize] as u32);
        }
        if eb > 0 {
            w.put(ev, eb);
        }
    }
    Written { codes, lens: lens.to_vec(), single: used.len() == 1 }
}

#[inline(always)]
fn emit(w: &mut LsbWriter, c: &Written, sym: usize) {
    if !c.single {
        w.put(c.codes[sym] as u32, c.lens[sym] as u32);
    }
}

/// Write an entropy-coded image: cache setup, (optionally) a single-group meta prefix flag, the
/// prefix codes and the tokens.
fn write_image(w: &mut LsbWriter, toks: &[Tok], cache_bits: u32, main: bool) {
    let cache_size = if cache_bits > 0 { 1usize << cache_bits } else { 0 };
    if cache_bits > 0 {
        w.put(1, 1);
        w.put(cache_bits, 4);
    } else {
        w.put(0, 1);
    }
    if main {
        w.put(0, 1); // one prefix group for the whole image
    }
    let mut hist = Hist::new(cache_size);
    for &t in toks {
        hist.add(t);
    }
    let written: Vec<Written> = hist.h.iter().map(|f| write_code(w, &code_lengths(f))).collect();
    for &t in toks {
        match t {
            Tok::Literal(p) => {
                emit(w, &written[0], ((p >> 8) & 255) as usize);
                emit(w, &written[1], ((p >> 16) & 255) as usize);
                emit(w, &written[2], (p & 255) as usize);
                emit(w, &written[3], (p >> 24) as usize);
            }
            Tok::Cache(i) => emit(w, &written[0], NUM_LITERAL + NUM_LENGTH + i as usize),
            Tok::Copy { len, code } => {
                let (lp, le, lv) = prefix_encode(len);
                emit(w, &written[0], NUM_LITERAL + lp as usize);
                if le > 0 {
                    w.put(lv, le);
                }
                let (dp, de, dv) = prefix_encode(code);
                emit(w, &written[4], dp as usize);
                if de > 0 {
                    w.put(dv, de);
                }
            }
        }
    }
}

/// Parse, pick a colour cache size, and write one image.
fn encode_image(w: &mut LsbWriter, px: &[u32], xsize: usize, level: u32, main: bool) {
    let plane = plane_table();
    let (depth, lazy) = match level {
        0..=1 => (2, false),
        2..=3 => (8, false),
        4..=5 => (24, true),
        6..=7 => (64, true),
        _ => (192, true),
    };
    let steps = parse(px, xsize, depth, lazy);
    let candidates: &[u32] = match level {
        0..=1 => &[0],
        2..=4 => &[0, 6],
        _ => &[0, 3, 6, 9],
    };
    let mut best: Option<(u64, u32, Vec<Tok>)> = None;
    for &bits in candidates {
        let toks = to_tokens(px, &steps, xsize, bits, &plane);
        let mut hist = Hist::new(if bits > 0 { 1 << bits } else { 0 });
        toks.iter().for_each(|&t| hist.add(t));
        let est = estimate_bits(&hist);
        if best.as_ref().is_none_or(|b| est < b.0) {
            best = Some((est, bits, toks));
        }
    }
    let (_, bits, toks) = best.expect("at least one cache size");
    write_image(w, &toks, bits, main);
}

// ------------------------------------------------------------------ whole image

fn predictor_bits(w: usize, h: usize) -> u32 {
    match w * h {
        0..=1_000_000 => 4,
        1_000_001..=8_000_000 => 5,
        _ => 6,
    }
}

fn header(w: &mut LsbWriter, width: usize, height: usize, alpha: bool) {
    w.put(SIGNATURE as u32, 8);
    w.put(width as u32 - 1, 14);
    w.put(height as u32 - 1, 14);
    w.put(alpha as u32, 1);
    w.put(0, 3);
}

fn encode_palette(argb: &[u32], width: usize, height: usize, alpha: bool, palette: &[u32], level: u32) -> Vec<u8> {
    let mut w = LsbWriter::new();
    header(&mut w, width, height, alpha);
    let bits = bundle_bits(palette.len());
    w.put(1, 1); // transform present
    w.put(3, 2); // colour indexing
    w.put(palette.len() as u32 - 1, 8);
    let mut deltas = palette.to_vec();
    for i in (1..deltas.len()).rev() {
        deltas[i] = sub_pixels(deltas[i], deltas[i - 1]);
    }
    encode_image(&mut w, &deltas, palette.len(), level.min(3), false);
    w.put(0, 1); // no more transforms
    let indexed = index_pixels(argb, width, height, palette, bits);
    encode_image(&mut w, &indexed, div_round_up(width, bits), level, true);
    w.finish()
}

fn encode_direct(argb: &[u32], width: usize, height: usize, alpha: bool, level: u32, predictor: bool) -> Vec<u8> {
    let mut w = LsbWriter::new();
    header(&mut w, width, height, alpha);
    let mut px = argb.to_vec();
    w.put(1, 1);
    w.put(2, 2); // subtract green
    subtract_green(&mut px);
    if predictor {
        let bits = predictor_bits(width, height);
        let (res, modes) = predictor_transform(&px, width, height, bits);
        w.put(1, 1);
        w.put(0, 2);
        w.put(bits - 2, 3);
        encode_image(&mut w, &modes, div_round_up(width, bits), level.min(3), false);
        px = res;
    }
    w.put(0, 1);
    encode_image(&mut w, &px, width, level, true);
    w.finish()
}

/// Encode packed ARGB pixels as a VP8L bitstream (the payload of a `VP8L` chunk).
pub fn encode(argb: &[u32], width: usize, height: usize, has_alpha: bool, opts: Options) -> Vec<u8> {
    assert_eq!(argb.len(), width * height);
    assert!((1..=16384).contains(&width) && (1..=16384).contains(&height));
    let mut px = argb.to_vec();
    if opts.clean_alpha && has_alpha {
        px.iter_mut().filter(|p| **p >> 24 == 0).for_each(|p| *p = 0);
    }
    let level = opts.level.min(9);
    let mut candidates: Vec<Vec<u8>> = Vec::new();
    if let Some(palette) = small_palette(&px) {
        candidates.push(encode_palette(&px, width, height, has_alpha, &palette, level));
        // Palette images with smooth gradients can still do better with prediction.
        if level >= 6 && palette.len() > 16 {
            candidates.push(encode_direct(&px, width, height, has_alpha, level, true));
        }
    } else {
        candidates.push(encode_direct(&px, width, height, has_alpha, level, true));
        if level >= 3 {
            candidates.push(encode_direct(&px, width, height, has_alpha, level, false));
        }
    }
    candidates.into_iter().min_by_key(|c| c.len()).expect("at least one candidate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_codes_decode_back_to_the_same_distance() {
        let plane = plane_table();
        for xsize in [1usize, 2, 3, 5, 8, 9, 10, 17, 40, 257] {
            for dist in 1..3000usize {
                let code = distance_code(dist, xsize, &plane) as usize;
                let back = if code > 120 {
                    code - 120
                } else {
                    let (dx, dy) = DISTANCE_MAP[code - 1];
                    (dx as isize + dy as isize * xsize as isize).max(1) as usize
                };
                assert_eq!(back, dist, "xsize {xsize} dist {dist} code {code}");
            }
        }
    }

    #[test]
    fn prefix_encode_matches_the_decoder_formula() {
        for value in 1..5000u32 {
            let (prefix, extra, ev) = prefix_encode(value);
            let back = if prefix < 4 { prefix + 1 } else { ((2 + (prefix & 1)) << ((prefix - 2) >> 1)) + ev + 1 };
            assert_eq!(back, value);
            assert_eq!(extra, if prefix < 4 { 0 } else { (prefix - 2) >> 1 });
        }
    }

    #[test]
    fn code_length_rle_decodes_back() {
        // Re-expand the tokens the way the decoder does and compare.
        let cases: Vec<Vec<u8>> = vec![
            vec![8; 40],
            vec![0; 300],
            vec![3, 3, 3, 3, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 8, 8, 8, 8],
            (0..280).map(|i| (i % 5) as u8).collect(),
            vec![1, 2, 3],
        ];
        for lens in cases {
            let mut out: Vec<u8> = Vec::new();
            let mut prev = 8u8;
            for (sym, _, ev) in rle_code_lengths(&lens) {
                match sym {
                    16 => out.extend(std::iter::repeat_n(prev, 3 + ev as usize)),
                    17 => out.extend(std::iter::repeat_n(0, 3 + ev as usize)),
                    18 => out.extend(std::iter::repeat_n(0, 11 + ev as usize)),
                    v => {
                        out.push(v);
                        if v != 0 {
                            prev = v;
                        }
                    }
                }
            }
            assert_eq!(out, lens);
        }
    }
}
