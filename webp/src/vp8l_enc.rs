//! VP8L (WebP lossless) encoder.
//!
//! Candidate pipelines are tried and the smallest kept: a colour-indexing transform for images with
//! at most 256 colours (with several pixels bundled per stored pixel), or subtract-green plus a
//! per-tile predictor (and optionally cross-colour) for everything else. The transformed pixels go
//! through LZ77 with plane-aware distance codes, an optional colour cache and canonical prefix codes.

use super::cluster::{self, ALPHABETS, Sparse};
use super::vp8l::{DISTANCE_MAP, SIGNATURE, cache_index, div_round_up, predict};
use rayon::prelude::*;
use rc_core::util::bitio::LsbWriter;
use rc_core::util::huffman::{build_lengths, lsb_codes};

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

/// `(t * c) >> 5` with both operands as signed bytes: the cross-colour correction term.
#[inline(always)]
fn color_delta(t: i8, c: i8) -> i32 {
    (t as i32 * c as i32) >> 5
}

/// Shannon entropy in bits of a 256-bin histogram.
fn entropy256(hist: &[u32; 256]) -> f64 {
    let total: f64 = hist.iter().map(|&c| c as f64).sum();
    if total == 0.0 {
        return 0.0;
    }
    hist.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (total / c as f64).log2()).sum()
}

/// Best of `candidates` by `cost`, then refined by +-3 around the winner.
fn best_coefficient(candidates: impl Iterator<Item = i32>, mut cost: impl FnMut(i8) -> f64) -> i8 {
    let mut best = (f64::MAX, 0i32);
    for t in candidates {
        let c = cost(t as i8);
        if c < best.0 || (c == best.0 && t.abs() < best.1.abs()) {
            best = (c, t);
        }
    }
    for t in best.1 - 3..=best.1 + 3 {
        if (-128..128).contains(&t) {
            let c = cost(t as i8);
            if c < best.0 {
                best = (c, t);
            }
        }
    }
    best.1 as i8
}

/// Per-tile decorrelation of red and blue from green (and blue from red). Returns the corrected
/// pixels and the coefficient image (red = red_to_blue, green = green_to_blue, blue = green_to_red).
fn cross_color_transform(src: &[u32], w: usize, h: usize, bits: u32) -> (Vec<u32>, Vec<u32>) {
    let (tw, th) = (div_round_up(w, bits), div_round_up(h, bits));
    let tile = 1usize << bits;
    let coefs: Vec<(i8, i8, i8)> = (0..th)
        .into_par_iter()
        .flat_map_iter(|ty| {
            (0..tw).map(move |tx| {
                let (x0, y0) = (tx * tile, ty * tile);
                let (x1, y1) = ((x0 + tile).min(w), (y0 + tile).min(h));
                let px = || (y0..y1).flat_map(move |y| (x0..x1).map(move |x| src[y * w + x]));
                let hist_of = |f: &dyn Fn(u32) -> u8| {
                    let mut hist = [0u32; 256];
                    px().for_each(|p| hist[f(p) as usize] += 1);
                    entropy256(&hist)
                };
                let sweep = || (-64..=64).step_by(8);
                let g2r = best_coefficient(sweep(), |t| {
                    hist_of(&|p| ((p >> 16) as i32 - color_delta(t, (p >> 8) as i8)) as u8)
                });
                let g2b =
                    best_coefficient(sweep(), |t| hist_of(&|p| (p as i32 - color_delta(t, (p >> 8) as i8)) as u8));
                let r2b = best_coefficient(sweep(), |t| {
                    hist_of(&|p| (p as i32 - color_delta(g2b, (p >> 8) as i8) - color_delta(t, (p >> 16) as i8)) as u8)
                });
                // Zero coefficients are free to code; only keep ones that pay for themselves.
                let base = hist_of(&|p| (p >> 16) as u8) + hist_of(&|p| p as u8);
                let tuned = hist_of(&|p| ((p >> 16) as i32 - color_delta(g2r, (p >> 8) as i8)) as u8)
                    + hist_of(&|p| {
                        (p as i32 - color_delta(g2b, (p >> 8) as i8) - color_delta(r2b, (p >> 16) as i8)) as u8
                    });
                if tuned + 12.0 < base { (g2r, g2b, r2b) } else { (0, 0, 0) }
            })
        })
        .collect();
    let mut out = vec![0u32; src.len()];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let p = src[y * w + x];
            let (g2r, g2b, r2b) = coefs[(y >> bits) * tw + (x >> bits)];
            let green = (p >> 8) as i8;
            let red = (p >> 16) as u8;
            let new_red = (red as i32 - color_delta(g2r, green)) as u8;
            let new_blue = (p as i32 - color_delta(g2b, green) - color_delta(r2b, red as i8)) as u8;
            row[x] = (p & 0xFF00_FF00) | (new_red as u32) << 16 | new_blue as u32;
        }
    });
    let data = coefs
        .into_iter()
        .map(|(g2r, g2b, r2b)| 0xFF00_0000 | (r2b as u8 as u32) << 16 | (g2b as u8 as u32) << 8 | g2r as u8 as u32)
        .collect();
    (out, data)
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

// ------------------------------------------------------------------ optimal parse

/// Matches at least this long are taken as they come: no candidate search inside them.
const LONG_MATCH: usize = 192;

/// Per-symbol costs in bits, derived from the statistics of a previous parse.
struct Costs {
    green: Vec<f32>,
    red: Vec<f32>,
    blue: Vec<f32>,
    alpha: Vec<f32>,
    dist: Vec<f32>,
}

fn costs_from(h: &Hist) -> Costs {
    let cost = |f: &Vec<u32>| -> Vec<f32> {
        let total = f.iter().map(|&x| x as f64).sum::<f64>().max(1.0);
        f.iter().map(|&x| if x > 0 { (total / x as f64).log2() as f32 } else { (total.log2() + 3.0) as f32 }).collect()
    };
    Costs { green: cost(&h.h[0]), red: cost(&h.h[1]), blue: cost(&h.h[2]), alpha: cost(&h.h[3]), dist: cost(&h.h[4]) }
}

/// Candidate matches for each position, found with hash chains over pixel pairs. Every pixel must be
/// visited in order (`candidates` also inserts the position).
struct Finder {
    head: Vec<u32>,
    prev: Vec<u32>,
    depth: usize,
}

impl Finder {
    fn new(depth: usize) -> Self {
        Finder { head: vec![u32::MAX; 1 << HASH_BITS], prev: vec![u32::MAX; WINDOW], depth }
    }

    /// Append (length, distance) candidates at `i`: strictly longer matches found along the chain
    /// (nearest first) plus the common 2-D neighbours, then index `i`.
    fn candidates(&mut self, px: &[u32], i: usize, xsize: usize, out: &mut Vec<(usize, usize)>) {
        let n = px.len();
        let max = (n - i).min(MAX_LEN);
        out.clear();
        if max >= 3 && i + 1 < n {
            let mut best = 2;
            for d in [1, xsize.saturating_sub(1), xsize, xsize + 1] {
                if d >= 1 && d <= i && d <= MAX_DIST {
                    let l = common_len(px, i - d, i, max);
                    if l >= 3 {
                        out.push((l, d));
                        best = best.max(l);
                    }
                }
            }
            let mut cand = self.head[hash_pair(px[i], px[i + 1])];
            let mut left = self.depth;
            while cand != u32::MAX && left > 0 {
                let c = cand as usize;
                if c >= i || i - c > MAX_DIST {
                    break;
                }
                if px[c + best.min(max - 1)] == px[i + best.min(max - 1)] {
                    let l = common_len(px, c, i, max);
                    if l > best {
                        best = l;
                        out.push((l, i - c));
                        if l == max {
                            break;
                        }
                    }
                }
                let next = self.prev[c & (WINDOW - 1)];
                if next == u32::MAX || next as usize >= c {
                    break;
                }
                cand = next;
                left -= 1;
            }
        }
        self.insert(px, i);
    }

    fn insert(&mut self, px: &[u32], i: usize) {
        if i + 1 < px.len() {
            let h = hash_pair(px[i], px[i + 1]);
            self.prev[i & (WINDOW - 1)] = self.head[h];
            self.head[h] = i as u32;
        }
    }
}

/// Lengths worth considering for a match whose longest usable length is `max`: all short ones,
/// then the ends of each length-code bucket (inside a bucket only the extra bits change).
fn lengths_to_try(max: usize, out: &mut Vec<usize>) {
    out.clear();
    for l in 3..=max.min(12) {
        out.push(l);
    }
    let mut l = 13;
    while l <= max {
        out.push(l);
        l += 1 + (l >> 3).max(1);
    }
    if max > 12 && out.last() != Some(&max) {
        out.push(max);
    }
}

/// Cheapest parse under `costs`: shortest path over literal/cache pixels and candidate copies.
fn optimal_parse(
    px: &[u32],
    xsize: usize,
    depth: usize,
    cache_bits: u32,
    costs: &Costs,
    plane: &[u8; 8 * 16],
) -> Vec<Step> {
    let n = px.len();
    let mut cost = vec![f32::INFINITY; n + 1];
    let mut step = vec![Step { len: 1, dist: 0 }; n + 1];
    cost[0] = 0.0;
    let mut finder = Finder::new(depth);
    let mut cache = vec![0u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
    let mut cands = Vec::new();
    let mut lens = Vec::new();
    let cache_base = NUM_LITERAL + NUM_LENGTH;
    // Inside a very long copy, re-measuring matches at every pixel is quadratic for little gain.
    let mut skip_until = 0usize;
    for i in 0..n {
        let ci = cost[i];
        let p = px[i];
        // A single pixel: a cache hit if the pixel is where the cache says, else a literal.
        let single = if cache_bits > 0 && cache[cache_index(p, cache_bits)] == p {
            costs.green[cache_base + cache_index(p, cache_bits)]
        } else {
            costs.green[((p >> 8) & 255) as usize]
                + costs.red[((p >> 16) & 255) as usize]
                + costs.blue[(p & 255) as usize]
                + costs.alpha[(p >> 24) as usize]
        };
        if ci + single < cost[i + 1] {
            cost[i + 1] = ci + single;
            step[i + 1] = Step { len: 1, dist: 0 };
        }
        if i < skip_until {
            finder.insert(px, i);
            if cache_bits > 0 {
                cache[cache_index(p, cache_bits)] = p;
            }
            continue;
        }
        finder.candidates(px, i, xsize, &mut cands);
        if let Some(&(l, _)) = cands.last()
            && l >= LONG_MATCH
        {
            skip_until = i + l - 1;
        }
        for &(max, dist) in &cands {
            let code = distance_code(dist, xsize, plane);
            let (dsym, dextra, _) = prefix_encode(code);
            let dcost = costs.dist[dsym as usize] + dextra as f32;
            lengths_to_try(max, &mut lens);
            for &l in &lens {
                let (lsym, lextra, _) = prefix_encode(l as u32);
                let c = ci + costs.green[NUM_LITERAL + lsym as usize] + lextra as f32 + dcost;
                if c < cost[i + l] {
                    cost[i + l] = c;
                    step[i + l] = Step { len: l as u32, dist: dist as u32 };
                }
            }
        }
        if cache_bits > 0 {
            cache[cache_index(p, cache_bits)] = p;
        }
    }
    // Walk back from the end.
    let mut steps = Vec::new();
    let mut i = n;
    while i > 0 {
        let s = step[i];
        steps.push(s);
        i -= s.len as usize;
    }
    steps.reverse();
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

/// Symbol (alphabet, index) events of one token, to feed histograms.
fn token_events(t: Tok, mut f: impl FnMut(usize, usize)) {
    match t {
        Tok::Literal(p) => {
            f(0, ((p >> 8) & 255) as usize);
            f(1, ((p >> 16) & 255) as usize);
            f(2, (p & 255) as usize);
            f(3, (p >> 24) as usize);
        }
        Tok::Cache(i) => f(0, NUM_LITERAL + NUM_LENGTH + i as usize),
        Tok::Copy { len, code } => {
            f(0, NUM_LITERAL + prefix_encode(len).0 as usize);
            f(4, prefix_encode(code).0 as usize);
        }
    }
}

/// Pixels a token covers.
fn token_len(t: Tok) -> usize {
    match t {
        Tok::Copy { len, .. } => len as usize,
        _ => 1,
    }
}

/// Plan for sharing prefix codes between tiles of the image.
struct GroupPlan {
    meta_bits: u32,
    tiles_w: usize,
    assign: Vec<u16>,
    groups: usize,
}

/// Cluster the image's tiles by their symbol statistics; `None` when one group is as good.
fn plan_groups(toks: &[Tok], xsize: usize, ysize: usize, cache_size: usize, level: u32) -> Option<GroupPlan> {
    if level < 3 || xsize * ysize < 4096 {
        return None;
    }
    // Tiles of 2^bits pixels a side, as large as needed to keep their number manageable.
    let mut meta_bits = 4;
    while meta_bits < 9 && div_round_up(xsize, meta_bits) * div_round_up(ysize, meta_bits) > 2500 {
        meta_bits += 1;
    }
    let (tiles_w, tiles_h) = (div_round_up(xsize, meta_bits), div_round_up(ysize, meta_bits));
    let sizes = [NUM_LITERAL + NUM_LENGTH + cache_size, NUM_LITERAL, NUM_LITERAL, NUM_LITERAL, NUM_DIST];
    let offsets: Vec<usize> = sizes
        .iter()
        .scan(0, |acc, &n| {
            let o = *acc;
            *acc += n;
            Some(o)
        })
        .collect();
    let stride: usize = sizes.iter().sum();
    let mut counts = vec![0u32; tiles_w * tiles_h * stride];
    let mut pos = 0usize;
    for &t in toks {
        let (y, x) = (pos / xsize, pos % xsize);
        let base = ((y >> meta_bits) * tiles_w + (x >> meta_bits)) * stride;
        token_events(t, |a, s| counts[base + offsets[a] + s] += 1);
        pos += token_len(t);
    }
    let tiles: Vec<Sparse> = counts
        .chunks_exact(stride)
        .map(|c| {
            let mut v = Sparse::new();
            for a in 0..ALPHABETS {
                for (s, &n) in c[offsets[a]..offsets[a] + sizes[a]].iter().enumerate() {
                    if n > 0 {
                        v.push((a as u8, s as u16, n));
                    }
                }
            }
            v
        })
        .collect();
    drop(counts);
    let (max_groups, passes) = match level {
        0..=4 => (24, 1),
        5..=7 => (96, 2),
        _ => (256, 3),
    };
    let g = cluster::cluster(&tiles, &sizes, max_groups, passes);
    // The meta image and its per-tile index cost something too.
    let overhead = (tiles_w * tiles_h) as f64 * 1.0 + 64.0;
    if g.groups < 2 || g.saving_bits <= overhead {
        return None;
    }
    Some(GroupPlan { meta_bits, tiles_w, assign: g.assign, groups: g.groups })
}

/// Write an entropy-coded image: cache setup, optional meta prefix groups, prefix codes, tokens.
fn write_image(w: &mut LsbWriter, toks: &[Tok], cache_bits: u32, main: bool, xsize: usize, ysize: usize, level: u32) {
    let cache_size = if cache_bits > 0 { 1usize << cache_bits } else { 0 };
    if cache_bits > 0 {
        w.put(1, 1);
        w.put(cache_bits, 4);
    } else {
        w.put(0, 1);
    }
    let plan = if main { plan_groups(toks, xsize, ysize, cache_size, level) } else { None };
    if main {
        match &plan {
            Some(p) => {
                w.put(1, 1);
                w.put(p.meta_bits - 2, 3);
                let meta: Vec<u32> = p.assign.iter().map(|&g| 0xFF00_0000 | (g as u32) << 8).collect();
                encode_image(w, &meta, p.tiles_w, level.min(3), false);
            }
            None => w.put(0, 1),
        }
    }
    let groups = plan.as_ref().map_or(1, |p| p.groups);
    let mut hists: Vec<Hist> = (0..groups).map(|_| Hist::new(cache_size)).collect();
    let group_at = |pos: usize| -> usize {
        match &plan {
            Some(p) => {
                let (y, x) = (pos / xsize, pos % xsize);
                p.assign[(y >> p.meta_bits) * p.tiles_w + (x >> p.meta_bits)] as usize
            }
            None => 0,
        }
    };
    let mut pos = 0usize;
    for &t in toks {
        hists[group_at(pos)].add(t);
        pos += token_len(t);
    }
    let written: Vec<Vec<Written>> =
        hists.iter().map(|h| h.h.iter().map(|f| write_code(w, &code_lengths(f))).collect()).collect();
    let mut pos = 0usize;
    for &t in toks {
        let written = &written[group_at(pos)];
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
        pos += token_len(t);
    }
}

/// Parse, pick a colour cache size, and write one image.
fn encode_image(w: &mut LsbWriter, px: &[u32], xsize: usize, level: u32, main: bool) {
    let plane = plane_table();
    let ysize = px.len() / xsize.max(1);
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
    let estimate = |toks: &[Tok], bits: u32| -> u64 {
        let mut hist = Hist::new(if bits > 0 { 1 << bits } else { 0 });
        toks.iter().for_each(|&t| hist.add(t));
        estimate_bits(&hist)
    };
    let mut best: Option<(u64, u32, Vec<Step>, Vec<Tok>)> = None;
    for &bits in candidates {
        let toks = to_tokens(px, &steps, xsize, bits, &plane);
        let est = estimate(&toks, bits);
        if best.as_ref().is_none_or(|b| est < b.0) {
            best = Some((est, bits, steps.clone(), toks));
        }
    }
    let (mut est, bits, mut steps, mut toks) = best.expect("at least one cache size");
    // Re-parse with costs learned from the current best parse, keeping whatever improves.
    let rounds = match level {
        0..=4 => 0,
        5..=6 => 1,
        7..=8 => 2,
        _ => 3,
    };
    for _ in 0..rounds {
        let mut hist = Hist::new(if bits > 0 { 1 << bits } else { 0 });
        toks.iter().for_each(|&t| hist.add(t));
        let candidate = optimal_parse(px, xsize, depth.min(96), bits, &costs_from(&hist), &plane);
        let cand_toks = to_tokens(px, &candidate, xsize, bits, &plane);
        let cand_est = estimate(&cand_toks, bits);
        if cand_est < est {
            (est, steps, toks) = (cand_est, candidate, cand_toks);
        } else {
            break;
        }
    }
    let _ = steps;
    write_image(w, &toks, bits, main, xsize, ysize, level);
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

/// One transform as it appears in the bitstream, with its side image.
enum Transform {
    SubtractGreen,
    Predictor {
        bits: u32,
        modes: Vec<u32>,
        xsize: usize,
    },
    CrossColor {
        bits: u32,
        data: Vec<u32>,
        xsize: usize,
    },
    /// Palette colours, already delta-coded.
    Palette {
        deltas: Vec<u32>,
    },
}

/// An image after its transforms: what is left to entropy-code.
struct Prepared {
    transforms: Vec<Transform>,
    pixels: Vec<u32>,
    xsize: usize,
}

/// Pipeline choices for the non-palette path.
#[derive(Clone, Copy)]
struct Direct {
    /// Subtract green from red and blue first.
    sg: bool,
    predictor: bool,
    cross: bool,
    /// Predictor tile size (log2); 0 = pick by image size.
    pred_bits: u32,
}

fn prepare_direct(argb: &[u32], width: usize, height: usize, d: Direct) -> Prepared {
    let mut px = argb.to_vec();
    let mut transforms = Vec::new();
    if d.sg {
        subtract_green(&mut px);
        transforms.push(Transform::SubtractGreen);
    }
    if d.predictor {
        let bits = if d.pred_bits > 0 { d.pred_bits } else { predictor_bits(width, height) };
        let (res, modes) = predictor_transform(&px, width, height, bits);
        transforms.push(Transform::Predictor { bits, modes, xsize: div_round_up(width, bits) });
        px = res;
    }
    if d.cross {
        let bits = 5;
        let (fixed, data) = cross_color_transform(&px, width, height, bits);
        transforms.push(Transform::CrossColor { bits, data, xsize: div_round_up(width, bits) });
        px = fixed;
    }
    Prepared { transforms, pixels: px, xsize: width }
}

fn prepare_palette(argb: &[u32], width: usize, height: usize, palette: &[u32]) -> Prepared {
    let bits = bundle_bits(palette.len());
    let mut deltas = palette.to_vec();
    for i in (1..deltas.len()).rev() {
        deltas[i] = sub_pixels(deltas[i], deltas[i - 1]);
    }
    Prepared {
        transforms: vec![Transform::Palette { deltas }],
        pixels: index_pixels(argb, width, height, palette, bits),
        xsize: div_round_up(width, bits),
    }
}

/// Quick size estimate in bits: greedy parse and entropy, with and without a small colour cache,
/// but no grouping.
fn quick_cost(px: &[u32], xsize: usize) -> u64 {
    let plane = plane_table();
    let steps = parse(px, xsize, 8, false);
    [0u32, 6]
        .iter()
        .map(|&bits| {
            let toks = to_tokens(px, &steps, xsize, bits, &plane);
            let mut hist = Hist::new(if bits > 0 { 1 << bits } else { 0 });
            toks.iter().for_each(|&t| hist.add(t));
            estimate_bits(&hist)
        })
        .min()
        .unwrap_or(0)
}

impl Prepared {
    fn estimate(&self) -> u64 {
        let side: u64 = self
            .transforms
            .iter()
            .map(|t| match t {
                Transform::Predictor { modes, xsize, .. } => quick_cost(modes, *xsize),
                Transform::CrossColor { data, xsize, .. } => quick_cost(data, *xsize),
                Transform::Palette { deltas } => quick_cost(deltas, deltas.len()) + 8,
                Transform::SubtractGreen => 0,
            })
            .sum();
        side + quick_cost(&self.pixels, self.xsize)
    }

    fn write(&self, width: usize, height: usize, alpha: bool, level: u32) -> Vec<u8> {
        let mut w = LsbWriter::new();
        header(&mut w, width, height, alpha);
        for t in &self.transforms {
            w.put(1, 1);
            match t {
                Transform::SubtractGreen => w.put(2, 2),
                Transform::Predictor { bits, modes, xsize } => {
                    w.put(0, 2);
                    w.put(bits - 2, 3);
                    encode_image(&mut w, modes, *xsize, level.min(3), false);
                }
                Transform::CrossColor { bits, data, xsize } => {
                    w.put(1, 2);
                    w.put(bits - 2, 3);
                    encode_image(&mut w, data, *xsize, level.min(3), false);
                }
                Transform::Palette { deltas } => {
                    w.put(3, 2);
                    w.put(deltas.len() as u32 - 1, 8);
                    encode_image(&mut w, deltas, deltas.len(), level.min(3), false);
                }
            }
        }
        w.put(0, 1);
        encode_image(&mut w, &self.pixels, self.xsize, level, true);
        w.finish()
    }
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
    let direct = |sg, predictor, cross, pred_bits| Direct { sg, predictor, cross, pred_bits };
    let mut candidates: Vec<Prepared> = Vec::new();
    if let Some(palette) = small_palette(&px) {
        candidates.push(prepare_palette(&px, width, height, &palette));
        // Palette images with smooth gradients can still do better with prediction.
        if level >= 6 && palette.len() > 16 {
            candidates.push(prepare_direct(&px, width, height, direct(true, true, true, 0)));
        }
    } else {
        candidates.push(prepare_direct(&px, width, height, direct(true, true, level >= 4, 0)));
        if level >= 3 {
            // Graphics often do best with no prediction at all.
            candidates.push(prepare_direct(&px, width, height, direct(true, false, false, 0)));
        }
        if level >= 5 {
            // Photographic content: let cross-colour do the green decorrelation instead.
            candidates.push(prepare_direct(&px, width, height, direct(false, true, true, 0)));
        }
        if level >= 6 {
            for bits in [3, 5] {
                candidates.push(prepare_direct(&px, width, height, direct(true, true, true, bits)));
            }
        }
    }
    // Rank the pipelines cheaply, then spend the full effort on the best one (more at higher levels).
    let keep = match level {
        0..=5 => 1,
        6..=8 => 2,
        _ => 3,
    };
    let mut ranked: Vec<(u64, Prepared)> = candidates.into_iter().map(|c| (c.estimate(), c)).collect();
    ranked.sort_by_key(|r| r.0);
    ranked
        .into_iter()
        .take(keep)
        .map(|(_, c)| c.write(width, height, has_alpha, level))
        .min_by_key(|c| c.len())
        .expect("at least one candidate")
}

/// Encode an 8-bit plane as the headerless lossless stream stored in an `ALPH` chunk (the plane
/// travels in the green channel).
pub fn encode_plane(plane: &[u8], width: usize, height: usize, level: u32) -> Vec<u8> {
    let argb: Vec<u32> = plane.iter().map(|&a| 0xFF00_0000 | (a as u32) << 8).collect();
    let mut stream = encode(&argb, width, height, false, Options { level, clean_alpha: false });
    // The 40-bit header (signature, size, alpha hint, version) is exactly five bytes.
    stream.drain(..5);
    stream
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
