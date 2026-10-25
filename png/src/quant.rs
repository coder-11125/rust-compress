//! Lossy palette quantization: median cut on a 5-5-5(+alpha) histogram, k-means refinement,
//! then error-diffusion remapping. The result is an ordinary RGB(A) image with at most
//! `colors` distinct colors, which the PNG encoder stores as an indexed image.

use rayon::prelude::*;
use rc_core::image::{ColorType, Image};
use std::collections::HashSet;

/// Perceptual channel weights (R, G, B, A) for color distance.
const W: [f32; 4] = [0.5, 1.0, 0.35, 0.8];
/// Squared weighted color distance above which a pixel counts as lying on an edge.
const EDGE_DIST2: f32 = 900.0;
const W_SUM: f32 = W[0] + W[1] + W[2] + W[3];

#[derive(Clone, Copy)]
struct Entry {
    c: [f32; 4],
    n: f32,
}

#[inline(always)]
fn dist2(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let mut d = 0.0;
    for i in 0..4 {
        let t = a[i] - b[i];
        d += W[i] * t * t;
    }
    d
}

#[inline(always)]
fn luma(c: &[f32; 4]) -> f32 {
    W[0] * c[0] + W[1] * c[1] + W[2] * c[2] + W[3] * c[3]
}

/// Palette sorted by weighted luma so nearest-color queries can prune by 1-D distance.
struct Palette {
    cols: Vec<[f32; 4]>,
    l: Vec<f32>,
}

impl Palette {
    fn new(mut cols: Vec<[f32; 4]>) -> Self {
        cols.sort_by(|a, b| luma(a).total_cmp(&luma(b)));
        let l = cols.iter().map(luma).collect();
        Palette { cols, l }
    }

    /// Index into `cols` of the nearest palette color.
    fn nearest(&self, p: &[f32; 4]) -> usize {
        let lp = luma(p);
        let start = self.l.partition_point(|&x| x < lp).min(self.cols.len() - 1);
        let mut best = start;
        let mut best_d = dist2(p, &self.cols[start]);
        // Cauchy-Schwarz: dist2 >= (delta luma)^2 / sum(weights).
        let mut lo = start;
        let mut hi = start + 1;
        loop {
            let mut progressed = false;
            if lo > 0 {
                let dl = lp - self.l[lo - 1];
                if dl * dl / W_SUM < best_d {
                    lo -= 1;
                    let d = dist2(p, &self.cols[lo]);
                    if d < best_d {
                        best_d = d;
                        best = lo;
                    }
                    progressed = true;
                }
            }
            if hi < self.cols.len() {
                let dl = self.l[hi] - lp;
                if dl * dl / W_SUM < best_d {
                    let d = dist2(p, &self.cols[hi]);
                    if d < best_d {
                        best_d = d;
                        best = hi;
                    }
                    hi += 1;
                    progressed = true;
                }
            }
            if !progressed {
                return best;
            }
        }
    }
}

fn rgba_at(img: &Image, i: usize) -> [u8; 4] {
    let ch = img.color.channels();
    let p = &img.data[i * ch..(i + 1) * ch];
    let px = match img.color {
        ColorType::Gray => [p[0], p[0], p[0], 255],
        ColorType::GrayAlpha => [p[0], p[0], p[0], p[1]],
        ColorType::Rgb => [p[0], p[1], p[2], 255],
        ColorType::Rgba => [p[0], p[1], p[2], p[3]],
    };
    // Invisible pixels all share one color.
    if px[3] == 0 { [0, 0, 0, 0] } else { px }
}

fn has_at_most_colors(px: &[[u8; 4]], limit: usize) -> bool {
    let mut set: HashSet<[u8; 4]> = HashSet::new();
    for &p in px {
        set.insert(p);
        if set.len() > limit {
            return false;
        }
    }
    true
}

fn median_cut(entries: &mut [Entry], target: usize) -> Vec<[f32; 4]> {
    struct B {
        start: usize,
        end: usize,
        score: f32,
        axis: usize,
    }
    fn stats(e: &[Entry]) -> ([f32; 4], f32, usize) {
        let total: f32 = e.iter().map(|x| x.n).sum();
        let mut mean = [0f32; 4];
        for x in e {
            for i in 0..4 {
                mean[i] += x.c[i] * x.n;
            }
        }
        for m in mean.iter_mut() {
            *m /= total;
        }
        let mut var = [0f32; 4];
        for x in e {
            for i in 0..4 {
                let d = x.c[i] - mean[i];
                var[i] += W[i] * d * d * x.n;
            }
        }
        let axis = (0..4).max_by(|&a, &b| var[a].total_cmp(&var[b])).unwrap();
        (mean, var.iter().sum(), axis)
    }
    let mk = |entries: &[Entry], start: usize, end: usize| {
        let (_, score, axis) = stats(&entries[start..end]);
        B { start, end, score: if end - start < 2 { 0.0 } else { score }, axis }
    };
    let mut boxes = vec![mk(entries, 0, entries.len())];
    while boxes.len() < target {
        let (bi, _) = match boxes.iter().enumerate().max_by(|a, b| a.1.score.total_cmp(&b.1.score)) {
            Some(x) => x,
            None => break,
        };
        if boxes[bi].score <= 0.0 {
            break;
        }
        let b = boxes.swap_remove(bi);
        let slice = &mut entries[b.start..b.end];
        slice.sort_unstable_by(|x, y| x.c[b.axis].total_cmp(&y.c[b.axis]));
        let total: f32 = slice.iter().map(|x| x.n).sum();
        let mut acc = 0.0;
        let mut split = 1;
        for (i, x) in slice.iter().enumerate() {
            acc += x.n;
            if acc >= total / 2.0 {
                split = (i + 1).clamp(1, slice.len() - 1);
                break;
            }
        }
        let mid = b.start + split;
        boxes.push(mk(entries, b.start, mid));
        boxes.push(mk(entries, mid, b.end));
    }
    boxes.iter().map(|b| stats(&entries[b.start..b.end]).0).collect()
}

fn refine(entries: &[Entry], cols: Vec<[f32; 4]>, iterations: usize) -> Vec<[f32; 4]> {
    let mut cols = cols;
    for _ in 0..iterations {
        let pal = Palette::new(cols.clone());
        let assign: Vec<usize> = entries.par_iter().map(|e| pal.nearest(&e.c)).collect();
        let mut sum = vec![[0f32; 4]; pal.cols.len()];
        let mut cnt = vec![0f32; pal.cols.len()];
        for (e, &a) in entries.iter().zip(&assign) {
            for i in 0..4 {
                sum[a][i] += e.c[i] * e.n;
            }
            cnt[a] += e.n;
        }
        cols = (0..pal.cols.len())
            .map(|k| {
                if cnt[k] > 0.0 {
                    let mut c = sum[k];
                    for v in c.iter_mut() {
                        *v /= cnt[k];
                    }
                    c
                } else {
                    pal.cols[k]
                }
            })
            .collect();
    }
    cols
}

/// Exact color counts, or `None` when there are too many distinct colors for that to be useful.
fn exact_histogram(px: &[[u8; 4]]) -> Option<Vec<Entry>> {
    const LIMIT: usize = 1 << 18;
    let mut keys: Vec<u32> = px.par_iter().map(|p| u32::from_le_bytes(*p)).collect();
    keys.par_sort_unstable();
    let mut entries = Vec::new();
    let mut i = 0;
    while i < keys.len() {
        let k = keys[i];
        let mut j = i + 1;
        while j < keys.len() && keys[j] == k {
            j += 1;
        }
        let p = k.to_le_bytes();
        entries.push(Entry { c: [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32], n: (j - i) as f32 });
        if entries.len() > LIMIT {
            return None;
        }
        i = j;
    }
    Some(entries)
}

/// 5-5-5 (+4-bit alpha) histogram with exact per-bucket means, for images with very many colors.
fn bucket_histogram(px: &[[u8; 4]], has_alpha: bool) -> Vec<Entry> {
    let size = if has_alpha { 1 << 19 } else { 1 << 15 };
    let mut hist = vec![[0u32; 5]; size];
    for p in px {
        let mut key = ((p[0] as usize >> 3) << 10) | ((p[1] as usize >> 3) << 5) | (p[2] as usize >> 3);
        if has_alpha {
            key |= (p[3] as usize >> 4) << 15;
        }
        let h = &mut hist[key];
        h[0] += 1;
        h[1] += p[0] as u32;
        h[2] += p[1] as u32;
        h[3] += p[2] as u32;
        h[4] += p[3] as u32;
    }
    hist.iter()
        .filter(|h| h[0] > 0)
        .map(|h| {
            let n = h[0] as f32;
            Entry { c: [h[1] as f32 / n, h[2] as f32 / n, h[3] as f32 / n, h[4] as f32 / n], n }
        })
        .collect()
}

/// Reduce `img` to at most `colors` colors. Images that already fit are returned unchanged.
pub fn quantize(img: &Image, colors: usize, dither: f32) -> Image {
    let colors = colors.clamp(2, 256);
    let src = img.to_depth8();
    let npx = src.width as usize * src.height as usize;
    // Packed RGBA once; everything below reads this instead of re-decoding the source layout.
    let px: Vec<[u8; 4]> = (0..npx).into_par_iter().map(|i| rgba_at(&src, i)).collect();
    if has_at_most_colors(&px, colors) {
        return src;
    }
    let has_alpha = px.iter().any(|p| p[3] != 255);

    let mut entries = exact_histogram(&px).unwrap_or_else(|| bucket_histogram(&px, has_alpha));

    let cols = median_cut(&mut entries, colors);
    let cols = refine(&entries, cols, 6);
    // Round the palette to what will actually be stored.
    let cols: Vec<[f32; 4]> = cols.iter().map(|c| c.map(|v| v.round().clamp(0.0, 255.0))).collect();
    let mut cols = cols;
    cols.dedup();
    let pal = Palette::new(cols);
    let transparent =
        if has_alpha { (0..pal.cols.len()).min_by(|&a, &b| pal.cols[a][3].total_cmp(&pal.cols[b][3])) } else { None };

    // Remap with serpentine Floyd-Steinberg error diffusion.
    let (w, h) = (src.width as usize, src.height as usize);
    let out_ch = if has_alpha { 4 } else { 3 };
    let mut out = vec![0u8; npx * out_ch];
    let mut cur = vec![[0f32; 4]; w + 2];
    let mut next = vec![[0f32; 4]; w + 2];
    let k = dither.clamp(0.0, 1.0);
    for y in 0..h {
        next.iter_mut().for_each(|e| *e = [0.0; 4]);
        let ltr = y % 2 == 0;
        for step in 0..w {
            let x = if ltr { step } else { w - 1 - step };
            let p = px[y * w + x];
            // Pixels on strong edges (text, borders) are mapped exactly: error diffusion would only
            // add noise there. Smooth regions (gradients, photos) get dithered.
            let edge = k > 0.0 && {
                let f = |q: [u8; 4]| [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32];
                let me = f(p);
                let left = if x > 0 { dist2(&me, &f(px[y * w + x - 1])) } else { 0.0 };
                let up = if y > 0 { dist2(&me, &f(px[(y - 1) * w + x])) } else { 0.0 };
                left.max(up) > EDGE_DIST2
            };
            let idx = match transparent {
                Some(t) if p[3] == 0 => t,
                _ => {
                    let mut v = [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32];
                    if k > 0.0 && !edge {
                        for i in 0..4 {
                            v[i] = (v[i] + cur[x + 1][i]).clamp(0.0, 255.0);
                        }
                    }
                    let idx = pal.nearest(&v);
                    if k > 0.0 && !edge {
                        let q = pal.cols[idx];
                        let mut err = [0f32; 4];
                        for i in 0..4 {
                            err[i] = (v[i] - q[i]).clamp(-40.0, 40.0) * k;
                        }
                        let (fwd, back) = if ltr { (x + 2, x) } else { (x, x + 2) };
                        for i in 0..4 {
                            cur[fwd][i] += err[i] * (7.0 / 16.0);
                            next[back][i] += err[i] * (3.0 / 16.0);
                            next[x + 1][i] += err[i] * (5.0 / 16.0);
                            next[fwd][i] += err[i] * (1.0 / 16.0);
                        }
                    }
                    idx
                }
            };
            let q = pal.cols[idx];
            let o = (y * w + x) * out_ch;
            out[o] = q[0] as u8;
            out[o + 1] = q[1] as u8;
            out[o + 2] = q[2] as u8;
            if has_alpha {
                out[o + 3] = q[3] as u8;
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    let color = if has_alpha { ColorType::Rgba } else { ColorType::Rgb };
    Image { width: src.width, height: src.height, color, depth: 8, data: out, meta: src.meta.clone() }
}
