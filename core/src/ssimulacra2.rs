//! SSIMULACRA 2: a perceptual image-quality metric (Jon Sneyers, Cloudinary).
//!
//! The score is 100 for identical images and falls as the distorted image departs from the reference; roughly, 90
//! is visually indistinguishable at normal viewing, 70 is good, 50 is clearly degraded, and it can go well below 0.
//! It models differences in the XYB colour space of JPEG XL at six scales: per-channel structural similarity,
//! plus how much new edge energy the distortion adds (ringing, blocking) and how much of the reference's it
//! removes (blur), pooled by the arithmetic mean and the 4th-power norm, weighted by 108 trained coefficients,
//! and mapped through a polynomial and a power curve.
//!
//! This is an independent Rust implementation of that published algorithm. The trained weights and constants,
//! and the structure of the computation, are those of the reference implementation in libjxl
//! (`tools/ssimulacra2.cc`, BSD-3-Clause), taken here through the py-ssimulacra2 port (BSD-3-Clause), which is
//! also what this code is checked against (see `THIRD_PARTY_NOTICES.md`). Like that port it blurs with a truncated
//! Gaussian of sigma 1.5 (11 taps, horizontal edge reflected, vertical edge zero) where libjxl uses a recursive
//! approximation of the same filter, so scores can differ from the official tool by a small amount that has not been
//! measured here.
//!
//! The statistics are accumulated in `f64` and the image is processed in row stripes, so memory stays bounded.

use crate::error::{Error, Result};
use crate::image::{ColorType, Image};
use rayon::prelude::*;

/// Both sides must be at least this long: the metric has no scales below that.
pub const MIN_SIDE: u32 = 8;

const NUM_SCALES: usize = 6;
const C2: f64 = 0.0009;
const SIGMA: f64 = 1.5;
/// Taps on each side of the centre of the blur (11 taps in all).
const RADIUS: usize = 5;
/// Output rows computed per parallel work item (each also reads `RADIUS` rows of context on both sides).
const STRIPE: usize = 96;
/// Border rule of the blur. `true`: samples beyond the left and right edges are zero and the top and bottom edges are
/// mirrored; `false`: the opposite. The py-ssimulacra2 port this is checked against ends up with `true` (its code
/// transposes the image in the XYB conversion, which swaps the rules its comments describe), and the scores here agree
/// with it to about 1e-6, so `true` is what this implements. Which rule the official libjxl tool uses has not been
/// checked: the two differ by up to about 5 points on a 640-pixel image at low quality, and by far less on large images
/// at high quality, because only pixels near the edges are affected.
const ZERO_LEFT_RIGHT: bool = true;

// Opsin absorbance matrix and bias of the XYB colour space (libjxl `opsin_params.h`).
const M02: f64 = 0.078;
const M00: f64 = 0.30;
const M01: f64 = 1.0 - M02 - M00;
const M12: f64 = 0.078;
const M10: f64 = 0.23;
const M11: f64 = 1.0 - M12 - M10;
const M20: f64 = 0.243_422_689_245_478_2;
const M21: f64 = 0.204_767_444_244_968_2;
const M22: f64 = 1.0 - M20 - M21;
const BIAS: f64 = 0.003_793_073_255_275_449_3;

const WEIGHTS: [f64; 108] = [
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
];

/// Linear RGB planes of one image at one scale.
struct Planes {
    w: usize,
    h: usize,
    ch: [Vec<f32>; 3],
}

/// sRGB transfer function (encoded 0..1 to linear light).
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Linear-light planes of `img`: alpha is composited onto mid-grey, grey is replicated to three channels, and 16-bit
/// samples keep their precision.
fn linear_planes(img: &Image) -> Planes {
    let flat;
    let img = if img.color.has_alpha() {
        flat = img.flatten([128, 128, 128]);
        &flat
    } else {
        img
    };
    let (w, h) = (img.width as usize, img.height as usize);
    let n = w * h;
    let channels = img.color.channels();
    let mut ch = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    if img.depth == 8 {
        let lut: Vec<f32> = (0..256).map(|v| srgb_to_linear(v as f64 / 255.0) as f32).collect();
        for (i, px) in img.data.chunks_exact(channels).enumerate() {
            if img.color == ColorType::Gray {
                let l = lut[px[0] as usize];
                ch[0][i] = l;
                ch[1][i] = l;
                ch[2][i] = l;
            } else {
                ch[0][i] = lut[px[0] as usize];
                ch[1][i] = lut[px[1] as usize];
                ch[2][i] = lut[px[2] as usize];
            }
        }
    } else {
        for (i, px) in img.data.chunks_exact(channels * 2).enumerate() {
            let s = |k: usize| srgb_to_linear(u16::from_be_bytes([px[2 * k], px[2 * k + 1]]) as f64 / 65535.0) as f32;
            if img.color == ColorType::Gray {
                let l = s(0);
                ch[0][i] = l;
                ch[1][i] = l;
                ch[2][i] = l;
            } else {
                ch[0][i] = s(0);
                ch[1][i] = s(1);
                ch[2][i] = s(2);
            }
        }
    }
    Planes { w, h, ch }
}

/// Halve both dimensions by averaging 2x2 blocks in linear light (a partial block at an odd edge is averaged over the
/// pixels it has).
fn downsample(p: &Planes) -> Planes {
    let (w, h) = (p.w, p.h);
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    let ch: [Vec<f32>; 3] = std::array::from_fn(|c| {
        let src = &p.ch[c];
        let mut out = vec![0f32; ow * oh];
        out.par_chunks_mut(ow).enumerate().for_each(|(oy, row)| {
            let (y0, y1) = (oy * 2, (oy * 2 + 2).min(h));
            for (ox, o) in row.iter_mut().enumerate() {
                let (x0, x1) = (ox * 2, (ox * 2 + 2).min(w));
                let mut sum = 0f64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += src[y * w + x] as f64;
                    }
                }
                *o = (sum / ((y1 - y0) * (x1 - x0)) as f64) as f32;
            }
        });
        out
    });
    Planes { w: ow, h: oh, ch }
}

/// XYB of one linear-RGB pixel, shifted positive as SSIMULACRA 2 does.
#[inline]
fn xyb(r: f64, g: f64, b: f64) -> [f64; 3] {
    let m0 = (M00 * r + M01 * g + M02 * b + BIAS).max(0.0).cbrt() - BIAS.cbrt();
    let m1 = (M10 * r + M11 * g + M12 * b + BIAS).max(0.0).cbrt() - BIAS.cbrt();
    let m2 = (M20 * r + M21 * g + M22 * b + BIAS).max(0.0).cbrt() - BIAS.cbrt();
    let x = 0.5 * (m0 - m1);
    let y = 0.5 * (m0 + m1);
    [x * 14.0 + 0.42, y + 0.01, (m2 - y) + 0.55]
}

/// The 11 normalised taps of the Gaussian.
fn kernel() -> [f64; 2 * RADIUS + 1] {
    let mut k = [0f64; 2 * RADIUS + 1];
    let mut sum = 0.0;
    for (i, v) in k.iter_mut().enumerate() {
        let x = i as f64 - RADIUS as f64;
        *v = (-0.5 / (SIGMA * SIGMA) * x * x).exp();
        sum += *v;
    }
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Horizontal blur of one row. Samples beyond the ends follow the border rule (see [`ZERO_LEFT_RIGHT`]). The interior is
/// computed tap by tap over whole slices, using the kernel's symmetry, so the compiler can vectorize it.
fn blur_row(src: &[f64], dst: &mut [f64], k: &[f64; 2 * RADIUS + 1]) {
    let w = src.len();
    let last = w as isize - 1;
    let at = |i: isize| -> f64 {
        if ZERO_LEFT_RIGHT {
            return if i < 0 || i > last { 0.0 } else { src[i as usize] };
        }
        let j = if i < 0 {
            -i - 1
        } else if i > last {
            2 * last + 1 - i
        } else {
            i
        };
        src[j as usize]
    };
    let edge = |x: usize| -> f64 {
        let mut s = 0.0;
        for (t, kv) in k.iter().enumerate() {
            s += kv * at(x as isize + t as isize - RADIUS as isize);
        }
        s
    };
    if w > 2 * RADIUS {
        let (lo, hi) = (RADIUS, w - RADIUS);
        let centre = k[RADIUS];
        for (d, s) in dst[lo..hi].iter_mut().zip(&src[lo..hi]) {
            *d = centre * s;
        }
        for t in 1..=RADIUS {
            let kt = k[RADIUS + t];
            let (left, right) = (&src[lo - t..hi - t], &src[lo + t..hi + t]);
            for ((d, l), r) in dst[lo..hi].iter_mut().zip(left).zip(right) {
                *d += kt * (l + r);
            }
        }
        for x in (0..lo).chain(hi..w) {
            dst[x] = edge(x);
        }
    } else {
        for x in 0..w {
            dst[x] = edge(x);
        }
    }
}

/// Per-channel sums over a scale: for the SSIM error map, the artifact (added edges) map and the detail-lost map, the
/// sum of the values and of their 4th powers.
#[derive(Clone, Copy, Default)]
struct Sums {
    v: [[f64; 6]; 3],
}

impl Sums {
    fn add(&mut self, o: &Sums) {
        for c in 0..3 {
            for i in 0..6 {
                self.v[c][i] += o.v[c][i];
            }
        }
    }
}

/// Statistics of output rows `y0..y1` of one scale.
fn stripe_sums(a: &Planes, b: &Planes, y0: usize, y1: usize, k: &[f64; 2 * RADIUS + 1]) -> Sums {
    let (w, h) = (a.w, a.h);
    let lo = y0.saturating_sub(RADIUS);
    let hi = (y1 + RADIUS).min(h);
    let rows = hi - lo;
    // XYB of the rows this stripe reads, per channel.
    let mut xa: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0f64; rows * w]);
    let mut xb: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0f64; rows * w]);
    for r in 0..rows {
        let base = (lo + r) * w;
        for x in 0..w {
            let i = base + x;
            let pa = xyb(a.ch[0][i] as f64, a.ch[1][i] as f64, a.ch[2][i] as f64);
            let pb = xyb(b.ch[0][i] as f64, b.ch[1][i] as f64, b.ch[2][i] as f64);
            for c in 0..3 {
                xa[c][r * w + x] = pa[c];
                xb[c][r * w + x] = pb[c];
            }
        }
    }
    let mut sums = Sums::default();
    // Horizontally blurred rows of the five quantities, then blurred down the columns for each output row.
    let mut h5: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0f64; rows * w]);
    let mut tmp = vec![0f64; w];
    let mut acc: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0f64; w]);
    for c in 0..3 {
        for r in 0..rows {
            let ra = &xa[c][r * w..(r + 1) * w];
            let rb = &xb[c][r * w..(r + 1) * w];
            blur_row(ra, &mut h5[0][r * w..(r + 1) * w], k);
            blur_row(rb, &mut h5[1][r * w..(r + 1) * w], k);
            for (t, p) in tmp.iter_mut().zip(ra) {
                *t = p * p;
            }
            blur_row(&tmp, &mut h5[2][r * w..(r + 1) * w], k);
            for (t, q) in tmp.iter_mut().zip(rb) {
                *t = q * q;
            }
            blur_row(&tmp, &mut h5[3][r * w..(r + 1) * w], k);
            for (t, (p, q)) in tmp.iter_mut().zip(ra.iter().zip(rb)) {
                *t = p * q;
            }
            blur_row(&tmp, &mut h5[4][r * w..(r + 1) * w], k);
        }
        let s = &mut sums.v[c];
        // A row of the horizontally blurred quantity `q` at (possibly out-of-image) row `yy`: mirrored or zero.
        let row_of = |q: usize, yy: isize| -> Option<&[f64]> {
            let hh = h as isize;
            let yy = if yy < 0 || yy >= hh {
                if !ZERO_LEFT_RIGHT {
                    return None;
                }
                if yy < 0 { -yy - 1 } else { 2 * hh - 1 - yy }
            } else {
                yy
            };
            let r = yy as usize - lo;
            Some(&h5[q][r * w..(r + 1) * w])
        };
        for y in y0..y1 {
            for q in 0..5 {
                let yc = y as isize;
                let centre = row_of(q, yc).expect("the row itself is in the image");
                for (a_, r_) in acc[q].iter_mut().zip(centre) {
                    *a_ = k[RADIUS] * r_;
                }
                for t in 1..=RADIUS {
                    let kt = k[RADIUS + t];
                    match (row_of(q, yc - t as isize), row_of(q, yc + t as isize)) {
                        (Some(up), Some(down)) => {
                            for ((a_, u), d) in acc[q].iter_mut().zip(up).zip(down) {
                                *a_ += kt * (u + d);
                            }
                        }
                        (Some(one), None) | (None, Some(one)) => {
                            for (a_, r_) in acc[q].iter_mut().zip(one) {
                                *a_ += kt * r_;
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
            let ra = &xa[c][(y - lo) * w..(y - lo + 1) * w];
            let rb = &xb[c][(y - lo) * w..(y - lo + 1) * w];
            for x in 0..w {
                let (m1, m2) = (acc[0][x], acc[1][x]);
                let (s11, s22, s12) = (acc[2][x], acc[3][x], acc[4][x]);
                let num_m = 1.0 - (m1 - m2) * (m1 - m2);
                let num_s = 2.0 * (s12 - m1 * m2) + C2;
                let denom_s = (s11 - m1 * m1) + (s22 - m2 * m2) + C2;
                let d = (1.0 - num_m * num_s / denom_s).max(0.0);
                let d1 = (1.0 + (rb[x] - m2).abs()) / (1.0 + (ra[x] - m1).abs()) - 1.0;
                let art = d1.max(0.0);
                let det = (-d1).max(0.0);
                s[0] += d;
                s[1] += (d * d) * (d * d);
                s[2] += art;
                s[3] += (art * art) * (art * art);
                s[4] += det;
                s[5] += (det * det) * (det * det);
            }
        }
    }
    sums
}

/// Pooled statistics of one scale: per channel `[ssim mean, ssim 4-norm, artifact mean, artifact 4-norm, detail mean,
/// detail 4-norm]`.
fn scale_stats(a: &Planes, b: &Planes) -> [[f64; 6]; 3] {
    let k = kernel();
    let n = a.h.div_ceil(STRIPE);
    let parts: Vec<Sums> =
        (0..n).into_par_iter().map(|i| stripe_sums(a, b, i * STRIPE, ((i + 1) * STRIPE).min(a.h), &k)).collect();
    let mut total = Sums::default();
    for p in &parts {
        total.add(p);
    }
    let count = (a.w * a.h) as f64;
    let mut out = [[0f64; 6]; 3];
    for c in 0..3 {
        for m in 0..3 {
            out[c][2 * m] = total.v[c][2 * m] / count;
            out[c][2 * m + 1] = (total.v[c][2 * m + 1] / count).powf(0.25);
        }
    }
    out
}

/// The SSIMULACRA 2 score of `distorted` against `reference`: 100 for identical images, lower is worse, and it can be
/// negative for severe damage. Both must have the same size, at least [`MIN_SIDE`] pixels on each side; alpha is
/// composited onto mid-grey and grey images are treated as RGB.
pub fn ssimulacra2(reference: &Image, distorted: &Image) -> Result<f64> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(Error::invalid("images differ in size"));
    }
    if reference.width < MIN_SIDE || reference.height < MIN_SIDE {
        return Err(Error::invalid(format!("SSIMULACRA 2 needs at least {MIN_SIDE}x{MIN_SIDE} pixels")));
    }
    let mut a = linear_planes(reference);
    let mut b = linear_planes(distorted);
    let mut scales: Vec<[[f64; 6]; 3]> = Vec::with_capacity(NUM_SCALES);
    for s in 0..NUM_SCALES {
        if a.w < MIN_SIDE as usize || a.h < MIN_SIDE as usize {
            break;
        }
        scales.push(scale_stats(&a, &b));
        if s + 1 < NUM_SCALES {
            a = downsample(&a);
            b = downsample(&b);
        }
    }
    // Weighted sum in the order of the reference: channel, scale, norm, then (ssim, artifact, detail). With fewer
    // than six scales the weights shift exactly as they do in the reference.
    let mut ssim = 0.0;
    let mut i = 0;
    for c in 0..3 {
        for sc in &scales {
            for n in 0..2 {
                ssim += WEIGHTS[i] * sc[c][n].abs();
                i += 1;
                ssim += WEIGHTS[i] * sc[c][2 + n].abs();
                i += 1;
                ssim += WEIGHTS[i] * sc[c][4 + n].abs();
                i += 1;
            }
        }
    }
    ssim *= 0.956_238_261_683_484_4;
    ssim = 2.326_765_642_916_932 * ssim - 0.020_884_521_182_843_837 * ssim * ssim
        + 6.248_496_625_763_138e-5 * ssim * ssim * ssim;
    let score = if ssim > 0.0 { 100.0 - 10.0 * ssim.powf(0.627_633_646_783_138_7) } else { 100.0 };
    if score.is_finite() { Ok(score) } else { Err(Error::invalid("SSIMULACRA 2 produced a non-finite score")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in 0..=255.
    fn noise(seed: &mut u32) -> u8 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 24) as u8
    }

    /// A smooth gradient with a little texture, optionally with added noise of the given amplitude.
    fn picture(w: u32, h: u32, amp: i32, seed: u32) -> Image {
        let mut s = seed;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let base =
                    [(x * 255 / w.max(1)) as i32, (y * 255 / h.max(1)) as i32, ((x + y) * 255 / (w + h).max(1)) as i32];
                for b in base {
                    let n = if amp == 0 { 0 } else { (noise(&mut s) as i32 - 128) * amp / 128 };
                    data.push((b + n).clamp(0, 255) as u8);
                }
            }
        }
        Image::new(w, h, ColorType::Rgb, 8, data).unwrap()
    }

    #[test]
    fn identical_images_score_exactly_100() {
        let a = picture(40, 33, 0, 1);
        assert_eq!(ssimulacra2(&a, &a).unwrap(), 100.0);
        let n = picture(64, 64, 30, 2);
        assert_eq!(ssimulacra2(&n, &n).unwrap(), 100.0);
    }

    #[test]
    fn more_damage_scores_lower() {
        let r = picture(96, 80, 0, 3);
        let scores: Vec<f64> =
            [4, 16, 48, 120].iter().map(|&amp| ssimulacra2(&r, &picture(96, 80, amp, 3)).unwrap()).collect();
        assert!(scores.windows(2).all(|p| p[0] > p[1]), "{scores:?}");
        assert!(scores[0] < 100.0);
    }

    #[test]
    fn sizes_must_match_and_be_at_least_8() {
        let a = picture(32, 32, 0, 4);
        assert!(ssimulacra2(&a, &picture(32, 33, 0, 4)).is_err());
        assert!(ssimulacra2(&picture(7, 32, 0, 4), &picture(7, 32, 0, 4)).is_err());
        assert!(ssimulacra2(&picture(32, 7, 0, 4), &picture(32, 7, 0, 4)).is_err());
        assert!(ssimulacra2(&picture(8, 8, 0, 4), &picture(8, 8, 5, 4)).is_ok());
    }

    #[test]
    fn awkward_sizes_give_finite_scores() {
        for (w, h) in [(8, 8), (9, 13), (8, 1000), (1000, 8), (17, 33), (255, 9), (257, 255), (300, 11)] {
            let s = ssimulacra2(&picture(w, h, 0, 5), &picture(w, h, 25, 5)).unwrap();
            assert!(s.is_finite(), "{w}x{h}: {s}");
        }
    }

    #[test]
    fn gray_is_scored_as_replicated_rgb() {
        let (w, h) = (48u32, 40u32);
        let mut s = 7;
        let gray: Vec<u8> = (0..w * h).map(|i| ((i % w) * 5 + noise(&mut s) as u32 / 8) as u8).collect();
        let mut s2 = 8;
        let gray2: Vec<u8> = gray.iter().map(|&v| v.saturating_add(noise(&mut s2) / 16)).collect();
        let rgb = |g: &[u8]| Image::new(w, h, ColorType::Rgb, 8, g.iter().flat_map(|&v| [v, v, v]).collect()).unwrap();
        let as_gray = |g: &[u8]| Image::new(w, h, ColorType::Gray, 8, g.to_vec()).unwrap();
        assert_eq!(
            ssimulacra2(&as_gray(&gray), &as_gray(&gray2)).unwrap(),
            ssimulacra2(&rgb(&gray), &rgb(&gray2)).unwrap()
        );
    }

    #[test]
    fn sixteen_bit_that_replicates_eight_bit_scores_identically() {
        let (a, b) = (picture(50, 44, 0, 9), picture(50, 44, 30, 9));
        let up = |i: &Image| {
            let data = i.data.iter().flat_map(|&v| (v as u16 * 257).to_be_bytes()).collect();
            Image::new(i.width, i.height, i.color, 16, data).unwrap()
        };
        assert_eq!(ssimulacra2(&a, &b).unwrap(), ssimulacra2(&up(&a), &up(&b)).unwrap());
    }

    #[test]
    fn alpha_is_composited_and_invisible_colour_is_ignored() {
        let (w, h) = (40u32, 40u32);
        let rgba = |hidden: u8| {
            let mut d = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    if x < 20 {
                        d.extend([200, 100, 50, 255]);
                    } else {
                        // Fully transparent: whatever colour is stored there is invisible.
                        d.extend([hidden, hidden / 2, 255 - hidden, 0]);
                    }
                    let _ = y;
                }
            }
            Image::new(w, h, ColorType::Rgba, 8, d).unwrap()
        };
        assert_eq!(ssimulacra2(&rgba(0), &rgba(255)).unwrap(), 100.0);
        // An opaque picture equals the same picture with an all-255 alpha channel.
        let rgb = picture(w, h, 0, 11);
        let with_alpha = {
            let d = rgb.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
            Image::new(w, h, ColorType::Rgba, 8, d).unwrap()
        };
        let other = picture(w, h, 40, 11);
        assert_eq!(ssimulacra2(&rgb, &other).unwrap(), ssimulacra2(&with_alpha, &other).unwrap());
    }

    #[test]
    fn repeated_calls_agree_bit_for_bit() {
        let (a, b) = (picture(130, 97, 0, 12), picture(130, 97, 20, 12));
        assert_eq!(ssimulacra2(&a, &b).unwrap().to_bits(), ssimulacra2(&a, &b).unwrap().to_bits());
    }
}
