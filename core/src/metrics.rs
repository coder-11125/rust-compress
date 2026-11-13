//! Image similarity metrics.

use crate::image::Image;
use rayon::prelude::*;

const C1: f64 = 6.5025; // (0.01 * 255)^2
const C2: f64 = 58.5225; // (0.03 * 255)^2
const WIN: usize = 8;
const STEP: usize = 4;

/// Mean structural similarity (SSIM) over all channels of two same-sized 8-bit images, using
/// 8x8 windows every 4 pixels. 1.0 means identical.
pub fn ssim(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height, a.color, a.depth), (b.width, b.height, b.color, b.depth));
    assert_eq!(a.depth, 8);
    let (w, h) = (a.width as usize, a.height as usize);
    let ch = a.color.channels();
    if w < WIN || h < WIN {
        // Too small for windows: fall back to the global statistics.
        return window_ssim(a, b, 0, 0, w, h, ch);
    }
    let ys: Vec<usize> = (0..=h - WIN).step_by(STEP).collect();
    let (sum, count) = ys
        .par_iter()
        .map(|&y| {
            let mut s = 0.0;
            let mut n = 0usize;
            for x in (0..=w - WIN).step_by(STEP) {
                s += window_ssim(a, b, x, y, WIN, WIN, ch);
                n += 1;
            }
            (s, n)
        })
        .reduce(|| (0.0, 0), |p, q| (p.0 + q.0, p.1 + q.1));
    sum / count as f64
}

fn window_ssim(a: &Image, b: &Image, x0: usize, y0: usize, ww: usize, wh: usize, ch: usize) -> f64 {
    let w = a.width as usize;
    let mut total = 0.0;
    for c in 0..ch {
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for y in y0..y0 + wh {
            let row = (y * w + x0) * ch + c;
            for i in 0..ww {
                let p = a.data[row + i * ch] as f64;
                let q = b.data[row + i * ch] as f64;
                sx += p;
                sy += q;
                sxx += p * p;
                syy += q * q;
                sxy += p * q;
            }
        }
        let n = (ww * wh) as f64;
        let (mx, my) = (sx / n, sy / n);
        let (vx, vy, cov) = (sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my);
        total += ((2.0 * mx * my + C1) * (2.0 * cov + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
    }
    total / ch as f64
}

/// Peak signal-to-noise ratio in dB over every sample of two same-sized 8-bit images.
/// Identical images return infinity.
pub fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height, a.color, a.depth), (b.width, b.height, b.color, b.depth));
    assert_eq!(a.depth, 8);
    let se: f64 = a.data.iter().zip(&b.data).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum();
    let mse = se / a.data.len().max(1) as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// Luminance-only 8-bit gray view used by [`ms_ssim`]. Alpha is ignored.
fn luma(img: &Image) -> Image {
    let src = img.to_depth8();
    let ch = src.color.channels();
    let data = match src.color {
        crate::image::ColorType::Gray => src.data,
        crate::image::ColorType::GrayAlpha => src.data.iter().step_by(2).copied().collect(),
        crate::image::ColorType::Rgb | crate::image::ColorType::Rgba => src
            .data
            .chunks_exact(ch)
            .map(|p| ((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32 + 500) / 1000) as u8)
            .collect(),
    };
    Image::new(src.width, src.height, crate::image::ColorType::Gray, 8, data).expect("luma keeps dimensions")
}

fn downsample2(img: &Image) -> Image {
    let (w, h) = (img.width as usize / 2, img.height as usize / 2);
    if w == 0 || h == 0 {
        return img.clone();
    }
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    sum += img.data[(2 * y + dy) * img.width as usize + 2 * x + dx] as u32;
                }
            }
            data.push((sum / 4) as u8);
        }
    }
    Image::new(w as u32, h as u32, crate::image::ColorType::Gray, 8, data).expect("downsample keeps dimensions")
}

/// Multi-scale SSIM approximation: luminance SSIM at five dyadic scales, combined with the usual
/// weights. Like [`ssim`] it is 1.0 for identical images and falls back to global statistics on
/// tiny inputs.
pub fn ms_ssim(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height, a.depth), (b.width, b.height, b.depth));
    assert_eq!(a.depth, 8);
    const W: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    let (mut x, mut y) = (luma(a), luma(b));
    let mut out = 1.0;
    for (i, w) in W.iter().enumerate() {
        let s = ssim(&x, &y).max(1e-9);
        out *= s.powf(*w);
        if i + 1 < W.len() {
            x = downsample2(&x);
            y = downsample2(&y);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ColorType;

    fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Image {
        let data = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect();
        Image::new(w, h, ColorType::Gray, 8, data).unwrap()
    }

    #[test]
    fn identical_images_score_one() {
        let a = img(40, 33, |x, y| (x * 5 + y * 3) as u8);
        assert!((ssim(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn more_distortion_scores_lower() {
        let a = img(64, 64, |x, y| ((x * 7 + y * 5) % 256) as u8);
        let slight = img(64, 64, |x, y| ((x * 7 + y * 5) % 256) as u8 / 8 * 8);
        let heavy = img(64, 64, |x, y| ((x * 7 + y * 5) % 256) as u8 / 64 * 64);
        let (s, h) = (ssim(&a, &slight), ssim(&a, &heavy));
        assert!(s > h && s < 1.0 && h > -1.0, "{s} {h}");
    }

    #[test]
    fn tiny_images_do_not_panic() {
        let a = img(3, 2, |x, y| (x + y) as u8);
        assert!(ssim(&a, &a) > 0.99);
    }

    #[test]
    fn psnr_is_infinite_for_identical_and_finite_for_noise() {
        let a = img(32, 32, |x, y| ((x * 3 + y * 5) % 256) as u8);
        assert_eq!(psnr(&a, &a), f64::INFINITY);
        let b = img(32, 32, |x, y| (((x * 3 + y * 5) % 256) as u8).wrapping_add(8));
        let p = psnr(&a, &b);
        assert!(p.is_finite() && p > 20.0 && p < 40.0, "{p}");
    }

    #[test]
    fn ms_ssim_scores_identical_highest() {
        let a = img(64, 64, |x, y| ((x * 7 + y * 11) % 256) as u8);
        let slight = img(64, 64, |x, y| (((x * 7 + y * 11) % 256) as u8) / 16 * 16);
        let (same, s) = (ms_ssim(&a, &a), ms_ssim(&a, &slight));
        assert!(same > 0.99 && s < same, "{same} {s}");
    }
}
