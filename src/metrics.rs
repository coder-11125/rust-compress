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
}
