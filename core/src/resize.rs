//! Downscaling with a Lanczos-3 filter.
//!
//! Resizing only ever shrinks: a target larger than the source leaves the image alone. Images with
//! alpha are resampled with premultiplied color, so invisible pixels cannot bleed into visible ones.
//! Filtering is done on the stored (gamma-encoded) values, as is conventional for image tools.

use crate::image::Image;
use rayon::prelude::*;

/// How large the result may be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spec {
    /// Fit inside a box; a missing side is unconstrained. Aspect ratio is kept.
    Fit { max_w: Option<u32>, max_h: Option<u32> },
    /// Scale both sides by this factor (0 < f < 1).
    Scale(f64),
}

/// Parse `WxH`, `Wx`, `xH` (pixels) or `P%` (1-99).
pub fn parse(s: &str) -> Result<Spec, String> {
    let s = s.trim();
    let err = || "expected WIDTHxHEIGHT, WIDTHx, xHEIGHT or a percentage such as 50%".to_string();
    if let Some(p) = s.strip_suffix('%') {
        let p: f64 = p.parse().map_err(|_| err())?;
        return if p > 0.0 && p < 100.0 {
            Ok(Spec::Scale(p / 100.0))
        } else {
            Err("percentage must be between 0 and 100 (images are never enlarged)".into())
        };
    }
    let (w, h) = s.split_once(['x', 'X']).ok_or_else(err)?;
    let side = |t: &str| -> Result<Option<u32>, String> {
        if t.is_empty() {
            return Ok(None);
        }
        match t.parse::<u32>() {
            Ok(v) if v > 0 => Ok(Some(v)),
            _ => Err(err()),
        }
    };
    let (max_w, max_h) = (side(w)?, side(h)?);
    if max_w.is_none() && max_h.is_none() {
        return Err(err());
    }
    Ok(Spec::Fit { max_w, max_h })
}

/// Output size for a `w` x `h` source: never larger than the source, aspect ratio kept, at least 1x1.
pub fn target_size(spec: Spec, w: u32, h: u32) -> (u32, u32) {
    let scale = match spec {
        Spec::Scale(f) => f,
        Spec::Fit { max_w, max_h } => {
            let sw = max_w.map_or(f64::INFINITY, |m| m as f64 / w as f64);
            let sh = max_h.map_or(f64::INFINITY, |m| m as f64 / h as f64);
            sw.min(sh)
        }
    };
    if scale >= 1.0 {
        return (w, h);
    }
    let round = |v: u32| ((v as f64 * scale).round() as u32).max(1);
    (round(w).min(w), round(h).min(h))
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn lanczos3(x: f64) -> f64 {
    if x.abs() >= 3.0 { 0.0 } else { sinc(x) * sinc(x / 3.0) }
}

/// For each output index: the first contributing input index and its normalized weights.
fn contributions(src_len: usize, dst_len: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = src_len as f64 / dst_len as f64;
    // Widen the kernel when shrinking so it low-passes instead of aliasing.
    let stretch = ratio.max(1.0);
    let support = 3.0 * stretch;
    (0..dst_len)
        .map(|i| {
            let center = (i as f64 + 0.5) * ratio;
            let lo = ((center - support).floor().max(0.0)) as usize;
            let hi = (((center + support).ceil()) as usize).min(src_len);
            let mut w: Vec<f64> = (lo..hi).map(|j| lanczos3((j as f64 + 0.5 - center) / stretch)).collect();
            let sum: f64 = w.iter().sum();
            if sum.abs() > 1e-12 {
                w.iter_mut().for_each(|v| *v /= sum);
            }
            (lo, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

/// Resample `img` to `new_w` x `new_h`.
pub fn resize(img: &Image, new_w: u32, new_h: u32) -> Image {
    let (w, h) = (img.width as usize, img.height as usize);
    let (nw, nh) = (new_w as usize, new_h as usize);
    if (w, h) == (nw, nh) || nw == 0 || nh == 0 {
        return img.clone();
    }
    let ch = img.color.channels();
    let max = if img.depth == 16 { 65535.0f32 } else { 255.0 };
    let alpha = img.color.has_alpha().then_some(ch - 1);

    // Samples as f32, premultiplied when there is alpha.
    let mut src = vec![0f32; w * h * ch];
    src.par_chunks_mut(w * ch).enumerate().for_each(|(y, row)| {
        let bps = img.bytes_per_sample();
        let line = &img.data[y * img.stride()..(y + 1) * img.stride()];
        for (i, v) in row.iter_mut().enumerate() {
            *v = if bps == 2 { u16::from_be_bytes([line[2 * i], line[2 * i + 1]]) as f32 } else { line[i] as f32 };
        }
        if let Some(a) = alpha {
            for px in row.chunks_exact_mut(ch) {
                let k = px[a] / max;
                for c in &mut px[..a] {
                    *c *= k;
                }
            }
        }
    });

    // Horizontal pass: w -> nw, keeping all h rows.
    let xs = contributions(w, nw);
    let mut mid = vec![0f32; nw * h * ch];
    mid.par_chunks_mut(nw * ch).enumerate().for_each(|(y, out)| {
        let row = &src[y * w * ch..(y + 1) * w * ch];
        for (x, (lo, weights)) in xs.iter().enumerate() {
            for c in 0..ch {
                let mut acc = 0f32;
                for (k, wt) in weights.iter().enumerate() {
                    acc += row[(lo + k) * ch + c] * wt;
                }
                out[x * ch + c] = acc;
            }
        }
    });
    drop(src);

    // Vertical pass: h -> nh.
    let ys = contributions(h, nh);
    let bps = img.bytes_per_sample();
    let mut data = vec![0u8; nw * nh * ch * bps];
    data.par_chunks_mut(nw * ch * bps).enumerate().for_each(|(y, out)| {
        let (lo, weights) = &ys[y];
        let mut row = vec![0f32; nw * ch];
        for (k, wt) in weights.iter().enumerate() {
            let src_row = &mid[(lo + k) * nw * ch..(lo + k + 1) * nw * ch];
            for (o, s) in row.iter_mut().zip(src_row) {
                *o += s * wt;
            }
        }
        for px in row.chunks_exact_mut(ch) {
            if let Some(a) = alpha {
                let av = px[a].clamp(0.0, max);
                let inv = if av > 0.0 { max / av } else { 0.0 };
                for c in &mut px[..a] {
                    *c = (*c * inv).clamp(0.0, max);
                }
                px[a] = av;
            }
        }
        for (i, v) in row.iter().enumerate() {
            let q = v.clamp(0.0, max).round();
            if bps == 2 {
                out[2 * i..2 * i + 2].copy_from_slice(&(q as u16).to_be_bytes());
            } else {
                out[i] = q as u8;
            }
        }
    });
    Image { width: new_w, height: new_h, data, ..img.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ColorType;

    fn solid(w: u32, h: u32, color: ColorType, depth: u8, px: &[u16]) -> Image {
        let mut data = Vec::new();
        for _ in 0..w * h {
            for &v in px {
                if depth == 16 {
                    data.extend_from_slice(&v.to_be_bytes());
                } else {
                    data.push(v as u8);
                }
            }
        }
        Image::new(w, h, color, depth, data).unwrap()
    }

    #[test]
    fn parses_specs() {
        assert_eq!(parse("800x600"), Ok(Spec::Fit { max_w: Some(800), max_h: Some(600) }));
        assert_eq!(parse("800x"), Ok(Spec::Fit { max_w: Some(800), max_h: None }));
        assert_eq!(parse("X300"), Ok(Spec::Fit { max_w: None, max_h: Some(300) }));
        assert_eq!(parse("50%"), Ok(Spec::Scale(0.5)));
        for bad in ["", "x", "0x10", "10x-1", "abc", "100%", "0%", "-5%", "12"] {
            assert!(parse(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn target_size_fits_keeps_aspect_and_never_enlarges() {
        let fit = |w, h| Spec::Fit { max_w: w, max_h: h };
        assert_eq!(target_size(fit(Some(100), Some(100)), 400, 200), (100, 50));
        assert_eq!(target_size(fit(Some(100), Some(100)), 200, 400), (50, 100));
        assert_eq!(target_size(fit(Some(100), None), 400, 200), (100, 50));
        assert_eq!(target_size(fit(None, Some(50)), 400, 200), (100, 50));
        assert_eq!(target_size(fit(Some(1000), Some(1000)), 400, 200), (400, 200), "no upscaling");
        assert_eq!(target_size(fit(Some(1), Some(1)), 1000, 10), (1, 1), "never below 1x1");
        assert_eq!(target_size(Spec::Scale(0.5), 101, 51), (51, 26));
    }

    #[test]
    fn solid_colors_stay_solid_at_both_depths() {
        let a = resize(&solid(40, 30, ColorType::Rgb, 8, &[200, 100, 50]), 13, 7);
        assert_eq!((a.width, a.height), (13, 7));
        assert!(a.data.chunks_exact(3).all(|p| p == [200, 100, 50]));
        let b = resize(&solid(40, 30, ColorType::Gray, 16, &[40000]), 10, 10);
        assert!(b.data.chunks_exact(2).all(|p| u16::from_be_bytes([p[0], p[1]]) == 40000));
    }

    #[test]
    fn same_size_is_a_no_op() {
        let img = solid(9, 9, ColorType::Rgba, 8, &[1, 2, 3, 4]);
        assert_eq!(resize(&img, 9, 9), img);
    }

    #[test]
    fn downscaled_ramp_stays_monotone_and_in_range() {
        let w = 200;
        let data: Vec<u8> = (0..w * 10).map(|i| (i % w * 255 / (w - 1)) as u8).collect();
        let img = Image::new(w as u32, 10, ColorType::Gray, 8, data).unwrap();
        let out = resize(&img, 50, 5);
        let row = &out.data[..50];
        assert!(row.windows(2).all(|p| p[0] <= p[1]), "{row:?}");
        assert!(row[0] <= 3 && row[49] >= 252);
    }

    #[test]
    fn hidden_pixels_do_not_bleed_into_visible_ones() {
        // Left half opaque red; right half fully transparent but carrying blue.
        let (w, h) = (64u32, 8u32);
        let mut data = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                data.extend_from_slice(if x < w / 2 { &[255, 0, 0, 255] } else { &[0, 0, 255, 0] });
            }
        }
        let img = Image::new(w, h, ColorType::Rgba, 8, data).unwrap();
        let out = resize(&img, 16, 2);
        for px in out.data.chunks_exact(4) {
            if px[3] > 0 {
                assert!(px[0] >= 250 && px[2] <= 5, "color bled: {px:?}");
            }
        }
        assert!(out.data.chunks_exact(4).any(|p| p[3] == 255) && out.data.chunks_exact(4).any(|p| p[3] == 0));
    }

    #[test]
    fn extreme_ratios_do_not_panic() {
        let img = solid(300, 3, ColorType::Rgb, 8, &[9, 9, 9]);
        for (w, h) in [(1, 1), (299, 1), (1, 2), (150, 3)] {
            let out = resize(&img, w, h);
            assert_eq!(out.data.len(), (w * h * 3) as usize);
        }
    }
}
