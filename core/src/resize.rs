//! Downscaling with a Lanczos-3 filter.
//!
//! Resizing only ever shrinks: a target larger than the source leaves the image alone. Images with
//! alpha are resampled with premultiplied color, so invisible pixels cannot bleed into visible ones.
//! By default the filter runs in linear light (the sRGB transfer curve is undone before filtering and
//! applied again after), so a fine black-and-white pattern averages to the brightness the eye sees
//! instead of coming out too dark; [`Space::Encoded`] filters the stored values directly instead.

use crate::image::Image;
use rayon::prelude::*;

/// What the filter averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Space {
    /// Linear light, assuming the sRGB transfer curve (the default).
    #[default]
    Linear,
    /// The stored, gamma-encoded values as they are.
    Encoded,
}

/// sRGB decode: encoded 0..=1 to linear light.
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// sRGB encode: linear light 0..=1 to the encoded value.
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

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

/// Resample `img` to `new_w` x `new_h` in linear light.
pub fn resize(img: &Image, new_w: u32, new_h: u32) -> Image {
    resize_in(img, new_w, new_h, Space::Linear)
}

/// Resample `img` to `new_w` x `new_h`, averaging in the given [`Space`].
pub fn resize_in(img: &Image, new_w: u32, new_h: u32, space: Space) -> Image {
    let (w, h) = (img.width as usize, img.height as usize);
    let (nw, nh) = (new_w as usize, new_h as usize);
    if (w, h) == (nw, nh) || nw == 0 || nh == 0 {
        return img.clone();
    }
    let ch = img.color.channels();
    let max = if img.depth == 16 { 65535.0f32 } else { 255.0 };
    let alpha = img.color.has_alpha().then_some(ch - 1);
    let linear = space == Space::Linear;
    let bps = img.bytes_per_sample();

    // Working values are 0..=max: linear light when `linear` (color channels only; alpha is coverage and stays
    // as it is), otherwise the stored values.
    let decode: Vec<f32> =
        if !linear { Vec::new() } else { (0..=max as usize).map(|v| srgb_to_linear(v as f32 / max) * max).collect() };

    // Samples as f32, premultiplied when there is alpha.
    let mut src = vec![0f32; w * h * ch];
    src.par_chunks_mut(w * ch).enumerate().for_each(|(y, row)| {
        let line = &img.data[y * img.stride()..(y + 1) * img.stride()];
        for (i, v) in row.iter_mut().enumerate() {
            let q =
                if bps == 2 { u16::from_be_bytes([line[2 * i], line[2 * i + 1]]) as usize } else { line[i] as usize };
            *v = if linear && alpha != Some(i % ch) { decode[q] } else { q as f32 };
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
    // 8-bit output: the encoded value is the number of decision thresholds the linear value reaches, where
    // threshold k lies halfway between codes k and k+1 in encoded space (exactly what rounding the encoded
    // value would give, without a pow per sample).
    let thresholds: Vec<f32> = if linear && bps == 1 {
        (0..255).map(|k| srgb_to_linear((k as f32 + 0.5) / 255.0) * 255.0).collect()
    } else {
        Vec::new()
    };
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
            let v = v.clamp(0.0, max);
            let color = linear && alpha != Some(i % ch);
            let q = if !color {
                v.round() as u32
            } else if bps == 1 {
                thresholds.partition_point(|&t| t <= v) as u32
            } else {
                (linear_to_srgb(v / max) * max).round().clamp(0.0, max) as u32
            };
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

    fn checkerboard(n: u32, depth: u8) -> Image {
        let hi: u16 = if depth == 16 { 65535 } else { 255 };
        let mut data = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let v = if (x + y) % 2 == 0 { hi } else { 0 };
                if depth == 16 {
                    data.extend_from_slice(&v.to_be_bytes());
                } else {
                    data.push(v as u8);
                }
            }
        }
        Image::new(n, n, ColorType::Gray, depth, data).unwrap()
    }

    #[test]
    fn a_fine_pattern_averages_in_linear_light_by_default() {
        // Half black, half white pixels look like 50% light, which is sRGB 188, not 128.
        let img = checkerboard(64, 8);
        let lin = resize(&img, 8, 8);
        let enc = resize_in(&img, 8, 8, Space::Encoded);
        let mid = |i: &Image| i.data[3 * 8 + 3] as i32;
        assert!((mid(&lin) - 188).abs() <= 3, "linear gave {}", mid(&lin));
        assert!((mid(&enc) - 128).abs() <= 3, "encoded gave {}", mid(&enc));
        let wide = checkerboard(64, 16);
        let v = u16::from_be_bytes([
            resize(&wide, 8, 8).data[2 * (3 * 8 + 3)],
            resize(&wide, 8, 8).data[2 * (3 * 8 + 3) + 1],
        ]);
        assert!((v as i32 - 188 * 257).abs() <= 3 * 257, "16-bit linear gave {v}");
    }

    #[test]
    fn linear_light_keeps_alpha_as_coverage_and_colors_exact() {
        let img = solid(20, 20, ColorType::Rgba, 8, &[10, 120, 250, 90]);
        let out = resize(&img, 7, 7);
        assert!(out.data.chunks_exact(4).all(|p| p == [10, 120, 250, 90]), "{:?}", &out.data[..4]);
    }

    #[test]
    fn every_8_bit_value_survives_a_round_trip_through_linear_light() {
        for v in 0..=255u16 {
            let out = resize(&solid(12, 12, ColorType::Gray, 8, &[v]), 5, 5);
            assert!(out.data.iter().all(|&p| p as u16 == v), "{v} -> {}", out.data[0]);
        }
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
