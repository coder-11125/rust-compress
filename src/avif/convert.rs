//! Conversion between decoded AV1 planes (YCbCr) and the crate's interleaved [`Image`].

use super::av1::frame::Frame;
use super::av1::obu::ColorConfig;
use crate::error::{Error, Result};
use crate::image::{ColorType, Image};

/// Luma coefficients (kr, kb) for a `matrix_coefficients` code; `None` for identity (GBR).
fn matrix(mc: u8) -> Result<Option<(f64, f64)>> {
    Ok(match mc {
        0 => None,
        1 => Some((0.2126, 0.0722)),
        4 => Some((0.30, 0.11)),
        5 | 6 | 2 => Some((0.299, 0.114)),
        7 => Some((0.212, 0.087)),
        9 | 10 => Some((0.2627, 0.0593)),
        _ => return Err(Error::unsupported(format!("AV1 matrix coefficients {mc}"))),
    })
}

/// Chroma sample of a plane at luma position (x, y), bilinearly interpolated for subsampled planes.
fn chroma_at(p: &super::av1::frame::Plane, x: usize, y: usize, sx: u8, sy: u8) -> u32 {
    // Fixed point with 2 fractional bits so quarter-sample centres stay exact.
    let sample = |cx: isize, cy: isize| -> u32 {
        let cx = cx.clamp(0, p.width as isize - 1) as usize;
        let cy = cy.clamp(0, p.height as isize - 1) as usize;
        p.at(cx, cy) as u32
    };
    let (fx, ix) = if sx == 1 {
        // Chroma sample i sits between luma 2i and 2i+1.
        let pos = x as isize * 2 - 1; // in quarter chroma samples: (x + 0.5) / 2 - 0.5 = (2x - 1) / 4
        (pos & 3, pos >> 2)
    } else {
        (0, x as isize)
    };
    let (fy, iy) = if sy == 1 {
        let pos = y as isize * 2 - 1;
        (pos & 3, pos >> 2)
    } else {
        (0, y as isize)
    };
    let (wx1, wy1) = (fx as u32, fy as u32);
    let (wx0, wy0) = (4 - wx1, 4 - wy1);
    let (wx0, wx1) = if sx == 1 { (wx0, wx1) } else { (4, 0) };
    let (wy0, wy1) = if sy == 1 { (wy0, wy1) } else { (4, 0) };
    let v = sample(ix, iy) * wx0 * wy0
        + sample(ix + 1, iy) * wx1 * wy0
        + sample(ix, iy + 1) * wx0 * wy1
        + sample(ix + 1, iy + 1) * wx1 * wy1;
    (v + 8) >> 4
}

/// Decoded planes to an RGB (or gray) image. Samples above 8 bits are widened to 16.
pub fn frame_to_image(frame: &Frame, color: &ColorConfig) -> Result<Image> {
    let (w, h) = (frame.width, frame.height);
    let bd = frame.bit_depth as u32;
    let maxv = ((1u32 << bd) - 1) as f64;
    let out16 = bd > 8;
    let mono = frame.planes.len() == 1;
    let coeffs = matrix(color.matrix_coefficients)?;
    let (ymin, yrange, cmid, crange) = if color.color_range {
        (0.0, maxv, (1u32 << (bd - 1)) as f64, maxv)
    } else {
        let s = (1u32 << (bd - 8)) as f64;
        (16.0 * s, 219.0 * s, 128.0 * s, 224.0 * s)
    };
    let channels = if mono { 1 } else { 3 };
    let color_type = if mono { ColorType::Gray } else { ColorType::Rgb };
    let mut data = Vec::with_capacity(w * h * channels * if out16 { 2 } else { 1 });
    let scale16 = 65535.0 / maxv;
    let mut put = |v: f64| {
        let v = v.round().clamp(0.0, maxv);
        if out16 {
            data.extend_from_slice(&((v * scale16).round() as u16).to_be_bytes());
        } else {
            data.push(v as u8);
        }
    };
    let (sx, sy) = (frame.sub_x, frame.sub_y);
    for y in 0..h {
        for x in 0..w {
            let yv = frame.planes[0].at(x, y) as f64;
            if mono {
                put((yv - ymin) * maxv / yrange);
                continue;
            }
            let cb = chroma_at(&frame.planes[1], x, y, sx, sy) as f64;
            let cr = chroma_at(&frame.planes[2], x, y, sx, sy) as f64;
            let (r, g, b);
            match coeffs {
                None => {
                    // Identity: planes are G, B, R.
                    let s = maxv / if color.color_range { maxv } else { yrange };
                    let off = if color.color_range { 0.0 } else { ymin };
                    g = (yv - off) * s;
                    b = (cb - off) * s;
                    r = (cr - off) * s;
                }
                Some((kr, kb)) => {
                    let kg = 1.0 - kr - kb;
                    let yn = (yv - ymin) / yrange;
                    let pb = (cb - cmid) / crange;
                    let pr = (cr - cmid) / crange;
                    let rn = yn + 2.0 * (1.0 - kr) * pr;
                    let bn = yn + 2.0 * (1.0 - kb) * pb;
                    let gn = (yn - kr * rn - kb * bn) / kg;
                    r = rn * maxv;
                    g = gn * maxv;
                    b = bn * maxv;
                }
            }
            put(r);
            put(g);
            put(b);
        }
    }
    Image::new(w as u32, h as u32, color_type, if out16 { 16 } else { 8 }, data)
}

/// Planes produced for encoding: 8-bit full-range BT.601 YCbCr with the given subsampling.
pub struct Yuv {
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub cw: usize,
    pub ch: usize,
}

/// RGB (8 or 16 bit, big-endian) to full-range YCbCr at `bit_depth`, averaging chroma over subsampled blocks.
pub fn rgb_to_yuv(img: &Image, bit_depth: u32, sub_x: usize, sub_y: usize, kr: f64, kb: f64) -> Yuv {
    let (w, h) = (img.width as usize, img.height as usize);
    let ch = img.color.channels();
    let maxv = ((1u32 << bit_depth) - 1) as f64;
    let src_max = if img.depth == 16 { 65535.0 } else { 255.0 };
    let get = |x: usize, y: usize, c: usize| -> f64 {
        let i = (y * w + x) * ch + c;
        let v = if img.depth == 16 {
            u16::from_be_bytes([img.data[2 * i], img.data[2 * i + 1]]) as f64
        } else {
            img.data[i] as f64
        };
        v * maxv / src_max
    };
    let kg = 1.0 - kr - kb;
    let (cw, chh) = ((w + sub_x) >> sub_x, (h + sub_y) >> sub_y);
    let mut y_plane = vec![0u16; w * h];
    let mut pb_sum = vec![0f64; cw * chh];
    let mut pr_sum = vec![0f64; cw * chh];
    let mut count = vec![0f64; cw * chh];
    let mid = (1u32 << (bit_depth - 1)) as f64;
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = if ch >= 3 {
                (get(x, y, 0), get(x, y, 1), get(x, y, 2))
            } else {
                let v = get(x, y, 0);
                (v, v, v)
            };
            let yy = kr * r + kg * g + kb * b;
            y_plane[y * w + x] = yy.round().clamp(0.0, maxv) as u16;
            let i = (y >> sub_y) * cw + (x >> sub_x);
            pb_sum[i] += (b - yy) / (2.0 * (1.0 - kb));
            pr_sum[i] += (r - yy) / (2.0 * (1.0 - kr));
            count[i] += 1.0;
        }
    }
    let conv = |sum: &[f64]| -> Vec<u16> {
        sum.iter().zip(&count).map(|(s, n)| (s / n + mid).round().clamp(0.0, maxv) as u16).collect()
    };
    Yuv { y: y_plane, u: conv(&pb_sum), v: conv(&pr_sum), cw, ch: chh }
}
