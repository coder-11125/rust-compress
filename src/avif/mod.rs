//! AVIF: an AV1 still picture inside a HEIF container, written from scratch.

pub mod av1;
pub mod container;
pub mod convert;

use crate::error::{Error, Result};
use crate::image::{ColorType, Image};
use av1::frame::{Frame, Plane};
use container::{AvifFile, ImageData, ImageItem};

/// True if `data` looks like an AVIF/HEIF file (`ftyp` with an AVIF brand).
pub fn is_avif(data: &[u8]) -> bool {
    data.len() >= 12 && &data[4..8] == b"ftyp" && {
        let size = (u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize).clamp(16, data.len());
        data[8..size].chunks(4).any(|b| b == b"avif" || b == b"avis")
    }
}

/// Decode the AV1 stream of one coded item.
fn decode_coded(item: &container::CodedItem) -> Result<av1::decoder::Decoded> {
    let mut stream = item.av1c.config_obus.clone();
    stream.extend_from_slice(&item.data);
    av1::decoder::decode(&stream)
}

/// Decode an image item (single stream or grid) into planes plus the colour description from its sequence header.
fn decode_item(item: &ImageItem) -> Result<(Frame, av1::obu::ColorConfig)> {
    match &item.data {
        ImageData::Coded(c) => {
            let d = decode_coded(c)?;
            Ok((d.frame, d.seq.color))
        }
        ImageData::Grid(g) => {
            let mut canvas: Option<Frame> = None;
            let mut color = av1::obu::ColorConfig::default();
            for (i, tile) in g.tiles.iter().enumerate() {
                let d = decode_coded(tile)?;
                let (r, c) = (i / g.columns as usize, i % g.columns as usize);
                let f = d.frame;
                if canvas.is_none() {
                    let (tw, th) = (f.width, f.height);
                    let (ow, oh) = (g.output_width as usize, g.output_height as usize);
                    let (cw, ch) = (tw * g.columns as usize, th * g.rows as usize);
                    if cw < ow || ch < oh || cw > 1 << 16 || ch > 1 << 16 {
                        return Err(Error::invalid("AVIF grid size mismatch"));
                    }
                    let planes = f
                        .planes
                        .iter()
                        .enumerate()
                        .map(|(p, _)| {
                            let (sx, sy) = if p == 0 { (0, 0) } else { (f.sub_x, f.sub_y) };
                            Plane::new((cw + sx as usize) >> sx, (ch + sy as usize) >> sy)
                        })
                        .collect();
                    canvas = Some(Frame {
                        planes,
                        bit_depth: f.bit_depth,
                        sub_x: f.sub_x,
                        sub_y: f.sub_y,
                        width: ow,
                        height: oh,
                    });
                    color = d.seq.color.clone();
                }
                let canvas = canvas.as_mut().unwrap();
                if f.planes.len() != canvas.planes.len() || f.bit_depth != canvas.bit_depth || f.sub_x != canvas.sub_x {
                    return Err(Error::invalid("AVIF grid tiles differ in format"));
                }
                for (p, src) in f.planes.iter().enumerate() {
                    let (sx, sy) = if p == 0 { (0, 0) } else { (f.sub_x as usize, f.sub_y as usize) };
                    let (ox, oy) = ((c * f.width) >> sx, (r * f.height) >> sy);
                    let dst = &mut canvas.planes[p];
                    for y in 0..src.height {
                        if oy + y >= dst.height {
                            break;
                        }
                        let n = src.width.min(dst.width.saturating_sub(ox));
                        dst.row_mut(oy + y)[ox..ox + n].copy_from_slice(&src.row(y)[..n]);
                    }
                }
            }
            let mut canvas = canvas.ok_or_else(|| Error::invalid("empty AVIF grid"))?;
            // Crop the canvas planes to the output size.
            for (p, plane) in canvas.planes.iter_mut().enumerate() {
                let (sx, sy) = if p == 0 { (0, 0) } else { (canvas.sub_x as usize, canvas.sub_y as usize) };
                let (w, h) = ((canvas.width + sx) >> sx, (canvas.height + sy) >> sy);
                let mut out = Plane::new(w, h);
                for y in 0..h {
                    out.row_mut(y).copy_from_slice(&plane.row(y)[..w]);
                }
                *plane = out;
            }
            Ok((canvas, color))
        }
    }
}

/// Apply `clap`, mirror and rotation to interleaved samples.
fn apply_transforms(img: Image, item: &ImageItem) -> Image {
    let bps = img.bytes_per_pixel();
    let (mut w, mut h) = (img.width as usize, img.height as usize);
    let mut data = img.data.clone();
    if let Some(c) = item.clap() {
        let frac = |n: u32, d: u32| if d == 0 { 0.0 } else { n as f64 / d as f64 };
        let cw = frac(c.width_n, c.width_d).round() as usize;
        let ch = frac(c.height_n, c.height_d).round() as usize;
        if cw > 0 && ch > 0 && cw <= w && ch <= h {
            let hx = if c.horiz_off_d == 0 { 0.0 } else { c.horiz_off_n as f64 / c.horiz_off_d as f64 };
            let vy = if c.vert_off_d == 0 { 0.0 } else { c.vert_off_n as f64 / c.vert_off_d as f64 };
            let x0 = (((w - cw) as f64 / 2.0) + hx).round().clamp(0.0, (w - cw) as f64) as usize;
            let y0 = (((h - ch) as f64 / 2.0) + vy).round().clamp(0.0, (h - ch) as f64) as usize;
            let mut out = Vec::with_capacity(cw * ch * bps);
            for y in y0..y0 + ch {
                out.extend_from_slice(&data[(y * w + x0) * bps..(y * w + x0 + cw) * bps]);
            }
            data = out;
            w = cw;
            h = ch;
        }
    }
    let o = item.net_orientation();
    if o.mirror {
        for y in 0..h {
            for x in 0..w / 2 {
                for k in 0..bps {
                    data.swap((y * w + x) * bps + k, (y * w + (w - 1 - x)) * bps + k);
                }
            }
        }
    }
    for _ in 0..o.rotate {
        // One anticlockwise quarter turn.
        let mut out = vec![0u8; data.len()];
        for y in 0..h {
            for x in 0..w {
                let (nx, ny) = (y, w - 1 - x);
                out[(ny * h + nx) * bps..(ny * h + nx + 1) * bps]
                    .copy_from_slice(&data[(y * w + x) * bps..(y * w + x + 1) * bps]);
            }
        }
        data = out;
        std::mem::swap(&mut w, &mut h);
    }
    Image { width: w as u32, height: h as u32, data, ..img }
}

/// Decode an AVIF file.
pub fn decode(data: &[u8]) -> Result<Image> {
    let file: AvifFile = container::parse(data)?;
    let (frame, seq_color) = decode_item(&file.primary)?;
    let mut color = seq_color;
    if let Some(n) = file.primary.colour.nclx {
        // The container's CICP wins over the sequence header's when both are present.
        color.matrix_coefficients = n.matrix_coefficients as u8;
        color.color_range = n.full_range;
    }
    let mut img = convert::frame_to_image(&frame, &color)?;
    if let Some(a) = &file.alpha {
        let (af, ac) = decode_item(&a.item)?;
        if af.width != frame.width || af.height != frame.height {
            return Err(Error::invalid("alpha plane size differs from the colour image"));
        }
        img = add_alpha(img, &af, ac.color_range, a.premultiplied)?;
    }
    img.meta = file.metadata();
    Ok(apply_transforms(img, &file.primary))
}

fn add_alpha(img: Image, alpha: &Frame, full_range: bool, premultiplied: bool) -> Result<Image> {
    let (w, h) = (img.width as usize, img.height as usize);
    let bd = alpha.bit_depth as u32;
    let maxv = ((1u32 << bd) - 1) as f64;
    let (lo, span) =
        if full_range { (0.0, maxv) } else { (16.0 * (1u32 << (bd - 8)) as f64, 219.0 * (1u32 << (bd - 8)) as f64) };
    let sixteen = img.depth == 16;
    let ch = img.color.channels();
    let new_color = if ch == 1 { ColorType::GrayAlpha } else { ColorType::Rgba };
    let bps = if sixteen { 2 } else { 1 };
    let mut out = Vec::with_capacity(w * h * (ch + 1) * bps);
    let top = if sixteen { 65535.0 } else { 255.0 };
    for y in 0..h {
        for x in 0..w {
            let a = ((alpha.planes[0].at(x, y) as f64 - lo) / span).clamp(0.0, 1.0);
            for c in 0..ch {
                let i = (y * w + x) * ch + c;
                let v = if sixteen {
                    u16::from_be_bytes([img.data[2 * i], img.data[2 * i + 1]]) as f64
                } else {
                    img.data[i] as f64
                };
                let v = if premultiplied && a > 0.0 { (v / a).min(top) } else { v };
                let v = v.round();
                if sixteen {
                    out.extend_from_slice(&(v as u16).to_be_bytes());
                } else {
                    out.push(v as u8);
                }
            }
            let av = (a * top).round();
            if sixteen {
                out.extend_from_slice(&(av as u16).to_be_bytes());
            } else {
                out.push(av as u8);
            }
        }
    }
    Ok(Image { color: new_color, data: out, ..img })
}
