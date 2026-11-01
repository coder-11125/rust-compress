//! Hierarchical JPEG (T.81 Annex J): an image coded as a sequence of frames. The first frame of a component is an
//! ordinary one (DCT or lossless); every later frame is *differential* and codes the difference between the source
//! and the previous reconstruction, upsampled by a factor of two horizontally and/or vertically as the `EXP` marker
//! says. Decoding keeps one reference plane per component and updates it frame by frame.
//!
//! This module holds the state and the pure parts (planes and the upsampling filter); the frame decoding itself is in
//! [`super::decode`].

/// One component's samples during reconstruction. Values are `i32` so differences and overshoot are representable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Plane {
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) data: Vec<i32>,
}

/// A component of the completed image, from the `DHP` marker.
#[derive(Clone, Debug)]
pub(super) struct HierComp {
    pub(super) id: u8,
    pub(super) h: usize,
    pub(super) v: usize,
}

/// State of a hierarchical decode: the size of the completed image, its components and the reference planes.
pub(super) struct Hier {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) precision: u8,
    pub(super) comps: Vec<HierComp>,
    /// The reconstruction so far, per component (`None` until a non-differential frame has supplied it).
    pub(super) planes: Vec<Option<Plane>>,
    /// Expansion requested by the `EXP` marker for the next frame: (horizontal, vertical).
    pub(super) expand: (bool, bool),
    /// Frames fully decoded.
    pub(super) frames: usize,
}

/// Upper bound on frames: a pyramid of 2x steps never needs more, and it bounds the work a hostile file can ask for.
pub(super) const MAX_FRAMES: usize = 32;

/// Upsample by two (T.81 J.1.1.2): bilinear interpolation with truncation, the first row and column kept and the last
/// ones replicated. A sample halfway between two neighbours is their average, `(Ra + Rb) / 2`; one in the middle of
/// four is `(Ra + Rb + Rc + Rd) / 4`, truncated once. T.81 words the two-dimensional case as "horizontal expansion,
/// then vertical", which read literally truncates twice and can differ by one; this follows the single truncation of
/// the JPEG committee's reference software (Richter's libjpeg), which was checked sample for sample.
pub(super) fn expand(p: &Plane, horizontal: bool, vertical: bool) -> Plane {
    let (w, h) = (p.w, p.h);
    let (nw, nh) = (if horizontal { 2 * w } else { w }, if vertical { 2 * h } else { h });
    let at = |x: usize, y: usize| p.data[y.min(h - 1) * w + x.min(w - 1)];
    let mut data = Vec::with_capacity(nw * nh);
    for oy in 0..nh {
        let (y, odd_y) = if vertical { (oy / 2, oy % 2 == 1) } else { (oy, false) };
        for ox in 0..nw {
            let (x, odd_x) = if horizontal { (ox / 2, ox % 2 == 1) } else { (ox, false) };
            data.push(match (odd_x, odd_y) {
                (false, false) => at(x, y),
                (true, false) => (at(x, y) + at(x + 1, y)) / 2,
                (false, true) => (at(x, y) + at(x, y + 1)) / 2,
                (true, true) => (at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1)) / 4,
            });
        }
    }
    Plane { w: nw, h: nh, data }
}

// ---------------------------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------------------------

use super::ZIGZAG;
use super::arith::{ArithOptions, write_arithmetic_with};
use super::decode::parse;
use super::encode::{
    CoefComponent, CoefImage, EncodeOptions, Subsampling, build_coefs, segment, write_jpeg, write_jpeg_with,
};
use super::lossless::{LosslessComp, LosslessFrame};
use super::progressive;
use super::quality::{chroma_table, luma_table};
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entropy {
    Huffman,
    Arithmetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// DCT-coded frames at `quality`; `lossless_last` ends with a differential lossless frame that makes the
    /// reconstruction exact.
    Dct { quality: u8, progressive: bool, lossless_last: bool },
    /// Every frame lossless: the smallest picture first, then exact differential frames. 8 or 16 bits per sample.
    Lossless,
}

/// How the resolution grows from one frame to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    /// Width and height double (`EXP` 1,1).
    #[default]
    Both,
    /// Only the width doubles (`EXP` 1,0).
    Horizontal,
    /// Only the height doubles (`EXP` 0,1).
    Vertical,
    /// Every frame has the full size and refines the one before (no `EXP`); DCT kinds quantize more finely each time.
    Refine,
}

#[derive(Clone, Copy, Debug)]
pub struct HierOptions {
    /// Number of frames: the image is coded at `levels` resolutions, each twice the size of the one before.
    pub levels: usize,
    pub entropy: Entropy,
    pub kind: Kind,
    /// DCT kinds: chroma at half resolution (Y sampled 2x2) instead of full.
    pub subsample_chroma: bool,
    pub shape: Shape,
}

/// Encode `img` as a hierarchical JPEG: a pyramid of `levels` resolutions whose first frame is non-differential and
/// whose others code the difference to the reconstruction so far. Made to exercise the decoder and to interchange;
/// it is not tuned for size.
pub fn write_hierarchical(img: &Image, opts: &HierOptions) -> Result<Vec<u8>> {
    if !(1..=6).contains(&opts.levels) {
        return Err(Error::invalid("hierarchical levels must be 1 to 6"));
    }
    let lossless = opts.kind == Kind::Lossless;
    // All frames of an image use the same entropy coding, lossless ones included.
    let cond = super::arith::Conditioning::default();
    let arith = (opts.entropy == Entropy::Arithmetic).then_some(&cond);
    let (sof_lossless, sof_diff_lossless) = if arith.is_some() { (0xCB, 0xCF) } else { (0xC3, 0xC7) };
    let src = if lossless && img.depth == 16 {
        match img.color {
            ColorType::Gray | ColorType::Rgb => img.clone(),
            _ => return Err(Error::unsupported("16-bit lossless hierarchical needs gray or RGB")),
        }
    } else {
        img.flatten([255, 255, 255])
    };
    let precision: u8 = if src.depth == 16 { 16 } else { 8 };
    let (w, h) = (src.width as usize, src.height as usize);
    let channels = src.color.channels();
    // Channel planes at full resolution, then the pyramid of halved ones.
    let mut full: Vec<Plane> = (0..channels).map(|_| Plane { w, h, data: Vec::with_capacity(w * h) }).collect();
    let bps = src.bytes_per_sample();
    for px in src.data.chunks_exact(channels * bps) {
        for (c, plane) in full.iter_mut().enumerate() {
            plane.data.push(if bps == 2 {
                u16::from_be_bytes([px[2 * c], px[2 * c + 1]]) as i32
            } else {
                px[c] as i32
            });
        }
    }
    let (grow_x, grow_y) = match opts.shape {
        Shape::Both => (true, true),
        Shape::Horizontal => (true, false),
        Shape::Vertical => (false, true),
        Shape::Refine => (false, false),
    };
    let mut pyramid = vec![full];
    for _ in 1..opts.levels {
        let next = pyramid.last().unwrap().iter().map(|p| shrink(p, grow_x, grow_y)).collect();
        pyramid.push(next);
    }
    pyramid.reverse(); // smallest first

    // Component layout of the completed image.
    let subsample = opts.subsample_chroma && channels == 3 && !lossless && opts.shape == Shape::Both;
    let (ids, sampling): (Vec<u8>, Vec<(usize, usize)>) = match (channels, lossless) {
        (1, _) => (vec![1], vec![(1, 1)]),
        (_, true) => (vec![b'R', b'G', b'B'], vec![(1, 1); 3]),
        (_, false) if subsample => (vec![1, 2, 3], vec![(2, 2), (1, 1), (1, 1)]),
        _ => (vec![1, 2, 3], vec![(1, 1); 3]),
    };
    let mut out = vec![0xFF, 0xD8];
    let mut dhp = vec![precision];
    dhp.extend_from_slice(&(h as u16).to_be_bytes());
    dhp.extend_from_slice(&(w as u16).to_be_bytes());
    dhp.push(channels as u8);
    for (id, (hs, vs)) in ids.iter().zip(&sampling) {
        dhp.extend_from_slice(&[*id, (*hs << 4 | *vs) as u8, 0]);
    }
    segment(&mut out, 0xDE, &dhp);

    for (level, planes) in pyramid.iter().enumerate() {
        let (lw, lh) = (planes[0].w, planes[0].h);
        // The component planes this frame has to reproduce, at the frame's own resolution.
        let sources: Vec<Plane> = if lossless || channels == 1 {
            planes.clone()
        } else {
            let mut ycc = rgb_to_ycc(planes);
            if subsample {
                for p in ycc.iter_mut().skip(1) {
                    *p = shrink(p, true, true);
                }
            }
            ycc
        };
        if level == 0 {
            match opts.kind {
                Kind::Lossless => {
                    let frame = lossless_frame(&sources, &ids, &sampling, lw, lh, precision, false);
                    out.extend_from_slice(&frame.frame_segments(1, sof_lossless, 0, arith)?);
                }
                Kind::Dct { quality, progressive, .. } => {
                    let image = to_image(planes, lw, lh)?;
                    let eopts = EncodeOptions {
                        quality,
                        subsampling: if subsample { Subsampling::S420 } else { Subsampling::S444 },
                        trellis: false,
                        ..EncodeOptions::default()
                    };
                    let ci = build_coefs(&image, &eopts);
                    let file = match (opts.entropy, progressive) {
                        (Entropy::Huffman, false) => write_jpeg(&ci),
                        (Entropy::Huffman, true) => progressive::write_progressive(&ci),
                        (Entropy::Arithmetic, p) => {
                            write_arithmetic_with(&ci, &ArithOptions { progressive: p, ..ArithOptions::default() })
                        }
                    };
                    out.extend_from_slice(frame_part(&file)?);
                }
            }
            continue;
        }
        // Differential frame against what a decoder holds after the frames so far.
        let mut probe = out.clone();
        probe.extend_from_slice(&[0xFF, 0xD9]);
        let reference = parse(&probe)?.hier_planes()?;
        let diffs: Vec<Plane> = sources
            .iter()
            .zip(&reference)
            .map(|(s, r)| {
                let up = expand(r, grow_x, grow_y);
                let mut data = Vec::with_capacity(s.w * s.h);
                for y in 0..s.h {
                    for x in 0..s.w {
                        data.push(s.data[y * s.w + x] - up.data[y * up.w + x]);
                    }
                }
                Plane { w: s.w, h: s.h, data }
            })
            .collect();
        if grow_x || grow_y {
            segment(&mut out, 0xDF, &[(grow_x as u8) << 4 | grow_y as u8]);
        }
        let last = level + 1 == pyramid.len();
        match opts.kind {
            Kind::Lossless => {
                let frame = lossless_frame(&diffs, &ids, &sampling, lw, lh, precision, true);
                out.extend_from_slice(&frame.frame_segments(0, sof_diff_lossless, 0, arith)?);
            }
            Kind::Dct { lossless_last: true, .. } if last => {
                let frame = lossless_frame(&diffs, &ids, &sampling, lw, lh, precision, true);
                out.extend_from_slice(&frame.frame_segments(0, sof_diff_lossless, 0, arith)?);
            }
            Kind::Dct { quality, progressive, .. } => {
                // A refinement of an image that is already at full size gets finer quantization each time.
                let quality =
                    if opts.shape == Shape::Refine { (quality as usize + 12 * level).min(98) as u8 } else { quality };
                let ci = differential_coefs(&diffs, &ids, &sampling, lw, lh, quality);
                let file = match (opts.entropy, progressive) {
                    (Entropy::Huffman, false) => write_jpeg_with(&ci, true),
                    (Entropy::Huffman, true) => progressive::emit_with(&ci, &progressive::plan(&ci).1, true),
                    (Entropy::Arithmetic, p) => write_arithmetic_with(
                        &ci,
                        &ArithOptions { progressive: p, differential: true, ..ArithOptions::default() },
                    ),
                };
                out.extend_from_slice(frame_part(&file)?);
            }
        }
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(out)
}

/// Halve the width and/or height with a box average and rounding; edges average what exists.
fn shrink(p: &Plane, horizontal: bool, vertical: bool) -> Plane {
    let (fx, fy) = (if horizontal { 2 } else { 1 }, if vertical { 2 } else { 1 });
    let (nw, nh) = (p.w.div_ceil(fx), p.h.div_ceil(fy));
    let mut data = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            let (mut sum, mut n) = (0i64, 0i64);
            for dy in 0..fy {
                for dx in 0..fx {
                    let (sx, sy) = (fx * x + dx, fy * y + dy);
                    if sx < p.w && sy < p.h {
                        sum += p.data[sy * p.w + sx] as i64;
                        n += 1;
                    }
                }
            }
            data.push(((sum + n / 2) / n) as i32);
        }
    }
    Plane { w: nw, h: nh, data }
}

/// BT.601 YCbCr of 8-bit RGB planes.
fn rgb_to_ycc(p: &[Plane]) -> Vec<Plane> {
    let mut out: Vec<Plane> = (0..3).map(|_| Plane { w: p[0].w, h: p[0].h, data: Vec::new() }).collect();
    for i in 0..p[0].data.len() {
        let (r, g, b) = (p[0].data[i] as f32, p[1].data[i] as f32, p[2].data[i] as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        for (plane, v) in out.iter_mut().zip([y, cb, cr]) {
            plane.data.push(v.round().clamp(0.0, 255.0) as i32);
        }
    }
    out
}

fn to_image(planes: &[Plane], w: usize, h: usize) -> Result<Image> {
    let mut data = Vec::with_capacity(w * h * planes.len());
    for i in 0..w * h {
        for p in planes {
            data.push(p.data[i] as u8);
        }
    }
    Image::new(w as u32, h as u32, if planes.len() == 1 { ColorType::Gray } else { ColorType::Rgb }, 8, data)
}

/// The frame-level part of a whole-file JPEG: from the first quantization table to just before the end marker.
fn frame_part(file: &[u8]) -> Result<&[u8]> {
    let mut pos = 2;
    while pos + 4 <= file.len() && file[pos] == 0xFF {
        if file[pos + 1] == 0xDB {
            return Ok(&file[pos..file.len() - 2]);
        }
        pos += 2 + u16::from_be_bytes([file[pos + 2], file[pos + 3]]) as usize;
    }
    Err(Error::invalid("no quantization table in the frame"))
}

/// Component samples (or differences, as 16-bit two's complement) as a lossless frame, padded to whole MCUs.
fn lossless_frame(
    planes: &[Plane],
    ids: &[u8],
    sampling: &[(usize, usize)],
    w: usize,
    h: usize,
    precision: u8,
    differential: bool,
) -> LosslessFrame {
    let hmax = sampling.iter().map(|s| s.0).max().unwrap_or(1);
    let vmax = sampling.iter().map(|s| s.1).max().unwrap_or(1);
    let (mcus_x, mcus_y) = (w.div_ceil(hmax), h.div_ceil(vmax));
    let comps = planes
        .iter()
        .zip(ids.iter().zip(sampling))
        .map(|(p, (&id, &(hs, vs)))| {
            let (stride, rows) = (mcus_x * hs, mcus_y * vs);
            let mut samples = Vec::with_capacity(stride * rows);
            for y in 0..rows {
                for x in 0..stride {
                    let v = p.data[y.min(p.h - 1) * p.w + x.min(p.w - 1)];
                    samples.push(if differential { v as i16 as u16 } else { v as u16 });
                }
            }
            LosslessComp { id, h: hs, v: vs, stride, samples }
        })
        .collect();
    LosslessFrame {
        width: w,
        height: h,
        precision,
        point_transform: 0,
        adobe_transform: None,
        meta: Default::default(),
        comps,
    }
}

/// Quantized DCT coefficients of difference planes: the transform without a level shift.
fn differential_coefs(
    planes: &[Plane],
    ids: &[u8],
    sampling: &[(usize, usize)],
    w: usize,
    h: usize,
    quality: u8,
) -> CoefImage {
    let hmax = sampling.iter().map(|s| s.0).max().unwrap_or(1);
    let vmax = sampling.iter().map(|s| s.1).max().unwrap_or(1);
    let (mcus_x, mcus_y) = (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax));
    let tables = [luma_table(quality), chroma_table(quality)];
    let comps = planes
        .iter()
        .zip(ids.iter().zip(sampling))
        .enumerate()
        .map(|(i, (p, (&id, &(hs, vs))))| {
            let (bw, bh) = (mcus_x * hs, mcus_y * vs);
            let q = &tables[i.min(1)];
            let mut coefs = vec![0i16; bw * bh * 64];
            for by in 0..bh {
                for bx in 0..bw {
                    let mut block = [0f32; 64];
                    for y in 0..8 {
                        for x in 0..8 {
                            let (sx, sy) = ((bx * 8 + x).min(p.w - 1), (by * 8 + y).min(p.h - 1));
                            block[y * 8 + x] = p.data[sy * p.w + sx] as f32;
                        }
                    }
                    let out = &mut coefs[(by * bw + bx) * 64..][..64];
                    let dct = naive_dct(&block);
                    for k in 0..64 {
                        let n = ZIGZAG[k];
                        out[k] = (dct[n] / q[n] as f32).round().clamp(-32767.0, 32767.0) as i16;
                    }
                }
            }
            CoefComponent { id, h: hs, v: vs, tq: i.min(1), blocks_w: bw, coefs }
        })
        .collect();
    CoefImage {
        width: w,
        height: h,
        comps,
        qtables: [Some(tables[0]), Some(tables[1]), None, None],
        meta: Default::default(),
        adobe_transform: None,
    }
}

/// The forward DCT of T.81 A.3.3, straight from the definition.
fn naive_dct(b: &[f32; 64]) -> [f32; 64] {
    let c = |k: usize| if k == 0 { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 };
    let mut out = [0f32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0f32;
            for y in 0..8 {
                for x in 0..8 {
                    sum += b[y * 8 + x]
                        * (((2 * x + 1) * u) as f32 * std::f32::consts::PI / 16.0).cos()
                        * (((2 * y + 1) * v) as f32 * std::f32::consts::PI / 16.0).cos();
                }
            }
            out[v * 8 + u] = 0.25 * c(u) * c(v) * sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> i32) -> Plane {
        Plane { w, h, data: (0..w * h).map(|i| f(i % w, i / w)).collect() }
    }

    /// The rule written out one output sample at a time, from the definition: neighbours at the sample positions
    /// below and to the right (clamped at the edges), averaged and truncated once.
    fn spec_sample(p: &Plane, x: usize, y: usize, horizontal: bool, vertical: bool) -> i32 {
        let at = |xx: usize, yy: usize| p.data[yy.min(p.h - 1) * p.w + xx.min(p.w - 1)];
        let (sx, sy) = (if horizontal { x / 2 } else { x }, if vertical { y / 2 } else { y });
        let between_x = horizontal && x % 2 == 1;
        let between_y = vertical && y % 2 == 1;
        let mut sum = at(sx, sy);
        let mut n = 1;
        if between_x {
            sum += at(sx + 1, sy);
            n += 1;
        }
        if between_y {
            sum += at(sx, sy + 1);
            n += 1;
        }
        if between_x && between_y {
            sum += at(sx + 1, sy + 1);
            n += 1;
        }
        sum / n
    }

    #[test]
    fn upsampling_follows_the_standard_for_every_direction() {
        let src = plane(7, 5, |x, y| ((x * 37 + y * 91 + x * y * 5) % 251) as i32);
        for (h, v) in [(true, false), (false, true), (true, true), (false, false)] {
            let up = expand(&src, h, v);
            assert_eq!((up.w, up.h), (src.w * (1 + h as usize), src.h * (1 + v as usize)));
            for y in 0..up.h {
                for x in 0..up.w {
                    assert_eq!(up.data[y * up.w + x], spec_sample(&src, x, y, h, v), "h={h} v={v} at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn the_middle_of_four_is_truncated_once() {
        // Averaging in two truncating steps would give ((0+1)/2 + (1+2)/2) / 2 = 0; the reference software, and this
        // function, give (0+1+1+2) / 4 = 1.
        let up = expand(&Plane { w: 2, h: 2, data: vec![0, 1, 1, 2] }, true, true);
        assert_eq!(up.data[up.w + 1], 1);
        assert_eq!(up.data[..up.w], [0, 0, 1, 1]);
    }

    #[test]
    fn upsampling_truncates_rather_than_rounds_and_keeps_the_edges() {
        // 0 and 3 average to 1 (not 2); the first column is kept and the last one repeated.
        let up = expand(&Plane { w: 2, h: 1, data: vec![0, 3] }, true, false);
        assert_eq!(up.data, vec![0, 1, 3, 3]);
        let up = expand(&Plane { w: 1, h: 2, data: vec![10, 21] }, false, true);
        assert_eq!(up.data, vec![10, 15, 21, 21]);
        // Truncation is toward zero, as integer division is, also for the negative values a difference can have.
        let up = expand(&Plane { w: 2, h: 1, data: vec![-3, 0] }, true, false);
        assert_eq!(up.data, vec![-3, -1, 0, 0]);
    }
}
