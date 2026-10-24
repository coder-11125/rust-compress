//! AVIF output: pixels -> AV1 still pictures (colour, plus a monochrome alpha picture) -> HEIF container.

use super::av1::encode::{EncodeParams, encode_picture};
use super::av1::frame::{Frame, Plane};
use super::av1::obu::ColorConfig;
use super::container::{self, Av1Config, Av1Stream, AvifOutput, ColourInfo, Nclx, SequenceOut};
use super::convert::rgb_to_yuv;
use crate::animation::Animation;
use crate::error::{Error, Result};
use crate::image::Image;

#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// 1-100, higher is better.
    pub quality: u8,
    /// Keep full-resolution chroma (4:4:4) instead of 4:2:0.
    pub full_chroma: bool,
    /// Search effort 0-9 (the CLI's `-l`): higher is smaller and slower.
    pub effort: u8,
    /// Coded bit depth, 8 or 10. 10-bit output keeps the precision of 16-bit sources.
    pub bit_depth: u8,
}

/// libaom's `--min-q/--max-q` scale (0-63) mapped to AV1 quantizer indices.
const QUANTIZER_TO_QINDEX: [u8; 64] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108,
    112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200,
    204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 249, 255,
];

/// Quality 1-100 to a base quantizer index. 100 is the finest lossy step (index 4), never lossless.
pub fn quality_to_qindex(quality: u8) -> u8 {
    let q = (100 - quality.clamp(1, 100)) as f64 * 63.0 / 99.0;
    QUANTIZER_TO_QINDEX[(q.round() as usize).clamp(1, 63)]
}

fn plane_from(v: &[u16], w: usize, h: usize) -> Plane {
    let mut p = Plane::new(w, h);
    p.data.copy_from_slice(v);
    p
}

fn stream_of(frame: &Frame, color: &ColorConfig, qidx: u8, effort: u8, still: bool) -> Result<Av1Stream> {
    let params = EncodeParams { base_q_idx: qidx, effort, still, ..EncodeParams::default() };
    let e = encode_picture(frame, color, &params)?;
    let level = e.seq.operating_points[0].level;
    let config = Av1Config {
        seq_profile: e.seq.profile,
        seq_level_idx_0: level,
        seq_tier_0: 0,
        high_bitdepth: color.bit_depth > 8,
        twelve_bit: color.bit_depth == 12,
        monochrome: color.mono_chrome,
        chroma_subsampling_x: color.sub_x,
        chroma_subsampling_y: color.sub_y,
        chroma_sample_position: 0,
        initial_presentation_delay_minus_one: None,
        config_obus: Vec::new(),
    };
    Ok(Av1Stream { data: e.data, config })
}

fn color_config(mono: bool, sub: u8, bit_depth: u8) -> ColorConfig {
    ColorConfig {
        bit_depth,
        mono_chrome: mono,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: if mono { 2 } else { 6 },
        color_range: true,
        sub_x: sub,
        sub_y: sub,
        chroma_sample_position: 0,
        separate_uv_delta_q: false,
    }
}

/// One channel of an image as samples scaled to `bit_depth` bits.
fn channel(img: &Image, c: usize, bit_depth: u32) -> Vec<u16> {
    let ch = img.color.channels();
    let max = (1u32 << bit_depth) - 1;
    if img.depth == 16 {
        img.data
            .chunks_exact(2 * ch)
            .map(|p| {
                let v = u16::from_be_bytes([p[2 * c], p[2 * c + 1]]) as u32;
                ((v * max + 32767) / 65535) as u16
            })
            .collect()
    } else {
        img.data.chunks_exact(ch).map(|p| ((p[c] as u32 * max + 127) / 255) as u16).collect()
    }
}

/// The picture planes of an image as AV1 input: colour (or gray) and, if it has alpha, a monochrome alpha frame.
struct Planes {
    color: Frame,
    color_cfg: ColorConfig,
    alpha: Option<Frame>,
    alpha_cfg: ColorConfig,
}

fn planes_of(src: &Image, bd: u8, full_chroma: bool) -> Planes {
    let (w, h) = (src.width as usize, src.height as usize);
    let gray = src.color.is_gray();
    let ch = src.color.channels();
    let (color, color_cfg) = if gray {
        let y = channel(src, 0, bd as u32);
        let f = Frame { planes: vec![plane_from(&y, w, h)], bit_depth: bd, sub_x: 1, sub_y: 1, width: w, height: h };
        (f, color_config(true, 1, bd))
    } else {
        let sub = if full_chroma { 0 } else { 1 };
        let yuv = rgb_to_yuv(src, bd as u32, sub as usize, sub as usize, 0.299, 0.114);
        let f = Frame {
            planes: vec![
                plane_from(&yuv.y, w, h),
                plane_from(&yuv.u, yuv.cw, yuv.ch),
                plane_from(&yuv.v, yuv.cw, yuv.ch),
            ],
            bit_depth: bd,
            sub_x: sub,
            sub_y: sub,
            width: w,
            height: h,
        };
        (f, color_config(false, sub, bd))
    };
    let alpha = src.color.has_alpha().then(|| {
        let a = channel(src, ch - 1, bd as u32);
        Frame { planes: vec![plane_from(&a, w, h)], bit_depth: bd, sub_x: 1, sub_y: 1, width: w, height: h }
    });
    Planes { color, color_cfg, alpha, alpha_cfg: color_config(true, 1, bd) }
}

/// Encode a picture's planes as AV1 streams: (colour, alpha).
fn streams_of(p: &Planes, qidx: u8, effort: u8, still: bool) -> Result<(Av1Stream, Option<Av1Stream>)> {
    let color = stream_of(&p.color, &p.color_cfg, qidx, effort, still)?;
    // Alpha edges matter more than colour detail: code it a little finer.
    let alpha = match &p.alpha {
        Some(a) => Some(stream_of(a, &p.alpha_cfg, qidx.saturating_sub(qidx / 8).max(4), effort, still)?),
        None => None,
    };
    Ok((color, alpha))
}

fn finish(out: &mut AvifOutput, img: &Image, gray: bool) {
    out.colour = ColourInfo {
        nclx: Some(Nclx { matrix_coefficients: if gray { 2 } else { 6 }, ..Nclx::SRGB }),
        icc: img.meta.icc.clone(),
    };
    out.exif = img.meta.exif.clone();
    out.xmp = img.meta.xmp.clone();
}

/// Encode an image as an AVIF file (8-bit or 10-bit 4:2:0 / 4:4:4, or monochrome).
pub fn encode(img: &Image, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let bd = opts.bit_depth.clamp(8, 10);
    let src = if bd == 8 { img.to_depth8() } else { img.clone() };
    let planes = planes_of(&src, bd, opts.full_chroma);
    let (color, alpha) = streams_of(&planes, quality_to_qindex(opts.quality), opts.effort, true)?;
    let mut out = AvifOutput::new(src.width, src.height, color);
    out.alpha = alpha;
    finish(&mut out, img, src.color.is_gray());
    Ok(container::write(&out))
}

/// Split off the leading OBU (the sequence header) of a stream whose first OBU has a size field.
fn split_first_obu(data: &[u8]) -> (&[u8], &[u8]) {
    let (mut n, mut shift, mut i) = (0usize, 0, 1);
    while let Some(&b) = data.get(i) {
        n |= ((b & 0x7f) as usize) << shift;
        i += 1;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    data.split_at_checked(i + n).unwrap_or((&[], data))
}

fn without_sequence_header(data: &[u8]) -> &[u8] {
    split_first_obu(data).1
}

/// Encode an animation as an AVIF image sequence (`avis`): every frame is a key frame, and the first is also the
/// still primary item. Frames are encoded in parallel.
pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let first = anim.frames.first().ok_or_else(|| Error::invalid("animation without frames"))?;
    let bd = opts.bit_depth.clamp(8, 10);
    let qidx = quality_to_qindex(opts.quality);
    let coded: Vec<Result<(Av1Stream, Option<Av1Stream>)>> = anim
        .frames
        .par_iter()
        .map(|f| {
            let src = if bd == 8 { f.image.to_depth8() } else { f.image.clone() };
            streams_of(&planes_of(&src, bd, opts.full_chroma), qidx, opts.effort, false)
        })
        .collect();
    let mut streams = Vec::with_capacity(coded.len());
    for c in coded {
        streams.push(c?);
    }
    let gray = first.image.color.is_gray();
    let mut it = streams.into_iter();
    let (mut c0, mut a0) = it.next().expect("at least one frame");
    // The sequence header goes into the track configuration, where decoders look for it first.
    c0.config.config_obus = split_first_obu(&c0.data).0.to_vec();
    if let Some(a) = a0.as_mut() {
        a.config.config_obus = split_first_obu(&a.data).0.to_vec();
    }
    let mut out = AvifOutput::new(anim.width, anim.height, c0);
    out.alpha = a0;
    let (mut color, mut alpha) = (Vec::new(), Vec::new());
    for (c, a) in it {
        color.push(without_sequence_header(&c.data).to_vec());
        if let Some(a) = a {
            alpha.push(without_sequence_header(&a.data).to_vec());
        }
    }
    out.sequence = Some(SequenceOut {
        timescale: 1000,
        durations: anim.frames.iter().map(|f| f.duration_ms).collect(),
        sync: vec![true; anim.frames.len()],
        color,
        alpha,
        loop_forever: anim.loop_forever,
    });
    finish(&mut out, &first.image, gray);
    Ok(container::write(&out))
}
