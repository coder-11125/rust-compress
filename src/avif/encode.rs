//! AVIF output: pixels -> AV1 still pictures (colour, plus a monochrome alpha picture) -> HEIF container.

use super::av1::encode::{EncodeParams, encode_picture};
use super::av1::frame::{Frame, Plane};
use super::av1::obu::ColorConfig;
use super::container::{self, Av1Config, Av1Stream, AvifOutput, ColourInfo, Nclx, SequenceOut};
use super::convert::rgb_to_yuv;
use crate::animation::Animation;
use crate::error::{Error, Result};
use crate::image::Image;
use std::sync::Arc;

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

fn stream_of(
    frame: &Frame,
    color: &ColorConfig,
    qidx: u8,
    effort: u8,
    still: bool,
    reference: Option<&Arc<Frame>>,
) -> Result<(Av1Stream, Arc<Frame>)> {
    let params =
        EncodeParams { base_q_idx: qidx, effort, still, reference: reference.cloned(), ..EncodeParams::default() };
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
    Ok((Av1Stream { data: e.data, config }, Arc::new(e.recon)))
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

/// What coding one picture produced: the colour and alpha streams and the pictures a decoder will reconstruct
/// (the references for the next frame).
struct Coded {
    color: Av1Stream,
    alpha: Option<Av1Stream>,
    color_recon: Arc<Frame>,
    alpha_recon: Option<Arc<Frame>>,
}

/// Encode a picture's planes as AV1 streams, as key frames or (given the previous pictures) inter frames.
fn streams_of(
    p: &Planes,
    qidx: u8,
    effort: u8,
    still: bool,
    reference: Option<(&Arc<Frame>, Option<&Arc<Frame>>)>,
) -> Result<Coded> {
    let (color, color_recon) = stream_of(&p.color, &p.color_cfg, qidx, effort, still, reference.map(|r| r.0))?;
    // Alpha edges matter more than colour detail: code it a little finer.
    let (alpha, alpha_recon) = match &p.alpha {
        Some(a) => {
            let aref = reference.and_then(|r| r.1);
            let (s, r) = stream_of(a, &p.alpha_cfg, qidx.saturating_sub(qidx / 8).max(4), effort, still, aref)?;
            (Some(s), Some(r))
        }
        None => (None, None),
    };
    Ok(Coded { color, alpha, color_recon, alpha_recon })
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
    let coded = streams_of(&planes, quality_to_qindex(opts.quality), opts.effort, true, None)?;
    let mut out = AvifOutput::new(src.width, src.height, coded.color);
    out.alpha = coded.alpha;
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

/// Encode an animation as an AVIF image sequence (`avis`): the first frame is a key frame (and also the still
/// primary item), the following ones are inter frames predicted from the previous output, except where a key
/// frame is smaller (scene changes).
pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let first = anim.frames.first().ok_or_else(|| Error::invalid("animation without frames"))?;
    let bd = opts.bit_depth.clamp(8, 10);
    let qidx = quality_to_qindex(opts.quality);
    let planes: Vec<Planes> = anim
        .frames
        .iter()
        .map(|f| {
            let src = if bd == 8 { f.image.to_depth8() } else { f.image.clone() };
            planes_of(&src, bd, opts.full_chroma)
        })
        .collect();
    let gray = first.image.color.is_gray();
    let mut prev: Option<(Arc<Frame>, Option<Arc<Frame>>)> = None;
    let mut codeds: Vec<(Coded, bool)> = Vec::with_capacity(planes.len());
    let mut key_size = 0usize;
    let mut verify = std::env::var_os("AV1_VERIFY").map(|_| super::av1::decoder::Decoder::new());
    for p in &planes {
        let size_of = |c: &Coded| c.color.data.len() + c.alpha.as_ref().map_or(0, |a| a.data.len());
        let coded = match &prev {
            _ if std::env::var_os("AV1_KEYONLY").is_some() => (streams_of(p, qidx, opts.effort, false, None)?, true),
            None => {
                let c = streams_of(p, qidx, opts.effort, false, None)?;
                key_size = size_of(&c);
                (c, true)
            }
            Some((cr, ar)) => {
                let inter = streams_of(p, qidx, opts.effort, false, Some((cr, ar.as_ref())))?;
                // An inter frame about as large as a key frame is probably a scene change: try the key frame too.
                if size_of(&inter) * 2 > key_size {
                    let key = streams_of(p, qidx, opts.effort, false, None)?;
                    if size_of(&key) < size_of(&inter) { (key, true) } else { (inter, false) }
                } else {
                    (inter, false)
                }
            }
        };
        if let Some(dec) = verify.as_mut() {
            // Debug aid (AV1_VERIFY): the decoder must reproduce the encoder's reconstruction exactly.
            let got = dec.decode_temporal_unit(&coded.0.color.data)?;
            let same = got.first().is_some_and(|d| {
                d.frame
                    .planes
                    .iter()
                    .zip(&coded.0.color_recon.planes)
                    .all(|(a, b)| (0..a.height).all(|y| a.row(y) == &b.row(y)[..a.width]))
            });
            eprintln!(
                "frame {}: key={} bytes={} decoder matches encoder: {same}",
                codeds.len(),
                coded.1,
                coded.0.color.data.len()
            );
            if !same && let Some(d) = got.first() {
                'find: for (pi, (a, b)) in d.frame.planes.iter().zip(&coded.0.color_recon.planes).enumerate() {
                    for y in 0..a.height {
                        for x in 0..a.width {
                            if a.at(x, y) != b.at(x, y) {
                                eprintln!(
                                    "  first difference: plane {pi} x {x} y {y}: decoder {} encoder {}",
                                    a.at(x, y),
                                    b.at(x, y)
                                );
                                break 'find;
                            }
                        }
                    }
                }
            }
        }
        prev = Some((coded.0.color_recon.clone(), coded.0.alpha_recon.clone()));
        codeds.push(coded);
    }
    let sync: Vec<bool> = codeds.iter().map(|c| c.1).collect();
    let mut it = codeds.into_iter().map(|c| c.0);
    let c0 = it.next().expect("at least one frame");
    let (mut c0_color, mut a0) = (c0.color, c0.alpha);
    // The sequence header goes into the track configuration, where decoders look for it first.
    c0_color.config.config_obus = split_first_obu(&c0_color.data).0.to_vec();
    if let Some(a) = a0.as_mut() {
        a.config.config_obus = split_first_obu(&a.data).0.to_vec();
    }
    let mut out = AvifOutput::new(anim.width, anim.height, c0_color);
    out.alpha = a0;
    let (mut color, mut alpha) = (Vec::new(), Vec::new());
    for c in it {
        color.push(without_sequence_header(&c.color.data).to_vec());
        if let Some(a) = c.alpha {
            alpha.push(without_sequence_header(&a.data).to_vec());
        }
    }
    out.sequence = Some(SequenceOut {
        timescale: 1000,
        durations: anim.frames.iter().map(|f| f.duration_ms).collect(),
        sync,
        color,
        alpha,
        loop_forever: anim.loop_forever,
    });
    finish(&mut out, &first.image, gray);
    Ok(container::write(&out))
}
