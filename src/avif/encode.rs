//! AVIF output: pixels -> AV1 still pictures (colour, plus a monochrome alpha picture) -> HEIF container.

use super::av1::encode::{EncodeParams, encode_picture};
use super::av1::frame::{Frame, Plane};
use super::av1::obu::ColorConfig;
use super::container::{self, Av1Config, Av1Stream, AvifOutput, ColourInfo, Nclx};
use super::convert::rgb_to_yuv;
use crate::error::Result;
use crate::image::Image;

#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// 1-100, higher is better.
    pub quality: u8,
    /// Keep full-resolution chroma (4:4:4) instead of 4:2:0.
    pub full_chroma: bool,
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

fn stream_of(frame: &Frame, color: &ColorConfig, qidx: u8) -> Result<Av1Stream> {
    let params = EncodeParams { base_q_idx: qidx, ..EncodeParams::default() };
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

fn color_config(mono: bool, sub: u8) -> ColorConfig {
    ColorConfig {
        bit_depth: 8,
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

/// Encode an image as an AVIF file (8-bit 4:2:0 or monochrome; 16-bit sources are reduced to 8 bits).
pub fn encode(img: &Image, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let img8 = img.to_depth8();
    let (w, h) = (img8.width as usize, img8.height as usize);
    let qidx = quality_to_qindex(opts.quality);
    let gray = img8.color.is_gray();
    let ch = img8.color.channels();

    let color_stream = if gray {
        let y: Vec<u16> = img8.data.chunks_exact(ch).map(|p| p[0] as u16).collect();
        let frame = Frame { planes: vec![plane_from(&y, w, h)], bit_depth: 8, sub_x: 1, sub_y: 1, width: w, height: h };
        stream_of(&frame, &color_config(true, 1), qidx)?
    } else {
        let sub = if opts.full_chroma { 0 } else { 1 };
        let yuv = rgb_to_yuv(&img8, 8, sub as usize, sub as usize, 0.299, 0.114);
        let frame = Frame {
            planes: vec![
                plane_from(&yuv.y, w, h),
                plane_from(&yuv.u, yuv.cw, yuv.ch),
                plane_from(&yuv.v, yuv.cw, yuv.ch),
            ],
            bit_depth: 8,
            sub_x: sub,
            sub_y: sub,
            width: w,
            height: h,
        };
        stream_of(&frame, &color_config(false, sub), qidx)?
    };

    let alpha_stream = if img8.color.has_alpha() {
        let a: Vec<u16> = img8.data.chunks_exact(ch).map(|p| p[ch - 1] as u16).collect();
        let frame = Frame { planes: vec![plane_from(&a, w, h)], bit_depth: 8, sub_x: 1, sub_y: 1, width: w, height: h };
        // Alpha edges matter more than colour detail: code it a little finer.
        Some(stream_of(&frame, &color_config(true, 1), qidx.saturating_sub(qidx / 8).max(4))?)
    } else {
        None
    };

    let mut out = AvifOutput::new(img8.width, img8.height, color_stream);
    out.alpha = alpha_stream;
    out.colour = ColourInfo {
        nclx: Some(Nclx { matrix_coefficients: if gray { 2 } else { 6 }, ..Nclx::SRGB }),
        icc: img.meta.icc.clone(),
    };
    out.exif = img.meta.exif.clone();
    out.xmp = img.meta.xmp.clone();
    Ok(container::write(&out))
}
