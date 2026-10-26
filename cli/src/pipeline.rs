//! Per-file processing: read -> decode -> encode -> never-grow check -> atomic write.

use crate::cli::{BitDepthArg, Cli, SubsamplingArg};
use crate::collect::{Format, Job, is_stdio, target_format};
use crate::report::{Report, Status};
use rc_avif as avif;
use rc_core::animation::{AnimFrame, Animation};
use rc_core::error::{Error, Result};
use rc_core::exif;
use rc_core::image::{ColorType, Image, Metadata};
use rc_core::resize;
use rc_gif as gif;
use rc_jpeg as jpeg;
use rc_png as png;
use rc_webp as webp;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// An encoded file plus an optional note for the report (e.g. the quality that was chosen).
struct Encoded {
    bytes: Vec<u8>,
    note: Option<String>,
    /// The user asked for a change (resizing) that the original bytes cannot satisfy, so the
    /// never-grow fallback to the original file must not apply.
    forced: bool,
}

impl From<Vec<u8>> for Encoded {
    fn from(bytes: Vec<u8>) -> Self {
        Encoded { bytes, note: None, forced: false }
    }
}

impl Encoded {
    /// Record that the image was resized, folding the description into the report note.
    fn resized(mut self, what: Option<String>) -> Self {
        if let Some(what) = what {
            self.forced = true;
            self.note = Some(match self.note.take() {
                Some(n) => format!("{what}, {n}"),
                None => what,
            });
        }
        self
    }
}

/// Shrink `img` as requested by `--resize`. Returns the image and a report note if it changed.
fn resize_image(img: Image, cli: &Cli) -> (Image, Option<String>) {
    let Some(spec) = cli.resize else { return (img, None) };
    let (w, h) = (img.width, img.height);
    let (nw, nh) = resize::target_size(spec, w, h);
    if (nw, nh) == (w, h) {
        return (img, None);
    }
    let space = if cli.resize_encoded { resize::Space::Encoded } else { resize::Space::Linear };
    let mut out = resize::resize_in(&img, nw, nh, space);
    let fix = exif::Fix { dims: Some((nw, nh)), drop_thumbnail: true, ..Default::default() };
    if let Some(e) = out.meta.exif.as_mut() {
        exif::tidy(e, fix);
    }
    (out, Some(format!("resized {w}x{h} -> {nw}x{nh}")))
}

/// Encode pixels as JPEG, honoring `--target-ssim` when given.
fn encode_jpeg_pixels(img: &Image, cli: &Cli, quality: u8) -> Result<Encoded> {
    match cli.target_ssim {
        None => Ok(jpeg::encode::encode(img, &jpeg_options(cli, quality))?.into()),
        Some(target) => {
            let reference = img.flatten(cli.background);
            let found = jpeg::target::find_quality(&reference, target, 100, |q| {
                jpeg::encode::build_coefs(&reference, &jpeg_options(cli, q))
            });
            let note = format!("q{}, ssim {:.4}", found.quality, found.ssim);
            Ok(Encoded { bytes: jpeg::encode::write_best(&found.coefs), note: Some(note), forced: false })
        }
    }
}

/// Apply the metadata policy to a freshly decoded image.
fn apply_metadata_policy(meta: &mut Metadata, cli: &Cli) {
    if !cli.keep_metadata {
        meta.drop_descriptive();
    }
    if cli.strip_icc {
        meta.icc = None;
        meta.png_color.clear();
    }
}

/// Bake an EXIF orientation into the pixels, keeping any carried EXIF block consistent with the result.
fn upright(img: Image, orientation: u8) -> Image {
    if orientation == 1 {
        return img;
    }
    let mut img = img.oriented(orientation);
    let fix = exif::Fix { reset_orientation: true, dims: Some((img.width, img.height)), drop_thumbnail: true };
    if let Some(e) = img.meta.exif.as_mut() {
        exif::tidy(e, fix);
    }
    img
}

fn decode(data: &[u8], fmt: Format) -> Result<Image> {
    match fmt {
        Format::Png => png::decode::decode(data),
        Format::Gif => gif::decode(data),
        Format::Webp => webp::decode(data),
        Format::Avif => avif::decode(data),
        // The orientation is baked into the pixels, so the tag must not rotate them a second time.
        Format::Jpeg => Ok(upright(jpeg::decode::decode(data)?, jpeg::decode::exif_orientation(data))),
    }
}

fn jpeg_options(cli: &Cli, quality: u8) -> jpeg::encode::EncodeOptions {
    jpeg::encode::EncodeOptions {
        quality,
        subsampling: match cli.subsampling {
            SubsamplingArg::Auto => jpeg::encode::Subsampling::Auto,
            SubsamplingArg::S444 => jpeg::encode::Subsampling::S444,
            SubsamplingArg::S420 => jpeg::encode::Subsampling::S420,
        },
        background: cli.background,
        trellis: !cli.no_trellis,
    }
}

/// JPEG -> JPEG. Works on DCT coefficients whenever pixels do not need to change: sources at or
/// below the target quality are transcoded losslessly (just better entropy coding), higher-quality
/// sources are re-quantized in the DCT domain. Falls back to a pixel round trip when chroma
/// subsampling or orientation must change.
fn jpeg_to_jpeg(data: &[u8], cli: &Cli) -> Result<Encoded> {
    let parsed = jpeg::decode::parse(data)?;
    let info = parsed.info();
    let orientation = jpeg::decode::exif_orientation(data);
    let three = info.sampling.len() == 3;
    let is444 = three && info.sampling.iter().all(|&s| s == (1, 1));
    let is420 = three && info.sampling[0] == (2, 2) && info.sampling[1..].iter().all(|&s| s == (1, 1));
    let effective_q = cli.quality.min(info.source_quality);
    // Dimensions as they will look once the orientation is applied.
    let (ow, oh) = if orientation >= 5 { (info.height, info.width) } else { (info.width, info.height) };
    let resizing = cli.resize.is_some_and(|s| resize::target_size(s, ow as u32, oh as u32) != (ow as u32, oh as u32));
    let resample = three
        && match cli.subsampling {
            // With a quality search the final quality is unknown up front: keep the source layout.
            SubsamplingArg::Auto => is444 && effective_q < 90 && cli.target_ssim.is_none(),
            SubsamplingArg::S420 => !is420,
            SubsamplingArg::S444 => !is444,
        };
    if orientation == 1 && !resample && !info.coded_rgb && !resizing {
        let mut ci = parsed.into_coefs()?;
        apply_metadata_policy(&mut ci.meta, cli);
        if let Some(target) = cli.target_ssim {
            let reference = jpeg::decode::reconstruct(&ci)?;
            let found = jpeg::target::find_quality(&reference, target, 100, |q| {
                let mut c = ci.clone();
                c.requantize(q, !cli.no_trellis);
                c
            });
            let note = format!("q{}, ssim {:.4}", found.quality, found.ssim);
            return Ok(Encoded { bytes: jpeg::encode::write_best(&found.coefs), note: Some(note), forced: false });
        }
        let before = ci.qtables;
        ci.requantize(cli.quality, !cli.no_trellis);
        let mut encoded: Encoded = jpeg::encode::write_best(&ci).into();
        if info.cmyk {
            encoded.note = Some(if ci.qtables == before { "CMYK, lossless" } else { "CMYK, requantized" }.into());
        }
        return Ok(encoded);
    }
    let img = upright(parsed.into_image()?, orientation);
    let mut img = img;
    apply_metadata_policy(&mut img.meta, cli);
    let (img, resized) = resize_image(img, cli);
    Ok(encode_jpeg_pixels(&img, cli, effective_q)?.resized(resized))
}

fn encode(img: &Image, fmt: Format, cli: &Cli) -> Result<Encoded> {
    match fmt {
        Format::Png => {
            let quantized;
            let img = if cli.lossy {
                quantized = png::quant::quantize(img, cli.colors as usize, cli.dither);
                &quantized
            } else {
                img
            };
            Ok(png::encode::encode(
                img,
                &png::encode::EncodeOptions { level: cli.level as u32, clean_alpha: !cli.keep_hidden_pixels },
            )
            .into())
        }
        Format::Jpeg => encode_jpeg_pixels(img, cli, cli.quality),
        Format::Webp => {
            if cli.lossy {
                let bytes = webp::encode_lossy(img, cli.quality, cli.level as u32, !cli.keep_hidden_pixels)?;
                let mut encoded: Encoded = bytes.into();
                encoded.forced = false;
                if img.depth == 16 {
                    encoded.note = Some("16-bit reduced to 8-bit".into());
                }
                return Ok(encoded);
            }
            if img.width > webp::MAX_DIMENSION || img.height > webp::MAX_DIMENSION {
                return Err(Error::unsupported(format!(
                    "WebP images are limited to {0}x{0} pixels",
                    webp::MAX_DIMENSION
                )));
            }
            let opts = webp::EncodeOptions { level: cli.level as u32, clean_alpha: !cli.keep_hidden_pixels };
            let mut encoded: Encoded = webp::encode(img, &opts).into();
            if img.depth == 16 {
                encoded.note = Some("16-bit reduced to 8-bit".into());
            }
            Ok(encoded)
        }
        Format::Avif => {
            let full_chroma = match cli.subsampling {
                SubsamplingArg::Auto => cli.quality >= 90,
                SubsamplingArg::S444 => true,
                SubsamplingArg::S420 => false,
            };
            let bit_depth = match cli.bit_depth {
                BitDepthArg::Auto if img.depth == 16 => 10,
                BitDepthArg::Auto | BitDepthArg::B8 => 8,
                BitDepthArg::B10 => 10,
            };
            let opts = avif::encode::EncodeOptions { quality: cli.quality, full_chroma, effort: cli.level, bit_depth };
            let mut encoded: Encoded = avif::encode::encode(img, &opts)?.into();
            if img.depth == 16 && bit_depth == 8 {
                encoded.note = Some("16-bit reduced to 8-bit".into());
            }
            Ok(encoded)
        }
        Format::Gif => {
            let frame = AnimFrame { image: img.clone(), duration_ms: rc_core::animation::DEFAULT_DELAY_MS };
            let still = Animation { width: img.width, height: img.height, frames: vec![frame], loop_forever: false };
            let mut encoded: Encoded = gif::encode_animation(&still, &gif_options(cli))?.into();
            if img.depth == 16 {
                encoded.note = Some("16-bit reduced to 8-bit".into());
            }
            Ok(encoded)
        }
    }
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all().ok();
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.map_err(Error::from)
}

/// Write the result to its destination: standard output for `-`, otherwise an atomically replaced file.
fn write_output(path: &Path, data: &[u8]) -> Result<()> {
    if is_stdio(path) {
        let mut out = std::io::stdout().lock();
        out.write_all(data)?;
        out.flush()?;
        return Ok(());
    }
    write_atomic(path, data)
}

fn run(job: &Job, cli: &Cli, data: &[u8]) -> Result<(Status, u64, Option<String>)> {
    let in_fmt = Format::sniff(data)
        .or_else(|| Format::from_extension(&job.input))
        .ok_or_else(|| Error::unsupported("unrecognized image format (supported: PNG, JPEG, WebP, AVIF, GIF)"))?;
    let out_fmt = target_format(cli, in_fmt);
    if cli.in_place && out_fmt != in_fmt {
        return Err(Error::unsupported("--in-place cannot be combined with a format change"));
    }
    if !cli.in_place && !cli.force && !cli.dry_run && !is_stdio(&job.output) && job.output.exists() {
        return Err(Error::unsupported(format!("{} already exists (use --force)", job.output.display())));
    }
    // Recompressing an animation as a still would drop every frame but the first. Lossy WebP and
    // lossy AVIF are left as they are unless --lossy explicitly asks for a re-encode (otherwise a
    // re-encode would only add generation loss).
    let untouchable = match (in_fmt, out_fmt) {
        (Format::Png, Format::Png) if png::is_animated(data) => Some("animated PNG left unchanged"),
        (Format::Webp, Format::Webp) => webp::unsupported_reason(data)
            .filter(|r| !(cli.lossy && matches!(*r, "lossy WebP" | "animated WebP")))
            .map(|r| {
                if r == "lossy WebP" {
                    "lossy WebP left unchanged"
                } else {
                    "animated WebP left unchanged (use --lossy to re-encode)"
                }
            }),
        // Re-encoding a lossy AVIF would only add generation loss, so it needs an explicit --lossy.
        (Format::Avif, Format::Avif) if !cli.lossy && cli.resize.is_none() => {
            Some("AVIF left unchanged (use --lossy to re-encode)")
        }
        _ => None,
    };
    if let Some(note) = untouchable {
        if !cli.in_place && !cli.dry_run {
            write_output(&job.output, data)?;
        }
        return Ok((Status::NoGain, data.len() as u64, Some(note.into())));
    }
    let animation = if matches!(out_fmt, Format::Avif | Format::Gif | Format::Webp) {
        decode_animation(data, in_fmt)?
    } else {
        None
    };
    let encoded = if let Some(anim) = animation {
        match animation_to(anim, in_fmt, out_fmt, cli)? {
            Some(encoded) => encoded,
            None => {
                // A GIF whose frames do not fit one palette would need a lossy re-quantization.
                if !cli.in_place && !cli.dry_run {
                    write_output(&job.output, data)?;
                }
                let note = "GIF left unchanged (use --lossy to re-quantize)";
                return Ok((Status::NoGain, data.len() as u64, Some(note.into())));
            }
        }
    } else if in_fmt == Format::Jpeg && out_fmt == Format::Jpeg {
        jpeg_to_jpeg(data, cli)?
    } else {
        let mut img = decode(data, in_fmt)?;
        apply_metadata_policy(&mut img.meta, cli);
        let (img, resized) = resize_image(img, cli);
        encode(&img, out_fmt, cli)?.resized(resized)
    };
    let note = encoded.note;
    let forced = encoded.forced;
    let encoded = encoded.bytes;
    let same_format = in_fmt == out_fmt;
    if same_format && !forced && encoded.len() >= data.len() {
        // Never make the file bigger. Keep the original bytes (copy them when writing elsewhere).
        if !cli.in_place && !cli.dry_run {
            write_output(&job.output, data)?;
        }
        return Ok((Status::NoGain, data.len() as u64, note));
    }
    if !cli.dry_run {
        write_output(&job.output, &encoded)?;
    }
    Ok((Status::Compressed, encoded.len() as u64, note))
}

/// The frames of an animated input, `None` for a still (or a format without animation here).
fn decode_animation(data: &[u8], fmt: Format) -> Result<Option<Animation>> {
    let anim = match fmt {
        Format::Gif => Some(gif::decode_animation(data)?),
        Format::Webp if webp::unsupported_reason(data) == Some("animated WebP") => Some(webp::decode_animation(data)?),
        Format::Png if png::is_animated(data) => Some(png::apng::decode_animation(data)?),
        Format::Avif => Some(avif::decode_animation(data)?),
        _ => None,
    };
    Ok(anim.filter(|a| a.frames.len() > 1))
}

fn gif_options(cli: &Cli) -> gif::EncodeOptions {
    gif::EncodeOptions { colors: cli.colors as usize, dither: cli.dither }
}

/// Encode an animation as an animated AVIF, WebP or GIF, resizing every frame when asked. `None` means the
/// original GIF should be kept: its frames do not fit one palette and re-quantizing them is lossy.
fn animation_to(mut anim: Animation, in_fmt: Format, out_fmt: Format, cli: &Cli) -> Result<Option<Encoded>> {
    let mut resized = None;
    if cli.resize.is_some() {
        for f in &mut anim.frames {
            let (img, r) =
                resize_image(std::mem::replace(&mut f.image, Image::new(1, 1, ColorType::Gray, 8, vec![0])?), cli);
            f.image = img;
            resized = r;
        }
        anim.width = anim.frames[0].image.width;
        anim.height = anim.frames[0].image.height;
    }
    let n = anim.frames.len();
    let mut encoded: Encoded = match out_fmt {
        Format::Avif => {
            let full_chroma = match cli.subsampling {
                SubsamplingArg::Auto => cli.quality >= 90,
                SubsamplingArg::S444 => true,
                SubsamplingArg::S420 => false,
            };
            let bit_depth = if cli.bit_depth == BitDepthArg::B10 { 10 } else { 8 };
            let opts = avif::encode::EncodeOptions { quality: cli.quality, full_chroma, effort: cli.level, bit_depth };
            let mut e: Encoded = avif::encode::encode_animation(&anim, &opts)?.into();
            e.forced = true;
            e
        }
        Format::Webp => {
            let opts = webp::AnimOptions {
                level: cli.level as u32,
                clean_alpha: !cli.keep_hidden_pixels,
                lossy_quality: cli.lossy.then_some(cli.quality),
            };
            webp::encode_animation(&anim, &opts)?.into()
        }
        Format::Gif => {
            let opts = gif_options(cli);
            if in_fmt == Format::Gif && !cli.lossy && resized.is_none() && !gif::palette_is_exact(&anim, opts.colors) {
                return Ok(None);
            }
            gif::encode_animation(&anim, &opts)?.into()
        }
        Format::Png | Format::Jpeg => {
            return Err(Error::unsupported("animations can be written as AVIF, WebP or GIF"));
        }
    };
    encoded.note = Some(format!("{n} frames"));
    Ok(Some(encoded.resized(resized)))
}

pub fn process(job: &Job, cli: &Cli) -> Report {
    let mut report = Report { input: job.input.clone(), in_size: 0, out_size: 0, status: Status::NoGain, note: None };
    let read = if is_stdio(&job.input) {
        let mut buf = Vec::new();
        std::io::stdin().lock().read_to_end(&mut buf).map(|_| buf)
    } else {
        fs::read(&job.input)
    };
    let data = match read {
        Ok(d) => d,
        Err(e) => {
            report.status = Status::Failed(e.to_string());
            return report;
        }
    };
    report.in_size = data.len() as u64;
    match run(job, cli, &data) {
        Ok((status, out_size, note)) => {
            report.note = note;
            report.status = status;
            report.out_size = out_size;
        }
        Err(e) => report.status = Status::Failed(e.to_string()),
    }
    report
}
