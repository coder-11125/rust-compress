//! Per-file processing: read -> decode -> encode -> never-grow check -> atomic write.

use crate::cli::{Cli, SubsamplingArg};
use crate::collect::{Format, Job, is_stdio, target_format};
use crate::error::{Error, Result};
use crate::exif;
use crate::image::{Image, Metadata};
use crate::report::{Report, Status};
use crate::{jpeg, png, resize};
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
    let mut out = resize::resize(&img, nw, nh);
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
        Format::Gif => crate::gif::decode(data),
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
        ci.requantize(cli.quality, !cli.no_trellis);
        return Ok(jpeg::encode::write_best(&ci).into());
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
        Format::Gif => Err(Error::unsupported("GIF output")),
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
        .ok_or_else(|| Error::unsupported("unrecognized image format (supported: PNG, JPEG, and GIF as input)"))?;
    let out_fmt = target_format(cli, in_fmt);
    if out_fmt == Format::Gif {
        return Err(Error::unsupported("GIF files can only be converted: use --format png or --format jpeg"));
    }
    if cli.in_place && out_fmt != in_fmt {
        return Err(Error::unsupported("--in-place cannot be combined with a format change"));
    }
    if !cli.in_place && !cli.force && !cli.dry_run && !is_stdio(&job.output) && job.output.exists() {
        return Err(Error::unsupported(format!("{} already exists (use --force)", job.output.display())));
    }
    // Recompressing an animation as a still would drop every frame but the first, so leave it as it is.
    if in_fmt == Format::Png && out_fmt == Format::Png && png::is_animated(data) {
        if !cli.in_place && !cli.dry_run {
            write_output(&job.output, data)?;
        }
        return Ok((Status::NoGain, data.len() as u64, Some("animated PNG left unchanged".into())));
    }
    let encoded = if in_fmt == Format::Jpeg && out_fmt == Format::Jpeg {
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

pub fn process(job: &Job, cli: &Cli) -> Report {
    let mut report = Report {
        input: job.input.clone(),
        output: job.output.clone(),
        in_size: 0,
        out_size: 0,
        status: Status::NoGain,
        note: None,
    };
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
