use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// Keep each file's own format.
    Keep,
    Png,
    Jpeg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SubsamplingArg {
    /// 4:4:4 for quality >= 90, otherwise 4:2:0.
    Auto,
    #[value(name = "444")]
    S444,
    #[value(name = "420")]
    S420,
}

/// Fast, high-quality image compression (PNG and JPEG) with a from-scratch engine.
#[derive(Parser, Debug)]
#[command(name = "rust-compress", version, about)]
pub struct Cli {
    /// Image files and/or directories to compress. Use `-` to read one image from standard input.
    #[arg(required = true, value_name = "INPUT")]
    pub inputs: Vec<PathBuf>,

    /// Descend into subdirectories.
    #[arg(short, long)]
    pub recursive: bool,

    /// Output file (single input file) or output directory; `-` writes the image to standard output.
    /// Default: write `<name>.min.<ext>` next to each input (standard input defaults to standard output).
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Replace originals, but only when the result is smaller.
    #[arg(long, conflicts_with = "output")]
    pub in_place: bool,

    /// JPEG quality ceiling (1-100). JPEG sources at or below it are transcoded losslessly; better ones are re-quantized down to it.
    #[arg(short, long, default_value_t = 80, value_parser = clap::value_parser!(u8).range(1..=100))]
    pub quality: u8,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = FormatArg::Keep)]
    pub format: FormatArg,

    /// JPEG chroma subsampling.
    #[arg(long, value_enum, default_value_t = SubsamplingArg::Auto)]
    pub subsampling: SubsamplingArg,

    /// PNG: allow lossy palette quantization for much smaller files.
    #[arg(long)]
    pub lossy: bool,

    /// With --lossy: number of palette colors (2-256).
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u16).range(2..=256))]
    pub colors: u16,

    /// With --lossy: dithering strength 0.0-1.0 (0 disables).
    #[arg(long, default_value_t = 0.5)]
    pub dither: f32,

    /// PNG effort (0-9). 1-6 use greedy/lazy parsing (6 is fast); 7-9 use optimal parsing (7 default, 9 smallest and slowest).
    #[arg(short, long, default_value_t = 7, value_parser = clap::value_parser!(u8).range(0..=9))]
    pub level: u8,

    /// JPEG: pick the lowest quality whose result still scores at least this SSIM (0-1, e.g. 0.97)
    /// against the original, instead of a fixed -q. Typical: 0.95 small, 0.97 good, 0.99 near-lossless.
    #[arg(long, value_name = "SSIM", value_parser = parse_ssim)]
    pub target_ssim: Option<f64>,

    /// JPEG: disable trellis (rate-distortion) quantization; faster, but ~5-10% larger at equal quality.
    #[arg(long)]
    pub no_trellis: bool,

    /// PNG: keep the exact (invisible) color values of fully transparent pixels instead of rewriting them for size.
    #[arg(long)]
    pub keep_hidden_pixels: bool,

    /// Drop embedded ICC profiles and PNG color chunks (smaller, but colors may shift on wide-gamut images).
    #[arg(long)]
    pub strip_icc: bool,

    /// Shrink images to fit: `WIDTHxHEIGHT` (a box; aspect ratio is kept), `WIDTHx`, `xHEIGHT`, or a
    /// percentage such as `50%`. Images are never enlarged. Resizing is not lossless and, unlike the
    /// default, may make a file's pixels differ from the original; the never-grow rule does not apply.
    #[arg(long, value_name = "SIZE", value_parser = crate::resize::parse)]
    pub resize: Option<crate::resize::Spec>,

    /// Keep descriptive metadata: EXIF, XMP, comments, PNG text chunks and pixel density. By default only
    /// color information (ICC profile, PNG color chunks) is kept. EXIF and XMP also carry over when
    /// converting between PNG and JPEG; orientation is reset once it has been baked into the pixels.
    #[arg(long)]
    pub keep_metadata: bool,

    /// Background color (hex RRGGBB) used when transparency must be flattened (e.g. PNG -> JPEG).
    #[arg(long, default_value = "ffffff", value_parser = parse_hex_color)]
    pub background: [u8; 3],

    /// Number of worker threads (default: all cores).
    #[arg(short, long)]
    pub jobs: Option<usize>,

    /// Report what would happen without writing anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Overwrite existing output files.
    #[arg(long)]
    pub force: bool,

    /// Only print errors and the final summary line.
    #[arg(long)]
    pub quiet: bool,
}

fn parse_hex_color(s: &str) -> Result<[u8; 3], String> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 || !s.is_ascii() {
        return Err("expected 6 hex digits, e.g. ffffff".into());
    }
    let v = u32::from_str_radix(s, 16).map_err(|_| "expected 6 hex digits, e.g. ffffff".to_string())?;
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

fn parse_ssim(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| "expected a number between 0 and 1".to_string())?;
    if v > 0.0 && v <= 1.0 { Ok(v) } else { Err("SSIM target must be in (0, 1]".into()) }
}
