use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// Keep each file's own format.
    Keep,
    /// PNG stills; animated inputs are written as APNG.
    Png,
    Jpeg,
    /// Lossless WebP.
    Webp,
    /// AVIF (always lossy; -q sets the quality).
    Avif,
    /// GIF (stills and animations; at most --colors palette entries, so always lossy for photos).
    Gif,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum BitDepthArg {
    /// 10 bits for 16-bit sources, otherwise 8.
    Auto,
    #[value(name = "8")]
    B8,
    #[value(name = "10")]
    B10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ArithmeticArg {
    /// Arithmetic coding for arithmetic-coded JPEG sources, Huffman otherwise.
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SubsamplingArg {
    /// 4:4:4 for quality >= 90, otherwise 4:2:0.
    Auto,
    #[value(name = "444")]
    S444,
    #[value(name = "422")]
    S422,
    #[value(name = "420")]
    S420,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Check GitHub for a newer release and replace this binary with it (a file named `update` must be written `./update`).
    Update(crate::update::UpdateArgs),
}

/// Fast, high-quality image compression (PNG and JPEG) with a from-scratch engine.
#[derive(Parser, Debug)]
#[command(
    name = "rc",
    version,
    about,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true,
    // Stated outright: clap's error messages otherwise list only the required arguments, which hides the options and
    // the `update` subcommand from anyone who just typed `rc`.
    override_usage = "rc [OPTIONS] <INPUT>...\n       rc update [OPTIONS]"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Image files and/or directories to compress. Use `-` to read one image from standard input.
    #[arg(required = true, value_name = "INPUT")]
    pub inputs: Vec<PathBuf>,

    /// Descend into subdirectories.
    #[arg(short, long)]
    pub recursive: bool,

    /// With directory inputs: only process files whose name or path relative to that directory matches GLOB
    /// (`*` and `?` stay within a path component, `**` crosses directories). Repeatable.
    #[arg(long, value_name = "GLOB")]
    pub include: Vec<String>,

    /// With directory inputs: skip files matching GLOB (same syntax as --include). Repeatable.
    #[arg(long, value_name = "GLOB")]
    pub exclude: Vec<String>,

    /// Suffix for default outputs: `<name>.<suffix>.<ext>` (default `min`). Files already ending in
    /// `.<suffix>.` are skipped during directory walks.
    #[arg(long, default_value = "min", value_parser = parse_suffix)]
    pub min_suffix: String,

    /// Output path template when no -o/--in-place is given. Placeholders: {root} (the directory input for
    /// walked files, else the input's parent), {dir} (the input file's parent), {rel} / {rel_stem} (path
    /// relative to {root}, without extension for the latter), {stem}, {name}, {ext} and {suffix}.
    /// Relative results are relative to the current directory; the default is `{root}/{rel_stem}.{suffix}.{ext}`.
    #[arg(long, value_name = "TEMPLATE", conflicts_with_all = ["output", "in_place"])]
    pub name_template: Option<String>,

    /// Output file (single input file) or output directory; `-` writes the image to standard output.
    /// Default: write `<name>.<suffix>.<ext>` next to each input (standard input defaults to standard output).
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Replace originals, but only when the result is smaller.
    #[arg(long, conflicts_with = "output")]
    pub in_place: bool,

    /// Quality (1-100): the JPEG quality ceiling (JPEG sources at or below it are transcoded losslessly; better ones are re-quantized down to it), the lossy WebP quality with --lossy, or the AVIF quality.
    #[arg(short, long, default_value_t = 80, value_parser = clap::value_parser!(u8).range(1..=100))]
    pub quality: u8,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = FormatArg::Keep)]
    pub format: FormatArg,

    /// JPEG chroma subsampling.
    #[arg(long, value_enum, default_value_t = SubsamplingArg::Auto)]
    pub subsampling: SubsamplingArg,

    /// JPEG output entropy coding. Arithmetic coding is lossless against Huffman and makes the file noticeably
    /// smaller, but browsers and many other programs cannot open such files; `auto` therefore uses it only when
    /// the source JPEG already was arithmetic-coded.
    #[arg(long, value_enum, default_value_t = ArithmeticArg::Auto)]
    pub arithmetic: ArithmeticArg,

    /// AVIF output bit depth: auto keeps 16-bit sources at 10 bits, everything else at 8.
    #[arg(long, value_enum, default_value_t = BitDepthArg::Auto)]
    pub bit_depth: BitDepthArg,

    /// Allow lossy output: PNG palette quantization, lossy (VP8) WebP when the output is WebP (animations too), re-encoding an AVIF source (AVIF is otherwise left unchanged), or re-quantizing a GIF whose frames do not fit one palette (such GIFs are otherwise left unchanged).
    #[arg(long)]
    pub lossy: bool,

    /// With --lossy on PNG, and for GIF output: number of palette colors (2-256).
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u16).range(2..=256))]
    pub colors: u16,

    /// With --lossy on PNG, and for still GIF output: dithering strength 0.0-1.0 (0 disables). Animated GIFs are never dithered.
    #[arg(long, default_value_t = 0.5, value_parser = parse_dither)]
    pub dither: f32,

    /// PNG / WebP / AVIF effort (0-9). For PNG, 1-6 use greedy/lazy parsing (6 is fast) and 7-9 optimal parsing
    /// (7 default, 9 smallest and slowest).
    #[arg(short, long, default_value_t = 7, value_parser = clap::value_parser!(u8).range(0..=9))]
    pub level: u8,

    /// Pick the lowest quality whose result still scores at least this SSIM (0-1, e.g. 0.97)
    /// against the original, instead of a fixed -q. Works for JPEG, and for lossy WebP, AVIF and
    /// `--lossy` PNG. Typical: 0.95 small, 0.97 good, 0.99 near-lossless.
    #[arg(long, value_name = "SSIM", value_parser = parse_ssim, conflicts_with_all = ["target_psnr", "target_ms_ssim", "target_size"])]
    pub target_ssim: Option<f64>,

    /// Like --target-ssim, but target a PSNR in dB (higher is better) for lossy WebP, AVIF and `--lossy` PNG.
    #[arg(long, value_name = "DB", value_parser = parse_psnr, conflicts_with_all = ["target_ssim", "target_ms_ssim", "target_size"])]
    pub target_psnr: Option<f64>,

    /// Like --target-ssim, but target multi-scale SSIM for lossy WebP, AVIF and `--lossy` PNG.
    #[arg(long, value_name = "SSIM", value_parser = parse_ssim, conflicts_with_all = ["target_ssim", "target_psnr", "target_size"])]
    pub target_ms_ssim: Option<f64>,

    /// Largest output size in bytes for lossy WebP, AVIF and `--lossy` PNG: picks the highest quality
    /// whose file still fits (or the smallest quality when even that is too big).
    #[arg(long, value_name = "BYTES", value_parser = clap::value_parser!(u64).range(1..), conflicts_with_all = ["target_ssim", "target_psnr", "target_ms_ssim"])]
    pub target_size: Option<u64>,

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
    #[arg(long, value_name = "SIZE", value_parser = rc_core::resize::parse)]
    pub resize: Option<rc_core::resize::Spec>,

    /// With --resize: filter the stored (gamma-encoded) values instead of linear light. A little faster and
    /// what most tools do, but fine detail comes out darker than the original looks.
    #[arg(long, requires = "resize")]
    pub resize_encoded: bool,

    /// Keep descriptive metadata: EXIF, XMP, comments, PNG text chunks and pixel density. By default only
    /// color information (ICC profile, PNG color chunks) is kept. EXIF and XMP also carry over when
    /// converting between PNG and JPEG; orientation is reset once it has been baked into the pixels.
    #[arg(long)]
    pub keep_metadata: bool,

    /// Background color (hex RRGGBB) used when transparency must be flattened (e.g. PNG -> JPEG).
    #[arg(long, default_value = "ffffff", value_parser = parse_hex_color)]
    pub background: [u8; 3],

    /// Reject images with more than N pixels (default and maximum: 150000000). Lower it to keep hostile or huge files
    /// from needing gigabytes: a 16-bit RGBA image takes about 20 bytes per pixel while it is processed.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..=150_000_000))]
    pub max_pixels: Option<u64>,

    /// Soft limit, in MiB, on the memory that images being processed at the same time may need (estimated from each
    /// image's size). Images wait for others to finish instead of being decoded together; one image bigger than the
    /// limit still runs, alone.
    #[arg(long, value_name = "MIB", default_value_t = 4096, value_parser = clap::value_parser!(u64).range(1..=4194304))]
    pub max_memory: u64,

    /// Number of worker threads, 1-1024 (default: all cores).
    #[arg(short, long, value_parser = parse_jobs)]
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

    /// Print the batch report as one JSON object instead of text lines (goes to stderr when the image itself
    /// is written to stdout).
    #[arg(long)]
    pub json: bool,
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

fn parse_psnr(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| "expected a number in dB".to_string())?;
    if v > 0.0 && v <= 99.0 { Ok(v) } else { Err("PSNR target must be in (0, 99]".into()) }
}

fn parse_dither(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| "expected a number between 0.0 and 1.0".to_string())?;
    if (0.0..=1.0).contains(&v) { Ok(v) } else { Err("dither strength must be between 0.0 and 1.0".into()) }
}

/// Far more threads than any machine has only fail later, while the thread pool is built.
const MAX_JOBS: usize = 1024;

fn parse_jobs(s: &str) -> Result<usize, String> {
    let n: usize = s.parse().map_err(|_| "expected a number of threads".to_string())?;
    if (1..=MAX_JOBS).contains(&n) { Ok(n) } else { Err(format!("--jobs must be between 1 and {MAX_JOBS}")) }
}

fn parse_suffix(s: &str) -> Result<String, String> {
    if s.is_empty() || s == "." || s == ".." || s.contains(['/', '\\']) {
        return Err("suffix must be a non-empty file-name component".into());
    }
    Ok(s.to_string())
}
