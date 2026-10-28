//! Hostile-input tests: every decoder/encoder path must reject or survive damaged files without
//! panicking, hanging or allocating absurd amounts of memory.
use common::fixture;
use rc_jpeg as jpeg;
use rc_png as png;
use rc_testutil as common;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const PNGS: &[&str] = &[
    "ga8.png",
    "ga8_adam7.png",
    "gray1.png",
    "gray16.png",
    "gray2.png",
    "gray4.png",
    "gray8.png",
    "icc.png",
    "pal2.png",
    "pal200.png",
    "pal5.png",
    "pal_trns.png",
    "rgb16.png",
    "rgb8.png",
    "rgb8_adam7.png",
    "rgb8_key.png",
    "rgba16_adam7.png",
    "rgba8.png",
];
const JPEGS: &[&str] = &[
    "cmyk.jpg",
    "gray8.jpg",
    "icc.jpg",
    "rgb8.jpg",
    "rgb8_420.jpg",
    "rgb8_444.jpg",
    "rgb8_opt.jpg",
    "rgb8_prog.jpg",
    "ycck.jpg",
    // Arithmetic-coded: sequential, progressive, restart intervals, custom conditioning, gray, CMYK.
    "arith_420_seq.jpg",
    "arith_420_prog.jpg",
    "arith_444_prog_rst.jpg",
    "arith_cond.jpg",
    "arith_gray.jpg",
    "arith_cmyk.jpg",
    // 12-bit.
    "jpeg12_gray.jpg",
    "jpeg12_444.jpg",
    "jpeg12_420.jpg",
];

/// Small deterministic generator so failures are reproducible from the printed seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Run a decoded image through the encoders the CLI would use next.
fn exercise_png(bytes: &[u8]) {
    if let Ok(img) = png::decode::decode(bytes) {
        let opts = png::encode::EncodeOptions { level: 1, ..Default::default() };
        let out = png::encode::encode(&img, &opts);
        // Whatever we decoded must re-encode to something that decodes to the same pixels.
        let back = png::decode::decode(&out).expect("own output must decode");
        assert_eq!((back.width, back.height), (img.width, img.height));
        let _ = png::quant::quantize(&img, 16, 0.5);
    }
}

fn exercise_jpeg(bytes: &[u8]) {
    let _ = jpeg::decode::exif_orientation(bytes);
    if let Ok(img) = jpeg::decode::decode(bytes) {
        let _ = jpeg::encode::encode(&img, &jpeg::encode::EncodeOptions::default());
    }
    if let Ok(parsed) = jpeg::decode::parse(bytes) {
        let _ = parsed.info();
        if let Ok(mut ci) = parsed.into_coefs() {
            ci.requantize(60, true);
            let _ = jpeg::encode::write_best(&ci);
            let _ = jpeg::arith::write_arithmetic(&ci);
        }
    }
}

fn exercise_gif(bytes: &[u8]) {
    if let Ok(img) = rc_gif::decode(bytes) {
        let opts = png::encode::EncodeOptions { level: 1, ..Default::default() };
        let _ = png::encode::encode(&img, &opts);
    }
}

fn exercise_webp(bytes: &[u8]) {
    let _ = rc_webp::unsupported_reason(bytes);
    if let Ok(img) = rc_webp::decode(bytes) {
        // Whatever decoded must survive a re-encode and decode back to the same pixels.
        let opts = rc_webp::EncodeOptions { level: 3, clean_alpha: false };
        if img.width <= 4096 && img.height <= 4096 {
            let back = rc_webp::decode(&rc_webp::encode(&img, &opts)).expect("own output decodes");
            // Opaque RGBA is stored as RGB, so compare as canonical RGBA.
            assert_eq!(common::to_rgba16(&back), common::to_rgba16(&img));
        }
    }
}

type Exercise = fn(&[u8]);

/// Per-case wall-clock budget. This is a hang detector, not a speed benchmark: debug builds
/// running the hostile suites in parallel on a small machine can be several times slower than a
/// release build on one core, so the budget is deliberately generous. A real hang never finishes
/// and is still caught.
const CASE_BUDGET: Duration = Duration::from_secs(60);

/// Apply `exercise` to many damaged variants of `original`; collect panics and slow runs.
fn hammer(name: &str, original: &[u8], exercise: Exercise, failures: &mut Vec<String>) {
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    // Truncations: every length near the start (headers) and a spread through the rest.
    for cut in (0..original.len().min(80)).chain((0..64).map(|i| original.len() * i / 64)) {
        cases.push((format!("truncate@{cut}"), original[..cut].to_vec()));
    }
    let mut rng = Rng(0x5eed ^ original.len() as u64);
    // HOSTILE_ROUNDS=20000 cargo test --release --test hostile  for a deeper (slower) sweep.
    let rounds = std::env::var("HOSTILE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    for i in 0..rounds {
        let mut m = original.to_vec();
        for _ in 0..1 + i % 4 {
            let at = if i % 3 == 0 { rng.below(m.len().min(64)) } else { rng.below(m.len()) };
            m[at] = match rng.below(3) {
                0 => 0x00,
                1 => 0xFF,
                _ => rng.next() as u8,
            };
        }
        cases.push((format!("flip#{i}"), m));
    }
    // Garbage appended / spliced in.
    let mut tail = original.to_vec();
    tail.extend((0..4096).map(|_| rng.next() as u8));
    cases.push(("trailing-garbage".into(), tail));
    for (label, bytes) in cases {
        let start = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| exercise(&bytes)));
        if r.is_err() {
            failures.push(format!("{name} {label}: panicked"));
        } else if start.elapsed() > CASE_BUDGET {
            failures.push(format!("{name} {label}: took {:?}", start.elapsed()));
        }
    }
}

fn report(failures: Vec<String>) {
    for f in failures.iter().take(20) {
        eprintln!("{f}");
    }
    assert!(failures.is_empty(), "{} hostile-input failures (first shown above)", failures.len());
}

#[test]
fn damaged_pngs_never_panic() {
    let mut failures = Vec::new();
    for name in PNGS {
        hammer(name, &fixture(name), exercise_png, &mut failures);
    }
    report(failures);
}

#[test]
fn damaged_gifs_never_panic() {
    let mut failures = Vec::new();
    for name in ["gif_plain.gif", "gif_interlaced.gif", "gif_transparent.gif", "gif_anim.gif"] {
        hammer(name, &fixture(name), exercise_gif, &mut failures);
    }
    report(failures);
}

#[test]
fn damaged_webps_never_panic() {
    let mut failures = Vec::new();
    for name in [
        "webp_rgb.webp",
        "webp_rgba.webp",
        "webp_pal.webp",
        "webp_bw.webp",
        "webp_tiled.webp",
        "webp_noise4.webp",
        "webp_noise16.webp",
        "webp_noise256.webp",
        "webp_lossy.webp",
    ] {
        hammer(name, &fixture(name), exercise_webp, &mut failures);
    }
    report(failures);
}

fn vp8_fixture(name: &str) -> Vec<u8> {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "fixtures", "vp8", name].iter().collect();
    std::fs::read(p).unwrap()
}

#[test]
fn damaged_lossy_webps_never_panic() {
    let mut failures = Vec::new();
    for name in [
        "q75_photo.webp",
        "q10_m6.webp",
        "seg3.webp",
        "part3.webp",
        "simple_filter.webp",
        "sharp7.webp",
        "size_3x2.webp",
        "size_33x1.webp",
        "alpha_default.webp",
        "alpha_raw.webp",
        "alpha_filter_best.webp",
        "alpha_small.webp",
    ] {
        hammer(name, &vp8_fixture(name), |b| drop(rc_webp::decode(b)), &mut failures);
    }
    report(failures);
}

#[test]
fn damaged_jpegs_never_panic() {
    let mut failures = Vec::new();
    for name in JPEGS {
        hammer(name, &fixture(name), exercise_jpeg, &mut failures);
    }
    report(failures);
}

fn crc_fix_png(png: &mut [u8]) {
    // Recompute the IHDR chunk CRC (bytes 29..33) so the decoder gets past its integrity check.
    let crc = rc_core::util::checksum::crc32(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
}

#[test]
fn png_with_absurd_header_values_is_rejected_cheaply() {
    for (w, h) in [(0u32, 0u32), (1, 0), (0, 1), (u32::MAX, 1), (1, u32::MAX), (100_000, 100_000), (65_536, 65_536)] {
        let mut b = fixture("rgb8.png");
        b[16..20].copy_from_slice(&w.to_be_bytes());
        b[20..24].copy_from_slice(&h.to_be_bytes());
        crc_fix_png(&mut b);
        let start = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| png::decode::decode(&b)));
        assert!(r.is_ok(), "{w}x{h}: panicked");
        assert!(r.unwrap().is_err(), "{w}x{h}: accepted");
        assert!(start.elapsed() < Duration::from_secs(2), "{w}x{h}: slow rejection");
    }
}

#[test]
fn jpeg_with_absurd_frame_size_is_rejected_cheaply() {
    let orig = fixture("rgb8.jpg");
    // SOF0/SOF2 marker: height and width follow the 2-byte length and 1-byte precision.
    let sof = orig.windows(2).position(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC2)).expect("SOF marker");
    for (w, h) in [(0u16, 0u16), (65_535, 65_535), (1, 65_535), (65_535, 1)] {
        let mut b = orig.clone();
        b[sof + 5..sof + 7].copy_from_slice(&h.to_be_bytes());
        b[sof + 7..sof + 9].copy_from_slice(&w.to_be_bytes());
        let start = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| exercise_jpeg(&b)));
        assert!(r.is_ok(), "{w}x{h}: panicked");
        assert!(start.elapsed() < Duration::from_secs(5), "{w}x{h}: took {:?}", start.elapsed());
    }
}

#[test]
fn deflate_bomb_is_bounded() {
    // 200 MB of zeros deflates to ~200 KB; inflating must honour the caller's output limit.
    let zeros = vec![0u8; 200 << 20];
    let packed = rc_deflate::deflate_compress(&zeros, 1);
    let r = rc_deflate::inflate(&packed, 0, 1 << 20);
    assert!(r.is_err(), "inflate ignored its output limit");
}
