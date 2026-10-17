//! AVIF end to end: our decoder against libavif/Pillow, our encoder checked by independent decoders, and
//! damaged AV1 / AVIF input.
//!
//! The `*.expected.png` files next to the fixtures were produced offline by Pillow 12.3 (libavif 1.4.2, libdav1d),
//! i.e. by decoders that share no code with this crate. Chroma upsampling is the one place decoders legitimately
//! differ, so the comparisons allow a difference of 2 (of 255); the `avif_enc_*` files were written by this
//! crate's encoder and opened with the same decoders.
mod common;
use common::{fixture, to_rgba16};
use rust_compress::avif;
use rust_compress::avif::encode::{EncodeOptions, encode, quality_to_qindex};
use rust_compress::image::{ColorType, Image};
use rust_compress::png;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

fn expected(name: &str) -> Image {
    png::decode::decode(&fixture(&format!("{name}.expected.png"))).unwrap()
}

fn max_diff(a: &Image, b: &Image) -> u32 {
    assert_eq!((a.width, a.height), (b.width, b.height), "dimensions");
    let (x, y) = (to_rgba16(a), to_rgba16(b));
    x.iter().zip(&y).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs()).max().unwrap_or(0) / 257
}

fn check_fixture(name: &str, tolerance: u32) {
    let img = avif::decode(&fixture(&format!("{name}.avif"))).unwrap_or_else(|e| panic!("{name}: {e}"));
    let d = max_diff(&img, &expected(name));
    assert!(d <= tolerance, "{name}: differs from the reference decoder by {d}");
}

#[test]
fn decodes_real_world_files_like_libavif() {
    for name in [
        "avif_rgb",
        "avif_rgba",
        "avif_ffmpeg",
        "avif_grid",
        "avif_red_ffmpeg",
        "avif_anim",
        "avif_written_full",
        "avif_written_red",
    ] {
        check_fixture(name, 2);
    }
}

#[test]
fn decodes_our_encoder_output_like_libavif() {
    for name in ["avif_enc_odd444", "avif_enc_rgba444", "avif_enc_one", "avif_enc_gray", "avif_enc_la"] {
        check_fixture(name, 1);
    }
    // 4:2:0 chroma upsampling filters differ between decoders, mostly at edges.
    for name in ["avif_enc_odd420", "avif_enc_rgba420", "avif_enc_thin420"] {
        let img = avif::decode(&fixture(&format!("{name}.avif"))).unwrap();
        let reference = expected(name);
        assert_eq!((img.width, img.height), (reference.width, reference.height));
        let (a, b) = (to_rgba16(&img), to_rgba16(&reference));
        let sse: f64 = a.iter().zip(&b).map(|(&p, &q)| ((p as f64 - q as f64) / 257.0).powi(2)).sum();
        let mse = sse / a.len() as f64;
        assert!(mse < 12.0, "{name}: mean squared difference {mse}");
    }
}

// ----------------------------------------------------------------------------------------------------------
// Encoder round trips
// ----------------------------------------------------------------------------------------------------------

fn synthetic(w: u32, h: u32, color: ColorType) -> Image {
    let ch = color.channels();
    let mut data = Vec::with_capacity((w * h) as usize * ch);
    let mut s = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (s >> 28) as i32 - 8;
            let v = [(x * 255 / w.max(1)) as i32, (y * 255 / h.max(1)) as i32, ((x + y) * 255 / (w + h).max(1)) as i32];
            for c in 0..ch {
                let base = if color.has_alpha() && c == ch - 1 { (x * 255 / w.max(1)) as i32 } else { v[c.min(2)] };
                data.push((base + noise).clamp(0, 255) as u8);
            }
        }
    }
    Image::new(w, h, color, 8, data).unwrap()
}

fn psnr(a: &Image, b: &Image, channels: usize) -> f64 {
    assert_eq!((a.width, a.height, a.color), (b.width, b.height, b.color));
    let ch = a.color.channels();
    let mut sse = 0f64;
    let mut n = 0f64;
    for (p, q) in a.data.chunks_exact(ch).zip(b.data.chunks_exact(ch)) {
        for c in 0..channels.min(ch) {
            sse += (p[c] as f64 - q[c] as f64).powi(2);
            n += 1.0;
        }
    }
    10.0 * (255.0f64 * 255.0 / (sse / n).max(1e-9)).log10()
}

#[test]
fn round_trips_every_colour_type() {
    for (w, h, color, min_psnr) in [
        (64, 48, ColorType::Rgb, 30.0),
        (37, 23, ColorType::Rgb, 30.0),
        (1, 1, ColorType::Rgb, 30.0),
        (300, 5, ColorType::Rgb, 28.0),
        (50, 40, ColorType::Gray, 36.0),
        (64, 48, ColorType::Rgba, 30.0),
        (31, 17, ColorType::GrayAlpha, 34.0),
    ] {
        for full_chroma in [false, true] {
            let img = synthetic(w, h, color);
            let bytes = encode(&img, &EncodeOptions { quality: 80, full_chroma, effort: 7 }).unwrap();
            assert!(avif::is_avif(&bytes));
            let back = avif::decode(&bytes).unwrap_or_else(|e| panic!("{w}x{h} {color:?}: {e}"));
            assert_eq!(back.color, color, "{w}x{h}");
            let p = psnr(&img, &back, color.channels());
            assert!(p >= min_psnr, "{w}x{h} {color:?} 444={full_chroma}: PSNR {p:.1}");
        }
    }
}

#[test]
fn alpha_survives_nearly_exactly() {
    let img = synthetic(64, 48, ColorType::Rgba);
    let back =
        avif::decode(&encode(&img, &EncodeOptions { quality: 80, full_chroma: true, effort: 7 }).unwrap()).unwrap();
    let worst = img
        .data
        .chunks_exact(4)
        .zip(back.data.chunks_exact(4))
        .map(|(p, q)| (p[3] as i32 - q[3] as i32).abs())
        .max()
        .unwrap();
    assert!(worst <= 12, "alpha error {worst}");
}

#[test]
fn quality_orders_size_and_fidelity() {
    let img = synthetic(96, 64, ColorType::Rgb);
    let mut last = (usize::MAX, 0.0);
    for q in [20u8, 50, 80, 95] {
        let bytes = encode(&img, &EncodeOptions { quality: q, full_chroma: true, effort: 7 }).unwrap();
        let p = psnr(&img, &avif::decode(&bytes).unwrap(), 3);
        assert!(
            !bytes.is_empty() && (last.0 == usize::MAX || bytes.len() >= last.0),
            "q{q}: size must not shrink as quality rises"
        );
        assert!(p >= last.1 - 0.2, "q{q}: PSNR {p} below {}", last.1);
        last = (bytes.len(), p);
    }
    assert!(quality_to_qindex(1) > quality_to_qindex(50) && quality_to_qindex(50) > quality_to_qindex(100));
    assert!(quality_to_qindex(100) > 0, "lossy encoding never uses the lossless index");
}

#[test]
fn metadata_round_trips() {
    let mut img = synthetic(32, 32, ColorType::Rgb);
    img.meta.icc = Some(fixture("icc_profile.bin"));
    img.meta.xmp = Some(b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>".to_vec());
    // A minimal big-endian TIFF block with an empty IFD.
    img.meta.exif = Some(vec![b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0]);
    let back =
        avif::decode(&encode(&img, &EncodeOptions { quality: 70, full_chroma: false, effort: 7 }).unwrap()).unwrap();
    assert_eq!(back.meta.icc, img.meta.icc);
    assert_eq!(back.meta.xmp, img.meta.xmp);
    assert_eq!(back.meta.exif, img.meta.exif);
}

#[test]
fn sixteen_bit_input_is_reduced() {
    let data: Vec<u8> = (0..16 * 16 * 3).flat_map(|i| (((i * 257) % 65536) as u16).to_be_bytes()).collect();
    let img = Image::new(16, 16, ColorType::Rgb, 16, data).unwrap();
    let back =
        avif::decode(&encode(&img, &EncodeOptions { quality: 90, full_chroma: true, effort: 7 }).unwrap()).unwrap();
    assert_eq!((back.width, back.height, back.depth), (16, 16, 8));
}

// ----------------------------------------------------------------------------------------------------------
// Damaged input
// ----------------------------------------------------------------------------------------------------------

/// Mutation rounds scale with `AVIF_FUZZ_SCALE` for longer local runs.
fn scale() -> usize {
    std::env::var("AVIF_FUZZ_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

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

fn survives(name: &str, f: impl FnOnce()) {
    let start = Instant::now();
    assert!(catch_unwind(AssertUnwindSafe(f)).is_ok(), "{name}: panicked");
    assert!(start.elapsed() < Duration::from_secs(20), "{name}: took {:?}", start.elapsed());
}

const CORPUS: &[&str] = &[
    "avif_rgb.avif",
    "avif_rgba.avif",
    "avif_ffmpeg.avif",
    "avif_grid.avif",
    "avif_meta.avif",
    "avif_enc_odd420.avif",
    "avif_enc_rgba444.avif",
    "avif_enc_gray.avif",
    "avif_enc_la.avif",
    "avif_enc_one.avif",
];

#[test]
fn damaged_avif_files_never_panic() {
    let mut rng = Rng(0x5eed_a11f);
    for name in CORPUS {
        let original = fixture(name);
        for round in 0..60 * scale() {
            let mut data = original.clone();
            match round % 4 {
                0 => data.truncate(rng.below(data.len())),
                1 => {
                    for _ in 0..1 + rng.below(4) {
                        let i = rng.below(data.len());
                        data[i] ^= 1 << rng.below(8);
                    }
                }
                2 => {
                    let i = rng.below(data.len());
                    let n = rng.below(16).min(data.len() - i);
                    for b in &mut data[i..i + n] {
                        *b = rng.next() as u8;
                    }
                }
                _ => {
                    let i = rng.below(data.len());
                    data.insert(i, rng.next() as u8);
                }
            }
            survives(&format!("{name} round {round}"), || {
                let _ = avif::decode(&data);
            });
        }
    }
}

#[test]
fn damaged_av1_payloads_never_panic() {
    // Corrupt only the coded bytes so that the container parses and the AV1 decoder sees the damage.
    let mut rng = Rng(0xbad_c0de);
    for name in
        ["avif_enc_odd420.avif", "avif_enc_rgba444.avif", "avif_enc_gray.avif", "avif_ffmpeg.avif", "avif_rgb.avif"]
    {
        let original = fixture(name);
        let mid = original.len() / 2;
        for round in 0..80 * scale() {
            let mut data = original.clone();
            for _ in 0..1 + rng.below(3) {
                let i = mid + rng.below(original.len() - mid);
                data[i] ^= 1 << rng.below(8);
            }
            survives(&format!("{name} round {round}"), || {
                let _ = avif::decode(&data);
            });
        }
    }
}

#[test]
fn raw_obu_streams_survive_mutation() {
    let original = fixture("av1_red_64.obu");
    let mut rng = Rng(0x0b0b_0b0b);
    for round in 0..200 * scale() {
        let mut data = original.clone();
        for _ in 0..1 + rng.below(4) {
            let i = rng.below(data.len());
            data[i] = rng.next() as u8;
        }
        if round % 5 == 0 {
            data.truncate(rng.below(data.len()));
        }
        survives(&format!("obu round {round}"), || {
            let _ = avif::av1::decoder::decode(&data);
        });
    }
}

#[test]
fn absurd_dimensions_are_refused_quickly() {
    // Take a real AVIF and claim a gigantic image in the sequence header / ispe: must error, not allocate.
    let original = fixture("avif_enc_gray.avif");
    for pos in 0..original.len().min(400) {
        let mut data = original.clone();
        data[pos] = 0xff;
        survives(&format!("0xff at {pos}"), || {
            let _ = avif::decode(&data);
        });
    }
}
