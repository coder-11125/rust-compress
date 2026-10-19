mod common;
use common::{fixture, reference_rgba16, to_rgba16};
use rust_compress::webp;

const LOSSLESS: &[&str] = &[
    "webp_rgb.webp",
    "webp_rgb_fast.webp",
    "webp_rgba.webp",
    "webp_pal.webp",
    "webp_bw.webp",
    "webp_tiled.webp",
    "webp_mixed.webp",
    "webp_noise2.webp",
    "webp_noise3.webp",
    "webp_noise4.webp",
    "webp_noise9.webp",
    "webp_noise16.webp",
    "webp_noise100.webp",
    "webp_noise256.webp",
];

#[test]
fn decoder_is_pixel_exact_against_the_reference_decoder() {
    for name in LOSSLESS {
        let bytes = fixture(name);
        let ours = webp::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (w, h, want) = reference_rgba16(&bytes);
        assert_eq!((ours.width, ours.height), (w, h), "{name}");
        let got = to_rgba16(&ours);
        // Colours behind fully transparent pixels are not part of the picture.
        let bad = got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .filter(|(a, b)| a[3] != b[3] || (a[3] != 0 && a != b))
            .count();
        assert_eq!(bad, 0, "{name}: {bad} wrong pixels");
    }
}

#[test]
fn lossy_files_decode_close_to_the_reference_and_garbage_is_an_error() {
    let bytes = fixture("webp_lossy.webp");
    let ours = webp::decode(&bytes).unwrap();
    let (w, h, want) = reference_rgba16(&bytes);
    assert_eq!((ours.width, ours.height), (w, h));
    // The reference decoder may upsample chroma differently; the pictures must still agree closely.
    let got = to_rgba16(&ours);
    let mean: f64 = got.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs() as f64 / 257.0).sum::<f64>()
        / got.len() as f64;
    assert!(mean < 3.0, "mean difference {mean:.2} levels");
    assert!(webp::decode(b"RIFF\0\0\0\0WEBP").is_err());
    assert!(webp::decode(b"not webp at all").is_err());
}

fn same_pixels(a: &[u16], b: &[u16]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x[3] != y[3] || (x[3] != 0 && x != y)).count()
}

#[test]
fn encoder_output_decodes_exactly_with_ours_and_the_reference() {
    for name in LOSSLESS {
        let img = webp::decode(&fixture(name)).unwrap();
        for level in [0u32, 3, 7, 9] {
            let bytes = webp::encode(&img, &webp::EncodeOptions { level, clean_alpha: false });
            let back = webp::decode(&bytes).unwrap_or_else(|e| panic!("{name} L{level}: {e}"));
            assert_eq!(back.data, img.data, "{name} L{level}: our decoder");
            let (w, h, got) = reference_rgba16(&bytes);
            assert_eq!((w, h), (img.width, img.height), "{name} L{level}");
            assert_eq!(same_pixels(&got, &to_rgba16(&img)), 0, "{name} L{level}: reference decoder");
        }
    }
}

#[test]
fn compression_is_competitive_with_libwebp() {
    // Not a hard requirement per file, but the total must stay close to what libwebp produced.
    let (mut ours, mut theirs) = (0usize, 0usize);
    for name in LOSSLESS {
        let bytes = fixture(name);
        let img = webp::decode(&bytes).unwrap();
        let out = webp::encode(&img, &webp::EncodeOptions { level: 9, clean_alpha: false });
        println!("{name}: libwebp {} ours {}", bytes.len(), out.len());
        ours += out.len();
        theirs += bytes.len();
    }
    println!("total: libwebp {theirs} ours {ours} ({:+.1}%)", (ours as f64 / theirs as f64 - 1.0) * 100.0);
    assert!(ours as f64 <= theirs as f64 * 1.35, "{ours} vs {theirs}");
}

#[test]
fn metadata_and_icc_survive_in_the_container() {
    let mut img = webp::decode(&fixture("webp_rgba.webp")).unwrap();
    img.meta.icc = Some(vec![7; 33]);
    img.meta.exif = Some(b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec());
    img.meta.xmp = Some(b"<x:xmpmeta/>".to_vec());
    let bytes = webp::encode(&img, &Default::default());
    let back = webp::decode(&bytes).unwrap();
    assert_eq!(back.meta.icc, img.meta.icc);
    assert_eq!(back.meta.exif, img.meta.exif);
    assert_eq!(back.meta.xmp, img.meta.xmp);
    assert!(image::load_from_memory(&bytes).is_ok(), "reference decoder accepts the extended container");
}

mod cli {
    use super::*;
    use assert_cmd::Command;
    use std::fs;
    use tempfile::tempdir;

    fn cmd() -> Command {
        Command::cargo_bin("rc").unwrap()
    }

    #[test]
    fn webp_to_webp_is_valid_and_never_larger() {
        let d = tempdir().unwrap();
        for name in ["webp_rgb.webp", "webp_mixed.webp", "webp_rgba.webp"] {
            let input = d.path().join(name);
            fs::write(&input, fixture(name)).unwrap();
            let out = d.path().join(format!("out_{name}"));
            cmd().arg(&input).arg("-o").arg(&out).arg("--keep-hidden-pixels").assert().success();
            let bytes = fs::read(&out).unwrap();
            assert!(bytes.len() <= fixture(name).len(), "{name}");
            let (_, _, got) = reference_rgba16(&bytes);
            let (_, _, want) = reference_rgba16(&fixture(name));
            assert_eq!(same_pixels(&got, &want), 0, "{name}");
        }
    }

    #[test]
    fn lossy_flag_writes_lossy_webp_and_only_then_reencodes_lossy_input() {
        let d = tempdir().unwrap();
        let input = d.path().join("in.png");
        fs::write(&input, fixture("rgb8.png")).unwrap();
        let lossy = d.path().join("lossy.webp");
        cmd().arg(&input).args(["-f", "webp", "--lossy", "-q", "60", "-o"]).arg(&lossy).assert().success();
        let bytes = fs::read(&lossy).unwrap();
        assert!(bytes.windows(4).any(|w| w == b"VP8 "));
        reference_rgba16(&bytes);
        // Without --lossy a lossy WebP is passed through untouched; with it, it is recoded.
        let plain = d.path().join("plain.webp");
        cmd().arg(&lossy).arg("-o").arg(&plain).assert().success();
        assert_eq!(fs::read(&plain).unwrap(), bytes);
        let again = d.path().join("again.webp");
        cmd().arg(&lossy).args(["--lossy", "-q", "30", "-o"]).arg(&again).assert().success();
        let recoded = fs::read(&again).unwrap();
        assert!(recoded.len() <= bytes.len());
        reference_rgba16(&recoded);
    }

    #[test]
    fn png_jpeg_and_gif_convert_to_webp() {
        for (name, lossless_source) in
            [("rgb8.png", true), ("rgba8.png", true), ("pal5.png", true), ("gif_plain.gif", true), ("rgb8.jpg", false)]
        {
            let d = tempdir().unwrap();
            let input = d.path().join(name);
            fs::write(&input, fixture(name)).unwrap();
            cmd().arg(&input).args(["-f", "webp"]).assert().success();
            let out = d.path().join(format!("{}.min.webp", input.file_stem().unwrap().to_string_lossy()));
            let bytes = fs::read(&out).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(bytes.starts_with(b"RIFF"));
            let (w, h, got) = reference_rgba16(&bytes);
            let (sw, sh, want) = reference_rgba16(&fixture(name));
            assert_eq!((w, h), (sw, sh), "{name}");
            if lossless_source {
                assert_eq!(same_pixels(&got, &want), 0, "{name}: WebP must hold the same pixels");
            }
        }
    }

    #[test]
    fn webp_converts_to_png_and_jpeg() {
        let d = tempdir().unwrap();
        let input = d.path().join("a.webp");
        fs::write(&input, fixture("webp_rgba.webp")).unwrap();
        cmd().arg(&input).args(["-f", "png"]).assert().success();
        let (_, _, got) = reference_rgba16(&fs::read(d.path().join("a.min.png")).unwrap());
        let (_, _, want) = reference_rgba16(&fixture("webp_rgba.webp"));
        assert_eq!(same_pixels(&got, &want), 0);
        cmd().arg(&input).args(["-f", "jpeg"]).assert().success();
        assert!(image::open(d.path().join("a.min.jpg")).is_ok());
    }

    #[test]
    fn lossy_webp_is_left_untouched_not_failed() {
        let d = tempdir().unwrap();
        let input = d.path().join("lossy.webp");
        fs::write(&input, fixture("webp_lossy.webp")).unwrap();
        let out = cmd().arg(&input).assert().success().get_output().clone();
        assert!(String::from_utf8_lossy(&out.stdout).contains("lossy WebP left unchanged"));
        assert_eq!(fs::read(d.path().join("lossy.min.webp")).unwrap(), fixture("webp_lossy.webp"));
        // Converting is fine: lossy files can be decoded.
        cmd().arg(&input).args(["-f", "png"]).assert().success();
        assert!(image::open(d.path().join("lossy.min.png")).is_ok());
    }

    #[test]
    fn oversized_images_are_rejected_for_webp() {
        // 16385x1 does not fit in WebP's 14-bit dimensions.
        let img = rust_compress::image::Image::new(16385, 1, rust_compress::image::ColorType::Gray, 8, vec![7; 16385])
            .unwrap();
        let png = rust_compress::png::encode::encode(&img, &Default::default());
        let out = cmd().args(["-", "-f", "webp"]).write_stdin(png).assert().failure().get_output().clone();
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("limited to"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn stdin_to_stdout_webp() {
        let out =
            cmd().args(["-", "-f", "webp"]).write_stdin(fixture("rgb8.png")).assert().success().get_output().clone();
        assert!(out.stdout.starts_with(b"RIFF") && &out.stdout[8..12] == b"WEBP");
    }
}

fn synthetic(w: u32, h: u32, color: rust_compress::image::ColorType) -> rust_compress::image::Image {
    use rust_compress::image::{ColorType, Image};
    let ch = color.channels();
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = ((x * 255 / w.max(1)) as u8, (y * 255 / h.max(1)) as u8, ((x * 7 + y * 3) % 256) as u8);
            match color {
                ColorType::Gray => data.push(r),
                ColorType::GrayAlpha => data.extend([r, if (x / 4 + y / 4) % 3 == 0 { 0 } else { g }]),
                ColorType::Rgb => data.extend([r, g, b]),
                ColorType::Rgba => {
                    data.extend([r, g, b, if (x / 5 + y / 3) % 4 == 0 { 0 } else { (x * 3 % 256) as u8 }])
                }
            }
        }
    }
    assert_eq!(data.len(), (w * h) as usize * ch);
    Image::new(w, h, color, 8, data).unwrap()
}

#[test]
fn lossy_encoder_output_is_valid_close_and_keeps_alpha_exact() {
    use rust_compress::image::ColorType;
    for (w, h) in [(1, 1), (2, 3), (17, 33), (64, 48), (130, 70)] {
        for color in [ColorType::Rgb, ColorType::Rgba, ColorType::Gray, ColorType::GrayAlpha] {
            for quality in [1u8, 50, 100] {
                let img = synthetic(w, h, color);
                let bytes = webp::encode_lossy(&img, quality, 6, true).unwrap();
                let ours = webp::decode(&bytes).unwrap_or_else(|e| panic!("{w}x{h} {color:?} q{quality}: {e}"));
                assert_eq!((ours.width, ours.height), (w, h));
                // The reference decoder must agree with ours on the decoded picture.
                let (rw, rh, want) = reference_rgba16(&bytes);
                assert_eq!((rw, rh), (w, h));
                let got = to_rgba16(&ours);
                let mean: f64 =
                    got.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs() as f64 / 257.0).sum::<f64>()
                        / got.len() as f64;
                assert!(mean < 0.5, "{w}x{h} {color:?} q{quality}: decoders differ by {mean:.2}");
                // Transparency is lossless; colour error is bounded at high quality.
                let src = to_rgba16(&img);
                let alpha_bad = got.chunks_exact(4).zip(src.chunks_exact(4)).filter(|(a, b)| a[3] != b[3]).count();
                assert_eq!(alpha_bad, 0, "{w}x{h} {color:?} q{quality}: alpha changed");
                if quality == 100 && w >= 17 {
                    let (mut sum, mut n) = (0.0, 0.0);
                    for (a, b) in got.chunks_exact(4).zip(src.chunks_exact(4)).filter(|(a, _)| a[3] != 0) {
                        for c in 0..3 {
                            sum += (a[c] as f64 - b[c] as f64).abs() / 257.0;
                            n += 1.0;
                        }
                    }
                    assert!(n == 0.0 || sum / n < 6.0, "{w}x{h} {color:?}: mean error {:.2}", sum / n);
                }
            }
        }
    }
}

#[test]
fn lossy_encoder_keeps_metadata_and_rejects_oversized_images() {
    use rust_compress::image::{ColorType, Image};
    let mut img = synthetic(40, 40, ColorType::Rgb);
    img.meta.icc = Some(vec![1, 2, 3, 4, 5]);
    img.meta.exif = Some(b"abc".to_vec());
    let back = webp::decode(&webp::encode_lossy(&img, 70, 4, true).unwrap()).unwrap();
    assert_eq!(back.meta.icc, img.meta.icc);
    assert_eq!(back.meta.exif, img.meta.exif);
    let huge = Image::new(16384, 1, ColorType::Gray, 8, vec![0; 16384]).unwrap();
    assert!(webp::encode_lossy(&huge, 70, 4, true).is_err());
}

#[test]
fn quality_trades_size_for_fidelity() {
    use rust_compress::image::ColorType;
    let img = synthetic(160, 120, ColorType::Rgb);
    let small = webp::encode_lossy(&img, 10, 6, true).unwrap().len();
    let big = webp::encode_lossy(&img, 95, 6, true).unwrap().len();
    assert!(small < big, "{small} !< {big}");
}
