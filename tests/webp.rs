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
fn lossy_and_garbage_are_clean_errors() {
    let e = webp::decode(&fixture("webp_lossy.webp")).unwrap_err().to_string();
    assert!(e.contains("lossy"), "{e}");
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
        Command::cargo_bin("rust-compress").unwrap()
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
        // Converting it is a different matter: there is no lossy decoder.
        cmd().arg(&input).args(["-f", "png"]).assert().failure();
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
