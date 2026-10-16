mod common;
use assert_cmd::Command;
use common::fixture;
use rust_compress::{jpeg, png};
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rust-compress").unwrap()
}

fn run(input: &[u8], ext: &str, args: &[&str]) -> (Vec<u8>, String) {
    let d = tempdir().unwrap();
    let inp = d.path().join(format!("in.{ext}"));
    fs::write(&inp, input).unwrap();
    let out = d.path().join("out.bin");
    let assert = cmd().arg(&inp).arg("-o").arg(&out).args(args).assert().success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    (fs::read(&out).unwrap(), stdout)
}

fn dims(bytes: &[u8]) -> (u32, u32) {
    let im = image::load_from_memory(bytes).unwrap();
    (im.width(), im.height())
}

#[test]
fn png_is_shrunk_to_fit_and_decodes() {
    let src = fixture("rgb8.png");
    let (w, h) = dims(&src);
    let (out, log) = run(&src, "png", &["--resize", &format!("{}x", w / 2)]);
    let (nw, nh) = dims(&out);
    assert_eq!(nw, w / 2);
    assert!(nh.abs_diff(h / 2) <= 1);
    assert!(log.contains("resized"), "{log}");
}

#[test]
fn jpeg_is_shrunk_through_the_pixel_route() {
    let src = fixture("rgb8_444.jpg");
    let (w, h) = dims(&src);
    let (out, _) = run(&src, "jpg", &["--resize", "50%"]);
    let (nw, nh) = dims(&out);
    assert!(nw.abs_diff(w / 2) <= 1 && nh.abs_diff(h / 2) <= 1, "{nw}x{nh} from {w}x{h}");
}

#[test]
fn images_are_never_enlarged() {
    let src = fixture("rgb8.jpg");
    let (out, log) = run(&src, "jpg", &["--resize", "100000x100000"]);
    assert_eq!(dims(&out), dims(&src));
    assert!(!log.contains("resized"));
}

#[test]
fn resizing_is_honoured_even_if_the_result_is_not_smaller() {
    // A tiny image resized to 1x1 can be larger than the original in header overhead; the
    // never-grow fallback must not silently hand back the original size.
    let src = fixture("gray1.png");
    let (w, _) = dims(&src);
    let (out, _) = run(&src, "png", &["--resize", &format!("{}x", (w / 2).max(1))]);
    let (nw, _) = dims(&out);
    assert!(nw < w || w == 1);
}

#[test]
fn bad_specs_are_rejected() {
    for bad in ["", "x", "10", "0x5", "150%", "abc"] {
        cmd().arg("whatever.png").args(["--resize", bad]).assert().failure();
    }
}

#[test]
fn resize_updates_exif_dimensions_and_drops_the_thumbnail() {
    // Reuse the metadata test's idea: a JPEG carrying EXIF with recorded dimensions and a thumbnail.
    let base = fixture("rgb8_444.jpg");
    let img = jpeg::decode::decode(&base).unwrap();
    let mut tiff = b"II*\0".to_vec();
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&1u16.to_le_bytes());
    tiff.extend_from_slice(&[0x69, 0x87, 4, 0, 1, 0, 0, 0, 26, 0, 0, 0]); // ExifIFD pointer -> 26
    tiff.extend_from_slice(&68u32.to_le_bytes()); // IFD1 at 68 (after the Exif IFD)
    assert_eq!(tiff.len(), 26);
    tiff.extend_from_slice(&2u16.to_le_bytes());
    for (tag, v) in [(0xA002u16, img.width), (0xA003, img.height)] {
        tiff.extend_from_slice(&tag.to_le_bytes());
        tiff.extend_from_slice(&4u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&v.to_le_bytes());
    }
    tiff.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(tiff.len(), 56);
    // Fix the IFD1 pointer to the real end of the Exif IFD (56), then add thumbnail bytes.
    tiff[22..26].copy_from_slice(&56u32.to_le_bytes());
    tiff.extend_from_slice(&0u16.to_le_bytes());
    tiff.extend_from_slice(&0u32.to_le_bytes());
    tiff.extend_from_slice(&[0xCD; 200]);
    let mut src = vec![0xFF, 0xD8, 0xFF, 0xE1];
    let body = [&b"Exif\0\0"[..], &tiff].concat();
    src.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    src.extend_from_slice(&body);
    src.extend_from_slice(&base[2..]);

    let (out, _) = run(&src, "jpg", &["--resize", "50%", "--keep-metadata"]);
    let after = jpeg::decode::parse(&out).unwrap().into_image().unwrap();
    let block = after.meta.exif.expect("exif kept");
    assert!(!block.ends_with(&[0xCD; 200]), "stale thumbnail kept");
    let want_w = after.width.to_le_bytes();
    assert_eq!(&block[28 + 8..28 + 12], &want_w, "recorded width follows the resize");
}

#[test]
fn resize_with_format_conversion_and_lossy_palette() {
    let src = fixture("rgb8.png");
    let (out, _) = run(&src, "png", &["--resize", "50%", "-f", "jpeg"]);
    assert!(out.starts_with(&[0xFF, 0xD8]));
    let (out, _) = run(&src, "png", &["--resize", "50%", "--lossy", "--colors", "16"]);
    let img = png::decode::decode(&out).unwrap();
    assert!(img.width < dims(&src).0);
}
