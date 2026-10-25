use assert_cmd::Command;
use common::fixture;
use rc_core::exif;
use rc_core::util::checksum::crc32;
use rc_jpeg as jpeg;
use rc_png as png;
use rc_testutil as common;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

/// Little-endian EXIF: IFD0 {orientation, ExifIFD}, ExifIFD {PixelX, PixelY}, then a fake thumbnail.
fn exif_block(orientation: u16, w: u32, h: u32) -> Vec<u8> {
    let mut b = b"II*\0".to_vec();
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    for (tag, ty, val) in [(0x0112u16, 3u16, orientation as u32), (0x8769, 4, 38)] {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    }
    b.extend_from_slice(&68u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    for (tag, v) in [(0xA002u16, w), (0xA003, h)] {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&[0xAB; 300]); // "thumbnail"
    b
}

const XMP: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF/></x:xmpmeta>";

fn jpeg_with_meta(orientation: u16) -> Vec<u8> {
    let base = fixture("rgb8_444.jpg");
    let img = jpeg::decode::decode(&base).unwrap();
    let mut out = vec![0xFF, 0xD8];
    let mut seg = |marker: u8, body: &[u8]| {
        out.extend_from_slice(&[0xFF, marker]);
        out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(body);
    };
    seg(0xE1, &[&b"Exif\0\0"[..], &exif_block(orientation, img.width, img.height)].concat());
    seg(0xE1, &[&b"http://ns.adobe.com/xap/1.0/\0"[..], XMP].concat());
    seg(0xFE, b"shot on a potato");
    out.extend_from_slice(&base[2..]);
    out
}

fn png_chunk(ty: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut c = (body.len() as u32).to_be_bytes().to_vec();
    let mut tb = ty.to_vec();
    tb.extend_from_slice(body);
    c.extend_from_slice(&tb);
    c.extend_from_slice(&crc32(&tb).to_be_bytes());
    c
}

fn png_with_meta() -> Vec<u8> {
    let base = fixture("rgb8.png");
    let ihdr_end = 8 + 12 + 13;
    let mut out = base[..ihdr_end].to_vec();
    out.extend(png_chunk(b"eXIf", &exif_block(1, 1, 1)));
    out.extend(png_chunk(b"tEXt", b"Comment\0hello"));
    let mut itxt = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
    itxt.extend_from_slice(XMP);
    out.extend(png_chunk(b"iTXt", &itxt));
    out.extend(png_chunk(b"pHYs", &[0, 0, 0x0B, 0x13, 0, 0, 0x0B, 0x13, 1]));
    out.extend(png_chunk(b"tIME", &[0x07, 0xE9, 1, 2, 3, 4, 5]));
    out.extend_from_slice(&base[ihdr_end..]);
    out
}

fn run(input: &[u8], args: &[&str], ext: &str) -> Vec<u8> {
    let d = tempdir().unwrap();
    let inp = d.path().join(format!("in.{ext}"));
    fs::write(&inp, input).unwrap();
    let out = d.path().join("out.bin");
    cmd().arg(&inp).arg("-o").arg(&out).args(args).assert().success();
    fs::read(&out).unwrap()
}

fn jpeg_meta(bytes: &[u8]) -> rc_core::image::Metadata {
    jpeg::decode::parse(bytes).unwrap().into_image().unwrap().meta
}

#[test]
fn by_default_descriptive_metadata_is_stripped() {
    let j = jpeg_meta(&run(&jpeg_with_meta(1), &[], "jpg"));
    assert!(!j.has_descriptive());
    let p = png::decode::decode(&run(&png_with_meta(), &[], "png")).unwrap().meta;
    assert!(!p.has_descriptive());
}

#[test]
fn jpeg_keeps_exif_xmp_and_comment() {
    let out = run(&jpeg_with_meta(1), &["--keep-metadata"], "jpg");
    let m = jpeg_meta(&out);
    assert_eq!(exif::orientation(m.exif.as_ref().unwrap()), Some(1));
    assert_eq!(m.xmp.as_deref(), Some(XMP));
    assert_eq!(m.jpeg_extra, vec![(0xFE, b"shot on a potato".to_vec())]);
    // The thumbnail is still valid when the pixels did not change shape.
    assert!(m.exif.as_ref().unwrap().ends_with(&[0xAB; 300]));
    assert!(image::load_from_memory(&out).is_ok());
}

#[test]
fn png_keeps_exif_xmp_text_density_and_time() {
    let out = run(&png_with_meta(), &["--keep-metadata"], "png");
    let m = png::decode::decode(&out).unwrap().meta;
    assert!(m.exif.is_some());
    assert_eq!(m.xmp.as_deref(), Some(XMP));
    let kinds: Vec<_> = m.png_extra.iter().map(|(t, _)| String::from_utf8_lossy(t).into_owned()).collect();
    for k in ["tEXt", "pHYs", "tIME"] {
        assert!(kinds.contains(&k.to_string()), "{k} missing from {kinds:?}");
    }
    assert!(image::load_from_memory(&out).is_ok());
}

#[test]
fn exif_and_xmp_survive_format_conversion_both_ways() {
    let to_jpeg = run(&png_with_meta(), &["--keep-metadata", "-f", "jpeg"], "png");
    let m = jpeg_meta(&to_jpeg);
    assert!(m.exif.is_some() && m.xmp.as_deref() == Some(XMP));
    assert!(m.jpeg_extra.is_empty(), "PNG-only chunks must not leak into a JPEG");
    let to_png = run(&jpeg_with_meta(1), &["--keep-metadata", "-f", "png"], "jpg");
    let m = png::decode::decode(&to_png).unwrap().meta;
    assert!(m.exif.is_some() && m.xmp.as_deref() == Some(XMP));
    assert!(m.png_extra.is_empty());
}

#[test]
fn baked_orientation_resets_the_tag_and_drops_the_stale_thumbnail() {
    let src = jpeg_with_meta(6); // rotate 90 degrees: width and height swap
    let before = jpeg::decode::decode(&src).unwrap();
    let out = run(&src, &["--keep-metadata"], "jpg");
    let after = jpeg::decode::decode(&out).unwrap();
    assert_eq!((after.width, after.height), (before.height, before.width), "pixels were rotated");
    let exif_block = jpeg_meta(&out).exif.unwrap();
    assert_eq!(exif::orientation(&exif_block), Some(1), "tag must not rotate the image a second time");
    assert!(!exif_block.ends_with(&[0xAB; 300]), "stale thumbnail kept");
    // Recorded dimensions follow the rotated pixels.
    assert_eq!(&exif_block[38 + 2 + 8..38 + 2 + 12], &after.width.to_le_bytes());
}

#[test]
fn strip_icc_keeps_other_metadata_when_asked() {
    let out = run(&jpeg_with_meta(1), &["--keep-metadata", "--strip-icc"], "jpg");
    let m = jpeg_meta(&out);
    assert!(m.icc.is_none() && m.exif.is_some());
}

#[test]
fn icc_profile_is_kept_by_default() {
    let m = jpeg_meta(&run(&fixture("icc.jpg"), &[], "jpg"));
    assert!(m.icc.is_some());
}
