//! AVIF through the command line: conversion in both directions and the leave-it-alone policy.
mod common;
use assert_cmd::Command;
use common::fixture;
use rust_compress::{avif, png};
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rust-compress").unwrap()
}

#[test]
fn png_to_avif_and_back() {
    let d = tempdir().unwrap();
    let input = d.path().join("a.png");
    fs::write(&input, fixture("rgb8.png")).unwrap();
    cmd().args(["-f", "avif", "-q", "85", "--quiet"]).arg(&input).assert().success();
    let out = d.path().join("a.min.avif");
    let bytes = fs::read(&out).unwrap();
    assert!(avif::is_avif(&bytes));
    let original = png::decode::decode(&fixture("rgb8.png")).unwrap();
    let decoded = avif::decode(&bytes).unwrap();
    assert_eq!((decoded.width, decoded.height), (original.width, original.height));
    // And the CLI can read it back as PNG.
    let back = d.path().join("back.png");
    cmd().args(["-f", "png", "--quiet", "-o"]).arg(&back).arg(&out).assert().success();
    let roundtrip = png::decode::decode(&fs::read(&back).unwrap()).unwrap();
    assert_eq!((roundtrip.width, roundtrip.height), (original.width, original.height));
}

#[test]
fn avif_is_left_alone_unless_lossy_is_given() {
    let d = tempdir().unwrap();
    let input = d.path().join("p.avif");
    fs::write(&input, fixture("avif_rgb.avif")).unwrap();
    let out = d.path().join("same.avif");
    cmd().arg(&input).arg("-o").arg(&out).arg("--quiet").assert().success();
    assert_eq!(fs::read(&out).unwrap(), fixture("avif_rgb.avif"));
    // Explicit --lossy re-encodes (and still never grows the file).
    let out2 = d.path().join("again.avif");
    cmd().arg(&input).args(["--lossy", "-q", "30", "--quiet", "-o"]).arg(&out2).assert().success();
    let bytes = fs::read(&out2).unwrap();
    assert!(avif::decode(&bytes).is_ok());
    assert!(bytes.len() <= fixture("avif_rgb.avif").len());
}

#[test]
fn alpha_and_gray_convert_through_the_cli() {
    let d = tempdir().unwrap();
    for (fix, channels) in [("rgba8.png", 4usize), ("gray8.png", 1)] {
        let input = d.path().join(fix);
        fs::write(&input, fixture(fix)).unwrap();
        let out = d.path().join(format!("{fix}.avif"));
        cmd().args(["-f", "avif", "--quiet", "-o"]).arg(&out).arg(&input).assert().success();
        let img = avif::decode(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(img.color.channels(), channels, "{fix}");
    }
}

#[test]
fn avif_via_stdio() {
    let assert =
        cmd().args(["-f", "avif", "-q", "60", "-", "-o", "-"]).write_stdin(fixture("rgb8.png")).assert().success();
    let bytes = assert.get_output().stdout.clone();
    assert!(avif::is_avif(&bytes), "stdout must carry the AVIF file");
    assert!(avif::decode(&bytes).is_ok());
}

#[test]
fn resize_applies_to_avif() {
    let d = tempdir().unwrap();
    let input = d.path().join("in.png");
    fs::write(&input, fixture("rgb8.png")).unwrap();
    let out = d.path().join("small.avif");
    cmd().args(["-f", "avif", "--resize", "50%", "--quiet", "-o"]).arg(&out).arg(&input).assert().success();
    let original = png::decode::decode(&fixture("rgb8.png")).unwrap();
    let img = avif::decode(&fs::read(&out).unwrap()).unwrap();
    assert!(img.width < original.width && img.height < original.height);
}
