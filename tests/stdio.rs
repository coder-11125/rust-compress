mod common;
use assert_cmd::Command;
use common::fixture;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rust-compress").unwrap()
}

#[test]
fn stdin_to_stdout_is_a_valid_smaller_image_and_keeps_stdout_clean() {
    let input = fixture("rgb8.png");
    let out = cmd().arg("-").write_stdin(input.clone()).assert().success().get_output().clone();
    // stdout is exactly one PNG: nothing from the progress/summary lines leaked into it.
    assert!(out.stdout.starts_with(&rust_compress::png::SIGNATURE));
    assert!(out.stdout.len() <= input.len());
    assert!(image::load_from_memory(&out.stdout).is_ok());
    assert!(String::from_utf8_lossy(&out.stderr).contains("1 file(s)"));
}

#[test]
fn file_to_stdout_with_dash_output() {
    let d = tempdir().unwrap();
    let p = d.path().join("a.jpg");
    fs::write(&p, fixture("rgb8_444.jpg")).unwrap();
    let out = cmd().arg(&p).args(["-o", "-"]).assert().success().get_output().clone();
    assert!(out.stdout.starts_with(&[0xFF, 0xD8]));
    assert!(image::load_from_memory(&out.stdout).is_ok());
    // No sibling file was written.
    assert!(!d.path().join("a.min.jpg").exists());
}

#[test]
fn stdin_to_a_file() {
    let d = tempdir().unwrap();
    let out = d.path().join("o.png");
    cmd().arg("-").arg("-o").arg(&out).write_stdin(fixture("rgb8.png")).assert().success();
    assert!(image::open(&out).is_ok());
}

#[test]
fn stdin_with_format_conversion() {
    let out = cmd().args(["-", "-f", "jpeg"]).write_stdin(fixture("rgb8.png")).assert().success().get_output().clone();
    assert!(out.stdout.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn stdin_garbage_fails_without_writing_stdout() {
    let out = cmd().arg("-").write_stdin(&b"not an image"[..]).assert().failure().get_output().clone();
    assert!(out.stdout.is_empty());
}

#[test]
fn stream_misuse_is_rejected() {
    let d = tempdir().unwrap();
    let a = d.path().join("a.png");
    fs::write(&a, fixture("rgb8.png")).unwrap();
    // Two inputs cannot share one stdout.
    cmd().arg(&a).arg(&a).args(["-o", "-"]).assert().failure();
    cmd().arg("-").arg(&a).assert().failure();
    cmd().arg("-").arg("--in-place").write_stdin(fixture("rgb8.png")).assert().failure();
    cmd().arg(d.path()).args(["-o", "-"]).assert().failure();
}

#[test]
fn no_gain_still_streams_the_original_bytes() {
    // A 1x1-ish already-minimal file: whatever happens, stdout must hold a valid image.
    let input = fixture("gray1.png");
    let out = cmd().arg("-").write_stdin(input.clone()).assert().success().get_output().clone();
    assert!(out.stdout.len() <= input.len());
    assert!(image::load_from_memory(&out.stdout).is_ok());
}
