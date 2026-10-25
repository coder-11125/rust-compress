use assert_cmd::Command;
use common::fixture;
use rc_testutil as common;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

fn put(dir: &std::path::Path, name: &str, fixture_name: &str) -> std::path::PathBuf {
    let p = dir.join(name);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, fixture(fixture_name)).unwrap();
    p
}

#[test]
fn single_file_default_output_is_min_sibling() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.jpg", "rgb8_444.jpg");
    cmd().arg(&input).assert().success();
    let out = d.path().join("photo.min.jpg");
    assert!(out.exists());
    assert!(fs::metadata(&out).unwrap().len() <= fs::metadata(&input).unwrap().len());
    // Original untouched.
    assert_eq!(fs::read(&input).unwrap(), fixture("rgb8_444.jpg"));
}

#[test]
fn explicit_output_file_and_refuse_overwrite() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "a.png", "rgb8.png");
    let out = d.path().join("result.png");
    cmd().arg(&input).arg("-o").arg(&out).assert().success();
    assert!(out.exists());
    cmd().arg(&input).arg("-o").arg(&out).assert().failure();
    cmd().arg(&input).arg("-o").arg(&out).arg("--force").assert().success();
}

#[test]
fn directory_batch_recursive_mirrors_tree() {
    let d = tempdir().unwrap();
    put(d.path(), "in/a.png", "rgb8.png");
    put(d.path(), "in/sub/b.jpg", "rgb8.jpg");
    put(d.path(), "in/sub/deeper/c.png", "pal5.png");
    fs::write(d.path().join("in/notes.txt"), "ignore me").unwrap();
    let out = d.path().join("out");
    cmd().arg(d.path().join("in")).arg("-r").arg("-o").arg(&out).assert().success();
    assert!(out.join("a.png").exists());
    assert!(out.join("sub/b.jpg").exists());
    assert!(out.join("sub/deeper/c.png").exists());
    assert!(!out.join("notes.txt").exists());
}

#[test]
fn non_recursive_directory_only_takes_top_level() {
    let d = tempdir().unwrap();
    put(d.path(), "in/a.png", "rgb8.png");
    put(d.path(), "in/sub/b.png", "rgb8.png");
    let out = d.path().join("out");
    cmd().arg(d.path().join("in")).arg("-o").arg(&out).assert().success();
    assert!(out.join("a.png").exists());
    assert!(!out.join("sub/b.png").exists());
}

#[test]
fn format_conversion_both_ways() {
    let d = tempdir().unwrap();
    let png = put(d.path(), "x.png", "rgba8.png");
    let jpg = put(d.path(), "y.jpg", "rgb8.jpg");
    cmd().arg(&png).args(["-f", "jpeg", "-o"]).arg(d.path().join("x_conv.jpg")).assert().success();
    cmd().arg(&jpg).args(["-f", "png", "-o"]).arg(d.path().join("y_conv.png")).assert().success();
    let a = image::open(d.path().join("x_conv.jpg")).unwrap();
    let b = image::open(d.path().join("y_conv.png")).unwrap();
    assert_eq!((a.width(), a.height()), (67, 45));
    assert_eq!((b.width(), b.height()), (67, 45));
}

#[test]
fn dry_run_writes_nothing() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "p.jpg", "rgb8.jpg");
    cmd().arg(&input).arg("--dry-run").assert().success();
    assert!(!d.path().join("p.min.jpg").exists());
}

#[test]
fn in_place_never_grows_the_file() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "p.png", "gray2.png"); // already tiny
    let before = fs::metadata(&input).unwrap().len();
    cmd().arg(&input).arg("--in-place").assert().success();
    assert!(fs::metadata(&input).unwrap().len() <= before);
    let after = image::open(&input).unwrap();
    assert_eq!((after.width(), after.height()), (67, 45));
}

#[test]
fn corrupt_file_fails_but_others_still_processed() {
    let d = tempdir().unwrap();
    let good = put(d.path(), "good.png", "rgb8.png");
    let bad = d.path().join("bad.png");
    fs::write(&bad, b"this is not a png at all").unwrap();
    let out = cmd().arg(&bad).arg(&good).assert().failure();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(stderr.contains("bad.png") && stderr.contains("FAILED"), "{stderr}");
    assert!(stdout.contains("1 failed"), "{stdout}");
    assert!(d.path().join("good.min.png").exists());
}

#[test]
fn lossy_png_reduces_colors() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.png", "rgb8.png");
    let out = d.path().join("q.png");
    cmd().arg(&input).args(["--lossy", "--colors", "16", "-o"]).arg(&out).assert().success();
    let img = image::open(&out).unwrap().to_rgb8();
    let distinct: std::collections::HashSet<_> = img.pixels().map(|p| p.0).collect();
    assert!(distinct.len() <= 16, "{} colors", distinct.len());
    assert!(fs::metadata(&out).unwrap().len() < fs::metadata(&input).unwrap().len());
}

#[test]
fn missing_input_is_an_error() {
    cmd().arg("/definitely/not/here.png").assert().failure();
}

#[test]
fn quality_flag_validated() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "p.jpg", "rgb8.jpg");
    cmd().arg(&input).args(["-q", "0"]).assert().failure();
    cmd().arg(&input).args(["-q", "101"]).assert().failure();
}

#[test]
fn target_ssim_picks_a_quality_and_reports_it() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.jpg", "rgb8_444.jpg");
    let out = cmd().arg(&input).args(["--target-ssim", "0.9", "-o"]).arg(d.path().join("t.jpg")).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(stdout.contains("ssim"), "{stdout}");
    let img = image::open(d.path().join("t.jpg")).unwrap();
    assert_eq!((img.width(), img.height()), (67, 45));
    cmd().arg(&input).args(["--target-ssim", "1.5"]).assert().failure();
    cmd().arg(&input).args(["--target-ssim", "0"]).assert().failure();
}
