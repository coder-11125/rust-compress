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
fn include_and_exclude_globs_filter_directory_walks() {
    let d = tempdir().unwrap();
    put(d.path(), "in/a.png", "rgb8.png");
    put(d.path(), "in/b.jpg", "rgb8.jpg");
    put(d.path(), "in/sub/c.png", "pal5.png");
    fs::write(d.path().join("in/notes.txt"), "ignore me").unwrap();
    let out = d.path().join("out");
    cmd()
        .arg(d.path().join("in"))
        .arg("-r")
        .arg("-o")
        .arg(&out)
        .args(["--include", "**/*.png", "--exclude", "sub/*"])
        .assert()
        .success();
    assert!(out.join("a.png").exists());
    assert!(!out.join("sub/c.png").exists(), "excluded by sub/*");
    assert!(!out.join("b.jpg").exists(), "not matched by --include");
    assert!(!out.join("notes.txt").exists());
}

#[test]
fn include_can_match_paths_relative_to_each_input_directory() {
    let d = tempdir().unwrap();
    put(d.path(), "in/a.png", "rgb8.png");
    put(d.path(), "in/sub/c.png", "pal5.png");
    let out = d.path().join("out");
    cmd().arg(d.path().join("in")).arg("-r").arg("-o").arg(&out).args(["--include", "sub/*.png"]).assert().success();
    assert!(!out.join("a.png").exists());
    assert!(out.join("sub/c.png").exists());
}

#[test]
fn min_suffix_names_default_outputs_and_marks_them_for_walks() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.jpg", "rgb8_444.jpg");
    cmd().arg(&input).args(["--min-suffix", "optimized"]).assert().success();
    assert!(d.path().join("photo.optimized.jpg").exists());

    let dir = tempdir().unwrap();
    put(dir.path(), "in/a.jpg", "rgb8.jpg");
    put(dir.path(), "in/a.optimized.jpg", "rgb8.jpg"); // already an output name: skipped
    let out = dir.path().join("out");
    cmd().arg(dir.path().join("in")).arg("-o").arg(&out).args(["--min-suffix", "optimized"]).assert().success();
    assert!(out.join("a.jpg").exists());
    assert!(!out.join("a.optimized.jpg").exists());
}

#[test]
fn name_template_controls_default_output_paths() {
    let d = tempdir().unwrap();
    put(d.path(), "in/a.png", "rgb8.png");
    put(d.path(), "in/sub/b.jpg", "rgb8.jpg");
    cmd()
        .arg(d.path().join("in"))
        .arg("-r")
        .args(["--name-template", "{root}/tpl/{rel_stem}.{suffix}.{ext}"])
        .assert()
        .success();
    assert!(d.path().join("in/tpl/a.min.png").exists());
    assert!(d.path().join("in/tpl/sub/b.min.jpg").exists());
    assert!(!d.path().join("in/a.min.png").exists());
    assert!(!d.path().join("in/sub/b.min.jpg").exists());
}

#[test]
fn name_template_rejects_unknown_placeholders() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "a.png", "rgb8.png");
    let res = cmd().arg(&input).args(["--name-template", "{bogus}"]).assert().failure();
    let stderr = String::from_utf8_lossy(&res.get_output().stderr);
    assert!(stderr.contains("unknown placeholder"), "{stderr}");
}

#[test]
fn json_report_replaces_text_output() {
    let d = tempdir().unwrap();
    let good = put(d.path(), "good.png", "rgb8.png");
    let res = cmd().arg(&good).arg("--json").assert().success();
    let stdout = String::from_utf8_lossy(&res.get_output().stdout);
    assert!(stdout.trim_start().starts_with('{'), "{stdout}");
    assert!(stdout.contains("\"files\":["), "{stdout}");
    assert!(stdout.contains("good.png"), "{stdout}");
    assert!(stdout.contains("\"status\":"), "{stdout}");
    assert!(stdout.contains("\"summary\":{"), "{stdout}");
    assert!(!stdout.contains("file(s):"), "{stdout}");

    let bad = d.path().join("bad.png");
    fs::write(&bad, b"not a png").unwrap();
    let res = cmd().arg(&bad).arg("--json").assert().failure();
    let stdout = String::from_utf8_lossy(&res.get_output().stdout);
    assert!(stdout.contains("\"status\":\"failed\""), "{stdout}");
    assert!(stdout.contains("\"failed\":1"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn in_place_preserves_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let d = tempdir().unwrap();
    let input = put(d.path(), "p.jpg", "rgb8.jpg");
    fs::set_permissions(&input, fs::Permissions::from_mode(0o640)).unwrap();
    cmd().arg(&input).arg("--in-place").assert().success();
    assert_eq!(fs::metadata(&input).unwrap().permissions().mode() & 0o777, 0o640);
}

#[cfg(unix)]
#[test]
fn in_place_follows_a_symlink_and_keeps_the_link() {
    let d = tempdir().unwrap();
    let real = put(d.path(), "real/a.png", "rgb8.png");
    let link = d.path().join("link.png");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let before = fs::metadata(&real).unwrap().len();
    cmd().arg(&link).arg("--in-place").assert().success();
    assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink(), "the link must stay a link");
    assert!(fs::metadata(&real).unwrap().len() < before, "the target is what gets optimized");
}

#[cfg(unix)]
#[test]
fn in_place_refuses_a_hard_linked_file() {
    let d = tempdir().unwrap();
    let a = put(d.path(), "a.png", "rgb8.png");
    let b = d.path().join("b.png");
    fs::hard_link(&a, &b).unwrap();
    let out = cmd().arg(&a).arg("--in-place").assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(stderr.contains("hard links"), "{stderr}");
    assert_eq!(fs::read(&a).unwrap(), fixture("rgb8.png"));
    assert_eq!(fs::read(&b).unwrap(), fixture("rgb8.png"));
}

// Windows forbids control characters in file names, so such a file cannot even be created there.
#[cfg(unix)]
#[test]
fn control_characters_in_file_names_are_not_printed_raw() {
    let d = tempdir().unwrap();
    let name = "x\u{1b}[31mRED.png";
    let input = d.path().join(name);
    fs::write(&input, fixture("rgb8.png")).unwrap();
    let out = cmd().arg(&input).arg("--dry-run").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(!stdout.contains('\u{1b}'), "{stdout:?}");
    assert!(stdout.contains("\\x1b[31mRED.png"), "{stdout:?}");
}

#[test]
fn out_of_range_dither_and_jobs_are_rejected() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "a.png", "rgb8.png");
    for bad in ["NaN", "inf", "5", "-3"] {
        cmd().arg(&input).arg("--lossy").arg(format!("--dither={bad}")).arg("--dry-run").assert().code(2);
    }
    cmd().arg(&input).args(["-j", "100000", "--dry-run"]).assert().code(2);
    cmd().arg(&input).args(["-j", "0", "--dry-run"]).assert().code(2);
    cmd().arg(&input).args(["--lossy", "--dither", "0.25", "-j", "2", "--dry-run"]).assert().success();
}

#[test]
fn a_format_change_that_grows_the_file_is_reported_as_larger() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.jpg", "rgb8.jpg");
    let out = cmd().arg(&input).args(["-f", "png", "--json"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(stdout.contains("\"status\":\"larger\""), "{stdout}");
    assert!(stdout.contains("\"larger\":1"), "{stdout}");
    let text = cmd().arg(&input).args(["-f", "png", "--force"]).assert().success();
    let text = String::from_utf8_lossy(&text.get_output().stdout).to_string();
    assert!(text.contains("1 larger"), "{text}");
}

#[cfg(unix)]
#[test]
fn pipes_and_devices_are_not_reported_as_missing() {
    let out = cmd().arg("/dev/null").assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(stderr.contains("not a regular file"), "{stderr}");
}

#[test]
fn max_pixels_refuses_bigger_images_and_accepts_the_rest() {
    let d = tempdir().unwrap();
    let png = put(d.path(), "a.png", "rgb8.png"); // 67 x 45 = 3015 pixels
    let jpg = put(d.path(), "b.jpg", "rgb8.jpg");
    for input in [&png, &jpg] {
        let out = cmd().arg(input).args(["--max-pixels", "3000", "--dry-run"]).assert().failure();
        let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
        assert!(stderr.contains("FAILED"), "{stderr}");
        cmd().arg(input).args(["--max-pixels", "3015", "--dry-run"]).assert().success();
    }
    cmd().arg(&png).args(["--max-pixels", "0"]).assert().code(2);
    cmd().arg(&png).args(["--max-pixels", "150000001"]).assert().code(2);
}

#[test]
fn a_tiny_memory_budget_runs_a_batch_one_image_at_a_time_without_stalling() {
    let d = tempdir().unwrap();
    for i in 0..12 {
        put(d.path(), &format!("in/{i}.png"), "rgb8.png");
        put(d.path(), &format!("in/{i}.jpg"), "rgb8.jpg");
    }
    let out = cmd()
        .arg(d.path().join("in"))
        .args(["--max-memory", "1", "--json", "-o"])
        .arg(d.path().join("out"))
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(stdout.contains("\"files\":24") && stdout.contains("\"failed\":0"), "{stdout}");
    cmd().arg(d.path().join("in")).args(["--max-memory", "0"]).assert().code(2);
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

#[test]
fn lossy_webp_honors_target_size() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.png", "rgb8.png");
    let limit = fs::metadata(&input).unwrap().len();
    let out = cmd()
        .arg(&input)
        .args(["-f", "webp", "--lossy", "--target-size", &limit.to_string(), "-o"])
        .arg(d.path().join("t.webp"))
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(stdout.contains("size target"), "{stdout}");
    assert!(fs::metadata(d.path().join("t.webp")).unwrap().len() <= limit);
}

#[test]
fn lossy_formats_accept_psnr_and_ms_ssim_targets() {
    let d = tempdir().unwrap();
    let png = put(d.path(), "photo.png", "rgb8.png");
    let webp = cmd()
        .arg(&png)
        .args(["-f", "webp", "--lossy", "--target-psnr", "35", "-o"])
        .arg(d.path().join("t.webp"))
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&webp.get_output().stdout).to_string();
    assert!(stdout.contains("psnr"), "{stdout}");

    let png = cmd()
        .arg(&png)
        .args(["--lossy", "--colors", "64", "--target-ms-ssim", "0.9", "-o"])
        .arg(d.path().join("t.png"))
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&png.get_output().stdout).to_string();
    assert!(stdout.contains("ms-ssim"), "{stdout}");
}

#[test]
fn jpeg_rejects_non_ssim_quality_targets() {
    let d = tempdir().unwrap();
    let input = put(d.path(), "photo.jpg", "rgb8.jpg");
    let out = cmd().arg(&input).args(["--target-size", "1000"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(stderr.contains("target-size"), "{stderr}");
}

#[test]
fn running_without_arguments_shows_the_full_usage_including_update() {
    let out = cmd().assert().failure().code(2).get_output().clone();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("required arguments were not provided"), "{err}");
    assert!(err.contains("rc [OPTIONS] <INPUT>..."), "{err}");
    assert!(err.contains("rc update [OPTIONS]"), "{err}");
    // --help and the error agree about how the tool is used.
    let help = cmd().arg("--help").assert().success().get_output().clone();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("rc [OPTIONS] <INPUT>...") && help.contains("rc update [OPTIONS]"), "{help}");
}
