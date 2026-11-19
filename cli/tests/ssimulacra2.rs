//! SSIMULACRA 2 against the reference values in `fixtures/ssim2/expected.csv`: 54 pairs, sizes 8x8 to 513x385 (so every
//! scale count and the weight shift for small images), blur, noise, JPEG and WebP damage, grayscale, flat and
//! identical images. The expected scores come from the py-ssimulacra2 port; see `fixtures/ssim2/README.md`.
use rc_png::decode::decode;
use rc_testutil::fixture;

/// Observed worst difference is about 7e-6 (the planes are held as f32 between scales); this leaves room.
const TOLERANCE: f64 = 1e-4;

#[test]
fn scores_match_the_reference_on_every_fixture_pair() {
    let csv = String::from_utf8(fixture("ssim2/expected.csv")).unwrap();
    let mut checked = 0;
    let mut failures = Vec::new();
    for line in csv.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.trim().split(',').collect();
        assert_eq!(f.len(), 4, "malformed line: {line}");
        let reference = decode(&fixture(&format!("ssim2/{}", f[1]))).unwrap();
        let distorted = decode(&fixture(&format!("ssim2/{}", f[2]))).unwrap();
        let want: f64 = f[3].parse().unwrap();
        let got = rc_core::ssimulacra2::ssimulacra2(&reference, &distorted).unwrap();
        if !got.is_finite() || (got - want).abs() > TOLERANCE {
            failures.push(format!("{}: reference {want:.9}, ours {got:.9}, difference {:+.3e}", f[0], got - want));
        }
        checked += 1;
    }
    assert!(checked >= 50, "only {checked} pairs found");
    assert!(failures.is_empty(), "{} of {checked} pairs differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn identical_images_score_exactly_100() {
    let csv = String::from_utf8(fixture("ssim2/expected.csv")).unwrap();
    let line = csv.lines().find(|l| l.starts_with("identical_96x72,")).expect("identical pair in the fixtures");
    let f: Vec<&str> = line.split(',').collect();
    let img = decode(&fixture(&format!("ssim2/{}", f[1]))).unwrap();
    assert_eq!(rc_core::ssimulacra2::ssimulacra2(&img, &img).unwrap(), 100.0);
}

mod target {
    //! `--target-ssimulacra2` through the real binary.
    use assert_cmd::Command;
    use rc_testutil::fixture;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn cmd() -> Command {
        Command::cargo_bin("rc").unwrap()
    }

    fn put(dir: &Path, name: &str, fixture_name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, fixture(fixture_name)).unwrap();
        p
    }

    fn stdout(a: &assert_cmd::assert::Assert) -> String {
        String::from_utf8_lossy(&a.get_output().stdout).to_string()
    }

    #[test]
    fn jpeg_from_a_jpeg_source_picks_a_quality_and_names_the_metric() {
        let d = tempdir().unwrap();
        let input = put(d.path(), "photo.jpg", "rgb8_444.jpg");
        let out =
            cmd().arg(&input).args(["--target-ssimulacra2", "60", "-o"]).arg(d.path().join("t.jpg")).assert().success();
        let text = stdout(&out);
        assert!(text.contains("ssimulacra2"), "{text}");
        let img = image::open(d.path().join("t.jpg")).unwrap();
        assert_eq!((img.width(), img.height()), (67, 45));
    }

    #[test]
    fn jpeg_from_pixels_and_a_higher_target_never_gives_a_smaller_file() {
        let d = tempdir().unwrap();
        let input = put(d.path(), "pic.png", "rgb8.png");
        let size = |target: &str| {
            let o = d.path().join(format!("o{target}.jpg"));
            cmd().arg(&input).args(["-f", "jpeg", "--target-ssimulacra2", target, "-o"]).arg(&o).assert().success();
            std::fs::metadata(&o).unwrap().len()
        };
        let (low, high) = (size("30"), size("85"));
        assert!(low <= high, "target 30 gave {low} bytes, target 85 gave {high}");
    }

    #[test]
    fn lossy_webp_honours_the_target() {
        let d = tempdir().unwrap();
        let input = put(d.path(), "pic.png", "rgb8.png");
        let out = cmd()
            .arg(&input)
            .args(["-f", "webp", "--lossy", "--target-ssimulacra2", "50", "-o"])
            .arg(d.path().join("t.webp"))
            .assert()
            .success();
        assert!(stdout(&out).contains("ssimulacra2"), "{}", stdout(&out));
        assert!(d.path().join("t.webp").exists());
    }

    #[test]
    fn conflicting_or_out_of_range_targets_are_rejected() {
        let d = tempdir().unwrap();
        let input = put(d.path(), "pic.png", "rgb8.png");
        cmd().arg(&input).args(["--target-ssim", "0.9", "--target-ssimulacra2", "70"]).assert().code(2);
        cmd().arg(&input).args(["--target-ssimulacra2", "70", "--target-size", "5000"]).assert().code(2);
        cmd().arg(&input).args(["--target-ssimulacra2", "0"]).assert().code(2);
        cmd().arg(&input).args(["--target-ssimulacra2", "101"]).assert().code(2);
        cmd().arg(&input).args(["--target-ssimulacra2", "abc"]).assert().code(2);
    }

    #[test]
    fn an_image_under_8x8_is_a_clean_per_file_error_not_a_panic() {
        let d = tempdir().unwrap();
        let tiny = d.path().join("tiny.png");
        image::RgbImage::from_pixel(4, 4, image::Rgb([200, 30, 30])).save(&tiny).unwrap();
        for extra in [&["-f", "jpeg"][..], &["-f", "webp", "--lossy"][..]] {
            let out = cmd().arg(&tiny).args(extra).args(["--target-ssimulacra2", "70", "--force"]).assert().failure();
            let err = String::from_utf8_lossy(&out.get_output().stderr).to_string()
                + &String::from_utf8_lossy(&out.get_output().stdout);
            assert!(err.contains("8x8"), "{extra:?}: {err}");
        }
    }
}
