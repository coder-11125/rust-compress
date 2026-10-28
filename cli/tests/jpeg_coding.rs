//! JPEG entropy coding and sample depth through the CLI: arithmetic-coded sources and outputs, and 12-bit sources.
use assert_cmd::Command;
use rc_jpeg as jpeg;
use rc_testutil::{fixture, reference_rgba16, to_rgba16};
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

/// Run the CLI on a fixture; returns the output bytes and the report text.
fn run(name: &str, args: &[&str]) -> (Vec<u8>, String) {
    let d = tempdir().unwrap();
    let input = d.path().join(name);
    fs::write(&input, fixture(name)).unwrap();
    let out = d.path().join("out.bin");
    let res = cmd().arg(&input).arg("-o").arg(&out).args(args).assert().success().get_output().clone();
    (fs::read(&out).unwrap(), String::from_utf8_lossy(&res.stdout).into_owned())
}

fn is_arithmetic(bytes: &[u8]) -> bool {
    jpeg::decode::parse(bytes).unwrap().info().arithmetic
}

#[test]
fn arithmetic_sources_stay_arithmetic_unless_told_otherwise() {
    for (arith, huffman) in
        [("arith_420_seq.jpg", "rgb8_420.jpg"), ("arith_420_prog.jpg", "rgb8_420.jpg"), ("arith_cmyk.jpg", "cmyk.jpg")]
    {
        // Default (auto): same coding, same picture, never larger.
        let (kept, _) = run(arith, &["-q", "100"]);
        assert!(is_arithmetic(&kept), "{arith}");
        assert!(kept.len() <= fixture(arith).len(), "{arith}");
        assert_eq!(jpeg::decode::decode(&kept).unwrap(), jpeg::decode::decode(&fixture(huffman)).unwrap(), "{arith}");
        // Asking for Huffman is honoured even though the result is larger.
        let (plain, log) = run(arith, &["-q", "100", "--arithmetic", "off"]);
        assert!(!is_arithmetic(&plain), "{arith}");
        assert!(plain.len() > fixture(arith).len(), "{arith}: {log}");
        assert_eq!(jpeg::decode::decode(&plain).unwrap(), jpeg::decode::decode(&fixture(huffman)).unwrap(), "{arith}");
    }
}

#[test]
fn arithmetic_output_is_opt_in_and_lossless_for_huffman_sources() {
    for name in ["rgb8_420.jpg", "rgb8_444.jpg", "gray8.jpg", "cmyk.jpg", "ycck.jpg"] {
        let src = fixture(name);
        let (default, _) = run(name, &["-q", "100"]);
        assert!(!is_arithmetic(&default), "{name}: arithmetic coding must be opt-in");
        let (arith, log) = run(name, &["-q", "100", "--arithmetic", "on"]);
        assert!(is_arithmetic(&arith), "{name}");
        assert!(log.contains("arithmetic coding"), "{name}: {log}");
        // The coefficients are the same, so the picture is the same.
        assert_eq!(jpeg::decode::decode(&arith).unwrap(), jpeg::decode::decode(&src).unwrap(), "{name}");
        assert!(arith.len() < default.len(), "{name}: {} !< {}", arith.len(), default.len());
    }
}

#[test]
fn arithmetic_output_works_for_pixel_conversions_too() {
    let d = tempdir().unwrap();
    let input = d.path().join("p.png");
    fs::write(&input, fixture("rgb8.png")).unwrap();
    let out = d.path().join("p.jpg");
    cmd().arg(&input).args(["-f", "jpeg", "--arithmetic", "on", "-o"]).arg(&out).assert().success();
    let bytes = fs::read(&out).unwrap();
    assert!(is_arithmetic(&bytes));
    assert!(jpeg::decode::decode(&bytes).is_ok());
    // And a quality search does too.
    let out2 = d.path().join("q.jpg");
    cmd()
        .arg(&input)
        .args(["-f", "jpeg", "--arithmetic", "on", "--target-ssim", "0.95", "-o"])
        .arg(&out2)
        .assert()
        .success();
    assert!(is_arithmetic(&fs::read(&out2).unwrap()));
}

#[test]
fn twelve_bit_jpeg_converts_to_16_bit_png_without_losing_precision() {
    let d = tempdir().unwrap();
    let input = d.path().join("w.jpg");
    fs::write(&input, fixture("jpeg12_444.jpg")).unwrap();
    let out = d.path().join("w.png");
    cmd().arg(&input).args(["-f", "png", "-o"]).arg(&out).assert().success();
    let (w, h, got) = reference_rgba16(&fs::read(&out).unwrap());
    let (ew, eh, want) = reference_rgba16(&fixture("jpeg12_444.expected.png"));
    assert_eq!((w, h), (ew, eh));
    // libjpeg-turbo's 12-bit decode, within the IDCT's rounding (a few levels of 4095).
    let worst = got.iter().zip(&want).map(|(&a, &b)| ((a >> 4) as i32 - (b >> 4) as i32).abs()).max().unwrap();
    assert!(worst <= 4, "worst difference {worst} levels");
    // More than 256 distinct levels in a channel: the extra precision really arrived.
    let mut reds: Vec<u16> = got.chunks_exact(4).map(|p| p[0]).collect();
    reds.sort_unstable();
    reds.dedup();
    assert!(reds.len() > 256, "{} distinct red levels", reds.len());
}

#[test]
fn twelve_bit_jpeg_recompresses_to_8_bit_jpeg_with_a_note() {
    for name in ["jpeg12_gray.jpg", "jpeg12_444.jpg", "jpeg12_420.jpg"] {
        let (out, log) = run(name, &[]);
        assert!(log.contains("12-bit reduced to 8-bit"), "{name}: {log}");
        let info = jpeg::decode::parse(&out).unwrap().info();
        assert_eq!(info.precision, 8, "{name}");
        // Similar picture: compare the 8-bit result with the 12-bit source's top 8 bits.
        let src = jpeg::decode::decode(&fixture(name)).unwrap();
        let dst = jpeg::decode::decode(&out).unwrap();
        assert_eq!((src.width, src.height), (dst.width, dst.height), "{name}");
        let (a, b) = (to_rgba16(&src), to_rgba16(&dst));
        let mean: f64 =
            a.iter().zip(&b).map(|(&x, &y)| (x as i32 - y as i32).abs() as f64 / 257.0).sum::<f64>() / a.len() as f64;
        assert!(mean < 12.0, "{name}: mean difference {mean:.1} of 255");
    }
    // The 12-bit source can be written with arithmetic coding as well.
    let (out, _) = run("jpeg12_444.jpg", &["--arithmetic", "on"]);
    assert!(is_arithmetic(&out) && jpeg::decode::parse(&out).unwrap().info().precision == 8);
}

#[test]
fn damaged_arithmetic_and_twelve_bit_files_fail_cleanly_through_the_cli() {
    let d = tempdir().unwrap();
    for name in ["arith_420_prog.jpg", "arith_444_prog_rst.jpg", "jpeg12_420.jpg"] {
        let good = fixture(name);
        for cut in [20, good.len() / 3, good.len() / 2, good.len() - 5] {
            let input = d.path().join("bad.jpg");
            fs::write(&input, &good[..cut]).unwrap();
            let out = cmd().arg(&input).args(["-o"]).arg(d.path().join("o.jpg")).arg("--force").output().unwrap();
            // Either a clean failure or a recovered partial picture, never a crash.
            assert!(out.status.code().is_some(), "{name} cut at {cut}: terminated by signal");
        }
    }
}
