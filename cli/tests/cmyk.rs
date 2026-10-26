use assert_cmd::Command;
use common::{fixture, reference_rgba16, to_rgba16};
use rc_jpeg as jpeg;
use rc_testutil as common;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

#[test]
fn cmyk_decodes_to_the_same_rgb_as_the_reference_conversion() {
    // cmyk_expected.png is Pillow's RGB conversion of cmyk.jpg.
    let ours = jpeg::decode::decode(&fixture("cmyk.jpg")).unwrap();
    let (w, h, want) = reference_rgba16(&fixture("cmyk_expected.png"));
    assert_eq!((ours.width, ours.height), (w, h));
    let got = to_rgba16(&ours);
    let mean: f64 = got.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs() as f64 / 257.0).sum::<f64>()
        / got.len() as f64;
    assert!(mean < 2.0, "mean abs diff {mean:.2} levels");
}

/// Run the CLI on a fixture and return the output bytes.
fn compress(name: &str, args: &[&str]) -> Vec<u8> {
    let d = tempdir().unwrap();
    let inp = d.path().join("in.jpg");
    fs::write(&inp, fixture(name)).unwrap();
    let out = d.path().join("out.jpg");
    cmd().arg(&inp).arg("-o").arg(&out).args(args).assert().success();
    fs::read(&out).unwrap()
}

fn mean_abs_diff(a: &[u16], b: &[u16]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).abs() as f64 / 257.0).sum::<f64>() / a.len() as f64
}

#[test]
fn cmyk_at_full_quality_is_transcoded_losslessly_with_its_colour_flag() {
    // Nothing is ever requantized to a finer table than the source had, so -q 100 changes no coefficient.
    for name in ["cmyk.jpg", "ycck.jpg", "ycck_444.jpg"] {
        let src = fixture(name);
        let out_bytes = compress(name, &["-q", "100"]);
        assert!(out_bytes.len() <= src.len(), "{name}");
        let a = jpeg::decode::parse(&src).unwrap().into_coefs().unwrap();
        let b = jpeg::decode::parse(&out_bytes).unwrap().into_coefs().unwrap();
        assert_eq!(b.comps.len(), 4);
        assert_eq!(a.adobe_transform, b.adobe_transform, "{name}: Adobe colour flag lost");
        assert_eq!(a.qtables, b.qtables, "{name}");
        for (x, y) in a.comps.iter().zip(&b.comps) {
            assert_eq!(x.coefs, y.coefs, "{name}");
        }
    }
}

#[test]
fn ycck_decodes_like_libjpeg() {
    // The expected PNGs are Pillow's (libjpeg's) RGB conversion of the files; ycck*.jpg were built by
    // jpeg/examples/make_ycck.rs and carry Adobe transform 2 with 4:2:0-style and 4:4:4 sampling.
    for (jpg, png) in [("ycck.jpg", "ycck_expected.png"), ("ycck_444.jpg", "ycck_444_expected.png")] {
        assert_eq!(jpeg::decode::parse(&fixture(jpg)).unwrap().into_coefs().unwrap().adobe_transform, Some(2));
        let ours = jpeg::decode::decode(&fixture(jpg)).unwrap();
        let (w, h, want) = reference_rgba16(&fixture(png));
        assert_eq!((ours.width, ours.height), (w, h), "{jpg}");
        let mean = mean_abs_diff(&to_rgba16(&ours), &want);
        assert!(mean < 2.0, "{jpg}: mean abs diff {mean:.2} levels");
    }
}

#[test]
fn cmyk_and_ycck_are_requantized_to_the_requested_quality() {
    for name in ["cmyk.jpg", "ycck.jpg", "ycck_444.jpg"] {
        for extra in [&[][..], &["--no-trellis"][..]] {
            let src = fixture(name);
            let mut args = vec!["-q", "30"];
            args.extend_from_slice(extra);
            let out = compress(name, &args);
            assert!(out.len() < src.len(), "{name} {extra:?}: {} !< {}", out.len(), src.len());
            let a = jpeg::decode::parse(&src).unwrap().into_coefs().unwrap();
            let b = jpeg::decode::parse(&out).unwrap().into_coefs().unwrap();
            assert_eq!(b.comps.len(), 4, "{name}");
            assert_eq!(a.adobe_transform, b.adobe_transform, "{name}: Adobe colour flag lost");
            // Never a finer table than the source had at any position, and coarser somewhere.
            let mut coarser = false;
            for (x, y) in a.comps.iter().zip(&b.comps) {
                let (old, new) = (a.qtables[x.tq].unwrap(), b.qtables[y.tq].unwrap());
                assert!(old.iter().zip(&new).all(|(o, n)| n >= o), "{name}: finer table");
                coarser |= old.iter().zip(&new).any(|(o, n)| n > o);
            }
            assert!(coarser, "{name}: nothing was requantized");
            // Channel roles: ink channels (and YCCK's Y and K) share the luma table, Cb/Cr the chroma one.
            let tq: Vec<usize> = b.comps.iter().map(|c| c.tq).collect();
            if name.starts_with("ycck") {
                assert_eq!(tq[0], tq[3], "{name}: Y and K");
                assert_eq!(tq[1], tq[2], "{name}: Cb and Cr");
                assert_ne!(tq[0], tq[1], "{name}: luma vs chroma");
            } else {
                assert!(tq.iter().all(|&t| t == tq[0]), "{name}: CMYK channels differ: {tq:?}");
            }
            // And the picture is still the picture.
            let before = to_rgba16(&jpeg::decode::decode(&src).unwrap());
            let after = to_rgba16(&jpeg::decode::decode(&out).unwrap());
            let mean = mean_abs_diff(&before, &after);
            assert!(mean < 14.0, "{name} {extra:?}: mean abs diff {mean:.2} levels");
        }
    }
}

#[test]
fn cmyk_converts_to_png_and_resizes_to_rgb() {
    let d = tempdir().unwrap();
    let inp = d.path().join("c.jpg");
    fs::write(&inp, fixture("cmyk.jpg")).unwrap();
    cmd().arg(&inp).args(["-f", "png"]).assert().success();
    let png = fs::read(d.path().join("c.min.png")).unwrap();
    let (_, _, got) = reference_rgba16(&png);
    let (_, _, want) = reference_rgba16(&fixture("cmyk_expected.png"));
    let mean: f64 = got.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs() as f64 / 257.0).sum::<f64>()
        / got.len() as f64;
    assert!(mean < 2.0, "{mean}");
    let out = d.path().join("small.jpg");
    cmd().arg(&inp).arg("-o").arg(&out).args(["--resize", "50%"]).assert().success();
    let img = image::open(&out).unwrap();
    assert_eq!(img.width(), 32);
}

#[test]
fn damaged_cmyk_never_panics() {
    let src = fixture("cmyk.jpg");
    for cut in (0..src.len()).step_by(7) {
        let _ = jpeg::decode::decode(&src[..cut]);
        if let Ok(p) = jpeg::decode::parse(&src[..cut]) {
            let _ = p.into_coefs().map(|c| jpeg::encode::write_best(&c));
        }
    }
}

#[test]
fn both_writers_roundtrip_four_components() {
    let src = fixture("cmyk.jpg");
    let ci = jpeg::decode::parse(&src).unwrap().into_coefs().unwrap();
    for (name, bytes) in
        [("baseline", jpeg::encode::write_jpeg(&ci)), ("progressive", jpeg::progressive::write_progressive(&ci))]
    {
        let back = jpeg::decode::parse(&bytes).unwrap_or_else(|e| panic!("{name}: {e}")).into_coefs().unwrap();
        assert_eq!(back.adobe_transform, ci.adobe_transform, "{name}");
        for (x, y) in ci.comps.iter().zip(&back.comps) {
            assert_eq!(x.coefs, y.coefs, "{name}");
        }
        if let Ok(dir) = std::env::var("CMYK_DUMP") {
            fs::write(format!("{dir}/{name}.jpg"), &bytes).unwrap();
        }
    }
}
