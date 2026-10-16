mod common;
use assert_cmd::Command;
use common::{fixture, reference_rgba16, to_rgba16};
use rust_compress::jpeg;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rust-compress").unwrap()
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

#[test]
fn cmyk_jpeg_is_transcoded_losslessly_with_its_colour_flag() {
    let src = fixture("cmyk.jpg");
    let d = tempdir().unwrap();
    let inp = d.path().join("c.jpg");
    fs::write(&inp, &src).unwrap();
    let out = d.path().join("o.jpg");
    cmd().arg(&inp).arg("-o").arg(&out).args(["-q", "30"]).assert().success();
    let out_bytes = fs::read(&out).unwrap();
    assert!(out_bytes.len() <= src.len());
    // Same coefficients and quantization tables: nothing was requantized, whatever -q said.
    let a = jpeg::decode::parse(&src).unwrap().into_coefs().unwrap();
    let b = jpeg::decode::parse(&out_bytes).unwrap().into_coefs().unwrap();
    assert_eq!(b.comps.len(), 4);
    assert_eq!(a.adobe_transform, b.adobe_transform, "Adobe colour flag lost");
    assert_eq!(a.qtables, b.qtables);
    for (x, y) in a.comps.iter().zip(&b.comps) {
        assert_eq!(x.coefs, y.coefs);
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
