mod common;
use assert_cmd::Command;
use common::{fixture, reference_rgba16, to_rgba16};
use rust_compress::gif;
use std::fs;
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

#[test]
fn decoder_matches_the_reference_decoder() {
    for name in ["gif_plain.gif", "gif_interlaced.gif", "gif_transparent.gif"] {
        let bytes = fixture(name);
        let ours = gif::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (w, h, want) = reference_rgba16(&bytes);
        assert_eq!((ours.width, ours.height), (w, h), "{name}");
        let got = to_rgba16(&ours);
        // Fully transparent pixels may differ in their (invisible) color.
        let bad = got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .filter(|(a, b)| a[3] != b[3] || (a[3] != 0 && a != b))
            .count();
        assert_eq!(bad, 0, "{name}: {bad} differing pixels");
    }
}

#[test]
fn transparency_becomes_an_alpha_channel_only_when_needed() {
    assert!(!gif::decode(&fixture("gif_plain.gif")).unwrap().color.has_alpha());
    assert!(gif::decode(&fixture("gif_transparent.gif")).unwrap().color.has_alpha());
}

#[test]
fn interlaced_and_sequential_decode_to_the_same_picture() {
    // The two fixtures hold the same image, stored differently.
    let a = gif::decode(&fixture("gif_plain.gif")).unwrap();
    let b = gif::decode(&fixture("gif_interlaced.gif")).unwrap();
    assert_eq!(a, b);
}

#[test]
fn animated_gif_is_refused() {
    let err = gif::decode(&fixture("gif_anim.gif")).unwrap_err().to_string();
    assert!(err.contains("animated"), "{err}");
}

#[test]
fn converts_to_png_and_jpeg_via_the_cli() {
    let d = tempdir().unwrap();
    let input = d.path().join("pic.gif");
    fs::write(&input, fixture("gif_transparent.gif")).unwrap();
    cmd().arg(&input).args(["-f", "png"]).assert().success();
    let png = fs::read(d.path().join("pic.min.png")).unwrap();
    let (w, h, want) = reference_rgba16(&fixture("gif_transparent.gif"));
    let (pw, ph, got) = reference_rgba16(&png);
    assert_eq!((pw, ph), (w, h));
    let bad =
        got.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a[3] != b[3] || (a[3] != 0 && a != b)).count();
    assert_eq!(bad, 0, "PNG conversion must be lossless");
    cmd().arg(&input).args(["-f", "jpeg", "-o"]).arg(d.path().join("pic.jpg")).assert().success();
    assert!(image::open(d.path().join("pic.jpg")).is_ok());
}

#[test]
fn keeping_the_format_is_a_clear_error_and_directories_skip_gifs() {
    let d = tempdir().unwrap();
    let input = d.path().join("pic.gif");
    fs::write(&input, fixture("gif_plain.gif")).unwrap();
    let out = cmd().arg(&input).assert().failure().get_output().clone();
    assert!(String::from_utf8_lossy(&out.stderr).contains("--format"), "{}", String::from_utf8_lossy(&out.stderr));
    // A plain directory run ignores the GIF instead of failing the whole batch.
    fs::write(d.path().join("a.png"), fixture("rgb8.png")).unwrap();
    cmd().arg(d.path()).assert().success();
    assert!(d.path().join("a.min.png").exists());
    assert!(!d.path().join("pic.min.gif").exists());
    // With a target format the GIF is picked up too.
    fs::remove_file(d.path().join("a.min.png")).unwrap();
    cmd().arg(d.path()).args(["-f", "png"]).assert().success();
    assert!(d.path().join("pic.min.png").exists());
}

#[test]
fn animated_gif_through_the_cli_fails_cleanly() {
    let out =
        cmd().args(["-", "-f", "png"]).write_stdin(fixture("gif_anim.gif")).assert().failure().get_output().clone();
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("animated GIF"));
}
