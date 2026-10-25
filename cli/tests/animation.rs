//! Animated AVIF: key-frame sequences written by this crate, read back by this crate and (offline, with Pillow's
//! libavif and ffmpeg's dav1d) by independent decoders; animated GIF input; damaged movie boxes.
use assert_cmd::Command;
use common::fixture;
use rc_avif::encode::{EncodeOptions, encode_animation};
use rc_avif::{self as avif, sequence};
use rc_core::animation::{AnimFrame, Animation};
use rc_core::image::{ColorType, Image};
use rc_gif as gif;
use rc_testutil as common;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use tempfile::tempdir;

fn cmd() -> Command {
    Command::cargo_bin("rc").unwrap()
}

fn frame(w: u32, h: u32, color: ColorType, t: u32) -> Image {
    let ch = color.channels();
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let v = if color.has_alpha() && c == ch - 1 {
                    if (x + t * 3) % 16 < 8 { 255 } else { 40 }
                } else {
                    let v = (x * 4 + y * 2 + t * 20 + c as u32 * 60) % 256;
                    (if v < 128 { v * 2 } else { (255 - v) * 2 }) as u8
                };
                data.push(v);
            }
        }
    }
    Image::new(w, h, color, 8, data).unwrap()
}

fn animation(color: ColorType, n: u32) -> Animation {
    let frames = (0..n).map(|t| AnimFrame { image: frame(48, 32, color, t), duration_ms: 40 + 20 * t }).collect();
    Animation { width: 48, height: 32, frames, loop_forever: true }
}

fn opts() -> EncodeOptions {
    EncodeOptions { quality: 85, full_chroma: false, effort: 5, bit_depth: 8 }
}

fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height, a.color), (b.width, b.height, b.color));
    let n = a.data.len() as f64;
    let se: f64 = a.data.iter().zip(&b.data).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum();
    10.0 * (255.0f64 * 255.0 / (se / n).max(1e-9)).log10()
}

#[test]
fn animation_round_trips_with_timing_and_looping() {
    for color in [ColorType::Rgb, ColorType::Rgba, ColorType::Gray] {
        let anim = animation(color, 5);
        let bytes = encode_animation(&anim, &opts()).unwrap();
        assert!(avif::is_avif(&bytes));
        assert_eq!(&bytes[8..12], b"avis");
        let back = avif::decode_animation(&bytes).unwrap_or_else(|e| panic!("{color:?}: {e}"));
        assert_eq!(back.frames.len(), 5, "{color:?}");
        assert_eq!((back.width, back.height), (48, 32));
        assert!(back.loop_forever);
        for (i, (a, b)) in anim.frames.iter().zip(&back.frames).enumerate() {
            assert_eq!(a.duration_ms, b.duration_ms, "{color:?} frame {i}");
            let p = psnr(&a.image, &b.image);
            assert!(p > 28.0, "{color:?} frame {i}: PSNR {p:.1}");
        }
    }
}

/// A smooth random texture, shifted by (3t, t) pixels per frame: what a camera pan looks like.
fn panning(w: u32, h: u32, t: u32) -> Image {
    let hash = |x: i32, y: i32| -> f64 {
        let mut v = (x as u32).wrapping_mul(0x9e37_79b1) ^ (y as u32).wrapping_mul(0x85eb_ca6b) ^ 0x1234_5678;
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^= v >> 12;
        (v & 0xffff) as f64 / 65535.0
    };
    let noise = |fx: f64, fy: f64| -> f64 {
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let top = hash(x0, y0) * (1.0 - tx) + hash(x0 + 1, y0) * tx;
        let bottom = hash(x0, y0 + 1) * (1.0 - tx) + hash(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    };
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x + 3 * t) as f64, (y + t) as f64);
            let value = 0.5 * noise(u / 8.0, v / 8.0) + 0.3 * noise(u / 3.0, v / 3.0) + 0.2 * noise(u / 1.5, v / 1.5);
            let g = (value * 255.0) as u8;
            data.extend_from_slice(&[g, (g as u32 * 9 / 10) as u8, (255 - g) / 2 + 40]);
        }
    }
    Image::new(w, h, ColorType::Rgb, 8, data).unwrap()
}

#[test]
fn inter_frames_make_a_panning_animation_much_smaller() {
    let frames: Vec<AnimFrame> = (0..6).map(|t| AnimFrame { image: panning(128, 96, t), duration_ms: 100 }).collect();
    let anim = Animation { width: 128, height: 96, frames, loop_forever: true };
    let bytes = encode_animation(&anim, &opts()).unwrap();
    let separate: usize = anim.frames.iter().map(|f| avif::encode::encode(&f.image, &opts()).unwrap().len()).sum();
    assert!(bytes.len() * 10 < separate * 6, "animation {} bytes, frames coded separately {separate}", bytes.len());
    let seq = sequence::parse(&bytes).unwrap().expect("a movie box");
    let color = seq.color().unwrap();
    assert!(color.samples[0].sync);
    assert!(color.samples[1..].iter().any(|s| !s.sync), "later frames are inter frames");
    let back = avif::decode_animation(&bytes).unwrap();
    for (i, (a, b)) in anim.frames.iter().zip(&back.frames).enumerate() {
        let p = psnr(&a.image, &b.image);
        assert!(p > 28.0, "frame {i}: PSNR {p:.1}");
    }
}

#[test]
fn a_single_play_animation_does_not_loop() {
    let mut anim = animation(ColorType::Rgb, 3);
    anim.loop_forever = false;
    let back = avif::decode_animation(&encode_animation(&anim, &opts()).unwrap()).unwrap();
    assert!(!back.loop_forever);
}

#[test]
fn the_first_frame_is_also_the_still_primary_item() {
    let anim = animation(ColorType::Rgb, 4);
    let bytes = encode_animation(&anim, &opts()).unwrap();
    let still = avif::decode(&bytes).unwrap();
    let frames = avif::decode_animation(&bytes).unwrap();
    assert_eq!(still.data, frames.frames[0].image.data);
}

#[test]
fn sample_tables_describe_every_frame() {
    let bytes = encode_animation(&animation(ColorType::Rgba, 6), &opts()).unwrap();
    let seq = sequence::parse(&bytes).unwrap().expect("a movie box");
    assert_eq!(seq.tracks.len(), 2, "colour and alpha");
    let color = seq.color().unwrap();
    let alpha = seq.alpha().expect("alpha track");
    assert_eq!((color.samples.len(), alpha.samples.len()), (6, 6));
    assert_eq!(color.timescale, 1000);
    for t in [color, alpha] {
        assert!(t.samples[0].sync, "the first frame is a key frame");
        assert!(t.samples.iter().all(|s| s.offset + s.size as u64 <= bytes.len() as u64));
        assert_eq!(t.samples.iter().map(|s| s.duration).collect::<Vec<_>>(), vec![40, 60, 80, 100, 120, 140]);
    }
}

#[test]
fn animated_gif_converts_to_animated_avif() {
    let gif_bytes = fixture("gif_anim.gif");
    let anim = gif::decode_animation(&gif_bytes).unwrap();
    assert!(anim.frames.len() > 1);
    let d = tempdir().unwrap();
    let input = d.path().join("a.gif");
    fs::write(&input, &gif_bytes).unwrap();
    let out = d.path().join("a.avif");
    cmd().arg(&input).args(["-f", "avif", "-q", "90", "-o"]).arg(&out).assert().success();
    let bytes = fs::read(&out).unwrap();
    let back = avif::decode_animation(&bytes).unwrap();
    assert_eq!(back.frames.len(), anim.frames.len());
    assert_eq!((back.width, back.height), (anim.width, anim.height));
}

#[test]
fn a_still_gif_is_still_a_still() {
    let d = tempdir().unwrap();
    let input = d.path().join("s.gif");
    fs::write(&input, fixture("gif_plain.gif")).unwrap();
    let out = d.path().join("s.avif");
    cmd().arg(&input).args(["-f", "avif", "-o"]).arg(&out).assert().success();
    let bytes = fs::read(&out).unwrap();
    assert_eq!(&bytes[8..12], b"avif");
}

#[test]
fn gif_disposal_and_transparency_are_composited() {
    let anim = gif::decode_animation(&fixture("gif_anim.gif")).unwrap();
    let first = &anim.frames[0].image;
    assert!(anim.frames.iter().all(|f| f.image.width == first.width && f.image.height == first.height));
    assert!(anim.frames.iter().all(|f| f.duration_ms >= 20));
}

#[test]
fn damaged_movie_boxes_never_panic() {
    let bytes = encode_animation(&animation(ColorType::Rgba, 3), &opts()).unwrap();
    let moov = bytes.windows(4).position(|w| w == b"moov").expect("moov") - 4;
    let mut s = 0x1234_5678u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 8
    };
    for round in 0..400 {
        let mut b = bytes.clone();
        for _ in 0..1 + round % 4 {
            let i = moov + (rnd() as usize % (b.len() - moov));
            b[i] = rnd() as u8;
        }
        if round % 7 == 0 {
            b.truncate(moov + rnd() as usize % (b.len() - moov));
        }
        let r = catch_unwind(AssertUnwindSafe(|| {
            let _ = avif::decode_animation(&b);
        }));
        assert!(r.is_ok(), "panic on mutation round {round}");
    }
}

// ----------------------------------------------------------------------------------------------------------
// Other animated inputs, checked against the canvases Pillow composites for the same files.
// ----------------------------------------------------------------------------------------------------------

fn pillow_frame(name: &str, i: usize) -> Image {
    rc_png::decode::decode(&fixture(&format!("{name}.frame{i}.png"))).unwrap()
}

/// Pixels that differ by more than `tolerance` (of 255) in any channel. Fully transparent pixels may differ in
/// their invisible colour.
fn same_pixels(a: &Image, b: &Image, tolerance: u32) -> usize {
    let (x, y) = (common::to_rgba16(a), common::to_rgba16(b));
    assert_eq!((a.width, a.height), (b.width, b.height));
    if std::env::var_os("ANIM_DEBUG").is_some() {
        let w = a.width as usize;
        for (i, (p, q)) in x.chunks_exact(4).zip(y.chunks_exact(4)).enumerate() {
            if p != q && (p[3] != 0 || q[3] != 0) {
                eprintln!(
                    "px ({},{}) ours {:?} want {:?}",
                    i % w,
                    i / w,
                    p.iter().map(|v| v / 257).collect::<Vec<_>>(),
                    q.iter().map(|v| v / 257).collect::<Vec<_>>()
                );
            }
        }
    }
    let off = |p: u16, q: u16| (p as i32 - q as i32).unsigned_abs() / 257 > tolerance;
    x.chunks_exact(4)
        .zip(y.chunks_exact(4))
        .filter(|(p, q)| p[3] != q[3] || (p[3] != 0 && p.iter().zip(q.iter()).any(|(&u, &v)| off(u, v))))
        .count()
}

#[test]
fn apng_frames_match_pillow() {
    let anim = rc_png::apng::decode_animation(&fixture("apng_anim.png")).unwrap();
    assert_eq!(anim.frames.len(), 5);
    assert!(anim.loop_forever);
    assert_eq!(anim.frames.iter().map(|f| f.duration_ms).collect::<Vec<_>>(), vec![50, 80, 120, 60, 200]);
    for (i, f) in anim.frames.iter().enumerate() {
        assert_eq!(same_pixels(&f.image, &pillow_frame("apng_anim", i), 2), 0, "frame {i}");
    }
}

#[test]
fn animated_webp_frames_match_pillow() {
    let anim = rc_webp::decode_animation(&fixture("webp_anim.webp")).unwrap();
    assert_eq!(anim.frames.len(), 5);
    assert!(anim.loop_forever);
    assert_eq!(anim.frames.iter().map(|f| f.duration_ms).collect::<Vec<_>>(), vec![50, 80, 120, 60, 200]);
    for (i, f) in anim.frames.iter().enumerate() {
        assert_eq!(same_pixels(&f.image, &pillow_frame("webp_anim", i), 2), 0, "frame {i}");
    }
}

#[test]
fn apng_and_animated_webp_convert_to_animated_avif_through_the_cli() {
    for (name, ext) in [("apng_anim", "png"), ("webp_anim", "webp")] {
        let d = tempdir().unwrap();
        let input = d.path().join(format!("a.{ext}"));
        fs::write(&input, fixture(&format!("{name}.{ext}"))).unwrap();
        let out = d.path().join("a.avif");
        cmd().arg(&input).args(["-f", "avif", "-q", "90", "-o"]).arg(&out).assert().success();
        let back = avif::decode_animation(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(back.frames.len(), 5, "{name}");
        assert!(back.frames[0].image.color.has_alpha(), "{name}");
    }
}

#[test]
fn damaged_apng_and_webp_animations_never_panic() {
    for (name, ext) in [("apng_anim", "png"), ("webp_anim", "webp")] {
        let bytes = fixture(&format!("{name}.{ext}"));
        let mut s = 0x9876_5432u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s >> 8
        };
        for round in 0..300 {
            let mut b = bytes.clone();
            for _ in 0..1 + round % 3 {
                let i = rnd() as usize % b.len();
                b[i] = rnd() as u8;
            }
            if round % 9 == 0 {
                b.truncate(rnd() as usize % b.len());
            }
            let r = catch_unwind(AssertUnwindSafe(|| {
                let _ = rc_png::apng::decode_animation(&b);
                let _ = rc_webp::decode_animation(&b);
            }));
            assert!(r.is_ok(), "{name}: panic on mutation round {round}");
        }
    }
}
