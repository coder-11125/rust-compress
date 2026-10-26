//! Animated WebP writing, read back through our own reader.
use rc_core::animation::{AnimFrame, Animation};
use rc_core::image::{ColorType, Image};
use rc_webp::{AnimOptions, decode_animation, encode_animation};

const W: usize = 48;
const H: usize = 34;

fn anim(frames: Vec<(Vec<[u8; 4]>, u32)>, alpha: bool, loop_forever: bool) -> Animation {
    let color = if alpha { ColorType::Rgba } else { ColorType::Rgb };
    let frames = frames
        .into_iter()
        .map(|(px, ms)| {
            let data: Vec<u8> = px.iter().flat_map(|p| if alpha { p.to_vec() } else { p[..3].to_vec() }).collect();
            AnimFrame { image: Image::new(W as u32, H as u32, color, 8, data).unwrap(), duration_ms: ms }
        })
        .collect();
    Animation { width: W as u32, height: H as u32, frames, loop_forever }
}

/// A gradient with fixed pseudo-random texture, as photographic footage has (smooth images code for free).
fn background(x: usize, y: usize) -> [u8; 4] {
    let h = (x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663);
    let h = h.wrapping_mul(2_654_435_761) >> 13;
    let n = |k: u32| ((h >> k) % 31) as i32 - 15;
    let c = |base: usize, k: u32| (base as i32 + n(k)).clamp(0, 255) as u8;
    [c(x * 5, 0), c(y * 7, 5), c((x + y) * 3, 10), 255]
}

/// A textured background with an 8x8 square sliding over it. `mode` picks how the square looks.
fn frames(n: usize, mode: &str) -> Vec<(Vec<[u8; 4]>, u32)> {
    (0..n)
        .map(|t| {
            let px = (0..W * H)
                .map(|p| {
                    let (x, y) = (p % W, p / W);
                    let (cx, cy) = (3 + 4 * t, 5 + t);
                    if x >= cx && x < cx + 8 && y >= cy && y < cy + 8 {
                        match mode {
                            "opaque" => [255, (40 * t) as u8, 30, 255],
                            "glass" => [255, (40 * t) as u8, 30, 120],
                            _ => [0, 0, 0, 0],
                        }
                    } else {
                        background(x, y)
                    }
                })
                .collect();
            (px, 80 + 20 * t as u32)
        })
        .collect()
}

fn pixels(img: &Image) -> Vec<[u8; 4]> {
    let ch = img.color.channels();
    img.data
        .chunks_exact(ch)
        .map(|p| if ch == 4 { [p[0], p[1], p[2], p[3]] } else { [p[0], p[1], p[2], 255] })
        .collect()
}

fn assert_lossless(a: &Animation, back: &Animation) {
    assert_eq!(back.frames.len(), a.frames.len());
    for (i, (got, want)) in back.frames.iter().zip(&a.frames).enumerate() {
        assert_eq!(got.duration_ms, want.duration_ms, "frame {i}");
        for (p, (g, w)) in pixels(&got.image).iter().zip(pixels(&want.image)).enumerate() {
            assert!(*g == w || (g[3] == 0 && w[3] == 0), "frame {i} pixel {p}: {g:?} vs {w:?}");
        }
    }
}

#[test]
fn opaque_changes_round_trip_exactly_and_beat_full_frames() {
    let a = anim(frames(7, "opaque"), false, true);
    let bytes = encode_animation(&a, &AnimOptions::default()).unwrap();
    assert!(rc_webp::is_webp(&bytes));
    assert_eq!(rc_webp::unsupported_reason(&bytes), Some("animated WebP"));
    let back = decode_animation(&bytes).unwrap();
    assert!(back.loop_forever);
    assert_lossless(&a, &back);
    let still = rc_webp::encode(&a.frames[0].image, &rc_webp::EncodeOptions::default()).len();
    assert!(bytes.len() < still * 3, "{} bytes for 7 frames, a still is {still}", bytes.len());
}

#[test]
fn partially_transparent_changes_round_trip_exactly() {
    // Blending cannot reproduce a half-transparent change, so those frames must replace the canvas.
    let a = anim(frames(6, "glass"), true, false);
    let back = decode_animation(&encode_animation(&a, &AnimOptions::default()).unwrap()).unwrap();
    assert!(!back.loop_forever);
    assert_lossless(&a, &back);
}

#[test]
fn pixels_that_become_transparent_again_round_trip_exactly() {
    let a = anim(frames(6, "clear"), true, true);
    let back = decode_animation(&encode_animation(&a, &AnimOptions::default()).unwrap()).unwrap();
    assert_lossless(&a, &back);
}

#[test]
fn identical_frames_are_nearly_free() {
    let one = frames(1, "opaque").remove(0).0;
    let many = anim((0..8).map(|_| (one.clone(), 100)).collect(), false, true);
    let single = anim(vec![(one, 100)], false, true);
    let big = encode_animation(&many, &AnimOptions::default()).unwrap();
    let small = encode_animation(&single, &AnimOptions::default()).unwrap();
    assert!(big.len() < small.len() + 7 * 80, "{} vs {}", big.len(), small.len());
    assert_lossless(&many, &decode_animation(&big).unwrap());
}

fn psnr(a: &Image, b: &Image) -> f64 {
    let (pa, pb) = (pixels(a), pixels(b));
    let se: f64 =
        pa.iter().zip(&pb).map(|(p, q)| (0..3).map(|c| (p[c] as f64 - q[c] as f64).powi(2)).sum::<f64>()).sum();
    10.0 * (255.0f64 * 255.0 / (se / (pa.len() * 3) as f64).max(1e-9)).log10()
}

#[test]
fn lossy_frames_stay_close_to_the_source_and_do_not_drift() {
    let a = anim(frames(9, "opaque"), false, true);
    let opts = AnimOptions { lossy_quality: Some(80), ..AnimOptions::default() };
    let bytes = encode_animation(&a, &opts).unwrap();
    let back = decode_animation(&bytes).unwrap();
    assert_eq!(back.frames.len(), 9);
    for (i, (got, want)) in back.frames.iter().zip(&a.frames).enumerate() {
        // Compare with a standalone lossy encode of the same frame: coding the frame as part of an animation
        // (changed rectangles measured against the decoded canvas) must not cost more than a few dB.
        let alone = rc_webp::decode(&rc_webp::encode_lossy(&want.image, 80, 7, true).unwrap()).unwrap();
        let (q, reference) = (psnr(&got.image, &want.image), psnr(&alone, &want.image));
        assert!(q > reference - 3.0, "frame {i}: {q:.1} dB in the animation, {reference:.1} dB alone");
    }
    let lossless = encode_animation(&a, &AnimOptions::default()).unwrap();
    assert!(bytes.len() < lossless.len(), "lossy {} vs lossless {}", bytes.len(), lossless.len());
}

#[test]
fn rejects_what_webp_cannot_hold() {
    let empty = Animation { width: 4, height: 4, frames: vec![], loop_forever: false };
    assert!(encode_animation(&empty, &AnimOptions::default()).is_err());
    let huge = Animation { width: 20_000, height: 2, frames: vec![], loop_forever: false };
    assert!(encode_animation(&huge, &AnimOptions::default()).is_err());
}
