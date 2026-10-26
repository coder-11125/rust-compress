//! GIF writing: animations come back through our own reader pixel for pixel when their colors fit the palette.
use rc_core::animation::{AnimFrame, Animation};
use rc_core::image::{ColorType, Image};
use rc_gif::{EncodeOptions, decode_animation, encode_animation, palette_is_exact};

const W: usize = 40;
const H: usize = 30;

fn anim(frames: Vec<(Vec<[u8; 4]>, u32)>, color: ColorType, loop_forever: bool) -> Animation {
    let frames = frames
        .into_iter()
        .map(|(px, ms)| {
            let data: Vec<u8> =
                px.iter().flat_map(|p| if color == ColorType::Rgba { p.to_vec() } else { p[..3].to_vec() }).collect();
            AnimFrame { image: Image::new(W as u32, H as u32, color, 8, data).unwrap(), duration_ms: ms }
        })
        .collect();
    Animation { width: W as u32, height: H as u32, frames, loop_forever }
}

/// Four-color background with a moving square.
fn opaque_frames(n: usize) -> Vec<(Vec<[u8; 4]>, u32)> {
    (0..n)
        .map(|t| {
            let px = (0..W * H)
                .map(|p| {
                    let (x, y) = (p % W, p / W);
                    if (4 + 3 * t..12 + 3 * t).contains(&x) && (6..14).contains(&y) {
                        [250, 20, 20, 255]
                    } else if (x / 10 + y / 10) % 2 == 0 {
                        [10, 40, 200, 255]
                    } else {
                        [240, 240, 60, 255]
                    }
                })
                .collect();
            (px, 60 + 10 * t as u32)
        })
        .collect()
}

/// A sprite on a transparent canvas: it also leaves pixels behind that must become transparent again.
fn sprite_frames(n: usize) -> Vec<(Vec<[u8; 4]>, u32)> {
    (0..n)
        .map(|t| {
            let px = (0..W * H)
                .map(|p| {
                    let (x, y) = (p % W, p / W);
                    let (cx, cy) = (6 + 4 * t, 8 + t);
                    if x >= cx && x < cx + 10 && y >= cy && y < cy + 10 {
                        [(30 * t % 256) as u8, 200, 90, 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect();
            (px, 50)
        })
        .collect()
}

fn same_picture(a: &Image, b: &Image) {
    let pa = to_rgba(a);
    let pb = to_rgba(b);
    for (i, (p, q)) in pa.iter().zip(&pb).enumerate() {
        assert!(p == q || (p[3] == 0 && q[3] == 0), "pixel {i}: {p:?} vs {q:?}");
    }
}

fn to_rgba(img: &Image) -> Vec<[u8; 4]> {
    let ch = img.color.channels();
    img.data
        .chunks_exact(ch)
        .map(|p| if ch == 4 { [p[0], p[1], p[2], p[3]] } else { [p[0], p[1], p[2], 255] })
        .collect()
}

#[test]
fn opaque_animation_round_trips_exactly_and_is_delta_coded() {
    let a = anim(opaque_frames(8), ColorType::Rgb, true);
    let bytes = encode_animation(&a, &EncodeOptions::default()).unwrap();
    assert!(bytes.starts_with(b"GIF89a"));
    let back = decode_animation(&bytes).unwrap();
    assert_eq!(back.frames.len(), 8);
    assert!(back.loop_forever);
    for (got, want) in back.frames.iter().zip(&a.frames) {
        same_picture(&got.image, &want.image);
        assert_eq!(got.duration_ms, want.duration_ms);
    }
    // Only the square moves, so later frames are far smaller than a frame of raw pixels.
    assert!(bytes.len() < 8 * W * H / 3, "{} bytes", bytes.len());
}

#[test]
fn sprite_that_leaves_transparent_pixels_behind_round_trips() {
    let a = anim(sprite_frames(7), ColorType::Rgba, false);
    let bytes = encode_animation(&a, &EncodeOptions::default()).unwrap();
    let back = decode_animation(&bytes).unwrap();
    assert!(!back.loop_forever);
    assert_eq!(back.frames.len(), 7);
    for (got, want) in back.frames.iter().zip(&a.frames) {
        same_picture(&got.image, &want.image);
    }
}

#[test]
fn something_appearing_then_disappearing_over_a_picture_round_trips() {
    // Opaque background; in the middle frames the square is cut out to transparency and comes back.
    let mut frames = opaque_frames(6);
    for t in [2usize, 3] {
        for p in 0..W * H {
            let (x, y) = (p % W, p / W);
            if x < 10 && y < 10 {
                frames[t].0[p] = [0, 0, 0, 0];
            }
        }
    }
    let a = anim(frames, ColorType::Rgba, true);
    let back = decode_animation(&encode_animation(&a, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.frames.len(), 6);
    for (got, want) in back.frames.iter().zip(&a.frames) {
        same_picture(&got.image, &want.image);
    }
}

#[test]
fn identical_frames_cost_almost_nothing() {
    let one = opaque_frames(1).remove(0).0;
    let a = anim((0..6).map(|_| (one.clone(), 100)).collect(), ColorType::Rgb, true);
    let single = anim(vec![(one, 100)], ColorType::Rgb, true);
    let big = encode_animation(&a, &EncodeOptions::default()).unwrap();
    let small = encode_animation(&single, &EncodeOptions::default()).unwrap();
    assert!(big.len() < small.len() + 5 * 40, "{} vs {}", big.len(), small.len());
    let back = decode_animation(&big).unwrap();
    assert_eq!(back.frames.len(), 6);
}

#[test]
fn a_still_with_many_colors_is_quantized_and_stays_close() {
    let px: Vec<[u8; 4]> = (0..W * H)
        .map(|p| {
            let (x, y) = (p % W, p / W);
            [(x * 6) as u8, (y * 8) as u8, ((x + y) * 3) as u8, 255]
        })
        .collect();
    let a = anim(vec![(px.clone(), 100)], ColorType::Rgb, false);
    assert!(!palette_is_exact(&a, 256));
    let back = decode_animation(&encode_animation(&a, &EncodeOptions { colors: 64, dither: 0.5 }).unwrap()).unwrap();
    let got = to_rgba(&back.frames[0].image);
    let colors: std::collections::HashSet<_> = got.iter().collect();
    assert!(colors.len() <= 64, "{} colors", colors.len());
    let mse: f64 = got
        .iter()
        .zip(&px)
        .map(|(p, q)| (0..3).map(|c| (p[c] as f64 - q[c] as f64).powi(2)).sum::<f64>() / 3.0)
        .sum::<f64>()
        / (W * H) as f64;
    assert!(10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10() > 24.0, "mse {mse}");
}

#[test]
fn a_many_color_animation_is_quantized_to_one_palette() {
    let frames: Vec<_> = (0..4)
        .map(|t| {
            let px = (0..W * H)
                .map(|p| {
                    let (x, y) = (p % W, p / W);
                    [(x * 6 + t * 5) as u8, (y * 8) as u8, ((x + y) * 3) as u8, 255]
                })
                .collect();
            (px, 80)
        })
        .collect();
    let a = anim(frames, ColorType::Rgb, true);
    assert!(!palette_is_exact(&a, 256));
    let back = decode_animation(&encode_animation(&a, &EncodeOptions { colors: 32, dither: 0.5 }).unwrap()).unwrap();
    let mut all = std::collections::HashSet::new();
    for f in &back.frames {
        all.extend(to_rgba(&f.image));
    }
    assert!(all.len() <= 32, "{} colors", all.len());
}

#[test]
fn palette_exactness_accounts_for_the_reserved_entry() {
    let a = anim(opaque_frames(3), ColorType::Rgb, true); // 3 colors, several frames: needs 3 + 1 entries
    assert!(palette_is_exact(&a, 4));
    assert!(!palette_is_exact(&a, 3));
    let still = anim(opaque_frames(1), ColorType::Rgb, true); // a single opaque frame needs no spare entry
    assert!(palette_is_exact(&still, 3));
}

#[test]
fn rejects_what_a_gif_cannot_hold() {
    let empty = Animation { width: 4, height: 4, frames: vec![], loop_forever: false };
    assert!(encode_animation(&empty, &EncodeOptions::default()).is_err());
    let huge = Animation { width: 70_000, height: 2, frames: vec![], loop_forever: false };
    assert!(encode_animation(&huge, &EncodeOptions::default()).is_err());
}
