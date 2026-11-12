use common::*;
use rc_png::{
    decode::decode,
    encode::{EncodeOptions, encode},
};
use rc_testutil as common;

const PNGS: &[&str] = &[
    "rgb8.png",
    "rgb8_adam7.png",
    "rgb16.png",
    "rgba16_adam7.png",
    "rgba8.png",
    "ga8.png",
    "ga8_adam7.png",
    "gray8.png",
    "gray1.png",
    "gray2.png",
    "gray4.png",
    "gray16.png",
    "pal2.png",
    "pal5.png",
    "pal200.png",
    "pal_trns.png",
    "rgb8_key.png",
];

#[test]
fn decoder_matches_reference_decoder() {
    for name in PNGS {
        let bytes = fixture(name);
        let ours = decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (w, h, want) = reference_rgba16(&bytes);
        assert_eq!((ours.width, ours.height), (w, h), "{name}");
        assert_eq!(to_rgba16(&ours), want, "{name}: pixel mismatch");
    }
}

/// RGBA16 with the hidden color of fully transparent pixels zeroed (what viewers actually show).
fn visible(mut px: Vec<u16>) -> Vec<u16> {
    for p in px.chunks_exact_mut(4) {
        if p[3] == 0 {
            p[..3].fill(0);
        }
    }
    px
}

#[test]
fn hidden_pixels_are_preserved_with_clean_alpha_off() {
    for name in ["rgba8.png", "ga8.png", "rgba16_adam7.png"] {
        let img = decode(&fixture(name)).unwrap();
        let want = to_rgba16(&img);
        let out = encode(&img, &EncodeOptions { level: 7, clean_alpha: false });
        assert_eq!(to_rgba16(&decode(&out).unwrap()), want, "{name}");
    }
}

#[test]
fn transparent_pixels_cleaning_shrinks_and_keeps_visible_pixels() {
    use rc_core::image::{ColorType, Image};
    // Noisy RGB under alpha=0 in half the image, flat opaque elsewhere.
    let (w, h) = (96u32, 64u32);
    let mut data = Vec::new();
    let mut s = 1u32;
    for y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                data.extend_from_slice(&[(s >> 24) as u8, (s >> 16) as u8, (s >> 8) as u8, 0]);
            } else {
                data.extend_from_slice(&[(y * 3) as u8, 40, 200, 255]);
            }
        }
    }
    let img = Image::new(w, h, ColorType::Rgba, 8, data).unwrap();
    let exact = encode(&img, &EncodeOptions { level: 7, clean_alpha: false });
    let cleaned = encode(&img, &EncodeOptions { level: 7, clean_alpha: true });
    assert!(cleaned.len() * 2 < exact.len(), "{} vs {}", cleaned.len(), exact.len());
    assert_eq!(visible(to_rgba16(&decode(&cleaned).unwrap())), visible(to_rgba16(&img)));
    let (_, _, theirs) = reference_rgba16(&cleaned);
    assert_eq!(visible(theirs), visible(to_rgba16(&img)));
}

#[test]
fn encoder_roundtrips_losslessly_at_every_level() {
    for name in PNGS {
        let img = decode(&fixture(name)).unwrap();
        let want = to_rgba16(&img);
        for level in [1, 6, 9] {
            let out = encode(&img, &EncodeOptions { level, ..Default::default() });
            // Our decoder.
            let back = decode(&out).unwrap_or_else(|e| panic!("{name} L{level}: {e}"));
            assert_eq!(visible(to_rgba16(&back)), visible(want.clone()), "{name} L{level}: own decoder mismatch");
            // Reference decoder must also accept it and agree.
            let (_, _, theirs) = reference_rgba16(&out);
            assert_eq!(visible(theirs), visible(want.clone()), "{name} L{level}: reference decoder mismatch");
        }
    }
}

#[test]
fn reduction_shrinks_files() {
    // Palette, gray and low-bit-depth sources should not grow and should usually shrink a lot.
    for name in ["rgb8.png", "gray8.png", "gray2.png", "pal5.png", "ga8.png"] {
        let orig = fixture(name);
        let img = decode(&orig).unwrap();
        let out = encode(&img, &EncodeOptions { level: 9, ..Default::default() });
        assert!(out.len() <= orig.len() + orig.len() / 10, "{name}: {} -> {}", orig.len(), out.len());
    }
}

#[test]
fn corrupt_png_never_panics() {
    let good = fixture("rgb8_adam7.png");
    let mut s = 99u32;
    for _ in 0..300 {
        let mut m = good.clone();
        for _ in 0..3 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let i = (s >> 8) as usize % m.len();
            m[i] = (s >> 4) as u8;
        }
        let cut = (s >> 6) as usize % (m.len() + 1);
        let _ = decode(&m);
        let _ = decode(&m[..cut]);
    }
}

#[test]
fn icc_profile_and_color_chunks_survive_png_roundtrip() {
    let img = decode(&fixture("icc.png")).unwrap();
    let profile = fixture("icc_profile.bin");
    assert_eq!(img.meta.icc.as_deref(), Some(&profile[..]));
    let out = encode(&img, &EncodeOptions::default());
    let back = decode(&out).unwrap();
    assert_eq!(back.meta.icc.as_deref(), Some(&profile[..]));
    // The reference decoder must still accept the file.
    let _ = reference_rgba16(&out);
}

#[test]
fn absurd_dimensions_are_rejected_without_huge_allocation() {
    use rc_core::util::checksum::crc32;
    // 100000 x 1000 RGBA8 header (400 MB of pixels) followed by a tiny IDAT.
    fn chunk(ty: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u32).to_be_bytes().to_vec();
        let mut tb = ty.to_vec();
        tb.extend_from_slice(body);
        v.extend_from_slice(&tb);
        v.extend_from_slice(&crc32(&tb).to_be_bytes());
        v
    }
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&100_000u32.to_be_bytes());
    ihdr.extend_from_slice(&1000u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    let mut png = rc_png::SIGNATURE.to_vec();
    png.extend(chunk(b"IHDR", &ihdr));
    png.extend(chunk(b"IDAT", &rc_deflate::zlib_compress(&[0u8; 16], 6)));
    png.extend(chunk(b"IEND", &[]));
    assert!(decode(&png).is_err());
}

#[test]
fn animated_png_is_refused_instead_of_flattened() {
    use rc_core::util::checksum::crc32;
    let mut png = fixture("rgb8.png");
    // Insert an acTL chunk right after IHDR (8 signature + 25 IHDR bytes).
    let mut body = Vec::new();
    body.extend_from_slice(&2u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    let mut tb = b"acTL".to_vec();
    tb.extend_from_slice(&body);
    let mut chunk = (body.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(&tb);
    chunk.extend_from_slice(&crc32(&tb).to_be_bytes());
    png.splice(33..33, chunk);
    let err = decode(&png).unwrap_err().to_string();
    assert!(err.contains("APNG"), "{err}");
}

fn anim_frame(w: u32, h: u32, color: rc_core::image::ColorType, t: u32) -> rc_core::image::Image {
    let ch = color.channels();
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let v = (x * 5 + y * 11 + t * 17 + c as u32 * 29) % 256;
                data.push(if color.has_alpha() && c == ch - 1 {
                    [255, 128, 0][(x + y + t) as usize % 3]
                } else {
                    v as u8
                });
            }
        }
    }
    rc_core::image::Image::new(w, h, color, 8, data).unwrap()
}

#[test]
fn apng_encode_round_trips_frames_timing_and_looping() {
    use rc_core::animation::{AnimFrame, Animation};
    use rc_core::image::ColorType;
    for (color, looping) in [(ColorType::Rgb, true), (ColorType::Rgba, false)] {
        let frames =
            (0..4).map(|t| AnimFrame { image: anim_frame(9, 7, color, t), duration_ms: 30 + 20 * t }).collect();
        let anim = Animation { width: 9, height: 7, frames, loop_forever: looping };
        let bytes = rc_png::apng::encode_animation(&anim, &EncodeOptions { level: 7, clean_alpha: false }).unwrap();
        assert!(rc_png::is_animated(&bytes));
        let back = rc_png::apng::decode_animation(&bytes).unwrap();
        assert_eq!((back.width, back.height), (anim.width, anim.height), "{color:?}");
        assert_eq!(back.frames.len(), anim.frames.len(), "{color:?}");
        assert_eq!(back.loop_forever, looping, "{color:?}");
        for (i, (a, b)) in anim.frames.iter().zip(&back.frames).enumerate() {
            assert_eq!(a.duration_ms, b.duration_ms, "{color:?} frame {i}");
            assert_eq!(to_rgba16(&a.image), to_rgba16(&b.image), "{color:?} frame {i}");
        }
    }
}

#[test]
fn apng_long_delays_survive() {
    use rc_core::animation::{AnimFrame, Animation};
    use rc_core::image::ColorType;
    let durations = [655_350, 120_000, 20];
    let frames = durations
        .iter()
        .enumerate()
        .map(|(t, &ms)| AnimFrame { image: anim_frame(4, 3, ColorType::Rgb, t as u32), duration_ms: ms })
        .collect();
    let anim = Animation { width: 4, height: 3, frames, loop_forever: false };
    let bytes = rc_png::apng::encode_animation(&anim, &EncodeOptions::default()).unwrap();
    let back = rc_png::apng::decode_animation(&bytes).unwrap();
    assert_eq!(back.frames.iter().map(|f| f.duration_ms).collect::<Vec<_>>(), durations);
}

#[test]
fn apng_sequence_numbers_count_fctl_and_fdat_chunks() {
    use rc_core::animation::{AnimFrame, Animation};
    use rc_core::image::ColorType;
    let frames = (0..3).map(|t| AnimFrame { image: anim_frame(5, 4, ColorType::Rgb, t), duration_ms: 100 }).collect();
    let anim = Animation { width: 5, height: 4, frames, loop_forever: true };
    let bytes = rc_png::apng::encode_animation(&anim, &EncodeOptions::default()).unwrap();
    let mut pos = 8;
    let mut seq = 0u32;
    let mut fdat = 0;
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let body = &bytes[pos + 8..pos + 8 + len];
        if matches!(kind, b"fcTL" | b"fdAT") {
            let got = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
            assert_eq!(got, seq, "{kind:?} sequence number");
            seq += 1;
            fdat += usize::from(kind == b"fdAT");
        }
        pos += 12 + len;
        if kind == b"IEND" {
            break;
        }
    }
    assert_eq!((fdat, seq), (2, 5));
}
