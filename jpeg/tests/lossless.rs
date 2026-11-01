//! Lossless JPEG (SOF3): files written by libjpeg-turbo's lossless encoder decode to exactly the samples it was given.
use rc_jpeg::decode::{decode, parse};
use rc_testutil::{fixture, reference_rgba16, to_rgba16};
use std::path::Path;

fn lossless_fixtures() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("fixtures");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("lossless_") && n.ends_with(".jpg"))
        .collect();
    names.sort();
    names
}

#[test]
fn every_precision_predictor_and_component_count_decodes_exactly() {
    let names = lossless_fixtures();
    assert_eq!(names.len(), 12, "fixture set changed: {names:?}");
    for name in names {
        let bytes = fixture(&name);
        let info = parse(&bytes).unwrap().info();
        assert!(info.lossless, "{name}");
        let img = decode(&bytes).unwrap();
        let (w, h, want) = reference_rgba16(&fixture(&name.replace(".jpg", ".expected.png")));
        assert_eq!((img.width, img.height), (w, h), "{name}");
        let got = to_rgba16(&img);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{name}: {bad} of {} samples differ", got.len());
        // Up to 8 bits of precision come out as 8-bit images, everything else as 16-bit.
        let want_depth = if info.precision <= 8 { 8 } else { 16 };
        assert_eq!(img.depth, want_depth, "{name}");
    }
}

#[test]
fn lossless_files_have_no_dct_coefficients() {
    let bytes = fixture("lossless_gray_p12_pred5.jpg");
    let parsed = parse(&bytes).unwrap();
    assert!(parsed.info().lossless);
    assert!(parsed.into_coefs().is_err());
}

#[test]
fn damaged_lossless_files_never_panic() {
    let mut state = 0xA5A5_1234u32;
    for name in ["lossless_gray_p16_pred7.jpg", "lossless_rgb_p12_pred6.jpg", "lossless_cmyk_p8_pred2.jpg"] {
        let good = fixture(name);
        for cut in (0..good.len()).step_by(13) {
            let _ = decode(&good[..cut]);
        }
        for _ in 0..300 {
            let mut b = good.clone();
            for _ in 0..3 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let i = (state >> 8) as usize % b.len();
                b[i] ^= 1 << (state & 7);
            }
            let _ = decode(&b);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------------------------------------------

use rc_jpeg::lossless::{LosslessComp, LosslessFrame, LosslessOptions, recompress};

fn frame_of(name: &str) -> LosslessFrame {
    parse(&fixture(name)).unwrap().lossless_frame().unwrap()
}

#[test]
fn recompression_keeps_every_sample_with_every_predictor_and_restart_setting() {
    for name in lossless_fixtures() {
        let original = decode(&fixture(&name)).unwrap();
        for predictor in std::iter::once(None).chain((1..=7).map(Some)) {
            for restart_rows in [0, 1, 3] {
                let bytes = frame_of(&name)
                    .write(&LosslessOptions { predictor, restart_rows, ..Default::default() })
                    .unwrap()
                    .bytes;
                assert_eq!(
                    decode(&bytes).unwrap(),
                    original,
                    "{name}: predictor {predictor:?}, restart rows {restart_rows}"
                );
            }
        }
        // The automatic choice is the smallest of the seven, up to the byte stuffing its estimate leaves out.
        let auto = recompress(&fixture(&name)).unwrap();
        for p in 1..=7 {
            let size = frame_of(&name)
                .write(&LosslessOptions { predictor: Some(p), restart_rows: 0, ..Default::default() })
                .unwrap()
                .bytes
                .len();
            assert!(
                auto.bytes.len() <= size + size / 200 + 4,
                "{name}: predictor {p} gives {size}, auto {}",
                auto.bytes.len()
            );
        }
    }
}

#[test]
fn point_transform_scales_the_samples_back_up() {
    let mut f = frame_of("lossless_gray_p12_pred5.jpg");
    let original = decode(&fixture("lossless_gray_p12_pred5.jpg")).unwrap();
    f.point_transform = 3;
    f.comps[0].samples.iter_mut().for_each(|v| *v >>= 3);
    let bytes = f.write(&LosslessOptions::default()).unwrap().bytes;
    let back = decode(&bytes).unwrap();
    // Every sample lost its three low bits (widened to 16 bits, those are the low bits of the 12-bit value).
    let gray = |img: &rc_core::image::Image| -> Vec<u16> { to_rgba16(img).chunks_exact(4).map(|p| p[0]).collect() };
    for (o, b) in gray(&original).iter().zip(gray(&back)) {
        assert_eq!((*o >> 4) & !7, b >> 4, "sample {o} vs {b}");
    }
}

/// Components with their own sampling factors (a 4:2:0-style layout): the MCU holds 2x2 samples of the first
/// component and one of each of the others, and the planes are padded to whole MCUs.
#[test]
fn subsampled_components_round_trip() {
    let (w, h) = (37usize, 29usize);
    let (mcus_x, mcus_y) = (w.div_ceil(2), h.div_ceil(2));
    let mut state = 7u32;
    let mut plane = |sw: usize, sh: usize| -> Vec<u16> {
        (0..sw * sh)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((i % sw) as u32 * 6 + (i / sw) as u32 * 4 + (state >> 28)).min(1023) as u16
            })
            .collect()
    };
    let comps = vec![
        LosslessComp { id: 1, h: 2, v: 2, stride: mcus_x * 2, samples: plane(mcus_x * 2, mcus_y * 2) },
        LosslessComp { id: 2, h: 1, v: 1, stride: mcus_x, samples: plane(mcus_x, mcus_y) },
        LosslessComp { id: 3, h: 1, v: 1, stride: mcus_x, samples: plane(mcus_x, mcus_y) },
    ];
    let want: Vec<Vec<u16>> = comps.iter().map(|c| c.samples.clone()).collect();
    let frame = LosslessFrame {
        width: w,
        height: h,
        precision: 10,
        point_transform: 0,
        adobe_transform: None,
        meta: Default::default(),
        comps,
    };
    for predictor in 1..=7 {
        let bytes = frame
            .write(&LosslessOptions { predictor: Some(predictor), restart_rows: 2, ..Default::default() })
            .unwrap()
            .bytes;
        let back = parse(&bytes).unwrap().lossless_frame().unwrap();
        assert_eq!((back.width, back.height, back.precision), (w, h, 10));
        for (i, (b, w)) in back.comps.iter().zip(&want).enumerate() {
            assert!(b.samples == *w, "predictor {predictor}: component {i} differs");
        }
        // And it still converts to an image.
        assert_eq!(decode(&bytes).unwrap().width as usize, w);
    }
}

#[test]
fn structures_the_writer_cannot_reproduce_are_refused_cleanly() {
    assert!(recompress(&fixture("rgb8.jpg")).is_err(), "a DCT file is not a lossless one");
    assert!(recompress(b"not a jpeg").is_err());
    let f = frame_of("lossless_gray_p8_pred1.jpg");
    assert!(f.write(&LosslessOptions { predictor: Some(0), restart_rows: 0, ..Default::default() }).is_err());
    assert!(f.write(&LosslessOptions { predictor: Some(8), restart_rows: 0, ..Default::default() }).is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// Arithmetic coding (SOF11): from the standard's text only, so these tests establish self-consistency, not
// agreement with another implementation.
// ---------------------------------------------------------------------------------------------------------------

use rc_jpeg::arith::Conditioning;

#[test]
fn arithmetic_lossless_round_trips_every_fixture_in_every_mode() {
    let custom = Conditioning { dc_l: [2, 1, 0, 0], dc_u: [5, 3, 1, 1], ac_k: [5; 4] };
    let mut files = 0;
    for name in lossless_fixtures() {
        let original = decode(&fixture(&name)).unwrap();
        for (predictor, restart_rows, conditioning) in [
            (Some(1), 0, Conditioning::default()),
            (Some(4), 0, Conditioning::default()),
            (Some(7), 2, Conditioning::default()),
            (None, 0, custom.clone()),
            (None, 3, custom.clone()),
        ] {
            let opts = LosslessOptions { arithmetic: true, conditioning, predictor, restart_rows };
            let bytes = frame_of(&name).write(&opts).unwrap().bytes;
            let info = parse(&bytes).unwrap().info();
            assert!(info.lossless && info.arithmetic, "{name}");
            assert!(bytes.windows(2).any(|w| w == [0xFF, 0xCB]), "{name}: no SOF11");
            assert_eq!(
                decode(&bytes).unwrap(),
                original,
                "{name}: predictor {predictor:?} restart rows {restart_rows}"
            );
            files += 1;
        }
    }
    assert_eq!(files, 60);
}

#[test]
fn arithmetic_lossless_is_usually_smaller_than_huffman() {
    // Adaptive statistics beat per-file Huffman tables on all but the smallest files.
    let (mut huffman, mut arithmetic) = (0usize, 0usize);
    for name in lossless_fixtures() {
        huffman += frame_of(&name).write(&LosslessOptions::default()).unwrap().bytes.len();
        arithmetic +=
            frame_of(&name).write(&LosslessOptions { arithmetic: true, ..Default::default() }).unwrap().bytes.len();
    }
    assert!(arithmetic < huffman, "arithmetic {arithmetic} !< huffman {huffman}");
}

#[test]
fn arithmetic_lossless_subsampled_components_and_a_recode_to_huffman() {
    let (w, h) = (37usize, 29usize);
    let (mcus_x, mcus_y) = (w.div_ceil(2), h.div_ceil(2));
    let mut state = 11u32;
    let mut plane = |sw: usize, sh: usize| -> Vec<u16> {
        (0..sw * sh)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((i % sw) as u32 * 5 + (i / sw) as u32 * 3 + (state >> 28)).min(1023) as u16
            })
            .collect()
    };
    let comps = vec![
        LosslessComp { id: 1, h: 2, v: 2, stride: mcus_x * 2, samples: plane(mcus_x * 2, mcus_y * 2) },
        LosslessComp { id: 2, h: 1, v: 1, stride: mcus_x, samples: plane(mcus_x, mcus_y) },
        LosslessComp { id: 3, h: 1, v: 1, stride: mcus_x, samples: plane(mcus_x, mcus_y) },
    ];
    let want: Vec<Vec<u16>> = comps.iter().map(|c| c.samples.clone()).collect();
    let frame = LosslessFrame {
        width: w,
        height: h,
        precision: 10,
        point_transform: 0,
        adobe_transform: None,
        meta: Default::default(),
        comps,
    };
    let arith = frame
        .write(&LosslessOptions { arithmetic: true, predictor: Some(4), restart_rows: 2, ..Default::default() })
        .unwrap()
        .bytes;
    let back = parse(&arith).unwrap().lossless_frame().unwrap();
    for (b, w) in back.comps.iter().zip(&want) {
        assert!(b.samples == *w);
    }
    // An arithmetic-coded lossless file can be recoded as a Huffman one, and back.
    let huffman = recompress(&arith).unwrap().bytes;
    assert!(!parse(&huffman).unwrap().info().arithmetic);
    assert_eq!(decode(&huffman).unwrap(), decode(&arith).unwrap());
}

#[test]
fn damaged_arithmetic_lossless_files_never_panic() {
    let good = frame_of("lossless_rgb_p12_pred6.jpg")
        .write(&LosslessOptions { arithmetic: true, restart_rows: 4, ..Default::default() })
        .unwrap()
        .bytes;
    for cut in (0..good.len()).step_by(11) {
        let _ = decode(&good[..cut]);
    }
    let mut state = 0xFACE_B00Cu32;
    for _ in 0..400 {
        let mut b = good.clone();
        for _ in 0..3 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let i = (state >> 8) as usize % b.len();
            b[i] ^= 1 << (state & 7);
        }
        let _ = decode(&b);
    }
}
