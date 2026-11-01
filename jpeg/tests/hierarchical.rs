//! Hierarchical JPEG (T.81 Annex J). The streams come from this crate's own pyramid writer, which builds every
//! differential frame against what the decoder reconstructs; the tests check the decoder against the known source.
use rc_core::image::{ColorType, Image};
use rc_jpeg::decode::{decode, parse};
use rc_jpeg::encode::{EncodeOptions, Subsampling};
use rc_jpeg::hierarchical::{Entropy, HierOptions, Kind, write_hierarchical};

/// A picture with smooth areas, edges and texture, of any size (odd ones included).
fn picture(w: usize, h: usize, color: ColorType, depth: u8) -> Image {
    let ch = color.channels();
    let max = if depth == 16 { 65535.0 } else { 255.0 };
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let smooth = (x as f64 * (3.0 + c as f64) + y as f64 * 2.0) / (w + h) as f64;
                let edge = if (x / 7 + y / 5) % 3 == 0 { 0.25 } else { 0.0 };
                let tex = (((x * 31 + y * 17 + c * 5) % 13) as f64) / 130.0;
                let v = ((smooth + edge + tex).fract() * max).round() as u32;
                if depth == 16 {
                    data.extend_from_slice(&(v as u16).to_be_bytes());
                } else {
                    data.push(v as u8);
                }
            }
        }
    }
    Image::new(w as u32, h as u32, color, depth, data).unwrap()
}

fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height, a.color, a.depth), (b.width, b.height, b.color, b.depth));
    let n = a.data.len() as f64;
    let se: f64 = a.data.iter().zip(&b.data).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum();
    10.0 * (255.0f64 * 255.0 / (se / n).max(1e-9)).log10()
}

fn opts(levels: usize, entropy: Entropy, kind: Kind, subsample_chroma: bool) -> HierOptions {
    HierOptions { levels, entropy, kind, subsample_chroma }
}

#[test]
fn lossless_pyramids_reproduce_the_source_exactly() {
    for (w, h) in [(64, 48), (37, 29), (9, 7), (1, 1), (2, 31)] {
        for color in [ColorType::Gray, ColorType::Rgb] {
            for depth in [8u8, 16] {
                let src = picture(w, h, color, depth);
                for levels in 1..=4 {
                    let bytes =
                        write_hierarchical(&src, &opts(levels, Entropy::Huffman, Kind::Lossless, false)).unwrap();
                    let info = parse(&bytes).unwrap().info();
                    assert!(info.hierarchical);
                    assert_eq!((info.width, info.height, info.precision), (w, h, depth));
                    let got = decode(&bytes).unwrap();
                    assert!(got == src, "{w}x{h} {color:?} {depth}-bit, {levels} levels: not exact");
                }
            }
        }
    }
}

#[test]
fn arithmetic_lossless_pyramids_reproduce_the_source_exactly() {
    // SOF11 first frame, SOF15 differential frames.
    for (w, h) in [(64, 48), (37, 29), (3, 5)] {
        for color in [ColorType::Gray, ColorType::Rgb] {
            for depth in [8u8, 16] {
                let src = picture(w, h, color, depth);
                for levels in 1..=3 {
                    let bytes =
                        write_hierarchical(&src, &opts(levels, Entropy::Arithmetic, Kind::Lossless, false)).unwrap();
                    assert!(bytes.windows(2).any(|p| p == [0xFF, 0xCB]));
                    if levels > 1 {
                        assert!(bytes.windows(2).any(|p| p == [0xFF, 0xCF]));
                    }
                    assert!(decode(&bytes).unwrap() == src, "{w}x{h} {color:?} {depth}-bit, {levels} levels");
                }
            }
        }
    }
}

#[test]
fn arithmetic_dct_pyramids_ending_in_an_arithmetic_lossless_frame_are_exact_for_gray() {
    for progressive in [false, true] {
        let src = picture(53, 41, ColorType::Gray, 8);
        let kind = Kind::Dct { quality: 70, progressive, lossless_last: true };
        let bytes = write_hierarchical(&src, &opts(3, Entropy::Arithmetic, kind, false)).unwrap();
        assert!(decode(&bytes).unwrap() == src, "progressive={progressive}");
    }
}

#[test]
fn dct_pyramids_ending_in_a_lossless_frame_are_exact_for_gray() {
    for progressive in [false, true] {
        for (w, h) in [(64, 48), (45, 33)] {
            let src = picture(w, h, ColorType::Gray, 8);
            for levels in 2..=4 {
                let kind = Kind::Dct { quality: 60, progressive, lossless_last: true };
                let bytes = write_hierarchical(&src, &opts(levels, Entropy::Huffman, kind, false)).unwrap();
                assert!(decode(&bytes).unwrap() == src, "{w}x{h} progressive={progressive} levels={levels}");
            }
        }
    }
}

#[test]
fn every_dct_process_combination_decodes_close_to_the_source() {
    // Huffman / arithmetic, sequential / progressive, gray and colour, chroma at full or half resolution.
    let mut checked = 0;
    for entropy in [Entropy::Huffman, Entropy::Arithmetic] {
        for progressive in [false, true] {
            for (color, subsample) in [(ColorType::Gray, false), (ColorType::Rgb, false), (ColorType::Rgb, true)] {
                let src = picture(56, 40, color, 8);
                let kind = Kind::Dct { quality: 90, progressive, lossless_last: false };
                let coarse = decode(&write_hierarchical(&src, &opts(1, entropy, kind, subsample)).unwrap()).unwrap();
                let fine = decode(&write_hierarchical(&src, &opts(3, entropy, kind, subsample)).unwrap()).unwrap();
                let (c, f) = (psnr(&src, &coarse), psnr(&src, &fine));
                // The differential frames refine the picture, and the result is as good a copy of the source as an
                // ordinary JPEG at the same quality and chroma resolution (halved chroma limits both on this
                // picture, whose colour edges are sharp).
                let plain = {
                    let o = EncodeOptions {
                        quality: 90,
                        subsampling: if subsample { Subsampling::S420 } else { Subsampling::S444 },
                        ..EncodeOptions::default()
                    };
                    psnr(&src, &decode(&rc_jpeg::encode::encode(&src, &o).unwrap()).unwrap())
                };
                assert!(
                    f > plain - 3.0,
                    "{entropy:?} progressive={progressive} {color:?} subsample={subsample}: {f:.1} dB vs {plain:.1} for a plain JPEG"
                );
                assert!(
                    f >= c - 0.5,
                    "{entropy:?} progressive={progressive} {color:?}: {f:.1} dB after refinement vs {c:.1} before"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 12);
}

#[test]
fn colour_pyramids_with_a_lossless_end_are_within_the_colour_conversion() {
    // The lossless frame makes the YCC planes exact; converting them to RGB rounds, so allow a couple of levels.
    let src = picture(48, 40, ColorType::Rgb, 8);
    let kind = Kind::Dct { quality: 50, progressive: false, lossless_last: true };
    let got = decode(&write_hierarchical(&src, &opts(3, Entropy::Huffman, kind, false)).unwrap()).unwrap();
    let worst = got.data.iter().zip(&src.data).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
    assert!(worst <= 3, "worst channel difference {worst}");
}

#[test]
fn refinement_steps_improve_the_picture() {
    let src = picture(64, 64, ColorType::Gray, 8);
    let kind = Kind::Dct { quality: 40, progressive: false, lossless_last: false };
    let mut last = 0.0;
    for levels in 1..=4 {
        let got = decode(&write_hierarchical(&src, &opts(levels, Entropy::Huffman, kind, false)).unwrap()).unwrap();
        let q = psnr(&src, &got);
        assert!(q > last - 1.0, "levels {levels}: {q:.1} dB after {last:.1}");
        last = q;
    }
}

#[test]
fn malformed_hierarchical_streams_are_refused_not_crashed_on() {
    let src = picture(40, 30, ColorType::Gray, 8);
    let kind = Kind::Dct { quality: 70, progressive: false, lossless_last: false };
    let good = write_hierarchical(&src, &opts(3, Entropy::Huffman, kind, false)).unwrap();
    assert!(decode(&good).is_ok());
    let find = |needle: &[u8]| good.windows(needle.len()).position(|w| w == needle).unwrap();

    // EXP says no expansion although the frame is twice as large as the reference.
    let mut b = good.clone();
    let exp = find(&[0xFF, 0xDF]);
    b[exp + 4] = 0x00;
    assert!(decode(&b).is_err(), "reference size mismatch must be an error");
    // EXP values other than 0 and 1.
    let mut b = good.clone();
    b[exp + 4] = 0x22;
    assert!(decode(&b).is_err());
    // A differential frame with no frame before it.
    let dhp = find(&[0xFF, 0xDE]);
    let first_sof = find(&[0xFF, 0xC0]);
    let mut b = good[..first_sof].to_vec();
    b.extend_from_slice(&[0xFF, 0xC5]);
    b.extend_from_slice(&good[first_sof + 2..]);
    assert!(decode(&b).is_err());
    // DHP twice.
    let mut b = good.clone();
    let dhp_len = u16::from_be_bytes([good[dhp + 2], good[dhp + 3]]) as usize + 2;
    let copy = good[dhp..dhp + dhp_len].to_vec();
    b.splice(dhp..dhp, copy);
    assert!(decode(&b).is_err());
    // Truncations and bit flips never panic.
    for cut in (0..good.len()).step_by(5) {
        let _ = decode(&good[..cut]);
    }
    let mut state = 0xBEEFu32;
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

#[test]
fn a_hierarchical_image_has_no_dct_domain_form() {
    let src = picture(32, 32, ColorType::Gray, 8);
    let kind = Kind::Dct { quality: 70, progressive: false, lossless_last: false };
    let bytes = write_hierarchical(&src, &opts(2, Entropy::Huffman, kind, false)).unwrap();
    assert!(parse(&bytes).unwrap().into_coefs().is_err());
    assert!(parse(&bytes).unwrap().lossless_frame().is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// Fixed files
// ---------------------------------------------------------------------------------------------------------------

use rc_testutil::{fixture, reference_rgba16, to_rgba16};

fn matches_expected(name: &str, exact: bool) {
    let img = decode(&fixture(&format!("{name}.jpg"))).unwrap();
    let (w, h, want) = reference_rgba16(&fixture(&format!("{name}.expected.png")));
    assert_eq!((img.width, img.height), (w, h), "{name}");
    let got = to_rgba16(&img);
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(
        bad,
        0,
        "{name}: {bad} samples differ{}",
        if exact { " from the source" } else { " from the recorded decode" }
    );
}

#[test]
fn the_committed_hierarchical_files_decode_to_their_expected_pictures() {
    // Exact: these reproduce their source picture bit for bit.
    for name in ["hier_gray_lossless", "hier_rgb16_lossless", "hier_gray_dct_then_lossless"] {
        matches_expected(name, true);
    }
    // Recorded: the decoder's own output when the file was made (nothing else reads hierarchical JPEG).
    for name in ["hier_dct_huffman", "hier_dct_progressive", "hier_dct_arith", "hier_dct_arith_progressive"] {
        matches_expected(name, false);
    }
}

#[test]
fn the_same_coefficients_give_the_same_picture_in_every_entropy_coding() {
    // The four DCT fixtures carry identical quantized data through SOF5, SOF6, SOF13 and SOF14.
    let reference = decode(&fixture("hier_dct_huffman.jpg")).unwrap();
    for name in ["hier_dct_progressive", "hier_dct_arith", "hier_dct_arith_progressive"] {
        assert!(decode(&fixture(&format!("{name}.jpg"))).unwrap() == reference, "{name}");
    }
}
