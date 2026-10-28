//! Arithmetic-coded JPEG: coefficients written by our encoder come back identical through our decoder, in every
//! mode (sequential, progressive, restart intervals, non-default conditioning).
use rc_jpeg::arith::{ArithOptions, Conditioning, write_arithmetic, write_arithmetic_with};
use rc_jpeg::decode::{decode, parse};
use rc_jpeg::encode::CoefImage;
use rc_testutil::fixture;

/// Sequential and progressive, 4:2:0 / 4:4:4, gray, ICC, CMYK and both YCCK layouts.
const FILES: [&str; 10] = [
    "rgb8.jpg",
    "rgb8_420.jpg",
    "rgb8_444.jpg",
    "rgb8_opt.jpg",
    "rgb8_prog.jpg",
    "gray8.jpg",
    "icc.jpg",
    "cmyk.jpg",
    "ycck.jpg",
    "ycck_444.jpg",
];

fn coefs(bytes: &[u8]) -> CoefImage {
    parse(bytes).unwrap().into_coefs().unwrap()
}

fn same(a: &CoefImage, b: &CoefImage, what: &str) {
    assert_eq!((a.width, a.height, a.comps.len()), (b.width, b.height, b.comps.len()), "{what}");
    assert_eq!(a.adobe_transform, b.adobe_transform, "{what}");
    for (i, (x, y)) in a.comps.iter().zip(&b.comps).enumerate() {
        assert_eq!((x.id, x.h, x.v, x.blocks_w), (y.id, y.h, y.v, y.blocks_w), "{what}: component {i}");
        assert_eq!(a.qtables[x.tq], b.qtables[y.tq], "{what}: component {i} table");
        assert!(x.coefs == y.coefs, "{what}: component {i} coefficients differ");
    }
}

fn opts(progressive: bool, restart: usize, cond: Conditioning) -> ArithOptions {
    ArithOptions { progressive, conditioning: cond, restart_interval: restart }
}

#[test]
fn every_mode_round_trips_the_coefficients() {
    let custom = Conditioning { dc_l: [2, 1, 0, 0], dc_u: [6, 9, 1, 1], ac_k: [12, 30, 5, 5] };
    let mut tested = 0;
    for name in FILES {
        let src = fixture(name);
        let ci = coefs(&src);
        for (label, o) in [
            ("sequential", opts(false, 0, Conditioning::default())),
            ("progressive", opts(true, 0, Conditioning::default())),
            ("sequential, restarts", opts(false, 3, Conditioning::default())),
            ("progressive, restarts", opts(true, 5, Conditioning::default())),
            ("sequential, custom conditioning", opts(false, 0, custom.clone())),
            ("progressive, custom conditioning, restarts", opts(true, 7, custom.clone())),
        ] {
            let bytes = write_arithmetic_with(&ci, &o);
            let back = coefs(&bytes);
            same(&ci, &back, &format!("{name}, {label}"));
            tested += 1;
        }
    }
    assert_eq!(tested, FILES.len() * 6);
}

#[test]
fn the_decoded_picture_is_the_huffman_files_picture() {
    for name in FILES {
        let src = fixture(name);
        let want = decode(&src).unwrap();
        let got = decode(&write_arithmetic(&coefs(&src))).unwrap();
        assert_eq!(want, got, "{name}");
    }
}

/// Arithmetic-coded files made by `examples/make_arith.rs` whose decode by libjpeg-turbo (via Pillow) was checked to be
/// pixel-identical to their Huffman original; keeping the bytes pins the format independently of our own encoder.
const VERIFIED: [(&str, &str); 6] = [
    ("arith_420_seq.jpg", "rgb8_420.jpg"),
    ("arith_420_prog.jpg", "rgb8_420.jpg"),
    ("arith_gray.jpg", "gray8.jpg"),
    ("arith_444_prog_rst.jpg", "rgb8_444.jpg"),
    ("arith_cond.jpg", "rgb8.jpg"),
    ("arith_cmyk.jpg", "cmyk.jpg"),
];

#[test]
fn verified_arithmetic_files_decode_to_their_huffman_originals() {
    for (arith, huffman) in VERIFIED {
        let bytes = fixture(arith);
        let info = parse(&bytes).unwrap().info();
        assert!(info.arithmetic, "{arith}");
        assert_eq!(decode(&bytes).unwrap(), decode(&fixture(huffman)).unwrap(), "{arith}");
        assert!(bytes.windows(2).any(|w| w == [0xFF, 0xC9] || w == [0xFF, 0xCA]), "{arith} has no SOF9/SOF10");
    }
}

#[test]
fn arithmetic_coding_is_smaller_than_the_huffman_original() {
    // The arithmetic coder adapts to the data, so it beats even the optimized Huffman tables of our own writer.
    for name in ["rgb8.jpg", "rgb8_444.jpg", "rgb8_420.jpg"] {
        let src = fixture(name);
        let ci = coefs(&src);
        let huffman = rc_jpeg::encode::write_best(&ci).len();
        let arith = write_arithmetic(&ci).len();
        assert!(arith < huffman, "{name}: arithmetic {arith} !< huffman {huffman}");
    }
}

#[test]
fn damaged_arithmetic_streams_never_panic() {
    let src = fixture("rgb8_444.jpg");
    let ci = coefs(&src);
    for progressive in [false, true] {
        let good = write_arithmetic_with(&ci, &opts(progressive, 4, Conditioning::default()));
        // Truncations.
        for cut in (0..good.len()).step_by(11) {
            let _ = decode(&good[..cut]);
        }
        // Bit flips in the entropy data and headers.
        let mut state = 0x2545_F491u32;
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
// 12-bit
// ---------------------------------------------------------------------------------------------------------------

/// Differences between our decode of a 12-bit file and libjpeg-turbo's, in 12-bit levels: (largest, mean).
fn diff12(name: &str) -> (f64, f64) {
    use rc_testutil::{reference_rgba16, to_rgba16};
    let ours = decode(&fixture(&format!("{name}.jpg"))).unwrap();
    assert_eq!(ours.depth, 16, "{name}");
    let (w, h, want) = reference_rgba16(&fixture(&format!("{name}.expected.png")));
    assert_eq!((ours.width, ours.height), (w, h), "{name}");
    let got = to_rgba16(&ours);
    // Both sides hold 12-bit values widened to 16 bits; compare the 12-bit values.
    let d: Vec<f64> = got.iter().zip(&want).map(|(&a, &b)| ((a >> 4) as f64 - (b >> 4) as f64).abs()).collect();
    (d.iter().cloned().fold(0.0, f64::max), d.iter().sum::<f64>() / d.len() as f64)
}

#[test]
fn twelve_bit_files_decode_like_libjpeg_turbo() {
    for name in ["jpeg12_gray", "jpeg12_444", "jpeg12_420", "jpeg12_422"] {
        let (max, mean) = diff12(name);
        eprintln!("{name}: max {max} levels, mean {mean:.3} levels (of 4095)");
        // Float versus integer IDCT, and chroma upsampling rounding: a few levels in 4096 at most.
        assert!(max <= 4.0 && mean < 0.2, "{name}: max {max}, mean {mean:.3}");
    }
}

#[test]
fn twelve_bit_info_and_dct_domain_refusal() {
    let data = fixture("jpeg12_444.jpg");
    let parsed = parse(&data).unwrap();
    let info = parsed.info();
    assert_eq!((info.precision, info.arithmetic), (12, false));
    assert_eq!(info.source_quality, 100, "12-bit tables say nothing about quality");
    assert!(parsed.into_coefs().is_err(), "12-bit has no DCT-domain path");
    assert_eq!(parse(&fixture("rgb8.jpg")).unwrap().info().precision, 8);
}

#[test]
fn damaged_twelve_bit_files_never_panic() {
    let good = fixture("jpeg12_420.jpg");
    for cut in (0..good.len()).step_by(7) {
        let _ = decode(&good[..cut]);
    }
    let mut state = 0x9E37_79B9u32;
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
