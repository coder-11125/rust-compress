use common::*;
use rc_core::image::ColorType;
use rc_jpeg::decode::decode;
use rc_testutil as common;

const JPEGS: &[&str] = &["rgb8.jpg", "rgb8_444.jpg", "rgb8_420.jpg", "rgb8_prog.jpg", "rgb8_opt.jpg", "gray8.jpg"];

fn diff_stats(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len());
    let max = a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y)).max().unwrap_or(0);
    let mean = a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y) as f64).sum::<f64>() / a.len().max(1) as f64;
    (max, mean)
}

#[test]
fn decoder_matches_reference_decoder() {
    for name in JPEGS {
        let bytes = fixture(name);
        let ours = decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let theirs = image::load_from_memory(&bytes).unwrap();
        assert_eq!((ours.width, ours.height), (theirs.width(), theirs.height()), "{name}");
        let want =
            if ours.color == ColorType::Gray { theirs.to_luma8().into_raw() } else { theirs.to_rgb8().into_raw() };
        let (max, mean) = diff_stats(&ours.data, &want);
        println!("{name}: max diff {max}, mean {mean:.3}");
        assert!(mean < 1.5, "{name}: mean abs diff {mean}");
    }
}

#[test]
fn roundtrip_through_own_encoder() {
    use rc_jpeg::encode::{EncodeOptions, Subsampling, encode};
    let img = decode(&fixture("rgb8_444.jpg")).unwrap();
    let jpg =
        encode(&img, &EncodeOptions { quality: 92, subsampling: Subsampling::S444, ..Default::default() }).unwrap();
    let back = decode(&jpg).unwrap();
    let (_, mean) = diff_stats(&img.data, &back.data);
    assert!(mean < 6.0, "mean {mean}");
}

#[test]
fn truncated_and_corrupt_jpegs_never_panic() {
    for name in JPEGS {
        let good = fixture(name);
        let mut s = 7u32;
        for _ in 0..200 {
            let mut m = good.clone();
            for _ in 0..3 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let i = 2 + (s >> 8) as usize % (m.len() - 2);
                m[i] = (s >> 4) as u8;
            }
            let cut = (s >> 6) as usize % (m.len() + 1);
            let _ = decode(&m);
            let _ = decode(&m[..cut]);
        }
    }
}

#[test]
fn icc_profile_survives_jpeg_roundtrip_and_conversion() {
    use rc_jpeg::encode::{EncodeOptions, encode};
    let profile = fixture("icc_profile.bin");
    let img = decode(&fixture("icc.jpg")).unwrap();
    assert_eq!(img.meta.icc.as_deref(), Some(&profile[..]));
    let jpg = encode(&img, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&jpg).unwrap().meta.icc.as_deref(), Some(&profile[..]));
    // JPEG -> PNG keeps the profile too.
    let png = rc_png::encode::encode(&img, &Default::default());
    assert_eq!(rc_png::decode::decode(&png).unwrap().meta.icc.as_deref(), Some(&profile[..]));
}

#[test]
fn progressive_writer_is_lossless_and_decodable() {
    use rc_jpeg::decode::parse;
    use rc_jpeg::progressive::write_progressive;
    for name in ["rgb8.jpg", "rgb8_444.jpg", "rgb8_420.jpg", "gray8.jpg", "rgb8_prog.jpg"] {
        let bytes = fixture(name);
        let want = decode(&bytes).unwrap();
        let ci = parse(&bytes).unwrap().into_coefs().unwrap();
        let prog = write_progressive(&ci);
        // Our own decoder reproduces the exact same pixels (the coefficients are identical).
        let ours = decode(&prog).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(ours.data, want.data, "{name}: own decoder pixel mismatch");
        // And the independent reference decoder agrees.
        let theirs = image::load_from_memory(&prog).unwrap_or_else(|e| panic!("{name}: reference decoder: {e}"));
        let t = if want.color == ColorType::Gray { theirs.to_luma8().into_raw() } else { theirs.to_rgb8().into_raw() };
        let (max, mean) = diff_stats(&want.data, &t);
        assert!(mean < 1.5, "{name}: reference mean diff {mean} (max {max})");
    }
}

#[test]
fn baseline_transcode_is_lossless() {
    use rc_jpeg::decode::parse;
    use rc_jpeg::encode::write_jpeg;
    for name in JPEGS {
        let bytes = fixture(name);
        let want = decode(&bytes).unwrap();
        let ci = parse(&bytes).unwrap().into_coefs().unwrap();
        let back = decode(&write_jpeg(&ci)).unwrap();
        assert_eq!(back.data, want.data, "{name}");
    }
}

/// Repeat the third scan of a progressive JPEG (a first-pass AC scan, which can legally be decoded again) `copies`
/// more times right after itself. Each scan costs a pass over the image, so a file that does this is a time bomb:
/// 20,000 repeats on a 16-megapixel image took the decoder half a minute.
fn with_repeated_scan(jpeg: &[u8], copies: usize) -> Vec<u8> {
    let sos = jpeg.windows(2).enumerate().filter(|(_, w)| *w == [0xFF, 0xDA]).map(|(i, _)| i).nth(2).expect("a scan");
    let header_len = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
    let mut end = sos + 2 + header_len;
    while !(jpeg[end] == 0xFF && jpeg[end + 1] != 0 && !(0xD0..=0xD7).contains(&jpeg[end + 1])) {
        end += 1;
    }
    let mut out = jpeg[..end].to_vec();
    for _ in 0..copies {
        out.extend_from_slice(&jpeg[sos..end]);
    }
    out.extend_from_slice(&jpeg[end..]);
    out
}

#[test]
fn a_flood_of_repeated_scans_is_refused_but_a_few_repeats_are_fine() {
    let prog = fixture("rgb8_prog.jpg");
    assert!(decode(&prog).is_ok());
    let few = decode(&with_repeated_scan(&prog, 5));
    assert!(few.is_ok(), "a handful of extra scans is harmless: {:?}", few.err());
    let err = decode(&with_repeated_scan(&prog, 1500)).unwrap_err().to_string();
    assert!(err.contains("too many scans"), "{err}");
}
