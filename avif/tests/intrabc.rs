//! Intra block copy: streams written by libaom with `tune-content=screen` (they use `allow_intrabc`), decoded
//! by this crate and compared, through a CRC of the raw planes, with what dav1d produced for the same streams.
//! Damaged versions of them must fail cleanly.
use common::fixture;
use rc_avif::av1::decoder::decode;
use rc_core::util::checksum::crc32;
use rc_testutil as common;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The planes in the layout ffmpeg writes raw video: row after row, 16-bit little endian above 8 bits.
fn raw_planes(d: &rc_avif::av1::decoder::Decoded) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &d.frame.planes {
        for y in 0..p.height {
            for &v in p.row(y) {
                if d.frame.bit_depth > 8 {
                    out.extend_from_slice(&v.to_le_bytes());
                } else {
                    out.push(v as u8);
                }
            }
        }
    }
    out
}

#[test]
fn block_copy_streams_decode_like_dav1d() {
    for name in ["av1_ibc_420", "av1_ibc_444", "av1_ibc_420_10"] {
        let d = decode(&fixture(&format!("{name}.obu"))).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(d.hdr.allow_intrabc, "{name} should use intra block copy");
        let want = String::from_utf8(fixture(&format!("{name}.crc32"))).unwrap();
        let got = format!("{:08x}", crc32(&raw_planes(&d)));
        assert_eq!(got, want.trim(), "{name}");
    }
}

#[test]
fn damaged_block_copy_streams_never_panic() {
    for name in ["av1_ibc_420", "av1_ibc_444"] {
        let bytes = fixture(&format!("{name}.obu"));
        let mut s = 0xdead_beefu32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s >> 8
        };
        for round in 0..300 {
            let mut b = bytes.clone();
            for _ in 0..1 + round % 4 {
                let i = rnd() as usize % b.len();
                b[i] = rnd() as u8;
            }
            if round % 11 == 0 {
                b.truncate(rnd() as usize % b.len());
            }
            let r = catch_unwind(AssertUnwindSafe(|| {
                let _ = decode(&b);
            }));
            assert!(r.is_ok(), "{name}: panic on mutation round {round}");
        }
    }
}

// ----------------------------------------------------------------------------------------------------------
// The encoder
// ----------------------------------------------------------------------------------------------------------

use rc_avif::av1::encode::{EncodeParams, encode_picture};
use rc_avif::av1::frame::{Frame, Plane};
use rc_avif::av1::obu::ColorConfig;
use rc_avif::container::{self, ImageData};
use rc_avif::convert::rgb_to_yuv;
use rc_avif::encode::{EncodeOptions, encode};
use rc_core::image::{ColorType, Image};

/// Text-like content: a few small glyph tiles reused all over a picture (so exact repeats exist far enough back).
fn repeated_tiles(w: usize, h: usize) -> Image {
    let mut s = 0x1357_9bdfu32;
    let mut rnd = |n: u32| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) % n
    };
    let (tw, th) = (24usize, 16usize);
    let tiles: Vec<Vec<[u8; 3]>> = (0..6)
        .map(|_| {
            let mut t = vec![[250u8; 3]; tw * th];
            for _ in 0..14 {
                let (x, y) = (rnd(tw as u32 - 8) as usize, rnd(th as u32 - 8) as usize);
                let c = [rnd(256) as u8, rnd(256) as u8, rnd(256) as u8];
                for j in 0..2 + rnd(6) as usize {
                    for i in 0..2 + rnd(6) as usize {
                        t[(y + j) * tw + x + i] = c;
                    }
                }
            }
            t
        })
        .collect();
    let mut data = vec![250u8; w * h * 3];
    for ty in (0..h).step_by(th) {
        for tx in (0..w).step_by(tw) {
            let t = &tiles[rnd(tiles.len() as u32) as usize];
            for j in 0..th.min(h - ty) {
                for i in 0..tw.min(w - tx) {
                    data[((ty + j) * w + tx + i) * 3..][..3].copy_from_slice(&t[j * tw + i]);
                }
            }
        }
    }
    Image::new(w as u32, h as u32, ColorType::Rgb, 8, data).unwrap()
}

fn frame_of(img: &Image) -> Frame {
    let (w, h) = (img.width as usize, img.height as usize);
    let yuv = rgb_to_yuv(img, 8, 1, 1, 0.299, 0.114);
    let plane = |v: &Vec<u16>, pw: usize, ph: usize| {
        let mut p = Plane::new(pw, ph);
        p.data.copy_from_slice(v);
        p
    };
    Frame {
        planes: vec![plane(&yuv.y, w, h), plane(&yuv.u, yuv.cw, yuv.ch), plane(&yuv.v, yuv.cw, yuv.ch)],
        bit_depth: 8,
        sub_x: 1,
        sub_y: 1,
        width: w,
        height: h,
    }
}

fn color() -> ColorConfig {
    ColorConfig {
        bit_depth: 8,
        mono_chrome: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
        color_range: true,
        sub_x: 1,
        sub_y: 1,
        chroma_sample_position: 0,
        separate_uv_delta_q: false,
    }
}

#[test]
fn repeated_content_is_coded_with_block_copy_and_is_smaller() {
    let img = repeated_tiles(768, 256);
    let frame = frame_of(&img);
    let params = EncodeParams { base_q_idx: 120, effort: 5, ..EncodeParams::default() };
    let with = encode_picture(&frame, &color(), &params).unwrap();
    let without = encode_picture(&frame, &color(), &EncodeParams { intrabc: false, ..params.clone() }).unwrap();
    assert!(with.hdr.allow_intrabc, "the trial should keep intra block copy for this picture");
    assert!(!without.hdr.allow_intrabc);
    assert!(
        with.data.len() * 10 < without.data.len() * 8,
        "block copy {} bytes vs plain {} bytes",
        with.data.len(),
        without.data.len()
    );
    // The stream decodes to exactly the picture the encoder said it would.
    let mut stream = Vec::new();
    stream.extend_from_slice(&with.data);
    let d = decode(&stream).unwrap();
    for (a, b) in d.frame.planes.iter().zip(&with.recon.planes) {
        assert_eq!((a.width, a.height), (b.width, b.height));
        for y in 0..a.height {
            assert_eq!(a.row(y), b.row(y), "row {y}");
        }
    }
}

#[test]
fn avif_files_of_repeated_content_use_block_copy_and_round_trip() {
    let img = repeated_tiles(768, 256);
    let bytes = encode(&img, &EncodeOptions { quality: 70, full_chroma: true, effort: 5, bit_depth: 8 }).unwrap();
    let file = container::parse(&bytes).unwrap();
    let ImageData::Coded(c) = &file.primary.data else { panic!("a single coded image") };
    let mut stream = c.av1c.config_obus.clone();
    stream.extend_from_slice(&c.data);
    assert!(decode(&stream).unwrap().hdr.allow_intrabc);
    let back = rc_avif::decode(&bytes).unwrap();
    assert_eq!((back.width, back.height), (768, 256));
    let mse: f64 = img.data.iter().zip(&back.data).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>()
        / img.data.len() as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
    assert!(psnr > 28.0, "PSNR {psnr:.1}");
}

#[test]
fn pictures_without_repeats_do_not_use_block_copy() {
    // Smooth noise: nothing repeats exactly.
    let (w, h) = (512usize, 192usize);
    let mut s = 7u32;
    let data: Vec<u8> = (0..w * h * 3)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (((i / 3 % w) * 255 / w) as u32 + (s >> 29)) as u8
        })
        .collect();
    let img = Image::new(w as u32, h as u32, ColorType::Rgb, 8, data).unwrap();
    let frame = frame_of(&img);
    let e =
        encode_picture(&frame, &color(), &EncodeParams { base_q_idx: 120, effort: 5, ..Default::default() }).unwrap();
    assert!(!e.hdr.allow_intrabc);
}
