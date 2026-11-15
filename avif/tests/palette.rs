//! Palette mode. The `av1_pal_*` fixtures were written by this crate's encoder and decoded by dav1d and libaom
//! (which agree with each other and with this decoder bit for bit); the CRC of dav1d's raw planes is stored next
//! to each. The other tests round-trip pictures through the encoder and the decoder.
use common::fixture;
use rc_avif::av1::decoder::{Decoded, decode};
use rc_avif::av1::encode::{EncodeParams, encode_picture};
use rc_avif::av1::frame::{Frame, Plane};
use rc_avif::av1::obu::ColorConfig;
use rc_core::util::checksum::crc32;
use rc_testutil as common;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The planes in the layout ffmpeg writes raw video: row after row, 16-bit little endian above 8 bits.
fn raw_planes(d: &Decoded) -> Vec<u8> {
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
fn palette_streams_decode_like_dav1d() {
    for name in ["av1_pal_420", "av1_pal_444", "av1_pal_420_10", "av1_pal_mono"] {
        let d = decode(&fixture(&format!("{name}.obu"))).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(d.hdr.allow_screen_content_tools, "{name} should allow screen content tools");
        let want = String::from_utf8(fixture(&format!("{name}.crc32"))).unwrap();
        assert_eq!(format!("{:08x}", crc32(&raw_planes(&d))), want.trim(), "{name}");
    }
}

fn color(sub_x: u8, sub_y: u8, bit_depth: u8) -> ColorConfig {
    ColorConfig {
        bit_depth,
        mono_chrome: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
        color_range: true,
        sub_x,
        sub_y,
        chroma_sample_position: 0,
        separate_uv_delta_q: false,
    }
}

/// A UI-like picture: flat panels in a few colours (with V following U in some, not in others), one-pixel lines and a
/// smooth gradient patch that is not palette material.
fn ui_picture(w: usize, h: usize, sx: u8, sy: u8, bit_depth: u8) -> Frame {
    let sh = bit_depth as u32 - 8;
    let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
    let (mut y, mut u, mut v) = (Plane::new(w, h), Plane::new(cw, ch), Plane::new(cw, ch));
    let mut s = 12345u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 8
    };
    let panels: Vec<(usize, usize, usize, usize, [u16; 3])> = (0..24)
        .map(|i| {
            let (x0, y0) = (rnd() as usize % (w - 24), rnd() as usize % (h - 16));
            let c = if i % 2 == 0 {
                [(40 + 20 * (i % 7)) as u16, (100 + 12 * (i % 5)) as u16, (130 + 9 * (i % 5)) as u16]
            } else {
                [(rnd() % 220) as u16 + 16, (rnd() % 200) as u16 + 28, (rnd() % 200) as u16 + 28]
            };
            (x0, y0, x0 + 8 + rnd() as usize % 56, y0 + 8 + rnd() as usize % 40, c)
        })
        .collect();
    for py in 0..h {
        for px in 0..w {
            let mut c = [235u16, 128, 128];
            for &(x0, y0, x1, y1, col) in &panels {
                if px >= x0 && px < x1 && py >= y0 && py < y1 {
                    c = col;
                }
            }
            if py % 17 == 0 || px % 29 == 0 {
                c = [16, 128, 128];
            }
            if px > w - 40 && py > h - 40 {
                c = [(px * 5 + py * 3) as u16 % 200 + 20, 128 + (px % 9) as u16, 128 - (py % 7) as u16];
            }
            y.data[py * w + px] = c[0] << sh;
            if px % (1 << sx) == 0 && py % (1 << sy) == 0 {
                u.data[(py >> sy) * cw + (px >> sx)] = c[1] << sh;
                v.data[(py >> sy) * cw + (px >> sx)] = c[2] << sh;
            }
        }
    }
    Frame { planes: vec![y, u, v], bit_depth, sub_x: sx, sub_y: sy, width: w, height: h }
}

fn same_planes(a: &Frame, b: &Frame) -> bool {
    a.planes.len() == b.planes.len()
        && a.planes.iter().zip(&b.planes).all(|(p, q)| (0..p.height.min(q.height)).all(|r| p.row(r) == q.row(r)))
}

#[test]
fn encoder_palette_frames_round_trip() {
    for (sub, depth) in [(1u8, 8u8), (0, 8), (1, 10)] {
        let src = ui_picture(203, 139, sub, sub, depth);
        let c = color(sub, sub, depth);
        let params = EncodeParams { base_q_idx: 90, intrabc: false, ..EncodeParams::default() };
        let plain = encode_picture(&src, &c, &EncodeParams { palette: false, ..params.clone() }).unwrap();
        let pal = encode_picture(&src, &c, &EncodeParams { palette_frame: true, ..params.clone() }).unwrap();
        assert!(pal.hdr.allow_screen_content_tools && !pal.hdr.allow_intrabc);
        assert!(!plain.hdr.allow_screen_content_tools);
        assert!(
            pal.data.len() < plain.data.len(),
            "sub {sub} depth {depth}: {} vs {}",
            pal.data.len(),
            plain.data.len()
        );
        // the decoder reproduces what the encoder reconstructed
        let mut stream = vec![0x12, 0x00];
        stream.extend_from_slice(&pal.data);
        let d = decode(&stream).unwrap();
        assert!(same_planes(&d.frame, &pal.recon), "sub {sub} depth {depth}");
        // and the trial keeps the palette frame for this picture
        let chosen = encode_picture(&src, &c, &params).unwrap();
        assert!(chosen.hdr.allow_screen_content_tools, "sub {sub} depth {depth}: trial should pick palette");
    }
}

#[test]
fn photographic_pictures_do_not_try_palette() {
    let mut f = ui_picture(160, 96, 1, 1, 8);
    let mut s = 99u32;
    for p in &mut f.planes {
        for v in &mut p.data {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = 60 + ((s >> 24) as u16 % 120);
        }
    }
    let e = encode_picture(&f, &color(1, 1, 8), &EncodeParams { intrabc: false, ..EncodeParams::default() }).unwrap();
    assert!(!e.hdr.allow_screen_content_tools);
}

#[test]
fn damaged_palette_streams_never_panic() {
    for name in ["av1_pal_420", "av1_pal_444"] {
        let bytes = fixture(&format!("{name}.obu"));
        let mut s = 0xfeed_f00du32;
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

/// Every subsampling, every bit depth and the extreme (non-lossless) quantizers, with palette mode forced on, and the
/// same pictures without it as a control. `PAL_DUMP=<dir>` writes the palette streams and the encoder's reconstruction
/// (raw planes) for checking with other decoders.
#[test]
fn palette_in_every_format_round_trips() {
    let mut failures = Vec::new();
    for (sx, sy) in [(0u8, 0u8), (1, 0), (1, 1)] {
        for depth in [8u8, 10, 12] {
            for qidx in [1u8, 60, 255] {
                for palette in [true, false] {
                    let name = format!("fmt_{sx}{sy}_{depth}b_q{qidx}_{}", if palette { "pal" } else { "plain" });
                    let src = ui_picture(75, 51, sx, sy, depth);
                    let params = EncodeParams {
                        base_q_idx: qidx,
                        intrabc: false,
                        palette: false,
                        palette_frame: palette,
                        ..EncodeParams::default()
                    };
                    let enc = catch_unwind(AssertUnwindSafe(|| encode_picture(&src, &color(sx, sy, depth), &params)));
                    let e = match enc {
                        Ok(Ok(e)) => e,
                        Ok(Err(err)) => {
                            failures.push(format!("{name}: encoder error: {err}"));
                            continue;
                        }
                        Err(_) => {
                            failures.push(format!("{name}: encoder PANIC"));
                            continue;
                        }
                    };
                    let mut stream = vec![0x12, 0x00];
                    stream.extend_from_slice(&e.data);
                    let dec = catch_unwind(AssertUnwindSafe(|| decode(&stream)));
                    let Ok(dec) = dec else {
                        failures.push(format!("{name}: decoder PANIC"));
                        continue;
                    };
                    match dec {
                        Err(err) => failures.push(format!("{name}: our decoder: {err}")),
                        Ok(d) if !same_planes(&d.frame, &e.recon) => {
                            failures.push(format!("{name}: decoder != encoder"))
                        }
                        Ok(d) => {
                            if let Some(dir) = std::env::var_os("PAL_DUMP") {
                                let dir = std::path::Path::new(&dir);
                                std::fs::write(dir.join(format!("{name}.obu")), &stream).unwrap();
                                std::fs::write(dir.join(format!("{name}.raw")), raw_planes(&d)).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn lossless_is_an_error_not_a_panic() {
    let src = ui_picture(64, 64, 1, 1, 8);
    let params = EncodeParams { base_q_idx: 0, ..EncodeParams::default() };
    assert!(encode_picture(&src, &color(1, 1, 8), &params).is_err());
}

/// Palette blocks inside an inter frame: the second frame has a new few-colour panel that the first frame cannot
/// predict, and is coded with palette mode forced on. `PAL_DUMP` writes both frames as one OBU stream plus the raw
/// planes both decoders must produce.
#[test]
fn palette_in_inter_frames_round_trips() {
    use rc_avif::av1::decoder::Decoder;
    use std::sync::Arc;
    for (sx, sy, depth) in [(1u8, 1u8, 8u8), (0, 0, 8), (1, 1, 10)] {
        let name = format!("inter_{sx}{sy}_{depth}b");
        let c = color(sx, sy, depth);
        let f0 = ui_picture(96, 64, sx, sy, depth);
        let mut f1 = f0.clone();
        let sh = depth as u32 - 8;
        for y in 8..40 {
            for x in 16..80 {
                let v = [(60 + 40 * ((x / 8 + y / 8) % 3)) as u16, 90 + 30 * ((x / 16) % 2) as u16, 150];
                f1.planes[0].data[y * 96 + x] = v[0] << sh;
                if x % (1 << sx) == 0 && y % (1 << sy) == 0 {
                    let cw = 96usize.div_ceil(1 << sx);
                    f1.planes[1].data[(y >> sy) * cw + (x >> sx)] = v[1] << sh;
                    f1.planes[2].data[(y >> sy) * cw + (x >> sx)] = v[2] << sh;
                }
            }
        }
        let key = EncodeParams {
            base_q_idx: 60,
            still: false,
            intrabc: false,
            palette_frame: true,
            ..EncodeParams::default()
        };
        let e0 = encode_picture(&f0, &c, &key).unwrap();
        let inter = EncodeParams { reference: Some(Arc::new(e0.recon.clone())), ..key.clone() };
        let e1 = encode_picture(&f1, &c, &inter).unwrap();
        let plain_inter = EncodeParams { palette_frame: false, palette: false, ..inter.clone() };
        let e1_plain = encode_picture(&f1, &c, &plain_inter).unwrap();
        assert!(e1.hdr.allow_screen_content_tools && !e1_plain.hdr.allow_screen_content_tools, "{name}");

        let mut dec = Decoder::new();
        let mut stream = Vec::new();
        let mut raw = Vec::new();
        for (e, label) in [(&e0, "key"), (&e1, "inter")] {
            let mut tu = vec![0x12, 0x00];
            tu.extend_from_slice(&e.data);
            let got = dec.decode_temporal_unit(&tu).unwrap_or_else(|err| panic!("{name} {label}: {err}"));
            let d = got.first().unwrap_or_else(|| panic!("{name} {label}: no picture"));
            assert!(same_planes(&d.frame, &e.recon), "{name} {label}: decoder != encoder");
            raw.extend(raw_planes(d));
            stream.extend(tu);
        }
        if let Some(dir) = std::env::var_os("PAL_DUMP") {
            let dir = std::path::Path::new(&dir);
            std::fs::write(dir.join(format!("{name}.obu")), &stream).unwrap();
            std::fs::write(dir.join(format!("{name}.raw")), &raw).unwrap();
        }
        eprintln!("{name}: inter frame {} bytes with palette allowed, {} without", e1.data.len(), e1_plain.data.len());
    }
}
