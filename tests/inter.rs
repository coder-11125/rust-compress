//! Inter prediction: streams written by libaom (warped motion, overlapped and
//! masked prediction and global motion switched off for the first group, all tools on for the rest), decoded frame by frame with this crate's stateful decoder
//! and compared, through a CRC of all raw output planes, with what dav1d produced for the same streams.
mod common;
use common::fixture;
use rust_compress::avif::av1::decoder::Decoder;
use rust_compress::util::checksum::crc32;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// All shown frames in the layout ffmpeg writes raw video: plane after plane, 16-bit little endian above 8 bits.
fn decode_all(data: &[u8]) -> rust_compress::error::Result<(usize, Vec<u8>)> {
    let frames = Decoder::new().decode_temporal_unit(data)?;
    let mut out = Vec::new();
    for d in &frames {
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
    }
    Ok((frames.len(), out))
}

const STREAMS: [(&str, usize); 14] = [
    ("single", 12),
    ("altref", 16),
    ("pan_aq", 14),
    ("short_gop", 14),
    ("dist_comp", 14),
    ("tiles_odd", 8),
    ("444", 8),
    ("420_10", 8),
    ("mandel", 10),
    ("temporal", 16),
    ("temporal_pan", 14),
    ("all_tools", 16),
    ("rotate_zoom", 12),
    ("zoom_444", 10),
];

#[test]
fn inter_streams_decode_like_dav1d() {
    for (name, frames) in STREAMS {
        let (n, raw) = decode_all(&fixture(&format!("av1_inter_{name}.obu"))).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(n, frames, "{name}: number of shown frames");
        let want = String::from_utf8(fixture(&format!("av1_inter_{name}.crc32"))).unwrap();
        assert_eq!(format!("{:08x}", crc32(&raw)), want.trim(), "{name}");
    }
}

#[test]
fn damaged_inter_streams_never_panic() {
    for name in ["altref", "tiles_odd"] {
        let bytes = fixture(&format!("av1_inter_{name}.obu"));
        let mut s = 0x1234_5678u32;
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
                let _ = decode_all(&b);
            }));
            assert!(r.is_ok(), "{name}: panic on mutation round {round}");
        }
    }
}

/// Development helper: with `AV1_FILE` and `AV1_OUT` set, decode that stream and write the raw frames.
#[test]
fn dump_stream() {
    let (Ok(f), Ok(o)) = (std::env::var("AV1_FILE"), std::env::var("AV1_OUT")) else { return };
    std::fs::write(o, decode_all(&std::fs::read(f).unwrap()).unwrap().1).unwrap();
}
