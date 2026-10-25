//! The VP8 decoder must reproduce libwebp's decoded planes bit for bit. Each fixture's expected
//! CRC-32 per plane came from libwebp itself (`WebPDecodeYUV`).
use rc_core::util::checksum::crc32;
use rc_webp::vp8::decode::decode_frame;
use rc_webp::vp8::yuv::yuv420_to_rgb;
use std::path::PathBuf;

fn dir() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "..", "fixtures", "vp8"].iter().collect()
}

/// The VP8 frame inside a lossy WebP file (skipping any ALPH chunk).
fn vp8_payload(file: &[u8]) -> &[u8] {
    let mut pos = 12;
    while pos + 8 <= file.len() {
        let len = u32::from_le_bytes(file[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &file[pos..pos + 4] == b"VP8 " {
            return &file[pos + 8..pos + 8 + len];
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no VP8 chunk");
}

fn crop(plane: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    (0..h).flat_map(|r| plane[r * stride..r * stride + w].to_vec()).collect()
}

#[test]
fn decoded_planes_match_libwebp() {
    let index = std::fs::read_to_string(dir().join("index.txt")).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in index.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let (name, w, h): (&str, usize, usize) = (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
        let want = [f[3], f[4], f[5]].map(|s| u32::from_str_radix(s, 16).unwrap());
        let want_rgb = u32::from_str_radix(f[6], 16).unwrap();
        let file = std::fs::read(dir().join(format!("{name}.webp"))).unwrap();
        let frame = match decode_frame(vp8_payload(&file)) {
            Ok(fr) => fr,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        count += 1;
        assert_eq!((frame.width, frame.height), (w, h), "{name}");
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let got = [
            crc32(&crop(&frame.y, frame.y_stride, w, h)),
            crc32(&crop(&frame.u, frame.uv_stride, cw, ch)),
            crc32(&crop(&frame.v, frame.uv_stride, cw, ch)),
        ];
        let rgb = yuv420_to_rgb(w, h, &frame.y, frame.y_stride, &frame.u, &frame.v, frame.uv_stride);
        if crc32(&rgb) != want_rgb {
            failures.push(format!("{name}: RGB conversion differs from libwebp"));
        }
        if got != want {
            failures.push(format!(
                "{name}: planes differ (Y {} U {} V {})",
                got[0] == want[0],
                got[1] == want[1],
                got[2] == want[2]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} streams wrong:\n{}",
        failures.len(),
        count + failures.len(),
        failures.join("\n")
    );
}

#[test]
fn whole_file_decode_matches_libwebp_rgba_including_alpha() {
    let index = std::fs::read_to_string(dir().join("index.txt")).unwrap();
    let mut failures = Vec::new();
    for line in index.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let name = f[0];
        let want = u32::from_str_radix(f[7], 16).unwrap();
        let file = std::fs::read(dir().join(format!("{name}.webp"))).unwrap();
        let img = match rc_webp::decode(&file) {
            Ok(i) => i,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let rgba: Vec<u8> = match img.color {
            rc_core::image::ColorType::Rgba => img.data.clone(),
            _ => img.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        };
        if crc32(&rgba) != want {
            failures.push(format!("{name}: RGBA differs from libwebp (alpha: {})", img.color.has_alpha()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
