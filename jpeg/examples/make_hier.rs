//! Developer tool: write the hierarchical JPEG test fixtures and their expected pictures.
//!
//! usage: make_hier <out_dir>
//! The lossless ones decode to exactly their source (`*.expected.png` is the source). The DCT ones have no external
//! reference (no other tool writes hierarchical JPEG): their `*.expected.png` is this decoder's own output when the
//! file was made, so they pin the format against later changes rather than prove it.
use rc_core::image::{ColorType, Image};
use rc_jpeg::decode::decode;
use rc_jpeg::hierarchical::{Entropy, HierOptions, Kind, write_hierarchical};
use std::path::Path;

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

fn main() {
    let dir = std::env::args().nth(1).expect("out_dir");
    let png = |img: &Image| rc_png::encode::encode(img, &rc_png::encode::EncodeOptions { level: 6, clean_alpha: true });
    let save = |name: &str, jpeg: &[u8], expected: &Image| {
        std::fs::write(Path::new(&dir).join(format!("{name}.jpg")), jpeg).unwrap();
        std::fs::write(Path::new(&dir).join(format!("{name}.expected.png")), png(expected)).unwrap();
    };
    let o = |levels, entropy, kind, subsample_chroma| HierOptions { levels, entropy, kind, subsample_chroma };
    let dct = |progressive, lossless_last| Kind::Dct { quality: 85, progressive, lossless_last };

    let gray = picture(61, 47, ColorType::Gray, 8);
    save(
        "hier_gray_lossless",
        &write_hierarchical(&gray, &o(3, Entropy::Huffman, Kind::Lossless, false)).unwrap(),
        &gray,
    );
    let rgb16 = picture(40, 30, ColorType::Rgb, 16);
    save(
        "hier_rgb16_lossless",
        &write_hierarchical(&rgb16, &o(3, Entropy::Huffman, Kind::Lossless, false)).unwrap(),
        &rgb16,
    );
    save(
        "hier_gray_dct_then_lossless",
        &write_hierarchical(&gray, &o(3, Entropy::Huffman, dct(false, true), false)).unwrap(),
        &gray,
    );
    let rgb = picture(64, 48, ColorType::Rgb, 8);
    for (name, entropy, progressive) in [
        ("hier_dct_huffman", Entropy::Huffman, false),
        ("hier_dct_progressive", Entropy::Huffman, true),
        ("hier_dct_arith", Entropy::Arithmetic, false),
        ("hier_dct_arith_progressive", Entropy::Arithmetic, true),
    ] {
        let bytes = write_hierarchical(&rgb, &o(3, entropy, dct(progressive, false), true)).unwrap();
        let expected = decode(&bytes).unwrap();
        save(name, &bytes, &expected);
    }
}
