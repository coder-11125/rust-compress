//! Developer tool: write a matrix of hierarchical JPEGs with their source pictures, for cross-checking decoders.
//!
//! usage: hier_matrix <out_dir>
//! For each case writes `NAME.jpg`, `NAME.raw` (the source samples, big-endian when 16-bit) and a line in `manifest.txt`:
//! `NAME width height channels depth exact|lossy`.
use rc_core::image::{ColorType, Image};
use rc_jpeg::hierarchical::{Entropy, HierOptions, Kind, Shape, write_hierarchical};
use std::fmt::Write as _;
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
    let mut manifest = String::new();
    let mut add = |name: String, img: &Image, opts: HierOptions, exact: bool| {
        let bytes = write_hierarchical(img, &opts).unwrap();
        std::fs::write(Path::new(&dir).join(format!("{name}.jpg")), bytes).unwrap();
        std::fs::write(Path::new(&dir).join(format!("{name}.raw")), &img.data).unwrap();
        let _ = writeln!(
            manifest,
            "{name} {} {} {} {} {}",
            img.width,
            img.height,
            img.color.channels(),
            img.depth,
            if exact { "exact" } else { "lossy" }
        );
    };
    for entropy in [Entropy::Huffman, Entropy::Arithmetic] {
        let e = if entropy == Entropy::Huffman { "huff" } else { "arith" };
        for (w, h) in [(64, 48), (37, 29), (9, 7)] {
            for (cn, color) in [("gray", ColorType::Gray), ("rgb", ColorType::Rgb)] {
                for depth in [8u8, 16] {
                    for levels in [1usize, 2, 3] {
                        let img = picture(w, h, color, depth);
                        let opts = HierOptions {
                            levels,
                            entropy,
                            kind: Kind::Lossless,
                            subsample_chroma: false,
                            shape: Shape::Both,
                        };
                        add(format!("ll_{e}_{cn}{depth}_{w}x{h}_L{levels}"), &img, opts, true);
                    }
                }
            }
        }
        // DCT pyramids ending in a lossless frame, and plain DCT ones.
        for progressive in [false, true] {
            let p = if progressive { "prog" } else { "seq" };
            let gray = picture(53, 41, ColorType::Gray, 8);
            for levels in [2usize, 3] {
                let kind = Kind::Dct { quality: 70, progressive, lossless_last: entropy == Entropy::Huffman };
                let opts = HierOptions { levels, entropy, kind, subsample_chroma: false, shape: Shape::Both };
                add(format!("dct_{e}_{p}_gray_L{levels}"), &gray, opts, entropy == Entropy::Huffman);
            }
            for subsample in [false, true] {
                let s = if subsample { "420" } else { "444" };
                let rgb = picture(64, 48, ColorType::Rgb, 8);
                let kind = Kind::Dct { quality: 85, progressive, lossless_last: false };
                let opts = HierOptions { levels: 3, entropy, kind, subsample_chroma: subsample, shape: Shape::Both };
                add(format!("dct_{e}_{p}_rgb{s}_L3"), &rgb, opts, false);
            }
        }
    }
    // Colour pyramids that end in a lossless frame: the YCbCr planes (not the RGB) are reproduced exactly.
    for (s, sub) in [("444", false), ("420", true)] {
        let rgb = picture(64, 48, ColorType::Rgb, 8);
        let kind = Kind::Dct { quality: 85, progressive: false, lossless_last: true };
        let opts =
            HierOptions { levels: 3, entropy: Entropy::Huffman, kind, subsample_chroma: sub, shape: Shape::Both };
        add(format!("ycc_huff_rgb{s}_L3"), &rgb, opts, false);
    }
    // A smooth picture, where any chroma upsampling reproduces the full-resolution chroma closely.
    {
        let (w, h) = (64usize, 48usize);
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[(x * 3 + 20) as u8, (y * 4 + 30) as u8, ((x + y) * 2 + 40) as u8]);
            }
        }
        let img = Image::new(w as u32, h as u32, ColorType::Rgb, 8, data).unwrap();
        let kind = Kind::Dct { quality: 85, progressive: false, lossless_last: true };
        let opts =
            HierOptions { levels: 3, entropy: Entropy::Huffman, kind, subsample_chroma: true, shape: Shape::Both };
        add("ycc_smooth_huff_rgb420_L3".to_string(), &img, opts, false);
    }
    // Expansion in one direction only, and refinement frames of the full size (no EXP at all).
    for (sn, shape) in [("horiz", Shape::Horizontal), ("vert", Shape::Vertical), ("refine", Shape::Refine)] {
        for entropy in [Entropy::Huffman, Entropy::Arithmetic] {
            let e = if entropy == Entropy::Huffman { "huff" } else { "arith" };
            for (cn, color) in [("gray", ColorType::Gray), ("rgb", ColorType::Rgb)] {
                let img = picture(61, 47, color, 8);
                let ll = HierOptions { levels: 3, entropy, kind: Kind::Lossless, subsample_chroma: false, shape };
                add(format!("shape_{sn}_{e}_{cn}_lossless"), &img, ll, true);
                // Lossy DCT frames, with a lossless one at the end (Huffman only: the writer keeps one entropy coder).
                let lossless_last = entropy == Entropy::Huffman && color == ColorType::Gray;
                let kind = Kind::Dct { quality: 75, progressive: false, lossless_last };
                let dct = HierOptions { levels: 3, entropy, kind, subsample_chroma: false, shape };
                add(format!("shape_{sn}_{e}_{cn}_dct"), &img, dct, lossless_last);
            }
        }
    }
    std::fs::write(Path::new(&dir).join("manifest.txt"), manifest).unwrap();
}
