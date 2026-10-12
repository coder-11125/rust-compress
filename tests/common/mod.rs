#![allow(dead_code)]
use rust_compress::image::{ColorType, Image};
use std::path::PathBuf;

pub fn fixture(name: &str) -> Vec<u8> {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name].iter().collect();
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Canonical RGBA16 view of one of our images.
pub fn to_rgba16(img: &Image) -> Vec<u16> {
    let ch = img.color.channels();
    let samples: Vec<u16> = if img.depth == 16 {
        img.data.chunks_exact(2).map(|s| u16::from_be_bytes([s[0], s[1]])).collect()
    } else {
        img.data.iter().map(|&v| v as u16 * 257).collect()
    };
    let mut out = Vec::with_capacity(samples.len() / ch * 4);
    for p in samples.chunks_exact(ch) {
        match img.color {
            ColorType::Gray => out.extend([p[0], p[0], p[0], 65535]),
            ColorType::GrayAlpha => out.extend([p[0], p[0], p[0], p[1]]),
            ColorType::Rgb => out.extend([p[0], p[1], p[2], 65535]),
            ColorType::Rgba => out.extend([p[0], p[1], p[2], p[3]]),
        }
    }
    out
}

pub fn reference_rgba16(bytes: &[u8]) -> (u32, u32, Vec<u16>) {
    let im = image::load_from_memory(bytes).expect("reference decoder failed");
    let rgba = im.to_rgba16();
    (rgba.width(), rgba.height(), rgba.into_raw())
}
