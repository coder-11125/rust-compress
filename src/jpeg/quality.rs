//! Mapping of a 1..=100 quality setting to quantization tables.

use super::{STD_CHROMA_Q, STD_LUMA_Q};

/// IJG-style scaling of a base table.
pub fn scale_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u8; 64];
    for (o, &b) in out.iter_mut().zip(base) {
        *o = ((b as u32 * scale + 50) / 100).clamp(1, 255) as u8;
    }
    out
}

/// Estimate the IJG quality a luminance table was generated with (approximate for non-IJG tables).
pub fn estimate_quality(luma: &[u8; 64]) -> u8 {
    let s: f64 = luma.iter().zip(&STD_LUMA_Q).map(|(&t, &b)| t as f64 * 100.0 / b as f64).sum::<f64>() / 64.0;
    let q = if s <= 100.0 { (200.0 - s) / 2.0 } else { 5000.0 / s };
    q.round().clamp(1.0, 100.0) as u8
}

pub fn luma_table(quality: u8) -> [u8; 64] {
    scale_table(&STD_LUMA_Q, quality)
}

pub fn chroma_table(quality: u8) -> [u8; 64] {
    scale_table(&STD_CHROMA_Q, quality)
}
