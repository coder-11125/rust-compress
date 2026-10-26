//! PNG codec (RFC 2083 / ISO 15948) written from scratch.

pub mod apng;
pub mod decode;
pub mod encode;
pub mod filter;
pub use rc_core::quant;

pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Whether `data` is an animated PNG (an `acTL` chunk before the first `IDAT`).
pub fn is_animated(data: &[u8]) -> bool {
    if !data.starts_with(&SIGNATURE) {
        return false;
    }
    let mut pos = 8;
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        match &data[pos + 4..pos + 8] {
            b"acTL" => return true,
            b"IDAT" | b"IEND" => return false,
            _ => {}
        }
        pos = match pos.checked_add(12).and_then(|p| p.checked_add(len)) {
            Some(p) => p,
            None => return false,
        };
    }
    false
}
