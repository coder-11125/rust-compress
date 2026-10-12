//! PNG codec (RFC 2083 / ISO 15948) written from scratch.

pub mod decode;
pub mod encode;
pub mod filter;
pub mod quant;

pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
