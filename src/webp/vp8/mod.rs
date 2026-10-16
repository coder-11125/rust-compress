//! VP8 key-frame codec (the lossy half of WebP), written from RFC 6386.

pub mod boolcoder;
pub mod decode;
pub mod loopfilter;
pub mod predict;
#[rustfmt::skip]
pub mod tables;
pub mod transform;
pub mod yuv;
