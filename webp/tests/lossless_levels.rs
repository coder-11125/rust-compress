//! Every lossless effort level round-trips, and more effort never makes the file larger.
use rc_core::image::{ColorType, Image};
use rc_webp::{EncodeOptions, decode, encode};

const W: usize = 192;
const H: usize = 128;

/// A textured gradient with a block of flat "interface" pixels, so both the predictor and the
/// no-prediction pipelines have something to win.
fn picture() -> Image {
    let mut data = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            let h = (x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663);
            let n = (h.wrapping_mul(2_654_435_761) >> 20) as i32 % 9 - 4;
            let px = if x < 64 && y < 48 {
                if (x / 8 + y / 8) % 2 == 0 { [250, 250, 250] } else { [20, 90, 200] }
            } else {
                let c = |v: usize| (v as i32 + n).clamp(0, 255) as u8;
                [c(x + y), c(y * 2), c(255 - x)]
            };
            data.extend_from_slice(&px);
        }
    }
    Image::new(W as u32, H as u32, ColorType::Rgb, 8, data).unwrap()
}

#[test]
fn levels_round_trip_and_never_grow() {
    let img = picture();
    let mut last = usize::MAX;
    for level in 0..=5 {
        let bytes = encode(&img, &EncodeOptions { level, clean_alpha: true });
        let back = decode(&bytes).unwrap();
        assert_eq!(back.data, img.data, "level {level} is not lossless");
        assert!(bytes.len() <= last, "level {level} is larger ({}) than the level below ({last})", bytes.len());
        last = bytes.len();
    }
}
