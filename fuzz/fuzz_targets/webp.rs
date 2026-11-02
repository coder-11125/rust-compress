#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_webp::unsupported_reason(data);
    let _ = rc_webp::decode_animation(data);
    if let Ok(img) = rc_webp::decode(data) {
        // Keep re-encoding bounded: the fuzzer is hunting parser bugs, not benchmarking the encoder.
        if img.width <= 1024 && img.height <= 1024 {
            let opts = rc_webp::EncodeOptions { level: 1, clean_alpha: false };
            let back = rc_webp::decode(&rc_webp::encode(&img, &opts)).expect("own WebP output must decode");
            assert_eq!((back.width, back.height), (img.width, img.height));
        }
    }
});
