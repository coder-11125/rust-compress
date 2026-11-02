#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_png::is_animated(data);
    let _ = rc_png::apng::decode_animation(data);
    if let Ok(img) = rc_png::decode::decode(data) {
        // Anything that decodes must re-encode to a stream that decodes to the same size.
        let opts = rc_png::encode::EncodeOptions { level: 1, ..Default::default() };
        let out = rc_png::encode::encode(&img, &opts);
        let back = rc_png::decode::decode(&out).expect("own PNG output must decode");
        assert_eq!((back.width, back.height), (img.width, img.height));
        let _ = rc_png::quant::quantize(&img, 16, 0.5);
    }
});
