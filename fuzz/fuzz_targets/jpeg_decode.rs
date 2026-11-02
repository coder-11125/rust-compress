#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_jpeg::decode::exif_orientation(data);
    if let Ok(img) = rc_jpeg::decode::decode(data) {
        let _ = rc_jpeg::encode::encode(&img, &rc_jpeg::encode::EncodeOptions::default());
    }
});
