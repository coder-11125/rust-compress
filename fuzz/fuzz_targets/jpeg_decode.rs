#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_jpeg::decode::exif_orientation(data);
    if let Ok(img) = rc_jpeg::decode::decode(data) {
        // A header can legitimately declare up to MAX_PIXELS; re-encoding that much is slow by design
        // and would only show up as a fuzzer timeout. Bound the encode, not the decode.
        if (img.width as u64) * (img.height as u64) <= 1 << 20 {
            let _ = rc_jpeg::encode::encode(&img, &rc_jpeg::encode::EncodeOptions::default());
        }
    }
});
