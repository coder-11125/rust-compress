#![no_main]
use libfuzzer_sys::fuzz_target;

// The lossless-recompression path: parse coefficients, requantize, and write both entropy coders.
fuzz_target!(|data: &[u8]| {
    if let Ok(parsed) = rc_jpeg::decode::parse(data) {
        let _ = parsed.info();
        if let Ok(mut ci) = parsed.into_coefs() {
            ci.requantize(60, true);
            let _ = rc_jpeg::encode::write_best(&ci);
            let _ = rc_jpeg::arith::write_arithmetic(&ci);
        }
    }
});
