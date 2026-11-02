#![no_main]
use libfuzzer_sys::fuzz_target;

// A bare AV1 OBU stream, bypassing the AVIF container so the fuzzer reaches the decoder directly.
fuzz_target!(|data: &[u8]| {
    let _ = rc_avif::av1::decoder::decode(data);
});
