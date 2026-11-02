#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_gif::decode_animation(data);
    let _ = rc_gif::decode(data);
});
