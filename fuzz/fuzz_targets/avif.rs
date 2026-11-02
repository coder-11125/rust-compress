#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = rc_avif::container::parse(data);
    let _ = rc_avif::sequence::parse(data);
    let _ = rc_avif::decode_animation(data);
    let _ = rc_avif::decode(data);
});
