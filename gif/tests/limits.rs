//! Hostile GIFs: a one-pixel canvas passes any pixel budget, so the number of frames has its own limit.
use rc_core::animation::MAX_ANIMATION_FRAMES;
use rc_gif::decode_animation;

/// A 1x1 GIF with `n` one-pixel frames (the pixel alternates so no frame equals its neighbour).
fn gif_with_frames(n: usize) -> Vec<u8> {
    let mut g = b"GIF89a".to_vec();
    g.extend_from_slice(&[1, 0, 1, 0, 0xF0, 0, 0]); // 1x1, global table of 2 colours
    g.extend_from_slice(&[0, 0, 0, 255, 255, 255]);
    for i in 0..n {
        g.extend_from_slice(&[0x21, 0xF9, 4, 0, 0, 0, 0, 0]); // graphic control extension
        g.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0]); // image descriptor
        g.extend_from_slice(&[2, 2, if i % 2 == 0 { 0x44 } else { 0x4C }, 1, 0]); // LZW data
    }
    g.push(0x3B);
    g
}

#[test]
fn an_animation_with_too_many_frames_is_refused() {
    let ok = decode_animation(&gif_with_frames(100)).unwrap();
    assert_eq!(ok.frames.len(), 100);
    let err = decode_animation(&gif_with_frames(MAX_ANIMATION_FRAMES + 1)).unwrap_err().to_string();
    assert!(err.contains("too large"), "{err}");
}
