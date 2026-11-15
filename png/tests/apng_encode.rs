use rc_core::animation::{AnimFrame, Animation};
use rc_core::image::{ColorType, Image};
use rc_png::apng::{decode_animation, encode_animation};
use rc_png::encode::EncodeOptions;

const W: u32 = 64;
const H: u32 = 48;

/// A moving opaque square over a static background; `alpha` makes the background partly transparent.
fn make(alpha: bool, n: usize, still_at: Option<usize>) -> Animation {
    let ch = if alpha { 4 } else { 3 };
    let frames = (0..n)
        .map(|i| {
            let t = if Some(i) == still_at { i - 1 } else { i };
            let mut d = Vec::new();
            for y in 0..H {
                for x in 0..W {
                    let inside = x >= 4 + 3 * t as u32 && x < 14 + 3 * t as u32 && (10..20).contains(&y);
                    let mut px = if inside { [200, 30, 30, 255] } else { [(x * 4) as u8, (y * 5) as u8, 90, 255] };
                    if alpha && !inside && (x / 8 + y / 8) % 2 == 0 {
                        px = [0, 0, 0, 0];
                    } else if alpha && !inside && x % 5 == 0 {
                        px[3] = 128;
                    }
                    d.extend_from_slice(&px[..ch]);
                }
            }
            let color = if alpha { ColorType::Rgba } else { ColorType::Rgb };
            AnimFrame { image: Image::new(W, H, color, 8, d).unwrap(), duration_ms: 40 + 10 * i as u32 }
        })
        .collect();
    Animation { width: W, height: H, frames, loop_forever: true }
}

fn rgba(img: &Image) -> Vec<u8> {
    match img.color {
        ColorType::Rgba => img.data.clone(),
        _ => img.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
    }
}

/// Opaque-or-transparent pixels compared with hidden colour ignored.
fn same(a: &Image, b: &Image) -> bool {
    rgba(a).chunks_exact(4).zip(rgba(b).chunks_exact(4)).all(|(p, q)| p == q || (p[3] == 0 && q[3] == 0))
}

fn fctls(png: &[u8]) -> Vec<[u32; 5]> {
    let mut v = Vec::new();
    let mut i = 8;
    while i + 8 <= png.len() {
        let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
        if &png[i + 4..i + 8] == b"fcTL" {
            let b = &png[i + 8..i + 8 + len];
            let g = |o: usize| u32::from_be_bytes(b[o..o + 4].try_into().unwrap());
            v.push([g(4), g(8), g(12), g(16), b[25] as u32]);
        }
        i += 12 + len;
    }
    v
}

#[test]
fn dirty_rectangles_roundtrip_and_shrink() {
    for alpha in [false, true] {
        let anim = make(alpha, 8, Some(4));
        let png = encode_animation(&anim, &EncodeOptions::default()).unwrap();
        let back = decode_animation(&png).unwrap();
        assert_eq!(back.frames.len(), 8);
        for (a, b) in anim.frames.iter().zip(&back.frames) {
            assert!(same(&a.image, &b.image), "alpha {alpha}");
            assert_eq!(a.duration_ms, b.duration_ms);
        }
        let ctl = fctls(&png);
        assert_eq!(ctl[0][..2], [W, H]);
        // later frames only cover the moving square (plus its previous position)
        assert!(ctl[1][0] < W && ctl[1][1] < H, "alpha {alpha}: {:?}", ctl[1]);
        // the repeated frame is a 1x1 no-op
        assert_eq!(ctl[4][..2], [1, 1]);
        if alpha {
            assert!(ctl[1..].iter().any(|c| c[4] == 1), "OVER should win for opaque additions");
        }
        // Compared with every frame stored in full this is much smaller.
        let full: usize = anim.frames.iter().map(|f| rc_deflate::zlib_compress(&f.image.data, 6).len()).sum();
        assert!(png.len() < full / 2, "alpha {alpha}: {} vs {full}", png.len());
    }
}

#[test]
fn translucent_changes_fall_back_to_source() {
    let mut anim = make(true, 3, None);
    // frame 2 turns an opaque pixel of frame 1 translucent: OVER cannot lower alpha.
    let i = (30 * W as usize + 40) * 4;
    anim.frames[1].image.data[i + 3] = 255;
    anim.frames[2].image.data[i + 3] = 100;
    let png = encode_animation(&anim, &EncodeOptions { clean_alpha: false, ..Default::default() }).unwrap();
    let back = decode_animation(&png).unwrap();
    for (a, b) in anim.frames.iter().zip(&back.frames) {
        assert_eq!(rgba(&a.image), rgba(&b.image));
    }
}

/// Streams written by the dirty-rectangle encoder (cropped frames, OVER blending, 1x1 no-ops), then damaged: random
/// byte changes and truncation anywhere, and frame control chunks with absurd geometry and operations whose CRC is
/// repaired so the decoder really looks at them. None of it may panic.
#[test]
fn damaged_dirty_rectangle_streams_never_panic() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let bytes = encode_animation(&make(true, 8, Some(4)), &EncodeOptions::default()).unwrap();
    let mut s = 0x1357_9bdfu32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 8
    };
    for round in 0..300 {
        let mut b = bytes.clone();
        for _ in 0..1 + round % 3 {
            let i = rnd() as usize % b.len();
            b[i] = rnd() as u8;
        }
        if round % 7 == 0 {
            b.truncate(rnd() as usize % b.len());
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| drop(decode_animation(&b)))).is_ok(),
            "panic on mutation round {round}"
        );
    }
    // Rewrite every fcTL with extreme values, fixing the CRC.
    let crc = |b: &[u8]| rc_core::util::checksum::crc32(b);
    let extremes: [(u32, u32, u32, u32, u8, u8); 8] = [
        (0, 0, 0, 0, 0, 0),
        (u32::MAX, u32::MAX, 0, 0, 0, 0),
        (1, 1, u32::MAX, u32::MAX, 0, 0),
        (W, H, 1, 1, 0, 0),
        (W + 1, H, 0, 0, 0, 0),
        (4, 4, 2, 2, 9, 0),
        (4, 4, 2, 2, 0, 9),
        (W, H, 0, 0, 2, 1),
    ];
    for (w, h, x, y, dispose, blend) in extremes {
        let mut b = bytes.clone();
        let mut i = 8;
        while i + 12 <= b.len() {
            let len = u32::from_be_bytes(b[i..i + 4].try_into().unwrap()) as usize;
            if &b[i + 4..i + 8] == b"fcTL" {
                let body = &mut b[i + 8..i + 8 + len];
                body[4..8].copy_from_slice(&w.to_be_bytes());
                body[8..12].copy_from_slice(&h.to_be_bytes());
                body[12..16].copy_from_slice(&x.to_be_bytes());
                body[16..20].copy_from_slice(&y.to_be_bytes());
                body[24] = dispose;
                body[25] = blend;
                let c = crc(&b[i + 4..i + 8 + len]);
                b[i + 8 + len..i + 12 + len].copy_from_slice(&c.to_be_bytes());
            }
            i += 12 + len;
        }
        let r = catch_unwind(AssertUnwindSafe(|| decode_animation(&b)));
        assert!(r.is_ok(), "panic on frame geometry {w}x{h} at {x},{y} dispose {dispose} blend {blend}");
    }
}
