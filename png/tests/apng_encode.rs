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
