//! Developer tool: write adversarial animations as APNG (`<dir>/<name>.png`) with the exact canvases they must
//! decode to as raw RGBA (`<dir>/<name>.rgba`, frames back to back), for checking other decoders.
//! usage: apng_cases <dir>
use rc_core::animation::{AnimFrame, Animation};
use rc_core::image::{ColorType, Image};
use rc_png::apng::encode_animation;
use rc_png::encode::EncodeOptions;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

fn frame(w: u32, h: u32, alpha: bool, px: &dyn Fn(u32, u32) -> [u8; 4]) -> Image {
    let mut d = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let p = px(x, y);
            d.extend_from_slice(if alpha { &p[..] } else { &p[..3] });
        }
    }
    Image::new(w, h, if alpha { ColorType::Rgba } else { ColorType::Rgb }, 8, d).unwrap()
}

fn anim(w: u32, h: u32, frames: Vec<Image>) -> Animation {
    let frames =
        frames.into_iter().enumerate().map(|(i, image)| AnimFrame { image, duration_ms: 50 + 10 * i as u32 }).collect();
    Animation { width: w, height: h, frames, loop_forever: true }
}

fn cases() -> Vec<(&'static str, Animation)> {
    let mut v = Vec::new();
    // Opaque sprite moving over a background.
    for alpha in [false, true] {
        let (w, h) = (96u32, 64u32);
        let fs = (0..8u32)
            .map(|t| {
                frame(w, h, alpha, &|x, y| {
                    if x >= 4 + 7 * t && x < 20 + 7 * t && (8..24).contains(&y) {
                        [220, 40 + 20 * t as u8, 30, 255]
                    } else if alpha && (x / 8 + y / 8) % 2 == 0 {
                        [0, 0, 0, 0]
                    } else {
                        [(x * 2) as u8, (y * 3) as u8, 100, 255]
                    }
                })
            })
            .collect();
        v.push((if alpha { "sprite_rgba" } else { "sprite_rgb" }, anim(w, h, fs)));
    }
    // Content that appears and disappears (disappearing needs SOURCE) with partial alpha mixed in.
    let mut r = Rng(7);
    let (w, h) = (40u32, 30u32);
    let fs = (0..10)
        .map(|_| {
            let mut px = vec![[0u8; 4]; (w * h) as usize];
            for p in px.iter_mut() {
                *p = match r.next() % 6 {
                    0 => [r.next() as u8, r.next() as u8, r.next() as u8, 255],
                    1 => [r.next() as u8, r.next() as u8, r.next() as u8, 128],
                    _ => [0, 0, 0, 0],
                };
            }
            frame(w, h, true, &|x, y| px[(y * w + x) as usize])
        })
        .collect();
    v.push(("random_alpha", anim(w, h, fs)));
    // Mostly static with sparse opaque changes (the case OVER is for) on a partly transparent background.
    let mut r = Rng(99);
    let mut base: Vec<[u8; 4]> = (0..(w * h) as usize)
        .map(|i| {
            if i % 7 == 0 {
                [0, 0, 0, 0]
            } else if i % 5 == 0 {
                [200, 100, 50, 90]
            } else {
                [30, 60, 90, 255]
            }
        })
        .collect();
    let mut fs = Vec::new();
    for _ in 0..12 {
        for _ in 0..(r.next() % 6) {
            let i = (r.next() % (w * h)) as usize;
            base[i] = [r.next() as u8, r.next() as u8, r.next() as u8, 255];
        }
        fs.push(frame(w, h, true, &|x, y| base[(y * w + x) as usize]));
    }
    v.push(("sparse_opaque", anim(w, h, fs)));
    // Repeated frames, a single changed corner pixel, and the last pixel.
    let still = |c: [u8; 4]| frame(16, 12, true, &|_, _| c);
    let corner = |cx: u32, cy: u32| {
        frame(16, 12, true, &|x, y| if (x, y) == (cx, cy) { [255, 0, 0, 255] } else { [10, 20, 30, 255] })
    };
    v.push((
        "repeats_corners",
        anim(
            16,
            12,
            vec![
                still([10, 20, 30, 255]),
                still([10, 20, 30, 255]),
                corner(0, 0),
                corner(15, 11),
                corner(15, 11),
                corner(0, 11),
                corner(15, 0),
            ],
        ),
    ));
    // Degenerate sizes.
    v.push((
        "one_pixel",
        anim(1, 1, vec![still_px([1, 2, 3, 255]), still_px([9, 8, 7, 255]), still_px([9, 8, 7, 255])]),
    ));
    let strip = |t: u32| frame(1, 20, true, &|_, y| if y == t { [255, 255, 0, 255] } else { [0, 0, 0, 255] });
    v.push(("one_wide", anim(1, 20, vec![strip(0), strip(5), strip(19)])));
    let row = |t: u32| frame(25, 1, false, &|x, _| if x == t { [255, 255, 0, 255] } else { [0, 0, 0, 255] });
    v.push(("one_high", anim(25, 1, vec![row(0), row(24), row(3)])));
    v.push((
        "two_frames",
        anim(
            8,
            8,
            vec![
                frame(8, 8, true, &|x, _| [x as u8 * 30, 0, 0, 255]),
                frame(8, 8, true, &|x, y| if y > 3 { [x as u8 * 30, 255, 0, 255] } else { [x as u8 * 30, 0, 0, 255] }),
            ],
        ),
    ));
    v
}

fn still_px(c: [u8; 4]) -> Image {
    frame(1, 1, true, &|_, _| c)
}

fn main() {
    let dir = std::env::args().nth(1).expect("output directory");
    for (name, a) in cases() {
        for (clean, tag) in [(true, ""), (false, "_keephidden")] {
            let png = encode_animation(&a, &EncodeOptions { clean_alpha: clean, ..Default::default() }).unwrap();
            std::fs::write(format!("{dir}/{name}{tag}.png"), png).unwrap();
            let mut raw = Vec::new();
            for f in &a.frames {
                for p in f.image.data.chunks_exact(f.image.color.channels()) {
                    match f.image.color {
                        ColorType::Rgba => raw.extend_from_slice(p),
                        _ => raw.extend_from_slice(&[p[0], p[1], p[2], 255]),
                    }
                }
            }
            std::fs::write(format!("{dir}/{name}{tag}.rgba"), raw).unwrap();
            std::fs::write(format!("{dir}/{name}{tag}.info"), format!("{} {} {}", a.width, a.height, a.frames.len()))
                .unwrap();
        }
    }
}
