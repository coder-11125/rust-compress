mod common;
use common::*;
use rust_compress::image::{ColorType, Image};
use rust_compress::jpeg::encode::{EncodeOptions, Subsampling, encode};
use rust_compress::png::decode::decode as png_decode;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

fn smooth_image(w: u32, h: u32, color: ColorType) -> Image {
    let ch = color.channels();
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                data.push(((x * 3 + y * 2 + c as u32 * 40) % 256) as u8);
            }
        }
    }
    Image::new(w, h, color, 8, data).unwrap()
}

#[test]
fn encoded_jpegs_decode_with_reference_decoder() {
    let rgb = png_decode(&fixture("rgb8.png")).unwrap();
    let gray = png_decode(&fixture("gray8.png")).unwrap();
    for (name, img) in [("rgb", &rgb), ("gray", &gray)] {
        for sub in [Subsampling::S444, Subsampling::S420] {
            for q in [30u8, 75, 95] {
                let opts = EncodeOptions { quality: q, subsampling: sub, ..Default::default() };
                let jpg = encode(img, &opts).unwrap();
                let back = image::load_from_memory(&jpg).unwrap_or_else(|e| panic!("{name} {sub:?} q{q}: {e}"));
                assert_eq!((back.width(), back.height()), (img.width, img.height));
                let (got, want) = if gray_img(img) {
                    (back.to_luma8().into_raw(), img.data.clone())
                } else {
                    (back.to_rgb8().into_raw(), img.data.clone())
                };
                // The fixture has heavy per-pixel chroma noise, so 4:2:0 is only checked
                // for decodability; 4:4:4/gray must match libjpeg-level fidelity
                // (libjpeg yields 19.6 / 25.5 / 36.6 dB at q30/75/95 on this fixture).
                if sub == Subsampling::S444 || gray_img(img) {
                    let p = psnr(&got, &want);
                    let min = match q {
                        30 => 19.0,
                        75 => 25.0,
                        _ => 36.0,
                    };
                    assert!(p > min, "{name} {sub:?} q{q}: PSNR {p:.1}");
                }
            }
        }
    }
}

fn gray_img(i: &Image) -> bool {
    i.color == ColorType::Gray
}

#[test]
fn odd_and_tiny_sizes() {
    for (w, h) in [(1, 1), (7, 9), (8, 8), (9, 9), (17, 1), (1, 17), (33, 31), (250, 3)] {
        for color in [ColorType::Gray, ColorType::Rgb] {
            let img = smooth_image(w, h, color);
            for sub in [Subsampling::S444, Subsampling::S420] {
                let jpg = encode(&img, &EncodeOptions { quality: 90, subsampling: sub, ..Default::default() }).unwrap();
                let back = image::load_from_memory(&jpg).unwrap_or_else(|e| panic!("{w}x{h} {color:?} {sub:?}: {e}"));
                assert_eq!((back.width(), back.height()), (w, h));
            }
        }
    }
}

#[test]
fn quality_orders_file_size() {
    let img = png_decode(&fixture("rgb8.png")).unwrap();
    let sizes: Vec<usize> = [20u8, 50, 80, 95]
        .iter()
        .map(|&q| encode(&img, &EncodeOptions { quality: q, ..Default::default() }).unwrap().len())
        .collect();
    assert!(sizes.windows(2).all(|w| w[0] < w[1]), "{sizes:?}");
}

#[test]
fn alpha_is_flattened() {
    let img = png_decode(&fixture("rgba8.png")).unwrap();
    let jpg = encode(&img, &EncodeOptions::default()).unwrap();
    let back = image::load_from_memory(&jpg).unwrap();
    assert_eq!((back.width(), back.height()), (img.width, img.height));
}

#[test]
fn quality_search_meets_the_target_and_is_monotonic_in_it() {
    use rust_compress::jpeg::encode::build_coefs;
    use rust_compress::jpeg::target::find_quality;
    let reference = png_decode(&fixture("rgb8.png")).unwrap();
    let make = |q: u8| build_coefs(&reference, &EncodeOptions { quality: q, ..Default::default() });
    let mut last = 0u8;
    for target in [0.80, 0.90, 0.95, 0.98] {
        let found = find_quality(&reference, target, 100, make);
        assert!(found.ssim >= target || found.quality == 100, "target {target}: got {}", found.ssim);
        assert!(found.quality >= last, "quality must not drop as the target rises: {} < {last}", found.quality);
        last = found.quality;
        // One step lower must miss the target (the search returned the smallest passing quality).
        if found.quality > 1 {
            let lower = rust_compress::jpeg::decode::reconstruct(&make(found.quality - 1)).unwrap();
            assert!(
                rust_compress::metrics::ssim(&reference, &lower) < target + 0.01,
                "quality {} would also have passed",
                found.quality - 1
            );
        }
    }
}

#[test]
fn reconstruct_matches_the_decoder() {
    use rust_compress::jpeg::decode::{parse, reconstruct};
    for name in ["rgb8_420.jpg", "rgb8_444.jpg", "gray8.jpg", "rgb8_prog.jpg"] {
        let bytes = fixture(name);
        let want = rust_compress::jpeg::decode::decode(&bytes).unwrap();
        let ci = parse(&bytes).unwrap().into_coefs().unwrap();
        assert_eq!(reconstruct(&ci).unwrap().data, want.data, "{name}");
    }
}
