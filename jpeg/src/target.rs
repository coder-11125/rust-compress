//! Perceptual quality targeting: find the lowest JPEG quality whose reconstruction still scores at
//! least a target SSIM against the reference, instead of guessing a quality number.

use super::decode::reconstruct;
use super::encode::CoefImage;
use rc_core::image::Image;
use rc_core::metrics::ssim;

/// Result of a quality search.
pub struct Found {
    pub quality: u8,
    pub ssim: f64,
    pub coefs: CoefImage,
}

/// Bisect `quality` in `1..=max_quality` for the smallest value whose coefficients (from `make`)
/// reconstruct to at least `target` SSIM against `reference`. If even `max_quality` misses the
/// target, that best-effort result is returned. SSIM is assumed to grow with quality; every
/// accepted candidate has been checked against the target, so a non-monotonic stretch can only
/// make the answer slightly larger than necessary, never worse than the target.
pub fn find_quality(reference: &Image, target: f64, max_quality: u8, mut make: impl FnMut(u8) -> CoefImage) -> Found {
    let score = |ci: &CoefImage| -> f64 { reconstruct(ci).map(|img| ssim(reference, &img)).unwrap_or(0.0) };
    let (mut lo, mut hi) = (1u8, max_quality.max(1));
    let top = make(hi);
    let top_score = score(&top);
    let mut best = Found { quality: hi, ssim: top_score, coefs: top };
    if top_score < target {
        return best;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let ci = make(mid);
        let s = score(&ci);
        if s >= target {
            hi = mid;
            best = Found { quality: mid, ssim: s, coefs: ci };
        } else {
            lo = mid + 1;
        }
    }
    best
}
