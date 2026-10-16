//! VP8 in-loop deblocking filters (RFC 6386 section 15): a simple filter touching two pixels on
//! each side of an edge, and a "normal" filter that also uses the inner pixels and adapts to
//! high-edge-variance.

/// Per-macroblock filter parameters.
#[derive(Clone, Copy, Default, Debug)]
pub struct FilterInfo {
    /// Edge limit; 0 disables filtering for this macroblock.
    pub limit: u8,
    /// Interior difference limit.
    pub ilevel: u8,
    /// High edge variance threshold.
    pub hev_thresh: u8,
    /// Whether the edges between the 4x4 sub-blocks are filtered too.
    pub inner: bool,
}

#[inline(always)]
fn sclip1(v: i32) -> i32 {
    v.clamp(-128, 127)
}

#[inline(always)]
fn sclip2(v: i32) -> i32 {
    v.clamp(-16, 15)
}

#[inline(always)]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline(always)]
fn px(p: &[u8], at: usize, step: usize, k: isize) -> i32 {
    p[(at as isize + k * step as isize) as usize] as i32
}

/// Adjust the two pixels next to the edge (used by the simple filter and for high-variance edges).
#[inline(always)]
fn filter2(p: &mut [u8], at: usize, step: usize) {
    let (p1, p0, q0, q1) = (px(p, at, step, -2), px(p, at, step, -1), px(p, at, step, 0), px(p, at, step, 1));
    let a = 3 * (q0 - p0) + sclip1(p1 - q1);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    p[at - step] = clip1(p0 + a2);
    p[at] = clip1(q0 - a1);
}

/// Adjust two pixels on each side (inner edges, low variance).
#[inline(always)]
fn filter4(p: &mut [u8], at: usize, step: usize) {
    let (p1, p0, q0, q1) = (px(p, at, step, -2), px(p, at, step, -1), px(p, at, step, 0), px(p, at, step, 1));
    let a = 3 * (q0 - p0);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    let a3 = (a1 + 1) >> 1;
    p[at - 2 * step] = clip1(p1 + a3);
    p[at - step] = clip1(p0 + a2);
    p[at] = clip1(q0 - a1);
    p[at + step] = clip1(q1 - a3);
}

/// Adjust three pixels on each side (macroblock edges, low variance).
#[inline(always)]
fn filter6(p: &mut [u8], at: usize, step: usize) {
    let (p2, p1, p0) = (px(p, at, step, -3), px(p, at, step, -2), px(p, at, step, -1));
    let (q0, q1, q2) = (px(p, at, step, 0), px(p, at, step, 1), px(p, at, step, 2));
    let a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    let a1 = (27 * a + 63) >> 7;
    let a2 = (18 * a + 63) >> 7;
    let a3 = (9 * a + 63) >> 7;
    p[at - 3 * step] = clip1(p2 + a3);
    p[at - 2 * step] = clip1(p1 + a2);
    p[at - step] = clip1(p0 + a1);
    p[at] = clip1(q0 - a1);
    p[at + step] = clip1(q1 - a2);
    p[at + 2 * step] = clip1(q2 - a3);
}

#[inline(always)]
fn hev(p: &[u8], at: usize, step: usize, thresh: i32) -> bool {
    let (p1, p0, q0, q1) = (px(p, at, step, -2), px(p, at, step, -1), px(p, at, step, 0), px(p, at, step, 1));
    (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh
}

#[inline(always)]
fn needs_filter(p: &[u8], at: usize, step: usize, t: i32) -> bool {
    let (p1, p0, q0, q1) = (px(p, at, step, -2), px(p, at, step, -1), px(p, at, step, 0), px(p, at, step, 1));
    4 * (p0 - q0).abs() + (p1 - q1).abs() <= t
}

#[inline(always)]
fn needs_filter2(p: &[u8], at: usize, step: usize, t: i32, it: i32) -> bool {
    let (p3, p2, p1, p0) = (px(p, at, step, -4), px(p, at, step, -3), px(p, at, step, -2), px(p, at, step, -1));
    let (q0, q1, q2, q3) = (px(p, at, step, 0), px(p, at, step, 1), px(p, at, step, 2), px(p, at, step, 3));
    if 4 * (p0 - q0).abs() + (p1 - q1).abs() > t {
        return false;
    }
    (p3 - p2).abs() <= it
        && (p2 - p1).abs() <= it
        && (p1 - p0).abs() <= it
        && (q3 - q2).abs() <= it
        && (q2 - q1).abs() <= it
        && (q1 - q0).abs() <= it
}

/// Simple filter along an edge of `len` pixels. `step` crosses the edge, `along` runs along it.
fn simple_edge(p: &mut [u8], at: usize, step: usize, along: usize, len: usize, thresh: i32) {
    let t2 = 2 * thresh + 1;
    for i in 0..len {
        let a = at + i * along;
        if needs_filter(p, a, step, t2) {
            filter2(p, a, step);
        }
    }
}

/// Normal filter along an edge. `mb_edge` selects the wider macroblock-edge adjustment.
#[allow(clippy::too_many_arguments)]
fn normal_edge(
    p: &mut [u8],
    at: usize,
    step: usize,
    along: usize,
    len: usize,
    thresh: i32,
    ithresh: i32,
    hev_thresh: i32,
    mb_edge: bool,
) {
    let t2 = 2 * thresh + 1;
    for i in 0..len {
        let a = at + i * along;
        if needs_filter2(p, a, step, t2, ithresh) {
            if hev(p, a, step, hev_thresh) {
                filter2(p, a, step);
            } else if mb_edge {
                filter6(p, a, step);
            } else {
                filter4(p, a, step);
            }
        }
    }
}

/// A plane being filtered: pixels plus the row stride.
pub struct Plane<'a> {
    pub data: &'a mut [u8],
    pub stride: usize,
}

/// Filter one macroblock's edges in place. `(mbx, mby)` is its position; planes are the full
/// frame planes. `simple` selects the simple filter (luma only).
pub fn filter_macroblock(
    info: &FilterInfo,
    simple: bool,
    mbx: usize,
    mby: usize,
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
) {
    if info.limit == 0 {
        return;
    }
    let limit = info.limit as i32;
    let ilevel = info.ilevel as i32;
    let hev_t = info.hev_thresh as i32;
    let ys = y.stride;
    let yat = mby * 16 * ys + mbx * 16;
    if simple {
        if mbx > 0 {
            simple_edge(y.data, yat, 1, ys, 16, limit + 4);
        }
        if info.inner {
            for k in 1..4 {
                simple_edge(y.data, yat + 4 * k, 1, ys, 16, limit);
            }
        }
        if mby > 0 {
            simple_edge(y.data, yat, ys, 1, 16, limit + 4);
        }
        if info.inner {
            for k in 1..4 {
                simple_edge(y.data, yat + 4 * k * ys, ys, 1, 16, limit);
            }
        }
        return;
    }
    let cs = u.stride;
    let cat = mby * 8 * cs + mbx * 8;
    if mbx > 0 {
        normal_edge(y.data, yat, 1, ys, 16, limit + 4, ilevel, hev_t, true);
        normal_edge(u.data, cat, 1, cs, 8, limit + 4, ilevel, hev_t, true);
        normal_edge(v.data, cat, 1, cs, 8, limit + 4, ilevel, hev_t, true);
    }
    if info.inner {
        for k in 1..4 {
            normal_edge(y.data, yat + 4 * k, 1, ys, 16, limit, ilevel, hev_t, false);
        }
        normal_edge(u.data, cat + 4, 1, cs, 8, limit, ilevel, hev_t, false);
        normal_edge(v.data, cat + 4, 1, cs, 8, limit, ilevel, hev_t, false);
    }
    if mby > 0 {
        normal_edge(y.data, yat, ys, 1, 16, limit + 4, ilevel, hev_t, true);
        normal_edge(u.data, cat, cs, 1, 8, limit + 4, ilevel, hev_t, true);
        normal_edge(v.data, cat, cs, 1, 8, limit + 4, ilevel, hev_t, true);
    }
    if info.inner {
        for k in 1..4 {
            normal_edge(y.data, yat + 4 * k * ys, ys, 1, 16, limit, ilevel, hev_t, false);
        }
        normal_edge(u.data, cat + 4 * cs, cs, 1, 8, limit, ilevel, hev_t, false);
        normal_edge(v.data, cat + 4 * cs, cs, 1, 8, limit, ilevel, hev_t, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_areas_are_left_alone() {
        let mut y = vec![100u8; 32 * 32];
        let mut u = vec![100u8; 16 * 16];
        let mut v = vec![100u8; 16 * 16];
        let info = FilterInfo { limit: 40, ilevel: 5, hev_thresh: 1, inner: true };
        let before = (y.clone(), u.clone(), v.clone());
        for simple in [false, true] {
            filter_macroblock(
                &info,
                simple,
                1,
                1,
                &mut Plane { data: &mut y, stride: 32 },
                &mut Plane { data: &mut u, stride: 16 },
                &mut Plane { data: &mut v, stride: 16 },
            );
        }
        assert_eq!((y, u, v), before);
    }

    #[test]
    fn a_small_step_is_smoothed_and_a_big_one_kept() {
        for (step, smoothed) in [(6i32, true), (200, false)] {
            let mut y = vec![0u8; 32 * 32];
            for row in 0..32 {
                for col in 0..32 {
                    y[row * 32 + col] = if col < 16 { 100 } else { (100 + step) as u8 };
                }
            }
            let orig = y.clone();
            let mut u = vec![128u8; 16 * 16];
            let mut v = vec![128u8; 16 * 16];
            let info = FilterInfo { limit: 30, ilevel: 10, hev_thresh: 0, inner: false };
            filter_macroblock(
                &info,
                false,
                1,
                0,
                &mut Plane { data: &mut y, stride: 32 },
                &mut Plane { data: &mut u, stride: 16 },
                &mut Plane { data: &mut v, stride: 16 },
            );
            assert_eq!(y != orig, smoothed, "step {step}");
        }
    }
}
