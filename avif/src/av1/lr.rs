//! Loop restoration (AV1 specification section 7.17): the Wiener and self-guided (sgrproj) filters.
//!
//! The picture is cut into horizontal *stripes* of 64 luma rows, the first one 8 rows short, so every stripe
//! boundary sits 8 rows above a multiple of 64. Rows inside a stripe are read from the CDEF output; the rows
//! just above and below (at most two) come from the deblocked, pre-CDEF picture, and further rows repeat the
//! nearest of those. Left, right, top and bottom picture edges repeat the edge sample. Inside a stripe the
//! picture is cut into restoration units, each with its own filter and coefficients.

use super::frame::{Frame, Plane};

/// Restoration filter chosen for one unit (after `RESTORE_SWITCHABLE` has been resolved).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LrType {
    #[default]
    None,
    Wiener,
    Sgrproj,
}

/// The restoration parameters of one unit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LrUnit {
    pub kind: LrType,
    /// Wiener taps `[pass][i]` with pass 0 = vertical, 1 = horizontal, as the specification's `LrWiener`.
    /// Only the three outer taps are coded; the centre tap is `128 - 2 * (c0 + c1 + c2)`.
    pub wiener: [[i8; 3]; 2],
    /// Index into `Sgr_Params` (0..16).
    pub sgr_set: u8,
    /// The two projection weights `LrSgrXqd`.
    pub sgr_xqd: [i8; 2],
}

/// The restoration units of one plane, row-major.
#[derive(Clone, Debug, Default)]
pub struct LrPlane {
    /// Unit size in samples of this plane (`LoopRestorationSize`: 64, 128 or 256 for luma; 32..256 for chroma).
    pub unit_size: usize,
    pub units_per_row: usize,
    pub units_per_col: usize,
    pub units: Vec<LrUnit>,
}

impl LrPlane {
    /// A plane of `plane_width` x `plane_height` visible samples (`Round2(UpscaledWidth, sub_x)` and
    /// `Round2(FrameHeight, sub_y)`), every unit set to [`LrType::None`].
    pub fn new(unit_size: usize, plane_width: usize, plane_height: usize) -> LrPlane {
        let units_per_row = count_units(unit_size, plane_width);
        let units_per_col = count_units(unit_size, plane_height);
        LrPlane {
            unit_size,
            units_per_row,
            units_per_col,
            units: vec![LrUnit::default(); units_per_row * units_per_col],
        }
    }

    fn is_identity(&self) -> bool {
        self.units.iter().all(|u| u.kind == LrType::None)
    }
}

/// `count_units_in_frame`: the last unit absorbs a remainder of less than half a unit.
pub fn count_units(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

const SGRPROJ_RST_BITS: u32 = 4;
const SGRPROJ_PRJ_BITS: u32 = 7;
const SGRPROJ_MTABLE_BITS: u32 = 20;
const SGRPROJ_SGR_BITS: u32 = 8;
const SGRPROJ_RECIP_BITS: u32 = 12;
const FILTER_BITS: u32 = 7;

/// `Sgr_Params`: `[r0, e0, r1, e1]` per set. Pass 0 uses the 5x5 window (`r = 2`), pass 1 the 3x3 window
/// (`r = 1`); a radius of 0 disables that pass.
pub(crate) const SGR_PARAMS: [[u8; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

/// Applies loop restoration to the whole picture.
///
/// * `cdef_out` is the post-CDEF (and post-superres) picture, `deblocked` the post-deblock, pre-CDEF picture
///   brought to the same (upscaled) size. Both must have identical geometry.
/// * `out` must be shaped like `cdef_out`; it is overwritten: samples outside the visible area and the planes
///   or units without restoration are copied from `cdef_out`.
/// * `planes` has one entry per plane of the frame. A plane whose units are all [`LrType::None`] is copied.
///
/// The visible plane size is `Round2(frame.width, sub_x)` x `Round2(frame.height, sub_y)`.
pub fn loop_restoration_frame(cdef_out: &Frame, deblocked: &Frame, out: &mut Frame, planes: &[LrPlane]) {
    assert_eq!(cdef_out.planes.len(), out.planes.len());
    assert_eq!(cdef_out.planes.len(), deblocked.planes.len());
    assert_eq!(planes.len(), cdef_out.planes.len());
    for (dst, src) in out.planes.iter_mut().zip(&cdef_out.planes) {
        dst.data.clear();
        dst.data.extend_from_slice(&src.data);
        dst.stride = src.stride;
        dst.width = src.width;
        dst.height = src.height;
    }
    for (index, lr) in planes.iter().enumerate() {
        if lr.is_identity() {
            continue;
        }
        let (sx, sy) = if index == 0 { (0, 0) } else { (cdef_out.sub_x as usize, cdef_out.sub_y as usize) };
        let plane_w = (cdef_out.width + sx) >> sx;
        let plane_h = (cdef_out.height + sy) >> sy;
        let ctx = PlaneCtx {
            cdef: &cdef_out.planes[index],
            deblocked: &deblocked.planes[index],
            plane_w,
            plane_h,
            bit_depth: cdef_out.bit_depth as u32,
        };
        restore_plane(&ctx, lr, sy, &mut out.planes[index]);
    }
}

pub(crate) struct PlaneCtx<'a> {
    pub(crate) cdef: &'a Plane,
    pub(crate) deblocked: &'a Plane,
    pub(crate) plane_w: usize,
    pub(crate) plane_h: usize,
    pub(crate) bit_depth: u32,
}

fn restore_plane(ctx: &PlaneCtx, lr: &LrPlane, sub_y: usize, out: &mut Plane) {
    assert!(lr.units_per_row * lr.units_per_col == lr.units.len() && lr.unit_size > 0);
    let stripe_h = 64 >> sub_y;
    let mut stripe = 0usize;
    loop {
        // Rows of this stripe in plane coordinates; the first stripe starts 8 luma rows above the picture.
        let start = ((64 * stripe) as isize - 8) >> sub_y;
        let end = start + stripe_h as isize - 1;
        if start >= ctx.plane_h as isize {
            break;
        }
        let y0 = start.max(0) as usize;
        let y1 = (end.min(ctx.plane_h as isize - 1)) as usize;
        let unit_row = ((stripe * stripe_h) / lr.unit_size).min(lr.units_per_col - 1);
        for unit_col in 0..lr.units_per_row {
            let unit = &lr.units[unit_row * lr.units_per_row + unit_col];
            if unit.kind == LrType::None {
                continue;
            }
            let x0 = unit_col * lr.unit_size;
            let x1 = if unit_col + 1 == lr.units_per_row { ctx.plane_w } else { (unit_col + 1) * lr.unit_size };
            if x0 >= x1 {
                continue;
            }
            let block = Block { x0, y0, w: x1 - x0, h: y1 - y0 + 1, stripe_start: start, stripe_end: end };
            let window = Window::gather(ctx, &block);
            let mut result = vec![0u16; block.w * block.h];
            match unit.kind {
                LrType::Wiener => wiener(&window, ctx.bit_depth, &unit.wiener, &mut result),
                LrType::Sgrproj => sgrproj(&window, ctx.bit_depth, unit, &mut result),
                LrType::None => unreachable!(),
            }
            for r in 0..block.h {
                let row = out.row_mut(block.y0 + r);
                row[x0..x1].copy_from_slice(&result[r * block.w..(r + 1) * block.w]);
            }
        }
        stripe += 1;
    }
}

/// One unit within one stripe.
pub(crate) struct Block {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) stripe_start: isize,
    pub(crate) stripe_end: isize,
}

/// Margin of source samples kept around a block: Wiener needs 3, the box filters need 2 + 1.
pub(crate) const MARGIN: usize = 3;

/// The source samples around a block as `get_source_sample` returns them, with a margin of [`MARGIN`].
pub(crate) struct Window {
    pub(crate) data: Vec<u16>,
    pub(crate) stride: usize,
}

impl Window {
    pub(crate) fn gather(ctx: &PlaneCtx, b: &Block) -> Window {
        let stride = b.w + 2 * MARGIN;
        let rows = b.h + 2 * MARGIN;
        let mut data = vec![0u16; stride * rows];
        let last_x = ctx.plane_w as isize - 1;
        let last_y = ctx.plane_h as isize - 1;
        for r in 0..rows {
            let mut y = (b.y0 as isize + r as isize - MARGIN as isize).clamp(0, last_y);
            let plane = if y < b.stripe_start {
                y = y.max(b.stripe_start - 2);
                ctx.deblocked
            } else if y > b.stripe_end {
                y = y.min(b.stripe_end + 2);
                ctx.deblocked
            } else {
                ctx.cdef
            };
            let src = plane.row(y as usize);
            let dst = &mut data[r * stride..(r + 1) * stride];
            for (c, d) in dst.iter_mut().enumerate() {
                let x = (b.x0 as isize + c as isize - MARGIN as isize).clamp(0, last_x);
                *d = src[x as usize];
            }
        }
        Window { data, stride }
    }

    /// The sample at block-relative `(x, y)`, each of which may reach `MARGIN` outside the block.
    #[inline(always)]
    pub(crate) fn at(&self, x: isize, y: isize) -> u16 {
        self.data[(y + MARGIN as isize) as usize * self.stride + (x + MARGIN as isize) as usize]
    }
}

#[inline(always)]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// Expands the three coded taps to the symmetric 7-tap filter.
fn wiener_taps(c: &[i8; 3]) -> [i32; 7] {
    let (c0, c1, c2) = (c[0] as i32, c[1] as i32, c[2] as i32);
    [c0, c1, c2, 128 - 2 * (c0 + c1 + c2), c2, c1, c0]
}

/// 7.17.4: horizontal pass into a clipped intermediate, then the vertical pass.
pub(crate) fn wiener(win: &Window, bit_depth: u32, coeffs: &[[i8; 3]; 2], out: &mut [u16]) {
    let (round0, round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
    let vfilter = wiener_taps(&coeffs[0]);
    let hfilter = wiener_taps(&coeffs[1]);
    let w = win.stride - 2 * MARGIN;
    let h = win.data.len() / win.stride - 2 * MARGIN;
    let offset = 1i32 << (bit_depth + FILTER_BITS - round0 - 1);
    let limit = (1i32 << (bit_depth + 1 + FILTER_BITS - round0)) - 1;
    let mut inter = vec![0i32; (h + 6) * w];
    for r in 0..h + 6 {
        let row = &win.data[r * win.stride..(r + 1) * win.stride];
        for c in 0..w {
            let mut sum = 0i32;
            for t in 0..7 {
                sum += hfilter[t] * row[c + t] as i32;
            }
            inter[r * w + c] = round2(sum, round0).clamp(-offset, limit - offset);
        }
    }
    let max = (1i32 << bit_depth) - 1;
    for r in 0..h {
        for c in 0..w {
            let mut sum = 0i32;
            for t in 0..7 {
                sum += vfilter[t] * inter[(r + t) * w + c];
            }
            out[r * w + c] = round2(sum, round1).clamp(0, max) as u16;
        }
    }
}

/// 7.17.3: the self-guided filter, a blend of up to two box-filter passes projected onto the source.
fn sgrproj(win: &Window, bit_depth: u32, unit: &LrUnit, out: &mut [u16]) {
    let w = win.stride - 2 * MARGIN;
    let h = win.data.len() / win.stride - 2 * MARGIN;
    let params = &SGR_PARAMS[(unit.sgr_set & 15) as usize];
    let (r0, e0, r1, e1) = (params[0] as usize, params[1] as u32, params[2] as usize, params[3] as u32);
    let w0 = unit.sgr_xqd[0] as i32;
    let w1 = unit.sgr_xqd[1] as i32;
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let flt0 = (r0 != 0).then(|| box_filter(win, bit_depth, r0, e0, w, h));
    let flt1 = (r1 != 0).then(|| box_filter(win, bit_depth, r1, e1, w, h));
    let max = (1i32 << bit_depth) - 1;
    for i in 0..h {
        for j in 0..w {
            let u = (win.at(j as isize, i as isize) as i32) << SGRPROJ_RST_BITS;
            let mut v = w1 * u;
            v += w0 * flt0.as_ref().map_or(u, |f| f[i * w + j]);
            v += w2 * flt1.as_ref().map_or(u, |f| f[i * w + j]);
            out[i * w + j] = round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS).clamp(0, max) as u16;
        }
    }
}

/// The box filter process: per-pixel `a`/`b` over a `(2r+1)^2` window, then a weighted 3x3 blend of the
/// neighbours' values, returned at `SGRPROJ_RST_BITS` extra precision.
pub(crate) fn box_filter(win: &Window, bit_depth: u32, r: usize, eps: u32, w: usize, h: usize) -> Vec<i32> {
    let d = 2 * r + 1;
    let n = (d * d) as u32;
    let n2e = n * n * eps;
    let s = ((1u32 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let one_over_n = ((1u32 << SGRPROJ_RECIP_BITS) + n / 2) / n;
    let bd8 = bit_depth - 8;

    // Horizontal window sums for every row of the margin-padded block, for columns -1..=w.
    let aw = w + 2;
    let rows = h + 2 * MARGIN;
    let mut hsum = vec![0u32; rows * aw];
    let mut hsq = vec![0u32; rows * aw];
    for y in 0..rows {
        let base = y * win.stride;
        for j in 0..aw {
            // Column j of the A/B grid is block column j - 1; its window starts r samples to the left.
            let first = base + MARGIN + j - 1 - r;
            let (mut sum, mut sq) = (0u32, 0u32);
            for &c in &win.data[first..first + d] {
                let c = c as u32;
                sum += c;
                sq += c * c;
            }
            hsum[y * aw + j] = sum;
            hsq[y * aw + j] = sq;
        }
    }

    // A and B for rows -1..=h (grid row i + 1). With r = 2 only odd rows are ever read.
    let mut a_grid = vec![0i32; (h + 2) * aw];
    let mut b_grid = vec![0i32; (h + 2) * aw];
    for gi in 0..h + 2 {
        let i = gi as isize - 1;
        if r == 2 && i & 1 == 0 {
            continue;
        }
        for j in 0..aw {
            let top = (i + MARGIN as isize - r as isize) as usize;
            let (mut sum, mut sq) = (0u32, 0u32);
            for y in top..top + d {
                sum += hsum[y * aw + j];
                sq += hsq[y * aw + j];
            }
            let a = (sq + ((1u32 << (2 * bd8)) >> 1)) >> (2 * bd8);
            let dd = (sum + ((1u32 << bd8) >> 1)) >> bd8;
            let p = (a * n).saturating_sub(dd * dd);
            let z = ((p as u64 * s as u64 + (1u64 << (SGRPROJ_MTABLE_BITS - 1))) >> SGRPROJ_MTABLE_BITS) as u32;
            let a2 = if z >= 255 {
                256
            } else if z == 0 {
                1
            } else {
                ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)
            };
            let b2 = ((1u64 << SGRPROJ_SGR_BITS) - a2 as u64) * sum as u64 * one_over_n as u64;
            a_grid[gi * aw + j] = a2 as i32;
            b_grid[gi * aw + j] = ((b2 + (1u64 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS) as i32;
        }
    }

    let mut flt = vec![0i32; w * h];
    for i in 0..h {
        let shift = if r == 2 && i & 1 == 1 { 4 } else { 5 };
        for j in 0..w {
            let (mut a, mut b) = (0i32, 0i32);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let weight = if r == 2 {
                        if (i as isize + dy) & 1 != 0 { if dx == 0 { 6 } else { 5 } } else { 0 }
                    } else if dx == 0 || dy == 0 {
                        4
                    } else {
                        3
                    };
                    if weight != 0 {
                        let g = (i as isize + dy + 1) as usize * aw + (j as isize + dx + 1) as usize;
                        a += weight * a_grid[g];
                        b += weight * b_grid[g];
                    }
                }
            }
            let v = a * win.at(j as isize, i as isize) as i32 + b;
            flt[i * w + j] = round2(v, SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS);
        }
    }
    flt
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u32)) as i32
        }
    }

    /// A smooth, noisy test picture (coded area rounded up to 8 luma samples; the padding is random too).
    fn golden_frame(rng: &mut Rng, bd: u32, sub: (u8, u8), planes: usize, w: usize, h: usize) -> Frame {
        let max = (1i32 << bd) - 1;
        let mut out = Vec::new();
        for p in 0..planes {
            let (sx, sy) = if p == 0 { (0, 0) } else { sub };
            let (cw, ch) = ((w.div_ceil(8) * 8) >> sx, (h.div_ceil(8) * 8) >> sy);
            let mut plane = Plane::new(cw, ch);
            let base = rng.range(0, max);
            let amp = (max >> 4).max(1);
            for y in 0..ch {
                for x in 0..cw {
                    let v =
                        base + (((x * 3 + y * 5) as i32 * (max >> 8).max(1)) % (max + 1)) / 8 + rng.range(-amp, amp);
                    plane.data[y * cw + x] = v.clamp(0, max) as u16;
                }
            }
            out.push(plane);
        }
        Frame { planes: out, bit_depth: bd as u8, sub_x: sub.0, sub_y: sub.1, width: w, height: h }
    }

    /// `mode`: 0 = Wiener everywhere, 1 = self-guided everywhere (set cycles through all 16), 2 = a mix with `None`.
    #[allow(clippy::too_many_arguments)]
    fn golden_case(
        seed: u64,
        bd: u32,
        sub: (u8, u8),
        planes: usize,
        w: usize,
        h: usize,
        unit: usize,
        mode: u8,
    ) -> (Frame, Frame, Vec<LrPlane>) {
        let mut rng = Rng(seed);
        let cdef = golden_frame(&mut rng, bd, sub, planes, w, h);
        let deblocked = golden_frame(&mut rng, bd, sub, planes, w, h);
        let mut lr = Vec::new();
        let mut counter = 0u8;
        for p in 0..planes {
            let (sx, sy) = if p == 0 { (0, 0) } else { sub };
            let mut plane = LrPlane::new(unit, (w + sx as usize) >> sx, (h + sy as usize) >> sy);
            for u in plane.units.iter_mut() {
                let kind = match mode {
                    0 => LrType::Wiener,
                    1 => LrType::Sgrproj,
                    _ => [LrType::Wiener, LrType::None, LrType::Sgrproj][rng.range(0, 2) as usize],
                };
                let mins = [-5, -23, -17];
                let maxs = [10, 8, 46];
                let mut wiener = [[0i8; 3]; 2];
                for pass in wiener.iter_mut() {
                    for k in 0..3 {
                        pass[k] = rng.range(mins[k], maxs[k]) as i8;
                    }
                    if p > 0 {
                        pass[0] = 0;
                    }
                }
                let set = if mode == 1 {
                    counter += 1;
                    (counter - 1) % 16
                } else {
                    rng.range(0, 15) as u8
                };
                let xqd0 = if (10..=13).contains(&set) { 0 } else { rng.range(-96, 31) as i8 };
                *u = LrUnit { kind, wiener, sgr_set: set, sgr_xqd: [xqd0, rng.range(-32, 95) as i8] };
            }
            lr.push(plane);
        }
        (cdef, deblocked, lr)
    }

    /// FNV-1a over the visible samples of every plane.
    fn visible_hash(frame: &Frame) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for (p, plane) in frame.planes.iter().enumerate() {
            let (sx, sy) = if p == 0 { (0, 0) } else { (frame.sub_x as usize, frame.sub_y as usize) };
            for y in 0..(frame.height + sy) >> sy {
                for &v in &plane.row(y)[..(frame.width + sx) >> sx] {
                    for byte in v.to_le_bytes() {
                        h = (h ^ byte as u64).wrapping_mul(0x100_0000_01b3);
                    }
                }
            }
        }
        h
    }

    /// (seed, bit depth, subsampling, planes, width, height, unit size, mode)
    #[allow(clippy::type_complexity)]
    const GOLDEN_CASES: [(u64, u32, (u8, u8), usize, usize, usize, usize, u8); 8] = [
        (1, 8, (1, 1), 3, 67, 70, 64, 0),
        (2, 8, (1, 1), 3, 150, 200, 64, 1),
        (3, 10, (1, 0), 3, 131, 133, 64, 1),
        (4, 12, (0, 0), 3, 100, 129, 128, 0),
        (5, 12, (1, 1), 3, 97, 190, 64, 2),
        (6, 10, (0, 0), 1, 61, 59, 64, 2),
        (7, 8, (0, 0), 3, 33, 125, 256, 1),
        (8, 10, (1, 1), 3, 200, 70, 128, 2),
    ];

    /// Hashes of the visible output of `GOLDEN_CASES`, produced by rav1e's loop restoration (checked equal to this
    /// implementation on those inputs; rav1e's Wiener edge rows were first aligned with the specification and
    /// libaom/dav1d, which take the rows above the picture and below the last stripe from the CDEF output).
    const GOLDEN_HASHES: [u64; 8] = [
        0x49ec30de41cafad7,
        0xa7f644355a12e0d4,
        0xab3e9fe15beb5bab,
        0x57d241ca063624b2,
        0xb031ab7481f76b65,
        0xefbaf8d421afcc94,
        0x6d99b6be4a2bd46b,
        0x3e341bd2b65c1d91,
    ];

    #[test]
    fn matches_reference_vectors() {
        for (n, &(seed, bd, sub, planes, w, h, unit, mode)) in GOLDEN_CASES.iter().enumerate() {
            let (cdef, deblocked, lr) = golden_case(seed, bd, sub, planes, w, h, unit, mode);
            let mut out = cdef.clone();
            loop_restoration_frame(&cdef, &deblocked, &mut out, &lr);
            assert_eq!(visible_hash(&out), GOLDEN_HASHES[n], "case {n}");
        }
    }

    #[test]
    fn none_units_and_padding_are_copied() {
        let (cdef, deblocked, mut lr) = golden_case(11, 10, (1, 1), 3, 70, 90, 64, 2);
        for plane in lr.iter_mut() {
            plane.units[0].kind = LrType::None;
        }
        let mut out = cdef.clone();
        loop_restoration_frame(&cdef, &deblocked, &mut out, &lr);
        // The first unit is untouched, and so is everything beyond the visible area.
        for y in 0..32 {
            assert_eq!(out.planes[1].row(y)[..16], cdef.planes[1].row(y)[..16]);
        }
        for (p, plane) in out.planes.iter().enumerate() {
            let (sx, sy) = if p == 0 { (0, 0) } else { (1, 1) };
            let (vw, vh) = ((70 + sx) >> sx, (90 + sy) >> sy);
            for y in 0..plane.height {
                for x in 0..plane.width {
                    if x >= vw || y >= vh {
                        assert_eq!(plane.at(x, y), cdef.planes[p].at(x, y));
                    }
                }
            }
        }
        for plane in lr.iter_mut() {
            plane.units.iter_mut().for_each(|u| u.kind = LrType::None);
        }
        loop_restoration_frame(&cdef, &deblocked, &mut out, &lr);
        assert!(out.planes.iter().zip(&cdef.planes).all(|(a, b)| a.data == b.data));
    }

    #[test]
    fn wiener_keeps_flat_pictures_flat() {
        for bd in [8, 10, 12] {
            let (mut cdef, mut deblocked, lr) = golden_case(5, bd, (1, 0), 3, 90, 100, 64, 0);
            let v = (1u16 << bd) / 3;
            for f in [&mut cdef, &mut deblocked] {
                f.planes.iter_mut().for_each(|p| p.data.fill(v));
            }
            let mut out = cdef.clone();
            loop_restoration_frame(&cdef, &deblocked, &mut out, &lr);
            assert!(out.planes.iter().all(|p| p.data.iter().all(|&s| s == v)), "bit depth {bd}");
        }
    }

    #[test]
    fn stripe_context_comes_from_the_deblocked_picture() {
        // Luma stripe 1 starts at row 56. Its first rows see the deblocked rows 54 and 55; rows further away do not.
        let (cdef, mut deblocked, lr) = golden_case(21, 8, (1, 1), 1, 40, 130, 64, 0);
        let mut a = cdef.clone();
        loop_restoration_frame(&cdef, &deblocked, &mut a, &lr);
        // Deblocked rows that are not stripe context are never read.
        for y in [10usize, 40, 70, 129] {
            deblocked.planes[0].row_mut(y).fill(0);
        }
        let mut b = cdef.clone();
        loop_restoration_frame(&cdef, &deblocked, &mut b, &lr);
        assert_eq!(a.planes[0].data, b.planes[0].data);
        // The row just above stripe 1 is read by its first rows only (and by nothing in stripe 0).
        deblocked.planes[0].row_mut(55).fill(255);
        let mut c = cdef.clone();
        loop_restoration_frame(&cdef, &deblocked, &mut c, &lr);
        assert_ne!(a.planes[0].row(56), c.planes[0].row(56));
        assert_eq!(a.planes[0].row(60), c.planes[0].row(60));
        assert_eq!(a.planes[0].row(54), c.planes[0].row(54));
    }

    #[test]
    fn unit_counts() {
        assert_eq!(count_units(64, 1), 1);
        assert_eq!(count_units(64, 95), 1);
        assert_eq!(count_units(64, 96), 2);
        assert_eq!(count_units(256, 4000), 16);
    }
}
