//! AV1 intra prediction (specification section 7.11.2, plus chroma-from-luma, 7.11.5).
//!
//! The work is split the way the specification splits it:
//!
//! 1. [`build_edges`] reads the already reconstructed neighbours of a transform block into [`Edges`], applying the
//!    substitution rules for neighbours that do not exist.
//! 2. [`predict`] turns those edges into the predicted block (DC, directional with the intra edge filter and
//!    upsampling, smooth, Paeth or recursive filter-intra).
//! 3. [`cfl_predict`] adds the chroma-from-luma correction on top of a DC prediction.
//!
//! Edge layout. For a `w` x `h` block `above[0]` is the top-left corner sample, `above[1 + i]` is the spec's
//! `AboveRow[i]` and `left[1 + i]` its `LeftCol[i]`, for `i` in `0..w + h`; `left[0]` repeats the corner.
//!
//! Deviation from the requested API: [`Edges`] also carries `avail_w` / `avail_h` (how many columns / rows of the
//! block are inside the coded plane, `max_x + 1 - x` and `max_y + 1 - y`), which the edge filter needs and
//! [`build_edges`] fills in. [`predict`] only reads the edges (`&Edges`, so a `&mut Edges` argument also works).

// Intra mode numbers (spec values).
pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const D45_PRED: u8 = 3;
pub const D135_PRED: u8 = 4;
pub const D113_PRED: u8 = 5;
pub const D157_PRED: u8 = 6;
pub const D203_PRED: u8 = 7;
pub const D67_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const SMOOTH_V_PRED: u8 = 10;
pub const SMOOTH_H_PRED: u8 = 11;
pub const PAETH_PRED: u8 = 12;

// Filter-intra modes.
pub const FILTER_DC: u8 = 0;
pub const FILTER_V: u8 = 1;
pub const FILTER_H: u8 = 2;
pub const FILTER_D157: u8 = 3;
pub const FILTER_PAETH: u8 = 4;

/// Largest edge array needed: the corner plus `w + h` samples with `w = h = 64`.
pub const EDGE_LEN: usize = 1 + 64 + 64;

/// Neighbouring samples of one block (see the module documentation for the layout).
#[derive(Clone, Debug)]
pub struct Edges {
    pub above: [u16; EDGE_LEN],
    pub left: [u16; EDGE_LEN],
    /// Columns of the block that lie inside the coded plane (`max_x + 1 - x`); set by [`build_edges`].
    pub avail_w: usize,
    /// Rows of the block that lie inside the coded plane (`max_y + 1 - y`); set by [`build_edges`].
    pub avail_h: usize,
}

impl Default for Edges {
    fn default() -> Edges {
        Edges { above: [0; EDGE_LEN], left: [0; EDGE_LEN], avail_w: usize::MAX, avail_h: usize::MAX }
    }
}

/// Parameters of one intra prediction.
#[derive(Clone, Copy, Debug)]
pub struct IntraParams {
    pub mode: u8,
    /// `-3..=3`, only used by the directional modes.
    pub angle_delta: i8,
    /// Recursive filter-intra mode (luma only); overrides `mode`.
    pub filter_intra: Option<u8>,
    pub w: usize,
    pub h: usize,
    pub have_left: bool,
    pub have_above: bool,
    /// Sequence flag `enable_intra_edge_filter`.
    pub enable_edge_filter: bool,
    /// The spec's `get_filter_type()`: true when the above or left block uses a smooth mode.
    pub smooth_neighbor: bool,
    pub bit_depth: u8,
}

/// Weights for smooth prediction: the weights for a dimension of `n` samples start at index `n`.
#[rustfmt::skip]
const SM_WEIGHTS: [u8; 128] = [
    // unused
    0, 0,
    // 2
    255, 128,
    // 4
    255, 149, 85, 64,
    // 8
    255, 197, 146, 105, 73, 50, 37, 32,
    // 16
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
    // 32
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
    66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
    // 64
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150,
    144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
    65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20,
    18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// `Dr_Intra_Derivative`, indexed by `angle >> 1` (zero for angles that never occur).
#[rustfmt::skip]
const DR_INTRA_DERIVATIVE: [u16; 44] = [
    0, 1023, 0, 547, 372, 0, 0, 273, 215, 0, 178,
    151, 0, 132, 116, 0, 102, 0, 90, 80, 0, 71,
    64, 0, 57, 51, 0, 45, 0, 40, 35, 0, 31,
    27, 0, 23, 19, 0, 15, 0, 11, 0, 7, 3,
];

/// `Intra_Filter_Taps[mode][pixel of the 4x2 cell][tap]`: the taps apply to the corner, four above, two left samples.
#[rustfmt::skip]
const FILTER_INTRA_TAPS: [[[i8; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// `Mode_To_Angle` for the directional modes `V_PRED..=D67_PRED` (index = mode).
const MODE_TO_ANGLE: [i32; 9] = [0, 90, 180, 45, 135, 113, 157, 203, 67];

/// Degrees per `angle_delta` step.
const ANGLE_STEP: i32 = 3;

#[inline(always)]
fn dr_derivative(angle: i32) -> i32 {
    i32::from(DR_INTRA_DERIVATIVE[(angle >> 1) as usize])
}

/// Fills `edges` from the reconstructed `plane` for the block at (`x`, `y`) of size `w` x `h` (spec 7.11.2).
///
/// `have_left` / `have_above` say whether the neighbouring column / row exists, `have_above_right` /
/// `have_below_left` whether the extra samples beyond the block may be read. Like libaom and dav1d, at most
/// `w` samples are read to the right of the block (and `h` below it); anything further is the last sample
/// replicated. `max_x` / `max_y` are the last valid column / row of the plane and clamp every read.
#[allow(clippy::too_many_arguments)]
pub fn build_edges(
    plane: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    have_left: bool,
    have_above: bool,
    have_above_right: bool,
    have_below_left: bool,
    max_x: usize,
    max_y: usize,
    bit_depth: u8,
    edges: &mut Edges,
) {
    let mid = 1u16 << (bit_depth - 1);
    let n = w + h;
    edges.avail_w = (max_x + 1).saturating_sub(x);
    edges.avail_h = (max_y + 1).saturating_sub(y);

    // Above row.
    if have_above {
        let row = &plane[(y - 1) * stride..y * stride];
        let reach = if have_above_right { w + w.min(h) } else { w };
        for i in 0..reach {
            edges.above[1 + i] = row[max_x.min(x + i)];
        }
        let last = edges.above[reach];
        edges.above[1 + reach..=n].fill(last);
    } else if have_left {
        edges.above[1..=n].fill(plane[y * stride + x - 1]);
    } else {
        edges.above[1..=n].fill(mid - 1);
    }

    // Left column.
    if have_left {
        let reach = if have_below_left { h + h.min(w) } else { h };
        for i in 0..reach {
            edges.left[1 + i] = plane[max_y.min(y + i) * stride + x - 1];
        }
        let last = edges.left[reach];
        edges.left[1 + reach..=n].fill(last);
    } else if have_above {
        edges.left[1..=n].fill(plane[(y - 1) * stride + x]);
    } else {
        edges.left[1..=n].fill(mid + 1);
    }

    let corner = match (have_above, have_left) {
        (true, true) => plane[(y - 1) * stride + x - 1],
        (true, false) => plane[(y - 1) * stride + x],
        (false, true) => plane[y * stride + x - 1],
        (false, false) => mid,
    };
    edges.above[0] = corner;
    edges.left[0] = corner;
}

/// Predicts one `w` x `h` block into `dst` (row stride `dst_stride`) from `edges`.
///
/// The spec's `Round2`/`Clip1` arithmetic is reproduced exactly. Palette prediction is not handled here.
pub fn predict(dst: &mut [u16], dst_stride: usize, edges: &Edges, p: &IntraParams) {
    debug_assert!((4..=64).contains(&p.w) && (4..=64).contains(&p.h) && p.w.is_power_of_two() && p.h.is_power_of_two());
    if let Some(mode) = p.filter_intra {
        filter_intra(dst, dst_stride, edges, p, mode);
        return;
    }
    match p.mode {
        DC_PRED => dc(dst, dst_stride, edges, p),
        V_PRED..=D67_PRED => directional(dst, dst_stride, edges, p),
        SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED => smooth(dst, dst_stride, edges, p),
        PAETH_PRED => paeth(dst, dst_stride, edges, p),
        _ => panic!("not an intra prediction mode: {}", p.mode),
    }
}

/// Calls `value(row, column)` for every sample of the `w` x `h` block and stores the result.
fn fill_rows(dst: &mut [u16], stride: usize, w: usize, h: usize, mut value: impl FnMut(usize, usize) -> u16) {
    for (i, row) in dst.chunks_mut(stride).take(h).enumerate() {
        for (j, out) in row[..w].iter_mut().enumerate() {
            *out = value(i, j);
        }
    }
}

/// DC prediction (7.11.2.5), averaging whichever of the two edges exist.
fn dc(dst: &mut [u16], stride: usize, e: &Edges, p: &IntraParams) {
    let (w, h) = (p.w, p.h);
    let above: u32 = e.above[1..=w].iter().map(|&v| u32::from(v)).sum();
    let left: u32 = e.left[1..=h].iter().map(|&v| u32::from(v)).sum();
    let value = match (p.have_above, p.have_left) {
        (true, true) => (above + left + ((w + h) >> 1) as u32) / (w + h) as u32,
        (true, false) => (above + (w >> 1) as u32) / w as u32,
        (false, true) => (left + (h >> 1) as u32) / h as u32,
        (false, false) => 1 << (p.bit_depth - 1),
    } as u16;
    fill_rows(dst, stride, w, h, |_, _| value);
}

/// Paeth prediction (7.11.2.2).
fn paeth(dst: &mut [u16], stride: usize, e: &Edges, p: &IntraParams) {
    let corner = i32::from(e.above[0]);
    fill_rows(dst, stride, p.w, p.h, |i, j| {
        let (top, left) = (i32::from(e.above[1 + j]), i32::from(e.left[1 + i]));
        let base = top + left - corner;
        let (d_left, d_top, d_corner) = ((base - left).abs(), (base - top).abs(), (base - corner).abs());
        if d_left <= d_top && d_left <= d_corner {
            left as u16
        } else if d_top <= d_corner {
            top as u16
        } else {
            corner as u16
        }
    });
}

/// Smooth, smooth-vertical and smooth-horizontal prediction (7.11.2.6).
fn smooth(dst: &mut [u16], stride: usize, e: &Edges, p: &IntraParams) {
    let (w, h) = (p.w, p.h);
    let (weights_w, weights_h) = (&SM_WEIGHTS[w..2 * w], &SM_WEIGHTS[h..2 * h]);
    let bottom_left = u32::from(e.left[h]);
    let top_right = u32::from(e.above[w]);
    let mode = p.mode;
    fill_rows(dst, stride, w, h, |i, j| {
        let (wh, ww) = (u32::from(weights_h[i]), u32::from(weights_w[j]));
        let vertical = wh * u32::from(e.above[1 + j]) + (256 - wh) * bottom_left;
        let horizontal = ww * u32::from(e.left[1 + i]) + (256 - ww) * top_right;
        match mode {
            SMOOTH_PRED => ((vertical + horizontal + 256) >> 9) as u16,
            SMOOTH_V_PRED => ((vertical + 128) >> 8) as u16,
            _ => ((horizontal + 128) >> 8) as u16,
        }
    });
}

/// Recursive filter-intra (7.11.2.3): the block is predicted in 4x2 cells, each from seven neighbours.
fn filter_intra(dst: &mut [u16], stride: usize, e: &Edges, p: &IntraParams, mode: u8) {
    let max = (1i32 << p.bit_depth) - 1;
    let taps = &FILTER_INTRA_TAPS[usize::from(mode)];
    for cell_row in 0..p.h / 2 {
        for cell_col in 0..p.w / 4 {
            let (row, col) = (cell_row * 2, cell_col * 4);
            // Corner, four samples above, two to the left.
            let mut nb = [0i32; 7];
            for (k, v) in nb.iter_mut().enumerate() {
                *v = i32::from(if k < 5 {
                    if cell_row == 0 {
                        e.above[col + k]
                    } else if cell_col == 0 && k == 0 {
                        e.left[row]
                    } else {
                        dst[(row - 1) * stride + col + k - 1]
                    }
                } else if cell_col == 0 {
                    e.left[row + k - 4]
                } else {
                    dst[(row + k - 5) * stride + col - 1]
                });
            }
            for (cell, cell_taps) in taps.iter().enumerate() {
                let sum: i32 = cell_taps.iter().zip(&nb).map(|(&t, &v)| i32::from(t) * v).sum();
                // Round2Signed(sum, 4) followed by Clip1: every negative sum clips to zero.
                let value = if sum < 0 { 0 } else { ((sum + 8) >> 4).min(max) };
                dst[(row + cell / 4) * stride + col + cell % 4] = value as u16;
            }
        }
    }
}

/// `intra_edge_filter_strength_selection` (7.11.2.9).
fn edge_filter_strength(w: usize, h: usize, smooth: bool, delta: i32) -> usize {
    let d = delta.abs();
    if smooth {
        match w + h {
            0..=8 => usize::from(d >= 40) + usize::from(d >= 64),
            9..=16 => usize::from(d >= 20) + usize::from(d >= 48),
            17..=24 => 3 * usize::from(d >= 4),
            _ => 3 * usize::from(d >= 1),
        }
    } else {
        match w + h {
            0..=8 => usize::from(d >= 56),
            9..=16 => usize::from(d >= 40),
            17..=24 => usize::from(d >= 8) + usize::from(d >= 16) + usize::from(d >= 32),
            25..=32 => usize::from(d >= 1) + usize::from(d >= 4) + usize::from(d >= 32),
            _ => 3 * usize::from(d >= 1),
        }
    }
}

/// `intra_edge_upsample_selection` (7.11.2.10).
fn use_upsample(w: usize, h: usize, smooth: bool, delta: i32) -> bool {
    let d = delta.abs();
    if d == 0 || d >= 40 {
        false
    } else if smooth {
        w + h <= 8
    } else {
        w + h <= 16
    }
}

/// Working copy of one edge: `AboveRow[i]` / `LeftCol[i]` lives at `OFF + i`, so that the indices `-1` (corner)
/// and `-2` (created by upsampling) exist.
struct Edge {
    buf: [u16; Edge::LEN],
}

impl Edge {
    const OFF: usize = 2;
    const LEN: usize = Edge::OFF + 2 * 64 + 1;

    /// Copies the corner and the first `n` samples out of `src`.
    fn new(src: &[u16; EDGE_LEN], n: usize) -> Edge {
        let mut buf = [0; Edge::LEN];
        buf[Edge::OFF - 1..=Edge::OFF + n - 1].copy_from_slice(&src[..=n]);
        Edge { buf }
    }

    /// The sample at spec index `i` (`i >= -2`).
    #[inline(always)]
    fn at(&self, i: i32) -> i32 {
        i32::from(self.buf[(Edge::OFF as i32 + i) as usize])
    }

    /// Intra edge filter (7.11.2.12) over `size` samples starting at the corner, i.e. over `size - 1` edge samples.
    fn filter(&mut self, size: usize, strength: usize) {
        const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
        if strength == 0 {
            return;
        }
        let mut src = [0i32; 2 * 64 + 1];
        for (i, v) in src[..size].iter_mut().enumerate() {
            *v = self.at(i as i32 - 1);
        }
        for i in 1..size {
            let sum: i32 = (0..5).map(|j| KERNEL[strength - 1][j] * src[(i + j).saturating_sub(2).min(size - 1)]).sum();
            self.buf[Edge::OFF - 1 + i] = ((sum + 8) >> 4) as u16;
        }
    }

    /// Intra edge upsample (7.11.2.11): doubles the resolution of the first `count` samples.
    fn upsample(&mut self, count: usize, bit_depth: u8) {
        let max = (1i32 << bit_depth) - 1;
        let mut dup = [0i32; 20];
        dup[0] = self.at(-1);
        for i in -1..count as i32 {
            dup[(i + 2) as usize] = self.at(i);
        }
        dup[count + 2] = self.at(count as i32 - 1);
        self.buf[Edge::OFF - 2] = dup[0] as u16;
        for i in 0..count {
            let sum = 9 * (dup[i + 1] + dup[i + 2]) - dup[i] - dup[i + 3];
            self.buf[Edge::OFF + 2 * i - 1] = ((sum + 8) >> 4).clamp(0, max) as u16;
            self.buf[Edge::OFF + 2 * i] = dup[i + 2] as u16;
        }
    }
}

/// Directional prediction (7.11.2.4), which also covers `V_PRED` / `H_PRED` (angles 90 and 180).
fn directional(dst: &mut [u16], stride: usize, e: &Edges, p: &IntraParams) {
    let (w, h) = (p.w, p.h);
    let angle = MODE_TO_ANGLE[usize::from(p.mode)] + i32::from(p.angle_delta) * ANGLE_STEP;
    let mut above = Edge::new(&e.above, w + h);
    let mut left = Edge::new(&e.left, w + h);
    let (mut up_above, mut up_left) = (0u32, 0u32);

    if p.enable_edge_filter {
        let smooth = p.smooth_neighbor;
        if angle != 90 && angle != 180 {
            if angle > 90 && angle < 180 && w + h >= 24 {
                // Filter the corner sample (7.11.2.7).
                let corner = (u32::from(left.buf[Edge::OFF]) * 5
                    + u32::from(above.buf[Edge::OFF - 1]) * 6
                    + u32::from(above.buf[Edge::OFF]) * 5
                    + 8)
                    >> 4;
                above.buf[Edge::OFF - 1] = corner as u16;
                left.buf[Edge::OFF - 1] = corner as u16;
            }
            if p.have_above {
                let size = w.min(e.avail_w) + if angle < 90 { h } else { 0 } + 1;
                above.filter(size, edge_filter_strength(w, h, smooth, angle - 90));
            }
            if p.have_left {
                let size = h.min(e.avail_h) + if angle > 180 { w } else { 0 } + 1;
                left.filter(size, edge_filter_strength(w, h, smooth, angle - 180));
            }
        }
        if use_upsample(w, h, smooth, angle - 90) {
            up_above = 1;
            above.upsample(w + if angle < 90 { h } else { 0 }, p.bit_depth);
        }
        if use_upsample(w, h, smooth, angle - 180) {
            up_left = 1;
            left.upsample(h + if angle > 180 { w } else { 0 }, p.bit_depth);
        }
    }

    let interpolate = |edge: &Edge, base: i32, shift: i32| -> u16 {
        ((edge.at(base) * (32 - shift) + edge.at(base + 1) * shift + 16) >> 5) as u16
    };
    if angle == 90 {
        fill_rows(dst, stride, w, h, |_, j| above.buf[Edge::OFF + j]);
    } else if angle == 180 {
        fill_rows(dst, stride, w, h, |i, _| left.buf[Edge::OFF + i]);
    } else if angle < 90 {
        let dx = dr_derivative(angle);
        let max_base = ((w + h - 1) << up_above) as i32;
        fill_rows(dst, stride, w, h, |i, j| {
            let idx = (i as i32 + 1) * dx;
            let base = (idx >> (6 - up_above)) + ((j as i32) << up_above);
            let shift = ((idx << up_above) >> 1) & 31;
            if base < max_base { interpolate(&above, base, shift) } else { above.at(max_base) as u16 }
        });
    } else if angle < 180 {
        let (dx, dy) = (dr_derivative(180 - angle), dr_derivative(angle - 90));
        fill_rows(dst, stride, w, h, |i, j| {
            let (i, j) = (i as i32, j as i32);
            let idx = (j << 6) - (i + 1) * dx;
            let base = idx >> (6 - up_above);
            if base >= -(1 << up_above) {
                interpolate(&above, base, ((idx << up_above) >> 1) & 31)
            } else {
                let idx = (i << 6) - (j + 1) * dy;
                interpolate(&left, idx >> (6 - up_left), ((idx << up_left) >> 1) & 31)
            }
        });
    } else {
        let dy = dr_derivative(270 - angle);
        fill_rows(dst, stride, w, h, |i, j| {
            let idx = (j as i32 + 1) * dy;
            let base = (idx >> (6 - up_left)) + ((i as i32) << up_left);
            interpolate(&left, base, ((idx << up_left) >> 1) & 31)
        });
    }
}

/// Chroma-from-luma (7.11.5): adds `alpha` times the mean-removed, subsampled luma to the DC prediction in `dst`.
///
/// `luma` is the reconstructed luma plane; (`luma_x`, `luma_y`) is the luma sample co-located with the top-left
/// chroma sample, and `luma_avail_w` / `luma_avail_h` count the luma samples from there that lie inside the coded
/// area (reads beyond them replicate the last valid sample). `alpha` is the signed scale (-16..=16).
#[allow(clippy::too_many_arguments)]
pub fn cfl_predict(
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    luma: &[u16],
    luma_stride: usize,
    luma_x: usize,
    luma_y: usize,
    luma_avail_w: usize,
    luma_avail_h: usize,
    sub_x: u8,
    sub_y: u8,
    alpha: i32,
    bit_depth: u8,
) {
    let mut ac = [0i32; 32 * 32];
    let mut total = 0i32;
    for i in 0..h {
        let y = (i << sub_y).min(luma_avail_h - (1 << sub_y));
        for j in 0..w {
            let x = (j << sub_x).min(luma_avail_w - (1 << sub_x));
            let mut sum = 0i32;
            for dy in 0..=usize::from(sub_y) {
                let row = (luma_y + y + dy) * luma_stride + luma_x + x;
                for dx in 0..=usize::from(sub_x) {
                    sum += i32::from(luma[row + dx]);
                }
            }
            let value = sum << (3 - sub_x - sub_y);
            ac[i * w + j] = value;
            total += value;
        }
    }
    let log2_area = (w * h).trailing_zeros();
    let average = (total + (1 << (log2_area - 1))) >> log2_area;
    let max = (1i32 << bit_depth) - 1;
    for i in 0..h {
        for j in 0..w {
            let scaled = alpha * (ac[i * w + j] - average);
            let scaled = if scaled < 0 { -((-scaled + 32) >> 6) } else { (scaled + 32) >> 6 };
            let out = &mut dst[i * dst_stride + j];
            *out = (i32::from(*out) + scaled).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Xs(u64);

    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 11) as usize % n
        }
    }

    /// A noisy test picture: full-range noise, a gradient with a little noise, or a flat area with outliers.
    fn test_picture(r: &mut Xs, w: usize, h: usize, bd: u32) -> Vec<u16> {
        let max = (1usize << bd) - 1;
        let kind = r.below(3);
        let base = r.below(max + 1) as i32;
        let (ga, gb) = (r.below(8) as i32 - 3, r.below(8) as i32 - 3);
        let mut v = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let val = match kind {
                    0 => r.below(max + 1) as i32,
                    1 => base + (ga * x as i32 + gb * y as i32) * (1 << (bd - 8)) + r.below(5) as i32,
                    _ if r.below(8) == 0 => [0, max as i32][r.below(2)],
                    _ => base,
                };
                v[y * w + x] = val.clamp(0, max as i32) as u16;
            }
        }
        v
    }

    struct Case {
        bd: u32,
        w: usize,
        h: usize,
        filter: Option<u8>,
        mode: u8,
        delta: i8,
        left: bool,
        top: bool,
        tr: bool,
        bl: bool,
        smooth: bool,
        edge_filter: bool,
        x: usize,
        y: usize,
        avail_w: usize,
        avail_h: usize,
        frame: Vec<u16>,
    }

    fn golden_case(i: u64) -> Case {
        let mut r = Xs(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        let bd = [8, 10, 12][r.below(3)];
        let sizes = [4usize, 8, 16, 32, 64];
        let (w, h) = loop {
            let (w, h) = (sizes[r.below(5)], sizes[r.below(5)]);
            if w <= 4 * h && h <= 4 * w {
                break (w, h);
            }
        };
        let filter = (i % 9 == 8 && w <= 32 && h <= 32).then(|| r.below(5) as u8);
        let mode = (i % 13) as u8;
        let delta = if (V_PRED..=D67_PRED).contains(&mode) { r.below(7) as i8 - 3 } else { 0 };
        let left = r.below(5) != 0;
        let top = r.below(5) != 0;
        let x = if left { 4 * (1 + r.below(16)) } else { 4 * r.below(3) };
        let y = if top { 4 * (1 + r.below(16)) } else { 4 * r.below(3) };
        let avail_w = 4 * (1 + r.below(2 * w / 4 + 3));
        let avail_h = 4 * (1 + r.below(2 * h / 4 + 3));
        let frame = test_picture(&mut r, x + avail_w, y + avail_h, bd);
        let (tr, bl, smooth, edge_filter) = (r.below(2) == 1, r.below(2) == 1, r.below(2) == 1, r.below(4) != 0);
        Case { bd, w, h, filter, mode, delta, left, top, tr, bl, smooth, edge_filter, x, y, avail_w, avail_h, frame }
    }

    fn fnv(samples: &[u16]) -> u64 {
        samples.iter().fold(0xcbf2_9ce4_8422_2325, |h, &s| (h ^ u64::from(s)).wrapping_mul(0x0000_0100_0000_01b3))
    }

    fn run_golden_case(i: u64) -> u64 {
        let c = golden_case(i);
        let fw = c.x + c.avail_w;
        let fh = c.y + c.avail_h;
        let mut edges = Edges::default();
        build_edges(
            &c.frame,
            fw,
            c.x,
            c.y,
            c.w,
            c.h,
            c.left,
            c.top,
            c.tr,
            c.bl,
            fw - 1,
            fh - 1,
            c.bd as u8,
            &mut edges,
        );
        let params = IntraParams {
            mode: c.mode,
            angle_delta: c.delta,
            filter_intra: c.filter,
            w: c.w,
            h: c.h,
            have_left: c.left,
            have_above: c.top,
            enable_edge_filter: c.edge_filter,
            smooth_neighbor: c.smooth,
            bit_depth: c.bd as u8,
        };
        let mut out = vec![0u16; c.w * c.h];
        predict(&mut out, c.w, &edges, &params);
        fnv(&out)
    }

    fn run_golden_cfl(i: u64) -> u64 {
        let mut r = Xs((i + 1000).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        let bd = [8u32, 10, 12][r.below(3)];
        let sizes = [4usize, 8, 16, 32];
        let (w, h) = loop {
            let (w, h) = (sizes[r.below(4)], sizes[r.below(4)]);
            if w <= 4 * h && h <= 4 * w {
                break (w, h);
            }
        };
        let (sx, sy) = [(0u8, 0u8), (1, 0), (1, 1)][r.below(3)];
        let (valid_w, valid_h) = (4 * (1 + r.below(w / 4)), 4 * (1 + r.below(h / 4)));
        let (lx, ly) = (r.below(9), r.below(9));
        let (lw, lh) = (lx + (w << sx) + 2, ly + (h << sy) + 2);
        let luma = test_picture(&mut r, lw, lh, bd);
        let alpha = r.below(33) as i32 - 16;
        let dc = r.below(1 << bd) as u16;
        let mut out = vec![dc; w * h];
        cfl_predict(&mut out, w, w, h, &luma, lw, lx, ly, valid_w << sx, valid_h << sy, sx, sy, alpha, bd as u8);
        fnv(&out)
    }

    /// Hashes of `run_golden_case(i)` for `i = 0..`, produced by a transcription of dav1d's (rav1d 1.1.0) edge
    /// preparation and predictors from the same inputs (2 000 000 random cases were also compared offline).
    #[rustfmt::skip]
    const GOLDEN_INTRA: [u64; 130] = [
        0xe537a7e83dd52725, 0xb56c555814e9dca5, 0x51d88627df287325, 0xdca4dfd387f55625,
        0x96f14bc9753f4f63, 0x3184a0076d639725, 0x8eb06fb30687ee9a, 0xbb5b8f081086f725,
        0x7d0929193d839725, 0x00672598d98530f7, 0xb4bee02c089e901b, 0x8913ba271e6256d0,
        0x73a3038c4b8545e5, 0xfbbcf65e920c5b25, 0x220c95b5a14de779, 0xe0ef9e5d79771918,
        0xda277ee00bb7759b, 0x263fbd8e93ff8fe5, 0x2e2278af1fc68e14, 0x001cb2c2b1f4a865,
        0x8a2007bef582e391, 0x88201fb960ff6465, 0xe959c3661169b9bb, 0x5f9016ea631208e9,
        0xbab16346de95adbb, 0x3558038d6fcffd60, 0x9c9b3c322eee18e8, 0x84cc4da0e20ecde5,
        0x7ce3da40e6a239a6, 0x34574dd8cfb0056e, 0x290ccd2992bec855, 0xacf20b6a1d52a221,
        0x8503a1f1eee2346a, 0x9248cafe1652db85, 0x51d88627df287325, 0x7cf4df459a0989df,
        0x51d88627df287325, 0xab7fcc4c01f27df6, 0x33d30209c4441e25, 0xebef254c52804325,
        0x8c3588f529994721, 0x2a3d15590add3325, 0xfc5c044b5ca3b185, 0x412d124bc206d525,
        0x68c8efd5c07cd905, 0xe7cb1b5f1c3d5973, 0xaac9a700e8467b41, 0x4c9589c93a4d6349,
        0xf0de576da0fd76c0, 0x4492bb636a4930cc, 0x50174529cd639687, 0x7c05970a5933a7b2,
        0xd44997e06eb25485, 0xb0467c5abc76b33a, 0x1bc3d895cb06b17b, 0x79659f66de48b502,
        0x08819c2a328f5d16, 0x51d88627df287325, 0x0480bee1eb9d7eeb, 0xfc5afa424e4f7325,
        0x71dfb853b62e4c62, 0x9eed3d4141d93625, 0xea6f21d2ad978d51, 0xaf6eadfeac0eb886,
        0x93ccbb5f35483ea6, 0x5c33b4a3bb63d825, 0xb42f728185ffcf25, 0x7d4b653aa310cda3,
        0x1dd6c7008d5dd725, 0x9eed3d4141d93625, 0xea0569380cce69b8, 0xa5c0ccfb7f9661ec,
        0xb30ec36450d030a5, 0x1708ac8a087a7f54, 0x07c29b5f187e11bc, 0xf13314a1cca455c9,
        0xfe018d270bde48c5, 0x0440d81567bc8c07, 0x7f5e66a928d905c5, 0xc4e538fca8648f05,
        0xfa9b806ee0f78bc1, 0x03a67794da288455, 0x9878153a661f1e46, 0x030b69b790b63aa5,
        0x17078bab1f4b589c, 0x9ac5e12119fd3f85, 0x3c522ed41d5eeef4, 0x1d75180c343dba05,
        0x89a719235c62450b, 0x18429f143dea272d, 0x22682b8e068bb47b, 0x02fe0c1c5992a325,
        0x2c60ea351043d725, 0x8004e784b454a93f, 0x2545458bc3bf4e31, 0x86f3ab3108477efe,
        0xaddbb4c6e2cfabf8, 0x25cf8cb7a87296d2, 0xb5fca21b272c64af, 0x0d6b8c02f6429c20,
        0x2205a55efa54ffaf, 0x9eed3d4141d93625, 0x9ac5e12119fd3f85, 0xb33ea37f2dab5434,
        0x7da788d16c913325, 0x99d7a53370ee71d2, 0x8cbd46b952f8fbf0, 0xf44a57a71c5ae573,
        0xd0b2eb1528f000f0, 0x00b6dbc5fd901d45, 0x962c39425c06ce76, 0x140f7ec78bfb701a,
        0xf06ec1cf602a6994, 0xe2b18a3ff2cdf4b9, 0xcb238cf366b35501, 0xd53c630e784df2e5,
        0x36891eb243adbe9c, 0x6253159083ac6325, 0xbf213b211259d1a5, 0x4567325fa1898ab2,
        0x7e404b70d8228aae, 0xac5e72ba070760e0, 0x3184a0076d639725, 0x3f0294e78e97cc3e,
        0x07928e91f97ac463, 0xe8e94e38b88d0a7b, 0xb4018697f40eba33, 0x3308d1108e074425,
        0x3b333ca94d6d7ffb, 0xf3cb2b95e4c54215,
    ];

    /// Same for `run_golden_cfl`.
    #[rustfmt::skip]
    const GOLDEN_CFL: [u64; 26] = [
        0x8ac12e0e0f7e59e5, 0x6cffb7ba22ba013b, 0xf8080ef63a96eabf, 0xebe9a55e64d1b261,
        0xd6f7c21cdd0a0e4e, 0x2567dc98a8bf00b6, 0x58cee222d428143e, 0x5e29f8c3bbcd9b78,
        0x2c3a8bcebe30267b, 0xfe389f86421bff65, 0xf2b1656f99294ceb, 0x2a4db359a1882907,
        0xa2b6760c473fa2bd, 0xf6027a48312d6d84, 0xae17609007dbd1c4, 0xe613d7dbb8e536b3,
        0x6cf82e7080a490f4, 0xd5f252b8e172f178, 0x810479e997b95b1e, 0x6ef36de994b7a80e,
        0x6101bf321c27628c, 0x42993e12744b40cd, 0x33236e3f622b0a74, 0x923a98e6fb91f9dd,
        0x5e56b2c54121916c, 0xf8e169617d76c834,
    ];

    #[test]
    fn matches_dav1d_golden_vectors() {
        for (i, &expected) in GOLDEN_INTRA.iter().enumerate() {
            assert_eq!(run_golden_case(i as u64), expected, "intra case {i}");
        }
    }

    #[test]
    fn cfl_matches_dav1d_golden_vectors() {
        for (i, &expected) in GOLDEN_CFL.iter().enumerate() {
            assert_eq!(run_golden_cfl(i as u64), expected, "cfl case {i}");
        }
    }

    fn params(mode: u8, w: usize, h: usize) -> IntraParams {
        IntraParams {
            mode,
            angle_delta: 0,
            filter_intra: None,
            w,
            h,
            have_left: true,
            have_above: true,
            enable_edge_filter: true,
            smooth_neighbor: false,
            bit_depth: 8,
        }
    }

    /// Edges where `AboveRow[i] = 10 + i` and `LeftCol[i] = 100 + 2 * i`, corner 50.
    fn ramp_edges() -> Edges {
        let mut e = Edges::default();
        e.above[0] = 50;
        e.left[0] = 50;
        for i in 0..64 + 64 {
            e.above[1 + i] = 10 + i as u16;
            e.left[1 + i] = 100 + 2 * i as u16;
        }
        e
    }

    #[test]
    fn dc_uses_the_available_edges() {
        let e = ramp_edges();
        let mut out = [0u16; 4 * 8];
        // 4x8: above 10..=13 (sum 46), left 100, 102, .. 114 (sum 856); (46 + 856 + 6) / 12 = 75.
        predict(&mut out, 4, &e, &params(DC_PRED, 4, 8));
        assert!(out.iter().all(|&v| v == 75));
        let mut p = params(DC_PRED, 4, 8);
        p.have_left = false;
        predict(&mut out, 4, &e, &p); // (46 + 2) / 4
        assert!(out.iter().all(|&v| v == 12));
        p.have_left = true;
        p.have_above = false;
        predict(&mut out, 4, &e, &p); // (856 + 4) / 8
        assert!(out.iter().all(|&v| v == 107));
        p.bit_depth = 10;
        p.have_left = false;
        predict(&mut out, 4, &e, &p);
        assert!(out.iter().all(|&v| v == 512));
    }

    #[test]
    fn v_h_and_paeth() {
        let e = ramp_edges();
        let mut out = [0u16; 16];
        predict(&mut out, 4, &e, &params(V_PRED, 4, 4));
        assert_eq!(out[..4], [10, 11, 12, 13]);
        assert_eq!(out[12..], [10, 11, 12, 13]);
        predict(&mut out, 4, &e, &params(H_PRED, 4, 4));
        assert_eq!(out.iter().step_by(4).copied().collect::<Vec<_>>(), [100, 102, 104, 106]);
        assert!(out[4..8].iter().all(|&v| v == 102));
        // Paeth: base = above + left - corner (60 for the first sample) is closest to the corner sample, 50.
        predict(&mut out, 4, &e, &params(PAETH_PRED, 4, 4));
        assert_eq!(out[0], 50);
        assert_eq!(out[3], 50);
    }

    #[test]
    fn smooth_of_flat_edges_is_flat() {
        let mut e = Edges::default();
        e.above.fill(77);
        e.left.fill(77);
        for mode in [SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
            let mut out = [0u16; 16 * 4];
            predict(&mut out, 16, &e, &params(mode, 16, 4));
            assert!(out.iter().all(|&v| v == 77), "mode {mode}");
        }
    }

    #[test]
    fn missing_edges_use_the_spec_base_values() {
        let plane = vec![200u16; 16 * 16];
        let mut e = Edges::default();
        build_edges(&plane, 16, 0, 0, 4, 4, false, false, false, false, 15, 15, 8, &mut e);
        assert_eq!((e.above[0], e.above[1], e.above[8], e.left[1], e.left[8]), (128, 127, 127, 129, 129));
        build_edges(&plane, 16, 0, 0, 4, 4, false, false, false, false, 15, 15, 10, &mut e);
        assert_eq!((e.above[1], e.left[1], e.left[0]), (511, 513, 512));
        // Only the left column exists: the top row repeats the left neighbour of the first row.
        let mut plane = vec![0u16; 16 * 16];
        for y in 0..16 {
            plane[y * 16 + 3] = 20 + y as u16;
        }
        build_edges(&plane, 16, 4, 4, 4, 4, true, false, false, false, 15, 15, 8, &mut e);
        assert!(e.above[0..=8].iter().all(|&v| v == 24));
        assert_eq!(e.left[1..=4], [24, 25, 26, 27]);
        // Below-left is read when allowed, otherwise the last left sample repeats; reads are clamped to `max_y`.
        build_edges(&plane, 16, 4, 4, 4, 4, true, false, false, true, 15, 15, 8, &mut e);
        assert_eq!(e.left[1..=8], [24, 25, 26, 27, 28, 29, 30, 31]);
        build_edges(&plane, 16, 4, 4, 4, 4, true, false, false, true, 15, 5, 8, &mut e);
        assert_eq!(e.left[1..=8], [24, 25, 25, 25, 25, 25, 25, 25]);
        assert_eq!((e.avail_w, e.avail_h), (12, 2));
    }

    #[test]
    fn above_right_reaches_at_most_one_block_width() {
        let mut plane = vec![0u16; 64 * 8];
        for (x, v) in plane[3 * 64..4 * 64].iter_mut().enumerate() {
            *v = x as u16;
        }
        let mut e = Edges::default();
        // A 4x16 block at x = 8: only the four samples 12..=15 to the right exist, the rest repeats 15.
        build_edges(&plane, 64, 8, 4, 4, 16, true, true, true, false, 63, 7, 8, &mut e);
        assert_eq!(e.above[0], 7);
        assert_eq!(e.above[1..=8], [8, 9, 10, 11, 12, 13, 14, 15]);
        assert!(e.above[9..=20].iter().all(|&v| v == 15));
        // Clamped by the plane width.
        build_edges(&plane, 64, 8, 4, 8, 8, true, true, true, false, 12, 7, 8, &mut e);
        assert_eq!(e.above[1..=16], [8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]);
    }

    #[test]
    fn edge_filter_strength_table() {
        // (w + h, smooth neighbour, |delta|) -> strength, spot checks of every row of the spec table.
        let cases = [
            (8, false, 55, 0),
            (8, false, 56, 1),
            (16, false, 39, 0),
            (16, false, 40, 1),
            (24, false, 7, 0),
            (24, false, 8, 1),
            (24, false, 16, 2),
            (24, false, 32, 3),
            (32, false, 3, 1),
            (32, false, 4, 2),
            (32, false, 32, 3),
            (40, false, 1, 3),
            (8, true, 39, 0),
            (8, true, 40, 1),
            (8, true, 64, 2),
            (16, true, 19, 0),
            (16, true, 20, 1),
            (16, true, 48, 2),
            (24, true, 3, 0),
            (24, true, 4, 3),
            (40, true, 1, 3),
        ];
        for (wh, smooth, delta, expected) in cases {
            assert_eq!(edge_filter_strength(wh / 2, wh / 2, smooth, delta), expected, "{wh} {smooth} {delta}");
        }
        assert!(use_upsample(8, 8, false, 39) && !use_upsample(8, 8, false, 40) && !use_upsample(8, 8, false, 0));
        assert!(use_upsample(4, 4, true, 3) && !use_upsample(8, 8, true, 3) && !use_upsample(8, 16, false, 3));
    }

    #[test]
    fn cfl_of_flat_luma_keeps_the_dc() {
        let luma = vec![321u16; 20 * 20];
        let mut out = vec![99u16; 8 * 8];
        cfl_predict(&mut out, 8, 8, 8, &luma, 20, 2, 2, 16, 16, 1, 1, 9, 10);
        assert!(out.iter().all(|&v| v == 99));
    }

    #[test]
    fn cfl_scales_and_clips() {
        // Left half dark, right half bright luma (4:4:4, 4x4): ac = +-(8 * 50) after removing the mean.
        let mut luma = vec![100u16; 4 * 4];
        for row in luma.chunks_mut(4) {
            row[2..].fill(200);
        }
        let mut out = vec![128u16; 16];
        cfl_predict(&mut out, 4, 4, 4, &luma, 4, 0, 0, 4, 4, 0, 0, 2, 8);
        // ac = 8 * (100 - 150) = -400 and +400; Round2Signed(2 * 400, 6) = 13.
        assert_eq!(out[..4], [115, 115, 141, 141]);
        let mut out = vec![128u16; 16];
        cfl_predict(&mut out, 4, 4, 4, &luma, 4, 0, 0, 4, 4, 0, 0, -16, 8);
        assert_eq!(out[..4], [128 + 100, 128 + 100, 28, 28]);
        let mut out = vec![250u16; 16];
        cfl_predict(&mut out, 4, 4, 4, &luma, 4, 0, 0, 4, 4, 0, 0, 16, 8);
        assert_eq!(out[..4], [150, 150, 255, 255]);
    }

    #[test]
    fn every_mode_stays_in_range_for_every_shape() {
        let mut state = Xs(0x1234_5678_9abc_def1);
        for bd in [8u8, 10, 12] {
            let max = (1u16 << bd) - 1;
            let mut e = Edges::default();
            for v in e.above.iter_mut().chain(e.left.iter_mut()) {
                *v = if state.below(3) == 0 {
                    [0, max][state.below(2)]
                } else {
                    state.below(usize::from(max) + 1) as u16
                };
            }
            e.left[0] = e.above[0];
            for (w, h) in [(4, 4), (4, 16), (16, 4), (64, 64), (64, 16), (16, 64), (8, 32), (32, 8), (4, 8), (8, 4)] {
                for mode in 0..=PAETH_PRED {
                    for delta in -3..=3 {
                        for (have_left, have_above, edge_filter, smooth_neighbor) in
                            [(true, true, true, true), (true, true, true, false), (false, true, false, false)]
                        {
                            let p = IntraParams {
                                mode,
                                angle_delta: delta,
                                filter_intra: None,
                                w,
                                h,
                                have_left,
                                have_above,
                                enable_edge_filter: edge_filter,
                                smooth_neighbor,
                                bit_depth: bd,
                            };
                            let mut out = vec![0u16; w * h];
                            predict(&mut out, w, &e, &p);
                            assert!(out.iter().all(|&v| v <= max));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn filter_intra_modes_are_distinct_and_in_range() {
        let e = ramp_edges();
        let mut seen = Vec::new();
        for mode in FILTER_DC..=FILTER_PAETH {
            let mut p = params(DC_PRED, 8, 8);
            p.filter_intra = Some(mode);
            let mut out = [0u16; 64];
            predict(&mut out, 8, &e, &p);
            assert!(out.iter().all(|&v| v <= 255));
            assert!(!seen.contains(&out));
            seen.push(out);
        }
    }
}
