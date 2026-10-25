//! The AV1 deblocking loop filter (specification section 7.14).
//!
//! The filter runs on the reconstructed frame, plane by plane: first every vertical edge of the plane (left to right
//! across each row of 4x4 units), then every horizontal edge. An edge between two 4x4 units is filtered when it is a
//! transform edge (and, for skipped inter blocks, a block edge); the filter *length* depends on the transform sizes on
//! both sides (4, 6 for chroma, 8, or 14 for luma), and the filter *strength* on the loop-filter level of the block
//! (frame level, per-block `delta_lf`, per-segment feature, reference/mode deltas) and on the frame's sharpness.
//!
//! The decoder describes the frame with [`DeblockParams`] (frame header values) and [`DeblockMi`] (per-4x4 mode info
//! saved while decoding blocks). Everything here is for intra frames as well as inter frames; for an intra frame every
//! `ref_frame` is `INTRA_FRAME` (0).

use super::frame::{Frame, Plane};

/// `MAX_LOOP_FILTER`: the largest loop-filter level.
const MAX_LEVEL: i32 = 63;

/// First inter prediction mode (`NEARESTMV`) in the spec's `YMode` numbering; modes below it are intra modes.
pub const NEARESTMV: u8 = 13;
/// `GLOBALMV` in the spec's `YMode` numbering.
pub const GLOBALMV: u8 = 15;
/// `GLOBAL_GLOBALMV` in the spec's `YMode` numbering.
pub const GLOBAL_GLOBALMV: u8 = 23;

/// Block width and height in units of 4 samples, indexed by the spec's `BLOCK_*` number
/// (`BLOCK_4X4` = 0 .. `BLOCK_64X16` = 21).
const BLOCK_DIMS_4: [(u8, u8); 22] = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (1, 4),
    (4, 1),
    (2, 8),
    (8, 2),
    (4, 16),
    (16, 4),
];

/// Transform width and height in units of 4 samples, indexed by the spec's `TX_*` number
/// (`TX_4X4` = 0 .. `TX_64X16` = 18).
const TX_DIMS_4: [(u8, u8); 19] = [
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (1, 2),
    (2, 1),
    (2, 4),
    (4, 2),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (1, 4),
    (4, 1),
    (2, 8),
    (8, 2),
    (4, 16),
    (16, 4),
];

/// Frame-level loop-filter syntax (`loop_filter_params()`, `delta_lf_params()`, segmentation) needed by the filter.
#[derive(Clone, Debug, Default)]
pub struct DeblockParams {
    /// `loop_filter_level[0..4]`: Y vertical edges, Y horizontal edges, U, V. When both luma levels are 0 nothing is
    /// filtered at all (the chroma levels are not even coded then); a chroma plane with level 0 is not filtered.
    pub filter_level: [u8; 4],
    /// `loop_filter_sharpness` (0..=7).
    pub sharpness: u8,
    /// `loop_filter_delta_enabled`.
    pub delta_enabled: bool,
    /// `loop_filter_ref_deltas` (index 0 = `INTRA_FRAME`, the only one an intra frame uses).
    pub ref_deltas: [i8; 8],
    /// `loop_filter_mode_deltas` (index 0: zero-motion modes, 1: the other inter modes).
    pub mode_deltas: [i8; 2],
    /// `delta_lf_multi`: when set the per-block `delta_lf` has four independent values (one per
    /// [`DeblockParams::filter_level`] entry), otherwise value 0 applies to all of them. (`delta_lf_present` need not
    /// be passed: without it every block's `delta_lf` is zero.)
    pub delta_lf_multi: bool,
    /// Per-segment loop-filter adjustment, `FeatureData[segment][SEG_LVL_ALT_LF_Y_V + i]` for `i` in
    /// `[y_vertical, y_horizontal, u, v]`, or 0 where the feature is disabled (or segmentation is off).
    pub seg_lf: [[i8; 4]; 8],
}

/// Per-4x4 ("mode info") data of the decoded frame, row-major with `mi_cols` columns. Chroma planes are addressed
/// with the luma 4x4 position `(row, col)` as the spec does (the bottom-right unit of each subsampled group).
#[derive(Clone, Debug, Default)]
pub struct DeblockMi {
    pub mi_cols: usize,
    pub mi_rows: usize,
    /// `MiSizes`: the block size (`BLOCK_*`, spec numbering 0..=21) covering each unit.
    pub mi_size: Vec<u8>,
    /// `Skips`: the block's `skip` flag.
    pub skip: Vec<bool>,
    /// `RefFrames[..][0]`: 0 (`INTRA_FRAME`) for intra blocks (and every block of an intra frame), 1.. for inter.
    pub ref_frame: Vec<i8>,
    /// `YModes`: the luma prediction mode (spec numbering; only compared against the inter modes).
    pub y_mode: Vec<u8>,
    /// `SegmentIds`.
    pub segment_id: Vec<u8>,
    /// `DeltaLFs`: the `delta_lf` state in force for the block; all zero without `delta_lf_present`.
    pub delta_lf: Vec<[i8; 4]>,
    /// `LoopfilterTxSizes[plane]` (`TX_*`, spec numbering 0..=18). Plane 0 is indexed `[row * mi_cols + col]`; planes
    /// 1 and 2 are indexed `[(row >> sub_y) * ((mi_cols + sub_x) >> sub_x) + (col >> sub_x)]`. Unused planes may be
    /// empty.
    pub tx_size: [Vec<u8>; 3],
}

impl DeblockMi {
    /// A frame of 4x4 intra blocks with transform size `TX_4X4`, ready to be filled in.
    pub fn new(mi_cols: usize, mi_rows: usize, sub_x: usize, sub_y: usize, num_planes: usize) -> DeblockMi {
        let n = mi_cols * mi_rows;
        let chroma = ((mi_cols + sub_x) >> sub_x) * ((mi_rows + sub_y) >> sub_y);
        let tx = |plane: usize| match plane {
            _ if plane >= num_planes => Vec::new(),
            0 => vec![0; n],
            _ => vec![0; chroma],
        };
        DeblockMi {
            mi_cols,
            mi_rows,
            mi_size: vec![0; n],
            skip: vec![false; n],
            ref_frame: vec![0; n],
            y_mode: vec![0; n],
            segment_id: vec![0; n],
            delta_lf: vec![[0; 4]; n],
            tx_size: [tx(0), tx(1), tx(2)],
        }
    }

    /// Row stride of `tx_size[plane]`.
    pub fn tx_stride(&self, sub_x: usize) -> usize {
        (self.mi_cols + sub_x) >> sub_x
    }
}

/// Filter limits of one edge, already scaled to the bit depth.
#[derive(Clone, Copy)]
struct Limits {
    limit: i32,
    blimit: i32,
    thresh: i32,
    /// `1 << (BitDepth - 8)`: the flatness threshold.
    flat: i32,
    bit_depth: u32,
}

/// Number of taps of the filter: `Narrow` uses 2 samples either side, the others 3 (`Six`, chroma), 4 and 7 (`Wide`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Length {
    Four,
    Six,
    Eight,
    Fourteen,
}

/// Applies the deblocking filter to a whole frame, in place. `mi` must describe the same frame (`mi_cols * 4` x
/// `mi_rows * 4` luma samples, the planes' coded area); `frame.width` / `frame.height` (the visible size) bound which
/// edges are filtered: an edge starting at or beyond the visible area is left alone, exactly as in the spec.
pub fn deblock_frame(frame: &mut Frame, p: &DeblockParams, mi: &DeblockMi) {
    if p.filter_level[0] == 0 && p.filter_level[1] == 0 {
        return;
    }
    let (frame_w, frame_h) = (frame.width, frame.height);
    let (frame_sx, frame_sy) = (frame.sub_x as usize, frame.sub_y as usize);
    let bit_depth = frame.bit_depth as u32;
    let sharp = SharpnessLut::new(p.sharpness);
    for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
        if plane_idx > 0 && p.filter_level[plane_idx + 1] == 0 {
            continue;
        }
        let (sx, sy) = if plane_idx == 0 { (0, 0) } else { (frame_sx, frame_sy) };
        for pass in 0..2 {
            let ctx = PassCtx { p, mi, sharp: &sharp, plane_idx, pass, sx, sy, bit_depth, frame_w, frame_h };
            filter_plane_pass(plane, &ctx);
        }
    }
}

/// `limit` per level for the frame's sharpness (spec 7.14.4).
struct SharpnessLut {
    limit: [u8; 64],
}

impl SharpnessLut {
    fn new(sharpness: u8) -> SharpnessLut {
        let sharp = sharpness as i32;
        let shift = if sharp > 4 {
            2
        } else if sharp > 0 {
            1
        } else {
            0
        };
        let mut limit = [0u8; 64];
        for (lvl, l) in limit.iter_mut().enumerate() {
            let v = lvl as i32 >> shift;
            *l = if sharp > 0 { v.clamp(1, 9 - sharp) } else { v.max(1) } as u8;
        }
        SharpnessLut { limit }
    }
}

struct PassCtx<'a> {
    p: &'a DeblockParams,
    mi: &'a DeblockMi,
    sharp: &'a SharpnessLut,
    plane_idx: usize,
    /// 0: vertical edges (filter across columns), 1: horizontal edges.
    pass: usize,
    sx: usize,
    sy: usize,
    bit_depth: u32,
    frame_w: usize,
    frame_h: usize,
}

impl PassCtx<'_> {
    /// The loop-filter level of the block at mode-info index `idx` for this plane and pass (spec 7.14.4).
    fn level(&self, idx: usize) -> i32 {
        let (p, mi) = (self.p, self.mi);
        let i = if self.plane_idx == 0 { self.pass } else { self.plane_idx + 1 };
        let delta = mi.delta_lf[idx][if p.delta_lf_multi { i } else { 0 }] as i32;
        let base = (delta + p.filter_level[i] as i32).clamp(0, MAX_LEVEL);
        let seg = mi.segment_id[idx] as usize & 7;
        let mut lvl = (base + p.seg_lf[seg][i] as i32).clamp(0, MAX_LEVEL);
        if p.delta_enabled {
            let shift = lvl >> 5;
            let r = mi.ref_frame[idx];
            if r <= 0 {
                lvl += (p.ref_deltas[0] as i32) << shift;
            } else {
                let mode = mi.y_mode[idx];
                let mode_type = (mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV) as usize;
                lvl += ((p.ref_deltas[r as usize & 7] as i32) << shift) + ((p.mode_deltas[mode_type] as i32) << shift);
            }
            lvl = lvl.clamp(0, MAX_LEVEL);
        }
        lvl
    }

    fn tx_size(&self, row: usize, col: usize) -> usize {
        let stride = self.mi.tx_stride(self.sx);
        self.mi.tx_size[self.plane_idx][(row >> self.sy) * stride + (col >> self.sx)] as usize
    }
}

/// One pass (all vertical or all horizontal edges) over one plane: the spec's edge loop filter process (7.14.2).
fn filter_plane_pass(plane: &mut Plane, c: &PassCtx) {
    let mi = c.mi;
    let (dx, dy) = if c.pass == 0 { (1usize, 0usize) } else { (0, 1) };
    let stride = plane.stride;
    // Moving across the edge / along the edge, in samples of the plane buffer.
    let (across, along) = if c.pass == 0 { (1, stride) } else { (stride, 1) };
    let mut row = 0;
    while row < mi.mi_rows {
        let y = row * 4;
        let mut col = 0;
        while col < mi.mi_cols {
            let x = col * 4;
            let step = 1usize << c.sx;
            let on_screen = x < c.frame_w && y < c.frame_h && !(c.pass == 0 && x == 0) && !(c.pass == 1 && y == 0);
            if on_screen {
                // Chroma units are described by the bottom-right luma unit of their group.
                let r = (row | c.sy).min(mi.mi_rows - 1);
                let cl = (col | c.sx).min(mi.mi_cols - 1);
                let prev_r = r - (dy << c.sy);
                let prev_c = cl - (dx << c.sx);
                let idx = r * mi.mi_cols + cl;
                let prev_idx = prev_r * mi.mi_cols + prev_c;
                let x_p = x >> c.sx;
                let y_p = y >> c.sy;
                let tx = c.tx_size(r, cl);
                let prev_tx = c.tx_size(prev_r, prev_c);
                let (tx_w, tx_h) = TX_DIMS_4[tx];
                let (blk_w, blk_h) = BLOCK_DIMS_4[mi.mi_size[idx] as usize];
                // Sizes in samples of this plane.
                let (tx_len, blk_len, pos) = if c.pass == 0 {
                    (tx_w as usize * 4, ((blk_w as usize * 4) >> c.sx).max(4), x_p)
                } else {
                    (tx_h as usize * 4, ((blk_h as usize * 4) >> c.sy).max(4), y_p)
                };
                let is_tx_edge = pos % tx_len == 0;
                let is_block_edge = pos % blk_len == 0;
                let is_intra = mi.ref_frame[idx] <= 0;
                if is_tx_edge && (is_block_edge || !mi.skip[idx] || is_intra) {
                    let (pw, ph) = TX_DIMS_4[prev_tx];
                    let (cur_len, prev_len) = if c.pass == 0 { (tx_w, pw) } else { (tx_h, ph) };
                    let base = cur_len.min(prev_len) as usize * 4;
                    let length = if c.plane_idx == 0 {
                        match base {
                            4 => Length::Four,
                            8 => Length::Eight,
                            _ => Length::Fourteen,
                        }
                    } else if base >= 8 {
                        Length::Six
                    } else {
                        Length::Four
                    };
                    let mut lvl = c.level(idx);
                    if lvl == 0 {
                        lvl = c.level(prev_idx);
                    }
                    if lvl > 0 {
                        let shift = c.bit_depth - 8;
                        let limit = c.sharp.limit[lvl as usize] as i32;
                        let lim = Limits {
                            limit: limit << shift,
                            blimit: (2 * (lvl + 2) + limit) << shift,
                            thresh: (lvl >> 4) << shift,
                            flat: 1 << shift,
                            bit_depth: c.bit_depth,
                        };
                        let start = y_p * stride + x_p;
                        for i in 0..4 {
                            filter_sample(&mut plane.data, start + i * along, across, length, &lim);
                        }
                    }
                }
            }
            col += step;
        }
        row += 1 << c.sy;
    }
}

/// Filters the samples across one edge position. `pos` is the index of `q0`; `across` the distance between
/// consecutive samples perpendicular to the edge (spec 7.14.6).
fn filter_sample(data: &mut [u16], pos: usize, across: usize, length: Length, lim: &Limits) {
    let at = |k: isize| data[(pos as isize + k * across as isize) as usize] as i32;
    // Sample k >= 0 is q_k, k < 0 is p_(-k-1).
    let (p1, p0, q0, q1) = (at(-2), at(-1), at(0), at(1));
    let mut mask = (p1 - p0).abs() <= lim.limit
        && (q1 - q0).abs() <= lim.limit
        && (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= lim.blimit;
    let (mut p2, mut q2, mut p3, mut q3) = (0, 0, 0, 0);
    if length != Length::Four {
        p2 = at(-3);
        q2 = at(2);
        mask &= (p2 - p1).abs() <= lim.limit && (q2 - q1).abs() <= lim.limit;
        if length != Length::Six {
            p3 = at(-4);
            q3 = at(3);
            mask &= (p3 - p2).abs() <= lim.limit && (q3 - q2).abs() <= lim.limit;
        }
    }
    if !mask {
        return;
    }
    let t = lim.flat;
    let mut flat = false;
    if length != Length::Four {
        flat = (p1 - p0).abs() <= t && (q1 - q0).abs() <= t && (p2 - p0).abs() <= t && (q2 - q0).abs() <= t;
        if length != Length::Six {
            flat &= (p3 - p0).abs() <= t && (q3 - q0).abs() <= t;
        }
    }
    if !flat {
        narrow_filter(data, pos, across, lim, [p1, p0, q0, q1]);
        return;
    }
    let mut flat2 = false;
    if length == Length::Fourteen {
        flat2 = (4..=6).all(|k| (at(-1 - k) - p0).abs() <= t && (at(k) - q0).abs() <= t);
    }
    let mut px = [0i32; 14];
    for (k, v) in px.iter_mut().enumerate() {
        // Only the taps the chosen filter needs are read; the rest stay 0 and are never used.
        let k = k as isize - 7;
        let needed = match length {
            Length::Six => (-3..=2).contains(&k),
            Length::Eight => (-4..=3).contains(&k),
            Length::Fourteen => flat2 || (-4..=3).contains(&k),
            Length::Four => false,
        };
        if needed {
            *v = at(k);
        }
    }
    let (n, n2, log2) = match length {
        _ if flat2 => (6, 1, 4),
        Length::Six => (2, 1, 3),
        _ => (3, 0, 3),
    };
    wide_filter(data, pos, across, &px, n, n2, log2);
}

/// The narrow (4-tap) filter, spec 7.14.6.3.
fn narrow_filter(data: &mut [u16], pos: usize, across: usize, lim: &Limits, [p1, p0, q0, q1]: [i32; 4]) {
    let bd = lim.bit_depth;
    let off = 0x80 << (bd - 8);
    let (lo, hi) = (-(1i32 << (bd - 1)), (1i32 << (bd - 1)) - 1);
    let c = |v: i32| v.clamp(lo, hi);
    let hev = (p1 - p0).abs() > lim.thresh || (q1 - q0).abs() > lim.thresh;
    let (ps1, ps0, qs0, qs1) = (p1 - off, p0 - off, q0 - off, q1 - off);
    let mut filter = if hev { c(ps1 - qs1) } else { 0 };
    filter = c(filter + 3 * (qs0 - ps0));
    let filter1 = c(filter + 4) >> 3;
    let filter2 = c(filter + 3) >> 3;
    let mut put = |k: isize, v: i32| data[(pos as isize + k * across as isize) as usize] = (c(v) + off) as u16;
    put(0, qs0 - filter1);
    put(-1, ps0 + filter2);
    if !hev {
        let f = (filter1 + 1) >> 1;
        put(1, qs1 - f);
        put(-2, ps1 + f);
    }
}

/// The wide filters, spec 7.14.6.4. `px[k + 7]` holds sample `k` (negative: p side); modifies samples `-n..n`.
fn wide_filter(data: &mut [u16], pos: usize, across: usize, px: &[i32; 14], n: isize, n2: isize, log2: u32) {
    for i in -n..n {
        let mut t = 0;
        for j in -n..=n {
            let k = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            t += px[(k + 7) as usize] * tap;
        }
        data[(pos as isize + i * across as isize) as usize] = ((t + (1 << (log2 - 1))) >> log2) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- deterministic scene generator shared with the reference harness (this block is pasted verbatim) ----

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 33) as usize % n
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + self.below((hi - lo + 1) as usize) as i32
        }

        fn chance(&mut self, pct: usize) -> bool {
            self.below(100) < pct
        }
    }

    fn fnv(frame: &Frame) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for p in &frame.planes {
            for y in 0..p.height {
                for &s in p.row(y) {
                    h = (h ^ s as u64).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        h
    }

    /// Transform size one split down (`Split_Tx_Size`), in 4-sample units.
    fn split_tx(t: (usize, usize)) -> (usize, usize) {
        let (w, h) = t;
        if w == h {
            ((w / 2).max(1), (h / 2).max(1))
        } else if w == 2 * h || h == 2 * w {
            (w.min(h), w.min(h))
        } else if w > h {
            (w / 2, h)
        } else {
            (w, h / 2)
        }
    }

    fn block_index(w: usize, h: usize) -> u8 {
        BLOCK_DIMS_4.iter().position(|&(bw, bh)| (bw as usize, bh as usize) == (w, h)).unwrap() as u8
    }

    fn tx_index(w: usize, h: usize) -> u8 {
        TX_DIMS_4.iter().position(|&(tw, th)| (tw as usize, th as usize) == (w, h)).unwrap() as u8
    }

    /// One block of a generated scene (sizes in 4-sample units).
    #[derive(Clone, Copy)]
    struct Leaf {
        bx: usize,
        by: usize,
        w4: usize,
        h4: usize,
        ytx: (usize, usize),
        uvtx: (usize, usize),
        has_chroma: bool,
        segment: usize,
        lf_delta: [i8; 4],
        ref_frame: usize,
        mode_type: usize,
        skip: bool,
    }

    #[allow(dead_code)] // `leaves` feeds the reference harness
    struct Scene {
        frame: Frame,
        params: DeblockParams,
        mi: DeblockMi,
        leaves: Vec<Leaf>,
    }

    /// A 64x64-luma frame tiled with blocks of random shapes (one shape per 16x16 area), random transform sizes,
    /// skip/ref/segment/delta_lf values and a pixel pattern with blocky steps plus noise.
    fn scene(seed: u64, bit_depth: u8, sx: usize, sy: usize, mono: bool, inter: bool) -> Scene {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let (mi_cols, mi_rows) = (16, 16);
        let is422 = sx == 1 && sy == 0;
        let shapes: &[(usize, usize)] =
            &[(4, 4), (4, 2), (2, 4), (2, 2), (4, 1), (1, 4), (2, 1), (1, 2), (1, 1), (4, 4), (2, 2)];
        let mut leaves = Vec::new();
        for qy in 0..4 {
            for qx in 0..4 {
                let (w4, h4) = loop {
                    let s = shapes[rng.below(shapes.len())];
                    if !(is422 && s.0 == 1 && s.1 > 1) {
                        break s;
                    }
                };
                for y in (0..4).step_by(h4) {
                    for x in (0..4).step_by(w4) {
                        leaves.push(Leaf {
                            bx: qx * 4 + x,
                            by: qy * 4 + y,
                            w4,
                            h4,
                            ytx: (0, 0),
                            uvtx: (0, 0),
                            has_chroma: false,
                            segment: 0,
                            lf_delta: [0; 4],
                            ref_frame: 0,
                            mode_type: 0,
                            skip: false,
                        });
                    }
                }
            }
        }
        let seg_on = rng.chance(50);
        let multi = rng.chance(50);
        let lf_present = rng.chance(50);
        let skip_pct = rng.below(100);
        let mut mi = DeblockMi::new(mi_cols, mi_rows, sx, sy, if mono { 1 } else { 3 });
        for lf in leaves.iter_mut() {
            let mut ytx = (lf.w4.min(16), lf.h4.min(16));
            for _ in 0..rng.below(3) {
                ytx = split_tx(ytx);
            }
            lf.skip = rng.chance(skip_pct);
            lf.ref_frame = if inter && rng.chance(60) { rng.range(1, 7) as usize } else { 0 };
            if lf.skip && lf.ref_frame != 0 {
                ytx = (lf.w4, lf.h4);
            }
            lf.ytx = ytx;
            let y_mode = if lf.ref_frame == 0 { rng.below(13) as u8 } else { rng.range(13, 24) as u8 };
            lf.mode_type = (y_mode >= 13 && y_mode != 15 && y_mode != 23) as usize;
            lf.segment = if seg_on { rng.below(8) } else { 0 };
            if lf_present {
                for d in lf.lf_delta.iter_mut() {
                    *d = rng.range(-30, 30) as i8;
                }
            }
            lf.has_chroma =
                !mono && (sx == 0 || lf.w4 > 1 || lf.bx & 1 == 1) && (sy == 0 || lf.h4 > 1 || lf.by & 1 == 1);
            lf.uvtx = ((((lf.w4 * 4) >> sx).max(4) / 4).min(8), (((lf.h4 * 4) >> sy).max(4) / 4).min(8));
            for y in 0..lf.h4 {
                for x in 0..lf.w4 {
                    let i = (lf.by + y) * mi_cols + lf.bx + x;
                    mi.mi_size[i] = block_index(lf.w4, lf.h4);
                    mi.skip[i] = lf.skip;
                    mi.ref_frame[i] = lf.ref_frame as i8;
                    mi.y_mode[i] = y_mode;
                    mi.segment_id[i] = lf.segment as u8;
                    mi.delta_lf[i] = lf.lf_delta;
                    mi.tx_size[0][i] = tx_index(lf.ytx.0, lf.ytx.1);
                }
            }
            if lf.has_chroma {
                let stride = mi.tx_stride(sx);
                for y in 0..(lf.h4 + sy) >> sy {
                    for x in 0..(lf.w4 + sx) >> sx {
                        let i = ((lf.by >> sy) + y) * stride + (lf.bx >> sx) + x;
                        mi.tx_size[1][i] = tx_index(lf.uvtx.0, lf.uvtx.1);
                        mi.tx_size[2][i] = tx_index(lf.uvtx.0, lf.uvtx.1);
                    }
                }
            }
        }
        let mut params = DeblockParams {
            filter_level: [
                rng.range(1, 63) as u8,
                rng.range(0, 63) as u8,
                rng.range(0, 63) as u8,
                rng.range(0, 63) as u8,
            ],
            sharpness: if rng.chance(40) { 0 } else { rng.range(1, 7) as u8 },
            delta_enabled: rng.chance(60),
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [rng.range(-8, 8) as i8, rng.range(-8, 8) as i8],
            delta_lf_multi: multi,
            seg_lf: [[0; 4]; 8],
        };
        if rng.chance(30) {
            for l in params.filter_level.iter_mut() {
                *l = (*l % 10).max(1);
            }
        }
        if seg_on {
            for s in params.seg_lf.iter_mut() {
                for v in s.iter_mut() {
                    *v = rng.range(-20, 20) as i8;
                }
            }
        }
        if mono {
            params.filter_level[2] = 0;
            params.filter_level[3] = 0;
        }
        // pixels: smooth ramp + a random step per 4x4 + noise
        let shift = bit_depth as i32 - 8;
        let step = [0, 1, 3, 8][rng.below(4)] << shift;
        let noise = [0, 1, 2, 5][rng.below(4)] << shift;
        let (vis_w, vis_h) = (61, 58);
        let mut planes = Vec::new();
        for pl in 0..if mono { 1 } else { 3 } {
            let (a, b) = if pl == 0 { (0, 0) } else { (sx, sy) };
            let (w, h) = ((mi_cols * 4) >> a, (mi_rows * 4) >> b);
            let mut p = Plane::new(w, h);
            let base = rng.range(40, 200) << shift;
            let offs: Vec<i32> = (0..(w / 4) * (h / 4)).map(|_| rng.range(-step, step)).collect();
            for y in 0..h {
                for x in 0..w {
                    let v = base
                        + ((x as i32 - y as i32) << shift) / 8
                        + offs[(y / 4) * (w / 4) + x / 4]
                        + rng.range(-noise, noise);
                    p.data[y * p.stride + x] = v.clamp(0, (1 << bit_depth) - 1) as u16;
                }
            }
            planes.push(p);
        }
        let frame = Frame { planes, bit_depth, sub_x: sx as u8, sub_y: sy as u8, width: vis_w, height: vis_h };
        Scene { frame, params, mi, leaves }
    }

    /// (filter length, bit depth, blimit, limit, thresh before bit-depth scaling, input, expected output): the
    /// samples `p6..p0 q0..q6` around one edge, filtered by the rav1d (dav1d) reference kernel.
    #[allow(clippy::type_complexity)]
    const KERNEL_VECTORS: &[(u32, u32, i32, i32, i32, [u16; 14], [u16; 14])] = &[
        (
            6,
            8,
            43,
            13,
            0,
            [92, 90, 91, 92, 92, 90, 91, 92, 93, 93, 92, 93, 93, 92],
            [92, 90, 91, 92, 92, 91, 91, 92, 93, 93, 92, 93, 93, 92],
        ),
        (
            16,
            8,
            40,
            4,
            1,
            [193, 194, 192, 194, 193, 193, 193, 188, 186, 186, 187, 187, 188, 187],
            [193, 194, 192, 194, 193, 193, 192, 189, 186, 186, 187, 187, 188, 187],
        ),
        (
            6,
            12,
            63,
            5,
            1,
            [1402, 1393, 1420, 1417, 1449, 1395, 1414, 1463, 1458, 1477, 1511, 1477, 1511, 1471],
            [1402, 1393, 1420, 1417, 1449, 1395, 1424, 1452, 1458, 1477, 1511, 1477, 1511, 1471],
        ),
        (
            16,
            10,
            132,
            6,
            3,
            [368, 368, 368, 368, 368, 368, 368, 356, 356, 356, 356, 356, 356, 356],
            [368, 367, 367, 366, 365, 364, 363, 361, 360, 359, 358, 358, 357, 356],
        ),
        (
            8,
            10,
            100,
            8,
            2,
            [251, 260, 251, 259, 253, 262, 255, 279, 272, 270, 276, 268, 270, 283],
            [251, 260, 251, 259, 253, 267, 264, 270, 267, 270, 276, 268, 270, 283],
        ),
        (
            6,
            12,
            119,
            3,
            3,
            [2530, 2537, 2514, 2530, 2513, 2543, 2534, 2440, 2436, 2461, 2462, 2443, 2434, 2440],
            [2530, 2537, 2514, 2530, 2513, 2526, 2499, 2475, 2453, 2461, 2462, 2443, 2434, 2440],
        ),
        (
            8,
            8,
            14,
            2,
            0,
            [182, 182, 182, 182, 182, 182, 182, 188, 188, 188, 188, 188, 188, 188],
            [182, 182, 182, 182, 182, 182, 182, 188, 188, 188, 188, 188, 188, 188],
        ),
        (
            16,
            8,
            37,
            11,
            0,
            [24, 27, 27, 27, 27, 27, 26, 29, 30, 29, 28, 28, 31, 29],
            [24, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 31, 29],
        ),
        (
            6,
            8,
            111,
            5,
            3,
            [183, 182, 183, 183, 183, 182, 183, 184, 184, 183, 185, 183, 184, 184],
            [183, 182, 183, 183, 183, 183, 183, 183, 184, 183, 185, 183, 184, 184],
        ),
        (
            4,
            10,
            112,
            8,
            3,
            [660, 660, 660, 660, 660, 660, 660, 668, 668, 668, 668, 668, 668, 668],
            [660, 660, 660, 660, 660, 662, 663, 665, 666, 668, 668, 668, 668, 668],
        ),
        (
            16,
            12,
            67,
            7,
            1,
            [1335, 1321, 1327, 1344, 1332, 1329, 1333, 1400, 1403, 1421, 1405, 1412, 1402, 1412],
            [1335, 1321, 1327, 1344, 1332, 1342, 1358, 1375, 1390, 1421, 1405, 1412, 1402, 1412],
        ),
        (
            4,
            12,
            39,
            7,
            0,
            [873, 880, 839, 871, 837, 819, 835, 873, 897, 887, 908, 894, 882, 895],
            [873, 880, 839, 871, 837, 819, 839, 868, 897, 887, 908, 894, 882, 895],
        ),
        (
            8,
            12,
            50,
            4,
            1,
            [2495, 2505, 2495, 2518, 2515, 2506, 2481, 2556, 2512, 2519, 2540, 2536, 2569, 2547],
            [2495, 2505, 2495, 2518, 2515, 2506, 2508, 2529, 2512, 2519, 2540, 2536, 2569, 2547],
        ),
        (
            16,
            8,
            74,
            2,
            2,
            [172, 172, 172, 172, 172, 172, 172, 176, 176, 176, 176, 176, 176, 176],
            [172, 172, 173, 173, 173, 173, 174, 174, 175, 175, 175, 176, 176, 176],
        ),
        (
            8,
            10,
            38,
            2,
            1,
            [656, 656, 656, 656, 656, 656, 656, 676, 676, 676, 676, 676, 676, 676],
            [656, 656, 656, 656, 659, 661, 664, 669, 671, 674, 676, 676, 676, 676],
        ),
        (
            16,
            12,
            120,
            2,
            3,
            [1381, 1392, 1408, 1388, 1376, 1394, 1379, 1446, 1453, 1449, 1431, 1448, 1430, 1452],
            [1381, 1392, 1408, 1388, 1392, 1402, 1408, 1422, 1430, 1434, 1431, 1448, 1430, 1452],
        ),
        (
            6,
            10,
            92,
            2,
            2,
            [294, 292, 296, 291, 293, 296, 293, 300, 300, 302, 299, 302, 299, 303],
            [294, 292, 296, 291, 293, 295, 296, 298, 300, 302, 299, 302, 299, 303],
        ),
        (
            6,
            12,
            187,
            61,
            3,
            [398, 404, 392, 390, 415, 391, 400, 298, 302, 306, 291, 299, 301, 308],
            [398, 404, 392, 390, 415, 391, 362, 337, 315, 306, 291, 299, 301, 308],
        ),
        (
            8,
            8,
            73,
            23,
            1,
            [85, 89, 85, 89, 87, 89, 88, 88, 85, 85, 87, 88, 87, 86],
            [85, 89, 85, 89, 87, 89, 88, 87, 85, 85, 87, 88, 87, 86],
        ),
        (
            4,
            8,
            57,
            3,
            1,
            [135, 135, 133, 135, 134, 135, 135, 138, 140, 140, 139, 140, 140, 139],
            [135, 135, 133, 135, 134, 135, 135, 137, 140, 140, 139, 140, 140, 139],
        ),
        (
            16,
            8,
            52,
            4,
            1,
            [142, 140, 142, 141, 140, 141, 140, 141, 141, 140, 140, 141, 139, 139],
            [142, 140, 142, 141, 141, 141, 141, 141, 141, 140, 140, 141, 139, 139],
        ),
        (
            8,
            12,
            133,
            43,
            2,
            [1691, 1691, 1665, 1680, 1692, 1689, 1679, 1683, 1680, 1683, 1672, 1677, 1682, 1668],
            [1691, 1691, 1665, 1680, 1684, 1684, 1683, 1683, 1680, 1678, 1672, 1677, 1682, 1668],
        ),
    ];

    #[test]
    fn kernel_matches_reference_vectors() {
        for (n, &(wd, bd, e, i, h, input, expected)) in KERNEL_VECTORS.iter().enumerate() {
            let s = bd - 8;
            let lim = Limits { limit: i << s, blimit: e << s, thresh: h << s, flat: 1 << s, bit_depth: bd };
            let length = match wd {
                4 => Length::Four,
                6 => Length::Six,
                8 => Length::Eight,
                _ => Length::Fourteen,
            };
            let mut data = input;
            filter_sample(&mut data, 7, 1, length, &lim);
            assert_eq!(data, expected, "vector {n}");
        }
    }

    /// (seed, bit depth, sub_x, sub_y, monochrome, inter blocks, FNV-1a of the filtered planes from the reference).
    const GOLDEN: &[(u64, u8, usize, usize, bool, bool, u64)] = &[
        (1, 8, 1, 1, false, false, 0x83cb929814edfa49),
        (2, 8, 1, 1, false, true, 0xb680a21e9af5ad9d),
        (3, 8, 0, 0, false, false, 0x2b3ba7fe4205dd2c),
        (4, 10, 1, 0, false, true, 0xff9918ada966cb87),
        (5, 12, 1, 1, false, false, 0x81c919ae6a67ec15),
        (6, 10, 0, 0, false, true, 0x5efeec37f44219db),
        (8, 12, 1, 0, false, false, 0x85a140117e2c0dd1),
        (9, 8, 1, 1, false, true, 0x2b63d80337e788a0),
        (10, 10, 1, 1, false, false, 0xe39dad30e885632f),
        (11, 8, 0, 0, false, true, 0x1e166e1ce62e2b7d),
        (12, 12, 1, 1, false, true, 0xbd288827a07f56a0),
        (13, 8, 1, 1, true, false, 0x614bae58e20b63f6),
        (14, 10, 1, 1, true, true, 0x59b3373690969dd6),
    ];

    #[test]
    fn frames_match_reference_hashes() {
        for &(seed, bd, sx, sy, mono, inter, hash) in GOLDEN {
            let s = scene(seed, bd, sx, sy, mono, inter);
            let mut f = s.frame.clone();
            deblock_frame(&mut f, &s.params, &s.mi);
            assert_ne!(fnv(&f), fnv(&s.frame), "scene {seed} should change the picture");
            assert_eq!(fnv(&f), hash, "scene {seed}");
        }
    }

    fn mono_frame(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Frame {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                p.data[y * w + x] = f(x, y);
            }
        }
        Frame { planes: vec![p], bit_depth: 8, sub_x: 1, sub_y: 1, width: w, height: h }
    }

    fn params(level: u8) -> DeblockParams {
        DeblockParams { filter_level: [level, level, level, level], ..Default::default() }
    }

    /// 16x8 luma: two 8x8 blocks with 8x8 transforms and a small step between them.
    fn two_blocks(skip: bool, inter: bool) -> (Frame, DeblockMi) {
        let frame = mono_frame(16, 8, |x, _| if x < 8 { 100 } else { 104 });
        let mut mi = DeblockMi::new(4, 2, 1, 1, 1);
        mi.mi_size.fill(3);
        mi.skip.fill(skip);
        mi.ref_frame.fill(inter as i8);
        mi.tx_size[0].fill(1);
        (frame, mi)
    }

    #[test]
    fn intra_block_edge_is_smoothed() {
        let (mut frame, mi) = two_blocks(false, false);
        deblock_frame(&mut frame, &params(32), &mi);
        let row = frame.planes[0].row(3).to_vec();
        assert_eq!(&row[..4], &[100; 4]);
        assert_eq!(&row[12..], &[104; 4]);
        // Both sides are flat, so the 8-tap filter smooths the step (p2 = (3*100 + 2*100 + 100 + 100 + 104 + 4) >> 3).
        assert_eq!(&row[4..12], &[100, 101, 101, 102, 103, 103, 104, 104]);
        // every row is treated alike
        assert!((0..8).all(|y| frame.planes[0].row(y) == row));
    }

    #[test]
    fn zero_levels_leave_the_frame_alone() {
        let (mut frame, mi) = two_blocks(false, false);
        let before = frame.planes[0].data.clone();
        deblock_frame(&mut frame, &params(0), &mi);
        assert_eq!(frame.planes[0].data, before);
        // luma levels of zero disable everything, whatever the chroma levels say
        let p = DeblockParams { filter_level: [0, 0, 20, 20], ..Default::default() };
        deblock_frame(&mut frame, &p, &mi);
        assert_eq!(frame.planes[0].data, before);
    }

    #[test]
    fn only_the_direction_with_a_level_is_filtered() {
        // Vertical edges (pass 0) use filter_level[0], horizontal ones filter_level[1].
        let mut frame = mono_frame(16, 16, |x, y| if (x < 8) == (y < 8) { 100 } else { 104 });
        let mut mi = DeblockMi::new(4, 4, 1, 1, 1);
        mi.mi_size.fill(3);
        mi.tx_size[0].fill(1);
        let before = frame.planes[0].data.clone();
        let p = DeblockParams { filter_level: [32, 0, 0, 0], ..Default::default() };
        deblock_frame(&mut frame, &p, &mi);
        // Samples on both sides of the horizontal edge (y = 8), away from the vertical one, are untouched.
        assert_eq!(frame.planes[0].at(0, 7), before[7 * 16]);
        assert_eq!(frame.planes[0].at(0, 8), before[8 * 16]);
        assert_ne!(frame.planes[0].at(7, 3), before[3 * 16 + 7]);
    }

    #[test]
    fn skipped_inter_blocks_only_filter_block_edges() {
        // One 16x8 block with 8x8 transforms: its inner transform edge (x = 8) is filtered only with residual.
        let build = |skip: bool, inter: bool| {
            let (mut frame, mut mi) = two_blocks(skip, inter);
            mi.mi_size.fill(5); // BLOCK_16X8
            deblock_frame(&mut frame, &params(32), &mi);
            frame
        };
        let skipped = build(true, true);
        assert_eq!(skipped.planes[0].row(3)[7], 100);
        assert_eq!(skipped.planes[0].row(3)[8], 104);
        assert_ne!(build(false, true).planes[0].row(3)[7], 100);
        // An intra block is always filtered, skip or not.
        assert_ne!(build(true, false).planes[0].row(3)[7], 100);
    }

    #[test]
    fn edges_outside_the_visible_area_are_skipped() {
        let (mut frame, mi) = two_blocks(false, false);
        frame.width = 8; // the edge at x = 8 starts outside the picture
        let before = frame.planes[0].data.clone();
        deblock_frame(&mut frame, &params(32), &mi);
        assert_eq!(frame.planes[0].data, before);
    }

    #[test]
    fn neighbour_level_is_used_when_the_block_level_is_zero() {
        // delta_lf zeroes the right block's level; the left block's level then applies to the shared edge.
        let (mut frame, mut mi) = two_blocks(false, false);
        for r in 0..2 {
            for c in 2..4 {
                mi.delta_lf[r * 4 + c] = [-63; 4];
            }
        }
        deblock_frame(&mut frame, &params(32), &mi);
        assert_ne!(frame.planes[0].row(3)[7], 100);
        // Zero on the left block only: the current (right-hand) block decides, so it is filtered as well.
        let (mut frame, mut mi) = two_blocks(false, false);
        for r in 0..2 {
            for c in 0..2 {
                mi.delta_lf[r * 4 + c] = [-63; 4];
            }
        }
        deblock_frame(&mut frame, &params(32), &mi);
        assert_ne!(frame.planes[0].row(3)[7], 100);
        // Zero on both: nothing happens.
        let (mut frame, mut mi) = two_blocks(false, false);
        mi.delta_lf.fill([-63; 4]);
        let before = frame.planes[0].data.clone();
        deblock_frame(&mut frame, &params(32), &mi);
        assert_eq!(frame.planes[0].data, before);
    }

    #[test]
    fn level_follows_segments_deltas_and_delta_lf() {
        let mut mi = DeblockMi::new(2, 2, 1, 1, 1);
        mi.segment_id[0] = 2;
        mi.delta_lf[0] = [5, 6, 7, 8];
        let mut p = DeblockParams { filter_level: [10, 20, 30, 40], ..Default::default() };
        p.seg_lf[2] = [1, 2, 3, 4];
        let sharp = SharpnessLut::new(0);
        let level = |mi: &DeblockMi, plane_idx: usize, pass: usize, p: &DeblockParams, idx: usize| {
            PassCtx { p, mi, sharp: &sharp, plane_idx, pass, sx: 1, sy: 1, bit_depth: 8, frame_w: 8, frame_h: 8 }
                .level(idx)
        };
        // Without delta_lf_multi the first delta applies to every component.
        assert_eq!(level(&mi, 0, 0, &p, 0), 10 + 5 + 1);
        assert_eq!(level(&mi, 0, 1, &p, 0), 20 + 5 + 2);
        assert_eq!(level(&mi, 1, 0, &p, 0), 30 + 5 + 3);
        p.delta_lf_multi = true;
        assert_eq!(level(&mi, 0, 1, &p, 0), 20 + 6 + 2);
        assert_eq!(level(&mi, 1, 0, &p, 0), 30 + 7 + 3);
        assert_eq!(level(&mi, 2, 0, &p, 0), 40 + 8 + 4);
        // Intra blocks add ref_deltas[0], scaled by 1 << (level >> 5).
        p.delta_enabled = true;
        p.ref_deltas[0] = 3;
        assert_eq!(level(&mi, 0, 0, &p, 0), 10 + 5 + 1 + 3);
        assert_eq!(level(&mi, 2, 0, &p, 0), 40 + 8 + 4 + (3 << 1));
        // Block 1 has no delta_lf and segment 0.
        assert_eq!(level(&mi, 0, 0, &p, 1), 10 + 3);
        // Clamping at both ends.
        p.ref_deltas[0] = -40;
        assert_eq!(level(&mi, 0, 0, &p, 1), 0);
        p.ref_deltas[0] = 60;
        assert_eq!(level(&mi, 0, 0, &p, 1), 63);
        // Inter blocks use the reference delta and the delta of their mode type.
        p.ref_deltas[0] = 0;
        p.ref_deltas[2] = 2;
        p.mode_deltas = [-1, 4];
        mi.ref_frame[1] = 2;
        mi.y_mode[1] = GLOBALMV;
        assert_eq!(level(&mi, 0, 0, &p, 1), 10 + 2 - 1);
        mi.y_mode[1] = NEARESTMV;
        assert_eq!(level(&mi, 0, 0, &p, 1), 10 + 2 + 4);
    }

    #[test]
    fn sharpness_limits() {
        let l = SharpnessLut::new(0);
        assert_eq!((l.limit[0], l.limit[1], l.limit[40], l.limit[63]), (1, 1, 40, 63));
        let l = SharpnessLut::new(3);
        assert_eq!((l.limit[1], l.limit[8], l.limit[63]), (1, 4, 6));
        let l = SharpnessLut::new(7);
        assert_eq!((l.limit[3], l.limit[63]), (1, 2));
    }

    #[test]
    fn mode_info_defaults() {
        let mi = DeblockMi::new(6, 4, 1, 0, 3);
        assert_eq!(mi.mi_size.len(), 24);
        assert_eq!(mi.tx_size[1].len(), 3 * 4);
        assert_eq!(mi.tx_stride(1), 3);
        assert!(DeblockMi::new(2, 2, 1, 1, 1).tx_size[1].is_empty());
    }
}
