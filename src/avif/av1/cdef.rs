//! The AV1 constrained directional enhancement filter, CDEF (specification section 7.15).
//!
//! CDEF runs after deblocking, on every 8x8 luma block (with its chroma counterpart) that is not skipped and whose
//! 64x64 filter block has a strength index. For each block the luma content is searched for its dominant direction;
//! every sample is then nudged towards its neighbours along that direction (primary taps) and the two directions
//! 45 degrees off it (secondary taps), where each neighbour's influence is limited by `constrain`, so strong edges are
//! preserved. All reads come from the deblocked frame and all writes go to a separate output frame.

use super::frame::{Frame, Plane};

/// Frame-level CDEF syntax (`cdef_params()`).
#[derive(Clone, Debug, Default)]
pub struct CdefParams {
    /// `CdefDamping` = `cdef_damping_minus_3 + 3`.
    pub damping: u8,
    /// `cdef_bits` (informational: `cdef_idx` values are below `1 << bits`).
    pub bits: u8,
    /// `cdef_y_pri_strength[i]` (0..=15), indexed by a 64x64 block's `cdef_idx`.
    pub y_pri: [u8; 8],
    /// `cdef_y_sec_strength[i]` with the spec's mapping already applied (the coded value 3 means 4): 0, 1, 2 or 4.
    pub y_sec: [u8; 8],
    /// `cdef_uv_pri_strength[i]`.
    pub uv_pri: [u8; 8],
    /// `cdef_uv_sec_strength[i]`, mapped like `y_sec`.
    pub uv_sec: [u8; 8],
}

/// `Cdef_Directions[dir][k]` as `(row, column)` offsets.
const DIRECTIONS: [[(i32, i32); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];

/// `Cdef_Pri_Taps[(priStr >> coeffShift) & 1][k]`.
const PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
/// `Cdef_Sec_Taps[k]` (identical for both rows of the spec's table).
const SEC_TAPS: [i32; 2] = [2, 1];
/// `Cdef_Uv_Dir` for 4:2:2 (sub_x = 1, sub_y = 0); the other layouts map a direction to itself.
const UV_DIR_422: [u8; 8] = [7, 0, 2, 4, 5, 6, 6, 6];

/// Marks a neighbour outside the frame in the padded block.
const UNAVAILABLE: i32 = -1;

/// Runs CDEF over a whole frame.
///
/// * `pre` is the deblocked frame (the spec's `CurrFrame`), never modified.
/// * `out` receives the result (`CdefFrame`): it is overwritten with a copy of `pre` first, then the filtered blocks.
/// * `cdef_idx` holds `cdef_idx[][]` per 64x64 luma block, row-major with `cdef_stride` entries per row; `-1` means
///   the block is not filtered (all its blocks were skipped, so no index was coded).
/// * `skip` holds the `skip` flag per 4x4 mode-info unit (`mi_cols` x `mi_rows`, row-major); an 8x8 block is filtered
///   unless all four of its units are skipped.
///
/// The planes' coded size must be `mi_cols * 4` x `mi_rows * 4` luma samples (halved per subsampling); samples
/// outside it count as unavailable, those inside it (even beyond the visible picture) as regular samples.
#[allow(clippy::too_many_arguments)]
pub fn cdef_frame(
    pre: &Frame,
    out: &mut Frame,
    p: &CdefParams,
    cdef_idx: &[i8],
    cdef_stride: usize,
    skip: &[bool],
    mi_cols: usize,
    mi_rows: usize,
) {
    copy_frame(pre, out);
    let num_planes = pre.planes.len();
    let (sx, sy) = (pre.sub_x as usize, pre.sub_y as usize);
    let coeff_shift = pre.bit_depth as u32 - 8;
    let skip_at = |r: usize, c: usize| skip[r.min(mi_rows - 1) * mi_cols + c.min(mi_cols - 1)];
    let mut r = 0;
    while r < mi_rows {
        let mut c = 0;
        while c < mi_cols {
            let idx = cdef_idx[(r >> 4) * cdef_stride + (c >> 4)];
            let all_skipped = skip_at(r, c) && skip_at(r + 1, c) && skip_at(r, c + 1) && skip_at(r + 1, c + 1);
            if idx >= 0 && !all_skipped {
                let idx = idx as usize & 7;
                let (y_dir, var) = find_direction(&pre.planes[0], c * 4, r * 4, pre.bit_depth as u32);

                // Luma: the direction only matters with a primary strength, whose size follows the local variance.
                let pri = (p.y_pri[idx] as i32) << coeff_shift;
                let sec = (p.y_sec[idx] as i32) << coeff_shift;
                let dir = if pri == 0 { 0 } else { y_dir };
                let var_str = if (var >> 6) != 0 { floor_log2((var >> 6) as u32).min(12) } else { 0 };
                let pri = if var != 0 { (pri * (4 + var_str) + 8) >> 4 } else { 0 };
                let damping = p.damping as i32 + coeff_shift as i32;
                let blk = Block { c, r, sx: 0, sy: 0, mi_cols, mi_rows };
                filter_block(&pre.planes[0], &mut out.planes[0], &blk, [pri, sec, damping, dir as i32], coeff_shift);

                if num_planes > 1 {
                    let pri = (p.uv_pri[idx] as i32) << coeff_shift;
                    let sec = (p.uv_sec[idx] as i32) << coeff_shift;
                    let dir = if pri == 0 {
                        0
                    } else if sx == 1 && sy == 0 {
                        UV_DIR_422[y_dir] as i32
                    } else {
                        y_dir as i32
                    };
                    let damping = damping - 1;
                    let blk = Block { c, r, sx, sy, mi_cols, mi_rows };
                    for plane in 1..num_planes {
                        filter_block(
                            &pre.planes[plane],
                            &mut out.planes[plane],
                            &blk,
                            [pri, sec, damping, dir],
                            coeff_shift,
                        );
                    }
                }
            }
            c += 2;
        }
        r += 2;
    }
}

/// Makes `out` an exact copy of `pre` (reusing its buffers when the shapes already match).
fn copy_frame(pre: &Frame, out: &mut Frame) {
    let same_shape = out.planes.len() == pre.planes.len()
        && out.planes.iter().zip(&pre.planes).all(|(a, b)| a.data.len() == b.data.len() && a.stride == b.stride);
    if same_shape {
        for (a, b) in out.planes.iter_mut().zip(&pre.planes) {
            a.data.copy_from_slice(&b.data);
            a.width = b.width;
            a.height = b.height;
        }
        out.bit_depth = pre.bit_depth;
        out.sub_x = pre.sub_x;
        out.sub_y = pre.sub_y;
        out.width = pre.width;
        out.height = pre.height;
    } else {
        *out = pre.clone();
    }
}

fn floor_log2(v: u32) -> i32 {
    31 - v.leading_zeros() as i32
}

/// The CDEF direction process (7.15.2) on the 8x8 luma block at `(x0, y0)`: returns the dominant direction (0..8) and
/// the directional variance used to scale the primary strength.
fn find_direction(plane: &Plane, x0: usize, y0: usize, bit_depth: u32) -> (usize, i32) {
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        let row = &plane.row(y0 + i)[x0..x0 + 8];
        for (j, &s) in row.iter().enumerate() {
            let x = (s as i32 >> (bit_depth - 8)) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    const DIV_TABLE: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let sq = |v: i32| (v as i64) * (v as i64);
    let mut cost = [0i64; 8];
    for i in 0..8 {
        cost[2] += sq(partial[2][i]);
        cost[6] += sq(partial[6][i]);
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (sq(partial[0][i]) + sq(partial[0][14 - i])) * DIV_TABLE[i + 1];
        cost[4] += (sq(partial[4][i]) + sq(partial[4][14 - i])) * DIV_TABLE[i + 1];
    }
    cost[0] += sq(partial[0][7]) * DIV_TABLE[8];
    cost[4] += sq(partial[4][7]) * DIV_TABLE[8];
    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += sq(partial[i][3 + j]);
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (sq(partial[i][j]) + sq(partial[i][10 - j])) * DIV_TABLE[2 * j + 2];
        }
    }
    let mut best_dir = 0;
    let mut best_cost = 0;
    for (d, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            best_dir = d;
        }
    }
    (best_dir, ((best_cost - cost[(best_dir + 4) & 7]) >> 10) as i32)
}

/// Position of an 8x8 luma block (in 4x4 units) and the plane's subsampling.
struct Block {
    c: usize,
    r: usize,
    sx: usize,
    sy: usize,
    mi_cols: usize,
    mi_rows: usize,
}

/// `constrain()` of the spec: `diff` limited so that large differences (edges) contribute little or nothing.
#[inline]
fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let adj = (damping - floor_log2(threshold as u32)).max(0);
    let mag = diff.abs();
    let v = mag.min((threshold - (mag >> adj)).max(0));
    if diff < 0 { -v } else { v }
}

/// The CDEF filter process (7.15.3) for the part of an 8x8 luma block lying in one plane. `[pri, sec, damping, dir]`.
fn filter_block(src: &Plane, dst: &mut Plane, blk: &Block, params: [i32; 4], coeff_shift: u32) {
    let [pri, sec, damping, dir] = params;
    if pri == 0 && sec == 0 {
        return;
    }
    let w = 8 >> blk.sx;
    let h = 8 >> blk.sy;
    let x0 = (blk.c * 4) >> blk.sx;
    let y0 = (blk.r * 4) >> blk.sy;
    // Samples at or beyond these limits are outside the frame (CdefAvailable = 0).
    let lim_w = ((blk.mi_cols * 4) >> blk.sx) as i32;
    let lim_h = ((blk.mi_rows * 4) >> blk.sy) as i32;
    const PAD: usize = 2;
    const TS: usize = 8 + 2 * PAD;
    let mut tmp = [UNAVAILABLE; TS * TS];
    for ty in 0..h + 2 * PAD {
        let y = (y0 + ty) as i32 - PAD as i32;
        if y < 0 || y >= lim_h {
            continue;
        }
        for tx in 0..w + 2 * PAD {
            let x = (x0 + tx) as i32 - PAD as i32;
            if x >= 0 && x < lim_w {
                tmp[ty * TS + tx] = src.at(x as usize, y as usize) as i32;
            }
        }
    }
    let pri_taps = PRI_TAPS[((pri >> coeff_shift) & 1) as usize];
    let dirs = [dir as usize, (dir as usize + 2) & 7, (dir as usize + 6) & 7];
    for i in 0..h {
        for j in 0..w {
            let at = |dy: i32, dx: i32| tmp[((i + PAD) as i32 + dy) as usize * TS + ((j + PAD) as i32 + dx) as usize];
            let x = at(0, 0);
            let (mut sum, mut max, mut min) = (0, x, x);
            for k in 0..2 {
                for sign in [-1, 1] {
                    for (n, &d) in dirs.iter().enumerate() {
                        let (dy, dx) = DIRECTIONS[d][k];
                        let s = at(sign * dy, sign * dx);
                        if s == UNAVAILABLE {
                            continue;
                        }
                        sum += if n == 0 {
                            pri_taps[k] * constrain(s - x, pri, damping)
                        } else {
                            SEC_TAPS[k] * constrain(s - x, sec, damping)
                        };
                        max = max.max(s);
                        min = min.min(s);
                    }
                }
            }
            let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
            dst.data[(y0 + i) * dst.stride + x0 + j] = v.clamp(min, max) as u16;
        }
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

    struct Scene {
        frame: Frame,
        params: CdefParams,
        cdef_idx: Vec<i8>,
        cdef_stride: usize,
        skip: Vec<bool>,
        mi_cols: usize,
        mi_rows: usize,
    }

    /// A 128x80-luma frame of directional stripes, steps and noise with random strengths, filter-block indices and skips.
    fn scene(seed: u64, bit_depth: u8, sx: usize, sy: usize, mono: bool) -> Scene {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let (mi_cols, mi_rows) = (32, 20);
        let max = (1i32 << bit_depth) - 1;
        let mut planes = Vec::new();
        for pl in 0..if mono { 1 } else { 3 } {
            let (a, b) = if pl == 0 { (0, 0) } else { (sx, sy) };
            let mut p = Plane::new((mi_cols * 4) >> a, (mi_rows * 4) >> b);
            let extreme = rng.chance(15);
            let base = if extreme { 0 } else { rng.range(0, max) };
            let (ax, ay) = (rng.range(-6, 6), rng.range(-6, 6));
            let period = rng.range(3, 20);
            let amp = if extreme { max } else { rng.range(0, 60) << (bit_depth - 8) };
            let noise = [0, 1, 3, 8, 20][rng.below(5)] << (bit_depth - 8);
            for y in 0..p.height {
                for x in 0..p.width {
                    let t = (ax * x as i32 + ay * y as i32).rem_euclid(period * 2);
                    let v = base + if t < period { amp } else { 0 } + rng.range(-noise, noise);
                    p.data[y * p.stride + x] = v.clamp(0, max) as u16;
                }
            }
            planes.push(p);
        }
        let frame = Frame {
            planes,
            bit_depth,
            sub_x: sx as u8,
            sub_y: sy as u8,
            width: mi_cols * 4 - 3,
            height: mi_rows * 4 - 5,
        };
        let mut params = CdefParams { damping: rng.range(3, 6) as u8, bits: 3, ..Default::default() };
        for i in 0..8 {
            params.y_pri[i] = if rng.chance(25) { 0 } else { rng.range(0, 15) as u8 };
            params.y_sec[i] = [0, 1, 2, 4][rng.below(4)];
            params.uv_pri[i] = if rng.chance(25) { 0 } else { rng.range(0, 15) as u8 };
            params.uv_sec[i] = [0, 1, 2, 4][rng.below(4)];
        }
        let cdef_idx: Vec<i8> = (0..2 * 2).map(|_| if rng.chance(20) { -1 } else { rng.range(0, 7) as i8 }).collect();
        let skip_pct = rng.below(100);
        let skip = (0..mi_cols * mi_rows).map(|_| rng.chance(skip_pct)).collect();
        Scene { frame, params, cdef_idx, cdef_stride: 2, skip, mi_cols, mi_rows }
    }

    /// (seed, bit depth, sub_x, sub_y, monochrome, FNV-1a of the filtered planes from the reference).
    const GOLDEN: &[(u64, u8, usize, usize, bool, u64)] = &[
        (1, 8, 1, 1, false, 0x81e3cec6158f5c17),
        (2, 8, 0, 0, false, 0x94be028eb84cffe6),
        (3, 8, 1, 0, false, 0x4917d9d16b92209d),
        (4, 10, 1, 1, false, 0xd5ae03258e2d7e22),
        (5, 10, 0, 0, false, 0x8958e63432d41b72),
        (6, 12, 1, 0, false, 0x8b6371d88240341d),
        (7, 12, 1, 1, false, 0x6ffda8a65e330bfe),
        (8, 8, 1, 1, true, 0xb84a9280e758a3a6),
        (9, 10, 1, 1, true, 0x6bdcd9ce3a956633),
        (10, 8, 1, 1, false, 0x36d7c052a5d4b6a9),
        (11, 12, 0, 0, false, 0x91757a5aa33b919a),
        (12, 10, 1, 0, false, 0xb54a7855ac3f4b89),
    ];

    #[test]
    fn frames_match_reference_hashes() {
        for &(seed, bd, sx, sy, mono, hash) in GOLDEN {
            let s = scene(seed, bd, sx, sy, mono);
            let mut out = s.frame.clone();
            cdef_frame(&s.frame, &mut out, &s.params, &s.cdef_idx, s.cdef_stride, &s.skip, s.mi_cols, s.mi_rows);
            assert_ne!(fnv(&out), fnv(&s.frame), "scene {seed} should change the picture");
            assert_eq!(fnv(&out), hash, "scene {seed}");
        }
    }

    fn plane_from(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Plane {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                p.data[y * w + x] = f(x, y);
            }
        }
        p
    }

    fn mono(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Frame {
        Frame { planes: vec![plane_from(w, h, f)], bit_depth: 8, sub_x: 1, sub_y: 1, width: w, height: h }
    }

    fn strong() -> CdefParams {
        CdefParams { damping: 6, bits: 0, y_pri: [8; 8], y_sec: [2; 8], uv_pri: [8; 8], uv_sec: [2; 8] }
    }

    fn run(f: &Frame, p: &CdefParams, idx: &[i8], skip: &[bool]) -> Frame {
        let (mi_cols, mi_rows) = (f.planes[0].width / 4, f.planes[0].height / 4);
        let mut out = f.clone();
        cdef_frame(f, &mut out, p, idx, mi_cols.div_ceil(16), skip, mi_cols, mi_rows);
        out
    }

    #[test]
    fn directions_of_simple_patterns() {
        let horizontal = mono(8, 8, |_, y| if y % 2 == 0 { 40 } else { 200 });
        assert_eq!(find_direction(&horizontal.planes[0], 0, 0, 8).0, 2);
        let vertical = mono(8, 8, |x, _| if x % 2 == 0 { 40 } else { 200 });
        assert_eq!(find_direction(&vertical.planes[0], 0, 0, 8).0, 6);
        let anti = mono(8, 8, |x, y| if (x + y) % 4 == 0 { 200 } else { 40 });
        let (d, var) = find_direction(&anti.planes[0], 0, 0, 8);
        assert_eq!(d, 0);
        assert!(var > 0);
        let diag = mono(8, 8, |x, y| if (x + 8 - y) % 4 == 0 { 200 } else { 40 });
        assert_eq!(find_direction(&diag.planes[0], 0, 0, 8).0, 4);
        let flat = mono(8, 8, |_, _| 77);
        assert_eq!(find_direction(&flat.planes[0], 0, 0, 8), (0, 0));
        // High bit depths are scaled down to 8 bits first.
        let v10 = mono(8, 8, |x, _| if x % 2 == 0 { 160 } else { 800 });
        assert_eq!(find_direction(&v10.planes[0], 0, 0, 10), (6, find_direction(&vertical.planes[0], 0, 0, 8).1));
    }

    #[test]
    fn constrain_limits_large_differences() {
        assert_eq!(constrain(5, 0, 6), 0);
        assert_eq!(constrain(5, 8, 6), 5);
        assert_eq!(constrain(-5, 8, 6), -5);
        // damping 3, threshold 4 (log2 2): shift 1; 6 - (10 >> 1) = -1 -> 0
        assert_eq!(constrain(10, 4, 3), 0);
        assert_eq!(constrain(6, 4, 3), 1);
        assert_eq!(constrain(-6, 4, 3), -1);
        // never amplifies
        assert_eq!(constrain(2, 15, 6), 2);
    }

    #[test]
    fn flat_picture_and_no_strength_are_unchanged() {
        let flat = mono(16, 16, |_, _| 90);
        let out = run(&flat, &strong(), &[0], &[false; 16]);
        assert_eq!(out.planes[0].data, flat.planes[0].data);
        let noisy = mono(16, 16, |x, y| ((x * 7 + y * 13) % 31) as u16 + 50);
        let none = CdefParams { damping: 3, ..Default::default() };
        assert_eq!(run(&noisy, &none, &[0], &[false; 16]).planes[0].data, noisy.planes[0].data);
        assert_ne!(run(&noisy, &strong(), &[0], &[false; 16]).planes[0].data, noisy.planes[0].data);
    }

    #[test]
    fn skipped_blocks_and_unset_indices_are_copied() {
        let noisy = mono(16, 16, |x, y| ((x * 7 + y * 13) % 31) as u16 + 50);
        // No filter block index: nothing is filtered.
        assert_eq!(run(&noisy, &strong(), &[-1], &[false; 16]).planes[0].data, noisy.planes[0].data);
        // Skipping all four 4x4 units of the top-left 8x8 block leaves exactly that block alone.
        let mut skip = [false; 16];
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            skip[r * 4 + c] = true;
        }
        let out = run(&noisy, &strong(), &[0], &skip);
        let full = run(&noisy, &strong(), &[0], &[false; 16]);
        for y in 0..16 {
            for x in 0..16 {
                let want = if x < 8 && y < 8 { noisy.planes[0].at(x, y) } else { full.planes[0].at(x, y) };
                assert_eq!(out.planes[0].at(x, y), want, "({x},{y})");
            }
        }
        // One unskipped unit is enough to filter the block.
        skip[0] = false;
        assert_eq!(run(&noisy, &strong(), &[0], &skip).planes[0].data, full.planes[0].data);
    }

    #[test]
    fn filtering_reads_the_unfiltered_frame() {
        // Filtering block by block in place would differ at block borders; compare with a whole-frame reference run
        // that filters each 8x8 block independently from the original samples.
        let noisy = mono(16, 8, |x, y| ((x * 11 + y * 5) % 29) as u16 * 3 + 20);
        let both = run(&noisy, &strong(), &[0], &[false; 8]);
        for bx in [0usize, 8] {
            let left_only = {
                let mut skip = [true; 8];
                skip[bx / 4] = false;
                skip[bx / 4 + 1] = false;
                skip[4 + bx / 4] = false;
                skip[4 + bx / 4 + 1] = false;
                run(&noisy, &strong(), &[0], &skip)
            };
            for y in 0..8 {
                for x in bx..bx + 8 {
                    assert_eq!(both.planes[0].at(x, y), left_only.planes[0].at(x, y));
                }
            }
        }
    }

    #[test]
    fn chroma_uses_mapped_direction_and_reduced_damping() {
        // 4:2:2 maps the luma direction through a table; 4:2:0 and 4:4:4 use it directly. Vertical luma stripes
        // (direction 6) stay direction 6 in the table, so use a diagonal (direction 0 or 4) to see the mapping.
        let luma = |x: usize, y: usize| if (x + y).is_multiple_of(2) { 40u16 } else { 200 };
        let build = |sx: u8, sy: u8| {
            let cw = 16 >> sx;
            let ch = 16 >> sy;
            Frame {
                planes: vec![
                    plane_from(16, 16, luma),
                    plane_from(cw, ch, |x, y| ((x * 5 + y * 3) % 17) as u16 * 4 + 30),
                    plane_from(cw, ch, |x, y| ((x * 3 + y * 7) % 19) as u16 * 4 + 30),
                ],
                bit_depth: 8,
                sub_x: sx,
                sub_y: sy,
                width: 16,
                height: 16,
            }
        };
        let p = CdefParams { damping: 4, bits: 0, y_pri: [0; 8], y_sec: [0; 8], uv_pri: [6; 8], uv_sec: [0; 8] };
        for (sx, sy) in [(1, 1), (0, 0), (1, 0)] {
            let f = build(sx, sy);
            let mut out = f.clone();
            cdef_frame(&f, &mut out, &p, &[0], 1, &[false; 16], 4, 4);
            assert_eq!(out.planes[0].data, f.planes[0].data, "luma has no strength");
            assert_ne!(out.planes[1].data, f.planes[1].data, "chroma is filtered ({sx}{sy})");
        }
    }

    #[test]
    fn edges_of_the_frame_are_unavailable_but_the_padding_is_not() {
        // A step at the very left must not be smeared by data from outside: a constant frame stays constant even
        // with a strong filter, which would fail if outside samples were treated as 0.
        let f = mono(8, 8, |_, _| 200);
        assert_eq!(run(&f, &strong(), &[0], &[false; 4]).planes[0].data, f.planes[0].data);
        // Samples between the visible size and the coded size take part like any other.
        let mut g = mono(8, 8, |x, _| if x < 4 { 40 } else { 220 });
        g.width = 5;
        let a = run(&g, &strong(), &[0], &[false; 4]);
        g.width = 8;
        let b = run(&g, &strong(), &[0], &[false; 4]);
        assert_eq!(a.planes[0].data, b.planes[0].data);
    }

    #[test]
    fn output_buffers_of_any_shape_are_accepted() {
        let noisy = mono(16, 16, |x, y| ((x * 7 + y * 13) % 31) as u16 + 50);
        let mut small = mono(4, 4, |_, _| 0);
        cdef_frame(&noisy, &mut small, &strong(), &[0], 1, &[false; 16], 4, 4);
        assert_eq!(small.planes[0].width, 16);
        assert_eq!(small.planes[0].data, run(&noisy, &strong(), &[0], &[false; 16]).planes[0].data);
    }
}
