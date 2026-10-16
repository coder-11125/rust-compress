//! Intra prediction. Predictors work in a small workspace laid out like the neighbourhood of a
//! block: row 0 holds the pixels above (column 0 is the top-left corner, then the block's columns,
//! then up to four above-right pixels), and column 0 of every following row holds the pixel to the
//! left. `stride` is the workspace row length.

use super::tables::*;

/// Row length of the workspaces (large enough for 1 + 16 + 4 columns).
pub const WS: usize = 32;
/// Index of the first pixel of a luma macroblock inside its workspace.
pub const ORIGIN: usize = WS + 1;

#[inline(always)]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

#[inline(always)]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

/// DC prediction of a `size` x `size` block at `at`; which edges exist decides the divisor.
pub fn dc(ws: &mut [u8], at: usize, size: usize, have_top: bool, have_left: bool) {
    let shift = if size == 16 { 3 } else { 2 };
    let mut sum = 0u32;
    if have_top {
        sum += (0..size).map(|x| ws[at - WS + x] as u32).sum::<u32>();
    }
    if have_left {
        sum += (0..size).map(|y| ws[at + y * WS - 1] as u32).sum::<u32>();
    }
    let v = match (have_top, have_left) {
        (true, true) => (sum + size as u32) >> (shift + 2),
        (false, false) => 128,
        _ => (sum + (size as u32 >> 1)) >> (shift + 1),
    } as u8;
    for y in 0..size {
        ws[at + y * WS..at + y * WS + size].fill(v);
    }
}

pub fn vertical(ws: &mut [u8], at: usize, size: usize) {
    for y in 0..size {
        for x in 0..size {
            ws[at + y * WS + x] = ws[at - WS + x];
        }
    }
}

pub fn horizontal(ws: &mut [u8], at: usize, size: usize) {
    for y in 0..size {
        let v = ws[at + y * WS - 1];
        ws[at + y * WS..at + y * WS + size].fill(v);
    }
}

pub fn true_motion(ws: &mut [u8], at: usize, size: usize) {
    let tl = ws[at - WS - 1] as i32;
    for y in 0..size {
        let left = ws[at + y * WS - 1] as i32;
        for x in 0..size {
            ws[at + y * WS + x] = (left + ws[at - WS + x] as i32 - tl).clamp(0, 255) as u8;
        }
    }
}

/// Predict a whole 16x16 luma or 8x8 chroma block with one of DC/V/H/TM.
pub fn predict_block(ws: &mut [u8], at: usize, size: usize, mode: u8, have_top: bool, have_left: bool) {
    match mode {
        DC_PRED => dc(ws, at, size, have_top, have_left),
        V_PRED => vertical(ws, at, size),
        H_PRED => horizontal(ws, at, size),
        _ => true_motion(ws, at, size),
    }
}

/// Predict one 4x4 sub-block at `at` with one of the ten B_* modes.
pub fn predict_4x4(ws: &mut [u8], at: usize, mode: u8) {
    let top = |i: isize| ws[(at as isize - WS as isize + i) as usize];
    let left = |i: usize| ws[at + i * WS - 1];
    let (x, a, b, c, d, e, f, g, h) = (top(-1), top(0), top(1), top(2), top(3), top(4), top(5), top(6), top(7));
    let (i, j, k, l) = (left(0), left(1), left(2), left(3));
    let mut px = [[0u8; 4]; 4]; // px[y][x]
    match mode {
        B_DC_PRED => {
            let sum: u32 = [a, b, c, d, i, j, k, l].iter().map(|&v| v as u32).sum();
            px = [[((sum + 4) >> 3) as u8; 4]; 4];
        }
        B_TM_PRED => {
            for (y, row) in px.iter_mut().enumerate() {
                for (xx, v) in row.iter_mut().enumerate() {
                    *v = (left(y) as i32 + top(xx as isize) as i32 - x as i32).clamp(0, 255) as u8;
                }
            }
        }
        B_VE_PRED => {
            let v = [avg3(x, a, b), avg3(a, b, c), avg3(b, c, d), avg3(c, d, e)];
            px = [v; 4];
        }
        B_HE_PRED => {
            let rows = [avg3(x, i, j), avg3(i, j, k), avg3(j, k, l), avg3(k, l, l)];
            for (y, row) in px.iter_mut().enumerate() {
                *row = [rows[y]; 4];
            }
        }
        B_LD_PRED => {
            px[0][0] = avg3(a, b, c);
            px[0][1] = avg3(b, c, d);
            px[1][0] = px[0][1];
            px[0][2] = avg3(c, d, e);
            px[1][1] = px[0][2];
            px[2][0] = px[0][2];
            px[0][3] = avg3(d, e, f);
            px[1][2] = px[0][3];
            px[2][1] = px[0][3];
            px[3][0] = px[0][3];
            px[1][3] = avg3(e, f, g);
            px[2][2] = px[1][3];
            px[3][1] = px[1][3];
            px[2][3] = avg3(f, g, h);
            px[3][2] = px[2][3];
            px[3][3] = avg3(g, h, h);
        }
        B_RD_PRED => {
            px[3][0] = avg3(j, k, l);
            px[3][1] = avg3(i, j, k);
            px[2][0] = px[3][1];
            px[3][2] = avg3(x, i, j);
            px[2][1] = px[3][2];
            px[1][0] = px[3][2];
            px[3][3] = avg3(a, x, i);
            px[2][2] = px[3][3];
            px[1][1] = px[3][3];
            px[0][0] = px[3][3];
            px[2][3] = avg3(b, a, x);
            px[1][2] = px[2][3];
            px[0][1] = px[2][3];
            px[1][3] = avg3(c, b, a);
            px[0][2] = px[1][3];
            px[0][3] = avg3(d, c, b);
        }
        B_VR_PRED => {
            px[0][0] = avg2(x, a);
            px[2][1] = px[0][0];
            px[0][1] = avg2(a, b);
            px[2][2] = px[0][1];
            px[0][2] = avg2(b, c);
            px[2][3] = px[0][2];
            px[0][3] = avg2(c, d);
            px[3][0] = avg3(k, j, i);
            px[2][0] = avg3(j, i, x);
            px[1][0] = avg3(i, x, a);
            px[3][1] = px[1][0];
            px[1][1] = avg3(x, a, b);
            px[3][2] = px[1][1];
            px[1][2] = avg3(a, b, c);
            px[3][3] = px[1][2];
            px[1][3] = avg3(b, c, d);
        }
        B_VL_PRED => {
            px[0][0] = avg2(a, b);
            px[0][1] = avg2(b, c);
            px[2][0] = px[0][1];
            px[0][2] = avg2(c, d);
            px[2][1] = px[0][2];
            px[0][3] = avg2(d, e);
            px[2][2] = px[0][3];
            px[1][0] = avg3(a, b, c);
            px[1][1] = avg3(b, c, d);
            px[3][0] = px[1][1];
            px[1][2] = avg3(c, d, e);
            px[3][1] = px[1][2];
            px[1][3] = avg3(d, e, f);
            px[3][2] = px[1][3];
            px[2][3] = avg3(e, f, g);
            px[3][3] = avg3(f, g, h);
        }
        B_HD_PRED => {
            px[0][0] = avg2(i, x);
            px[1][2] = px[0][0];
            px[1][0] = avg2(j, i);
            px[2][2] = px[1][0];
            px[2][0] = avg2(k, j);
            px[3][2] = px[2][0];
            px[3][0] = avg2(l, k);
            px[0][3] = avg3(a, b, c);
            px[0][2] = avg3(x, a, b);
            px[0][1] = avg3(i, x, a);
            px[1][3] = px[0][1];
            px[1][1] = avg3(j, i, x);
            px[2][3] = px[1][1];
            px[2][1] = avg3(k, j, i);
            px[3][3] = px[2][1];
            px[3][1] = avg3(l, k, j);
        }
        _ => {
            // B_HU_PRED
            px[0][0] = avg2(i, j);
            px[0][2] = avg2(j, k);
            px[1][0] = px[0][2];
            px[1][2] = avg2(k, l);
            px[2][0] = px[1][2];
            px[0][1] = avg3(i, j, k);
            px[0][3] = avg3(j, k, l);
            px[1][1] = px[0][3];
            px[1][3] = avg3(k, l, l);
            px[2][1] = px[1][3];
            for (yy, xx) in [(2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)] {
                px[yy][xx] = l;
            }
        }
    }
    for (y, row) in px.iter().enumerate() {
        ws[at + y * WS..at + y * WS + 4].copy_from_slice(row);
    }
}

/// Fill the border of a workspace for the macroblock at (mbx, mby): 127 above the frame, 129 left
/// of it, otherwise pixels already reconstructed (unfiltered) in the plane.
#[allow(clippy::too_many_arguments)]
pub fn fill_borders(
    ws: &mut [u8],
    plane: &[u8],
    stride: usize,
    mbx: usize,
    mby: usize,
    mbw: usize,
    size: usize,
    above_right: bool,
) {
    let x0 = mbx * size;
    let y0 = mby * size;
    // Row 0: top-left corner, the pixels above, and (luma) the pixels above-right.
    let cols = 1 + size + if above_right { 4 } else { 0 };
    if mby == 0 {
        ws[..cols].fill(127);
    } else {
        let row = (y0 - 1) * stride;
        ws[0] = if mbx == 0 { 129 } else { plane[row + x0 - 1] };
        ws[1..=size].copy_from_slice(&plane[row + x0..row + x0 + size]);
        if above_right {
            if mbx + 1 < mbw {
                ws[size + 1..size + 5].copy_from_slice(&plane[row + x0 + size..row + x0 + size + 4]);
            } else {
                let last = plane[row + x0 + size - 1];
                ws[size + 1..size + 5].fill(last);
            }
        }
    }
    for r in 0..size {
        ws[(r + 1) * WS] = if mbx == 0 { 129 } else { plane[(y0 + r) * stride + x0 - 1] };
    }
}

pub fn store_block(ws: &[u8], plane: &mut [u8], stride: usize, mbx: usize, mby: usize, size: usize) {
    for r in 0..size {
        let dst = (mby * size + r) * stride + mbx * size;
        plane[dst..dst + size].copy_from_slice(&ws[ORIGIN + r * WS..ORIGIN + r * WS + size]);
    }
}
