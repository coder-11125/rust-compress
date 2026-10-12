//! PNG scanline filters (forward direction) and filter-selection strategies.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Every row uses the given filter type (0..=4).
    Fixed(u8),
    /// Per row, pick the filter minimizing the sum of absolute filtered values.
    MinSum,
    /// Per row, pick the filter minimizing the Shannon entropy of the filtered bytes.
    Entropy,
}

#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (a as i32, b as i32, c as i32);
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Apply filter `kind` to one row. `prev` must be the same length as `cur` (zeros for the first row).
pub fn filter_row(kind: u8, cur: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) {
    let n = cur.len();
    match kind {
        0 => out.copy_from_slice(cur),
        1 => {
            let head = bpp.min(n);
            out[..head].copy_from_slice(&cur[..head]);
            for i in bpp..n {
                out[i] = cur[i].wrapping_sub(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..n {
                out[i] = cur[i].wrapping_sub(prev[i]);
            }
        }
        3 => {
            for i in 0..bpp.min(n) {
                out[i] = cur[i].wrapping_sub(prev[i] >> 1);
            }
            for i in bpp..n {
                out[i] = cur[i].wrapping_sub(((cur[i - bpp] as u16 + prev[i] as u16) >> 1) as u8);
            }
        }
        _ => {
            for i in 0..bpp.min(n) {
                out[i] = cur[i].wrapping_sub(prev[i]);
            }
            for i in bpp..n {
                out[i] = cur[i].wrapping_sub(paeth(cur[i - bpp], prev[i], prev[i - bpp]));
            }
        }
    }
}

fn entropy_cost(row: &[u8]) -> f32 {
    let mut hist = [0u32; 256];
    for &b in row {
        hist[b as usize] += 1;
    }
    let n = row.len() as f32;
    let ln = n.log2();
    hist.iter().filter(|&&c| c > 0).map(|&c| c as f32 * (ln - (c as f32).log2())).sum()
}

#[inline]
fn sum_abs(row: &[u8]) -> u64 {
    row.iter().map(|&b| (b as i8).unsigned_abs() as u64).sum()
}

/// Filter a whole image. `rows` holds `height` packed scanlines of `row_bytes` each (no filter bytes).
/// The result has `height * (row_bytes + 1)` bytes: filter byte then filtered row.
pub fn filter_image(rows: &[u8], row_bytes: usize, bpp: usize, strategy: Strategy) -> Vec<u8> {
    let stride = row_bytes + 1;
    let height = rows.len() / row_bytes;
    let mut out = vec![0u8; height * stride];
    let zero = vec![0u8; row_bytes];
    // Batches of rows are independent (they only read the unfiltered input), so run them in parallel.
    const BATCH: usize = 64;
    out.par_chunks_mut(stride * BATCH).enumerate().for_each(|(b, chunk)| {
        let mut scratch = vec![0u8; row_bytes];
        let mut best = vec![0u8; row_bytes];
        for (j, dst) in chunk.chunks_exact_mut(stride).enumerate() {
            let y = b * BATCH + j;
            let cur = &rows[y * row_bytes..(y + 1) * row_bytes];
            let prev = if y == 0 { &zero[..] } else { &rows[(y - 1) * row_bytes..y * row_bytes] };
            match strategy {
                Strategy::Fixed(k) => {
                    dst[0] = k;
                    filter_row(k, cur, prev, bpp, &mut dst[1..]);
                }
                Strategy::MinSum | Strategy::Entropy => {
                    let mut best_kind = 0u8;
                    let mut best_score = f32::MAX;
                    for k in 0..5u8 {
                        filter_row(k, cur, prev, bpp, &mut scratch);
                        let s = if strategy == Strategy::MinSum {
                            sum_abs(&scratch) as f32
                        } else {
                            entropy_cost(&scratch)
                        };
                        if s < best_score {
                            best_score = s;
                            best_kind = k;
                            std::mem::swap(&mut best, &mut scratch);
                        }
                    }
                    dst[0] = best_kind;
                    dst[1..].copy_from_slice(&best);
                }
            }
        }
    });
    out
}
