//! PNG scanline filters (forward direction) and filter-selection strategies.

use rayon::prelude::*;
use rc_deflate::deflate_compress;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Every row uses the given filter type (0..=4).
    Fixed(u8),
    /// Per row, pick the filter minimizing the sum of absolute filtered values.
    MinSum,
    /// Per row, pick the filter minimizing the Shannon entropy of the filtered bytes.
    Entropy,
    /// Per row, pick the filter producing the fewest distinct byte pairs (a cheap proxy for how well
    /// LZ77 will match the row; much better than the statistical heuristics on graphics).
    Bigrams,
    /// Per row, pick the filter whose output actually compresses smallest when appended to the
    /// preceding filtered rows. Slow; meant for the highest effort levels.
    Brute,
}

/// Number of preceding filtered rows `Strategy::Brute` compresses together with each candidate.
const BRUTE_LINES: usize = 4;
/// Rows per independently processed batch (batches only read the unfiltered input).
const BATCH: usize = 64;

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

/// Number of distinct (byte, next byte) pairs in `row`. `seen` is a 65536-bit scratch set.
fn distinct_bigrams(row: &[u8], seen: &mut [u64; 1024]) -> u32 {
    seen.fill(0);
    let mut count = 0;
    for w in row.windows(2) {
        let i = (w[0] as usize) << 8 | w[1] as usize;
        let bit = 1u64 << (i & 63);
        let word = &mut seen[i >> 6];
        count += (*word & bit == 0) as u32;
        *word |= bit;
    }
    count
}

/// Reusable per-thread buffers for choosing row filters.
struct RowChooser {
    scratch: Vec<u8>,
    best: Vec<u8>,
    seen: Box<[u64; 1024]>,
}

impl RowChooser {
    fn new(row_bytes: usize) -> Self {
        Self { scratch: vec![0; row_bytes], best: vec![0; row_bytes], seen: Box::new([0; 1024]) }
    }

    /// Filter `cur` with every filter type, keep the one `score` likes best; its output is left in
    /// `self.best` and its type returned.
    fn pick(
        &mut self,
        cur: &[u8],
        prev: &[u8],
        bpp: usize,
        mut score: impl FnMut(&[u8], &mut [u64; 1024]) -> f64,
    ) -> u8 {
        let mut best_kind = 0u8;
        let mut best_score = f64::MAX;
        for k in 0..5u8 {
            filter_row(k, cur, prev, bpp, &mut self.scratch);
            let s = score(&self.scratch, &mut self.seen);
            if s < best_score {
                best_score = s;
                best_kind = k;
                std::mem::swap(&mut self.best, &mut self.scratch);
            }
        }
        best_kind
    }

    /// One row with a statistical strategy. The result is in `self.best`.
    fn choose(&mut self, strategy: Strategy, cur: &[u8], prev: &[u8], bpp: usize) -> u8 {
        match strategy {
            Strategy::Fixed(k) => {
                filter_row(k, cur, prev, bpp, &mut self.best);
                k
            }
            Strategy::Entropy => self.pick(cur, prev, bpp, |r, _| entropy_cost(r) as f64),
            Strategy::Bigrams => self.pick(cur, prev, bpp, |r, seen| distinct_bigrams(r, seen) as f64),
            Strategy::MinSum | Strategy::Brute => self.pick(cur, prev, bpp, |r, _| sum_abs(r) as f64),
        }
    }
}

/// Filter one batch of rows with `Strategy::Brute`: every candidate filter for a row is compressed
/// together with the previously chosen rows and the smallest wins. `first` is the index of the
/// batch's first row; rows just before it (not yet filtered by this batch) are approximated with
/// MinSum so the history is not empty.
fn brute_batch(
    rows: &[u8],
    row_bytes: usize,
    bpp: usize,
    first: usize,
    chunk: &mut [u8],
    zero: &[u8],
    ch: &mut RowChooser,
) {
    let stride = row_bytes + 1;
    // Window of the filter-byte-prefixed rows chosen so far, newest last.
    let keep_rows = BRUTE_LINES.min((32768 / stride).max(1));
    let mut hist: Vec<u8> = Vec::with_capacity((keep_rows * 2 + 2) * stride);
    for y in first.saturating_sub(keep_rows)..first {
        let cur = &rows[y * row_bytes..(y + 1) * row_bytes];
        let prev = if y == 0 { zero } else { &rows[(y - 1) * row_bytes..y * row_bytes] };
        let k = ch.choose(Strategy::MinSum, cur, prev, bpp);
        hist.push(k);
        hist.extend_from_slice(&ch.best);
    }
    let mut trial: Vec<u8> = Vec::with_capacity((keep_rows + 1) * stride);
    for (j, dst) in chunk.chunks_exact_mut(stride).enumerate() {
        let y = first + j;
        let cur = &rows[y * row_bytes..(y + 1) * row_bytes];
        let prev = if y == 0 { zero } else { &rows[(y - 1) * row_bytes..y * row_bytes] };
        let tail = hist.len().saturating_sub(keep_rows * stride);
        let mut best = (usize::MAX, 0u8);
        for k in 0..5u8 {
            filter_row(k, cur, prev, bpp, &mut ch.scratch);
            trial.clear();
            trial.extend_from_slice(&hist[tail..]);
            trial.push(k);
            trial.extend_from_slice(&ch.scratch);
            let size = deflate_compress(&trial, 1).len();
            if size < best.0 {
                best = (size, k);
            }
        }
        filter_row(best.1, cur, prev, bpp, &mut ch.scratch);
        dst[0] = best.1;
        dst[1..].copy_from_slice(&ch.scratch);
        hist.push(best.1);
        hist.extend_from_slice(&ch.scratch);
        if hist.len() > (keep_rows * 2 + 1) * stride {
            hist.drain(..hist.len() - keep_rows * stride);
        }
    }
}

/// Filter a whole image. `rows` holds `height` packed scanlines of `row_bytes` each (no filter bytes).
/// The result has `height * (row_bytes + 1)` bytes: filter byte then filtered row.
pub fn filter_image(rows: &[u8], row_bytes: usize, bpp: usize, strategy: Strategy) -> Vec<u8> {
    let stride = row_bytes + 1;
    let height = rows.len() / row_bytes;
    let mut out = vec![0u8; height * stride];
    let zero = vec![0u8; row_bytes];
    // Batches of rows are independent (they only read the unfiltered input), so run them in parallel.
    out.par_chunks_mut(stride * BATCH).enumerate().for_each(|(b, chunk)| {
        let mut ch = RowChooser::new(row_bytes);
        if strategy == Strategy::Brute {
            brute_batch(rows, row_bytes, bpp, b * BATCH, chunk, &zero, &mut ch);
            return;
        }
        for (j, dst) in chunk.chunks_exact_mut(stride).enumerate() {
            let y = b * BATCH + j;
            let cur = &rows[y * row_bytes..(y + 1) * row_bytes];
            let prev = if y == 0 { &zero[..] } else { &rows[(y - 1) * row_bytes..y * row_bytes] };
            dst[0] = ch.choose(strategy, cur, prev, bpp);
            dst[1..].copy_from_slice(&ch.best);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Undo the filters (the decoder's job) to check every strategy is lossless.
    fn unfilter(data: &[u8], row_bytes: usize, bpp: usize) -> Vec<u8> {
        let stride = row_bytes + 1;
        let mut out: Vec<u8> = Vec::new();
        for (y, line) in data.chunks_exact(stride).enumerate() {
            let prev_start = y.saturating_sub(1) * row_bytes;
            let mut cur = vec![0u8; row_bytes];
            for i in 0..row_bytes {
                let a = if i >= bpp { cur[i - bpp] } else { 0 };
                let b = if y > 0 { out[prev_start + i] } else { 0 };
                let c = if y > 0 && i >= bpp { out[prev_start + i - bpp] } else { 0 };
                let x = line[1 + i];
                cur[i] = match line[0] {
                    0 => x,
                    1 => x.wrapping_add(a),
                    2 => x.wrapping_add(b),
                    3 => x.wrapping_add(((a as u16 + b as u16) >> 1) as u8),
                    _ => x.wrapping_add(paeth(a, b, c)),
                };
            }
            out.extend_from_slice(&cur);
        }
        out
    }

    fn image(w: usize, h: usize, bpp: usize) -> Vec<u8> {
        let mut s = 7u32;
        (0..w * h * bpp)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                // Smooth gradient with some noise and flat patches.
                let (x, y) = ((i / bpp) % w, (i / bpp) / w);
                if (x / 20 + y / 20) % 3 == 0 { 40 } else { ((x + y) as u32 + (s >> 29)) as u8 }
            })
            .collect()
    }

    #[test]
    fn every_strategy_roundtrips() {
        for (w, h, bpp) in [(1, 1, 1), (37, 150, 3), (64, 70, 4), (200, 9, 6)] {
            let img = image(w, h, bpp);
            let row_bytes = w * bpp;
            for s in [
                Strategy::Fixed(0),
                Strategy::Fixed(4),
                Strategy::MinSum,
                Strategy::Entropy,
                Strategy::Bigrams,
                Strategy::Brute,
            ] {
                let f = filter_image(&img, row_bytes, bpp, s);
                assert_eq!(f.len(), h * (row_bytes + 1));
                assert_eq!(unfilter(&f, row_bytes, bpp), img, "{s:?} {w}x{h}x{bpp}");
            }
        }
    }

    #[test]
    fn brute_is_no_worse_than_fixed_filters_it_can_choose() {
        let (w, h, bpp) = (120, 130, 3);
        let img = image(w, h, bpp);
        let size = |s| deflate_compress(&filter_image(&img, w * bpp, bpp, s), 6).len();
        let brute = size(Strategy::Brute);
        for k in 0..5 {
            assert!(brute <= size(Strategy::Fixed(k)) + size(Strategy::Fixed(k)) / 50, "filter {k}");
        }
    }

    #[test]
    fn bigram_count_counts_distinct_pairs() {
        let mut seen = Box::new([0u64; 1024]);
        assert_eq!(distinct_bigrams(&[1, 2, 1, 2, 1, 2], &mut seen), 2);
        assert_eq!(distinct_bigrams(&[5, 5, 5, 5], &mut seen), 1);
        assert_eq!(distinct_bigrams(&[9], &mut seen), 0);
    }
}
