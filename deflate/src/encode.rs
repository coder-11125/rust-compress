use super::lz77::{Matcher, Params, Token, WINDOW};
use super::{CL_ORDER, DIST_BASE, DIST_EXTRA, LEN_BASE, LEN_EXTRA};
use rayon::prelude::*;
use rc_core::util::bitio::LsbWriter;
use rc_core::util::checksum::adler32;
use rc_core::util::huffman::{build_lengths, lsb_codes};

const BLOCK_TOKENS: usize = 1 << 14;

const fn build_len_code() -> [u8; 259] {
    let mut t = [0u8; 259];
    let mut code = 0;
    while code < 29 {
        let base = LEN_BASE[code] as usize;
        let mut l = base;
        while l < base + (1 << LEN_EXTRA[code]) && l <= 258 {
            t[l] = code as u8;
            l += 1;
        }
        code += 1;
    }
    t
}

const fn build_dist_codes() -> ([u8; 256], [u8; 256]) {
    let (mut lo, mut hi) = ([0u8; 256], [0u8; 256]);
    let mut code = 0;
    while code < 30 {
        let base = DIST_BASE[code] as usize;
        let mut d = base;
        while d < base + (1 << DIST_EXTRA[code]) {
            if d <= 256 {
                lo[d - 1] = code as u8;
            } else {
                hi[(d - 1) >> 7] = code as u8;
            }
            d += 1;
        }
        code += 1;
    }
    (lo, hi)
}

pub(super) static LEN_CODE: [u8; 259] = build_len_code();
static DIST_CODES: ([u8; 256], [u8; 256]) = build_dist_codes();

#[inline(always)]
pub(super) fn dist_code(d: usize) -> usize {
    if d <= 256 { DIST_CODES.0[d - 1] as usize } else { DIST_CODES.1[(d - 1) >> 7] as usize }
}

fn fixed_lengths() -> ([u8; 288], [u8; 30]) {
    let mut ll = [8u8; 288];
    ll[144..256].fill(9);
    ll[256..280].fill(7);
    (ll, [5u8; 30])
}

/// Run-length encode code lengths into code-length-alphabet (symbol, extra bits value) pairs.
fn rle_lengths(lens: &[u8]) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lens.len() {
        let v = lens[i];
        let mut run = 1;
        while i + run < lens.len() && lens[i + run] == v {
            run += 1;
        }
        i += run;
        if v == 0 {
            while run >= 11 {
                let n = run.min(138);
                out.push((18, (n - 11) as u8));
                run -= n;
            }
            if run >= 3 {
                out.push((17, (run - 3) as u8));
                run = 0;
            }
        } else {
            out.push((v, 0));
            run -= 1;
            while run >= 3 {
                let n = run.min(6);
                out.push((16, (n - 3) as u8));
                run -= n;
            }
        }
        for _ in 0..run {
            out.push((v, 0));
        }
    }
    out
}

struct Dynamic {
    ll_len: Vec<u8>,
    d_len: Vec<u8>,
    hlit: usize,
    hdist: usize,
    cl_len: Vec<u8>,
    hclen: usize,
    rle: Vec<(u8, u8)>,
    header_bits: u64,
}

fn plan_dynamic(ll_freq: &[u32; 288], d_freq: &[u32; 30]) -> Dynamic {
    let ll_len = build_lengths(ll_freq, 15);
    let mut d_freq = *d_freq;
    if d_freq.iter().all(|&f| f == 0) {
        d_freq[0] = 1;
    }
    let d_len = build_lengths(&d_freq, 15);
    let hlit = (ll_len.iter().rposition(|&l| l > 0).unwrap() + 1).max(257);
    let hdist = d_len.iter().rposition(|&l| l > 0).unwrap() + 1;
    let mut all = ll_len[..hlit].to_vec();
    all.extend_from_slice(&d_len[..hdist]);
    let rle = rle_lengths(&all);
    let mut cl_freq = [0u32; 19];
    for &(s, _) in &rle {
        cl_freq[s as usize] += 1;
    }
    let cl_len = build_lengths(&cl_freq, 7);
    let hclen = CL_ORDER.iter().rposition(|&i| cl_len[i] > 0).map_or(4, |p| p + 1).max(4);
    let mut bits = 5 + 5 + 4 + 3 * hclen as u64;
    for &(s, _) in &rle {
        bits += cl_len[s as usize] as u64
            + match s {
                16 => 2,
                17 => 3,
                18 => 7,
                _ => 0,
            };
    }
    Dynamic { ll_len, d_len, hlit, hdist, cl_len, hclen, rle, header_bits: bits }
}

fn write_tokens(w: &mut LsbWriter, tokens: &[Token], ll_len: &[u8], ll_code: &[u16], d_len: &[u8], d_code: &[u16]) {
    for &t in tokens {
        if t.is_match() {
            let len = t.length();
            let lc = LEN_CODE[len] as usize;
            let sym = 257 + lc;
            w.put(ll_code[sym] as u32, ll_len[sym] as u32);
            let eb = LEN_EXTRA[lc] as u32;
            if eb > 0 {
                w.put((len - LEN_BASE[lc] as usize) as u32, eb);
            }
            let dist = t.dist();
            let dc = dist_code(dist);
            w.put(d_code[dc] as u32, d_len[dc] as u32);
            let eb = DIST_EXTRA[dc] as u32;
            if eb > 0 {
                w.put((dist - DIST_BASE[dc] as usize) as u32, eb);
            }
        } else {
            let s = t.lit() as usize;
            w.put(ll_code[s] as u32, ll_len[s] as u32);
        }
    }
    w.put(ll_code[256] as u32, ll_len[256] as u32);
}

fn write_stored(w: &mut LsbWriter, raw: &[u8], last: bool) {
    let mut chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    if chunks.is_empty() {
        chunks.push(&[]);
    }
    let n = chunks.len();
    for (i, c) in chunks.into_iter().enumerate() {
        w.put((last && i + 1 == n) as u32, 1);
        w.put(0, 2);
        w.align();
        let len = c.len() as u16;
        w.put_bytes(&len.to_le_bytes());
        w.put_bytes(&(!len).to_le_bytes());
        w.put_bytes(c);
    }
}

/// Size in bits of `tokens` coded with their own optimal dynamic Huffman tables.
pub(super) fn estimate_bits(tokens: &[Token]) -> u64 {
    let mut ll_freq = [0u32; 288];
    let mut d_freq = [0u32; 30];
    let mut bits = 3u64;
    for &t in tokens {
        if t.is_match() {
            let lc = LEN_CODE[t.length()] as usize;
            ll_freq[257 + lc] += 1;
            let dc = dist_code(t.dist());
            d_freq[dc] += 1;
            bits += LEN_EXTRA[lc] as u64 + DIST_EXTRA[dc] as u64;
        } else {
            ll_freq[t.lit() as usize] += 1;
        }
    }
    ll_freq[256] = 1;
    let dynamic = plan_dynamic(&ll_freq, &d_freq);
    bits += dynamic.header_bits;
    for (f, l) in ll_freq.iter().zip(&dynamic.ll_len) {
        bits += *f as u64 * *l as u64;
    }
    for (f, l) in d_freq.iter().zip(&dynamic.d_len) {
        bits += *f as u64 * *l as u64;
    }
    bits
}

fn write_block(w: &mut LsbWriter, tokens: &[Token], raw: &[u8], last: bool) {
    let mut ll_freq = [0u32; 288];
    let mut d_freq = [0u32; 30];
    let mut extra_bits = 0u64;
    for &t in tokens {
        if t.is_match() {
            let lc = LEN_CODE[t.length()] as usize;
            ll_freq[257 + lc] += 1;
            let dc = dist_code(t.dist());
            d_freq[dc] += 1;
            extra_bits += LEN_EXTRA[lc] as u64 + DIST_EXTRA[dc] as u64;
        } else {
            ll_freq[t.lit() as usize] += 1;
        }
    }
    ll_freq[256] = 1;

    let dynamic = plan_dynamic(&ll_freq, &d_freq);
    let mut dyn_bits = 3 + dynamic.header_bits + extra_bits;
    for (f, l) in ll_freq.iter().zip(&dynamic.ll_len) {
        dyn_bits += *f as u64 * *l as u64;
    }
    for (f, l) in d_freq.iter().zip(&dynamic.d_len) {
        dyn_bits += *f as u64 * *l as u64;
    }
    let (fl, fd) = fixed_lengths();
    let mut fix_bits = 3 + extra_bits;
    for (f, l) in ll_freq.iter().zip(fl.iter()) {
        fix_bits += *f as u64 * *l as u64;
    }
    for (f, l) in d_freq.iter().zip(fd.iter()) {
        fix_bits += *f as u64 * *l as u64;
    }
    let stored_bits = 8 * (raw.len() as u64 + 5 * raw.len().div_ceil(65535).max(1) as u64);

    if stored_bits <= dyn_bits.min(fix_bits) {
        write_stored(w, raw, last);
    } else if fix_bits <= dyn_bits {
        w.put(last as u32, 1);
        w.put(1, 2);
        write_tokens(w, tokens, &fl, &lsb_codes(&fl), &fd, &lsb_codes(&fd));
    } else {
        w.put(last as u32, 1);
        w.put(2, 2);
        w.put((dynamic.hlit - 257) as u32, 5);
        w.put((dynamic.hdist - 1) as u32, 5);
        w.put((dynamic.hclen - 4) as u32, 4);
        for &i in CL_ORDER.iter().take(dynamic.hclen) {
            w.put(dynamic.cl_len[i] as u32, 3);
        }
        let cl_code = lsb_codes(&dynamic.cl_len);
        for &(s, x) in &dynamic.rle {
            w.put(cl_code[s as usize] as u32, dynamic.cl_len[s as usize] as u32);
            match s {
                16 => w.put(x as u32, 2),
                17 => w.put(x as u32, 3),
                18 => w.put(x as u32, 7),
                _ => {}
            }
        }
        write_tokens(
            w,
            tokens,
            &dynamic.ll_len,
            &lsb_codes(&dynamic.ll_len),
            &dynamic.d_len,
            &lsb_codes(&dynamic.d_len),
        );
    }
}

/// Input is split into segments of this size, compressed independently (in parallel) but with the
/// preceding 32 KiB available as match history. Fixed, so output never depends on the thread count.
const SEGMENT: usize = 1 << 19;
/// Level 9 parses (and emits one block for) chunks of this many input bytes at a time.
const OPT_CHUNK: usize = 1 << 17;
/// Maximum recursion depth when splitting an optimal-parse chunk into several blocks.
const OPT_SPLIT_DEPTH: u32 = 3;

/// Emit `tokens` as one block, or split it where two blocks with their own Huffman tables are
/// cheaper (tried at the quarter points, recursively up to `depth` levels).
fn emit_split(w: &mut LsbWriter, tokens: &[Token], raw: &[u8], last: bool, depth: u32) {
    if depth > 0 && tokens.len() >= 4096 {
        let whole = estimate_bits(tokens);
        let mut best: Option<(u64, usize)> = None;
        for q in 1..=3 {
            let mid = tokens.len() * q / 4;
            let cost = estimate_bits(&tokens[..mid]) + estimate_bits(&tokens[mid..]);
            if cost < whole && best.is_none_or(|b| cost < b.0) {
                best = Some((cost, mid));
            }
        }
        if let Some((_, mid)) = best {
            let left_bytes: usize = tokens[..mid].iter().map(|t| if t.is_match() { t.length() } else { 1 }).sum();
            emit_split(w, &tokens[..mid], &raw[..left_bytes], false, depth - 1);
            emit_split(w, &tokens[mid..], &raw[left_bytes..], last, depth - 1);
            return;
        }
    }
    write_block(w, tokens, raw, last);
}

fn compress_range(data: &[u8], start: usize, end: usize, level: u32, is_last: bool) -> Vec<u8> {
    let d = &data[..end];
    let mut w = LsbWriter::with_capacity((end - start) / 2 + 64);
    let matcher_params = Params::for_level(level);
    let mut matcher = Matcher::new(matcher_params);
    matcher.prime(d, start.saturating_sub(WINDOW), start);
    let mut tokens = Vec::with_capacity(BLOCK_TOKENS);
    let mut pos = start;
    while pos < end {
        let block_start = pos;
        tokens.clear();
        if matcher_params.optimal {
            let chunk_end = (pos + OPT_CHUNK).min(end);
            tokens = super::optimal::optimal_tokens(&mut matcher, d, pos, chunk_end, matcher_params.iters);
            pos = chunk_end;
        } else {
            matcher.tokenize(d, &mut pos, &mut tokens, BLOCK_TOKENS);
        }
        let last_block = is_last && pos >= end;
        if matcher_params.optimal {
            emit_split(&mut w, &tokens, &d[block_start..pos], last_block, OPT_SPLIT_DEPTH);
        } else {
            write_block(&mut w, &tokens, &d[block_start..pos], last_block);
        }
    }
    if !is_last {
        // Empty stored block: byte-aligns the stream so segments can simply be concatenated.
        w.put(0, 3);
        w.align();
        w.put_bytes(&[0, 0, 0xFF, 0xFF]);
    }
    w.finish()
}

/// Compress to a raw DEFLATE stream. Levels: 0 = stored, 1 (fastest) .. 9 (smallest).
pub fn deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
    if level == 0 || data.is_empty() {
        let mut w = LsbWriter::with_capacity(data.len() + 16);
        write_stored(&mut w, data, true);
        return w.finish();
    }
    let nseg = data.len().div_ceil(SEGMENT);
    if nseg == 1 {
        return compress_range(data, 0, data.len(), level, true);
    }
    let parts: Vec<Vec<u8>> = (0..nseg)
        .into_par_iter()
        .map(|i| compress_range(data, i * SEGMENT, ((i + 1) * SEGMENT).min(data.len()), level, i + 1 == nseg))
        .collect();
    parts.concat()
}

pub fn zlib_compress(data: &[u8], level: u32) -> Vec<u8> {
    let flevel = match level {
        0 | 1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let cmf = 0x78u16;
    let mut flg = flevel << 6;
    flg += 31 - ((cmf << 8) + flg) % 31;
    let body = deflate_compress(data, level);
    let mut out = Vec::with_capacity(body.len() + 6);
    out.push(cmf as u8);
    out.push(flg as u8);
    out.extend_from_slice(&body);
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflate::{inflate, zlib_decompress};

    fn sample(n: usize, seed: u32) -> Vec<u8> {
        // Mixture of runs, repeats and noise.
        let mut s = seed;
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            match (s >> 28) & 3 {
                0 => v.extend(std::iter::repeat_n((s >> 8) as u8, (s >> 20) as usize % 300)),
                1 if v.len() > 100 => {
                    let st = (s >> 8) as usize % (v.len() - 50);
                    let l = 3 + (s >> 3) as usize % 100;
                    for k in 0..l {
                        let b = v[st + k % 50];
                        v.push(b);
                    }
                }
                _ => v.extend((0..(s >> 20) % 64).map(|i| (s >> (i % 24)) as u8)),
            }
        }
        v.truncate(n);
        v
    }

    #[test]
    fn roundtrip_all_levels() {
        for (n, seed) in [(0, 1), (1, 2), (5, 3), (1000, 4), (70_000, 5), (400_000, 6)] {
            let data = sample(n, seed);
            for level in 0..=9 {
                if level == 9 && n > 100_000 {
                    continue; // optimal parsing is slow in debug builds; covered by smaller inputs
                }
                let c = deflate_compress(&data, level);
                let d = inflate(&c, n, usize::MAX).unwrap();
                assert_eq!(d, data, "n={n} level={level}");
            }
        }
    }

    #[test]
    fn multi_segment_stream_roundtrips() {
        // > 2 segments, with matches that reach back across segment boundaries.
        let mut data = sample(400_000, 21);
        let copy = data.clone();
        data.extend_from_slice(&copy);
        data.extend_from_slice(&copy);
        data.extend_from_slice(&copy[..100_000]);
        assert!(data.len() > 2 * SEGMENT);
        for level in [1, 6, 8] {
            let c = deflate_compress(&data, level);
            assert_eq!(inflate(&c, 0, usize::MAX).unwrap(), data, "level {level}");
        }
    }

    #[test]
    fn zlib_roundtrip() {
        let data = sample(50_000, 9);
        let c = zlib_compress(&data, 6);
        assert_eq!(zlib_decompress(&c, 0, usize::MAX).unwrap(), data);
    }

    #[test]
    fn higher_level_not_larger_on_repetitive_data() {
        let data = sample(150_000, 11);
        let fast = deflate_compress(&data, 1).len();
        let best = deflate_compress(&data, 9).len();
        assert!(best <= fast, "{best} > {fast}");
        assert!(best < data.len());
    }

    #[test]
    fn skewed_noise_compresses_near_its_entropy() {
        // Independent symbols from a skewed distribution: nothing to match, so the best parse is all
        // literals. A parse seeded from coincidental short matches used to land ~10% above this.
        let weights = [60u32, 20, 10, 5, 3, 1, 1];
        let total: u32 = weights.iter().sum();
        let mut s = 777u32;
        let data: Vec<u8> = (0..200_000)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let mut r = (s >> 8) % total;
                weights
                    .iter()
                    .position(|&w| {
                        if r < w {
                            true
                        } else {
                            r -= w;
                            false
                        }
                    })
                    .unwrap() as u8
            })
            .collect();
        let entropy_bytes: f64 = weights
            .iter()
            .map(|&w| {
                let p = w as f64 / total as f64;
                -p * p.log2()
            })
            .sum::<f64>()
            * data.len() as f64
            / 8.0;
        for level in [7, 8, 9] {
            let c = deflate_compress(&data, level);
            assert_eq!(inflate(&c, 0, usize::MAX).unwrap(), data);
            assert!(
                (c.len() as f64) < entropy_bytes * 1.08,
                "level {level}: {} vs entropy {entropy_bytes:.0}",
                c.len()
            );
        }
    }

    #[test]
    fn incompressible_data_falls_back_to_stored() {
        let mut s = 12345u32;
        let data: Vec<u8> = (0..100_000)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let c = deflate_compress(&data, 6);
        assert!(c.len() <= data.len() + data.len() / 1000 + 64, "{} vs {}", c.len(), data.len());
        assert_eq!(inflate(&c, 0, usize::MAX).unwrap(), data);
    }

    #[test]
    fn all_same_byte_and_long_runs() {
        let data = vec![7u8; 1_000_000];
        let c = deflate_compress(&data, 6);
        assert!(c.len() < 3000);
        assert_eq!(inflate(&c, 0, usize::MAX).unwrap(), data);
    }

    #[test]
    fn inflate_rejects_garbage_without_panicking() {
        let data = sample(20_000, 3);
        let c = deflate_compress(&data, 6);
        for cut in [0, 1, 2, c.len() / 2, c.len() - 1] {
            let _ = inflate(&c[..cut], 0, 1 << 24);
        }
        let mut s = 1u32;
        for _ in 0..200 {
            let mut m = c.clone();
            for _ in 0..4 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let i = (s >> 8) as usize % m.len();
                m[i] ^= (s >> 3) as u8 | 1;
            }
            let _ = inflate(&m, 0, 1 << 24);
        }
    }
}
