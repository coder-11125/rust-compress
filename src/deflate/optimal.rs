//! Optimal parsing (Zopfli-style): shortest path over all matches under a cost model that is
//! refined from the statistics of the previous pass.

use super::encode::{LEN_CODE, dist_code, estimate_bits};
use super::lz77::{MAX_MATCH, Matcher, Token};
use super::{DIST_EXTRA, LEN_EXTRA};

struct Costs {
    lit: [f32; 256],
    len: [f32; MAX_MATCH + 1],
    dist: [f32; 30],
}

/// Symbol frequencies (literal/length, distance) of a token list.
#[derive(Clone)]
struct Freqs {
    ll: [f32; 286],
    d: [f32; 30],
}

fn freqs_of(tokens: &[Token]) -> Freqs {
    let mut f = Freqs { ll: [0.0; 286], d: [0.0; 30] };
    for &t in tokens {
        if t.is_match() {
            f.ll[257 + LEN_CODE[t.length()] as usize] += 1.0;
            f.d[dist_code(t.dist())] += 1.0;
        } else {
            f.ll[t.lit() as usize] += 1.0;
        }
    }
    f.ll[256] = 1.0;
    f
}

fn costs_from(f: &Freqs) -> Costs {
    let ll_total = f.ll.iter().sum::<f32>();
    let d_total = f.d.iter().sum::<f32>().max(1.0);
    let sym = |freq: f32, total: f32| if freq > 0.0 { (total / freq).log2() } else { total.log2() + 2.0 };
    let mut c = Costs { lit: [0.0; 256], len: [0.0; MAX_MATCH + 1], dist: [0.0; 30] };
    for b in 0..256 {
        c.lit[b] = sym(f.ll[b], ll_total);
    }
    for l in 3..=MAX_MATCH {
        let lc = LEN_CODE[l] as usize;
        c.len[l] = sym(f.ll[257 + lc], ll_total) + LEN_EXTRA[lc] as f32;
    }
    for ds in 0..30 {
        c.dist[ds] = sym(f.d[ds], d_total) + DIST_EXTRA[ds] as f32;
    }
    c
}

/// Cost model for a first pass that knows nothing about the parse: literals cost what the chunk's own
/// byte histogram says, and a match costs roughly an average length code plus a distance code with its
/// extra bits. Starting from a parse that already uses many literals avoids the slow drift the
/// statistics-driven refinement shows on noisy data, where a greedy parse seeds it with thousands of
/// coincidental short matches that make literals look expensive.
fn prior_costs(d: &[u8]) -> Costs {
    let mut hist = [0u32; 256];
    for &b in d {
        hist[b as usize] += 1;
    }
    let n = d.len().max(1) as f32;
    let mut c = Costs { lit: [0.0; 256], len: [0.0; MAX_MATCH + 1], dist: [0.0; 30] };
    for b in 0..256 {
        c.lit[b] = if hist[b] > 0 { (n / hist[b] as f32).log2() } else { n.log2() + 2.0 };
    }
    for l in 3..=MAX_MATCH {
        c.len[l] = 5.0 + LEN_EXTRA[LEN_CODE[l] as usize] as f32;
    }
    for ds in 0..30 {
        c.dist[ds] = 5.0 + DIST_EXTRA[ds] as f32;
    }
    c
}

/// Lengths up to this are always tried; above it only bucket edges and the last `TAIL` lengths.
const DENSE: usize = 32;
const TAIL: usize = 4;

/// Lengths above DENSE that start or end a length-code bucket.
fn length_edges() -> &'static [u16] {
    static EDGES: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    EDGES.get_or_init(|| {
        (DENSE + 1..=MAX_MATCH)
            .filter(|&l| LEN_CODE[l] != LEN_CODE[l - 1] || (l < MAX_MATCH && LEN_CODE[l] != LEN_CODE[l + 1]))
            .map(|l| l as u16)
            .collect()
    })
}

/// Reusable DP buffers.
#[derive(Default)]
struct Scratch {
    cost: Vec<f32>,
    /// Back-pointer per position: `length << 16 | distance` (distance 0 = literal).
    step: Vec<u32>,
}

/// Cheapest tokenization of `d[start..end]` under `costs`, given every position's Pareto matches.
fn shortest_path(
    d: &[u8],
    start: usize,
    end: usize,
    arena: &[(u16, u16)],
    offs: &[u32],
    costs: &Costs,
    sc: &mut Scratch,
) -> Vec<Token> {
    let n = end - start;
    sc.cost.clear();
    sc.cost.resize(n + 1, f32::INFINITY);
    sc.step.clear();
    sc.step.resize(n + 1, 0);
    let (cost, step) = (&mut sc.cost, &mut sc.step);
    let edges = length_edges();
    cost[0] = 0.0;
    for i in 0..n {
        let ci = cost[i];
        let c = ci + costs.lit[d[start + i] as usize];
        if c < cost[i + 1] {
            cost[i + 1] = c;
            step[i + 1] = 1 << 16;
        }
        let matches = &arena[offs[i] as usize..offs[i + 1] as usize];
        // Inside long repetitions every position has a maximum-length match; trying all shorter
        // lengths from each of them costs O(258 n) for almost no gain, so only take the full match.
        let long_run = matches.last().is_some_and(|m| m.0 as usize == MAX_MATCH);
        let mut prev_end = if long_run { MAX_MATCH - 1 } else { 2 };
        for &(le, dist) in matches {
            if long_run && (le as usize) < MAX_MATCH {
                continue;
            }
            let (lo, hi) = (prev_end + 1, le as usize);
            prev_end = hi;
            if lo > hi {
                continue;
            }
            let base = ci + costs.dist[dist_code(dist as usize)];
            // Dense for short lengths; above DENSE only the longest few and the edges of
            // length-code buckets (inside a bucket every length costs the same).
            let dense_hi = hi.min(DENSE);
            if lo <= dense_hi {
                let lens = &costs.len[lo..=dense_hi];
                let cs = &mut cost[i + lo..=i + dense_hi];
                let ss = &mut step[i + lo..=i + dense_hi];
                for (k, ((cc, st), lc)) in cs.iter_mut().zip(ss.iter_mut()).zip(lens).enumerate() {
                    let c = base + lc;
                    if c < *cc {
                        *cc = c;
                        *st = (((lo + k) as u32) << 16) | dist as u32;
                    }
                }
            }
            // Sparse part: bucket edges in (DENSE, hi - TAIL], then the last TAIL lengths.
            let from = lo.max(DENSE + 1);
            if from <= hi {
                let tail_start = from.max(hi.saturating_sub(TAIL - 1));
                let first = edges.partition_point(|&e| (e as usize) < from);
                for &e in &edges[first..] {
                    let l = e as usize;
                    if l >= tail_start {
                        break;
                    }
                    let c = base + costs.len[l];
                    if c < cost[i + l] {
                        cost[i + l] = c;
                        step[i + l] = ((l as u32) << 16) | dist as u32;
                    }
                }
                for l in tail_start..=hi {
                    let c = base + costs.len[l];
                    if c < cost[i + l] {
                        cost[i + l] = c;
                        step[i + l] = ((l as u32) << 16) | dist as u32;
                    }
                }
            }
        }
    }
    let mut tokens = Vec::new();
    let mut i = n;
    while i > 0 {
        let st = step[i];
        let l = (st >> 16) as usize;
        if l == 1 && st & 0xFFFF == 0 {
            tokens.push(Token::literal(d[start + i - 1]));
            i -= 1;
        } else {
            tokens.push(Token::matched(l, (st & 0xFFFF) as usize));
            i -= l;
        }
    }
    tokens.reverse();
    tokens
}

/// Greedy parse over the collected matches; used to seed the cost model.
fn greedy(d: &[u8], start: usize, end: usize, arena: &[(u16, u16)], offs: &[u32]) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let n = end - start;
    while i < n {
        match arena[offs[i] as usize..offs[i + 1] as usize].last() {
            Some(&(l, dist)) => {
                tokens.push(Token::matched(l as usize, dist as usize));
                i += l as usize;
            }
            None => {
                tokens.push(Token::literal(d[start + i]));
                i += 1;
            }
        }
    }
    tokens
}

/// Tokenize `d[start..end]` with optimal parsing. The matcher's hash state advances past `end`.
pub(super) fn optimal_tokens(m: &mut Matcher, d: &[u8], start: usize, end: usize, iterations: u32) -> Vec<Token> {
    let mut arena: Vec<(u16, u16)> = Vec::new();
    let mut offs: Vec<u32> = Vec::with_capacity(end - start + 1);
    for p in start..end {
        offs.push(arena.len() as u32);
        m.collect_matches(d, p, end, &mut arena);
    }
    offs.push(arena.len() as u32);
    // Two starting points: the longest-match parse, and a parse priced from the data's own byte
    // histogram. Which one leads to the better fixed point depends on the data (the first suits image
    // rows full of real repeats, the second noisy ones), so both are refined and the cheaper kept.
    let (a, b) = rayon::join(
        || {
            let seed = greedy(d, start, end, &arena, &offs);
            refine(d, start, end, &arena, &offs, seed, iterations, &mut Scratch::default())
        },
        || {
            let mut scratch = Scratch::default();
            let seed = shortest_path(d, start, end, &arena, &offs, &prior_costs(&d[start..end]), &mut scratch);
            refine(d, start, end, &arena, &offs, seed, iterations, &mut scratch)
        },
    );
    if b.0 < a.0 { b.1 } else { a.1 }
}

/// Repeatedly re-price the matches by the statistics of the previous parse and re-run the shortest
/// path, starting from `seed`; stops after two passes without improvement. Returns (bits, tokens).
#[allow(clippy::too_many_arguments)]
fn refine(
    d: &[u8],
    start: usize,
    end: usize,
    arena: &[(u16, u16)],
    offs: &[u32],
    seed: Vec<Token>,
    iterations: u32,
    scratch: &mut Scratch,
) -> (u64, Vec<Token>) {
    let mut best_bits = estimate_bits(&seed);
    let mut best = seed;
    let mut freqs = freqs_of(&best);
    let mut stale = 0;
    for _ in 0..iterations {
        let costs = costs_from(&freqs);
        let current = shortest_path(d, start, end, arena, offs, &costs, scratch);
        let bits = estimate_bits(&current);
        freqs = freqs_of(&current);
        if bits < best_bits {
            best_bits = bits;
            best = current;
            stale = 0;
        } else {
            stale += 1;
            if stale >= 2 {
                break;
            }
        }
    }
    (best_bits, best)
}
