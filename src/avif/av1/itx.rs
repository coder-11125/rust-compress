//! AV1 inverse and forward transforms (specification section 7.13), bit-exact on the inverse side.
//!
//! The 1-D DCT (4 to 64 points) and ADST (8 and 16 points) are expressed the way the specification does: a
//! permutation, then a list of butterflies. `B(a, b, angle, flip)` is a rotation by `angle / 128 * pi` with
//! 12-bit cosines and `H(a, b)` is the add/subtract pair (clamped on the decode side). The lists are
//! generated at compile time by `const fn`s that follow the specification's step loops, so there are no
//! hand-copied 600-line kernels, and the same list drives the forward transform: the transpose of the
//! inverse network is the forward network (rotations transpose to the opposite rotation, `H` is symmetric).
//!
//! Layout. `coefs` and the residual are row-major with stride `TX_WIDTH[tx_size]`. A transform type names the
//! vertical (column) transform first and the horizontal (row) transform second, as in the specification.

// ---------------------------------------------------------------------------------------------------------
// Public numbering
// ---------------------------------------------------------------------------------------------------------

pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;
pub const TX_64X64: u8 = 4;
pub const TX_4X8: u8 = 5;
pub const TX_8X4: u8 = 6;
pub const TX_8X16: u8 = 7;
pub const TX_16X8: u8 = 8;
pub const TX_16X32: u8 = 9;
pub const TX_32X16: u8 = 10;
pub const TX_32X64: u8 = 11;
pub const TX_64X32: u8 = 12;
pub const TX_4X16: u8 = 13;
pub const TX_16X4: u8 = 14;
pub const TX_8X32: u8 = 15;
pub const TX_32X8: u8 = 16;
pub const TX_16X64: u8 = 17;
pub const TX_64X16: u8 = 18;

pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;
pub const FLIPADST_DCT: u8 = 4;
pub const DCT_FLIPADST: u8 = 5;
pub const FLIPADST_FLIPADST: u8 = 6;
pub const ADST_FLIPADST: u8 = 7;
pub const FLIPADST_ADST: u8 = 8;
pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;

pub const TX_WIDTH_LOG2: [usize; 19] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
pub const TX_HEIGHT_LOG2: [usize; 19] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
pub const TX_WIDTH: [usize; 19] = pow2_table(&TX_WIDTH_LOG2);
pub const TX_HEIGHT: [usize; 19] = pow2_table(&TX_HEIGHT_LOG2);

const fn pow2_table(logs: &[usize; 19]) -> [usize; 19] {
    let mut out = [0; 19];
    let mut i = 0;
    while i < 19 {
        out[i] = 1 << logs[i];
        i += 1;
    }
    out
}

/// Right shift applied between the row and the column pass (specification `Transform_Row_Shift`).
const ROW_SHIFT: [u32; 19] = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

/// Shifts of the encoder's forward transform: `[before the column pass, after it, after the row pass]`.
/// Positive values shift left, negative ones round-shift right. The net shift of each size balances the
/// gain of `inverse_transform_add`.
const FWD_SHIFT: [[i32; 3]; 19] = [
    [2, 0, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -4, 0],
    [0, -2, -2],
    [2, 0, 0],
    [2, 0, 0],
    [2, -1, 0],
    [2, -1, 0],
    [2, -3, 0],
    [2, -3, 0],
    [0, -1, -2],
    [2, -3, -2],
    [2, -1, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -2, 0],
    [0, -2, 0],
    [2, -4, 0],
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// Vertical (column) transform of each transform type.
const COL_KIND: [Kind; 16] = [
    Kind::Dct,
    Kind::Adst,
    Kind::Dct,
    Kind::Adst,
    Kind::FlipAdst,
    Kind::Dct,
    Kind::FlipAdst,
    Kind::Adst,
    Kind::FlipAdst,
    Kind::Identity,
    Kind::Dct,
    Kind::Identity,
    Kind::Adst,
    Kind::Identity,
    Kind::FlipAdst,
    Kind::Identity,
];

/// Horizontal (row) transform of each transform type.
const ROW_KIND: [Kind; 16] = [
    Kind::Dct,
    Kind::Dct,
    Kind::Adst,
    Kind::Adst,
    Kind::Dct,
    Kind::FlipAdst,
    Kind::FlipAdst,
    Kind::FlipAdst,
    Kind::Adst,
    Kind::Identity,
    Kind::Identity,
    Kind::Dct,
    Kind::Identity,
    Kind::Adst,
    Kind::Identity,
    Kind::FlipAdst,
];

// ---------------------------------------------------------------------------------------------------------
// Butterfly networks
// ---------------------------------------------------------------------------------------------------------

/// `cos(i * pi / 128)` in 12-bit fixed point for `i` in 0..=64.
const COS128_TABLE: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703,
    3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896, 2824, 2751, 2675, 2598, 2520,
    2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700,
    601, 501, 401, 301, 201, 101, 0,
];

const fn cos128(angle: i32) -> i32 {
    let a = (angle & 255) as usize;
    if a <= 64 {
        COS128_TABLE[a]
    } else if a <= 128 {
        -COS128_TABLE[128 - a]
    } else if a <= 192 {
        -COS128_TABLE[a - 128]
    } else {
        COS128_TABLE[256 - a]
    }
}

const fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

const fn brev(bits: u32, x: usize) -> usize {
    let mut r = 0;
    let mut i = 0;
    while i < bits {
        r |= ((x >> i) & 1) << (bits - 1 - i);
        i += 1;
    }
    r
}

/// One step of a butterfly network, acting on elements `a` and `b` of the working array.
#[derive(Clone, Copy)]
struct Op {
    a: u8,
    b: u8,
    /// `true`: `(x, y) -> (x + y, x - y)` (clamped when decoding). `false`: the 2x2 matrix `m` followed by a
    /// rounding shift of 12 bits.
    add: bool,
    m: [i32; 4],
}

impl Op {
    const NOP: Op = Op { a: 0, b: 0, add: true, m: [0; 4] };
}

/// Collects ops; with `K == 0` it only counts them, which is how the final array length is found.
struct Builder<const K: usize> {
    ops: [Op; K],
    len: usize,
}

impl<const K: usize> Builder<K> {
    const fn new() -> Self {
        Builder { ops: [Op::NOP; K], len: 0 }
    }

    const fn push(&mut self, op: Op) {
        if self.len < K {
            self.ops[self.len] = op;
        }
        self.len += 1;
    }

    /// The specification's `B(a, b, angle, flip)`.
    const fn rot(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let (c, s) = (cos128(angle), sin128(angle));
        // x = T[a] c - T[b] s, y = T[a] s + T[b] c; without flip T[a] = x and T[b] = y, with flip they swap.
        let m = if flip { [s, c, c, -s] } else { [c, -s, s, c] };
        self.push(Op { a: a as u8, b: b as u8, add: false, m });
    }

    /// The specification's `H(a, b, flip)`: with `flip`, `H(b, a, 0)`.
    const fn add(&mut self, a: usize, b: usize, flip: bool) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        self.push(Op { a: a as u8, b: b as u8, add: true, m: [0; 4] });
    }
}

/// Specification 7.13.2.3: the inverse DCT of `2^n` points after the bit-reversal permutation.
const fn dct_build<const K: usize>(n: u32) -> Builder<K> {
    let mut g = Builder::<K>::new();
    let mut i;
    let mut j;
    if n == 6 {
        i = 0;
        while i < 16 {
            g.rot(32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 8 {
            g.rot(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i32) << 3), false);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 16 {
            g.add(32 + i * 2, 33 + i * 2, i & 1 == 1);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 4 {
            g.rot(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i32) << 4), false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 8 {
            g.add(16 + 2 * i, 17 + 2 * i, i & 1 == 1);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 2 {
                g.rot(62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * brev(2, i) as i32 + 64 * j as i32, true);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 3 {
        i = 0;
        while i < 2 {
            g.rot(4 + i, 7 - i, 56 - 32 * i as i32, false);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 4 {
            g.add(8 + 2 * i, 9 + 2 * i, i & 1 == 1);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 2 {
                g.rot(30 - 4 * i - j, 17 + 4 * i + j, 24 + ((j as i32) << 6) + ((1 - i as i32) << 5), true);
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            j = 0;
            while j < 2 {
                g.add(32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1);
                j += 1;
            }
            i += 1;
        }
    }
    i = 0;
    while i < 2 {
        g.rot(2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
        i += 1;
    }
    if n >= 3 {
        i = 0;
        while i < 2 {
            g.add(4 + 2 * i, 5 + 2 * i, i == 1);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            g.rot(14 - i, 9 + i, 48 + 64 * i as i32, true);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 2 {
                g.add(16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1);
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 4 {
                g.rot(61 - i * 8 - j, 34 + i * 8 + j, 56 - 32 * i as i32 + ((j >> 1) as i32) * 64, true);
                j += 1;
            }
            i += 1;
        }
    }
    i = 0;
    while i < 2 {
        g.add(i, 3 - i, false);
        i += 1;
    }
    if n >= 3 {
        g.rot(6, 5, 32, true);
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 2 {
                g.add(8 + 4 * i + j, 11 + 4 * i - j, i == 1);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            g.rot(29 - i, 18 + i, 48 + (((i >> 1) as i32) << 6), true);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 4 {
                g.add(32 + i * 8 + j, 39 + i * 8 - j, i & 1 == 1);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 3 {
        i = 0;
        while i < 4 {
            g.add(i, 7 - i, false);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            g.rot(13 - i, 10 + i, 32, true);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 4 {
                g.add(16 + i * 8 + j, 23 + i * 8 - j, i == 1);
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            g.rot(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 8 {
            g.add(i, 15 - i, false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            g.rot(27 - i, 20 + i, 32, true);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 8 {
                g.add(32 + i * 16 + j, 47 + i * 16 - j, i == 1);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 16 {
            g.add(i, 31 - i, false);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            g.rot(55 - i, 40 + i, 32, true);
            i += 1;
        }
        i = 0;
        while i < 32 {
            g.add(i, 63 - i, false);
            i += 1;
        }
    }
    g
}

/// Specification 7.13.2.6: the inverse ADST of 8 points (after its input permutation).
const fn adst8_build<const K: usize>() -> Builder<K> {
    let mut g = Builder::<K>::new();
    let mut i = 0;
    while i < 4 {
        g.rot(2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
        i += 1;
    }
    i = 0;
    while i < 4 {
        g.add(i, 4 + i, false);
        i += 1;
    }
    g.rot(4, 5, 48, true);
    g.rot(7, 6, 16, true);
    g.add(0, 2, false);
    g.add(1, 3, false);
    g.add(4, 6, false);
    g.add(5, 7, false);
    g.rot(2, 3, 32, true);
    g.rot(6, 7, 32, true);
    g
}

/// Specification 7.13.2.7: the inverse ADST of 16 points (after its input permutation).
const fn adst16_build<const K: usize>() -> Builder<K> {
    let mut g = Builder::<K>::new();
    let mut i = 0;
    while i < 8 {
        g.rot(2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
        i += 1;
    }
    i = 0;
    while i < 8 {
        g.add(i, 8 + i, false);
        i += 1;
    }
    g.rot(8, 9, 56, true);
    g.rot(10, 11, 24, true);
    g.rot(13, 12, 8, true);
    g.rot(15, 14, 40, true);
    i = 0;
    while i < 4 {
        g.add(i, 4 + i, false);
        g.add(8 + i, 12 + i, false);
        i += 1;
    }
    g.rot(4, 5, 48, true);
    g.rot(7, 6, 16, true);
    g.rot(12, 13, 48, true);
    g.rot(15, 14, 16, true);
    i = 0;
    while i < 16 {
        g.add(i, i + 2, false);
        g.add(i + 1, i + 3, false);
        i += 4;
    }
    i = 0;
    while i < 16 {
        g.rot(i + 2, i + 3, 32, true);
        i += 4;
    }
    g
}

const fn dct_input_perm<const N: usize>(bits: u32) -> [u8; N] {
    let mut p = [0; N];
    let mut i = 0;
    while i < N {
        p[i] = brev(bits, i) as u8;
        i += 1;
    }
    p
}

const fn identity_perm<const N: usize>() -> [u8; N] {
    let mut p = [0; N];
    let mut i = 0;
    while i < N {
        p[i] = i as u8;
        i += 1;
    }
    p
}

const fn adst_input_perm<const N: usize>() -> [u8; N] {
    let mut p = [0; N];
    let mut i = 0;
    while i < N {
        p[i] = if i & 1 == 1 { i - 1 } else { N - i - 1 } as u8;
        i += 1;
    }
    p
}

/// A complete 1-D transform: gather the input through `inp`, run `ops`, scatter through `out`.
struct Net {
    /// `work[i] = input[inp[i]]`.
    inp: &'static [u8],
    /// `output[j] = work[out[j]]`, negated for odd `j` when `neg_odd`.
    out: &'static [u8],
    neg_odd: bool,
    ops: &'static [Op],
}

macro_rules! dct_net {
    ($name:ident, $bits:expr, $n:expr) => {
        static $name: Net = {
            const OPS: [Op; dct_build::<0>($bits).len] = dct_build($bits).ops;
            const IN: [u8; $n] = dct_input_perm($bits);
            const OUT: [u8; $n] = identity_perm();
            Net { inp: &IN, out: &OUT, neg_odd: false, ops: &OPS }
        };
    };
}

dct_net!(DCT4, 2, 4);
dct_net!(DCT8, 3, 8);
dct_net!(DCT16, 4, 16);
dct_net!(DCT32, 5, 32);
dct_net!(DCT64, 6, 64);

static ADST8: Net = {
    const OPS: [Op; adst8_build::<0>().len] = adst8_build().ops;
    const IN: [u8; 8] = adst_input_perm();
    const OUT: [u8; 8] = [0, 4, 6, 2, 3, 7, 5, 1];
    Net { inp: &IN, out: &OUT, neg_odd: true, ops: &OPS }
};

static ADST16: Net = {
    const OPS: [Op; adst16_build::<0>().len] = adst16_build().ops;
    const IN: [u8; 16] = adst_input_perm();
    const OUT: [u8; 16] = [0, 8, 12, 4, 6, 14, 10, 2, 3, 11, 15, 7, 5, 13, 9, 1];
    Net { inp: &IN, out: &OUT, neg_odd: true, ops: &OPS }
};

fn dct_net(log2: usize) -> &'static Net {
    match log2 {
        2 => &DCT4,
        3 => &DCT8,
        4 => &DCT16,
        5 => &DCT32,
        _ => &DCT64,
    }
}

fn adst_net(log2: usize) -> &'static Net {
    if log2 == 3 { &ADST8 } else { &ADST16 }
}

#[inline(always)]
fn round12(v: i32) -> i32 {
    v.wrapping_add(2048) >> 12
}

#[inline(always)]
fn round_shift(v: i32, bits: u32) -> i32 {
    if bits == 0 { v } else { v.wrapping_add(1 << (bits - 1)) >> bits }
}

#[inline(always)]
fn mul_add(m0: i32, x: i32, m1: i32, y: i32) -> i32 {
    m0.wrapping_mul(x).wrapping_add(m1.wrapping_mul(y))
}

/// Runs the inverse network on `x` (n values), writing n values to `out`. Sums are clamped to `lo..=hi`.
fn run_inverse(net: &Net, x: &[i32], out: &mut [i32], lo: i32, hi: i32) {
    let n = net.inp.len();
    let mut t = [0i32; 64];
    for i in 0..n {
        t[i] = x[net.inp[i] as usize];
    }
    for op in net.ops {
        let (a, b) = (op.a as usize, op.b as usize);
        let (p, q) = (t[a], t[b]);
        if op.add {
            t[a] = p.wrapping_add(q).clamp(lo, hi);
            t[b] = p.wrapping_sub(q).clamp(lo, hi);
        } else {
            t[a] = round12(mul_add(op.m[0], p, op.m[1], q));
            t[b] = round12(mul_add(op.m[2], p, op.m[3], q));
        }
    }
    for j in 0..n {
        let v = t[net.out[j] as usize];
        out[j] = if net.neg_odd && j & 1 == 1 { v.wrapping_neg() } else { v };
    }
}

/// Runs the transposed network: the forward transform matching `run_inverse`, without clamping.
fn run_forward(net: &Net, x: &[i32], out: &mut [i32]) {
    let n = net.inp.len();
    let mut t = [0i32; 64];
    for j in 0..n {
        let v = x[j];
        t[net.out[j] as usize] = if net.neg_odd && j & 1 == 1 { v.wrapping_neg() } else { v };
    }
    for op in net.ops.iter().rev() {
        let (a, b) = (op.a as usize, op.b as usize);
        let (p, q) = (t[a], t[b]);
        if op.add {
            t[a] = p.wrapping_add(q);
            t[b] = p.wrapping_sub(q);
        } else {
            t[a] = round12(mul_add(op.m[0], p, op.m[2], q));
            t[b] = round12(mul_add(op.m[1], p, op.m[3], q));
        }
    }
    for i in 0..n {
        out[net.inp[i] as usize] = t[i];
    }
}

const SINPI_1_9: i32 = 1321;
const SINPI_2_9: i32 = 2482;
const SINPI_3_9: i32 = 3344;
const SINPI_4_9: i32 = 3803;

/// Specification 7.13.2.5: inverse ADST4.
fn inverse_adst4(x: &[i32], out: &mut [i32]) {
    let (x0, x1, x2, x3) = (x[0], x[1], x[2], x[3]);
    let s0 = SINPI_1_9.wrapping_mul(x0);
    let s1 = SINPI_2_9.wrapping_mul(x0);
    let s2 = SINPI_3_9.wrapping_mul(x1);
    let s3 = SINPI_4_9.wrapping_mul(x2);
    let s4 = SINPI_1_9.wrapping_mul(x2);
    let s5 = SINPI_2_9.wrapping_mul(x3);
    let s6 = SINPI_4_9.wrapping_mul(x3);
    let s7 = x0.wrapping_sub(x2).wrapping_add(x3);
    let a0 = s0.wrapping_add(s3).wrapping_add(s5);
    let a1 = s1.wrapping_sub(s4).wrapping_sub(s6);
    let a3 = s2;
    let a2 = SINPI_3_9.wrapping_mul(s7);
    out[0] = round12(a0.wrapping_add(a3));
    out[1] = round12(a1.wrapping_add(a3));
    out[2] = round12(a2);
    out[3] = round12(a0.wrapping_add(a1).wrapping_sub(a3));
}

/// Forward ADST4: the transpose of the inverse matrix.
fn forward_adst4(x: &[i32], out: &mut [i32]) {
    let (x0, x1, x2, x3) = (x[0], x[1], x[2], x[3]);
    let (s1, s2, s3, s4) = (SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9);
    out[0] = round12(
        s1.wrapping_mul(x0)
            .wrapping_add(s2.wrapping_mul(x1))
            .wrapping_add(s3.wrapping_mul(x2))
            .wrapping_add(s4.wrapping_mul(x3)),
    );
    out[1] = round12(s3.wrapping_mul(x0.wrapping_add(x1).wrapping_sub(x3)));
    out[2] = round12(
        s4.wrapping_mul(x0)
            .wrapping_sub(s1.wrapping_mul(x1))
            .wrapping_sub(s3.wrapping_mul(x2))
            .wrapping_add(s2.wrapping_mul(x3)),
    );
    out[3] = round12(
        s2.wrapping_mul(x0)
            .wrapping_sub(s4.wrapping_mul(x1))
            .wrapping_add(s3.wrapping_mul(x2))
            .wrapping_sub(s1.wrapping_mul(x3)),
    );
}

/// Identity transform (same in both directions): scale by sqrt(2), 2, 2 sqrt(2) or 4 for 4/8/16/32 points.
fn identity_scale(x: &[i32], out: &mut [i32], log2: usize) {
    let n = 1 << log2;
    match log2 {
        2 => (0..n).for_each(|i| out[i] = round12(x[i].wrapping_mul(5793))),
        3 => (0..n).for_each(|i| out[i] = x[i].wrapping_mul(2)),
        4 => (0..n).for_each(|i| out[i] = round12(x[i].wrapping_mul(11586))),
        _ => (0..n).for_each(|i| out[i] = x[i].wrapping_mul(4)),
    }
}

/// One inverse 1-D transform of `1 << log2` values.
fn inverse_1d(kind: Kind, log2: usize, x: &[i32], out: &mut [i32], lo: i32, hi: i32) {
    match kind {
        Kind::Dct => run_inverse(dct_net(log2), x, out, lo, hi),
        Kind::Adst | Kind::FlipAdst => {
            if log2 == 2 {
                inverse_adst4(x, out);
            } else {
                run_inverse(adst_net(log2), x, out, lo, hi);
            }
            if kind == Kind::FlipAdst {
                out[..1 << log2].reverse();
            }
        }
        Kind::Identity => identity_scale(x, out, log2),
    }
}

/// One forward 1-D transform of `1 << log2` values.
fn forward_1d(kind: Kind, log2: usize, x: &[i32], out: &mut [i32]) {
    let n = 1 << log2;
    match kind {
        Kind::Dct => run_forward(dct_net(log2), x, out),
        Kind::Adst | Kind::FlipAdst => {
            let mut flipped = [0i32; 16];
            let x = if kind == Kind::FlipAdst {
                for i in 0..n {
                    flipped[i] = x[n - 1 - i];
                }
                &flipped[..n]
            } else {
                x
            };
            if log2 == 2 { forward_adst4(x, out) } else { run_forward(adst_net(log2), x, out) }
        }
        Kind::Identity => identity_scale(x, out, log2),
    }
}

// ---------------------------------------------------------------------------------------------------------
// Walsh-Hadamard (lossless)
// ---------------------------------------------------------------------------------------------------------

/// Specification 7.13.2.10: inverse WHT of 4 values (input already shifted).
fn inverse_wht4(x: [i32; 4]) -> [i32; 4] {
    let (mut a, mut c, mut d, mut b) = (x[0], x[1], x[2], x[3]);
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Exact inverse of `inverse_wht4`.
fn forward_wht4(o: [i32; 4]) -> [i32; 4] {
    let s3 = o[1];
    let s1 = o[2];
    let s2 = o[3] - s1;
    let s0 = o[0] + s3;
    let s4 = (s0 - s2) >> 1;
    let x3 = s4 - s3;
    let x1 = s4 - s1;
    let x2 = s2 + x3;
    let x0 = s0 - x1;
    [x0, x1, x2, x3]
}

fn inverse_wht_add(coefs: &[i32], bit_depth: u8, dst: &mut [u16], dst_stride: usize) {
    let mut rows = [[0i32; 4]; 4];
    for i in 0..4 {
        let r = &coefs[i * 4..i * 4 + 4];
        rows[i] = inverse_wht4([r[0] >> 2, r[1] >> 2, r[2] >> 2, r[3] >> 2]);
    }
    let max = (1i32 << bit_depth) - 1;
    for j in 0..4 {
        let col = inverse_wht4([rows[0][j], rows[1][j], rows[2][j], rows[3][j]]);
        for i in 0..4 {
            let p = &mut dst[i * dst_stride + j];
            *p = (*p as i32 + col[i]).clamp(0, max) as u16;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// 2-D transforms
// ---------------------------------------------------------------------------------------------------------

/// Inverse transform of one block and add it to `dst` (clipping to the bit depth).
///
/// `coefs`: dequantized coefficients, row-major with row stride `TX_WIDTH[tx_size]` (the array holds
/// `TX_WIDTH * TX_HEIGHT` values). For 64-wide/high sizes only the top-left 32x32 can be non-zero. The caller
/// has already clamped them to signed `bit_depth + 8` bits. The array is used as scratch and clobbered.
/// `nz_rows`/`nz_cols` are upper bounds on the number of rows/columns that contain non-zero coefficients;
/// everything outside them must be zero (passing the full dimensions is always correct).
#[allow(clippy::too_many_arguments)]
pub fn inverse_transform_add(
    coefs: &mut [i32],
    tx_size: u8,
    tx_type: u8,
    lossless: bool,
    bit_depth: u8,
    nz_rows: usize,
    nz_cols: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let ts = tx_size as usize;
    let (lw, lh) = (TX_WIDTH_LOG2[ts], TX_HEIGHT_LOG2[ts]);
    let (w, h) = (1usize << lw, 1usize << lh);
    debug_assert!(coefs.len() >= w * h);
    if lossless {
        inverse_wht_add(coefs, bit_depth, dst, dst_stride);
        return;
    }
    let rows = nz_rows.min(h.min(32));
    let cols = nz_cols.min(w.min(32));
    if rows == 0 || cols == 0 {
        return;
    }
    let bd = bit_depth as i32;
    let (row_hi, col_hi) = ((1i32 << (bd + 7)) - 1, (1i32 << (bd + 5).max(15)) - 1);
    let (row_lo, col_lo) = (-row_hi - 1, -col_hi - 1);
    let pixel_max = (1i32 << bd) - 1;
    let rect = lw.abs_diff(lh) == 1;
    let row_shift = ROW_SHIFT[ts];
    let (col_kind, row_kind) = (COL_KIND[tx_type as usize], ROW_KIND[tx_type as usize]);

    // Only a DC coefficient with DCT in both directions: every output pixel gets the same value.
    if rows == 1 && cols == 1 && tx_type == DCT_DCT {
        let mut dc = coefs[0];
        if rect {
            dc = round12(dc.wrapping_mul(2896));
        }
        dc = dc.clamp(row_lo, row_hi);
        dc = round12(dc.wrapping_mul(2896));
        dc = round_shift(dc, row_shift).clamp(col_lo, col_hi);
        dc = round12(dc.wrapping_mul(2896));
        let add = round_shift(dc, 4);
        for i in 0..h {
            for p in &mut dst[i * dst_stride..i * dst_stride + w] {
                *p = (*p as i32 + add).clamp(0, pixel_max) as u16;
            }
        }
        return;
    }

    // Row transforms, in place; each result is rounded and clamped to the range of the column pass.
    let mut input = [0i32; 64];
    let mut output = [0i32; 64];
    for i in 0..rows {
        let row = &mut coefs[i * w..(i + 1) * w];
        for j in 0..cols {
            let v = if rect { round12(row[j].wrapping_mul(2896)) } else { row[j] };
            input[j] = v.clamp(row_lo, row_hi);
        }
        inverse_1d(row_kind, lw, &input, &mut output, row_lo, row_hi);
        for j in 0..w {
            row[j] = round_shift(output[j], row_shift).clamp(col_lo, col_hi);
        }
    }

    // Column transforms.
    input = [0; 64];
    for j in 0..w {
        for i in 0..rows {
            input[i] = coefs[i * w + j];
        }
        inverse_1d(col_kind, lh, &input, &mut output, col_lo, col_hi);
        for i in 0..h {
            let p = &mut dst[i * dst_stride + j];
            *p = (*p as i32 + round_shift(output[i], 4)).clamp(0, pixel_max) as u16;
        }
    }
}

/// Forward transform for the encoder. `residual` is a (th x tw) block of source-minus-prediction values with
/// row stride `res_stride`. `coefs` receives `TX_WIDTH * TX_HEIGHT` values (row stride `TX_WIDTH`), scaled so
/// that the AV1 dequantizer plus `inverse_transform_add` reconstructs the residual. For 64-sized transforms
/// only the 32x32 low-frequency corner is computed; the rest is zero. Lossless uses the WHT.
pub fn forward_transform(
    residual: &[i32],
    res_stride: usize,
    tx_size: u8,
    tx_type: u8,
    lossless: bool,
    _bit_depth: u8,
    coefs: &mut [i32],
) {
    let ts = tx_size as usize;
    let (lw, lh) = (TX_WIDTH_LOG2[ts], TX_HEIGHT_LOG2[ts]);
    let (w, h) = (1usize << lw, 1usize << lh);
    let coefs = &mut coefs[..w * h];
    if lossless {
        let mut cols = [[0i32; 4]; 4];
        for j in 0..4 {
            let col = forward_wht4([
                residual[j],
                residual[res_stride + j],
                residual[2 * res_stride + j],
                residual[3 * res_stride + j],
            ]);
            for i in 0..4 {
                cols[i][j] = col[i];
            }
        }
        for i in 0..4 {
            let row = forward_wht4(cols[i]);
            for j in 0..4 {
                coefs[i * 4 + j] = row[j] << 2;
            }
        }
        return;
    }
    let shift = FWD_SHIFT[ts];
    let (col_kind, row_kind) = (COL_KIND[tx_type as usize], ROW_KIND[tx_type as usize]);
    let rect = lw.abs_diff(lh) == 1;
    let scale = |v: i32, s: i32| if s >= 0 { v << s } else { round_shift(v, (-s) as u32) };

    let mut mid = vec![0i32; w * h];
    let mut input = [0i32; 64];
    let mut output = [0i32; 64];
    for j in 0..w {
        for i in 0..h {
            input[i] = scale(residual[i * res_stride + j], shift[0]);
        }
        forward_1d(col_kind, lh, &input, &mut output);
        for i in 0..h {
            mid[i * w + j] = scale(output[i], shift[1]);
        }
    }
    coefs.fill(0);
    for i in 0..h.min(32) {
        for j in 0..w {
            let v = mid[i * w + j];
            input[j] = if rect { round12(v.wrapping_mul(2896)) } else { v };
        }
        forward_1d(row_kind, lw, &input, &mut output);
        for j in 0..w.min(32) {
            coefs[i * w + j] = scale(output[j], shift[2]);
        }
    }
}

/// Pixel-domain energy per unit of coefficient-domain energy for a transform size and type (the dequantized
/// coefficients live on the forward transform's scale, which is not orthonormal). Measured once per combination
/// by inverse-transforming a fixed pseudo-random coefficient perturbation, so the encoder can compare the
/// distortion of candidate quantizations without reconstructing them.
pub fn dist_scale(tx_size: u8, tx_type: u8) -> f64 {
    static TABLE: [[std::sync::OnceLock<f64>; 16]; 19] = [const { [const { std::sync::OnceLock::new() }; 16] }; 19];
    *TABLE[tx_size as usize][tx_type as usize].get_or_init(|| {
        let ts = tx_size as usize;
        let (w, h) = (1usize << TX_WIDTH_LOG2[ts], 1usize << TX_HEIGHT_LOG2[ts]);
        let (cw, ch) = (w.min(32), h.min(32));
        let (mut pix, mut coef) = (0f64, 0f64);
        let mut state = 0x2545_f491u32 ^ (ts as u32 * 977 + tx_type as u32);
        for _ in 0..4 {
            let mut c = vec![0i32; w * h];
            for i in 0..ch {
                for j in 0..cw {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let v = ((state >> 16) % 129) as i32 - 64;
                    c[i * w + j] = v;
                    coef += (v * v) as f64;
                }
            }
            let mut dst = vec![2048u16; w * h];
            inverse_transform_add(&mut c, tx_size, tx_type, false, 12, h.min(32), w.min(32), &mut dst, w);
            pix += dst.iter().map(|&p| ((p as i32 - 2048) as f64).powi(2)).sum::<f64>();
        }
        if coef == 0.0 { 1.0 } else { pix / coef }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift generator so reference hashes stay valid.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            (self.next() >> 11) % n
        }
    }

    fn is_valid(ts: usize, tt: u8) -> bool {
        match TX_WIDTH[ts].max(TX_HEIGHT[ts]) {
            64 => tt == DCT_DCT,
            32 => tt == DCT_DCT || tt == IDTX,
            _ => true,
        }
    }

    struct Case {
        coefs: Vec<i32>,
        rows: usize,
        cols: usize,
        dst: Vec<u16>,
    }

    /// A random sparse block with moderate amplitude (no intermediate overflow) over a random destination.
    fn random_case(rng: &mut Rng, ts: usize, bd: u8) -> Case {
        let (w, h) = (TX_WIDTH[ts], TX_HEIGHT[ts]);
        let rows = 1 + rng.below(h.min(32) as u64) as usize;
        let cols = 1 + rng.below(w.min(32) as u64) as usize;
        let amp = 1u64 << (bd as u32 + 4);
        let mut coefs = vec![0i32; w * h];
        for r in 0..rows {
            for c in 0..cols {
                if rng.below(3) != 0 {
                    coefs[r * w + c] = rng.below(2 * amp + 1) as i32 - amp as i32;
                }
            }
        }
        let dst = (0..w * h).map(|_| rng.below(1 << bd) as u16).collect();
        Case { coefs, rows, cols, dst }
    }

    fn fnv(hash: &mut u64, values: &[u16]) {
        for &v in values {
            for b in v.to_le_bytes() {
                *hash = (*hash ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }

    /// Hash of the reconstruction of 6 random blocks for every valid transform type of the size.
    fn reference_hash(ts: usize, bd: u8) -> u64 {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((ts as u64) << 8) ^ bd as u64);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let w = TX_WIDTH[ts];
        for tt in 0..16u8 {
            if !is_valid(ts, tt) {
                continue;
            }
            for _ in 0..6 {
                let case = random_case(&mut rng, ts, bd);
                let mut dst = case.dst.clone();
                let mut coefs = case.coefs.clone();
                inverse_transform_add(&mut coefs, ts as u8, tt, false, bd, case.rows, case.cols, &mut dst, w);
                // The nz hints are only an optimization.
                let mut dst_full = case.dst.clone();
                let mut coefs = case.coefs.clone();
                inverse_transform_add(&mut coefs, ts as u8, tt, false, bd, TX_HEIGHT[ts], w, &mut dst_full, w);
                assert_eq!(dst, dst_full, "size {ts} type {tt}");
                fnv(&mut hash, &dst);
            }
        }
        hash
    }

    /// Hashes produced by rav1e's `inverse_transform_add` (bit-exact with the specification) on the same
    /// blocks; one row per transform size, one column per bit depth (8, 10, 12).
    const REFERENCE_HASHES: [[u64; 3]; 19] = [
        [0xfcb3869309447aaa, 0xec95d38652183f94, 0x4176dffd1f3b4ef9],
        [0xbc4f756b996ebf5b, 0x9b02bf9f22deb742, 0xdc4b1bf8a2a88077],
        [0xe625be2b562dfdf0, 0x14f71e328a8985c4, 0x21099a8aa5344f2c],
        [0x0c2e126a96a26310, 0xdba2221063cb824c, 0xfa1c84a27c04b421],
        [0x9e481a411b32813b, 0xeb5b21e6893fc247, 0x3caabf116a8e64f1],
        [0x5283f89859460711, 0xbb21a40357bb0d6d, 0xb2eae6ec2b2e3634],
        [0x0c34e1082061e022, 0x3f9a84bb0f2b4561, 0x41706ca3c06179a7],
        [0x6485ea18ff930ad9, 0x268fbafedff16015, 0x077c2723abec815a],
        [0xf679af8e45baf326, 0x0750a53698688688, 0x8bce1d1938872cd5],
        [0x3111a00454a010ad, 0x34718402e46ef155, 0xad07226a84af5c72],
        [0xda7ddfd3bf50642e, 0xe4c5569ef31d58d1, 0x51b98bbe695731cd],
        [0xa03f189f9a902f52, 0x376dfa8904ab151c, 0xb9c128b96823dcfe],
        [0xab16ebfcc4f9c865, 0xd788893f34f2d8be, 0xacf659d7c4e32e9a],
        [0x11510c004610d028, 0x31c7566849964d5e, 0xe6f19754a49d5fd8],
        [0x2649e5cea63e4ce3, 0xb40b9d7edbbc432d, 0x3c315fe23a99759c],
        [0x7318c5aee27e6b9e, 0xd54f8a87bec81aa1, 0x7d0e819b4ecd914e],
        [0xd8739a29913c9a6f, 0x3eed67cc16f48314, 0xfaaebf3b8ae5f318],
        [0xae1c8259ceeb43a6, 0x57d359cbb1034831, 0x7001a48fdb448f6b],
        [0x9cba65a52b1adddf, 0x4f3a54628dfe0b43, 0x8695d11e4375d512],
    ];

    #[test]
    fn matches_reference_decoder() {
        for ts in 0..19 {
            for (k, bd) in [8u8, 10, 12].into_iter().enumerate() {
                assert_eq!(reference_hash(ts, bd), REFERENCE_HASHES[ts][k], "size {ts} bit depth {bd}");
            }
        }
    }

    #[test]
    fn size_tables_are_consistent() {
        for ts in 0..19 {
            assert_eq!(TX_WIDTH[ts], 1 << TX_WIDTH_LOG2[ts]);
            assert_eq!(TX_HEIGHT[ts], 1 << TX_HEIGHT_LOG2[ts]);
        }
        assert_eq!((TX_WIDTH[TX_16X64 as usize], TX_HEIGHT[TX_16X64 as usize]), (16, 64));
        assert_eq!((TX_WIDTH[TX_64X16 as usize], TX_HEIGHT[TX_64X16 as usize]), (64, 16));
        assert_eq!((TX_WIDTH[TX_4X8 as usize], TX_HEIGHT[TX_4X8 as usize]), (4, 8));
    }

    #[test]
    fn dc_only_is_flat_and_fast_path_agrees() {
        for ts in 0..19 {
            let (w, h) = (TX_WIDTH[ts], TX_HEIGHT[ts]);
            for bd in [8u8, 10, 12] {
                for dc in [-3000, -77, -1, 1, 5, 64, 1000, 20000] {
                    let mut fast = vec![1u16 << (bd - 1); w * h];
                    let mut slow = fast.clone();
                    let mut a = vec![0i32; w * h];
                    a[0] = dc;
                    let mut b = a.clone();
                    inverse_transform_add(&mut a, ts as u8, DCT_DCT, false, bd, 1, 1, &mut fast, w);
                    inverse_transform_add(&mut b, ts as u8, DCT_DCT, false, bd, 2, 2, &mut slow, w);
                    assert_eq!(fast, slow, "size {ts} dc {dc}");
                    assert!(fast.iter().all(|&p| p == fast[0]));
                }
            }
        }
    }

    #[test]
    fn dc_gain_of_4x4() {
        // 8-bit 4x4 DCT: dc * (1/sqrt2)^2 / 16 = dc / 32.
        let mut coefs = [0i32; 16];
        coefs[0] = 320;
        let mut dst = [100u16; 16];
        inverse_transform_add(&mut coefs, TX_4X4, DCT_DCT, false, 8, 1, 1, &mut dst, 4);
        assert_eq!(dst, [110u16; 16]);
    }

    #[test]
    fn empty_hints_leave_destination_alone() {
        let mut coefs = vec![5i32; 64];
        let mut dst = vec![7u16; 64];
        inverse_transform_add(&mut coefs, TX_8X8, ADST_ADST, false, 8, 0, 8, &mut dst, 8);
        inverse_transform_add(&mut coefs, TX_8X8, ADST_ADST, false, 8, 8, 0, &mut dst, 8);
        assert!(dst.iter().all(|&p| p == 7));
    }

    #[test]
    fn flipadst_mirrors_adst() {
        let mut rng = Rng(42);
        for ts in [TX_4X4, TX_8X8, TX_16X16, TX_8X16, TX_16X4] {
            let ts = ts as usize;
            let (w, h) = (TX_WIDTH[ts], TX_HEIGHT[ts]);
            // (flipped type, plain type, mirror vertically?, mirror horizontally?)
            let pairs = [
                (FLIPADST_DCT, ADST_DCT, true, false),
                (DCT_FLIPADST, DCT_ADST, false, true),
                (FLIPADST_FLIPADST, ADST_ADST, true, true),
                (V_FLIPADST, V_ADST, true, false),
                (H_FLIPADST, H_ADST, false, true),
            ];
            for (flipped, plain, vert, horiz) in pairs {
                let mut case = random_case(&mut rng, ts, 8);
                case.dst.fill(128);
                for c in case.coefs.iter_mut() {
                    *c /= 64;
                }
                let mut a = case.dst.clone();
                let mut b = case.dst.clone();
                inverse_transform_add(&mut case.coefs.clone(), ts as u8, flipped, false, 8, h, w, &mut a, w);
                inverse_transform_add(&mut case.coefs, ts as u8, plain, false, 8, h, w, &mut b, w);
                for i in 0..h {
                    for j in 0..w {
                        let (si, sj) = (if vert { h - 1 - i } else { i }, if horiz { w - 1 - j } else { j });
                        assert_eq!(a[i * w + j], b[si * w + sj], "size {ts} type {flipped}");
                    }
                }
            }
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let mut rng = Rng(7);
        for ts in 0..19 {
            let (w, h) = (TX_WIDTH[ts], TX_HEIGHT[ts]);
            for tt in 0..16u8 {
                if !is_valid(ts, tt) {
                    continue;
                }
                for bd in [8u8, 10, 12] {
                    let mid = 1i32 << (bd - 1);
                    let big = w == 64 || h == 64;
                    // 64-point transforms drop the high frequencies, so give them a smooth residual.
                    let amp = (mid / 2) as u64;
                    let residual: Vec<i32> = (0..w * h)
                        .map(|k| {
                            if big {
                                let (i, j) = ((k / w) as i32, (k % w) as i32);
                                (i * mid / 2 / h as i32) - (j * mid / 4 / w as i32) + 3
                            } else {
                                rng.below(2 * amp + 1) as i32 - amp as i32
                            }
                        })
                        .collect();
                    let mut coefs = vec![0i32; w * h];
                    forward_transform(&residual, w, ts as u8, tt, false, bd, &mut coefs);
                    let mut dst = vec![mid as u16; w * h];
                    inverse_transform_add(&mut coefs, ts as u8, tt, false, bd, h, w, &mut dst, w);
                    let worst = (0..w * h).map(|k| (dst[k] as i32 - mid - residual[k]).abs()).max().unwrap();
                    assert!(worst <= if big { 6 } else { 3 }, "size {ts} type {tt} bit depth {bd}: error {worst}");
                }
            }
        }
    }

    #[test]
    fn forward_zeroes_high_frequencies_of_64_sizes() {
        let (w, h) = (64, 64);
        let residual = vec![9i32; w * h];
        let mut coefs = vec![1i32; w * h];
        forward_transform(&residual, w, TX_64X64, DCT_DCT, false, 8, &mut coefs);
        for i in 0..h {
            for j in 0..w {
                if i >= 32 || j >= 32 {
                    assert_eq!(coefs[i * w + j], 0);
                }
            }
        }
        assert_ne!(coefs[0], 0);
        assert!(coefs[1..32].iter().all(|&c| c.abs() <= 1));
    }

    #[test]
    fn lossless_roundtrip_is_exact() {
        let mut rng = Rng(99);
        for bd in [8u8, 10, 12] {
            for _ in 0..200 {
                let pred: Vec<u16> = (0..16).map(|_| rng.below(1 << bd) as u16).collect();
                let src: Vec<u16> = (0..16).map(|_| rng.below(1 << bd) as u16).collect();
                let residual: Vec<i32> = (0..16).map(|k| src[k] as i32 - pred[k] as i32).collect();
                let mut coefs = vec![0i32; 16];
                forward_transform(&residual, 4, TX_4X4, DCT_DCT, true, bd, &mut coefs);
                let mut dst = pred.clone();
                inverse_transform_add(&mut coefs, TX_4X4, DCT_DCT, true, bd, 4, 4, &mut dst, 4);
                assert_eq!(dst, src);
            }
        }
    }

    #[test]
    fn lossless_flat_residual_has_a_single_coefficient() {
        let residual = [3i32; 16];
        let mut coefs = [0i32; 16];
        forward_transform(&residual, 4, TX_4X4, DCT_DCT, true, 8, &mut coefs);
        assert_eq!(coefs.iter().filter(|&&c| c != 0).count(), 1);
        let mut dst = [10u16; 16];
        inverse_transform_add(&mut coefs, TX_4X4, DCT_DCT, true, 8, 4, 4, &mut dst, 4);
        assert_eq!(dst, [13u16; 16]);
    }

    #[test]
    fn destination_stride_is_respected() {
        let mut coefs = vec![0i32; 64];
        coefs[0] = 640;
        let mut dst = vec![50u16; 8 * 20];
        inverse_transform_add(&mut coefs, TX_8X8, DCT_DCT, false, 8, 1, 1, &mut dst, 20);
        for i in 0..8 {
            for j in 0..20 {
                let expect = if j < 8 { 50 + 10 } else { 50 };
                assert_eq!(dst[i * 20 + j], expect);
            }
        }
    }

    #[test]
    fn output_is_clipped_to_the_bit_depth() {
        let mut coefs = vec![0i32; 16];
        coefs[0] = 30000;
        let mut dst = vec![250u16; 16];
        inverse_transform_add(&mut coefs, TX_4X4, DCT_DCT, false, 8, 1, 1, &mut dst, 4);
        assert!(dst.iter().all(|&p| p == 255));
        let mut coefs = vec![0i32; 16];
        coefs[0] = -30000;
        let mut dst = vec![10u16; 16];
        inverse_transform_add(&mut coefs, TX_4X4, DCT_DCT, false, 10, 1, 1, &mut dst, 4);
        assert!(dst.iter().all(|&p| p == 0));
    }
}
