//! Small constant tables of the AV1 specification: block sizes, partitions, transform sizes.
//! Numbering follows the specification.

pub const BLOCK_4X4: u8 = 0;
pub const BLOCK_4X8: u8 = 1;
pub const BLOCK_8X4: u8 = 2;
pub const BLOCK_8X8: u8 = 3;
pub const BLOCK_8X16: u8 = 4;
pub const BLOCK_16X8: u8 = 5;
pub const BLOCK_16X16: u8 = 6;
pub const BLOCK_16X32: u8 = 7;
pub const BLOCK_32X16: u8 = 8;
pub const BLOCK_32X32: u8 = 9;
pub const BLOCK_32X64: u8 = 10;
pub const BLOCK_64X32: u8 = 11;
pub const BLOCK_64X64: u8 = 12;
pub const BLOCK_64X128: u8 = 13;
pub const BLOCK_128X64: u8 = 14;
pub const BLOCK_128X128: u8 = 15;
pub const BLOCK_4X16: u8 = 16;
pub const BLOCK_16X4: u8 = 17;
pub const BLOCK_8X32: u8 = 18;
pub const BLOCK_32X8: u8 = 19;
pub const BLOCK_16X64: u8 = 20;
pub const BLOCK_64X16: u8 = 21;
pub const BLOCK_SIZES: usize = 22;
pub const BLOCK_INVALID: u8 = 22;

/// Block width and height in 4-sample units.
pub const NUM_4X4_WIDE: [u8; BLOCK_SIZES] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
pub const NUM_4X4_HIGH: [u8; BLOCK_SIZES] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

pub fn block_width(b: u8) -> usize {
    NUM_4X4_WIDE[b as usize] as usize * 4
}

pub fn block_height(b: u8) -> usize {
    NUM_4X4_HIGH[b as usize] as usize * 4
}

/// The block size with the given 4-sample dimensions, if there is one.
pub fn block_size_from(w4: usize, h4: usize) -> u8 {
    for b in 0..BLOCK_SIZES {
        if NUM_4X4_WIDE[b] as usize == w4 && NUM_4X4_HIGH[b] as usize == h4 {
            return b as u8;
        }
    }
    BLOCK_INVALID
}

pub const PARTITION_NONE: u8 = 0;
pub const PARTITION_HORZ: u8 = 1;
pub const PARTITION_VERT: u8 = 2;
pub const PARTITION_SPLIT: u8 = 3;
pub const PARTITION_HORZ_A: u8 = 4;
pub const PARTITION_HORZ_B: u8 = 5;
pub const PARTITION_VERT_A: u8 = 6;
pub const PARTITION_VERT_B: u8 = 7;
pub const PARTITION_HORZ_4: u8 = 8;
pub const PARTITION_VERT_4: u8 = 9;

/// `Partition_Subsize[partition][bsize]`.
pub fn partition_subsize(partition: u8, bsize: u8) -> u8 {
    let w4 = NUM_4X4_WIDE[bsize as usize] as usize;
    let h4 = NUM_4X4_HIGH[bsize as usize] as usize;
    match partition {
        PARTITION_NONE => bsize,
        PARTITION_HORZ | PARTITION_HORZ_A | PARTITION_HORZ_B => block_size_from(w4, h4 / 2),
        PARTITION_VERT | PARTITION_VERT_A | PARTITION_VERT_B => block_size_from(w4 / 2, h4),
        PARTITION_SPLIT => block_size_from(w4 / 2, h4 / 2),
        PARTITION_HORZ_4 => block_size_from(w4, h4 / 4),
        PARTITION_VERT_4 => block_size_from(w4 / 4, h4),
        _ => BLOCK_INVALID,
    }
}

/// `Subsampled_Size[bsize][sub_x][sub_y]`: the block covering the chroma samples of a luma block.
pub fn subsampled_size(bsize: u8, sub_x: u8, sub_y: u8) -> u8 {
    let w4 = (NUM_4X4_WIDE[bsize as usize] as usize >> sub_x).max(1);
    let h4 = (NUM_4X4_HIGH[bsize as usize] as usize >> sub_y).max(1);
    let b = block_size_from(w4, h4);
    if b != BLOCK_INVALID {
        return b;
    }
    // 4:2:2 can leave shapes such as 2x16 samples; the specification maps these to the nearest valid block.
    let (mut w4, mut h4) = (w4, h4);
    while block_size_from(w4, h4) == BLOCK_INVALID && (w4 > 1 || h4 > 1) {
        if w4 * 4 < h4 {
            w4 *= 2;
        } else {
            h4 *= 2;
        }
        if w4 > 32 || h4 > 32 {
            break;
        }
    }
    block_size_from(w4, h4)
}

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

pub const TX_W: [u8; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
pub const TX_H: [u8; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
pub const TX_W_LOG2: [u8; 19] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
pub const TX_H_LOG2: [u8; 19] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];

/// `Tx_Size_Sqr`: the square size of the smaller dimension.
pub const TX_SQR: [u8; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
/// `Tx_Size_Sqr_Up`: the square size of the larger dimension.
pub const TX_SQR_UP: [u8; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

/// `Split_Tx_Size`.
pub const SPLIT_TX_SIZE: [u8; 19] = [
    TX_4X4, TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16, TX_32X32, TX_32X32,
    TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_16X32, TX_32X16,
];

/// `Max_Tx_Size_Rect` by block size.
pub const MAX_TX_SIZE_RECT: [u8; BLOCK_SIZES] = [
    TX_4X4, TX_4X8, TX_8X4, TX_8X8, TX_8X16, TX_16X8, TX_16X16, TX_16X32, TX_32X16, TX_32X32, TX_32X64, TX_64X32,
    TX_64X64, TX_64X64, TX_64X64, TX_64X64, TX_4X16, TX_16X4, TX_8X32, TX_32X8, TX_16X64, TX_64X16,
];

/// `Max_Tx_Depth` by block size.
pub const MAX_TX_DEPTH: [u8; BLOCK_SIZES] = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4];

/// `Adjusted_Tx_Size`: the size actually coded (64-sample dimensions are cut to 32).
pub const ADJUSTED_TX_SIZE: [u8; 19] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_32X32, TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_16X32, TX_32X16, TX_32X32,
    TX_32X32, TX_4X16, TX_16X4, TX_8X32, TX_32X8, TX_16X32, TX_32X16,
];

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
pub const UV_CFL_PRED: u8 = 13;

pub fn is_directional(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

/// `Intra_Mode_Context`: neighbour mode class for the key-frame luma mode CDF.
pub const INTRA_MODE_CONTEXT: [u8; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `Mode_To_Txfm` for chroma transform types.
pub const MODE_TO_TXFM: [u8; 14] = [
    0, // DC -> DCT_DCT
    1, // V -> ADST_DCT
    2, // H -> DCT_ADST
    0, // D45 -> DCT_DCT
    3, // D135 -> ADST_ADST
    1, // D113 -> ADST_DCT
    2, // D157 -> DCT_ADST
    2, // D203 -> DCT_ADST
    1, // D67 -> ADST_DCT
    3, // SMOOTH -> ADST_ADST
    1, // SMOOTH_V -> ADST_DCT
    2, // SMOOTH_H -> DCT_ADST
    3, // PAETH -> ADST_ADST
    0, // CFL -> DCT_DCT
];

/// `Filter_Intra_Mode_To_Intra_Dir`.
pub const FILTER_INTRA_MODE_TO_INTRA_DIR: [u8; 5] = [DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED];

pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;

/// Transform-type symbol → type for the 7-type intra set (spec `Tx_Type_Intra_Inv_Set1`).
pub const TX_TYPE_INTRA_INV_SET1: [u8; 7] = [IDTX, 0, V_DCT, H_DCT, 3, 1, 2];
/// Transform-type symbol → type for the 5-type intra set (`Tx_Type_Intra_Inv_Set2`).
pub const TX_TYPE_INTRA_INV_SET2: [u8; 5] = [IDTX, 0, 3, 1, 2];

/// The 2-D, horizontal and vertical transform classes.
pub const TX_CLASS_2D: u8 = 0;
pub const TX_CLASS_HORIZ: u8 = 1;
pub const TX_CLASS_VERT: u8 = 2;

pub fn tx_class(tx_type: u8) -> u8 {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    }
}

pub const SEG_LVL_SKIP: usize = 6;

/// Palette colour context from the neighbour score hash.
pub const PALETTE_COLOR_CONTEXT: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];
pub const PALETTE_COLOR_HASH_MULTIPLIERS: [u32; 3] = [1, 2, 2];

/// Box-filter radii `[r0, r1]` of each self-guided parameter set (`Sgr_Params`); 0 means the pass is off.
pub const SGR_RADII: [[u8; 2]; 16] = [
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [0, 2],
    [0, 2],
    [0, 2],
    [0, 2],
    [2, 0],
    [2, 0],
];
