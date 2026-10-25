//! Film grain synthesis (AV1 specification section 7.18.3).
//!
//! Grain is *not* part of the coded picture: the decoder keeps the clean frame as the reference and adds the
//! noise only to the copy it outputs. The noise comes from pseudo-random 73x82 templates (an auto-regressive
//! filter over Gaussian samples), cut into 32x32 blocks at random offsets, cross-faded where blocks overlap
//! and scaled by a piecewise-linear function of the underlying sample value.

use super::frame::Frame;

/// The `film_grain_params` syntax elements, with the values exactly as coded (offsets such as the `+ 128` of
/// the auto-regressive coefficients are removed internally).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        FilmGrainParams {
            apply_grain: false,
            grain_seed: 0,
            num_y_points: 0,
            point_y_value: [0; 14],
            point_y_scaling: [0; 14],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            point_cb_value: [0; 10],
            point_cb_scaling: [0; 10],
            num_cr_points: 0,
            point_cr_value: [0; 10],
            point_cr_scaling: [0; 10],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y_plus_128: [128; 24],
            ar_coeffs_cb_plus_128: [128; 25],
            ar_coeffs_cr_plus_128: [128; 25],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 128,
            cb_luma_mult: 192,
            cb_offset: 256,
            cr_mult: 128,
            cr_luma_mult: 192,
            cr_offset: 256,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }
}

/// Adds film grain to a copy of `frame` (the visible `frame.width` x `frame.height` area) and returns it.
///
/// `matrix_coefficients_identity` is true when the sequence's `matrix_coefficients` is `MC_IDENTITY` (RGB
/// coded as GBR); it only changes the upper clip limit of the chroma planes in restricted range.
/// With `apply_grain == false` the result is an unmodified copy.
pub fn apply_film_grain(frame: &Frame, params: &FilmGrainParams, matrix_coefficients_identity: bool) -> Frame {
    let mut out = frame.clone();
    if !params.apply_grain || frame.width == 0 || frame.height == 0 {
        return out;
    }
    let bd = frame.bit_depth as u32;
    let (sub_x, sub_y) = (frame.sub_x as usize, frame.sub_y as usize);
    let has_chroma = frame.planes.len() >= 3;
    let csfl = params.chroma_scaling_from_luma;

    let enabled = [
        params.num_y_points > 0,
        has_chroma && (params.num_cb_points > 0 || csfl),
        has_chroma && (params.num_cr_points > 0 || csfl),
    ];
    if !enabled.iter().any(|&e| e) {
        return out;
    }

    let luma_grain = luma_template(params, bd);
    let chroma_grain = [
        chroma_template(params, bd, 0, &luma_grain, sub_x, sub_y),
        chroma_template(params, bd, 1, &luma_grain, sub_x, sub_y),
    ];
    let grain_ref = [&luma_grain, &chroma_grain[0], &chroma_grain[1]];

    let luts = [
        scaling_lut(&params.point_y_value[..params.num_y_points as usize], &params.point_y_scaling),
        scaling_lut(&params.point_cb_value[..params.num_cb_points as usize], &params.point_cb_scaling),
        scaling_lut(&params.point_cr_value[..params.num_cr_points as usize], &params.point_cr_scaling),
    ];

    let offsets = block_offsets(params, frame.width, frame.height);
    let scaling_shift = params.grain_scaling_minus_8 as u32 + 8;
    let (min_value, max_luma, max_chroma) = if params.clip_to_restricted_range {
        let max_chroma = if matrix_coefficients_identity { 235 } else { 240 };
        (16 << (bd - 8), 235 << (bd - 8), max_chroma << (bd - 8))
    } else {
        let top = (256 << (bd - 8)) - 1;
        (0, top, top)
    };
    let max_sample = (1i32 << bd) - 1;

    // Luma noise.
    if enabled[0] {
        let full = full_scaling_lut(&luts[0], bd);
        let stripes = NoiseStripes::build(
            &luma_grain,
            GRAIN_W,
            0,
            0,
            frame.width,
            frame.height,
            &offsets,
            params.overlap_flag,
            bd,
        );
        let mut noise = vec![0i32; frame.width];
        let src = &frame.planes[0];
        for y in 0..frame.height {
            stripes.row(y, &mut noise);
            let (srow, drow) = (src.row(y), out.planes[0].row_mut(y));
            for x in 0..frame.width {
                let orig = srow[x] as i32;
                let n = round2(full[orig as usize] * noise[x], scaling_shift);
                drow[x] = (orig + n).clamp(min_value, max_luma) as u16;
            }
        }
    }

    // Chroma noise: the scaling input mixes the (clean) luma and chroma samples.
    for c in 0..2 {
        if !enabled[c + 1] {
            continue;
        }
        let (mult, luma_mult, offset) = if c == 0 {
            (params.cb_mult as i32 - 128, params.cb_luma_mult as i32 - 128, params.cb_offset as i32 - 256)
        } else {
            (params.cr_mult as i32 - 128, params.cr_luma_mult as i32 - 128, params.cr_offset as i32 - 256)
        };
        let full = full_scaling_lut(if csfl { &luts[0] } else { &luts[c + 1] }, bd);
        let (cw, ch) = ((frame.width + sub_x) >> sub_x, (frame.height + sub_y) >> sub_y);
        let tw = if sub_x == 1 { CHROMA_SUB_W } else { GRAIN_W };
        let stripes = NoiseStripes::build(
            grain_ref[c + 1],
            tw,
            sub_x,
            sub_y,
            frame.width,
            frame.height,
            &offsets,
            params.overlap_flag,
            bd,
        );
        let mut noise = vec![0i32; cw];
        let luma = &frame.planes[0];
        let src = &frame.planes[c + 1];
        for y in 0..ch {
            stripes.row(y, &mut noise);
            let lrow = luma.row(y << sub_y);
            let (srow, drow) = (src.row(y), out.planes[c + 1].row_mut(y));
            for x in 0..cw {
                let lx = x << sub_x;
                let average = if sub_x == 1 {
                    (lrow[lx] as i32 + lrow[(lx + 1).min(frame.width - 1)] as i32 + 1) >> 1
                } else {
                    lrow[lx] as i32
                };
                let orig = srow[x] as i32;
                let merged = if csfl {
                    average
                } else {
                    let combined = average * luma_mult + orig * mult;
                    ((combined >> 6) + (offset << (bd - 8))).clamp(0, max_sample)
                };
                let n = round2(full[merged as usize] * noise[x], scaling_shift);
                drow[x] = (orig + n).clamp(min_value, max_chroma) as u16;
            }
        }
    }
    out
}

const GRAIN_W: usize = 82;
const GRAIN_H: usize = 73;
const CHROMA_SUB_W: usize = 44;
const CHROMA_SUB_H: usize = 38;

#[inline(always)]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// The 16-bit linear-feedback shift register of 7.18.3.2.
struct Lfsr(u32);

impl Lfsr {
    fn bits(&mut self, n: u32) -> i32 {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.0 = (r >> 1) | (bit << 15);
        ((self.0 >> (16 - n)) & ((1 << n) - 1)) as i32
    }
}

fn grain_limits(bd: u32) -> (i32, i32) {
    let center = 128 << (bd - 8);
    (-center, center - 1)
}

/// Fills a template with scaled Gaussian samples.
fn gaussian_fill(grain: &mut [i32], rng: &mut Lfsr, params: &FilmGrainParams, bd: u32) {
    let shift = 12 - bd + params.grain_scale_shift as u32;
    for g in grain.iter_mut() {
        *g = round2(GAUSSIAN_SEQUENCE[rng.bits(11) as usize] as i32, shift);
    }
}

/// 7.18.3.3, luma part: a 73x82 template.
fn luma_template(params: &FilmGrainParams, bd: u32) -> Vec<i32> {
    let mut grain = vec![0i32; GRAIN_W * GRAIN_H];
    if params.num_y_points == 0 {
        return grain;
    }
    gaussian_fill(&mut grain, &mut Lfsr(params.grain_seed as u32), params, bd);
    let (gmin, gmax) = grain_limits(bd);
    let shift = params.ar_coeff_shift_minus_6 as u32 + 6;
    let lag = params.ar_coeff_lag as isize;
    for y in 3..GRAIN_H {
        for x in 3..GRAIN_W - 3 {
            let mut sum = 0;
            let mut pos = 0;
            'taps: for dr in -lag..=0 {
                for dc in -lag..=lag {
                    if dr == 0 && dc == 0 {
                        break 'taps;
                    }
                    let c = params.ar_coeffs_y_plus_128[pos] as i32 - 128;
                    sum += c * grain[(y as isize + dr) as usize * GRAIN_W + (x as isize + dc) as usize];
                    pos += 1;
                }
            }
            let g = &mut grain[y * GRAIN_W + x];
            *g = (*g + round2(sum, shift)).clamp(gmin, gmax);
        }
    }
    grain
}

/// 7.18.3.3, chroma part for Cb (`plane == 0`) or Cr (`plane == 1`): a template of 38 or 73 rows by 44 or 82
/// columns, whose auto-regression also looks at the co-located luma grain.
fn chroma_template(
    params: &FilmGrainParams,
    bd: u32,
    plane: usize,
    luma: &[i32],
    sub_x: usize,
    sub_y: usize,
) -> Vec<i32> {
    let w = if sub_x == 1 { CHROMA_SUB_W } else { GRAIN_W };
    let h = if sub_y == 1 { CHROMA_SUB_H } else { GRAIN_H };
    let mut grain = vec![0i32; w * h];
    let num_points = if plane == 0 { params.num_cb_points } else { params.num_cr_points };
    if num_points == 0 && !params.chroma_scaling_from_luma {
        return grain;
    }
    let (seed_xor, coeffs) =
        if plane == 0 { (0xb524, &params.ar_coeffs_cb_plus_128) } else { (0x49d8, &params.ar_coeffs_cr_plus_128) };
    gaussian_fill(&mut grain, &mut Lfsr(params.grain_seed as u32 ^ seed_xor), params, bd);
    let (gmin, gmax) = grain_limits(bd);
    let shift = params.ar_coeff_shift_minus_6 as u32 + 6;
    let lag = params.ar_coeff_lag as isize;
    for y in 3..h {
        for x in 3..w - 3 {
            let mut sum = 0;
            let mut pos = 0;
            'taps: for dr in -lag..=0 {
                for dc in -lag..=lag {
                    let c = coeffs[pos] as i32 - 128;
                    if dr == 0 && dc == 0 {
                        if params.num_y_points > 0 {
                            let luma_x = ((x - 3) << sub_x) + 3;
                            let luma_y = ((y - 3) << sub_y) + 3;
                            let mut l = 0;
                            for i in 0..=sub_y {
                                for j in 0..=sub_x {
                                    l += luma[(luma_y + i) * GRAIN_W + luma_x + j];
                                }
                            }
                            sum += round2(l, (sub_x + sub_y) as u32) * c;
                        }
                        break 'taps;
                    }
                    sum += c * grain[(y as isize + dr) as usize * w + (x as isize + dc) as usize];
                    pos += 1;
                }
            }
            let g = &mut grain[y * w + x];
            *g = (*g + round2(sum, shift)).clamp(gmin, gmax);
        }
    }
    grain
}

/// 7.18.3.4: the 256-entry scaling function from its piecewise-linear points (8-bit domain).
fn scaling_lut(values: &[u8], scalings: &[u8]) -> [i32; 256] {
    let mut lut = [0i32; 256];
    let n = values.len();
    if n == 0 {
        return lut;
    }
    for entry in lut.iter_mut().take(values[0] as usize) {
        *entry = scalings[0] as i32;
    }
    for i in 0..n - 1 {
        let delta_y = scalings[i + 1] as i32 - scalings[i] as i32;
        let delta_x = values[i + 1] as i32 - values[i] as i32;
        if delta_x <= 0 {
            continue; // not a valid stream: the points must increase
        }
        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            lut[(values[i] as i32 + x) as usize] = scalings[i] as i32 + ((x * delta + 32768) >> 16);
        }
    }
    for entry in lut.iter_mut().skip(values[n - 1] as usize) {
        *entry = scalings[n - 1] as i32;
    }
    lut
}

/// `scale_lut` for every sample value of the bit depth: higher depths interpolate between 8-bit entries.
fn full_scaling_lut(lut: &[i32; 256], bd: u32) -> Vec<i32> {
    let shift = bd - 8;
    (0..1usize << bd)
        .map(|index| {
            let x = index >> shift;
            if shift == 0 || x == 255 {
                return lut[x];
            }
            let rem = (index - (x << shift)) as i32;
            lut[x] + round2((lut[x + 1] - lut[x]) * rem, shift)
        })
        .collect()
}

/// The random template offset of every 32x32 block, as drawn in 7.18.3.5: `result[stripe][block]`.
fn block_offsets(params: &FilmGrainParams, width: usize, height: usize) -> Vec<Vec<u8>> {
    let blocks = width.div_ceil(2).div_ceil(16);
    let stripes = height.div_ceil(2).div_ceil(16);
    (0..stripes)
        .map(|stripe| {
            let mut rng = Lfsr(params.grain_seed as u32);
            rng.0 ^= (((stripe * 37 + 178) & 255) << 8) as u32;
            rng.0 ^= ((stripe * 173 + 105) & 255) as u32;
            (0..blocks).map(|_| rng.bits(8) as u8).collect()
        })
        .collect()
}

/// The noise of one plane as stripes of 32 luma rows (plus 2 rows shared with the next stripe), each built
/// from template blocks that were already cross-faded horizontally.
struct NoiseStripes {
    data: Vec<i16>,
    /// Rows stored per stripe (34, or 17 with vertical subsampling) and samples per stored row.
    rows: usize,
    stride: usize,
    sub_y: usize,
    overlap: bool,
    gmin: i32,
    gmax: i32,
}

impl NoiseStripes {
    #[allow(clippy::too_many_arguments)]
    fn build(
        template: &[i32],
        template_stride: usize,
        sub_x: usize,
        sub_y: usize,
        width: usize,
        height: usize,
        offsets: &[Vec<u8>],
        overlap: bool,
        bd: u32,
    ) -> NoiseStripes {
        let (gmin, gmax) = grain_limits(bd);
        let rows = 34 >> sub_y;
        let block_w = 34 >> sub_x;
        let nblocks = width.div_ceil(2).div_ceil(16);
        let stride = ((nblocks * 32) >> sub_x) + 2;
        let mut data = vec![0i16; offsets.len() * rows * stride];
        debug_assert_eq!(offsets.len(), height.div_ceil(2).div_ceil(16));
        for (n, stripe_offsets) in offsets.iter().enumerate() {
            for (b, &rand) in stripe_offsets.iter().enumerate() {
                let (off_x, off_y) = ((rand >> 4) as usize, (rand & 15) as usize);
                let tx = if sub_x == 1 { 6 + off_x } else { 9 + off_x * 2 };
                let ty = if sub_y == 1 { 6 + off_y } else { 9 + off_y * 2 };
                let base = (b * 32) >> sub_x;
                for i in 0..rows {
                    let row = &mut data[(n * rows + i) * stride..(n * rows + i + 1) * stride];
                    for j in 0..block_w {
                        let mut g = template[(ty + i) * template_stride + tx + j];
                        if overlap && b > 0 {
                            let old = row[base + j] as i32;
                            if sub_x == 0 && j < 2 {
                                g = if j == 0 { old * 27 + g * 17 } else { old * 17 + g * 27 };
                                g = round2(g, 5).clamp(gmin, gmax);
                            } else if sub_x == 1 && j == 0 {
                                g = round2(old * 23 + g * 22, 5).clamp(gmin, gmax);
                            }
                        }
                        row[base + j] = g as i16;
                    }
                }
            }
        }
        NoiseStripes { data, rows, stride, sub_y, overlap, gmin, gmax }
    }

    /// The final noise of image row `y` of the plane (cross-faded with the previous stripe), into `out`.
    fn row(&self, y: usize, out: &mut [i32]) {
        let stripe_h = 32 >> self.sub_y;
        let (n, i) = (y / stripe_h, y % stripe_h);
        let cur = &self.data[(n * self.rows + i) * self.stride..][..out.len()];
        if self.overlap && n > 0 && i < (2 >> self.sub_y) {
            let old = &self.data[((n - 1) * self.rows + i + stripe_h) * self.stride..][..out.len()];
            let (w_old, w_cur) = match (self.sub_y, i) {
                (0, 0) => (27, 17),
                (0, _) => (17, 27),
                _ => (23, 22),
            };
            for ((o, &a), &b) in out.iter_mut().zip(old).zip(cur) {
                *o = round2(a as i32 * w_old + b as i32 * w_cur, 5).clamp(self.gmin, self.gmax);
            }
        } else {
            for (o, &b) in out.iter_mut().zip(cur) {
                *o = b as i32;
            }
        }
    }
}

/// `Gaussian_Sequence`: 2048 samples of the film grain noise distribution (11-bit random index).
const GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388, -732, -104, -188, 800,
    112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500, -228, 12, 680, 272, -476, 972, -100, 652,
    368, 432, -196, -720, -192, 1000, -332, 652, -136, -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124,
    -24, 396, 540, -12, -104, 640, 464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40,
    -156, 252, -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60, -264, -96,
    -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792, 304, 48, -28, -1232, -1172, -448,
    104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548, 488, -344, -136, 488, -196, -224, 656, -236, -1128,
    60, 4, 140, 276, -676, -376, 168, -108, 464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116,
    436, 504, -232, 328, 844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4, 196, 44, -36, -648,
    932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448, 208, -228, -1044, -328, 660, 380, -148,
    -300, 588, 240, 540, 28, 136, -88, -436, 256, 296, -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484,
    -592, -344, 796, 124, -668, -768, 388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648,
    -136, 372, -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112, -220, -100,
    996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412, -320, 120, -300, -384, -604, -572,
    -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240, -208, -40, 100, -592, 344, -308, -452, -228, 20, 916,
    -1752, -136, -340, -804, 140, 40, 512, 340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752,
    -100, 560, -1020, 180, -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164, -376, -684, 592,
    -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864, 424, -728, 988, -604, 904, -592, 296,
    -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776, -824, -884, 524, -548, -564, -68, -164, -96, 692, 364,
    -692, -1012, -68, 260, -480, 876, -1116, 452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32,
    280, 200, 112, -440, -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172, -68, 24, 424, 264,
    1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276, -508, 940, 24, -400, -120, 756, 60,
    236, -412, 100, 376, -484, 400, -100, -740, -108, -260, 328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60,
    892, -888, 60, 164, 68, -760, 216, -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164,
    -1560, -776, 1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272, 460,
    -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560, 1040, -472, 116, -848,
    -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972, -584, -204, 264, 880, 528, -24, -184, 116,
    448, -144, 828, 524, 212, -212, 52, 12, 200, 268, -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492,
    -576, 16, 720, -108, 384, 124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436,
    316, 372, -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4, -1168, 368,
    476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272, -256, -420, 320, 208, -144, -156,
    156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360, 32, -388, 496, -680, -48, 208, -116, -408, 60, -604,
    -392, 548, -840, 784, -460, 656, -544, -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308,
    308, -112, -636, -760, 280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636,
    -76, 392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24, -1044, -32,
    40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424, -1008, -324, 588, -1496, 576, 460,
    -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52, -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104,
    -284, -404, 732, -520, 164, -304, -540, 120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148,
    716, -604, 404, -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508, 132,
    -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384, -188, -612, -576, 736, 508,
    892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508, -144, -8, 484, 48, 284, -260, -240, 256, -100,
    -292, -204, -44, 472, -204, 908, -188, -1000, -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760,
    -176, 376, -524, -452, -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832, 148, 248, 652, 616,
    1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92, -272, 296, 448, -464, 412, -200, 392, 440,
    -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64, 156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256,
    84, 116, -352, 48, 116, 304, -384, 412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524,
    -268, -720, 540, -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888, 8,
    552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468, -28, -120, -32, -44,
    1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156, 36, 492, -156, -808, 188, 1652, 68, -120,
    -116, 316, 160, -140, 352, 808, -416, 592, 316, -480, 56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324,
    -416, -600, 768, 268, -248, -88, -132, -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368,
    -480, -216, -92, 1688, -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332,
    204, 760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728, -520, 276, -188,
    80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272, -404, 180, 816, -796, 48, 152, 484, -88,
    -216, 988, 696, 188, -528, 648, -116, -180, 316, 476, 12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576,
    260, -352, 120, -16, -136, -260, -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80,
    920, 160, 228, 724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84, 192,
    852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484, -888, 64, 184, 352, 600,
    460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848, -344, 224, 208, -200, 484, 128, -20, 272, -468,
    -840, 384, 256, -720, -520, -464, -580, 112, -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84,
    200, -152, 0, -144, 584, 280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300, 24, -676, -596, 132,
    1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488, 424, 24, 756, -220, -192, 716, 120, 920,
    688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424, 888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628,
    -348, -228, -380, 1012, -452, -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344,
    428, 548, -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284, -12, 824,
    164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676, 1068, 424, -56, -360, 468, -156,
    720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648, -620, -684, -24, -376, -384, -108, -920, -1032, 768,
    180, -264, -508, -1268, -260, -60, 300, -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376,
    -56, -4, -216, -32, 836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228, 132, -48, -220, 868,
    -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376, -108, -92, 788, 368, -56, 544, 400, -672,
    -420, 728, 16, 320, 44, -284, -380, -796, 488, 132, 204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692,
    -200, -56, 276, -88, 484, -324, 948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236,
    -288, -116, 260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676, -396,
    -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212, 252, -660, -708, 484, -444,
    -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228, -288, 504, 560, -328, -88, 288, -1008, 460,
    -228, 468, -836, -196, 76, 388, 232, 412, -1168, -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168,
    -608, 448, 160, -532, -272, 28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524,
    -572, -556, -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488, -192,
    -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12, -156, -368, -340, 432, 252,
    -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716, 592, -80, 436, 360, 4, -248, 160, 516, 584, 732,
    44, -468, -280, -292, -156, -588, 28, 308, 912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300,
    -212, -1144, 32, -724, 800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552, -52, -100, -656, 208,
    -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272, -876, -664, -648, -332, -136, 16, 12, 1152,
    -28, 332, -536, 320, -672, -460, -316, 532, -260, 228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92,
    356, 404, -408, 252, 196, -176, -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592,
    952, 128, 656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508, -476, 324,
    -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348, -288, -276, -196, -500, 852,
    -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12, 168, 1020, 512, -552, 924, -148, 716, 188, 164,
    -340, -520, -184, 880, -152, -680, -208, -1156, -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676,
    -32, -232, -280, -224, 96, 568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428,
    -484,
];
