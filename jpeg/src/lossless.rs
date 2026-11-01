//! Lossless JPEG (SOF3, T.81 Annex H) writer: recode the samples of a lossless file without changing any of them.
//!
//! Every sample is predicted from its neighbours; the seven predictors are all tried (in parallel) and costed from
//! the symbol statistics of real optimized Huffman tables, and the cheapest is written. The difference to the
//! prediction is coded modulo 2^16 as in the standard. The reading side is [`super::decode`].

use super::arith::{Conditioning, LosslessEncoder};
use super::decode::predict_sample;
use super::encode::{HuffTable, MsbWriter, optimal_table, segment, write_dht, write_metadata};
use rayon::prelude::*;
use rc_core::error::{Error, Result};
use rc_core::image::Metadata;

/// One component's samples as decoded: `stride` per row, padded to whole MCUs, before the point transform is undone.
pub struct LosslessComp {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub stride: usize,
    pub samples: Vec<u16>,
}

pub struct LosslessFrame {
    pub width: usize,
    pub height: usize,
    /// Bits per sample, 2 to 16.
    pub precision: u8,
    pub point_transform: u32,
    pub adobe_transform: Option<u8>,
    pub meta: Metadata,
    pub comps: Vec<LosslessComp>,
}

#[derive(Clone, Debug, Default)]
pub struct LosslessOptions {
    /// Arithmetic instead of Huffman coding of the differences (SOF11).
    pub arithmetic: bool,
    /// Arithmetic conditioning bounds (written as a DAC marker when they differ from the defaults).
    pub conditioning: Conditioning,
    /// Predictor 1 to 7, or `None` to try them all and keep the smallest.
    pub predictor: Option<u8>,
    /// Write a restart marker every this many MCU rows (0 for none).
    pub restart_rows: usize,
}

/// A lossless file recoded.
pub struct Recoded {
    pub bytes: Vec<u8>,
    pub predictor: u8,
}

enum Event {
    /// A restart interval ends here.
    Restart,
    /// The coded difference (-32768..=32767) of the sample at (`x`, `y`) of component `ci`; `first_row` is the first
    /// sample row of the current restart interval.
    Diff { ci: usize, x: usize, y: usize, first_row: usize, d: i32 },
}

impl LosslessFrame {
    fn mcu_grid(&self) -> (usize, usize) {
        let hmax = self.comps.iter().map(|c| c.h).max().unwrap_or(1);
        let vmax = self.comps.iter().map(|c| c.v).max().unwrap_or(1);
        (self.width.div_ceil(hmax), self.height.div_ceil(vmax))
    }

    /// Visit every sample in scan order with the difference between it and its prediction.
    fn for_each_diff(&self, predictor: u8, restart_rows: usize, mut f: impl FnMut(Event)) {
        let (mcus_x, mcus_y) = self.mcu_grid();
        let start = 1i32 << (self.precision as u32 - self.point_transform - 1);
        let mut first_row = vec![0usize; self.comps.len()];
        for my in 0..mcus_y {
            if restart_rows > 0 && my > 0 && my % restart_rows == 0 {
                f(Event::Restart);
                for (ci, c) in self.comps.iter().enumerate() {
                    first_row[ci] = my * c.v;
                }
            }
            for mx in 0..mcus_x {
                for (ci, c) in self.comps.iter().enumerate() {
                    for by in 0..c.v {
                        for bx in 0..c.h {
                            let (x, y) = (mx * c.h + bx, my * c.v + by);
                            let pred = predict_sample(&c.samples, c.stride, x, y, first_row[ci], predictor, start);
                            let diff = (c.samples[y * c.stride + x] as i32 - pred) & 0xFFFF;
                            f(Event::Diff { ci, x, y, first_row: first_row[ci], d: diff as u16 as i16 as i32 });
                        }
                    }
                }
            }
        }
    }

    /// Estimated size in bytes of the entropy-coded data and tables for one predictor.
    fn cost(&self, predictor: u8) -> u64 {
        let mut freq = [[0u32; 256]; 2];
        let mut extra = 0u64;
        self.for_each_diff(predictor, 0, |e| {
            if let Event::Diff { ci, d, .. } = e {
                let t = category(d);
                freq[ci.min(1)][t as usize] += 1;
                extra += if t == 16 { 0 } else { t as u64 };
            }
        });
        let mut bits = extra;
        let mut tables = 0u64;
        for f in freq.iter().filter(|f| f.iter().any(|&n| n > 0)) {
            let t = optimal_table(f);
            bits += f.iter().zip(&t.size).map(|(&n, &s)| n as u64 * s as u64).sum::<u64>();
            tables += 5 + 16 + t.vals.len() as u64;
        }
        bits.div_ceil(8) + tables
    }

    /// The smallest file for this frame.
    pub fn write(&self, opts: &LosslessOptions) -> Result<Recoded> {
        let predictor = match opts.predictor {
            Some(p) if (1..=7).contains(&p) => p,
            Some(_) => return Err(Error::invalid("lossless predictor must be 1 to 7")),
            None => {
                let costs: Vec<(u64, u8)> = (1..=7u8).into_par_iter().map(|p| (self.cost(p), p)).collect();
                costs.into_iter().min().expect("seven candidates").1
            }
        };
        let sof = if opts.arithmetic { 0xCB } else { 0xC3 };
        let arith = opts.arithmetic.then_some(&opts.conditioning);
        let segments = self.frame_segments(predictor, sof, opts.restart_rows, arith)?;
        let mut out = vec![0xFF, 0xD8];
        if let Some(t) = self.adobe_transform {
            segment(&mut out, 0xEE, &[b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, t]);
        }
        write_metadata(&mut out, &self.meta);
        out.extend_from_slice(&segments);
        out.extend_from_slice(&[0xFF, 0xD9]);
        Ok(Recoded { bytes: out, predictor })
    }

    /// The frame header, Huffman tables, restart interval, scan header and entropy-coded data, without the markers
    /// that frame a whole file. `predictor` 0 (no prediction) is for differential frames (`sof` 0xC7) of a
    /// hierarchical image, where each sample is itself a difference.
    pub(super) fn frame_segments(
        &self,
        predictor: u8,
        sof: u8,
        restart_rows: usize,
        arithmetic: Option<&Conditioning>,
    ) -> Result<Vec<u8>> {
        let nc = self.comps.len();
        if nc == 0 || self.width == 0 || self.height == 0 || self.width > 65535 || self.height > 65535 {
            return Err(Error::unsupported("lossless JPEG dimensions"));
        }
        let (mcus_x, _) = self.mcu_grid();

        let (data, tables): (Vec<u8>, Vec<HuffTable>) = match arithmetic {
            None => {
                // Pass 1: the symbol statistics give the Huffman tables.
                let mut freq = [[0u32; 256]; 2];
                self.for_each_diff(predictor, restart_rows, |e| {
                    if let Event::Diff { ci, d, .. } = e {
                        freq[ci.min(1)][category(d) as usize] += 1;
                    }
                });
                let tables: Vec<HuffTable> = (0..nc.min(2)).map(|t| optimal_table(&freq[t])).collect();
                // Pass 2: the entropy-coded data.
                let mut w = MsbWriter { out: Vec::with_capacity(self.width * self.height * nc), acc: 0, n: 0 };
                let mut rst = 0u8;
                self.for_each_diff(predictor, restart_rows, |e| match e {
                    Event::Restart => {
                        w.finish();
                        w.out.extend_from_slice(&[0xFF, 0xD0 + rst]);
                        rst = (rst + 1) & 7;
                    }
                    Event::Diff { ci, d, .. } => {
                        let t = category(d);
                        let table = &tables[ci.min(1)];
                        w.put(table.code[t as usize] as u32, table.size[t as usize] as u32);
                        if t > 0 && t < 16 {
                            let v = if d < 0 { (d - 1) as u32 } else { d as u32 };
                            w.put(v & ((1 << t) - 1), t);
                        }
                    }
                });
                w.finish();
                (w.out, tables)
            }
            Some(cond) => {
                let mut enc = LosslessEncoder::new(cond);
                let mut data = Vec::new();
                let mut rst = 0u8;
                // The differences already coded condition the statistics of their right and lower neighbours.
                let mut diffs: Vec<Vec<i16>> = self.comps.iter().map(|c| vec![0i16; c.samples.len()]).collect();
                self.for_each_diff(predictor, restart_rows, |e| match e {
                    Event::Restart => {
                        let done = std::mem::replace(&mut enc, LosslessEncoder::new(cond));
                        data.extend_from_slice(&done.finish());
                        data.extend_from_slice(&[0xFF, 0xD0 + rst]);
                        rst = (rst + 1) & 7;
                    }
                    Event::Diff { ci, x, y, first_row, d } => {
                        let stride = self.comps[ci].stride;
                        let da = if x == 0 { 0 } else { diffs[ci][y * stride + x - 1] as i32 };
                        let db = if y == first_row { 0 } else { diffs[ci][(y - 1) * stride + x] as i32 };
                        enc.diff(ci.min(1), da, db, d);
                        diffs[ci][y * stride + x] = d as i16;
                    }
                });
                data.extend_from_slice(&enc.finish());
                (data, Vec::new())
            }
        };

        let mut out = Vec::with_capacity(data.len() + 256);
        let mut header = vec![self.precision];
        header.extend_from_slice(&(self.height as u16).to_be_bytes());
        header.extend_from_slice(&(self.width as u16).to_be_bytes());
        header.push(nc as u8);
        for c in &self.comps {
            header.extend_from_slice(&[c.id, (c.h << 4 | c.v) as u8, 0]);
        }
        segment(&mut out, sof, &header);
        for (t, table) in tables.iter().enumerate() {
            write_dht(&mut out, t as u8, table);
        }
        if let Some(cond) = arithmetic {
            let dflt = Conditioning::default();
            let mut dac = Vec::new();
            for t in 0..nc.min(2) {
                if cond.dc_l[t] != dflt.dc_l[t] || cond.dc_u[t] != dflt.dc_u[t] {
                    dac.extend_from_slice(&[t as u8, cond.dc_u[t] << 4 | cond.dc_l[t]]);
                }
            }
            if !dac.is_empty() {
                segment(&mut out, 0xCC, &dac);
            }
        }
        if restart_rows > 0 {
            let interval = (restart_rows * mcus_x).min(0xFFFF) as u16;
            segment(&mut out, 0xDD, &interval.to_be_bytes());
        }
        let mut sos = vec![nc as u8];
        for (ci, c) in self.comps.iter().enumerate() {
            sos.extend_from_slice(&[c.id, (ci.min(1) as u8) << 4]);
        }
        sos.extend_from_slice(&[predictor, 0, self.point_transform as u8]);
        segment(&mut out, 0xDA, &sos);
        out.extend_from_slice(&data);
        Ok(out)
    }
}

/// Difference category (Table H.2): the number of bits of the magnitude, 16 for -32768.
fn category(d: i32) -> u32 {
    32 - d.unsigned_abs().leading_zeros()
}

/// Recode a lossless JPEG file: the same samples, the smallest predictor and Huffman tables.
pub fn recompress(data: &[u8]) -> Result<Recoded> {
    super::decode::parse(data)?.lossless_frame()?.write(&LosslessOptions::default())
}
