//! AVIF image sequences (`avis`): the movie box. Reading returns each track's sample table (where every sample
//! lives, how long it lasts, whether it is a sync sample); writing builds a `moov` for tracks whose samples are
//! already placed in `mdat`.
//!
//! Only what AVIF sequences use is understood: `av01` tracks with one sample entry, per-sample or per-chunk
//! layouts, `stts`/`stsc`/`stsz`/`stco`/`co64`/`stss`, `tref`/`auxl` for the alpha track and `elst` for looping.
//! Sizes and counts are checked against the box they came from, and the sample count is capped.

use super::container::{Av1Config, Cur, FourCc, bx, fbx, show, split_boxes};
use crate::error::{Error, Result};

/// Upper bound on samples per track (an hour of video at 60 fps is about 216 000).
pub const MAX_SAMPLES: usize = 1 << 20;

/// One sample: a byte range of the file and its presentation duration in track timescale ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    pub duration: u32,
    pub sync: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Track {
    pub id: u32,
    /// `pict` for colour, `auxv` for auxiliary video such as alpha.
    pub handler: FourCc,
    pub timescale: u32,
    pub width: u32,
    pub height: u32,
    pub config: Option<Av1Config>,
    /// The track this one is auxiliary to (`tref`/`auxl`).
    pub aux_for: Option<u32>,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub tracks: Vec<Track>,
    /// The movie plays forever (`elst` repeat flag).
    pub loop_forever: bool,
}

impl Sequence {
    /// The colour track: the first `pict` track.
    pub fn color(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| &t.handler == b"pict")
    }

    /// The alpha track of the colour track, if there is one.
    pub fn alpha(&self) -> Option<&Track> {
        let color = self.color()?;
        self.tracks.iter().find(|t| t.aux_for == Some(color.id) && &t.handler == b"auxv")
    }
}

fn child<'a>(boxes: &[(FourCc, &'a [u8])], kind: &FourCc) -> Option<&'a [u8]> {
    boxes.iter().find(|(k, _)| k == kind).map(|(_, b)| *b)
}

/// Parse the `moov` box of a file, if it has one. `data` is the whole file.
pub fn parse(data: &[u8]) -> Result<Option<Sequence>> {
    let top = split_boxes(data)?;
    let Some(&(_, moov)) = top.iter().find(|(k, _)| k == b"moov") else {
        return Ok(None);
    };
    let moov = split_boxes(moov)?;
    let mut tracks = Vec::new();
    let mut loop_forever = false;
    for (kind, body) in &moov {
        if kind != b"trak" {
            continue;
        }
        if tracks.len() >= 16 {
            return Err(Error::unsupported("more than 16 tracks"));
        }
        let (track, repeat) = parse_trak(body)?;
        loop_forever |= repeat;
        tracks.push(track);
    }
    if tracks.is_empty() {
        return Err(Error::invalid("moov without tracks"));
    }
    Ok(Some(Sequence { tracks, loop_forever }))
}

fn parse_trak(body: &[u8]) -> Result<(Track, bool)> {
    let trak = split_boxes(body)?;
    // tkhd: id and the 16.16 width and height.
    let tkhd = child(&trak, b"tkhd").ok_or_else(|| Error::invalid("trak without tkhd"))?;
    let mut c = Cur::new(tkhd);
    let (version, _) = c.full()?;
    if version == 1 {
        c.take(16)?;
    } else {
        c.take(8)?;
    }
    let id = c.u32()?;
    let tail = if version == 1 { 4 + 8 } else { 4 + 4 };
    c.take(tail)?;
    c.take(8 + 2 + 2 + 2 + 2 + 36)?;
    let (width, height) = (c.u32()? >> 16, c.u32()? >> 16);

    let aux_for = match child(&trak, b"tref") {
        Some(t) => {
            let mut found = None;
            for (k, b) in split_boxes(t)? {
                if &k == b"auxl" && b.len() >= 4 {
                    found = Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
            found
        }
        None => None,
    };
    let mut repeat = false;
    if let Some(edts) = child(&trak, b"edts")
        && let Some(elst) = child(&split_boxes(edts)?, b"elst")
    {
        let mut c = Cur::new(elst);
        let (_, flags) = c.full()?;
        repeat = flags & 1 != 0;
    }

    let mdia = split_boxes(child(&trak, b"mdia").ok_or_else(|| Error::invalid("trak without mdia"))?)?;
    let mdhd = child(&mdia, b"mdhd").ok_or_else(|| Error::invalid("mdia without mdhd"))?;
    let mut c = Cur::new(mdhd);
    let (version, _) = c.full()?;
    c.take(if version == 1 { 16 } else { 8 })?;
    let timescale = c.u32()?;
    if timescale == 0 {
        return Err(Error::invalid("mdhd: zero timescale"));
    }
    let hdlr = child(&mdia, b"hdlr").ok_or_else(|| Error::invalid("mdia without hdlr"))?;
    let mut c = Cur::new(hdlr);
    c.full()?;
    c.u32()?;
    let handler = c.fourcc()?;

    let minf = split_boxes(child(&mdia, b"minf").ok_or_else(|| Error::invalid("mdia without minf"))?)?;
    let stbl = split_boxes(child(&minf, b"stbl").ok_or_else(|| Error::invalid("minf without stbl"))?)?;
    let config = parse_stsd(child(&stbl, b"stsd").ok_or_else(|| Error::invalid("stbl without stsd"))?)?;
    let samples = parse_samples(&stbl)?;
    Ok((Track { id, handler, timescale, width, height, config, aux_for, samples }, repeat))
}

fn parse_stsd(stsd: &[u8]) -> Result<Option<Av1Config>> {
    let mut c = Cur::new(stsd);
    c.full()?;
    let n = c.u32()?;
    if n == 0 {
        return Err(Error::invalid("stsd without entries"));
    }
    let entry_size = c.u32()? as usize;
    let kind = c.fourcc()?;
    if entry_size < 8 || entry_size - 8 > c.left() {
        return Err(Error::invalid("stsd entry size out of range"));
    }
    let entry = c.take(entry_size - 8)?;
    if &kind != b"av01" {
        return Err(Error::unsupported(format!("sample entry '{}'", show(&kind))));
    }
    // Visual sample entry: 6 reserved + data_reference_index + 16 + width/height + resolutions + 4 + frame_count +
    // 32 compressor name + depth + pre_defined = 78 bytes before the child boxes.
    if entry.len() < 78 {
        return Err(Error::invalid("av01 sample entry too short"));
    }
    let kids = split_boxes(&entry[78..])?;
    child(&kids, b"av1C").map(Av1Config::parse).transpose()
}

fn parse_samples(stbl: &[(FourCc, &[u8])]) -> Result<Vec<Sample>> {
    let mut c = Cur::new(child(stbl, b"stsz").ok_or_else(|| Error::invalid("stbl without stsz"))?);
    c.full()?;
    let uniform = c.u32()?;
    let count = c.u32()? as usize;
    if count > MAX_SAMPLES {
        return Err(Error::unsupported("too many samples"));
    }
    let mut sizes = Vec::with_capacity(count);
    for _ in 0..count {
        sizes.push(if uniform != 0 { uniform } else { c.u32()? });
    }

    let mut c = Cur::new(child(stbl, b"stts").ok_or_else(|| Error::invalid("stbl without stts"))?);
    c.full()?;
    let entries = c.u32()? as usize;
    let mut durations: Vec<u32> = Vec::with_capacity(count);
    for _ in 0..entries {
        let (n, d) = (c.u32()? as usize, c.u32()?);
        if durations.len() + n > count {
            return Err(Error::invalid("stts covers more samples than stsz"));
        }
        durations.extend(std::iter::repeat_n(d, n));
    }
    durations.resize(count, 0);

    // Chunk offsets and how many samples each chunk holds.
    let (co_box, co64) = match child(stbl, b"stco") {
        Some(b) => (b, false),
        None => (child(stbl, b"co64").ok_or_else(|| Error::invalid("stbl without stco"))?, true),
    };
    let mut c = Cur::new(co_box);
    c.full()?;
    let chunks = c.u32()? as usize;
    if chunks > count.max(1) {
        return Err(Error::invalid("more chunks than samples"));
    }
    let mut chunk_off = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        chunk_off.push(if co64 { c.u64()? } else { c.u32()? as u64 });
    }
    let mut c = Cur::new(child(stbl, b"stsc").ok_or_else(|| Error::invalid("stbl without stsc"))?);
    c.full()?;
    let runs = c.u32()? as usize;
    let mut table: Vec<(u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk)
    for _ in 0..runs {
        let (first, per, _desc) = (c.u32()?, c.u32()?, c.u32()?);
        table.push((first, per));
    }
    let mut offsets = Vec::with_capacity(count);
    'chunks: for (ci, &base) in chunk_off.iter().enumerate() {
        let chunk_no = ci as u32 + 1;
        let per = table.iter().rev().find(|(first, _)| *first <= chunk_no).map_or(0, |&(_, p)| p) as usize;
        let mut off = base;
        for _ in 0..per {
            let i = offsets.len();
            if i >= count {
                break 'chunks;
            }
            offsets.push(off);
            off = off.checked_add(sizes[i] as u64).ok_or_else(|| Error::invalid("sample offset overflow"))?;
        }
    }
    if offsets.len() != count {
        return Err(Error::invalid("stsc and stco do not place every sample"));
    }

    let sync_all = child(stbl, b"stss").is_none();
    let mut sync = vec![sync_all; count];
    if let Some(stss) = child(stbl, b"stss") {
        let mut c = Cur::new(stss);
        c.full()?;
        let n = c.u32()? as usize;
        for _ in 0..n {
            let s = c.u32()? as usize;
            if s >= 1 && s <= count {
                sync[s - 1] = true;
            }
        }
    }
    Ok((0..count)
        .map(|i| Sample { offset: offsets[i], size: sizes[i], duration: durations[i], sync: sync[i] })
        .collect())
}

// ---------------------------------------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------------------------------------

/// One track to write: samples already sit in `mdat` at `offsets`.
pub struct TrackOut<'a> {
    pub id: u32,
    pub alpha: bool,
    pub config: &'a Av1Config,
    pub width: u32,
    pub height: u32,
    /// Boxes that follow `av1C` in the sample entry (`colr`).
    pub extra_entry_boxes: Vec<u8>,
    pub offsets: Vec<u64>,
    pub sizes: Vec<u32>,
    pub durations: Vec<u32>,
    pub sync: Vec<bool>,
    /// Track this one is auxiliary to.
    pub aux_for: Option<u32>,
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

fn trak(t: &TrackOut, timescale: u32, loop_forever: bool) -> Vec<u8> {
    let total: u64 = t.durations.iter().map(|&d| d as u64).sum();
    let dur32 = total.min(u32::MAX as u64) as u32;
    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&[0; 8]); // creation, modification
    tkhd.extend_from_slice(&t.id.to_be_bytes());
    tkhd.extend_from_slice(&[0; 4]);
    tkhd.extend_from_slice(&dur32.to_be_bytes());
    tkhd.extend_from_slice(&[0; 8 + 2 + 2]); // reserved, layer, alternate group
    tkhd.extend_from_slice(&0u16.to_be_bytes()); // volume
    tkhd.extend_from_slice(&[0; 2]);
    tkhd.extend_from_slice(&MATRIX);
    tkhd.extend_from_slice(&(t.width << 16).to_be_bytes());
    tkhd.extend_from_slice(&(t.height << 16).to_be_bytes());
    let mut out = fbx(b"tkhd", 0, 3, &tkhd);

    if let Some(main) = t.aux_for {
        out.extend(bx(b"tref", &bx(b"auxl", &main.to_be_bytes())));
    }
    // Looping: an edit covering the whole track, flagged to repeat when it should play forever.
    let mut elst = 1u32.to_be_bytes().to_vec();
    elst.extend_from_slice(&dur32.to_be_bytes());
    elst.extend_from_slice(&0i32.to_be_bytes());
    elst.extend_from_slice(&[0, 1, 0, 0]);
    out.extend(bx(b"edts", &fbx(b"elst", 0, loop_forever as u32, &elst)));

    let mut mdhd = vec![0; 8];
    mdhd.extend_from_slice(&timescale.to_be_bytes());
    mdhd.extend_from_slice(&dur32.to_be_bytes());
    mdhd.extend_from_slice(&[0x55, 0xc4, 0, 0]);
    let handler: &[u8; 4] = if t.alpha { b"auxv" } else { b"pict" };
    let mut hdlr = vec![0; 4];
    hdlr.extend_from_slice(handler);
    hdlr.extend_from_slice(&[0; 12]);
    hdlr.push(0);

    // Sample entry.
    let mut entry = vec![0; 6];
    entry.extend_from_slice(&1u16.to_be_bytes());
    entry.extend_from_slice(&[0; 16]);
    entry.extend_from_slice(&(t.width as u16).to_be_bytes());
    entry.extend_from_slice(&(t.height as u16).to_be_bytes());
    entry.extend_from_slice(&0x0048_0000u32.to_be_bytes());
    entry.extend_from_slice(&0x0048_0000u32.to_be_bytes());
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(&1u16.to_be_bytes());
    entry.extend_from_slice(&[0; 32]);
    entry.extend_from_slice(&0x0018u16.to_be_bytes());
    entry.extend_from_slice(&0xffffu16.to_be_bytes());
    entry.extend(bx(b"av1C", &t.config.to_bytes()));
    entry.extend_from_slice(&t.extra_entry_boxes);
    if t.alpha {
        let mut urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha".to_vec();
        urn.push(0);
        entry.extend(fbx(b"auxi", 0, 0, &urn));
    }
    let mut stsd = 1u32.to_be_bytes().to_vec();
    stsd.extend(bx(b"av01", &entry));

    // Durations as runs.
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in &t.durations {
        match runs.last_mut() {
            Some((n, last)) if *last == d => *n += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut stts = (runs.len() as u32).to_be_bytes().to_vec();
    for (n, d) in &runs {
        stts.extend_from_slice(&n.to_be_bytes());
        stts.extend_from_slice(&d.to_be_bytes());
    }
    // One sample per chunk keeps the layout independent of what sits between samples in `mdat`.
    let mut stsc = 1u32.to_be_bytes().to_vec();
    stsc.extend_from_slice(&u32s(&[1, 1, 1]));
    let mut stsz = 0u32.to_be_bytes().to_vec();
    stsz.extend_from_slice(&(t.sizes.len() as u32).to_be_bytes());
    stsz.extend_from_slice(&u32s(&t.sizes));
    let wide = t.offsets.iter().any(|&o| o > u32::MAX as u64);
    let mut co = (t.offsets.len() as u32).to_be_bytes().to_vec();
    for &o in &t.offsets {
        if wide {
            co.extend_from_slice(&o.to_be_bytes());
        } else {
            co.extend_from_slice(&(o as u32).to_be_bytes());
        }
    }
    let syncs: Vec<u32> = t.sync.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i as u32 + 1).collect();

    let mut stbl = fbx(b"stsd", 0, 0, &stsd);
    stbl.extend(fbx(b"stts", 0, 0, &stts));
    stbl.extend(fbx(b"stsc", 0, 0, &stsc));
    stbl.extend(fbx(b"stsz", 0, 0, &stsz));
    stbl.extend(fbx(if wide { b"co64" } else { b"stco" }, 0, 0, &co));
    if syncs.len() != t.sync.len() {
        let mut ss = (syncs.len() as u32).to_be_bytes().to_vec();
        ss.extend_from_slice(&u32s(&syncs));
        stbl.extend(fbx(b"stss", 0, 0, &ss));
    }
    let mut minf = fbx(b"vmhd", 0, 1, &[0; 8]);
    let dref = fbx(b"dref", 0, 0, &[&1u32.to_be_bytes()[..], &fbx(b"url ", 0, 1, &[])].concat());
    minf.extend(bx(b"dinf", &dref));
    minf.extend(bx(b"stbl", &stbl));
    let mut mdia = fbx(b"mdhd", 0, 0, &mdhd);
    mdia.extend(fbx(b"hdlr", 0, 0, &hdlr));
    mdia.extend(bx(b"minf", &minf));
    out.extend(bx(b"mdia", &mdia));
    bx(b"trak", &out)
}

const MATRIX: [u8; 36] =
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0];

/// Build the `moov` box for the given tracks. Track ids must be distinct and non-zero.
pub fn build_moov(tracks: &[TrackOut], timescale: u32, loop_forever: bool) -> Vec<u8> {
    let total: u64 = tracks.iter().map(|t| t.durations.iter().map(|&d| d as u64).sum()).max().unwrap_or(0);
    let mut mvhd = vec![0; 8];
    mvhd.extend_from_slice(&timescale.to_be_bytes());
    mvhd.extend_from_slice(&(total.min(u32::MAX as u64) as u32).to_be_bytes());
    mvhd.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    mvhd.extend_from_slice(&0x0100u16.to_be_bytes());
    mvhd.extend_from_slice(&[0; 10]);
    mvhd.extend_from_slice(&MATRIX);
    mvhd.extend_from_slice(&[0; 24]);
    let next_id = tracks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
    mvhd.extend_from_slice(&next_id.to_be_bytes());
    let mut moov = fbx(b"mvhd", 0, 0, &mvhd);
    for t in tracks {
        moov.extend(trak(t, timescale, loop_forever));
    }
    bx(b"moov", &moov)
}
