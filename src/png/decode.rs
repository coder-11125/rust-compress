use super::SIGNATURE;
use crate::deflate::zlib_decompress;
use crate::error::{Error, Result};
use crate::image::{ColorType, Image, MAX_PIXELS, Metadata};
use crate::util::checksum::crc32;

struct Header {
    width: usize,
    height: usize,
    depth: u8,
    color_type: u8,
    interlaced: bool,
}

impl Header {
    fn channels_in(&self) -> usize {
        match self.color_type {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        }
    }

    fn bits_per_pixel(&self) -> usize {
        self.channels_in() * self.depth as usize
    }

    fn row_bytes(&self, width: usize) -> usize {
        (width * self.bits_per_pixel()).div_ceil(8)
    }
}

const ADAM7: [(usize, usize, usize, usize); 7] =
    [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

fn parse_header(body: &[u8]) -> Result<Header> {
    if body.len() != 13 {
        return Err(Error::invalid("bad IHDR length"));
    }
    let width = u32::from_be_bytes(body[0..4].try_into().unwrap()) as u64;
    let height = u32::from_be_bytes(body[4..8].try_into().unwrap()) as u64;
    let (depth, color_type, compression, filter, interlace) = (body[8], body[9], body[10], body[11], body[12]);
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_PIXELS || width > (1 << 24) || height > (1 << 24)
    {
        return Err(Error::invalid("unreasonable image dimensions"));
    }
    let ok = match color_type {
        0 => matches!(depth, 1 | 2 | 4 | 8 | 16),
        3 => matches!(depth, 1 | 2 | 4 | 8),
        2 | 4 | 6 => matches!(depth, 8 | 16),
        _ => false,
    };
    if !ok {
        return Err(Error::invalid("invalid color type / bit depth combination"));
    }
    if compression != 0 || filter != 0 || interlace > 1 {
        return Err(Error::unsupported("PNG compression/filter/interlace method"));
    }
    Ok(Header { width: width as usize, height: height as usize, depth, color_type, interlaced: interlace == 1 })
}

fn pass_dims(h: &Header) -> Vec<(usize, usize, usize, usize, usize, usize)> {
    // (x0, y0, dx, dy, pass_w, pass_h)
    if !h.interlaced {
        return vec![(0, 0, 1, 1, h.width, h.height)];
    }
    ADAM7
        .iter()
        .map(|&(x0, y0, dx, dy)| {
            let pw = if h.width > x0 { (h.width - x0).div_ceil(dx) } else { 0 };
            let ph = if h.height > y0 { (h.height - y0).div_ceil(dy) } else { 0 };
            (x0, y0, dx, dy, pw, ph)
        })
        .collect()
}

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

/// Undo a scanline filter in place. `prev` is the already-unfiltered previous row.
fn unfilter_row(filter: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Result<()> {
    let n = cur.len();
    match filter {
        0 => {}
        1 => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..n {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(prev[i] >> 1);
            }
            for i in bpp..n {
                let avg = ((cur[i - bpp] as u16 + prev[i] as u16) >> 1) as u8;
                cur[i] = cur[i].wrapping_add(avg);
            }
        }
        4 => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
            for i in bpp..n {
                let p = paeth(cur[i - bpp], prev[i], prev[i - bpp]);
                cur[i] = cur[i].wrapping_add(p);
            }
        }
        _ => return Err(Error::invalid("bad PNG filter type")),
    }
    Ok(())
}

/// Unfilter `rows` scanlines (each preceded by a filter byte) in place.
fn unfilter_pass(buf: &mut [u8], rows: usize, row_bytes: usize, bpp: usize) -> Result<()> {
    let stride = row_bytes + 1;
    let zero = vec![0u8; row_bytes];
    for y in 0..rows {
        let (before, rest) = buf.split_at_mut(y * stride);
        let prev: &[u8] = if y == 0 { &zero } else { &before[(y - 1) * stride + 1..] };
        let row = &mut rest[..stride];
        let filter = row[0];
        unfilter_row(filter, &mut row[1..], &prev[..row_bytes], bpp)?;
    }
    Ok(())
}

struct Ctx {
    depth: u8,
    color_type: u8,
    palette: Vec<[u8; 3]>,
    palette_alpha: Vec<u8>,
    key: Option<[u16; 3]>,
    out_color: ColorType,
    out_depth: u8,
}

#[inline(always)]
fn sample(row: &[u8], depth: u8, idx: usize) -> u16 {
    match depth {
        16 => u16::from_be_bytes([row[idx * 2], row[idx * 2 + 1]]),
        8 => row[idx] as u16,
        d => {
            let per_byte = 8 / d as usize;
            let byte = row[idx / per_byte];
            let shift = 8 - d as usize * (idx % per_byte + 1);
            ((byte >> shift) & ((1u8 << d) - 1)) as u16
        }
    }
}

#[inline(always)]
fn put(out: &mut [u8], at: &mut usize, v: u16, out_depth: u8) {
    if out_depth == 16 {
        out[*at..*at + 2].copy_from_slice(&v.to_be_bytes());
        *at += 2;
    } else {
        out[*at] = v as u8;
        *at += 1;
    }
}

fn convert_pixel_row(row: &[u8], pw: usize, ctx: &Ctx, mut emit: impl FnMut(usize, &[u16; 4], usize)) {
    let max_out: u16 = if ctx.out_depth == 16 { 65535 } else { 255 };
    let ch_out = ctx.out_color.channels();
    for x in 0..pw {
        let mut px = [0u16; 4];
        match ctx.color_type {
            0 => {
                let v = sample(row, ctx.depth, x);
                px[0] = if ctx.depth < 8 { (v as u32 * 255 / ((1u32 << ctx.depth) - 1)) as u16 } else { v };
                if let Some(k) = ctx.key {
                    px[1] = if v == k[0] { 0 } else { max_out };
                }
            }
            2 => {
                let r = sample(row, ctx.depth, x * 3);
                let g = sample(row, ctx.depth, x * 3 + 1);
                let b = sample(row, ctx.depth, x * 3 + 2);
                px[..3].copy_from_slice(&[r, g, b]);
                if let Some(k) = ctx.key {
                    px[3] = if [r, g, b] == k { 0 } else { max_out };
                }
            }
            3 => {
                let i = sample(row, ctx.depth, x) as usize;
                let c = ctx.palette.get(i).copied().unwrap_or([0, 0, 0]);
                px[..3].copy_from_slice(&[c[0] as u16, c[1] as u16, c[2] as u16]);
                px[3] = ctx.palette_alpha.get(i).copied().unwrap_or(255) as u16;
            }
            4 => {
                px[0] = sample(row, ctx.depth, x * 2);
                px[1] = sample(row, ctx.depth, x * 2 + 1);
            }
            _ => {
                for (c, p) in px.iter_mut().enumerate() {
                    *p = sample(row, ctx.depth, x * 4 + c);
                }
            }
        }
        emit(x, &px, ch_out);
    }
}

pub fn decode(data: &[u8]) -> Result<Image> {
    if data.len() < 8 || data[..8] != SIGNATURE {
        return Err(Error::invalid("not a PNG file"));
    }
    let mut pos = 8;
    let mut header: Option<Header> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut meta = Metadata::default();
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let ty: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
        if len > data.len() - pos - 12 {
            return Err(Error::invalid("truncated PNG chunk"));
        }
        let body = &data[pos + 8..pos + 8 + len];
        let crc = u32::from_be_bytes(data[pos + 8 + len..pos + 12 + len].try_into().unwrap());
        let ancillary = ty[0] & 0x20 != 0;
        if crc32(&data[pos + 4..pos + 8 + len]) != crc && !ancillary {
            return Err(Error::invalid(format!("CRC mismatch in {} chunk", String::from_utf8_lossy(&ty))));
        }
        pos += 12 + len;
        match &ty {
            b"IHDR" => {
                if header.is_some() {
                    return Err(Error::invalid("duplicate IHDR"));
                }
                header = Some(parse_header(body)?);
            }
            _ if header.is_none() => return Err(Error::invalid("IHDR must be first chunk")),
            b"PLTE" => {
                if !body.len().is_multiple_of(3) || body.is_empty() || body.len() / 3 > 256 {
                    return Err(Error::invalid("bad PLTE"));
                }
                palette = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"tRNS" => trns = body.to_vec(),
            b"IDAT" => idat.extend_from_slice(body),
            // Re-encoding would silently drop every frame but the first.
            b"acTL" => return Err(Error::unsupported("animated PNG (APNG)")),
            b"iCCP" => {
                // name (1-79 bytes) NUL, compression method, zlib-compressed profile.
                if let Some(nul) = body.iter().position(|&b| b == 0)
                    && body.len() > nul + 2
                    && body[nul + 1] == 0
                    && let Ok(profile) = zlib_decompress(&body[nul + 2..], 0, 1 << 24)
                {
                    meta.icc = Some(profile);
                }
            }
            b"sRGB" | b"gAMA" | b"cHRM" => meta.png_color.push((ty, body.to_vec())),
            b"IEND" => break,
            _ => {
                if !ancillary {
                    return Err(Error::unsupported(format!("critical chunk {}", String::from_utf8_lossy(&ty))));
                }
            }
        }
    }
    let h = header.ok_or_else(|| Error::invalid("missing IHDR"))?;
    if idat.is_empty() {
        return Err(Error::invalid("missing IDAT"));
    }
    if h.color_type == 3 && palette.is_empty() {
        return Err(Error::invalid("palette image without PLTE"));
    }

    let passes = pass_dims(&h);
    let mut raw_size = 0usize;
    for &(_, _, _, _, pw, ph) in &passes {
        if pw > 0 && ph > 0 {
            raw_size += ph * (1 + h.row_bytes(pw));
        }
    }
    // DEFLATE cannot expand more than ~1032:1, so a header promising far more pixel data than the
    // stream could hold is bogus (and must not trigger a huge allocation).
    if raw_size as u64 > (idat.len() as u64 + 64) * 1100 {
        return Err(Error::invalid("image dimensions exceed what the compressed data can contain"));
    }
    let mut raw = zlib_decompress(&idat, raw_size, raw_size)?;
    if raw.len() < raw_size {
        return Err(Error::invalid("image data shorter than expected"));
    }
    drop(idat);

    // Transparency handling and output format.
    let mut key = None;
    let mut palette_alpha = Vec::new();
    match h.color_type {
        0 if trns.len() >= 2 => key = Some([u16::from_be_bytes([trns[0], trns[1]]), 0, 0]),
        2 if trns.len() >= 6 => {
            key = Some([
                u16::from_be_bytes([trns[0], trns[1]]),
                u16::from_be_bytes([trns[2], trns[3]]),
                u16::from_be_bytes([trns[4], trns[5]]),
            ])
        }
        3 => palette_alpha = trns.clone(),
        _ => {}
    }
    let pal_has_alpha = palette_alpha.iter().any(|&a| a != 255);
    let out_color = match h.color_type {
        0 => {
            if key.is_some() {
                ColorType::GrayAlpha
            } else {
                ColorType::Gray
            }
        }
        2 => {
            if key.is_some() {
                ColorType::Rgba
            } else {
                ColorType::Rgb
            }
        }
        3 => {
            if pal_has_alpha {
                ColorType::Rgba
            } else {
                ColorType::Rgb
            }
        }
        4 => ColorType::GrayAlpha,
        _ => ColorType::Rgba,
    };
    let out_depth = if h.depth == 16 { 16 } else { 8 };
    let ctx = Ctx { depth: h.depth, color_type: h.color_type, palette, palette_alpha, key, out_color, out_depth };

    let bytes_out = out_depth as usize / 8;
    let bpp_out = out_color.channels() * bytes_out;
    let mut out = vec![0u8; h.width * h.height * bpp_out];
    let bpp_filter = h.bits_per_pixel().div_ceil(8).max(1);

    // Fast path: samples already in output layout.
    let direct = !h.interlaced && h.depth >= 8 && h.color_type != 3 && key.is_none();

    let mut off = 0;
    for &(x0, y0, dx, dy, pw, ph) in &passes {
        if pw == 0 || ph == 0 {
            continue;
        }
        let rb = h.row_bytes(pw);
        let pass_len = ph * (rb + 1);
        let pass = &mut raw[off..off + pass_len];
        off += pass_len;
        unfilter_pass(pass, ph, rb, bpp_filter)?;
        if direct {
            for y in 0..ph {
                let src = &pass[y * (rb + 1) + 1..y * (rb + 1) + 1 + rb];
                out[y * rb..(y + 1) * rb].copy_from_slice(src);
            }
            continue;
        }
        for y in 0..ph {
            let row = &pass[y * (rb + 1) + 1..(y + 1) * (rb + 1)];
            let oy = y0 + y * dy;
            convert_pixel_row(row, pw, &ctx, |x, px, ch| {
                let ox = x0 + x * dx;
                let mut at = (oy * h.width + ox) * bpp_out;
                for &v in &px[..ch] {
                    put(&mut out, &mut at, v, out_depth);
                }
            });
        }
    }
    let mut img = Image::new(h.width as u32, h.height as u32, out_color, out_depth, out)?;
    img.meta = meta;
    Ok(img)
}
