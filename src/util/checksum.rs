//! CRC-32 (IEEE, slice-by-8) and Adler-32.

const fn make_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut c = t[0][i];
        let mut s = 1;
        while s < 8 {
            c = t[0][(c & 0xFF) as usize] ^ (c >> 8);
            t[s][i] = c;
            s += 1;
        }
        i += 1;
    }
    t
}

static CRC_TABLES: [[u32; 256]; 8] = make_tables();

/// Incremental CRC-32 update. Start with `0` and feed chunks.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let t = &CRC_TABLES;
    let mut c = !crc;
    let (chunks, rest) = data.as_chunks::<8>();
    for ch in chunks {
        let lo = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) ^ c;
        let hi = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]);
        c = t[7][(lo & 0xFF) as usize]
            ^ t[6][((lo >> 8) & 0xFF) as usize]
            ^ t[5][((lo >> 16) & 0xFF) as usize]
            ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xFF) as usize]
            ^ t[2][((hi >> 8) & 0xFF) as usize]
            ^ t[1][((hi >> 16) & 0xFF) as usize]
            ^ t[0][(hi >> 24) as usize];
    }
    for &b in rest {
        c = t[0][((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    // Largest n such that 255*n*(n+1)/2 + (n+1)*(MOD-1) fits in u32.
    const NMAX: usize = 5552;
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(NMAX) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_incremental_matches_oneshot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let (a, b) = data.split_at(333);
        assert_eq!(crc32_update(crc32(a), b), crc32(&data));
    }

    #[test]
    fn adler32_known_vector() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn adler32_long_input_no_overflow() {
        let data = vec![255u8; 100_000];
        let mut a = 1u64;
        let mut b = 0u64;
        for &x in &data {
            a = (a + x as u64) % 65521;
            b = (b + a) % 65521;
        }
        assert_eq!(adler32(&data), ((b << 16) | a) as u32);
    }
}
