use rc_deflate::*;

fn sample(n: usize) -> Vec<u8> {
    let mut s = 12345u32;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            if i % 7 < 4 { b"the quick brown fox"[i % 19] } else { (s >> 24) as u8 }
        })
        .collect()
}

#[test]
fn gzip_roundtrip_and_members() {
    for level in [0, 1, 6, 9] {
        for n in [0, 1, 1000, 700_000] {
            let d = sample(n);
            assert_eq!(gzip_decompress(&gzip_compress(&d, level), 0, 1 << 24).unwrap(), d, "level {level} n {n}");
        }
    }
    let (a, b) = (sample(500), sample(900));
    let mut cat = gzip_compress(&a, 6);
    cat.extend(gzip_compress(&b, 6));
    cat.extend([0u8; 5]);
    assert_eq!(gzip_decompress(&cat, 0, 1 << 20).unwrap(), [a, b].concat());
}

#[test]
fn gzip_rejects_corruption() {
    let d = sample(3000);
    let g = gzip_compress(&d, 6);
    let mut bad = g.clone();
    let n = bad.len();
    bad[n - 6] ^= 1;
    assert!(gzip_decompress(&bad, 0, 1 << 20).is_err());
    let mut bad = g.clone();
    bad[n - 1] ^= 1;
    assert!(gzip_decompress(&bad, 0, 1 << 20).is_err());
    assert!(gzip_decompress(&g[..n - 3], 0, 1 << 20).is_err());
    assert!(gzip_decompress(&g, 0, 100).is_err());
    let mut junk = g.clone();
    junk.push(7);
    assert!(gzip_decompress(&junk, 0, 1 << 20).is_err());
}

#[test]
fn gzip_optional_header_fields() {
    let d = sample(2000);
    let body = deflate_compress(&d, 6);
    let mut g = vec![0x1F, 0x8B, 8, 4 | 8 | 16 | 2, 0, 0, 0, 0, 0, 3, 3, 0, 1, 2, 3];
    g.extend(b"name\0comment\0");
    let crc = rc_core::util::checksum::crc32(&g) as u16;
    g.extend(crc.to_le_bytes());
    g.extend(body);
    g.extend(rc_core::util::checksum::crc32(&d).to_le_bytes());
    g.extend((d.len() as u32).to_le_bytes());
    assert_eq!(gzip_decompress(&g, 0, 1 << 20).unwrap(), d);
}

#[test]
fn zlib_dictionary_roundtrip() {
    let dict = sample(5000);
    let data = [&dict[1000..3000], &sample(300)[..], &dict[10..800]].concat();
    for level in [1, 4, 6, 9] {
        let plain = zlib_compress(&data, level);
        let z = zlib_compress_dict(&data, level, &dict);
        assert!(z.len() < plain.len(), "dictionary should help at level {level}");
        assert_eq!(zlib_decompress_dict(&z, &dict, 0, 1 << 20).unwrap(), data);
        // needs the dictionary, and the right one
        assert!(zlib_decompress(&z, 0, 1 << 20).is_err());
        assert!(zlib_decompress_dict(&z, &sample(5001), 0, 1 << 20).is_err());
        // dictionary is ignored by streams that do not use one
        assert_eq!(zlib_decompress_dict(&plain, &dict, 0, 1 << 20).unwrap(), data);
    }
}

#[test]
fn dictionary_larger_than_window_and_multi_segment() {
    let dict = sample(100_000);
    let data = [&dict[70_000..], &sample(1_200_000)[..]].concat();
    let z = zlib_compress_dict(&data, 3, &dict);
    assert_eq!(zlib_decompress_dict(&z, &dict, 0, 1 << 22).unwrap(), data);
    let raw = deflate_compress_dict(&data, 7, &dict);
    let (out, _) = rc_deflate::inflate::inflate_with_end_dict(&raw, &dict, 0, 1 << 22).unwrap();
    assert_eq!(out, data);
    assert!(rc_deflate::inflate::inflate_with_end_dict(&raw, &dict, 0, 1000).is_err());
}

#[test]
fn empty_dictionary_gives_a_plain_zlib_stream() {
    let d = sample(2000);
    let z = zlib_compress_dict(&d, 6, &[]);
    assert_eq!(z[1] & 0x20, 0, "no FDICT flag without a dictionary");
    assert_eq!(zlib_decompress(&z, 0, 1 << 20).unwrap(), d);
    assert_eq!(zlib_decompress_dict(&z, &[], 0, 1 << 20).unwrap(), d);
}
