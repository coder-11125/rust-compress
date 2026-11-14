//! Writes gzip / zlib-with-dictionary streams for cross-checking against other implementations.
use rc_deflate::*;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let dict: Vec<u8> = (0..4000u32).map(|i| b"abcdefghij klmnop"[(i * i % 17) as usize]).collect();
    let data: Vec<u8> = [&dict[100..2500], b"unique tail 12345".as_slice(), &dict[5..900]].concat();
    std::fs::write(format!("{dir}/dict.bin"), &dict).unwrap();
    std::fs::write(format!("{dir}/data.bin"), &data).unwrap();
    std::fs::write(format!("{dir}/ours.gz"), gzip_compress(&data, 6)).unwrap();
    std::fs::write(format!("{dir}/ours.zd"), zlib_compress_dict(&data, 6, &dict)).unwrap();
    if let Some(p) = a.get(2) {
        let z = std::fs::read(p).unwrap();
        assert_eq!(zlib_decompress_dict(&z, &dict, 0, 1 << 20).unwrap(), data);
        println!("decoded zlib-dict stream from reference zlib");
    }
}
