use rc_avif::av1::itx::*;
fn main() {
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as i32
    };
    for tx in 0u8..19 {
        let (w, h) = (TX_WIDTH[tx as usize], TX_HEIGHT[tx as usize]);
        if w == 64 || h == 64 {
            continue;
        }
        let mut worst = vec![];
        for tt in 0u8..16 {
            // valid type combos only: 32 sized => DCT/IDTX; 16 sized: no flips; skip invalid
            let max_dim = w.max(h);
            if max_dim >= 32 && !(tt == 0 || tt == 9) {
                continue;
            }
            if max_dim == 16 && matches!(tt, 4..=8 | 14 | 15) {
                continue;
            }
            let mut err_max = 0;
            for _ in 0..20 {
                let res: Vec<i32> = (0..w * h).map(|_| rnd() % 255 - 127).collect();
                let mut coefs = vec![0i32; w * h];
                forward_transform(&res, w, tx, tt, false, 8, &mut coefs);
                let mut dst = vec![128u16; w * h];
                inverse_transform_add(&mut coefs, tx, tt, false, 8, h, w, &mut dst, w);
                for i in 0..w * h {
                    err_max = err_max.max((dst[i] as i32 - 128 - res[i]).abs());
                }
            }
            worst.push((tt, err_max));
        }
        println!("tx {tx} {w}x{h}: {:?}", worst);
    }
}
