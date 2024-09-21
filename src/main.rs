use std::arch::x86_64::{
    __m256, __m256i, _mm256_add_epi32, _mm256_broadcast_ss, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
};

fn pack_b(vals: &[i8], b_rows: usize, b_cols: usize) -> Vec<i8> {
    assert!(b_cols % 8 == 0);
    assert!(b_cols % 4 == 0);

    let b_row_stride = b_cols;

    let mut out = Vec::new();
    for col_block in 0..b_cols / 8 {
        for row_block in 0..b_rows / 4 {
            for col_off in 0..8 {
                for row_off in 0..4 {
                    let y = row_block * 4 + row_off;
                    let x = col_block * 8 + col_off;
                    out.push(vals[y * b_row_stride + x]);
                }
            }
        }
    }
    out
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_int(c: &mut [i32], a: &[u8], b: &[i8], m: usize, n: usize, k: usize) {
    const MR: usize = 6;
    assert!(k % 4 == 0);
    assert!(m % MR == 0);
    assert!(n % 8 == 0);

    let col_block_size = 8;
    let row_block_size = MR;
    let depth_block_size = 4;
    let n_col_blocks = n / col_block_size;
    let n_row_blocks = m / MR;
    let n_depth_blocks = k / depth_block_size;

    let b_ptr = b.as_ptr();
    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let a_row_stride = k;
    let c_row_stride = n;

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * 8 * 4;

        for row_block in 0..n_row_blocks {
            let mut tmp = [_mm256_setzero_si256(); MR];
            let one = _mm256_set1_epi16(1);
            for k_block in 0..n_depth_blocks {
                let bv = _mm256_loadu_si256(
                    b_ptr.add(b_off + k_block * size_of::<__m256i>()) as *const __m256i
                );
                for i in 0..MR {
                    let av = _mm256_broadcast_ss(std::mem::transmute::<*const u8, &f32>(
                        a_ptr.add((row_block * MR + i) * a_row_stride + k_block * 4),
                    ));
                    let av = std::mem::transmute::<__m256, __m256i>(av);

                    let z = _mm256_maddubs_epi16(av, bv);
                    let z = _mm256_madd_epi16(z, one);
                    tmp[i] = _mm256_add_epi32(tmp[i], z);
                }
            }
            for i in 0..MR {
                let c_off =
                    (row_block * row_block_size + i) * c_row_stride + (col_block * col_block_size);
                let z = _mm256_loadu_si256(c_ptr.add(c_off) as *const __m256i);
                let z = _mm256_add_epi32(tmp[i], z);
                _mm256_storeu_si256(c_ptr.add(c_off) as *mut __m256i, z);
            }
        }
    }
}

fn reference_matmul_int(a: &[u8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0; m * n];
    for row in 0..m {
        for col in 0..n {
            for depth in 0..k {
                out[row * n + col] += a[row * k + depth] as i32 * b[depth * n + col] as i32;
            }
        }
    }
    out
}

fn print_mat<E: std::fmt::Display>(mat: &[E], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            print!("{} ", mat[row * n + col]);
        }
        println!("");
    }
}

fn main() {
    let m = 6 * 2;
    let n = 8 * 2;
    let k = 4 * 2;

    let a: Vec<u8> = (0..m * k).map(|x| x as u8).collect();
    let b: Vec<i8> = (0..k * n).map(|x| x as i8).collect();
    let mut c = vec![0i32; m * n];

    let packed_b = pack_b(&b, k, n);

    let start = std::time::Instant::now();
    let n_iters = 1;
    for _ in 0..n_iters {
        unsafe { matmul_int(&mut c, &a, &packed_b, m, n, k) };
    }

    let duration = start.elapsed();
    let flops = (2 * m * n * k * n_iters as usize) as f64 / duration.as_secs_f64();
    let gflops = flops / (10f64).powi(9);
    let duration_ms = duration.as_secs_f64() * 1000.0;

    println!(
        "m {} n {} k {} . Duration {}ms. GFLOPS {}",
        m, n, k, duration_ms, gflops,
    );

    let ref_c = reference_matmul_int(&a, &b, m, n, k);

    println!("\n\nOptimized:");
    print_mat(&c, m, n);

    println!("");
    println!("Reference:");
    print_mat(&ref_c, m, n);
}
