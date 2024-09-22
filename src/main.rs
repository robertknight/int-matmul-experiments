use std::arch::x86_64::{
    __m256,
    __m256i,
    _mm256_add_epi32,
    _mm256_broadcast_ss, //_mm256_extract_epi32,
    _mm256_loadu_si256,
    _mm256_madd_epi16,
    _mm256_maddubs_epi16,
    _mm256_maskload_epi32,
    _mm256_mullo_epi32,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_set1_epi8,
    _mm256_setr_epi32,
    _mm256_setzero_si256,
    _mm256_storeu_si256,
    _mm256_sub_epi32,
};

const MR: usize = 6;

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

fn pack_a(vals: &[u8], a_rows: usize, a_cols: usize) -> Vec<u8> {
    assert!(a_rows % MR == 0);
    assert!(a_cols % 4 == 0);

    let a_row_stride = a_cols;

    let mut out = Vec::new();
    for col_block in 0..a_cols / 4 {
        for row_block in 0..a_rows / MR {
            for row_off in 0..MR {
                for col_off in 0..4 {
                    let y = row_block * MR + row_off;
                    let x = col_block * 4 + col_off;
                    out.push(vals[y * a_row_stride + x]);
                }
            }
        }
    }

    out
}

/// Sum each group of 4 adjacent i8 values and produce i32 results.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn add_i8x32_i32x8(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    let tmp = _mm256_maddubs_epi16(one, x);
    let one_i16 = _mm256_set1_epi16(1);
    _mm256_madd_epi16(tmp, one_i16)
}

/// Sum each group of 4 adjacent u8 values and produce i32 results.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn add_u8x32_i32x8(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    let tmp = _mm256_maddubs_epi16(x, one);
    let one_i16 = _mm256_set1_epi16(1);
    _mm256_madd_epi16(tmp, one_i16)
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_int(
    c: &mut [i32],
    a: &[u8],
    b: &[i8],
    a_zero_point: u8,
    b_zero_point: i8,
    m: usize,
    n: usize,
    k: usize,
) {
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

    // let a_row_stride = k;
    let c_row_stride = n;

    // The value for each output cell is computed as:
    //
    // c = (a[0] - a_zero_point) * (b[0] - b_zero_point) + ...
    //   = a[0]b[0] - a[0] * b_zero_point - b[0] * a_zero_point + a_zero_point * b_zero_point
    //   = dot(a, b) - sum(a) * b_zero_point - sum(b) * a_zero_point + len(a) * a_zero_point * b_zero_point
    let c_init = _mm256_mullo_epi32(
        _mm256_set1_epi32(k as i32),
        _mm256_mullo_epi32(
            _mm256_set1_epi32(a_zero_point as i32),
            _mm256_set1_epi32(b_zero_point as i32),
        ),
    );

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * 8 * 4;

        for row_block in 0..n_row_blocks {
            let a_off = row_block * n_depth_blocks * MR * 4;

            // Sums along each row of `a`.
            let mut a_sum = _mm256_setzero_si256();
            // Sums along each column of `b`.
            let mut b_sum = _mm256_setzero_si256();

            let mut tmp = [c_init; MR];
            let one = _mm256_set1_epi16(1);

            let mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

            for k_block in 0..n_depth_blocks {
                let bv = _mm256_loadu_si256(
                    b_ptr.add(b_off + k_block * size_of::<__m256i>()) as *const __m256i
                );
                b_sum = _mm256_add_epi32(b_sum, add_i8x32_i32x8(bv));

                // An MR * 4 slice of A.
                // let mut a_vals = [0i32; 8];
                let a_vals = _mm256_maskload_epi32(
                    a_ptr.add(a_off + k_block * size_of::<__m256i>()) as *const i32,
                    mask,
                );

                for i in 0..MR {
                    let av = _mm256_broadcast_ss(std::mem::transmute::<*const u8, &f32>(
                        a_ptr.add(a_off + k_block * size_of::<__m256i>() + i), // a_ptr.add((row_block * MR + i) * a_row_stride + k_block * 4),
                    ));
                    let av = std::mem::transmute::<__m256, __m256i>(av);

                    let z = _mm256_maddubs_epi16(av, bv);
                    let z = _mm256_madd_epi16(z, one);
                    tmp[i] = _mm256_add_epi32(tmp[i], z);
                }

                a_sum = _mm256_add_epi32(a_sum, add_u8x32_i32x8(a_vals));
            }

            let a_sum = _mm256_mullo_epi32(a_sum, _mm256_set1_epi32(b_zero_point as i32));
            let a_sums = to_array::<i32, 8>(a_sum);
            let b_sum = _mm256_mullo_epi32(b_sum, _mm256_set1_epi32(a_zero_point as i32));

            for i in 0..MR {
                let c_off =
                    (row_block * row_block_size + i) * c_row_stride + (col_block * col_block_size);
                let z = _mm256_loadu_si256(c_ptr.add(c_off) as *const __m256i);
                let z = _mm256_add_epi32(tmp[i], z);
                let z = _mm256_sub_epi32(z, b_sum);
                let z = _mm256_sub_epi32(z, _mm256_set1_epi32(a_sums[i]));
                _mm256_storeu_si256(c_ptr.add(c_off) as *mut __m256i, z);
            }
        }
    }
}

#[allow(unused)]
fn reference_matmul_int(
    a: &[u8],
    b: &[i8],
    a_zero_point: u8,
    b_zero_point: i8,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<i32> {
    let mut out = vec![0; m * n];
    for row in 0..m {
        for col in 0..n {
            for depth in 0..k {
                let a_val = a[row * k + depth] as i32 - a_zero_point as i32;
                let b_val = b[depth * n + col] as i32 - b_zero_point as i32;
                out[row * n + col] += a_val * b_val;
            }
        }
    }
    out
}

/// Variant of `reference_matmul_int` where the zero-point handling has been
/// optimized to more closely match the way it is handled in the real
/// implementation.
#[allow(unused)]
fn reference_matmul_int_opt_zp(
    a: &[u8],
    b: &[i8],
    a_zero_point: u8,
    b_zero_point: i8,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<i32> {
    let mut out = vec![0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0;
            let mut a_sum = 0;
            let mut b_sum = 0;

            for depth in 0..k {
                let a_val = a[row * k + depth] as i32;
                let b_val = b[depth * n + col] as i32;
                a_sum += a_val;
                b_sum += b_val;
                acc += a_val * b_val;
            }

            acc = acc - (a_sum * b_zero_point as i32) - (b_sum * a_zero_point as i32)
                + (k as i32 * a_zero_point as i32 * b_zero_point as i32);

            out[row * n + col] += acc;
        }
    }
    out
}

#[allow(unused)]
fn print_mat<E: std::fmt::Display>(mat: &[E], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            print!("{} ", mat[row * n + col]);
        }
        println!("");
    }
}

fn to_array<T: Copy + Default, const N: usize>(x: __m256i) -> [T; N] {
    assert_eq!(size_of::<T>() * N, size_of::<__m256i>());
    let mut out = [T::default(); N];
    unsafe {
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, x);
    }
    out
}

#[allow(unused)]
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn test_avx_funcs() {
    let x = _mm256_set1_epi8(2);
    let x = add_i8x32_i32x8(x);
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, x);
    println!("add_i8x16_i32x4 {:?}", out);

    let x = _mm256_set1_epi8(2);
    let x = add_u8x32_i32x8(x);
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, x);
    println!("add_u8x16_i32x4 {:?}", out);
}

fn main() {
    // unsafe { test_avx_funcs() };
    // // TESTING
    // return;

    let m = 6 * 100;
    let n = 8 * 100;
    let k = 4 * 100;

    let a: Vec<u8> = (0..m * k).map(|x| x as u8).collect();
    let b: Vec<i8> = (0..k * n).map(|x| x as i8).collect();
    let mut c = vec![0i32; m * n];
    let a_zero_point = 5;
    let b_zero_point = -5;

    // println!("A:");
    // print_mat(&a, m, k);
    // println!("\nB:");
    // print_mat(&b, k, n);

    let packed_a = pack_a(&a, m, k);
    let packed_b = pack_b(&b, k, n);

    let start = std::time::Instant::now();
    let n_iters = 1000;
    for _ in 0..n_iters {
        // TODO - Fold this into `matmul_int` as a `beta` argument.
        c.fill(0);
        unsafe {
            matmul_int(
                &mut c,
                &packed_a,
                &packed_b,
                a_zero_point,
                b_zero_point,
                m,
                n,
                k,
            )
        };
    }

    let duration = start.elapsed();
    let flops = (2 * m * n * k * n_iters as usize) as f64 / duration.as_secs_f64();
    let gflops = flops / (10f64).powi(9);
    let duration_ms = duration.as_secs_f64() * 1000.0;

    println!(
        "m {} n {} k {} . Duration {}ms. GFLOPS {}",
        m, n, k, duration_ms, gflops,
    );

    let ref_c = reference_matmul_int(&a, &b, a_zero_point, b_zero_point, m, n, k);

    if c != ref_c {
        println!("Reference and optimized implementations match.");
    } else {
        println!("Reference and optimized implementations DO NOT MATCH.");
    }

    // let ref_c_opt = reference_matmul_int_opt_zp(&a, &b, a_zero_point, b_zero_point, m, n, k);

    // println!("\nOptimized:");
    // print_mat(&c, m, n);

    // println!("");
    // println!("Reference:");
    // print_mat(&ref_c, m, n);

    // println!("");
    // println!("Reference (optimized zero-point):");
    // print_mat(&ref_c_opt, m, n);
}
