use crate::Kernel;

use std::arch::asm;
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
    _mm256_setzero_si256,
    _mm256_storeu_si256,
    _mm256_sub_epi32,
};

/// Whether to use AVX-VNNI instructions. This improves performance by
/// performing u8 x i8 -> i32 dot products with one instruction instead of
/// three.
const USE_VNNI: bool = false;

/// Number of rows in tiles used by microkernel. Empirically 7 is the maximum
/// value for the current implementation and performance drops a lot with
/// larger values.
const MR: usize = 7;

const NR: usize = size_of::<__m256i>() / size_of::<i32>();

/// Convert a SIMD vector into a `[T; N]` array.
fn to_array<T: Copy + Default, const N: usize>(x: __m256i) -> [T; N] {
    assert_eq!(size_of::<T>() * N, size_of::<__m256i>());
    let mut out = [T::default(); N];
    unsafe {
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, x);
    }
    out
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8i8x32_i32x4(mut out: __m256i, x: __m256i, y: __m256i) -> __m256i {
    if USE_VNNI {
        asm! {
            "vpdpbusd {result}, {x}, {y}",
            result = inout(ymm_reg) out,
            x = in(ymm_reg) x,
            y = in(ymm_reg) y,
            options(nostack)
        }
        out
    } else {
        let tmp = _mm256_maddubs_epi16(x, y);
        let tmp = _mm256_madd_epi16(tmp, _mm256_set1_epi16(1));
        _mm256_add_epi32(out, tmp)
    }
}

/// Sum each group of 4 adjacent i8 values and produce i32 results.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn add_i8x32_i32x8(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    dot_u8i8x32_i32x4(_mm256_setzero_si256(), one, x)
}

/// Sum each group of 4 adjacent u8 values and produce i32 results.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn add_u8x32_i32x8(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    dot_u8i8x32_i32x4(_mm256_setzero_si256(), x, one)
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
    //
    // This can be expanded and re-arranged into:
    //
    // c = a[0]b[0] - a[0] * b_zero_point - b[0] * a_zero_point + a_zero_point * b_zero_point + ...
    // c = dot(a, b) - sum(a) * b_zero_point - sum(b) * a_zero_point + ... + len(a) * a_zero_point * b_zero_point
    //
    // The final term of the above equation does not depend on any individual
    // elements, so is pre-computed and used as the initialization value for
    // each output tile. Note that this assumes a single zero-point value per
    // tensor.
    //
    // The accumulation of the sum of each row of a and each column of b are
    // interleaved with the dot product and subtracted from the temporary `c`
    // value at the end.
    let c_init = _mm256_mullo_epi32(
        _mm256_set1_epi32(k as i32),
        _mm256_mullo_epi32(
            _mm256_set1_epi32(a_zero_point as i32),
            _mm256_set1_epi32(b_zero_point as i32),
        ),
    );

    // Mask which is set for the first `MR` elements.
    let mask_values: [i32; 8] = std::array::from_fn(|i| if i < MR { -1 } else { 0 });
    let mask = _mm256_loadu_si256(mask_values.as_ptr() as *const __m256i);

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * 8 * 4;

        for row_block in 0..n_row_blocks {
            let a_off = row_block * n_depth_blocks * MR * 4;

            // Sums along each row of `a`.
            let mut a_sum = _mm256_setzero_si256();
            // Sums along each column of `b`.
            let mut b_sum = _mm256_setzero_si256();

            let mut tmp = [c_init; MR];

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
                        a_ptr.add(a_off + k_block * size_of::<__m256i>() + i),
                    ));
                    let av = std::mem::transmute::<__m256, __m256i>(av);

                    tmp[i] = dot_u8i8x32_i32x4(tmp[i], av, bv);
                }

                a_sum = _mm256_add_epi32(a_sum, add_u8x32_i32x8(a_vals));
            }

            let a_sum = _mm256_mullo_epi32(a_sum, _mm256_set1_epi32(b_zero_point as i32));
            let a_sums = to_array::<i32, 8>(a_sum);
            let b_sum = _mm256_mullo_epi32(b_sum, _mm256_set1_epi32(a_zero_point as i32));

            for i in 0..MR {
                let c_off =
                    (row_block * row_block_size + i) * c_row_stride + (col_block * col_block_size);
                let z = tmp[i];
                let z = _mm256_sub_epi32(z, b_sum);
                let z = _mm256_sub_epi32(z, _mm256_set1_epi32(a_sums[i]));
                _mm256_storeu_si256(c_ptr.add(c_off) as *mut __m256i, z);
            }
        }
    }
}

pub struct AvxKernel {}

unsafe impl Kernel for AvxKernel {
    fn new() -> Option<Self> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        Some(AvxKernel {})
    }

    /// Return number of rows in this kernel's microtile.
    fn mr(&self) -> usize {
        MR
    }

    /// Return number of columns in this kernel's microtile.
    fn nr(&self) -> usize {
        NR
    }

    /// Return size of packing buffer required by `pack_a`.
    fn packed_a_size(&self, a_rows: usize, a_cols: usize) -> usize {
        a_rows * a_cols
    }

    /// Pack an input LHS / "A" matrix
    fn pack_a(&self, out: &mut [u8], a: &[u8], a_rows: usize, a_cols: usize) {
        crate::packing::pack_a::<MR>(out, a, a_rows, a_cols)
    }

    /// Return size of packing buffer required by `pack_b`.
    fn packed_b_size(&self, b_rows: usize, b_cols: usize) -> usize {
        b_rows * b_cols
    }

    /// Pack an input RHS / "B" matrix
    fn pack_b(&self, out: &mut [i8], b: &[i8], b_rows: usize, b_cols: usize) {
        crate::packing::pack_b::<NR>(out, b, b_rows, b_cols)
    }

    fn matmul(
        &self,
        c: &mut [i32],
        a: &[u8],
        b: &[i8],
        a_zero_point: u8,
        b_zero_point: i8,
        m: usize,
        n: usize,
        k: usize,
    ) {
        unsafe { matmul_int(c, a, b, a_zero_point, b_zero_point, m, n, k) }
    }
}
