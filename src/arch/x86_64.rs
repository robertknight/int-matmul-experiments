use crate::Kernel;

use std::arch::asm;
use std::arch::x86_64::{
    __cpuid_count, __m256, __m256i, _mm256_add_epi32, _mm256_broadcast_ss, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_maskload_epi32, _mm256_mullo_epi32,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi8, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm256_sub_epi32,
};

const VNNI_NONE: u8 = 0;
const VNNI_AVX: u8 = 1;
const VNNI_AVX_512: u8 = 2;

#[derive(Copy, Clone, Debug, PartialEq)]
enum VnniType {
    None,
    Avx,
    Avx512,
}

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
unsafe fn dot_u8i8x32_i32x4<const VNNI_TYPE: u8>(
    mut out: __m256i,
    x: __m256i,
    y: __m256i,
) -> __m256i {
    if VNNI_TYPE == VNNI_AVX_512 {
        // This uses AVX-512 VNNI (EVEX-encoded) rather than AVX-VNNI
        // (VEX-encoded).
        asm! {
            "vpdpbusd {result}, {x}, {y}",
            result = inout(ymm_reg) out,
            x = in(ymm_reg) x,
            y = in(ymm_reg) y,
            options(nostack)
        }
        out
    } else if VNNI_TYPE == VNNI_AVX {
        asm! {
            "{{vex}} vpdpbusd {result}, {x}, {y}",
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
unsafe fn add_i8x32_i32x8<const VNNI_TYPE: u8>(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    dot_u8i8x32_i32x4::<VNNI_TYPE>(_mm256_setzero_si256(), one, x)
}

/// Sum each group of 4 adjacent u8 values and produce i32 results.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn add_u8x32_i32x8<const VNNI_TYPE: u8>(x: __m256i) -> __m256i {
    let one = _mm256_set1_epi8(1);
    dot_u8i8x32_i32x4::<VNNI_TYPE>(_mm256_setzero_si256(), x, one)
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_int<const VNNI_TYPE: u8>(
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
    assert!(n % NR == 0);

    let col_block_size = NR;
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
    let mask_values: [i32; NR] = std::array::from_fn(|i| if i < MR { -1 } else { 0 });
    let mask = _mm256_loadu_si256(mask_values.as_ptr() as *const __m256i);

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * NR * 4;

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
                b_sum = _mm256_add_epi32(b_sum, add_i8x32_i32x8::<VNNI_TYPE>(bv));

                let a_vals = _mm256_maskload_epi32(
                    a_ptr.add(a_off + k_block * size_of::<__m256i>()) as *const i32,
                    mask,
                );

                for i in 0..MR {
                    let av = _mm256_broadcast_ss(std::mem::transmute::<*const u8, &f32>(
                        a_ptr.add(a_off + k_block * size_of::<__m256i>() + i),
                    ));
                    let av = std::mem::transmute::<__m256, __m256i>(av);

                    tmp[i] = dot_u8i8x32_i32x4::<VNNI_TYPE>(tmp[i], av, bv);
                }

                a_sum = _mm256_add_epi32(a_sum, add_u8x32_i32x8::<VNNI_TYPE>(a_vals));
            }

            let a_sum = _mm256_mullo_epi32(a_sum, _mm256_set1_epi32(b_zero_point as i32));
            let b_sum = _mm256_mullo_epi32(b_sum, _mm256_set1_epi32(a_zero_point as i32));

            let a_sums = to_array::<i32, NR>(a_sum);
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

struct VnniInfo {
    avx512_vnni: bool,
    avx_vnni: bool,
}

/// Detect availability of VNNI instructions.
///
/// See https://www.felixcloutier.com/x86/cpuid or the Intel Instruction Set
/// Reference for cpuid.
fn detect_vnni() -> VnniInfo {
    let avx512_vnni = if is_avx512_supported() {
        let regs = unsafe { __cpuid_count(7, 0) };
        regs.ecx & (1 << 11) != 0
    } else {
        false
    };

    let regs = unsafe { __cpuid_count(7, 1) };
    let avx_vnni = regs.eax & (1 << 4) != 0;

    VnniInfo {
        avx512_vnni,
        avx_vnni,
    }
}

pub struct AvxKernel {
    vnni: VnniType,
}

unsafe impl Kernel for AvxKernel {
    fn new() -> Option<Self> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }

        // CPUs may have only AVX512 VNNI, only AVX VNNI or both.
        //
        // Since this is an AVX2 kernel we prefer AVX VNNI, but will use
        // AVX512-VNNI if that is the only option. Ideally AVX512-VNNI would
        // be used in the context of an AVX512 kernel, but AVX512 intrinsics
        // are not available in stable Rust. We can however use AVX512-VNNI via
        // inline assembly with ymm registers.
        let vnni_info = detect_vnni();
        let vnni = match vnni_info {
            VnniInfo { avx_vnni: true, .. } => VnniType::Avx,
            VnniInfo {
                avx512_vnni: true,
                avx_vnni: false,
            } => VnniType::Avx512,
            _ => VnniType::None,
        };

        Some(AvxKernel { vnni })
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
        match self.vnni {
            VnniType::Avx512 => {
                // Safety: AVX2 and AVX512-VNNI are supported.
                unsafe { matmul_int::<VNNI_AVX_512>(c, a, b, a_zero_point, b_zero_point, m, n, k) }
            }
            VnniType::Avx => {
                // Safety: AVX2 and AVX-VNNI are supported.
                unsafe { matmul_int::<VNNI_AVX>(c, a, b, a_zero_point, b_zero_point, m, n, k) }
            }
            VnniType::None => {
                // Safety: AVX2 is supported.
                unsafe { matmul_int::<VNNI_NONE>(c, a, b, a_zero_point, b_zero_point, m, n, k) }
            }
        }
    }
}

/// Detect availability of AVX-512 on macOS, where `is_x86_feature_detected`
/// can return false even if AVX-512 is available.
///
/// See https://github.com/golang/go/issues/43089. Go chose to use the
/// `commpage` to get the info. We use `sysctlbyname` instead since it is
/// a documented API.
#[cfg(target_os = "macos")]
fn test_for_avx512_on_macos() -> bool {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    #[link(name = "c")]
    extern "C" {
        /// See https://developer.apple.com/documentation/kernel/1387446-sysctlbyname.
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    static AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVX512_AVAILABLE.get_or_init(|| {
        unsafe {
            let mut ret = 0u64;
            let mut size = std::mem::size_of::<u64>();

            // We test only for avx512vl, as this implies avx512f.
            let sysctl_ret = sysctlbyname(
                CStr::from_bytes_with_nul(b"hw.optional.avx512vl\0")
                    .unwrap()
                    .as_ptr(),
                std::mem::transmute(&mut ret),
                &mut size,
                std::ptr::null(),
                0,
            );
            sysctl_ret == 0 && ret == 1
        }
    })
}

/// Test if the current system has basic AVX-512 support (AVX-512 F, AVX-512 VL).
///
/// This is unfortunately not as simple as using `is_x86_feature_detected`
/// because that can return incorrect results on macOS.
fn is_avx512_supported() -> bool {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
        true
    } else {
        #[cfg(target_os = "macos")]
        {
            test_for_avx512_on_macos()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

#[cfg(feature = "avx512")]
pub mod avx512 {
    use std::arch::asm;
    use std::arch::x86_64::{
        __m512i, _mm512_add_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
        _mm512_mask_load_epi32, _mm512_mullo_epi32, _mm512_set1_epi16, _mm512_set1_epi32,
        _mm512_set1_epi8, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi32,
    };

    use super::{detect_vnni, VnniInfo, VnniType, VNNI_AVX_512, VNNI_NONE};
    use crate::Kernel;

    /// Number of rows in tiles used by microkernel.
    const MR: usize = 14;
    const REG_SIZE: usize = size_of::<__m512i>() / size_of::<i32>();
    const NR: usize = REG_SIZE;

    /// Convert a SIMD vector into a `[T; N]` array.
    fn to_array<T: Copy + Default, const N: usize>(x: __m512i) -> [T; N] {
        assert_eq!(size_of::<T>() * N, size_of::<__m512i>());
        let mut out = [T::default(); N];
        unsafe {
            _mm512_storeu_si512(out.as_mut_ptr() as *mut i32, x);
        }
        out
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn dot_u8i8x32_i32x4<const VNNI_TYPE: u8>(
        mut out: __m512i,
        x: __m512i,
        y: __m512i,
    ) -> __m512i {
        if VNNI_TYPE == VNNI_AVX_512 {
            // This uses AVX-512 VNNI (EVEX-encoded) rather than AVX-VNNI
            // (VEX-encoded).
            asm! {
                "vpdpbusd {result}, {x}, {y}",
                result = inout(zmm_reg) out,
                x = in(zmm_reg) x,
                y = in(zmm_reg) y,
                options(nostack)
            }
            out
        } else {
            let tmp = _mm512_maddubs_epi16(x, y);
            let tmp = _mm512_madd_epi16(tmp, _mm512_set1_epi16(1));
            _mm512_add_epi32(out, tmp)
        }
    }

    /// Sum each group of 4 adjacent i8 values and produce i32 results.
    #[target_feature(enable = "avx512f")]
    unsafe fn add_i8x32_i32x8<const VNNI_TYPE: u8>(x: __m512i) -> __m512i {
        let one = _mm512_set1_epi8(1);
        dot_u8i8x32_i32x4::<VNNI_TYPE>(_mm512_setzero_si512(), one, x)
    }

    /// Sum each group of 4 adjacent u8 values and produce i32 results.
    #[target_feature(enable = "avx512f")]
    unsafe fn add_u8x32_i32x8<const VNNI_TYPE: u8>(x: __m512i) -> __m512i {
        let one = _mm512_set1_epi8(1);
        dot_u8i8x32_i32x4::<VNNI_TYPE>(_mm512_setzero_si512(), x, one)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn matmul_int<const VNNI_TYPE: u8>(
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
        assert!(n % NR == 0);

        let col_block_size = NR;
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
        let c_init = _mm512_mullo_epi32(
            _mm512_set1_epi32(k as i32),
            _mm512_mullo_epi32(
                _mm512_set1_epi32(a_zero_point as i32),
                _mm512_set1_epi32(b_zero_point as i32),
            ),
        );

        // Mask which is set for the first `MR` elements.
        assert!(MR <= 16);
        let mask = 0u16 >> 16 - MR;

        for col_block in 0..n_col_blocks {
            let b_off = col_block * n_depth_blocks * NR * 4;

            for row_block in 0..n_row_blocks {
                let a_off = row_block * n_depth_blocks * MR * 4;

                // Sums along each row of `a`.
                let mut a_sum = _mm512_setzero_si512();
                // Sums along each column of `b`.
                let mut b_sum = _mm512_setzero_si512();

                let mut tmp = [c_init; MR];

                for k_block in 0..n_depth_blocks {
                    let bv = _mm512_loadu_si512(
                        b_ptr.add(b_off + k_block * size_of::<__m512i>()) as *const i32
                    );
                    b_sum = _mm512_add_epi32(b_sum, add_i8x32_i32x8::<VNNI_TYPE>(bv));

                    let a_vals = _mm512_mask_load_epi32(
                        _mm512_setzero_si512(),
                        mask,
                        a_ptr.add(a_off + k_block * size_of::<__m512i>()) as *const i32,
                    );
                    let a_elts = to_array::<i32, 16>(a_vals);

                    for i in 0..MR {
                        let av = _mm512_set1_epi32(a_elts[i]);
                        tmp[i] = dot_u8i8x32_i32x4::<VNNI_TYPE>(tmp[i], av, bv);
                    }

                    a_sum = _mm512_add_epi32(a_sum, add_u8x32_i32x8::<VNNI_TYPE>(a_vals));
                }

                let a_sum = _mm512_mullo_epi32(a_sum, _mm512_set1_epi32(b_zero_point as i32));
                let b_sum = _mm512_mullo_epi32(b_sum, _mm512_set1_epi32(a_zero_point as i32));

                let a_sums = to_array::<i32, REG_SIZE>(a_sum);
                for i in 0..MR {
                    let c_off = (row_block * row_block_size + i) * c_row_stride
                        + (col_block * col_block_size);
                    let z = tmp[i];
                    let z = _mm512_sub_epi32(z, b_sum);
                    let z = _mm512_sub_epi32(z, _mm512_set1_epi32(a_sums[i]));
                    _mm512_storeu_si512(c_ptr.add(c_off), z);
                }
            }
        }
    }

    pub struct Avx512Kernel {
        vnni: VnniType,
    }

    unsafe impl Kernel for Avx512Kernel {
        fn new() -> Option<Self> {
            if !super::is_avx512_supported() {
                return None;
            }

            let vnni_info = detect_vnni();
            let vnni = match vnni_info {
                VnniInfo {
                    avx512_vnni: true, ..
                } => VnniType::Avx512,
                _ => VnniType::None,
            };

            Some(Avx512Kernel { vnni })
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
            match self.vnni {
                VnniType::Avx512 => {
                    // Safety: AVX2 and AVX512-VNNI are supported.
                    unsafe {
                        matmul_int::<VNNI_AVX_512>(c, a, b, a_zero_point, b_zero_point, m, n, k)
                    }
                }
                _ => {
                    // Safety: AVX512 is supported.
                    unsafe { matmul_int::<VNNI_NONE>(c, a, b, a_zero_point, b_zero_point, m, n, k) }
                }
            }
        }
    }
}
