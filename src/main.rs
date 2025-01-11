#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

mod arch;
mod packing;

#[allow(unused_imports)]
use arch::KernelHint;

/// Trait implemented by integer matmul kernels.
///
/// # Safety
///
/// The [`new`](Self::new) method must return `None` if the kernel is not
/// supported on the current system.
unsafe trait Kernel {
    /// Construct kernel if supported on current system.
    fn new() -> Option<Self>
    where
        Self: Sized;

    fn name(&self) -> &str;

    /// Return number of rows in this kernel's microtile.
    fn mr(&self) -> usize;

    /// Return number of columns in this kernel's microtile.
    fn nr(&self) -> usize;

    /// Return size of packing buffer required by `pack_a`.
    fn packed_a_size(&self, a_rows: usize, a_cols: usize) -> usize;

    /// Pack an input LHS / "A" matrix
    fn pack_a(&self, out: &mut [u8], a: &[u8], a_rows: usize, a_cols: usize);

    /// Return size of packing buffer required by `pack_b`.
    fn packed_b_size(&self, b_rows: usize, b_cols: usize) -> usize;

    /// Pack an input RHS / "B" matrix
    fn pack_b(&self, out: &mut [i8], b: &[i8], b_rows: usize, b_cols: usize);

    /// Perform a matrix multiplication of two packed inputs into an output
    /// `c` buffer.
    fn matmul(
        &self,
        c: &mut [i32],
        packed_a: &[u8],
        packed_b: &[i8],
        a_zero_point: u8,
        b_zero_point: i8,
        a_rows: usize,
        b_cols: usize,
        a_cols: usize,
    );
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

/// Print the values of a matrix.
#[allow(unused)]
fn print_mat<E: std::fmt::Display>(mat: &[E], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            print!("{} ", mat[row * n + col]);
        }
        println!("");
    }
}

#[derive(Copy, Clone, Debug)]
struct InputSize {
    m: usize,
    n: usize,
    k: usize,
}

impl InputSize {
    #[allow(unused)]
    fn uniform(scale: usize) -> Self {
        InputSize {
            m: scale,
            n: scale,
            k: scale,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum ErrorDetail {
    Short,
    Verbose,
}

fn test_kernel(kernel: &dyn Kernel, n_iters: usize, size: InputSize, detail: ErrorDetail) {
    let InputSize { m, n, k } = size;

    let a_zero_point = 5;
    let b_zero_point = -5;

    // When using non-VNNI instructions on x64, VPMADDUBSW may saturate when
    // adding adjacent signed 16-bit intermediate values. This can cause
    // incorrect results. To avoid this, limit the maximum values in the input.
    //
    // If all values in the input are <= `isqrt(i16::MAX / 2)` (127) then
    // overflow cannot happen. nb. This assumes that the zero point subtraction
    // is performed at the end on the i32 values.
    let max_val = 127;

    let a: Vec<u8> = (0..m * k).map(|x| (x % max_val) as u8).collect();
    let b: Vec<i8> = (0..k * n).map(|x| x as i8).collect();
    let mut c = vec![0i32; m * n];

    let mut packed_a = vec![0; kernel.packed_a_size(m, k)];
    let mut packed_b = vec![0; kernel.packed_b_size(k, n)];

    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        kernel.pack_a(&mut packed_a, &a, m, k);
        kernel.pack_b(&mut packed_b, &b, k, n);
        kernel.matmul(
            &mut c,
            &packed_a,
            &packed_b,
            a_zero_point,
            b_zero_point,
            m,
            n,
            k,
        )
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

    let n_differences = c.iter().zip(&ref_c).filter(|(x, y)| x != y).count();

    if n_differences == 0 {
        println!("Reference and optimized implementations match.");
    } else {
        println!(
            "Reference and optimized implementations DO NOT MATCH ({} differences)",
            n_differences
        );

        if detail == ErrorDetail::Verbose {
            println!("\nActual:\n");
            print_mat(&c, m, n);

            println!("\nExpected:\n");
            print_mat(&ref_c, m, n);
        }
    }
}

fn main() {
    let kernel = arch::new_kernel(None);
    println!("Kernel: {}", kernel.name());

    // Size chosen to not be a multiple of tile size along any dimension.
    let size = InputSize {
        m: 25,
        n: 13,
        k: 15,
    };
    let n_iters = 1;
    test_kernel(kernel.as_ref(), n_iters, size, ErrorDetail::Verbose);

    // Do a benchmark
    // #[cfg(not(debug_assertions))]
    {
        let n_iters = 500;
        let size = InputSize::uniform(512);
        test_kernel(kernel.as_ref(), n_iters, size, ErrorDetail::Short);
    }
}
