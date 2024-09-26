use crate::Kernel;

const MR: usize = 4;
const NR: usize = 4;
const NVEC: usize = 4;
const NVEC_I8: usize = 16;

fn add_i8x16_i32x4(x: [i8; 16]) -> [i32; 4] {
    let y: [i16; 8] = std::array::from_fn(|i| x[i * 2] as i16 * x[i * 2 + 1] as i16);
    std::array::from_fn(|i| y[i * 2] as i32 + y[i * 2 + 1] as i32)
}

fn add_u8x16_i32x4(x: [u8; 16]) -> [i32; 4] {
    let y: [u16; 8] = std::array::from_fn(|i| x[i * 2] as u16 * x[i * 2 + 1] as u16);
    std::array::from_fn(|i| y[i * 2] as i32 + y[i * 2 + 1] as i32)
}

fn dot_u8i8x16_i32x4(c: [i32; 4], a: [u8; 16], b: [i8; 16]) -> [i32; 4] {
    let y: [i16; 8] = std::array::from_fn(|i| a[i * 2] as i16 * b[i * 2 + 1] as i16);
    let y: [i32; 4] = std::array::from_fn(|i| y[i * 2] as i32 + y[i * 2 + 1] as i32);
    std::array::from_fn(|i| c[i] + y[i])
}

fn matmul_int(
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
    let c_row_stride = n;

    let b_ptr = b.as_ptr();
    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let c_init = [k as i32 * (a_zero_point as i32) * (b_zero_point as i32); NVEC];

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * NR * 4;

        for row_block in 0..n_row_blocks {
            let a_off = row_block * n_depth_blocks * MR * 4;

            let mut a_sum = [0; MR];
            let mut b_sum = [0; NR];

            let mut tmp = [c_init; MR];

            for k_block in 0..n_depth_blocks {
                let bv: [i8; 16] =
                    std::array::from_fn(|i| unsafe { *b_ptr.add(b_off + k_block * NVEC_I8 + i) });
                let bv_sum = add_i8x16_i32x4(bv);
                for i in 0..NR {
                    b_sum[i] += bv_sum[i];
                }

                let a_vals: [u8; MR * 4] = std::array::from_fn(|i| unsafe {
                    *a_ptr.add(a_off + k_block * NVEC_I8 + i * 4)
                });

                for i in 0..MR {
                    let av: [u8; 16] = std::array::from_fn(|j| a_vals[i * 4 + j % 4]);
                    tmp[i] = dot_u8i8x16_i32x4(tmp[i], av, bv);
                }

                let av_sum = add_u8x16_i32x4(a_vals);
                for i in 0..MR {
                    a_sum[i] += av_sum[i];
                }
            }

            for i in 0..MR {
                a_sum[i] *= b_zero_point as i32;
            }
            for i in 0..NR {
                b_sum[i] *= a_zero_point as i32;
            }

            for i in 0..MR {
                let c_off =
                    (row_block * row_block_size + i) * c_row_stride + (col_block * col_block_size);
                for j in 0..NVEC {
                    tmp[i][j] -= b_sum[j];
                    tmp[i][j] -= a_sum[i];
                    unsafe { *c_ptr.add(c_off + j) = tmp[i][j] };
                }
            }
        }
    }
}

pub struct GenericKernel {}

impl GenericKernel {
    pub fn new() -> Self {
        GenericKernel {}
    }
}

impl Default for GenericKernel {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Kernel for GenericKernel {
    fn new() -> Option<Self> {
        Some(GenericKernel::new())
    }

    fn mr(&self) -> usize {
        MR
    }

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
        matmul_int(c, a, b, a_zero_point, b_zero_point, m, n, k);
    }
}
