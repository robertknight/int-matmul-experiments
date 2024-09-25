use crate::Kernel;

const MR: usize = 4;
const NR: usize = 4;
const NVEC: usize = 4;

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

    let c_init = [k as i32 * (a_zero_point as i32) * (b_zero_point as i32); NVEC];

    for col_block in 0..n_col_blocks {
        let b_off = col_block * n_depth_blocks * NR * 4;

        for row_block in 0..n_row_blocks {
            let a_off = row_block * n_depth_blocks * MR * 4;

            let mut a_sum = [[0; NVEC]; MR];
            let mut b_sum = [[0; NVEC]; NR];

            let mut tmp = [c_init; MR];

            for k_block in 0..n_depth_blocks {
                // Load bv

                // Update `b_sum`

                // Load `a_vals`

                for i in 0..MR {}

                // Update `a_sum`
            }

            for i in 0..MR {
                let c_off =
                    (row_block * row_block_size + i) * c_row_stride + (col_block * col_block_size);
                for j in 0..NVEC {
                    tmp[i][j] -= b_sum[j];
                    tmp[i][j] -= a_sums[i];
                    unsafe { *c_ptr.add(c_off + j) = tmp[i][j] };
                }
            }
        }
    }
}

pub struct GenericKernel {}

impl GenericKernel {
    fn new() -> Self {
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
