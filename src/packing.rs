// Pack blocks of the B matrix for use by the matmul kernel.
//
// Pack B matrix of shape `[K, N]` into a layout with shape `[N / NR, K / 4, NR,
// 4]`. The last two dimensions are transposed. In the kernel a transposed `[NR,
// 4]` microtile of `B` is then multiplied with a `[MR, 4]` microtile of `A`
// using dot product instructions.
pub fn pack_b<const NR: usize>(out: &mut [i8], vals: &[i8], b_rows: usize, b_cols: usize) {
    assert!(b_cols % NR == 0);
    assert!(b_cols % 4 == 0);
    assert!(out.len() == b_rows * b_cols);

    let b_row_stride = b_cols;
    let mut out_off = 0;

    for col_block in 0..b_cols / NR {
        for row_block in 0..b_rows / 4 {
            for col_off in 0..NR {
                for row_off in 0..4 {
                    let y = row_block * 4 + row_off;
                    let x = col_block * NR + col_off;
                    unsafe {
                        *out.get_unchecked_mut(out_off) = *vals.get_unchecked(y * b_row_stride + x);
                        out_off += 1;
                    }
                }
            }
        }
    }
}

// Pack blocks of the A matrix for use by the matmul kernel.
//
// Pack A matrix of shape `[M, K]` into a layout with shape `[M / MR, K / 4, MR,
// 4]`.
pub fn pack_a<const MR: usize>(out: &mut [u8], vals: &[u8], a_rows: usize, a_cols: usize) {
    assert!(a_rows % MR == 0);
    assert!(a_cols % 4 == 0);
    assert!(out.len() == a_rows * a_cols);

    let a_row_stride = a_cols;
    let mut out_off = 0;

    for row_block in 0..a_rows / MR {
        for col_block in 0..a_cols / 4 {
            for row_off in 0..MR {
                for col_off in 0..4 {
                    let y = row_block * MR + row_off;
                    let x = col_block * 4 + col_off;
                    unsafe {
                        *out.get_unchecked_mut(out_off) = *vals.get_unchecked(y * a_row_stride + x);
                        out_off += 1;
                    }
                }
            }
        }
    }
}
