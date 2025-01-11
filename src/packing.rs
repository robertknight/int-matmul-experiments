// Pack blocks of the B matrix for use by the matmul kernel.
//
// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
// / 4, NR, 4]` u8 values, followed by `NR` i32 column sums.  In the kernel a
// transposed `[NR, 4]` microtile of `B` is then multiplied with a `[MR, 4]`
// microtile of `A` using dot product instructions. The column sums are used
// to handle subtraction of the zero point.
pub fn pack_b<const NR: usize>(out: &mut [i8], vals: &[i8], b_rows: usize, b_cols: usize) {
    assert!(b_cols % NR == 0);
    assert!(b_cols % 4 == 0);
    assert!(out.len() == b_rows * b_cols + b_cols * 4);

    let b_row_stride = b_cols;
    let mut out_off = 0;

    for col_block in 0..b_cols / NR {
        let mut col_sums = [0i32; NR];

        for row_block in 0..b_rows / 4 {
            for col_off in 0..NR {
                for row_off in 0..4 {
                    let y = row_block * 4 + row_off;
                    let x = col_block * NR + col_off;
                    unsafe {
                        let val = *vals.get_unchecked(y * b_row_stride + x);
                        col_sums[col_off] += val as i32;
                        *out.get_unchecked_mut(out_off) = val;
                        out_off += 1;
                    }
                }
            }
        }

        debug_assert_eq!(out_off % 4, 0);
        for col_off in 0..NR {
            let col_sum_i8 = col_sums[col_off].to_ne_bytes().map(|b| b as i8);
            for i in 0..4 {
                unsafe {
                    *out.get_unchecked_mut(out_off) = col_sum_i8[i];
                }
                out_off += 1;
            }
        }
    }
}

// Pack blocks of the A matrix for use by the matmul kernel.
//
// Pack A matrix of shape `[M, K]` into a series of row panels. Each panel
// contains elements from an `[MR, K]` slice of the input and is laid out as `[K
// / 4, MR, 4]` u8 values, followed by `MR` i32 row sums. The row sums are
// used to handle subtraction of the zero point.
pub fn pack_a<const MR: usize>(out: &mut [u8], vals: &[u8], a_rows: usize, a_cols: usize) {
    assert!(a_rows % MR == 0);
    assert!(a_cols % 4 == 0);
    assert!(out.len() == a_rows * a_cols + a_rows * 4);

    let a_row_stride = a_cols;
    let mut out_off = 0;

    for row_block in 0..a_rows / MR {
        let mut row_sums = [0i32; MR];

        for col_block in 0..a_cols / 4 {
            for row_off in 0..MR {
                for col_off in 0..4 {
                    let y = row_block * MR + row_off;
                    let x = col_block * 4 + col_off;
                    unsafe {
                        let val = *vals.get_unchecked(y * a_row_stride + x);
                        row_sums[row_off] += val as i32;
                        *out.get_unchecked_mut(out_off) = val;
                        out_off += 1;
                    }
                }
            }
            debug_assert_eq!(out_off % 4, 0);
        }

        for row_off in 0..MR {
            let row_sum_u8 = row_sums[row_off].to_ne_bytes();
            for i in 0..4 {
                unsafe {
                    *out.get_unchecked_mut(out_off) = row_sum_u8[i];
                }
                out_off += 1;
            }
        }
    }
}
