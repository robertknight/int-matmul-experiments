const K_TILE: usize = 4;

pub fn pack_b_size<const NR: usize>(b_rows: usize, b_cols: usize) -> usize {
    // Packed block is padded to a multiple of NR columns and K_TILE rows.
    let n_panels = b_cols.div_ceil(NR);
    let panel_data_size = b_rows.div_ceil(K_TILE) * NR * K_TILE;
    let col_sums_size = NR * 4;
    let panel_stride = panel_data_size + col_sums_size;
    n_panels * panel_stride
}

// Pack blocks of the B matrix for use by the matmul kernel.
//
// Pack B matrix of shape `[K, N]` into a series of column panels. Each panel
// contains elements from a `[K, NR]` slice of the input and is laid out as `[K
// / 4, NR, 4]` u8 values, followed by `NR` i32 column sums.  In the kernel a
// transposed `[NR, 4]` microtile of `B` is then multiplied with a `[MR, 4]`
// microtile of `A` using dot product instructions. The column sums are used
// to handle subtraction of the zero point.
pub fn pack_b<const NR: usize>(out: &mut [i8], vals: &[i8], b_rows: usize, b_cols: usize) {
    assert_eq!(out.len(), pack_b_size::<NR>(b_rows, b_cols));

    let b_row_stride = b_cols;
    let mut out_off = 0;

    for col_tile in 0..b_cols.div_ceil(NR) {
        let mut col_sums = [0i32; NR];
        let col_range = col_tile * NR..(col_tile * NR + NR).min(b_cols);

        for row_tile in 0..b_rows.div_ceil(K_TILE) {
            let row_range = row_tile * K_TILE..(row_tile * K_TILE + K_TILE).min(b_rows);
            if col_range.len() == NR && row_range.len() == K_TILE {
                // Full tile
                for col_off in 0..NR {
                    for row_off in 0..K_TILE {
                        let y = row_tile * K_TILE + row_off;
                        let x = col_tile * NR + col_off;
                        unsafe {
                            let val = *vals.get_unchecked(y * b_row_stride + x);
                            col_sums[col_off] += val as i32;
                            *out.get_unchecked_mut(out_off) = val;
                            out_off += 1;
                        }
                    }
                }
            } else {
                // Partial tile
                for col_off in 0..col_range.len() {
                    for row_off in 0..row_range.len() {
                        let y = row_tile * K_TILE + row_off;
                        let x = col_tile * NR + col_off;
                        unsafe {
                            let val = *vals.get_unchecked(y * b_row_stride + x);
                            col_sums[col_off] += val as i32;
                            *out.get_unchecked_mut(out_off) = val;
                            out_off += 1;
                        }
                    }
                    for _row_off in row_range.len()..K_TILE {
                        unsafe {
                            *out.get_unchecked_mut(out_off) = 0;
                        }
                        out_off += 1;
                    }
                }
                for _ in col_range.len()..NR {
                    for _row_off in 0..K_TILE {
                        unsafe {
                            *out.get_unchecked_mut(out_off) = 0;
                        }
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

    assert_eq!(out_off, out.len());
}

pub fn pack_a_size<const MR: usize>(a_rows: usize, a_cols: usize) -> usize {
    // Packed block is padded to a multiple of MR rows and K_TILE columns.
    let n_panels = a_rows.div_ceil(MR);
    let panel_data_size = a_cols.div_ceil(K_TILE) * MR * K_TILE;
    let row_sums_size = MR * 4;
    let panel_stride = panel_data_size + row_sums_size;
    n_panels * panel_stride
}

// Pack blocks of the A matrix for use by the matmul kernel.
//
// Pack A matrix of shape `[M, K]` into a series of row panels. Each panel
// contains elements from an `[MR, K]` slice of the input and is laid out as `[K
// / 4, MR, 4]` u8 values, followed by `MR` i32 row sums. The row sums are
// used to handle subtraction of the zero point.
pub fn pack_a<const MR: usize>(out: &mut [u8], vals: &[u8], a_rows: usize, a_cols: usize) {
    assert_eq!(out.len(), pack_a_size::<MR>(a_rows, a_cols));

    let a_row_stride = a_cols;
    let mut out_off = 0;

    for row_tile in 0..a_rows.div_ceil(MR) {
        let mut row_sums = [0i32; MR];
        let row_range = row_tile * MR..(row_tile * MR + MR).min(a_rows);

        for col_tile in 0..a_cols.div_ceil(K_TILE) {
            let col_range = col_tile * K_TILE..(col_tile * K_TILE + K_TILE).min(a_cols);

            if row_range.len() == MR && col_range.len() == K_TILE {
                // Full tile
                for row_off in 0..MR {
                    for col_off in 0..K_TILE {
                        let y = row_tile * MR + row_off;
                        let x = col_tile * K_TILE + col_off;
                        unsafe {
                            let val = *vals.get_unchecked(y * a_row_stride + x);
                            row_sums[row_off] += val as i32;
                            *out.get_unchecked_mut(out_off) = val;
                            out_off += 1;
                        }
                    }
                }
            } else {
                // Partial tile
                for row_off in 0..row_range.len() {
                    for col_off in 0..col_range.len() {
                        let y = row_tile * MR + row_off;
                        let x = col_tile * K_TILE + col_off;
                        unsafe {
                            let val = *vals.get_unchecked(y * a_row_stride + x);
                            row_sums[row_off] += val as i32;
                            *out.get_unchecked_mut(out_off) = val;
                            out_off += 1;
                        }
                    }
                    for _col_off in col_range.len()..K_TILE {
                        unsafe {
                            *out.get_unchecked_mut(out_off) = 0;
                            out_off += 1;
                        }
                    }
                }
                for _row_off in row_range.len()..MR {
                    for _col_off in 0..K_TILE {
                        unsafe { *out.get_unchecked_mut(out_off) = 0 };
                        out_off += 1;
                    }
                }
            }
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

    assert_eq!(out_off, out.len());
}
