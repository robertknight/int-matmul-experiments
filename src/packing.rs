const K_TILE: usize = 4;

/// Helper for incrementally filling a slice.
struct SliceWriter<'a, T> {
    offset: usize,
    slice: &'a mut [T],
}

impl<'a, T> SliceWriter<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        SliceWriter { slice, offset: 0 }
    }

    /// Return true if the slice has been fully written.
    fn completed(&self) -> bool {
        self.offset == self.slice.len()
    }

    /// Write the next element in the slice.
    unsafe fn write_unchecked(&mut self, val: T) {
        debug_assert!(self.offset < self.slice.len());
        *self.slice.get_unchecked_mut(self.offset) = val;
        self.offset += 1;
    }

    /// Write `len` copies of `val` to the slice.
    unsafe fn write_n_unchecked(&mut self, len: usize, val: T)
    where
        T: Copy,
    {
        for i in 0..len {
            *self.slice.get_unchecked_mut(self.offset + i) = val;
        }
        self.offset += len;
    }
}

/// Return the size of packing buffer required by `pack_b`.
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
    let mut out = SliceWriter::new(out);

    for col_tile in 0..b_cols.div_ceil(NR) {
        let mut col_sums = [0i32; NR];
        let col_range = col_tile * NR..(col_tile * NR + NR).min(b_cols);

        for row_tile in 0..b_rows.div_ceil(K_TILE) {
            let row_range = row_tile * K_TILE..(row_tile * K_TILE + K_TILE).min(b_rows);
            if col_range.len() == NR && row_range.len() == K_TILE {
                // Full tile
                for c in 0..NR {
                    for r in 0..K_TILE {
                        let y = row_tile * K_TILE + r;
                        let x = col_tile * NR + c;
                        unsafe {
                            let val = *vals.get_unchecked(y * b_row_stride + x);
                            col_sums[c] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                }
            } else {
                // Partial tile
                for c in 0..col_range.len() {
                    for r in 0..row_range.len() {
                        let y = row_tile * K_TILE + r;
                        let x = col_tile * NR + c;
                        unsafe {
                            let val = *vals.get_unchecked(y * b_row_stride + x);
                            col_sums[c] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                    // Pad to row tile size
                    unsafe { out.write_n_unchecked(K_TILE - row_range.len(), 0) };
                }
                // Pad to column tile size
                unsafe { out.write_n_unchecked((NR - col_range.len()) * K_TILE, 0) };
            }
        }

        for c in 0..NR {
            let col_sum_i8 = col_sums[c].to_ne_bytes().map(|b| b as i8);
            for i in 0..4 {
                unsafe {
                    out.write_unchecked(col_sum_i8[i]);
                }
            }
        }
    }

    assert!(out.completed());
}

/// Return the size of packing buffer required by `pack_a`.
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
    let mut out = SliceWriter::new(out);

    let a_row_stride = a_cols;

    for row_tile in 0..a_rows.div_ceil(MR) {
        let mut row_sums = [0i32; MR];
        let row_range = row_tile * MR..(row_tile * MR + MR).min(a_rows);

        for col_tile in 0..a_cols.div_ceil(K_TILE) {
            let col_range = col_tile * K_TILE..(col_tile * K_TILE + K_TILE).min(a_cols);

            if row_range.len() == MR && col_range.len() == K_TILE {
                // Full tile
                for r in 0..MR {
                    for c in 0..K_TILE {
                        let y = row_tile * MR + r;
                        let x = col_tile * K_TILE + c;
                        unsafe {
                            let val = *vals.get_unchecked(y * a_row_stride + x);
                            row_sums[r] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                }
            } else {
                // Partial tile
                for r in 0..row_range.len() {
                    for c in 0..col_range.len() {
                        let y = row_tile * MR + r;
                        let x = col_tile * K_TILE + c;
                        unsafe {
                            let val = *vals.get_unchecked(y * a_row_stride + x);
                            row_sums[r] += val as i32;
                            out.write_unchecked(val);
                        }
                    }
                    // Pad to column tile size
                    unsafe {
                        out.write_n_unchecked(K_TILE - col_range.len(), 0);
                    }
                }
                // Pad to row tile size
                unsafe {
                    out.write_n_unchecked((MR - row_range.len()) * K_TILE, 0);
                }
            }
        }

        for r in 0..MR {
            let row_sum_u8 = row_sums[r].to_ne_bytes();
            for i in 0..4 {
                unsafe {
                    out.write_unchecked(row_sum_u8[i]);
                }
            }
        }
    }

    assert!(out.completed());
}
