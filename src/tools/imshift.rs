use std::path::Path;

use crate::core::tiff_io;
use crate::core::{DwError, FimImage};

/// Shift a 3D image by (dx, dy, dz) using trilinear interpolation.
///
/// For each output voxel (m, n, p), the value is interpolated from the input at
/// position (m - dx, n - dy, p - dz). Out-of-bounds positions map to zero.
pub fn run_imshift(
    input: &Path,
    output: &Path,
    dx: f32,
    dy: f32,
    dz: f32,
) -> Result<(), DwError> {
    let (img, meta) = tiff_io::tiff_read(input)?;

    let (m_dim, n_dim, p_dim) = img.dims();
    let mut out = FimImage::zeros(m_dim, n_dim, p_dim);

    for pp in 0..p_dim {
        let src_z = pp as f64 - dz as f64;
        for nn in 0..n_dim {
            let src_y = nn as f64 - dy as f64;
            for mm in 0..m_dim {
                let src_x = mm as f64 - dx as f64;
                let val = img.interp3_linear(src_x, src_y, src_z);
                out.set(mm, nn, pp, val);
            }
        }
    }

    let mut out_meta = meta;
    out_meta.image_description = None;
    tiff_io::tiff_write_f32(output, &out, Some(&out_meta))?;

    Ok(())
}
