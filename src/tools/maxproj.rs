use std::path::Path;

use crate::core::tiff_io;
use crate::core::{DwError, FimImage};

/// Maximum projection modes.
pub enum MaxProjMode {
    /// Standard max projection along Z.
    Max,
    /// Extract a specific Z-slice.
    Slice(usize),
    /// 3-panel collage: XY max, XZ max, and YZ max side by side.
    Xyz,
    /// Extract the most in-focus slice using gradient magnitude.
    GradientMagnitude,
}

/// Run max projection on a TIFF file.
pub fn run_maxproj(
    input: &Path,
    output: &Path,
    mode: MaxProjMode,
) -> Result<(), crate::core::DwError> {
    let (img, meta) = tiff_io::tiff_read(input)?;

    let result = match mode {
        MaxProjMode::Max => img.max_projection(),
        MaxProjMode::Slice(z) => img.get_plane(z)?,
        MaxProjMode::Xyz => xyz_collage(&img)?,
        MaxProjMode::GradientMagnitude => best_focus_slice(&img)?,
    };

    let mut out_meta = meta;
    out_meta.image_description = None;
    tiff_io::tiff_write_u16(output, &result, Some(&out_meta), None)?;

    Ok(())
}

/// Create a 3-panel collage: XY max projection, XZ max projection (below), and
/// YZ max projection (to the right).
///
/// Layout (P=1 output):
///   +---------+-----+
///   |  XY     | YZ  |
///   |  (MxN)  | (PxN)|
///   +---------+-----+
///   |  XZ     |     |
///   |  (MxP)  |     |
///   +---------+-----+
fn xyz_collage(img: &FimImage) -> Result<FimImage, DwError> {
    let (m_dim, n_dim, p_dim) = img.dims();
    if m_dim == 0 || n_dim == 0 || p_dim == 0 {
        return Err(DwError::InvalidDimensions(
            "Cannot build XYZ collage from an empty image".into(),
        ));
    }

    // XY max projection: M x N
    let xy_proj = img.max_projection();

    // XZ max projection: M x P (project along Y/N axis)
    let mut xz_proj = FimImage::zeros(m_dim, p_dim, 1);
    for pp in 0..p_dim {
        for mm in 0..m_dim {
            let mut max_val = f32::NEG_INFINITY;
            for nn in 0..n_dim {
                let val = img.get(mm, nn, pp);
                if val > max_val {
                    max_val = val;
                }
            }
            xz_proj.set(mm, pp, 0, max_val);
        }
    }

    // YZ max projection: P x N (project along X/M axis)
    let mut yz_proj = FimImage::zeros(p_dim, n_dim, 1);
    for nn in 0..n_dim {
        for pp in 0..p_dim {
            let mut max_val = f32::NEG_INFINITY;
            for mm in 0..m_dim {
                let val = img.get(mm, nn, pp);
                if val > max_val {
                    max_val = val;
                }
            }
            yz_proj.set(pp, nn, 0, max_val);
        }
    }

    // Assemble collage: width = M + P, height = N + P
    let (out_m, out_n) = xyz_collage_dims(m_dim, n_dim, p_dim)?;
    let mut out = FimImage::zeros(out_m, out_n, 1);

    // Top-left: XY projection (M x N)
    for nn in 0..n_dim {
        for mm in 0..m_dim {
            out.set(mm, nn, 0, xy_proj.get(mm, nn, 0));
        }
    }

    // Top-right: YZ projection (P x N), placed at x-offset = M
    for nn in 0..n_dim {
        for pp in 0..p_dim {
            out.set(m_dim + pp, nn, 0, yz_proj.get(pp, nn, 0));
        }
    }

    // Bottom-left: XZ projection (M x P), placed at y-offset = N
    for pp in 0..p_dim {
        for mm in 0..m_dim {
            out.set(mm, n_dim + pp, 0, xz_proj.get(mm, pp, 0));
        }
    }

    Ok(out)
}

fn xyz_collage_dims(m: usize, n: usize, p: usize) -> Result<(usize, usize), DwError> {
    let out_m = m
        .checked_add(p)
        .ok_or_else(|| DwError::InvalidDimensions("XYZ collage width overflow".into()))?;
    let out_n = n
        .checked_add(p)
        .ok_or_else(|| DwError::InvalidDimensions("XYZ collage height overflow".into()))?;
    Ok((out_m, out_n))
}

/// Find and extract the most in-focus Z-slice using gradient magnitude.
///
/// The slice with the highest mean gradient magnitude is considered the most
/// in-focus.
fn best_focus_slice(img: &FimImage) -> Result<FimImage, DwError> {
    let (m_dim, n_dim, p_dim) = img.dims();

    if m_dim == 0 || n_dim == 0 || p_dim == 0 {
        return Err(DwError::InvalidDimensions(
            "Cannot select a focus slice from an empty image".into(),
        ));
    }

    if p_dim <= 1 {
        return Ok(img.clone());
    }

    // Compute gradient magnitude with a small smoothing sigma.
    let grad = img.gradient_magnitude(1.0);

    // Find the slice with the highest mean gradient magnitude.
    let mut best_plane = 0;
    let mut best_mean = f64::NEG_INFINITY;
    for pp in 0..p_dim {
        let mut sum = 0.0f64;
        for nn in 0..n_dim {
            for mm in 0..m_dim {
                sum += grad.get(mm, nn, pp) as f64;
            }
        }
        let mean = sum / (m_dim * n_dim) as f64;
        if mean > best_mean {
            best_mean = mean;
            best_plane = pp;
        }
    }

    log::info!(
        "Best focus slice: z={} (mean gradient magnitude={:.4})",
        best_plane,
        best_mean
    );

    // img.get_plane can't fail here since best_plane < p_dim.
    img.get_plane(best_plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyz_collage_dims_rejects_overflow() {
        assert_eq!(xyz_collage_dims(4, 3, 2).unwrap(), (6, 5));
        assert!(xyz_collage_dims(usize::MAX, 1, 1).is_err());
        assert!(xyz_collage_dims(1, usize::MAX, 1).is_err());
    }

    #[test]
    fn xyz_collage_rejects_empty_axes() {
        assert!(xyz_collage(&FimImage::zeros(0, 3, 2)).is_err());
        assert!(xyz_collage(&FimImage::zeros(3, 0, 2)).is_err());
        assert!(xyz_collage(&FimImage::zeros(3, 2, 0)).is_err());
    }

    #[test]
    fn best_focus_slice_rejects_empty_xy_plane() {
        let img = FimImage::zeros(0, 3, 2);
        assert!(best_focus_slice(&img).is_err());

        let img = FimImage::zeros(3, 0, 2);
        assert!(best_focus_slice(&img).is_err());
    }
}
