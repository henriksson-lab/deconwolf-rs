use std::path::Path;

use crate::core::tiff_io;
use crate::core::FimImage;

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
        MaxProjMode::Xyz => xyz_collage(&img),
        MaxProjMode::GradientMagnitude => best_focus_slice(&img),
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
fn xyz_collage(img: &FimImage) -> FimImage {
    let (m_dim, n_dim, p_dim) = img.dims();

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
    let out_m = m_dim + p_dim;
    let out_n = n_dim + p_dim;
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

    out
}

/// Find and extract the most in-focus Z-slice using gradient magnitude.
///
/// The slice with the highest mean gradient magnitude is considered the most
/// in-focus.
fn best_focus_slice(img: &FimImage) -> FimImage {
    let (m_dim, n_dim, p_dim) = img.dims();

    if p_dim <= 1 {
        return img.clone();
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
    img.get_plane(best_plane).expect("valid plane index")
}
