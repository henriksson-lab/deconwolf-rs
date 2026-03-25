use std::path::Path;

use crate::core::tiff_io;
use crate::core::DwError;

/// Sparse preprocessing with L1 regularization and total variation smoothness.
///
/// Uses a simplified ADMM approach:
/// 1. Soft thresholding to enforce sparsity (L1 penalty with weight `lambda`).
/// 2. Total variation smoothing to enforce spatial coherence (`lambda_s`).
///
/// The algorithm iterates `n_iter` times, producing a denoised version of the input.
pub fn run_sparse(
    input: &Path,
    output: &Path,
    lambda: f64,
    lambda_s: f64,
    n_iter: usize,
) -> Result<(), DwError> {
    let (img, meta) = tiff_io::tiff_read(input)?;
    let (m_dim, n_dim, p_dim) = img.dims();

    let lambda_f = lambda as f32;
    let lambda_s_f = lambda_s as f32;

    // Initialize u as a copy of the input image.
    let mut u = img.clone();

    for iter in 0..n_iter {
        // Step 1: Soft thresholding (L1 proximal operator).
        // v = u, then u = sign(v) * max(|v| - lambda, 0)
        {
            let slice = u.as_slice_mut();
            for v in slice.iter_mut() {
                let abs_v = v.abs();
                *v = v.signum() * (abs_v - lambda_f).max(0.0);
            }
        }

        // Step 2: Total variation smoothing step.
        // For each voxel, compute weighted average with neighbors:
        //   u_new = (image[i] + lambda_s * sum(u[neighbors])) / (1 + n_neighbors * lambda_s)
        let u_old = u.clone();
        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    let mut neighbor_sum = 0.0f32;
                    let mut n_neighbors = 0u32;

                    // 6-connected neighbors (face-adjacent).
                    if mm > 0 {
                        neighbor_sum += u_old.get(mm - 1, nn, pp);
                        n_neighbors += 1;
                    }
                    if mm + 1 < m_dim {
                        neighbor_sum += u_old.get(mm + 1, nn, pp);
                        n_neighbors += 1;
                    }
                    if nn > 0 {
                        neighbor_sum += u_old.get(mm, nn - 1, pp);
                        n_neighbors += 1;
                    }
                    if nn + 1 < n_dim {
                        neighbor_sum += u_old.get(mm, nn + 1, pp);
                        n_neighbors += 1;
                    }
                    if pp > 0 {
                        neighbor_sum += u_old.get(mm, nn, pp - 1);
                        n_neighbors += 1;
                    }
                    if pp + 1 < p_dim {
                        neighbor_sum += u_old.get(mm, nn, pp + 1);
                        n_neighbors += 1;
                    }

                    let data_term = img.get(mm, nn, pp);
                    let new_val = (data_term + lambda_s_f * neighbor_sum)
                        / (1.0 + n_neighbors as f32 * lambda_s_f);
                    u.set(mm, nn, pp, new_val);
                }
            }
        }

        log::debug!("Sparse iteration {}/{}", iter + 1, n_iter);
    }

    // Write output.
    let mut out_meta = meta;
    out_meta.image_description = None;
    tiff_io::tiff_write_f32(output, &u, Some(&out_meta))?;

    Ok(())
}
