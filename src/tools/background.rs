use std::path::{Path, PathBuf};

use crate::core::tiff_io;
use crate::core::DwError;

/// Estimate background by averaging multiple images and applying Gaussian low-pass filter.
///
/// Each input is max-projected to 2D (if 3D), then all projections are averaged.
/// Finally, a Gaussian smoothing with the given `sigma` is applied to produce a
/// smooth background / vignetting estimate.
pub fn run_background(
    output: &Path,
    inputs: &[PathBuf],
    sigma: f32,
) -> Result<(), DwError> {
    if inputs.is_empty() {
        return Err(DwError::Config("No input files provided".into()));
    }

    // Read and max-project the first image.
    let (first_img, first_meta) = tiff_io::tiff_read(&inputs[0])?;
    let mut acc = if first_img.p() > 1 {
        first_img.max_projection()
    } else {
        first_img
    };

    let ref_m = acc.m();
    let ref_n = acc.n();

    // Accumulate remaining images.
    for path in &inputs[1..] {
        let (img, _meta) = tiff_io::tiff_read(path)?;
        let proj = if img.p() > 1 {
            img.max_projection()
        } else {
            img
        };
        if proj.m() != ref_m || proj.n() != ref_n {
            return Err(DwError::InvalidDimensions(format!(
                "Image {:?} has dimensions {}x{}, expected {}x{}",
                path,
                proj.m(),
                proj.n(),
                ref_m,
                ref_n,
            )));
        }
        acc.add_image(&proj);
    }

    // Divide by number of images to get the average.
    let n_images = inputs.len() as f32;
    acc.mult_scalar(1.0 / n_images);

    // Apply Gaussian smoothing with given sigma.
    if sigma > 0.0 {
        acc.gsmooth(sigma);
    }

    // Write output 2D TIFF.
    let mut out_meta = first_meta;
    out_meta.image_description = None;
    tiff_io::tiff_write_f32(output, &acc, Some(&out_meta))?;

    Ok(())
}
