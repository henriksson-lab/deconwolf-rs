use std::path::{Path, PathBuf};

use crate::core::tiff_io;
use crate::core::{DwError, FimImage};

/// Merge multiple 2D TIFF files into a single 3D TIFF stack.
///
/// Each input TIFF contributes one or more Z-planes. All inputs must share the
/// same M x N (width x height) dimensions.
pub fn run_merge(output: &Path, inputs: &[PathBuf]) -> Result<(), DwError> {
    if inputs.is_empty() {
        return Err(DwError::InvalidDimensions(
            "No input files provided for merge".into(),
        ));
    }

    // Read the first image to get reference dimensions and metadata.
    let (first_img, first_meta) = tiff_io::tiff_read(&inputs[0])?;
    let ref_m = first_img.m();
    let ref_n = first_img.n();

    // Collect all images, verifying dimensions.
    let mut images: Vec<FimImage> = Vec::with_capacity(inputs.len());
    images.push(first_img);

    for path in &inputs[1..] {
        let (img, _meta) = tiff_io::tiff_read(path)?;
        if img.m() != ref_m || img.n() != ref_n {
            return Err(DwError::InvalidDimensions(format!(
                "Image {:?} has dimensions {}x{}, expected {}x{}",
                path,
                img.m(),
                img.n(),
                ref_m,
                ref_n,
            )));
        }
        images.push(img);
    }

    // Compute total number of Z-planes.
    let total_p: usize = images.iter().map(|img| img.p()).sum();

    // Concatenate all planes into one volume.
    let mut out = FimImage::zeros(ref_m, ref_n, total_p);
    let mut z_offset = 0;
    for img in &images {
        for pp in 0..img.p() {
            for nn in 0..ref_n {
                for mm in 0..ref_m {
                    out.set(mm, nn, z_offset + pp, img.get(mm, nn, pp));
                }
            }
        }
        z_offset += img.p();
    }

    // Write output with metadata from the first image.
    let mut out_meta = first_meta;
    out_meta.image_description = None; // Will be regenerated for the new stack size
    tiff_io::tiff_write_f32(output, &out, Some(&out_meta))?;

    Ok(())
}
