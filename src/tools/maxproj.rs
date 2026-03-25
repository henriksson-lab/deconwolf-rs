use std::path::Path;

use crate::core::tiff_io;

/// Maximum projection modes.
pub enum MaxProjMode {
    /// Standard max projection along Z.
    Max,
    /// Extract a specific Z-slice.
    Slice(usize),
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
    };

    let mut out_meta = meta;
    out_meta.image_description = None;
    tiff_io::tiff_write_u16(output, &result, Some(&out_meta), None)?;

    Ok(())
}
