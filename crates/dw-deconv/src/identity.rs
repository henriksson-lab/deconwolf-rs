use dw_core::FimImage;
use dw_fft::backend::FftContext;
use dw_fft::FftBackend;

use crate::config::DwOpts;
use crate::method::DeconvMethod;

/// Identity (pass-through) method. Returns a copy of the input.
/// Used for testing the I/O pipeline.
pub struct IdentityMethod;

impl DeconvMethod for IdentityMethod {
    fn deconvolve<B: FftBackend>(
        &self,
        image: &FimImage,
        _psf: &FimImage,
        _opts: &DwOpts,
        _fft: &FftContext<B>,
    ) -> Result<FimImage, dw_core::DwError> {
        Ok(image.clone())
    }
}
