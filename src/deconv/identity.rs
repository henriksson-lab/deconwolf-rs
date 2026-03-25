use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;

use super::config::DwOpts;
use super::method::DeconvMethod;

/// Identity (pass-through) method. Returns a copy of the input.
pub struct IdentityMethod;

impl DeconvMethod for IdentityMethod {
    fn deconvolve<B: FftBackend>(
        &self,
        image: &FimImage,
        _psf: &FimImage,
        _opts: &DwOpts,
        _fft: &FftContext<B>,
    ) -> Result<FimImage, crate::core::DwError> {
        Ok(image.clone())
    }
}
