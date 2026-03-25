use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;

use super::config::DwOpts;

/// Trait for deconvolution methods.
pub trait DeconvMethod {
    fn deconvolve<B: FftBackend>(
        &self,
        image: &FimImage,
        psf: &FimImage,
        opts: &DwOpts,
        fft: &FftContext<B>,
    ) -> Result<FimImage, crate::core::DwError>;
}
