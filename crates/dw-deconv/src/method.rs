use dw_core::FimImage;
use dw_fft::backend::FftContext;
use dw_fft::FftBackend;

use crate::config::DwOpts;

/// Trait for deconvolution methods.
pub trait DeconvMethod {
    /// Run deconvolution on `image` with given `psf` and options.
    ///
    /// The PSF should already be normalized (sum = 1).
    /// Returns the deconvolved image.
    fn deconvolve<B: FftBackend>(
        &self,
        image: &FimImage,
        psf: &FimImage,
        opts: &DwOpts,
        fft: &FftContext<B>,
    ) -> Result<FimImage, dw_core::DwError>;
}
