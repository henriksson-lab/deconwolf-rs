/// FFTW backend for 3D real-to-complex FFT.
///
/// Currently delegates to RustFftBackend with a compatibility wrapper.
/// TODO: Replace with actual FFTW3 bindings (via the `fftw` crate) when
/// the crate API is stable and FFTW3 is available on the build system.
/// To use actual FFTW3 bindings, install FFTW3 and update this module.

#[cfg(feature = "fftw-backend")]
use num_complex::Complex;
#[cfg(feature = "fftw-backend")]
use super::backend::{FftBackend, Result};
#[cfg(feature = "fftw-backend")]
use super::rustfft_backend::RustFftBackend;

/// FFTW-compatible FFT backend.
///
/// When the `fftw-backend` feature is enabled, this backend is selected
/// in the CLI. Currently it wraps [`RustFftBackend`] as a fallback until
/// proper FFTW3 C-library bindings are integrated.
#[cfg(feature = "fftw-backend")]
pub struct FftwBackend {
    inner: RustFftBackend,
}

#[cfg(feature = "fftw-backend")]
impl FftBackend for FftwBackend {
    fn new_context(m: usize, n: usize, p: usize, threads: usize) -> Result<Self> {
        log::warn!(
            "FFTW backend requested but actual FFTW3 bindings are not yet integrated. \
             Falling back to pure-Rust FFT (rustfft)."
        );
        Ok(Self {
            inner: RustFftBackend::new_context(m, n, p, threads)?,
        })
    }

    fn dims(&self) -> (usize, usize, usize) {
        self.inner.dims()
    }

    fn complex_size(&self) -> usize {
        self.inner.complex_size()
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<Complex<f32>>> {
        self.inner.forward(input)
    }

    fn inverse(&self, input: &[Complex<f32>]) -> Result<Vec<f32>> {
        self.inner.inverse(input)
    }
}
