use num_complex::Complex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FftError {
    #[error("FFT initialization error: {0}")]
    Init(String),
    #[error("FFT execution error: {0}")]
    Execution(String),
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
}

pub type Result<T> = std::result::Result<T, FftError>;

/// Trait for FFT backends (rustfft, FFTW, etc.).
pub trait FftBackend: Send + Sync {
    /// Create a new FFT context for given real dimensions.
    fn new_context(m: usize, n: usize, p: usize, threads: usize) -> Result<Self>
    where
        Self: Sized;

    /// Real dimensions.
    fn dims(&self) -> (usize, usize, usize);

    /// Number of complex elements in the frequency domain.
    /// For real-to-complex: (m/2 + 1) * n * p
    fn complex_size(&self) -> usize;

    /// Forward transform: real → complex.
    fn forward(&self, input: &[f32]) -> Result<Vec<Complex<f32>>>;

    /// Inverse transform: complex → real.
    /// Output is normalized by 1/(m*n*p).
    fn inverse(&self, input: &[Complex<f32>]) -> Result<Vec<f32>>;
}

/// High-level FFT operations built on an FftBackend.
pub struct FftContext<B: FftBackend> {
    pub backend: B,
}

impl<B: FftBackend> FftContext<B> {
    pub fn new(m: usize, n: usize, p: usize, threads: usize) -> Result<Self> {
        Ok(Self {
            backend: B::new_context(m, n, p, threads)?,
        })
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.backend.dims()
    }

    /// Forward FFT.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<Complex<f32>>> {
        self.backend.forward(input)
    }

    /// Inverse FFT (normalized).
    pub fn inverse(&self, input: &[Complex<f32>]) -> Result<Vec<f32>> {
        self.backend.inverse(input)
    }

    /// Convolve: IFFT(A * B).
    pub fn convolve(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
    ) -> Result<Vec<f32>> {
        let product = crate::complex_ops::complex_mul(a, b);
        self.inverse(&product)
    }

    /// Convolve with conjugate: IFFT(conj(A) * B).
    pub fn convolve_conj(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
    ) -> Result<Vec<f32>> {
        let product = crate::complex_ops::complex_mul_conj(a, b);
        self.inverse(&product)
    }
}
