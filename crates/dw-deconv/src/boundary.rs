use dw_core::FimImage;
use dw_fft::backend::FftContext;
use dw_fft::FftBackend;

/// Compute Bertero boundary weights for reducing edge artifacts.
///
/// W = 1 / (H^T * 1), where H^T is the adjoint (conjugate) PSF convolution
/// and 1 is an all-ones image of the original size embedded in the job size.
pub fn compute_bertero_weights<B: FftBackend>(
    m: usize,
    n: usize,
    p: usize,
    wm: usize,
    wn: usize,
    wp: usize,
    psf_fft: &[num_complex::Complex<f32>],
    fft: &FftContext<B>,
) -> Result<FimImage, dw_core::DwError> {
    // Create all-ones image of original size, embedded in job size
    let mut ones = FimImage::zeros(wm, wn, wp);
    for pp in 0..p {
        for nn in 0..n {
            for mm in 0..m {
                ones.set(mm, nn, pp, 1.0);
            }
        }
    }

    // Forward FFT of ones
    let ones_fft = fft
        .forward(ones.as_slice())
        .map_err(|e| dw_core::DwError::Config(e.to_string()))?;

    // Convolve with conjugate of PSF: H^T * 1
    let ht_ones = fft
        .convolve_conj(psf_fft, &ones_fft)
        .map_err(|e| dw_core::DwError::Config(e.to_string()))?;

    // W = 1 / (H^T * 1), with a minimum to avoid division by zero
    let mut weights = FimImage::from_vec(wm, wn, wp, ht_ones)?;
    let max_val = weights.max();
    let threshold = max_val * 1e-5;
    let w = weights.as_slice_mut();
    for v in w.iter_mut() {
        if *v > threshold {
            *v = 1.0 / *v;
        } else {
            *v = 0.0;
        }
    }

    Ok(weights)
}
