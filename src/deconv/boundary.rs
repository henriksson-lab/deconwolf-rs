use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;

/// Compute Bertero boundary weights for reducing edge artifacts.
pub fn compute_bertero_weights<B: FftBackend>(
    m: usize,
    n: usize,
    p: usize,
    wm: usize,
    wn: usize,
    wp: usize,
    psf_fft: &[num_complex::Complex<f32>],
    fft: &FftContext<B>,
) -> Result<FimImage, crate::core::DwError> {
    let mut ones = FimImage::zeros(wm, wn, wp);
    for pp in 0..p {
        for nn in 0..n {
            for mm in 0..m {
                ones.set(mm, nn, pp, 1.0);
            }
        }
    }

    let ones_fft = fft
        .forward(ones.as_slice())
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let ht_ones = fft
        .convolve_conj(psf_fft, &ones_fft)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

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
