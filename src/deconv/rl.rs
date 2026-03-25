use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;
use num_complex::Complex;

use super::boundary::compute_bertero_weights;
use super::config::{DwOpts, Metric};
use super::iterator::DwIterator;
use super::method::DeconvMethod;

/// Richardson-Lucy deconvolution method.
pub struct RlMethod;

impl DeconvMethod for RlMethod {
    fn deconvolve<B: FftBackend>(
        &self,
        image: &FimImage,
        psf: &FimImage,
        opts: &DwOpts,
        fft: &FftContext<B>,
    ) -> Result<FimImage, crate::core::DwError> {
        let (m, n, p) = image.dims();
        let (pm, pn, pp) = psf.dims();
        let (wm, wn, wp) = fft.dims();

        let mut psf_expanded = psf.expand(wm, wn, wp);
        let shift_m = -(pm as i64 / 2);
        let shift_n = -(pn as i64 / 2);
        let shift_p = -(pp as i64 / 2);
        psf_expanded.circshift(shift_m, shift_n, shift_p);

        let psf_fft = fft
            .forward(psf_expanded.as_slice())
            .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

        let weights = if opts.border_quality > 0 {
            Some(compute_bertero_weights::<B>(
                m, n, p, wm, wn, wp, &psf_fft, fft,
            )?)
        } else {
            None
        };

        let mut x = initial_guess(image, opts, wm, wn, wp);

        let mut iter = DwIterator::new(
            opts.iter_type,
            opts.n_iter,
            opts.max_iter,
            opts.rel_error,
            opts.abs_error,
        );

        while iter.next() {
            let error = rl_iteration(
                &mut x,
                image,
                &psf_fft,
                weights.as_ref(),
                fft,
                opts,
                m, n, p, wm, wn, wp,
            )?;
            iter.set_error(error);

            if opts.verbosity > 0 {
                log::info!("RL iteration {}: error = {:.6}", iter.current(), error);
            }
        }

        x.subregion(m, n, p)
    }
}

fn initial_guess(image: &FimImage, opts: &DwOpts, wm: usize, wn: usize, wp: usize) -> FimImage {
    match opts.start_condition {
        super::config::StartCondition::Identity => image.expand(wm, wn, wp),
        _ => {
            let mean = image.mean() as f32;
            FimImage::constant(wm, wn, wp, mean.max(opts.bg))
        }
    }
}

fn rl_iteration<B: FftBackend>(
    x: &mut FimImage,
    image: &FimImage,
    psf_fft: &[Complex<f32>],
    weights: Option<&FimImage>,
    fft: &FftContext<B>,
    opts: &DwOpts,
    m: usize, n: usize, p: usize,
    wm: usize, wn: usize, wp: usize,
) -> Result<f64, crate::core::DwError> {
    let x_fft = fft
        .forward(x.as_slice())
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let y_data = fft
        .convolve(psf_fft, &x_fft)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;
    let mut y = FimImage::from_vec(wm, wn, wp, y_data)?;

    let error = compute_error(image, &y, m, n, p, opts.metric);

    {
        let y_slice = y.as_slice_mut();
        let im_slice = image.as_slice();
        for pp in 0..p {
            for nn in 0..n {
                for mm in 0..m {
                    let idx_y = pp * wn * wm + nn * wm + mm;
                    let idx_im = pp * n * m + nn * m + mm;
                    let denom = y_slice[idx_y];
                    y_slice[idx_y] = if denom > 0.0 { im_slice[idx_im] / denom } else { 0.0 };
                }
            }
        }
        for pp in 0..wp {
            for nn in 0..wn {
                for mm in 0..wm {
                    if pp >= p || nn >= n || mm >= m {
                        y_slice[pp * wn * wm + nn * wm + mm] = 0.0;
                    }
                }
            }
        }
    }

    let ratio_fft = fft
        .forward(y.as_slice())
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let correction_data = fft
        .convolve_conj(psf_fft, &ratio_fft)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let x_slice = x.as_slice_mut();
    let bg = opts.bg;
    for i in 0..x_slice.len() {
        let mut update = x_slice[i] * correction_data[i];
        if let Some(w) = weights {
            update *= w.as_slice()[i];
        }
        x_slice[i] = update.max(bg);
    }

    Ok(error)
}

fn compute_error(image: &FimImage, y: &FimImage, m: usize, n: usize, p: usize, metric: Metric) -> f64 {
    let im = image.as_slice();
    let y_slice = y.as_slice();
    let (wm, wn, _) = y.dims();

    match metric {
        Metric::Mse => {
            let mut sum = 0.0f64;
            let mut count = 0;
            for pp in 0..p {
                for nn in 0..n {
                    for mm in 0..m {
                        let d = im[pp * n * m + nn * m + mm] as f64 - y_slice[pp * wn * wm + nn * wm + mm] as f64;
                        sum += d * d;
                        count += 1;
                    }
                }
            }
            sum / count as f64
        }
        Metric::Idiv => {
            let mut sum = 0.0f64;
            for pp in 0..p {
                for nn in 0..n {
                    for mm in 0..m {
                        let a = im[pp * n * m + nn * m + mm] as f64;
                        let b = y_slice[pp * wn * wm + nn * wm + mm] as f64;
                        if a > 0.0 && b > 0.0 {
                            sum += a * (a / b).ln() - a + b;
                        } else {
                            sum += b;
                        }
                    }
                }
            }
            sum
        }
    }
}
