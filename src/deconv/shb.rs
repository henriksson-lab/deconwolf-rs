use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;
use num_complex::Complex;
use rayon::prelude::*;

use super::boundary::compute_bertero_weights;
use super::config::{DwOpts, Metric, StartCondition};
use super::iterator::DwIterator;
use super::method::DeconvMethod;

/// Scaled Heavy Ball (SHB) deconvolution method.
///
/// Accelerated variant of Richardson-Lucy using Nesterov momentum.
pub struct ShbMethod;

impl DeconvMethod for ShbMethod {
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

        let mean_val = image.mean() as f32;
        let init_val = mean_val.max(opts.bg);
        let mut x = match opts.start_condition {
            StartCondition::Identity => image.expand(wm, wn, wp),
            _ => FimImage::constant(wm, wn, wp, init_val),
        };
        let mut x_prev = x.clone();

        let mut iter = DwIterator::new(
            opts.iter_type,
            opts.n_iter,
            opts.max_iter,
            opts.rel_error,
            opts.abs_error,
        );

        while iter.next() {
            let k = iter.current();
            let error = shb_iteration(
                &mut x, &mut x_prev, image, &psf_fft,
                weights.as_ref(), fft, opts, k,
                m, n, p, wm, wn, wp,
            )?;
            iter.set_error(error);

            if opts.verbosity > 0 {
                log::info!("SHB iteration {}: error = {:.6}", iter.current(), error);
            }
        }

        x.subregion(m, n, p)
    }
}

fn shb_iteration<B: FftBackend>(
    x: &mut FimImage,
    x_prev: &mut FimImage,
    image: &FimImage,
    psf_fft: &[Complex<f32>],
    weights: Option<&FimImage>,
    fft: &FftContext<B>,
    opts: &DwOpts,
    k: usize,
    m: usize, n: usize, p: usize,
    wm: usize, wn: usize, wp: usize,
) -> Result<f64, crate::core::DwError> {
    let total = wm * wn * wp;

    let alpha = if k > 1 {
        ((k as f32 - 1.0) / (k as f32 + 2.0)).min(opts.alpha_max)
    } else {
        0.0
    };

    let mut pk_data = vec![0.0f32; total];
    let x_slice = x.as_slice();
    let xp_slice = x_prev.as_slice();
    pk_data
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, v)| {
            *v = (x_slice[i] + alpha * (x_slice[i] - xp_slice[i])).max(0.0);
        });

    let pk_fft = fft
        .forward(&pk_data)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let y_data = fft
        .convolve(psf_fft, &pk_fft)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let error = compute_error_shb(image, &y_data, m, n, p, wm, wn, opts.metric);

    let mut ratio = vec![0.0f32; total];
    for pp_idx in 0..p {
        for nn in 0..n {
            for mm in 0..m {
                let idx_y = pp_idx * wn * wm + nn * wm + mm;
                let idx_im = pp_idx * n * m + nn * m + mm;
                let denom = y_data[idx_y];
                ratio[idx_y] = if denom > 0.0 {
                    image.as_slice()[idx_im] / denom
                } else {
                    0.0
                };
            }
        }
    }

    let ratio_fft = fft
        .forward(&ratio)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let correction = fft
        .convolve_conj(psf_fft, &ratio_fft)
        .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

    let x_slice_mut = x.as_slice_mut();
    let xp_slice_mut = x_prev.as_slice_mut();
    let bg = opts.bg;
    for i in 0..total {
        xp_slice_mut[i] = x_slice_mut[i];
        let mut update = pk_data[i] * correction[i];
        if let Some(w) = weights {
            update *= w.as_slice()[i];
        }
        x_slice_mut[i] = update.max(bg);
    }

    Ok(error)
}

fn compute_error_shb(
    image: &FimImage,
    y: &[f32],
    m: usize, n: usize, p: usize,
    wm: usize, wn: usize,
    metric: Metric,
) -> f64 {
    let im = image.as_slice();
    match metric {
        Metric::Mse => {
            let mut sum = 0.0f64;
            let mut count = 0;
            for pp in 0..p {
                for nn in 0..n {
                    for mm in 0..m {
                        let d = im[pp * n * m + nn * m + mm] as f64
                            - y[pp * wn * wm + nn * wm + mm] as f64;
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
                        let b = y[pp * wn * wm + nn * wm + mm] as f64;
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
