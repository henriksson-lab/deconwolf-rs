use std::path::Path;
use std::time::Instant;

use dw_core::tiff_io::{self, TiffMeta};
use dw_core::npy_io;
use dw_core::FimImage;
use dw_fft::backend::FftContext;
use dw_fft::FftBackend;

use crate::config::{DwOpts, Method, OutputFormat};
use crate::identity::IdentityMethod;
use crate::method::DeconvMethod;
use crate::rl::RlMethod;
use crate::shb::ShbMethod;

/// Run the full deconvolution pipeline.
pub fn dw_run<B: FftBackend>(opts: &DwOpts) -> Result<(), dw_core::DwError> {
    let start = Instant::now();

    // Load image
    if opts.verbosity > 0 {
        log::info!("Loading image: {:?}", opts.image_file);
    }
    let (mut image, meta) = load_image(&opts.image_file)?;
    let (m, n, p) = image.dims();
    if opts.verbosity > 0 {
        log::info!("Image size: {} x {} x {}", m, n, p);
    }

    // Z-cropping
    if opts.zcrop > 0 {
        image = image.zcrop(opts.zcrop)?;
        if opts.verbosity > 0 {
            log::info!("After zcrop: {} x {} x {}", image.m(), image.n(), image.p());
        }
    } else if opts.auto_zcrop > 0 {
        image = image.auto_zcrop(opts.auto_zcrop)?;
        if opts.verbosity > 0 {
            log::info!(
                "After auto_zcrop: {} x {} x {}",
                image.m(),
                image.n(),
                image.p()
            );
        }
    }

    // Add offset
    if opts.offset > 0.0 {
        image.add_scalar(opts.offset);
    }

    let (m, n, p) = image.dims();

    // Load PSF
    if opts.verbosity > 0 {
        log::info!("Loading PSF: {:?}", opts.psf_file);
    }
    let (mut psf, _) = load_image(&opts.psf_file)?;
    let (pm, pn, pp) = psf.dims();
    if opts.verbosity > 0 {
        log::info!("PSF size: {} x {} x {}", pm, pn, pp);
    }

    // Normalize PSF
    psf.normalize_sum1();

    // Check PSF is centered
    if !psf.max_at_origin() {
        log::warn!("PSF maximum is not at the center");
    }

    // Compute job size based on border quality
    let (wm, wn, wp) = compute_job_size(m, n, p, pm, pn, pp, opts.border_quality);
    if opts.verbosity > 0 {
        log::info!("Job size: {} x {} x {}", wm, wn, wp);
    }

    // Create FFT context
    let fft: FftContext<B> = FftContext::new(wm, wn, wp, opts.fft_threads())
        .map_err(|e| dw_core::DwError::Config(e.to_string()))?;

    // Run deconvolution
    let result = match opts.method {
        Method::Id => IdentityMethod.deconvolve(&image, &psf, opts, &fft)?,
        Method::Rl => RlMethod.deconvolve(&image, &psf, opts, &fft)?,
        Method::Shb => ShbMethod.deconvolve(&image, &psf, opts, &fft)?,
        #[cfg(feature = "gpu")]
        Method::ShbCl | Method::ShbCl2 => {
            return Err(dw_core::DwError::Config("GPU methods not yet implemented".into()));
        }
    };

    // Remove offset
    let mut result = result;
    if opts.offset > 0.0 {
        result.add_scalar(-opts.offset);
    }
    result.project_positive();

    // Write output
    let out_path = opts.out_file.clone().unwrap_or_else(|| {
        let stem = opts
            .image_file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        opts.image_file.with_file_name(format!("{}_{}.tif", opts.prefix, stem))
    });

    if opts.verbosity > 0 {
        log::info!("Writing output: {:?}", out_path);
    }

    match opts.output_format {
        OutputFormat::F32 => {
            tiff_io::tiff_write_f32(&out_path, &result, Some(&meta))?;
        }
        OutputFormat::U16 => {
            let scaling = if opts.scaling > 0.0 {
                Some(opts.scaling)
            } else {
                None
            };
            tiff_io::tiff_write_u16(&out_path, &result, Some(&meta), scaling)?;
        }
    }

    let elapsed = start.elapsed();
    if opts.verbosity > 0 {
        log::info!("Done in {:.2}s", elapsed.as_secs_f64());
    }

    Ok(())
}

fn load_image(path: &Path) -> Result<(FimImage, TiffMeta), dw_core::DwError> {
    if npy_io::is_npy_file(path) {
        let img = npy_io::npy_read(path)?;
        Ok((img, TiffMeta::default()))
    } else {
        tiff_io::tiff_read(path)
    }
}

/// Compute the job size (padded image dimensions for FFT) based on border quality.
fn compute_job_size(
    m: usize,
    n: usize,
    p: usize,
    pm: usize,
    pn: usize,
    pp: usize,
    border_quality: u8,
) -> (usize, usize, usize) {
    match border_quality {
        0 => {
            // Periodic: min dimensions
            (m.max(pm), n.max(pn), p.max(pp))
        }
        1 => {
            // Compromise: partial padding
            (m + pm / 2, n + pn / 2, p + pp / 2)
        }
        _ => {
            // Full: M + pM - 1
            (m + pm - 1, n + pn - 1, p + pp - 1)
        }
    }
}
