use std::path::Path;
use std::time::Instant;

use crate::core::tiff_io::{self, TiffMeta};
use crate::core::npy_io;
use crate::core::tiling::Tiling;
use crate::core::FimImage;
use crate::fft::backend::FftContext;
use crate::fft::FftBackend;

use super::config::{DwOpts, Method, OutputFormat};
use super::identity::IdentityMethod;
use super::method::DeconvMethod;
use super::rl::RlMethod;
use super::shb::ShbMethod;

/// Run the full deconvolution pipeline.
pub fn dw_run<B: FftBackend>(opts: &DwOpts) -> Result<(), crate::core::DwError> {
    let start = Instant::now();

    if opts.verbosity > 0 {
        log::info!("Loading image: {:?}", opts.image_file);
    }
    let (mut image, meta) = load_image(&opts.image_file)?;
    let (m, n, p) = image.dims();
    if opts.verbosity > 0 {
        log::info!("Image size: {} x {} x {}", m, n, p);
    }

    if opts.zcrop > 0 {
        image = image.zcrop(opts.zcrop)?;
        if opts.verbosity > 0 {
            log::info!("After zcrop: {} x {} x {}", image.m(), image.n(), image.p());
        }
    } else if opts.auto_zcrop > 0 {
        image = image.auto_zcrop(opts.auto_zcrop)?;
        if opts.verbosity > 0 {
            log::info!("After auto_zcrop: {} x {} x {}", image.m(), image.n(), image.p());
        }
    }

    // Flat-field correction
    if let Some(ref ff_path) = opts.flatfield_file {
        if opts.verbosity > 0 {
            log::info!("Applying flat-field correction: {:?}", ff_path);
        }
        let (ff, _) = load_image(ff_path)?;
        let mn = image.m() * image.n();
        let num_p = image.p();
        let ff_p = ff.p();
        let ff_data = ff.as_slice().to_vec();
        let im_slice = image.as_slice_mut();
        for pp in 0..num_p {
            for i in 0..mn {
                let ff_val = if ff_p == 1 {
                    ff_data[i]
                } else {
                    ff_data[pp * mn + i]
                };
                if ff_val > 0.0 {
                    im_slice[pp * mn + i] /= ff_val;
                }
            }
        }
    }

    // Pre-filter with Gaussian (Anscombe domain)
    if opts.psigma > 0.0 {
        if opts.verbosity > 0 {
            log::info!("Pre-filtering with sigma={}", opts.psigma);
        }
        image.anscombe();
        image.gsmooth(opts.psigma);
        image.ianscombe();
    }

    if opts.offset > 0.0 {
        image.add_scalar(opts.offset);
    }

    let (m, n, p) = image.dims();

    if opts.verbosity > 0 {
        log::info!("Loading PSF: {:?}", opts.psf_file);
    }
    let (mut psf, _) = load_image(&opts.psf_file)?;
    let (pm, pn, pp) = psf.dims();
    if opts.verbosity > 0 {
        log::info!("PSF size: {} x {} x {}", pm, pn, pp);
    }

    psf.normalize_sum1();

    if !psf.max_at_origin() {
        log::warn!("PSF maximum is not at the center");
    }

    let use_tiling = opts.tiling_max_size > 0
        && (m as i64 > opts.tiling_max_size || n as i64 > opts.tiling_max_size);

    let mut result = if use_tiling {
        let tiling = Tiling::new(
            m, n, p,
            opts.tiling_max_size as usize,
            opts.tiling_padding,
        );
        log::info!(
            "Using tiled processing ({} tiles, max_size={}, padding={})",
            tiling.num_tiles(),
            opts.tiling_max_size,
            opts.tiling_padding,
        );
        deconvolve_tiled::<B>(&image, &psf, opts, &tiling)?
    } else {
        let (wm, wn, wp) = compute_job_size(m, n, p, pm, pn, pp, opts.border_quality);
        if opts.verbosity > 0 {
            log::info!("Job size: {} x {} x {}", wm, wn, wp);
        }

        let fft: FftContext<B> = FftContext::new(wm, wn, wp, opts.fft_threads())
            .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

        match opts.method {
            Method::Id => IdentityMethod.deconvolve(&image, &psf, opts, &fft)?,
            Method::Rl => RlMethod.deconvolve(&image, &psf, opts, &fft)?,
            Method::Shb => ShbMethod.deconvolve(&image, &psf, opts, &fft)?,
        }
    };
    if opts.offset > 0.0 {
        result.add_scalar(-opts.offset);
    }
    result.project_positive();

    let out_path = opts.out_file.clone().unwrap_or_else(|| {
        let stem = opts.image_file.file_stem().unwrap_or_default().to_string_lossy();
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
            let scaling = if opts.scaling > 0.0 { Some(opts.scaling) } else { None };
            tiff_io::tiff_write_u16(&out_path, &result, Some(&meta), scaling)?;
        }
    }

    let elapsed = start.elapsed();
    if opts.verbosity > 0 {
        log::info!("Done in {:.2}s", elapsed.as_secs_f64());
    }

    Ok(())
}

fn load_image(path: &Path) -> Result<(FimImage, TiffMeta), crate::core::DwError> {
    if npy_io::is_npy_file(path) {
        let img = npy_io::npy_read(path)?;
        Ok((img, TiffMeta::default()))
    } else {
        tiff_io::tiff_read(path)
    }
}

/// Crop a PSF to fit within the given tile dimensions, keeping the center.
fn crop_psf_to_fit(psf: &FimImage, tile_m: usize, tile_n: usize, tile_p: usize) -> FimImage {
    let (pm, pn, pp) = psf.dims();
    let new_m = pm.min(tile_m);
    let new_n = pn.min(tile_n);
    let new_p = pp.min(tile_p);
    if new_m == pm && new_n == pn && new_p == pp {
        return psf.clone();
    }
    let m0 = (pm - new_m) / 2;
    let n0 = (pn - new_n) / 2;
    let p0 = (pp - new_p) / 2;
    let mut cropped = psf
        .get_cuboid(m0, m0 + new_m, n0, n0 + new_n, p0, p0 + new_p)
        .expect("PSF crop dimensions should be valid");
    cropped.normalize_sum1();
    cropped
}

/// Run tiled deconvolution: split the image into overlapping tiles,
/// deconvolve each tile independently, and blend the results back together.
fn deconvolve_tiled<B: FftBackend>(
    image: &FimImage,
    psf: &FimImage,
    opts: &DwOpts,
    tiling: &Tiling,
) -> Result<FimImage, crate::core::DwError> {
    let (m, n, p) = image.dims();
    let mut output = FimImage::zeros(m, n, p);
    let mut weights = FimImage::zeros(m, n, p);

    for tile_idx in 0..tiling.num_tiles() {
        let tile = &tiling.tiles[tile_idx];
        if opts.verbosity > 0 {
            log::info!(
                "Processing tile {}/{} ({}x{}x{})",
                tile_idx + 1,
                tiling.num_tiles(),
                tile.xsize[0],
                tile.xsize[1],
                tile.xsize[2],
            );
        }

        // Extract tile from input image
        let tile_image = tiling.extract_tile(image, tile_idx)?;
        let (tm, tn, tp) = tile_image.dims();

        // Crop PSF if it's larger than the tile
        let tile_psf = crop_psf_to_fit(psf, tm, tn, tp);
        let (tpm, tpn, tpp) = tile_psf.dims();

        // Compute job size for this tile
        let (wm, wn, wp) = compute_job_size(tm, tn, tp, tpm, tpn, tpp, opts.border_quality);

        // Create FFT context for this tile's dimensions
        let fft: FftContext<B> = FftContext::new(wm, wn, wp, opts.fft_threads())
            .map_err(|e| crate::core::DwError::Config(e.to_string()))?;

        // Deconvolve the tile
        let tile_result = match opts.method {
            Method::Id => IdentityMethod.deconvolve(&tile_image, &tile_psf, opts, &fft)?,
            Method::Rl => RlMethod.deconvolve(&tile_image, &tile_psf, opts, &fft)?,
            Method::Shb => ShbMethod.deconvolve(&tile_image, &tile_psf, opts, &fft)?,
        };

        // Blend tile result back into output
        tiling.blend_tile(&mut output, &mut weights, tile_idx, &tile_result);
    }

    // Finalize: divide by accumulated weights
    Tiling::finalize(&mut output, &weights);

    Ok(output)
}

fn compute_job_size(
    m: usize, n: usize, p: usize,
    pm: usize, pn: usize, pp: usize,
    border_quality: u8,
) -> (usize, usize, usize) {
    match border_quality {
        0 => (m.max(pm), n.max(pn), p.max(pp)),
        1 => (m + pm / 2, n + pn / 2, p + pp / 2),
        _ => (m + pm - 1, n + pn - 1, p + pp - 1),
    }
}
