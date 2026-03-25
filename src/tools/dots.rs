use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::core::tiff_io;
use crate::core::DwError;

/// A detected dot with its 3D coordinates and intensity.
#[derive(Debug, Clone)]
struct Dot {
    x: usize,
    y: usize,
    z: usize,
    intensity: f32,
}

/// Detect dots in a 3D image using Laplacian of Gaussian filtering.
///
/// The LoG sigma values are computed from the optical parameters:
///   sigma_xy = 0.21 * lambda / (NA * dx)
///   sigma_z  = 0.66 * lambda / (ni - sqrt(ni^2 - NA^2)) / dz
///
/// Local maxima (26-connected neighborhood) are found in the LoG-filtered image,
/// sorted by descending intensity, and optionally truncated to `n_dots`.
///
/// Results are written as a TSV or CSV text file.
pub fn run_dots(
    input: &Path,
    output: &Path,
    na: f64,
    ni: f64,
    lambda: f64,
    dx: f64,
    dz: f64,
    n_dots: Option<usize>,
    csv_format: bool,
) -> Result<(), DwError> {
    let (img, _meta) = tiff_io::tiff_read(input)?;

    // Compute sigma values from optical parameters.
    let sigma_xy = (0.21 * lambda / (na * dx)) as f32;
    let ni2_na2 = ni * ni - na * na;
    if ni2_na2 < 0.0 {
        return Err(DwError::Config(format!(
            "ni ({}) must be >= NA ({})",
            ni, na
        )));
    }
    let sigma_z = (0.66 * lambda / ((ni - ni2_na2.sqrt()) * dz)) as f32;

    log::info!(
        "LoG filter sigmas: sigma_xy={:.3}, sigma_z={:.3}",
        sigma_xy,
        sigma_z
    );

    // Apply LoG filter. The result is negative at bright spots.
    let log_img = img.log_filter(sigma_xy, sigma_z);

    // Find local maxima in the *negated* LoG (bright spots produce negative LoG values).
    let (m_dim, n_dim, p_dim) = log_img.dims();
    let mut dots: Vec<Dot> = Vec::new();

    if p_dim == 1 {
        // 2D case: 8-connected neighborhood.
        let pp = 0;
        for nn in 1..n_dim.saturating_sub(1) {
            for mm in 1..m_dim.saturating_sub(1) {
                let val = -log_img.get(mm, nn, pp);
                if val <= 0.0 {
                    continue;
                }
                let mut is_max = true;
                'check_2d: for dn in (nn - 1)..=(nn + 1) {
                    for dm in (mm - 1)..=(mm + 1) {
                        if dm == mm && dn == nn {
                            continue;
                        }
                        if -log_img.get(dm, dn, pp) >= val {
                            is_max = false;
                            break 'check_2d;
                        }
                    }
                }
                if is_max {
                    dots.push(Dot {
                        x: mm,
                        y: nn,
                        z: pp,
                        intensity: val,
                    });
                }
            }
        }
    } else {
        // 3D case: 26-connected neighborhood.
        for pp in 1..p_dim.saturating_sub(1) {
            for nn in 1..n_dim.saturating_sub(1) {
                for mm in 1..m_dim.saturating_sub(1) {
                    let val = -log_img.get(mm, nn, pp);
                    if val <= 0.0 {
                        continue;
                    }

                    let mut is_max = true;
                    'check_3d: for dp in (pp - 1)..=(pp + 1) {
                        for dn in (nn - 1)..=(nn + 1) {
                            for dm in (mm - 1)..=(mm + 1) {
                                if dm == mm && dn == nn && dp == pp {
                                    continue;
                                }
                                if -log_img.get(dm, dn, dp) >= val {
                                    is_max = false;
                                    break 'check_3d;
                                }
                            }
                        }
                    }

                    if is_max {
                        dots.push(Dot {
                            x: mm,
                            y: nn,
                            z: pp,
                            intensity: val,
                        });
                    }
                }
            }
        }
    }

    // Sort by intensity, descending.
    dots.sort_by(|a, b| {
        b.intensity
            .partial_cmp(&a.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to n_dots if requested.
    if let Some(limit) = n_dots {
        dots.truncate(limit);
    }

    log::info!("Detected {} dots", dots.len());

    // Write output file.
    let sep = if csv_format { "," } else { "\t" };
    let mut file = File::create(output)?;
    writeln!(file, "x{}y{}z{}intensity", sep, sep, sep)
        .map_err(|e| DwError::Io(e))?;
    for dot in &dots {
        writeln!(
            file,
            "{}{}{}{}{}{}{}",
            dot.x, sep, dot.y, sep, dot.z, sep, dot.intensity
        )
        .map_err(|e| DwError::Io(e))?;
    }

    Ok(())
}
