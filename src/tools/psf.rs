use std::f64::consts::PI;

use rayon::prelude::*;

use crate::core::error::{DwError, Result};
use crate::core::FimImage;

// ---------------------------------------------------------------------------
// Bessel J0 approximation (Abramowitz & Stegun, formulas 9.4.1 / 9.4.3)
// ---------------------------------------------------------------------------

/// Compute the Bessel function of the first kind, order zero.
///
/// Uses rational polynomial approximations from Abramowitz & Stegun.
/// Accurate to roughly 1e-7 for all x.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7
                    + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0
            + y * (1029532985.0
                + y * (9494680.718
                    + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let p0 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4
                    + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q0 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5
                    + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        (0.636619772 / ax).sqrt() * (xx.cos() * p0 - z * xx.sin() * q0)
    }
}

// ---------------------------------------------------------------------------
// Simpson's rule integration for complex-valued integrands
// ---------------------------------------------------------------------------

/// Integrate a complex-valued function f over [a, b] using composite Simpson's
/// rule with `n` sub-intervals (n must be even; will be rounded up if odd).
///
/// `f` returns (real_part, imag_part).
fn simpson_integrate<F: Fn(f64) -> (f64, f64)>(
    f: F,
    a: f64,
    b: f64,
    n: usize,
) -> (f64, f64) {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let (mut sum_r, mut sum_i) = {
        let (fa_r, fa_i) = f(a);
        let (fb_r, fb_i) = f(b);
        (fa_r + fb_r, fa_i + fb_i)
    };
    for j in 1..n {
        let t = a + j as f64 * h;
        let (vr, vi) = f(t);
        let w = if j % 2 == 0 { 2.0 } else { 4.0 };
        sum_r += w * vr;
        sum_i += w * vi;
    }
    (sum_r * h / 3.0, sum_i * h / 3.0)
}

// ---------------------------------------------------------------------------
// Born-Wolf radial integral
// ---------------------------------------------------------------------------

/// Compute the Born-Wolf PSF intensity at lateral distance `r` (in nm) and
/// axial distance `z` (in nm) from the focal point.
///
/// Uses numerical integration of
///   I(r,z) = |integral_0^1 J0(k*NA*r*rho) * exp(i*k*z*ni*sqrt(1-(NA*rho/ni)^2)) * rho d_rho|^2
///
/// where k = 2*pi/lambda.
fn born_wolf_radial(r: f64, z: f64, na: f64, ni: f64, k: f64, n_points: usize) -> f64 {
    let (re, im) = simpson_integrate(
        |rho: f64| {
            let j0_val = bessel_j0(k * na * r * rho);
            let sin_arg = na * rho / ni;
            // Clamp for numerical safety near the edge of the aperture
            let cos_theta = if sin_arg.abs() >= 1.0 {
                0.0
            } else {
                (1.0 - sin_arg * sin_arg).sqrt()
            };
            let phase = k * z * ni * cos_theta;
            let amplitude = j0_val * rho;
            (amplitude * phase.cos(), amplitude * phase.sin())
        },
        0.0,
        1.0,
        n_points,
    );
    re * re + im * im
}

// ---------------------------------------------------------------------------
// Widefield PSF generation
// ---------------------------------------------------------------------------

/// Generate a widefield PSF using scalar diffraction theory (Born-Wolf model).
///
/// # Arguments
/// * `na`      - Numerical aperture
/// * `ni`      - Refractive index of immersion medium
/// * `lambda`  - Emission wavelength in nm
/// * `dx`      - Lateral pixel size in nm
/// * `dz`      - Axial step size in nm
/// * `size_xy` - Lateral extent in pixels (must be odd)
/// * `size_z`  - Number of z-planes (must be odd)
///
/// Returns a normalised 3-D PSF stored as a `FimImage`.
pub fn generate_widefield_psf(
    na: f64,
    ni: f64,
    lambda: f64,
    dx: f64,
    dz: f64,
    size_xy: usize,
    size_z: usize,
) -> Result<FimImage> {
    if size_xy % 2 == 0 || size_z % 2 == 0 {
        return Err(DwError::InvalidDimensions(
            "PSF dimensions must be odd".into(),
        ));
    }
    if na <= 0.0 || ni <= 0.0 || lambda <= 0.0 || dx <= 0.0 || dz <= 0.0 {
        return Err(DwError::Config(
            "All physical parameters must be positive".into(),
        ));
    }

    let k = 2.0 * PI / lambda;
    let center_xy = (size_xy / 2) as f64;
    let center_z = (size_z / 2) as f64;
    let n_integration = 1000_usize;

    // Compute the radial profile for each z-plane in parallel, then fill the
    // 2-D slice from that profile.

    // Maximum radial distance we need (corner of the slice in nm).
    let max_r_pix = center_xy * std::f64::consts::SQRT_2;
    let max_r_samples = (max_r_pix.ceil() as usize) + 2;

    // Each z-plane is independent -> parallelise over z.
    let planes: Vec<Vec<f32>> = (0..size_z)
        .into_par_iter()
        .map(|pz| {
            let z_nm = (pz as f64 - center_z) * dz;

            // 1. Build radial profile (sampled at integer pixel radii)
            let radial: Vec<f64> = (0..max_r_samples)
                .map(|ir| {
                    let r_nm = ir as f64 * dx;
                    born_wolf_radial(r_nm, z_nm, na, ni, k, n_integration)
                })
                .collect();

            // 2. Fill 2-D slice using linear interpolation from radial profile
            let mut plane = vec![0.0f32; size_xy * size_xy];
            for iy in 0..size_xy {
                for ix in 0..size_xy {
                    let rx = ix as f64 - center_xy;
                    let ry = iy as f64 - center_xy;
                    let r = (rx * rx + ry * ry).sqrt();
                    let ri = r.floor() as usize;
                    let frac = r - ri as f64;
                    let val = if ri + 1 < radial.len() {
                        radial[ri] * (1.0 - frac) + radial[ri + 1] * frac
                    } else if ri < radial.len() {
                        radial[ri] * (1.0 - frac)
                    } else {
                        0.0
                    };
                    plane[iy * size_xy + ix] = val as f32;
                }
            }
            plane
        })
        .collect();

    // Flatten into a single buffer (z-major order)
    let mut data = Vec::with_capacity(size_z * size_xy * size_xy);
    for plane in &planes {
        data.extend_from_slice(plane);
    }

    // Normalise so the total sum == 1
    let total: f64 = data.iter().map(|&v| v as f64).sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        for v in &mut data {
            *v = (*v as f64 * inv) as f32;
        }
    }

    FimImage::from_vec(size_xy, size_xy, size_z, data)
}

// ---------------------------------------------------------------------------
// Confocal PSF generation
// ---------------------------------------------------------------------------

/// Generate a confocal PSF.
///
/// The confocal PSF is the product of an excitation PSF (at `lambda_ex`) and a
/// detection PSF (at `lambda_em`) that has been convolved with the pinhole
/// function.
///
/// # Arguments
/// * `na`         - Numerical aperture
/// * `ni`         - Refractive index
/// * `lambda_em`  - Emission wavelength (nm)
/// * `lambda_ex`  - Excitation wavelength (nm)
/// * `dx`         - Lateral pixel size (nm)
/// * `dz`         - Axial step size (nm)
/// * `size_xy`    - Lateral extent (odd)
/// * `size_z`     - Number of z-planes (odd)
/// * `pinhole_au` - Pinhole diameter in Airy Units (1.0 is one Airy disk)
pub fn generate_confocal_psf(
    na: f64,
    ni: f64,
    lambda_em: f64,
    lambda_ex: f64,
    dx: f64,
    dz: f64,
    size_xy: usize,
    size_z: usize,
    pinhole_au: f64,
) -> Result<FimImage> {
    // Generate excitation PSF
    let excitation = generate_widefield_psf(na, ni, lambda_ex, dx, dz, size_xy, size_z)?;

    // Generate emission (detection) PSF
    let mut detection = generate_widefield_psf(na, ni, lambda_em, dx, dz, size_xy, size_z)?;

    // Convolve the detection PSF with the pinhole disk (2-D, per z-plane).
    // Pinhole radius in pixels: pinhole_au * 0.61 * lambda_em / NA  (in nm),
    // then convert to pixels.
    let pinhole_radius_nm = pinhole_au * 0.61 * lambda_em / na;
    let pinhole_radius_px = pinhole_radius_nm / dx;

    // Build a 2-D normalised disk kernel
    let kr = pinhole_radius_px.ceil() as usize;
    let ksize = 2 * kr + 1;
    let mut kernel = vec![0.0f64; ksize * ksize];
    let mut ksum = 0.0f64;
    for ky in 0..ksize {
        for kx in 0..ksize {
            let rx = kx as f64 - kr as f64;
            let ry = ky as f64 - kr as f64;
            let r = (rx * rx + ry * ry).sqrt();
            if r <= pinhole_radius_px {
                kernel[ky * ksize + kx] = 1.0;
                ksum += 1.0;
            }
        }
    }
    if ksum > 0.0 {
        for v in &mut kernel {
            *v /= ksum;
        }
    }

    // Convolve each z-plane of detection with the disk kernel (direct spatial
    // convolution -- acceptable because kernel is small).
    for pz in 0..size_z {
        let mut conv = vec![0.0f64; size_xy * size_xy];
        for iy in 0..size_xy {
            for ix in 0..size_xy {
                let mut acc = 0.0f64;
                for ky in 0..ksize {
                    for kx in 0..ksize {
                        let sx = ix as isize + kx as isize - kr as isize;
                        let sy = iy as isize + ky as isize - kr as isize;
                        if sx >= 0
                            && sx < size_xy as isize
                            && sy >= 0
                            && sy < size_xy as isize
                        {
                            acc += detection.get(sx as usize, sy as usize, pz) as f64
                                * kernel[ky * ksize + kx];
                        }
                    }
                }
                conv[iy * size_xy + ix] = acc;
            }
        }
        for iy in 0..size_xy {
            for ix in 0..size_xy {
                detection.set(ix, iy, pz, conv[iy * size_xy + ix] as f32);
            }
        }
    }

    // Multiply excitation x detection
    let mut data = vec![0.0f32; size_z * size_xy * size_xy];
    for pz in 0..size_z {
        for iy in 0..size_xy {
            for ix in 0..size_xy {
                data[pz * size_xy * size_xy + iy * size_xy + ix] =
                    excitation.get(ix, iy, pz) * detection.get(ix, iy, pz);
            }
        }
    }

    // Normalise
    let total: f64 = data.iter().map(|&v| v as f64).sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        for v in &mut data {
            *v = (*v as f64 * inv) as f32;
        }
    }

    FimImage::from_vec(size_xy, size_xy, size_z, data)
}

// ---------------------------------------------------------------------------
// STED PSF generation
// ---------------------------------------------------------------------------

/// Generate a STED PSF using a Lorentzian lateral profile and Gaussian axial
/// profile.
///
/// # Arguments
/// * `lateral_fwhm` - Lateral FWHM in pixels
/// * `axial_fwhm`   - Axial FWHM in pixels
/// * `size_xy`      - Lateral extent in pixels (must be odd)
/// * `size_z`       - Number of z-planes (must be odd)
pub fn generate_sted_psf(
    lateral_fwhm: f64,
    axial_fwhm: f64,
    size_xy: usize,
    size_z: usize,
) -> Result<FimImage> {
    if size_xy % 2 == 0 || size_z % 2 == 0 {
        return Err(DwError::InvalidDimensions(
            "PSF dimensions must be odd".into(),
        ));
    }
    if lateral_fwhm <= 0.0 || axial_fwhm <= 0.0 {
        return Err(DwError::Config("FWHM values must be positive".into()));
    }

    let center_xy = (size_xy / 2) as f64;
    let center_z = (size_z / 2) as f64;

    // Lorentzian: FWHM = 2*gamma  =>  gamma = FWHM/2
    let gamma = lateral_fwhm / 2.0;
    let gamma2 = gamma * gamma;

    // Gaussian: FWHM = 2*sqrt(2*ln2)*sigma  =>  sigma = FWHM / (2*sqrt(2*ln2))
    let sigma = axial_fwhm / (2.0 * (2.0_f64 * 2.0_f64.ln()).sqrt());
    let two_sigma2 = 2.0 * sigma * sigma;

    let mut data = vec![0.0f32; size_z * size_xy * size_xy];

    for pz in 0..size_z {
        let dz = pz as f64 - center_z;
        let gauss = (-dz * dz / two_sigma2).exp();
        for iy in 0..size_xy {
            let dy = iy as f64 - center_xy;
            for ix in 0..size_xy {
                let dxx = ix as f64 - center_xy;
                let r2 = dxx * dxx + dy * dy;
                let lorentz = 1.0 / (1.0 + r2 / gamma2);
                data[pz * size_xy * size_xy + iy * size_xy + ix] =
                    (lorentz * gauss) as f32;
            }
        }
    }

    // Normalise
    let total: f64 = data.iter().map(|&v| v as f64).sum();
    if total > 0.0 {
        let inv = 1.0 / total;
        for v in &mut data {
            *v = (*v as f64 * inv) as f32;
        }
    }

    FimImage::from_vec(size_xy, size_xy, size_z, data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel_j0_at_zero() {
        let val = bessel_j0(0.0);
        assert!(
            (val - 1.0).abs() < 1e-6,
            "J0(0) should be 1, got {}",
            val
        );
    }

    #[test]
    fn test_bessel_j0_first_root() {
        // First zero of J0 is at approximately 2.4048255577
        let val = bessel_j0(2.4048255577);
        assert!(
            val.abs() < 1e-4,
            "J0(2.4048) should be ~0, got {}",
            val
        );
    }

    #[test]
    fn test_bessel_j0_known_values() {
        // J0(1) ~ 0.7651976866
        assert!((bessel_j0(1.0) - 0.7651976866).abs() < 1e-5);
        // J0(5) ~ -0.1775967713
        assert!((bessel_j0(5.0) - (-0.1775967713)).abs() < 1e-5);
        // J0(-3) = J0(3) ~ -0.2600519549
        assert!((bessel_j0(-3.0) - bessel_j0(3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sted_psf_symmetry() {
        let psf = generate_sted_psf(3.0, 5.0, 11, 7).unwrap();
        let (m, n, p) = psf.dims();
        assert_eq!(m, 11);
        assert_eq!(n, 11);
        assert_eq!(p, 7);

        // Symmetry: PSF(cx+d, cy, cz) == PSF(cx-d, cy, cz)
        let cx = m / 2;
        let cy = n / 2;
        let cz = p / 2;
        for d in 1..=cx {
            let v1 = psf.get(cx + d, cy, cz);
            let v2 = psf.get(cx - d, cy, cz);
            assert!(
                (v1 - v2).abs() < 1e-7,
                "Lateral symmetry broken at d={}: {} vs {}",
                d,
                v1,
                v2
            );
        }

        // Axial symmetry
        for d in 1..=cz {
            let v1 = psf.get(cx, cy, cz + d);
            let v2 = psf.get(cx, cy, cz - d);
            assert!(
                (v1 - v2).abs() < 1e-7,
                "Axial symmetry broken at d={}: {} vs {}",
                d,
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_sted_psf_normalization() {
        let psf = generate_sted_psf(3.0, 5.0, 21, 11).unwrap();
        let total: f64 = psf.as_slice().iter().map(|&v| v as f64).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "STED PSF should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_sted_psf_fwhm_lateral() {
        // Check that the lateral FWHM roughly matches the input
        let fwhm_in = 5.0;
        let psf = generate_sted_psf(fwhm_in, 10.0, 51, 11).unwrap();
        let cx = 25;
        let cy = 25;
        let cz = 5;
        let peak = psf.get(cx, cy, cz);
        let half_max = peak / 2.0;

        // Walk along x to find the half-maximum point
        let mut fwhm_measured = 0.0;
        for d in 1..cx {
            let v = psf.get(cx + d, cy, cz);
            if v <= half_max {
                // Linear interpolation
                let v_prev = psf.get(cx + d - 1, cy, cz);
                let frac = (v_prev - half_max) / (v_prev - v);
                fwhm_measured = 2.0 * ((d - 1) as f64 + frac as f64);
                break;
            }
        }

        assert!(
            (fwhm_measured - fwhm_in).abs() < 0.5,
            "Lateral FWHM should be ~{}, measured {}",
            fwhm_in,
            fwhm_measured
        );
    }

    #[test]
    fn test_widefield_psf_peak_at_center() {
        let psf = generate_widefield_psf(1.4, 1.515, 520.0, 65.0, 200.0, 11, 5).unwrap();
        let cx = 5;
        let cy = 5;
        let cz = 2;
        let peak = psf.get(cx, cy, cz);
        // Peak should be the maximum value
        for pz in 0..5 {
            for iy in 0..11 {
                for ix in 0..11 {
                    assert!(
                        psf.get(ix, iy, pz) <= peak + 1e-10,
                        "Value at ({},{},{})={} exceeds peak {}",
                        ix,
                        iy,
                        pz,
                        psf.get(ix, iy, pz),
                        peak
                    );
                }
            }
        }
    }

    #[test]
    fn test_widefield_psf_normalization() {
        let psf = generate_widefield_psf(1.4, 1.515, 520.0, 65.0, 200.0, 11, 5).unwrap();
        let total: f64 = psf.as_slice().iter().map(|&v| v as f64).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "Widefield PSF should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_widefield_psf_even_size_rejected() {
        let result = generate_widefield_psf(1.4, 1.515, 520.0, 65.0, 200.0, 10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sted_psf_even_size_rejected() {
        let result = generate_sted_psf(3.0, 5.0, 10, 5);
        assert!(result.is_err());
    }
}
