use rayon::prelude::*;

use super::image::FimImage;

/// Generate a normalized 1D Gaussian kernel with the given sigma.
///
/// Kernel size is `2 * ceil(3 * sigma) + 1`. The kernel is normalized to sum to 1.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let val = (-x * x / two_sigma_sq).exp();
        kernel.push(val);
        sum += val as f64;
    }
    let inv_sum = 1.0 / sum as f32;
    for v in &mut kernel {
        *v *= inv_sum;
    }
    kernel
}

/// Generate a 1D second-derivative-of-Gaussian kernel (d²G/dx²).
///
/// d²G/dx² = (x² / sigma⁴ - 1/sigma²) * G(x)
fn gaussian_d2_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![0.0];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;
    let two_sigma_sq = 2.0 * sigma2;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let g = (-x * x / two_sigma_sq).exp();
        let val = (x * x / sigma4 - 1.0 / sigma2) * g;
        kernel.push(val);
    }
    kernel
}

impl FimImage {
    /// Minimum pixel value.
    pub fn min(&self) -> f32 {
        self.as_slice()
            .par_iter()
            .copied()
            .reduce(|| f32::INFINITY, f32::min)
    }

    /// Maximum pixel value.
    pub fn max(&self) -> f32 {
        self.as_slice()
            .par_iter()
            .copied()
            .reduce(|| f32::NEG_INFINITY, f32::max)
    }

    /// Sum of all pixel values.
    pub fn sum(&self) -> f64 {
        self.as_slice()
            .par_iter()
            .map(|&v| v as f64)
            .sum()
    }

    /// Mean pixel value.
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.sum() / self.len() as f64
    }

    /// Standard deviation (normalizing by N-1).
    pub fn std(&self) -> f64 {
        let n = self.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let var: f64 = self
            .as_slice()
            .par_iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum();
        (var / (n - 1) as f64).sqrt()
    }

    /// Percentile value (0..100) using quickselect.
    pub fn percentile(&self, prct: f32) -> f32 {
        let mut data = self.as_slice().to_vec();
        let n = data.len();
        if n == 0 {
            return 0.0;
        }
        let k = ((prct / 100.0) * (n - 1) as f32).round() as usize;
        let k = k.min(n - 1);
        quickselect(&mut data, k)
    }

    /// Find coordinates (m, n, p) and value of the maximum element.
    pub fn argmax(&self) -> (usize, usize, usize, f32) {
        let slice = self.as_slice();
        let (idx, &max_val) = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let m_dim = self.m();
        let n_dim = self.n();
        let mm = idx % m_dim;
        let nn = (idx / m_dim) % n_dim;
        let pp = idx / (m_dim * n_dim);
        (mm, nn, pp, max_val)
    }

    /// Check if the maximum is at the image center (origin for odd dims).
    pub fn max_at_origin(&self) -> bool {
        let (am, an, ap, _) = self.argmax();
        am == self.m() / 2 && an == self.n() / 2 && ap == self.p() / 2
    }

    /// Normalize so that all elements sum to 1.0.
    pub fn normalize_sum1(&mut self) {
        let s = self.sum();
        if s == 0.0 {
            return;
        }
        let inv_s = 1.0 / s as f32;
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| *v *= inv_s);
    }

    /// Multiply all pixels by a scalar.
    pub fn mult_scalar(&mut self, x: f32) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| *v *= x);
    }

    /// Add a scalar to all pixels.
    pub fn add_scalar(&mut self, x: f32) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| *v += x);
    }

    /// Clamp negative values to 0.
    pub fn project_positive(&mut self) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| {
                if *v < 0.0 {
                    *v = 0.0;
                }
            });
    }

    /// Shift so the minimum value becomes 0.
    pub fn set_min_to_zero(&mut self) {
        let min_val = self.min();
        if min_val != 0.0 {
            self.add_scalar(-min_val);
        }
    }

    /// Element-wise addition: self += other.
    pub fn add_image(&mut self, other: &FimImage) {
        let a = self.as_slice_mut();
        let b = other.as_slice();
        assert_eq!(a.len(), b.len());
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(a, b)| *a += b);
    }

    /// Element-wise subtraction: result = a - b.
    pub fn subtract(a: &FimImage, b: &FimImage) -> FimImage {
        assert_eq!(a.len(), b.len());
        let mut out = a.clone();
        out.as_slice_mut()
            .par_iter_mut()
            .zip(b.as_slice().par_iter())
            .for_each(|(o, &bv)| *o -= bv);
        out
    }

    /// Element-wise division: result[i] = a[i] / b[i].
    pub fn divide(a: &FimImage, b: &FimImage) -> FimImage {
        assert_eq!(a.len(), b.len());
        let mut out = a.clone();
        out.as_slice_mut()
            .par_iter_mut()
            .zip(b.as_slice().par_iter())
            .for_each(|(o, &bv)| {
                *o = if bv != 0.0 { *o / bv } else { 0.0 };
            });
        out
    }

    /// Element-wise multiplication: self *= other.
    pub fn mult_image(&mut self, other: &FimImage) {
        let a = self.as_slice_mut();
        let b = other.as_slice();
        assert_eq!(a.len(), b.len());
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(a, b)| *a *= b);
    }

    /// Mean squared error between two images.
    pub fn mse(a: &FimImage, b: &FimImage) -> f64 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = a
            .as_slice()
            .par_iter()
            .zip(b.as_slice().par_iter())
            .map(|(&av, &bv)| {
                let d = av as f64 - bv as f64;
                d * d
            })
            .sum();
        sum_sq / n as f64
    }

    /// Maximum Z-projection: returns a 2D image (P=1).
    pub fn max_projection(&self) -> FimImage {
        let (m_dim, n_dim, p_dim) = self.dims();
        let mut out = FimImage::constant(m_dim, n_dim, 1, f32::NEG_INFINITY);
        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    let val = self.data[[pp, nn, mm]];
                    if val > out.data[[0, nn, mm]] {
                        out.data[[0, nn, mm]] = val;
                    }
                }
            }
        }
        out
    }

    /// Sum Z-projection: returns a 2D image (P=1).
    pub fn sum_projection(&self) -> FimImage {
        let (m_dim, n_dim, p_dim) = self.dims();
        let mut out = FimImage::zeros(m_dim, n_dim, 1);
        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    out.data[[0, nn, mm]] += self.data[[pp, nn, mm]];
                }
            }
        }
        out
    }

    /// Anscombe transform: x[i] = 2 * sqrt(x[i] + 3/8).
    pub fn anscombe(&mut self) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| {
                *v = 2.0 * (*v + 0.375).max(0.0).sqrt();
            });
    }

    /// Inverse Anscombe transform.
    pub fn ianscombe(&mut self) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| {
                let x = *v / 2.0;
                *v = (x * x - 0.375).max(0.0);
            });
    }

    /// Z-crop: remove `zcrop` slices from both ends.
    pub fn zcrop(&self, zcrop: usize) -> super::error::Result<FimImage> {
        let p_dim = self.p();
        if 2 * zcrop >= p_dim {
            return Err(super::error::DwError::InvalidDimensions(format!(
                "Cannot remove {} slices from each end of {}-slice image",
                zcrop, p_dim
            )));
        }
        self.get_cuboid(0, self.m(), 0, self.n(), zcrop, p_dim - zcrop)
    }

    /// Auto z-crop to `new_p` slices centered on the brightest plane.
    pub fn auto_zcrop(&self, new_p: usize) -> super::error::Result<FimImage> {
        let p_dim = self.p();
        if new_p >= p_dim {
            return Ok(self.clone());
        }
        // Find brightest plane
        let mut best_plane = 0;
        let mut best_sum = f64::NEG_INFINITY;
        for pp in 0..p_dim {
            let mut s = 0.0f64;
            for nn in 0..self.n() {
                for mm in 0..self.m() {
                    s += self.data[[pp, nn, mm]] as f64;
                }
            }
            if s > best_sum {
                best_sum = s;
                best_plane = pp;
            }
        }
        let half = new_p / 2;
        let p0 = if best_plane >= half {
            best_plane - half
        } else {
            0
        };
        let p0 = p0.min(p_dim - new_p);
        self.get_cuboid(0, self.m(), 0, self.n(), p0, p0 + new_p)
    }

    /// Invert all elements: x[i] = 1/x[i].
    pub fn invert(&mut self) {
        self.as_slice_mut()
            .par_iter_mut()
            .for_each(|v| {
                if *v != 0.0 {
                    *v = 1.0 / *v;
                }
            });
    }

    // ---------------------------------------------------------------
    // Convolution and smoothing
    // ---------------------------------------------------------------

    /// 1D convolution along a specified dimension (in-place).
    ///
    /// - `dim` 0 = M (x, stride-1), 1 = N (y), 2 = P (z)
    /// - If `normalized` is true, kernel weights are renormalized at edges
    ///   so that only the portion of the kernel that overlaps the image is used.
    pub fn convolve_1d(&mut self, kernel: &[f32], dim: usize, normalized: bool) {
        assert!(dim < 3, "dim must be 0, 1, or 2");
        let klen = kernel.len();
        if klen == 0 {
            return;
        }
        let radius = klen / 2;
        let (m_dim, n_dim, p_dim) = self.dims();

        match dim {
            0 => {
                // Convolve along M (x). Outer loop over planes and rows.
                let slice = self.as_slice_mut();
                let mn = m_dim * n_dim;
                // Process each plane*row in parallel
                (0..p_dim * n_dim).into_par_iter().for_each(|idx| {
                    let pp = idx / n_dim;
                    let nn = idx % n_dim;
                    let row_start = pp * mn + nn * m_dim;
                    // SAFETY: each parallel iteration writes to a non-overlapping row
                    let ptr = slice.as_ptr();
                    let mut buf = vec![0.0f32; m_dim];
                    for mm in 0..m_dim {
                        let mut acc = 0.0f32;
                        let mut wsum = 0.0f32;
                        for ki in 0..klen {
                            let src = mm as isize + ki as isize - radius as isize;
                            if src >= 0 && (src as usize) < m_dim {
                                let val = unsafe { *ptr.add(row_start + src as usize) };
                                acc += val * kernel[ki];
                                wsum += kernel[ki];
                            }
                        }
                        buf[mm] = if normalized && wsum != 0.0 {
                            acc / wsum
                        } else {
                            acc
                        };
                    }
                    // Write back
                    let out_ptr = slice.as_ptr() as *mut f32;
                    for mm in 0..m_dim {
                        unsafe {
                            *out_ptr.add(row_start + mm) = buf[mm];
                        }
                    }
                });
            }
            1 => {
                // Convolve along N (y). Outer loop over planes and columns.
                let slice = self.as_slice_mut();
                let mn = m_dim * n_dim;
                (0..p_dim * m_dim).into_par_iter().for_each(|idx| {
                    let pp = idx / m_dim;
                    let mm = idx % m_dim;
                    let ptr = slice.as_ptr();
                    let mut buf = vec![0.0f32; n_dim];
                    for nn in 0..n_dim {
                        let mut acc = 0.0f32;
                        let mut wsum = 0.0f32;
                        for ki in 0..klen {
                            let src = nn as isize + ki as isize - radius as isize;
                            if src >= 0 && (src as usize) < n_dim {
                                let offset = pp * mn + src as usize * m_dim + mm;
                                let val = unsafe { *ptr.add(offset) };
                                acc += val * kernel[ki];
                                wsum += kernel[ki];
                            }
                        }
                        buf[nn] = if normalized && wsum != 0.0 {
                            acc / wsum
                        } else {
                            acc
                        };
                    }
                    let out_ptr = slice.as_ptr() as *mut f32;
                    for nn in 0..n_dim {
                        let offset = pp * mn + nn * m_dim + mm;
                        unsafe {
                            *out_ptr.add(offset) = buf[nn];
                        }
                    }
                });
            }
            2 => {
                // Convolve along P (z). Outer loop over rows and columns.
                let slice = self.as_slice_mut();
                let mn = m_dim * n_dim;
                (0..n_dim * m_dim).into_par_iter().for_each(|idx| {
                    let nn = idx / m_dim;
                    let mm = idx % m_dim;
                    let ptr = slice.as_ptr();
                    let mut buf = vec![0.0f32; p_dim];
                    for pp in 0..p_dim {
                        let mut acc = 0.0f32;
                        let mut wsum = 0.0f32;
                        for ki in 0..klen {
                            let src = pp as isize + ki as isize - radius as isize;
                            if src >= 0 && (src as usize) < p_dim {
                                let offset = src as usize * mn + nn * m_dim + mm;
                                let val = unsafe { *ptr.add(offset) };
                                acc += val * kernel[ki];
                                wsum += kernel[ki];
                            }
                        }
                        buf[pp] = if normalized && wsum != 0.0 {
                            acc / wsum
                        } else {
                            acc
                        };
                    }
                    let out_ptr = slice.as_ptr() as *mut f32;
                    for pp in 0..p_dim {
                        let offset = pp * mn + nn * m_dim + mm;
                        unsafe {
                            *out_ptr.add(offset) = buf[pp];
                        }
                    }
                });
            }
            _ => unreachable!(),
        }
    }

    /// Isotropic Gaussian smoothing (separable, in-place).
    ///
    /// Uses edge-normalized weighting so border pixels are not darkened.
    pub fn gsmooth(&mut self, sigma: f32) {
        if sigma <= 0.0 {
            return;
        }
        let kernel = gaussian_kernel(sigma);
        self.convolve_1d(&kernel, 0, true);
        self.convolve_1d(&kernel, 1, true);
        if self.p() > 1 {
            self.convolve_1d(&kernel, 2, true);
        }
    }

    /// Anisotropic Gaussian smoothing (separable, in-place).
    ///
    /// `sigma_lateral` is applied to M and N; `sigma_axial` to P.
    pub fn gsmooth_aniso(&mut self, sigma_lateral: f32, sigma_axial: f32) {
        if sigma_lateral > 0.0 {
            let k_lat = gaussian_kernel(sigma_lateral);
            self.convolve_1d(&k_lat, 0, true);
            self.convolve_1d(&k_lat, 1, true);
        }
        if sigma_axial > 0.0 && self.p() > 1 {
            let k_ax = gaussian_kernel(sigma_axial);
            self.convolve_1d(&k_ax, 2, true);
        }
    }

    // ---------------------------------------------------------------
    // Derivatives and differential filters
    // ---------------------------------------------------------------

    /// Partial derivative along dimension `dim` with Gaussian pre-smoothing.
    ///
    /// Computes the central-difference derivative: `(f[i+1] - f[i-1]) / 2`.
    /// Boundary pixels use forward/backward differences.
    pub fn partial_derivative(&self, dim: usize, sigma: f32) -> FimImage {
        assert!(dim < 3, "dim must be 0, 1, or 2");
        let mut smoothed = self.clone();
        if sigma > 0.0 {
            smoothed.gsmooth(sigma);
        }
        let (m_dim, n_dim, p_dim) = smoothed.dims();
        let mut out = FimImage::zeros(m_dim, n_dim, p_dim);

        match dim {
            0 => {
                for pp in 0..p_dim {
                    for nn in 0..n_dim {
                        for mm in 0..m_dim {
                            let val = if mm == 0 {
                                smoothed.get(1.min(m_dim - 1), nn, pp) - smoothed.get(0, nn, pp)
                            } else if mm == m_dim - 1 {
                                smoothed.get(m_dim - 1, nn, pp)
                                    - smoothed.get((m_dim - 2).max(0), nn, pp)
                            } else {
                                (smoothed.get(mm + 1, nn, pp) - smoothed.get(mm - 1, nn, pp))
                                    * 0.5
                            };
                            out.set(mm, nn, pp, val);
                        }
                    }
                }
            }
            1 => {
                for pp in 0..p_dim {
                    for nn in 0..n_dim {
                        for mm in 0..m_dim {
                            let val = if nn == 0 {
                                smoothed.get(mm, 1.min(n_dim - 1), pp) - smoothed.get(mm, 0, pp)
                            } else if nn == n_dim - 1 {
                                smoothed.get(mm, n_dim - 1, pp)
                                    - smoothed.get(mm, (n_dim - 2).max(0), pp)
                            } else {
                                (smoothed.get(mm, nn + 1, pp) - smoothed.get(mm, nn - 1, pp))
                                    * 0.5
                            };
                            out.set(mm, nn, pp, val);
                        }
                    }
                }
            }
            2 => {
                for pp in 0..p_dim {
                    for nn in 0..n_dim {
                        for mm in 0..m_dim {
                            let val = if pp == 0 {
                                smoothed.get(mm, nn, 1.min(p_dim - 1)) - smoothed.get(mm, nn, 0)
                            } else if pp == p_dim - 1 {
                                smoothed.get(mm, nn, p_dim - 1)
                                    - smoothed.get(mm, nn, (p_dim - 2).max(0))
                            } else {
                                (smoothed.get(mm, nn, pp + 1) - smoothed.get(mm, nn, pp - 1))
                                    * 0.5
                            };
                            out.set(mm, nn, pp, val);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
        out
    }

    /// Gradient magnitude: `sqrt(dx² + dy² + dz²)`.
    ///
    /// Each partial derivative is computed with Gaussian pre-smoothing of the
    /// given `sigma`.
    pub fn gradient_magnitude(&self, sigma: f32) -> FimImage {
        let dx = self.partial_derivative(0, sigma);
        let dy = self.partial_derivative(1, sigma);
        let dz = if self.p() > 1 {
            self.partial_derivative(2, sigma)
        } else {
            FimImage::zeros(self.m(), self.n(), self.p())
        };

        let (m_dim, n_dim, p_dim) = self.dims();
        let mut out = FimImage::zeros(m_dim, n_dim, p_dim);
        let out_s = out.as_slice_mut();
        let dx_s = dx.as_slice();
        let dy_s = dy.as_slice();
        let dz_s = dz.as_slice();
        out_s
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| {
                *o = (dx_s[i] * dx_s[i] + dy_s[i] * dy_s[i] + dz_s[i] * dz_s[i]).sqrt();
            });
        out
    }

    /// Laplacian of Gaussian filter.
    ///
    /// LoG = d²G/dx² + d²G/dy² + d²G/dz²
    ///
    /// Implemented using separable convolution with Gaussian and second-derivative
    /// Gaussian kernels.
    pub fn log_filter(&self, sigma_xy: f32, sigma_z: f32) -> FimImage {
        let gk_xy = gaussian_kernel(sigma_xy);
        let d2k_xy = gaussian_d2_kernel(sigma_xy);
        let gk_z = gaussian_kernel(sigma_z);
        let d2k_z = gaussian_d2_kernel(sigma_z);
        let is_3d = self.p() > 1;

        // d²G/dx²: convolve with d²G in x, G in y, G in z
        let mut term_x = self.clone();
        term_x.convolve_1d(&d2k_xy, 0, false);
        term_x.convolve_1d(&gk_xy, 1, false);
        if is_3d {
            term_x.convolve_1d(&gk_z, 2, false);
        }

        // d²G/dy²: convolve with G in x, d²G in y, G in z
        let mut term_y = self.clone();
        term_y.convolve_1d(&gk_xy, 0, false);
        term_y.convolve_1d(&d2k_xy, 1, false);
        if is_3d {
            term_y.convolve_1d(&gk_z, 2, false);
        }

        // Sum
        term_x.add_image(&term_y);

        if is_3d {
            // d²G/dz²: convolve with G in x, G in y, d²G in z
            let mut term_z = self.clone();
            term_z.convolve_1d(&gk_xy, 0, false);
            term_z.convolve_1d(&gk_xy, 1, false);
            term_z.convolve_1d(&d2k_z, 2, false);
            term_x.add_image(&term_z);
        }

        term_x
    }

    // ---------------------------------------------------------------
    // Interpolation and shifting
    // ---------------------------------------------------------------

    /// Trilinear interpolation at a continuous coordinate `(x, y, z)`.
    ///
    /// Coordinates are in (M, N, P) order. Returns 0 if out of bounds.
    pub fn interp3_linear(&self, x: f64, y: f64, z: f64) -> f32 {
        let (m_dim, n_dim, p_dim) = self.dims();
        if x < 0.0
            || y < 0.0
            || z < 0.0
            || x >= (m_dim - 1) as f64
            || y >= (n_dim - 1) as f64
            || z >= (p_dim - 1) as f64
        {
            // Allow exact integer at upper bound
            if x < 0.0 || y < 0.0 || z < 0.0 {
                return 0.0;
            }
            if x > (m_dim - 1) as f64 || y > (n_dim - 1) as f64 || z > (p_dim - 1) as f64 {
                return 0.0;
            }
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(m_dim - 1);
        let y1 = (y0 + 1).min(n_dim - 1);
        let z1 = (z0 + 1).min(p_dim - 1);

        let xd = (x - x0 as f64) as f32;
        let yd = (y - y0 as f64) as f32;
        let zd = (z - z0 as f64) as f32;

        let c000 = self.get(x0, y0, z0);
        let c100 = self.get(x1, y0, z0);
        let c010 = self.get(x0, y1, z0);
        let c110 = self.get(x1, y1, z0);
        let c001 = self.get(x0, y0, z1);
        let c101 = self.get(x1, y0, z1);
        let c011 = self.get(x0, y1, z1);
        let c111 = self.get(x1, y1, z1);

        let c00 = c000 * (1.0 - xd) + c100 * xd;
        let c01 = c001 * (1.0 - xd) + c101 * xd;
        let c10 = c010 * (1.0 - xd) + c110 * xd;
        let c11 = c011 * (1.0 - xd) + c111 * xd;

        let c0 = c00 * (1.0 - yd) + c10 * yd;
        let c1 = c01 * (1.0 - yd) + c11 * yd;

        c0 * (1.0 - zd) + c1 * zd
    }

    /// Shift the image by fractional amounts using trilinear interpolation.
    ///
    /// `dm`, `dn`, `dp` are shifts in M, N, P dimensions respectively.
    /// Out-of-bounds pixels are set to 0.
    pub fn shift_linear(&self, dm: f32, dn: f32, dp: f32) -> FimImage {
        let (m_dim, n_dim, p_dim) = self.dims();
        let mut out = FimImage::zeros(m_dim, n_dim, p_dim);
        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    let sx = mm as f64 - dm as f64;
                    let sy = nn as f64 - dn as f64;
                    let sz = pp as f64 - dp as f64;
                    out.set(mm, nn, pp, self.interp3_linear(sx, sy, sz));
                }
            }
        }
        out
    }

    // ---------------------------------------------------------------
    // Connected components
    // ---------------------------------------------------------------

    /// 2D connected-component labeling with 6-connectivity (4-connectivity per plane).
    ///
    /// Returns a flat label map with the same dimensions as the image.
    /// Pixels with value > 0 are foreground; 0 is background (label 0).
    /// Labels are 1-based. Each z-plane is labeled independently.
    pub fn conncomp6_2d(&self) -> Vec<i32> {
        let (m_dim, n_dim, p_dim) = self.dims();
        let plane_size = m_dim * n_dim;
        let mut labels = vec![0i32; m_dim * n_dim * p_dim];
        let mut next_label = 1i32;

        for pp in 0..p_dim {
            // Union-Find for this plane
            let mut parent: Vec<i32> = Vec::new();

            fn find(parent: &mut Vec<i32>, mut x: i32) -> i32 {
                while parent[x as usize] != x {
                    parent[x as usize] = parent[parent[x as usize] as usize];
                    x = parent[x as usize];
                }
                x
            }

            fn union(parent: &mut Vec<i32>, a: i32, b: i32) {
                let ra = find(parent, a);
                let rb = find(parent, b);
                if ra != rb {
                    parent[ra as usize] = rb;
                }
            }

            let plane_offset = pp * plane_size;
            // First pass
            let local_start = next_label;
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    let idx = plane_offset + nn * m_dim + mm;
                    if self.as_slice()[idx] <= 0.0 {
                        continue;
                    }
                    let mut neighbors = Vec::new();
                    // Left neighbor
                    if mm > 0 {
                        let left = labels[idx - 1];
                        if left > 0 {
                            neighbors.push(left);
                        }
                    }
                    // Top neighbor
                    if nn > 0 {
                        let top = labels[idx - m_dim];
                        if top > 0 {
                            neighbors.push(top);
                        }
                    }

                    if neighbors.is_empty() {
                        // Ensure parent vec is large enough
                        while parent.len() <= (next_label - local_start) as usize {
                            let l = parent.len() as i32;
                            parent.push(l);
                        }
                        labels[idx] = next_label;
                        next_label += 1;
                    } else {
                        let min_label = *neighbors.iter().min().unwrap();
                        labels[idx] = min_label;
                        // Union all neighbors
                        for &nl in &neighbors {
                            union(&mut parent, (nl - local_start) as i32, (min_label - local_start) as i32);
                        }
                    }
                }
            }

            // Second pass: resolve labels
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    let idx = plane_offset + nn * m_dim + mm;
                    if labels[idx] > 0 {
                        let local = (labels[idx] - local_start) as i32;
                        let root = find(&mut parent, local);
                        labels[idx] = root + local_start;
                    }
                }
            }
        }

        labels
    }

    // ---------------------------------------------------------------
    // Thresholding
    // ---------------------------------------------------------------

    /// Otsu's threshold: finds the threshold that minimizes intra-class variance.
    ///
    /// Uses a 256-bin histogram between min and max values.
    pub fn otsu_threshold(&self) -> f32 {
        let min_val = self.min();
        let max_val = self.max();
        if (max_val - min_val).abs() < f32::EPSILON {
            return min_val;
        }

        let nbins = 256usize;
        let range = max_val - min_val;
        let bin_width = range / nbins as f32;
        let mut hist = vec![0u64; nbins];

        for &v in self.as_slice() {
            let bin = ((v - min_val) / bin_width) as usize;
            let bin = bin.min(nbins - 1);
            hist[bin] += 1;
        }

        let total = self.len() as f64;
        let mut sum_total = 0.0f64;
        for (i, &h) in hist.iter().enumerate() {
            sum_total += i as f64 * h as f64;
        }

        let mut sum_bg = 0.0f64;
        let mut w_bg = 0.0f64;
        let mut best_var = -1.0f64;
        let mut best_t = 0usize;

        for t in 0..nbins {
            w_bg += hist[t] as f64;
            if w_bg == 0.0 {
                continue;
            }
            let w_fg = total - w_bg;
            if w_fg == 0.0 {
                break;
            }
            sum_bg += t as f64 * hist[t] as f64;
            let mean_bg = sum_bg / w_bg;
            let mean_fg = (sum_total - sum_bg) / w_fg;
            let diff = mean_bg - mean_fg;
            let var_between = w_bg * w_fg * diff * diff;
            if var_between > best_var {
                best_var = var_between;
                best_t = t;
            }
        }

        min_val + (best_t as f32 + 0.5) * bin_width
    }

    // ---------------------------------------------------------------
    // Focus and quality measures
    // ---------------------------------------------------------------

    /// Focus gradient magnitude: returns one value per z-plane.
    ///
    /// Each value is the mean gradient magnitude of that z-slice,
    /// computed with the given Gaussian sigma.
    pub fn focus_gm(&self, sigma: f32) -> Vec<f32> {
        let (m_dim, n_dim, p_dim) = self.dims();
        let gm = self.gradient_magnitude(sigma);
        let plane_size = (m_dim * n_dim) as f64;
        let mut result = Vec::with_capacity(p_dim);
        for pp in 0..p_dim {
            let mut sum = 0.0f64;
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    sum += gm.get(mm, nn, pp) as f64;
                }
            }
            result.push((sum / plane_size) as f32);
        }
        result
    }

    // ---------------------------------------------------------------
    // Cumulative operations
    // ---------------------------------------------------------------

    /// Cumulative sum along a dimension (in-place).
    ///
    /// - `dim` 0 = M (x), 1 = N (y), 2 = P (z)
    pub fn cumsum(&mut self, dim: usize) {
        assert!(dim < 3, "dim must be 0, 1, or 2");
        let (m_dim, n_dim, p_dim) = self.dims();

        match dim {
            0 => {
                for pp in 0..p_dim {
                    for nn in 0..n_dim {
                        for mm in 1..m_dim {
                            let prev = self.get(mm - 1, nn, pp);
                            let cur = self.get(mm, nn, pp);
                            self.set(mm, nn, pp, cur + prev);
                        }
                    }
                }
            }
            1 => {
                for pp in 0..p_dim {
                    for nn in 1..n_dim {
                        for mm in 0..m_dim {
                            let prev = self.get(mm, nn - 1, pp);
                            let cur = self.get(mm, nn, pp);
                            self.set(mm, nn, pp, cur + prev);
                        }
                    }
                }
            }
            2 => {
                for pp in 1..p_dim {
                    for nn in 0..n_dim {
                        for mm in 0..m_dim {
                            let prev = self.get(mm, nn, pp - 1);
                            let cur = self.get(mm, nn, pp);
                            self.set(mm, nn, pp, cur + prev);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    // ---------------------------------------------------------------
    // Cross-correlation
    // ---------------------------------------------------------------

    /// Normalized 2D cross-correlation with a template.
    ///
    /// Computes the normalized cross-correlation for each z-plane independently.
    /// The output has the same dimensions as `self`. The template is centered on
    /// each output pixel position. Out-of-bounds regions are zero-padded.
    pub fn xcorr2(&self, template: &FimImage) -> FimImage {
        let (m_dim, n_dim, p_dim) = self.dims();
        let (tm, tn, _tp) = template.dims();
        let half_m = tm / 2;
        let half_n = tn / 2;

        // Pre-compute template mean and stddev (use plane 0 of template)
        let t_slice: Vec<f32> = {
            let mut v = Vec::with_capacity(tm * tn);
            for nn in 0..tn {
                for mm in 0..tm {
                    v.push(template.get(mm, nn, 0));
                }
            }
            v
        };
        let t_n = t_slice.len() as f64;
        let t_mean: f64 = t_slice.iter().map(|&v| v as f64).sum::<f64>() / t_n;
        let t_std: f64 = {
            let var: f64 = t_slice
                .iter()
                .map(|&v| {
                    let d = v as f64 - t_mean;
                    d * d
                })
                .sum();
            (var / t_n).sqrt()
        };

        let mut out = FimImage::zeros(m_dim, n_dim, p_dim);

        if t_std < 1e-12 {
            return out; // Template is constant; correlation undefined.
        }

        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    // Extract patch under template centered at (mm, nn)
                    let mut patch_sum = 0.0f64;
                    let mut patch_sq_sum = 0.0f64;
                    let mut cross_sum = 0.0f64;
                    let mut count = 0u32;

                    for tn_i in 0..tn {
                        let sy = nn as isize + tn_i as isize - half_n as isize;
                        if sy < 0 || sy >= n_dim as isize {
                            continue;
                        }
                        for tm_i in 0..tm {
                            let sx = mm as isize + tm_i as isize - half_m as isize;
                            if sx < 0 || sx >= m_dim as isize {
                                continue;
                            }
                            let iv = self.get(sx as usize, sy as usize, pp) as f64;
                            let tv = t_slice[tn_i * tm + tm_i] as f64;
                            patch_sum += iv;
                            patch_sq_sum += iv * iv;
                            cross_sum += iv * tv;
                            count += 1;
                        }
                    }

                    if count == 0 {
                        continue;
                    }
                    let cnt = count as f64;
                    let p_mean = patch_sum / cnt;
                    let p_std = (patch_sq_sum / cnt - p_mean * p_mean).max(0.0).sqrt();
                    if p_std < 1e-12 {
                        continue;
                    }
                    let ncc = (cross_sum / cnt - p_mean * t_mean) / (p_std * t_std);
                    out.set(mm, nn, pp, ncc as f32);
                }
            }
        }
        out
    }
}

/// Quickselect: find the k-th smallest element in `data`.
fn quickselect(data: &mut [f32], k: usize) -> f32 {
    if data.len() == 1 {
        return data[0];
    }
    let pivot_idx = data.len() / 2;
    let pivot = data[pivot_idx];

    let mut lo: Vec<f32> = Vec::new();
    let mut hi: Vec<f32> = Vec::new();
    let mut eq: Vec<f32> = Vec::new();

    for &v in data.iter() {
        if v < pivot {
            lo.push(v);
        } else if v > pivot {
            hi.push(v);
        } else {
            eq.push(v);
        }
    }

    if k < lo.len() {
        quickselect(&mut lo, k)
    } else if k < lo.len() + eq.len() {
        pivot
    } else {
        quickselect(&mut hi, k - lo.len() - eq.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_max_sum_mean() {
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let img = FimImage::from_slice(3, 2, 2, &data).unwrap();
        assert_eq!(img.min(), 1.0);
        assert_eq!(img.max(), 12.0);
        assert_eq!(img.sum(), 78.0);
        assert!((img.mean() - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let img = FimImage::from_slice(100, 1, 1, &data).unwrap();
        assert!((img.percentile(50.0) - 50.0).abs() < 1.0);
        assert_eq!(img.percentile(0.0), 0.0);
        assert_eq!(img.percentile(100.0), 99.0);
    }

    #[test]
    fn test_normalize_sum1() {
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let mut img = FimImage::from_slice(2, 2, 2, &data).unwrap();
        img.normalize_sum1();
        assert!((img.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_project_positive() {
        let data = vec![-1.0, 2.0, -3.0, 4.0];
        let mut img = FimImage::from_slice(2, 2, 1, &data).unwrap();
        img.project_positive();
        assert_eq!(img.as_slice(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_max_projection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = FimImage::from_slice(2, 2, 2, &data).unwrap();
        let proj = img.max_projection();
        assert_eq!(proj.dims(), (2, 2, 1));
        assert_eq!(proj.get(0, 0, 0), 5.0);
        assert_eq!(proj.get(1, 1, 0), 8.0);
    }

    #[test]
    fn test_mse() {
        let a = FimImage::constant(2, 2, 1, 1.0);
        let b = FimImage::constant(2, 2, 1, 2.0);
        assert!((FimImage::mse(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_argmax() {
        let mut img = FimImage::zeros(3, 3, 3);
        img.set(1, 2, 0, 99.0);
        let (m, n, p, v) = img.argmax();
        assert_eq!((m, n, p), (1, 2, 0));
        assert_eq!(v, 99.0);
    }
}
