use rayon::prelude::*;

use crate::image::FimImage;

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
    pub fn zcrop(&self, zcrop: usize) -> crate::error::Result<FimImage> {
        let p_dim = self.p();
        if 2 * zcrop >= p_dim {
            return Err(crate::error::DwError::InvalidDimensions(format!(
                "Cannot remove {} slices from each end of {}-slice image",
                zcrop, p_dim
            )));
        }
        self.get_cuboid(0, self.m(), 0, self.n(), zcrop, p_dim - zcrop)
    }

    /// Auto z-crop to `new_p` slices centered on the brightest plane.
    pub fn auto_zcrop(&self, new_p: usize) -> crate::error::Result<FimImage> {
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
