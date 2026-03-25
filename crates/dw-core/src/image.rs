use ndarray::Array3;

use crate::error::{DwError, Result};

/// 3D floating-point image.
///
/// Wraps an `ndarray::Array3<f32>` with shape `[P, N, M]` (z, y, x).
/// This matches the C memory layout where M (x) is stride-1.
#[derive(Debug, Clone)]
pub struct FimImage {
    pub data: Array3<f32>,
}

impl FimImage {
    /// Create a zero-initialized image of dimensions M x N x P.
    pub fn zeros(m: usize, n: usize, p: usize) -> Self {
        Self {
            data: Array3::zeros((p, n, m)),
        }
    }

    /// Create an image filled with a constant value.
    pub fn constant(m: usize, n: usize, p: usize, value: f32) -> Self {
        Self {
            data: Array3::from_elem((p, n, m), value),
        }
    }

    /// Create from a flat f32 slice in row-major order (M stride-1).
    /// The slice length must equal m * n * p.
    pub fn from_slice(m: usize, n: usize, p: usize, data: &[f32]) -> Result<Self> {
        if data.len() != m * n * p {
            return Err(DwError::InvalidDimensions(format!(
                "Expected {} elements, got {}",
                m * n * p,
                data.len()
            )));
        }
        let arr = Array3::from_shape_vec((p, n, m), data.to_vec())
            .map_err(|e| DwError::InvalidDimensions(e.to_string()))?;
        Ok(Self { data: arr })
    }

    /// Create from an owned Vec<f32>.
    pub fn from_vec(m: usize, n: usize, p: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != m * n * p {
            return Err(DwError::InvalidDimensions(format!(
                "Expected {} elements, got {}",
                m * n * p,
                data.len()
            )));
        }
        let arr = Array3::from_shape_vec((p, n, m), data)
            .map_err(|e| DwError::InvalidDimensions(e.to_string()))?;
        Ok(Self { data: arr })
    }

    /// Image width (x dimension, stride-1 in C layout).
    #[inline]
    pub fn m(&self) -> usize {
        self.data.shape()[2]
    }

    /// Image height (y dimension).
    #[inline]
    pub fn n(&self) -> usize {
        self.data.shape()[1]
    }

    /// Image depth (z dimension, number of slices).
    #[inline]
    pub fn p(&self) -> usize {
        self.data.shape()[0]
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the image is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dimensions as (M, N, P) tuple.
    #[inline]
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.m(), self.n(), self.p())
    }

    /// Access pixel at (m, n, p) = (x, y, z).
    #[inline]
    pub fn get(&self, m: usize, n: usize, p: usize) -> f32 {
        self.data[[p, n, m]]
    }

    /// Set pixel at (m, n, p) = (x, y, z).
    #[inline]
    pub fn set(&mut self, m: usize, n: usize, p: usize, value: f32) {
        self.data[[p, n, m]] = value;
    }

    /// Get a flat slice of the underlying data (contiguous, C-order).
    pub fn as_slice(&self) -> &[f32] {
        self.data
            .as_slice()
            .expect("FimImage data is not contiguous")
    }

    /// Get a mutable flat slice of the underlying data.
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        self.data
            .as_slice_mut()
            .expect("FimImage data is not contiguous")
    }

    /// Extract a single z-plane as a new 2D image (P=1).
    pub fn get_plane(&self, plane: usize) -> Result<FimImage> {
        if plane >= self.p() {
            return Err(DwError::InvalidDimensions(format!(
                "Plane {} out of range (P={})",
                plane,
                self.p()
            )));
        }
        let slice = self.data.index_axis(ndarray::Axis(0), plane);
        let mut out = FimImage::zeros(self.m(), self.n(), 1);
        out.data
            .index_axis_mut(ndarray::Axis(0), 0)
            .assign(&slice);
        Ok(out)
    }

    /// Insert image `src` into `self` at upper-left corner (0,0,0).
    /// Copies the overlapping region.
    pub fn insert(&mut self, src: &FimImage) {
        let cm = src.m().min(self.m());
        let cn = src.n().min(self.n());
        let cp = src.p().min(self.p());
        for pp in 0..cp {
            for nn in 0..cn {
                for mm in 0..cm {
                    self.data[[pp, nn, mm]] = src.data[[pp, nn, mm]];
                }
            }
        }
    }

    /// Extract a sub-region [0..m, 0..n, 0..p] as a new image.
    pub fn subregion(&self, m: usize, n: usize, p: usize) -> Result<FimImage> {
        if m > self.m() || n > self.n() || p > self.p() {
            return Err(DwError::InvalidDimensions(format!(
                "Subregion ({},{},{}) exceeds image ({},{},{})",
                m,
                n,
                p,
                self.m(),
                self.n(),
                self.p()
            )));
        }
        let sub = self
            .data
            .slice(ndarray::s![..p, ..n, ..m])
            .to_owned();
        Ok(FimImage { data: sub })
    }

    /// Extract a cuboid region [m0..m1, n0..n1, p0..p1] as a new image.
    pub fn get_cuboid(
        &self,
        m0: usize,
        m1: usize,
        n0: usize,
        n1: usize,
        p0: usize,
        p1: usize,
    ) -> Result<FimImage> {
        if m1 > self.m() || n1 > self.n() || p1 > self.p() || m0 > m1 || n0 > n1 || p0 > p1 {
            return Err(DwError::InvalidDimensions(format!(
                "Cuboid [{},{}]x[{},{}]x[{},{}] invalid for image ({},{},{})",
                m0,
                m1,
                n0,
                n1,
                p0,
                p1,
                self.m(),
                self.n(),
                self.p()
            )));
        }
        let sub = self
            .data
            .slice(ndarray::s![p0..p1, n0..n1, m0..m1])
            .to_owned();
        Ok(FimImage { data: sub })
    }

    /// Expand image to larger dimensions, zero-padding.
    pub fn expand(&self, m: usize, n: usize, p: usize) -> FimImage {
        let mut out = FimImage::zeros(m, n, p);
        out.insert(self);
        out
    }

    /// Flip all three dimensions: T = flip(flip(flip(A,1),2),3).
    pub fn flipall(&self) -> FimImage {
        let (m_dim, n_dim, p_dim) = self.dims();
        let mut out = FimImage::zeros(m_dim, n_dim, p_dim);
        for pp in 0..p_dim {
            for nn in 0..n_dim {
                for mm in 0..m_dim {
                    out.data[[p_dim - 1 - pp, n_dim - 1 - nn, m_dim - 1 - mm]] =
                        self.data[[pp, nn, mm]];
                }
            }
        }
        out
    }

    /// Circular shift by (sm, sn, sp) pixels in each dimension.
    pub fn circshift(&mut self, sm: i64, sn: i64, sp: i64) {
        let (m_dim, n_dim, p_dim) = self.dims();
        // Shift along M (axis 2)
        if sm != 0 && m_dim > 1 {
            Self::circshift_axis(&mut self.data, 2, m_dim, sm);
        }
        // Shift along N (axis 1)
        if sn != 0 && n_dim > 1 {
            Self::circshift_axis(&mut self.data, 1, n_dim, sn);
        }
        // Shift along P (axis 0)
        if sp != 0 && p_dim > 1 {
            Self::circshift_axis(&mut self.data, 0, p_dim, sp);
        }
    }

    fn circshift_axis(data: &mut Array3<f32>, axis: usize, len: usize, shift: i64) {
        let shift = ((shift % len as i64) + len as i64) as usize % len;
        if shift == 0 {
            return;
        }
        // For each "lane" along the given axis, perform circular shift
        let shape = data.shape().to_vec();
        let (d0, d1, d2) = (shape[0], shape[1], shape[2]);

        match axis {
            0 => {
                let mut buf = vec![0.0f32; d0];
                for n in 0..d1 {
                    for m in 0..d2 {
                        for p in 0..d0 {
                            buf[p] = data[[p, n, m]];
                        }
                        for p in 0..d0 {
                            data[[(p + shift) % d0, n, m]] = buf[p];
                        }
                    }
                }
            }
            1 => {
                let mut buf = vec![0.0f32; d1];
                for p in 0..d0 {
                    for m in 0..d2 {
                        for n in 0..d1 {
                            buf[n] = data[[p, n, m]];
                        }
                        for n in 0..d1 {
                            data[[p, (n + shift) % d1, m]] = buf[n];
                        }
                    }
                }
            }
            2 => {
                let mut buf = vec![0.0f32; d2];
                for p in 0..d0 {
                    for n in 0..d1 {
                        for m in 0..d2 {
                            buf[m] = data[[p, n, m]];
                        }
                        for m in 0..d2 {
                            data[[p, n, (m + shift) % d2]] = buf[m];
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let img = FimImage::zeros(10, 20, 3);
        assert_eq!(img.dims(), (10, 20, 3));
        assert_eq!(img.len(), 600);
        assert_eq!(img.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_from_slice() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = FimImage::from_slice(4, 3, 2, &data).unwrap();
        assert_eq!(img.dims(), (4, 3, 2));
        assert_eq!(img.get(0, 0, 0), 0.0);
        assert_eq!(img.get(1, 0, 0), 1.0);
        assert_eq!(img.get(0, 1, 0), 4.0); // stride M=4
    }

    #[test]
    fn test_get_set() {
        let mut img = FimImage::zeros(5, 5, 5);
        img.set(2, 3, 1, 42.0);
        assert_eq!(img.get(2, 3, 1), 42.0);
    }

    #[test]
    fn test_subregion() {
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let img = FimImage::from_slice(5, 4, 3, &data).unwrap();
        let sub = img.subregion(3, 2, 2).unwrap();
        assert_eq!(sub.dims(), (3, 2, 2));
        assert_eq!(sub.get(0, 0, 0), img.get(0, 0, 0));
        assert_eq!(sub.get(2, 1, 1), img.get(2, 1, 1));
    }

    #[test]
    fn test_flipall() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = FimImage::from_slice(2, 2, 2, &data).unwrap();
        let flipped = img.flipall();
        // First element should become last
        assert_eq!(flipped.get(1, 1, 1), img.get(0, 0, 0));
        assert_eq!(flipped.get(0, 0, 0), img.get(1, 1, 1));
    }

    #[test]
    fn test_circshift_roundtrip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let original = FimImage::from_slice(4, 3, 2, &data).unwrap();
        let mut shifted = original.clone();
        shifted.circshift(2, 1, 1);
        shifted.circshift(-2, -1, -1);
        for i in 0..original.len() {
            assert_eq!(original.as_slice()[i], shifted.as_slice()[i]);
        }
    }

    #[test]
    fn test_expand_insert() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = FimImage::from_slice(2, 2, 2, &data).unwrap();
        let expanded = img.expand(4, 4, 4);
        assert_eq!(expanded.dims(), (4, 4, 4));
        assert_eq!(expanded.get(0, 0, 0), img.get(0, 0, 0));
        assert_eq!(expanded.get(1, 1, 1), img.get(1, 1, 1));
        assert_eq!(expanded.get(3, 3, 3), 0.0);
    }
}
