use num_complex::Complex;
use rustfft::{FftPlanner, FftDirection};

use super::backend::{FftBackend, FftError, Result};

/// 3D FFT backend using the pure-Rust `rustfft` crate.
///
/// Implements 3D real-to-complex transform by decomposing into
/// sequences of 1D FFTs along each axis.
pub struct RustFftBackend {
    m: usize,
    n: usize,
    p: usize,
    /// Complex size: (m/2+1) * n * p
    csize: usize,
    /// Normalization factor: 1 / (m * n * p)
    norm: f32,
}

impl FftBackend for RustFftBackend {
    fn new_context(m: usize, n: usize, p: usize, _threads: usize) -> Result<Self> {
        if m == 0 || n == 0 || p == 0 {
            return Err(FftError::InvalidDimensions("All dimensions must be > 0".into()));
        }
        let cm = m / 2 + 1;
        Ok(Self {
            m,
            n,
            p,
            csize: cm * n * p,
            norm: 1.0 / (m * n * p) as f32,
        })
    }

    fn dims(&self) -> (usize, usize, usize) {
        (self.m, self.n, self.p)
    }

    fn complex_size(&self) -> usize {
        self.csize
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<Complex<f32>>> {
        let total = self.m * self.n * self.p;
        if input.len() != total {
            return Err(FftError::InvalidDimensions(format!(
                "Expected {} elements, got {}",
                total,
                input.len()
            )));
        }

        let (m, n, p) = (self.m, self.n, self.p);
        let cm = m / 2 + 1;

        // Work in full complex domain, then truncate
        let mut data: Vec<Complex<f32>> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let mut planner = FftPlanner::new();

        // 1D FFT along M (axis 2, innermost, stride 1)
        let fft_m = planner.plan_fft(m, FftDirection::Forward);
        for pp in 0..p {
            for nn in 0..n {
                let offset = pp * n * m + nn * m;
                fft_m.process(&mut data[offset..offset + m]);
            }
        }

        // 1D FFT along N (axis 1, stride m)
        let fft_n = planner.plan_fft(n, FftDirection::Forward);
        let mut buf = vec![Complex::new(0.0, 0.0); n];
        for pp in 0..p {
            for mm in 0..m {
                // Gather
                for nn in 0..n {
                    buf[nn] = data[pp * n * m + nn * m + mm];
                }
                fft_n.process(&mut buf);
                // Scatter
                for nn in 0..n {
                    data[pp * n * m + nn * m + mm] = buf[nn];
                }
            }
        }

        // 1D FFT along P (axis 0, stride m*n)
        if p > 1 {
            let fft_p = planner.plan_fft(p, FftDirection::Forward);
            let mut buf = vec![Complex::new(0.0, 0.0); p];
            for nn in 0..n {
                for mm in 0..m {
                    // Gather
                    for pp in 0..p {
                        buf[pp] = data[pp * n * m + nn * m + mm];
                    }
                    fft_p.process(&mut buf);
                    // Scatter
                    for pp in 0..p {
                        data[pp * n * m + nn * m + mm] = buf[pp];
                    }
                }
            }
        }

        // Truncate to real-to-complex half: only keep m/2+1 along M axis
        let mut output = vec![Complex::new(0.0, 0.0); cm * n * p];
        for pp in 0..p {
            for nn in 0..n {
                for mm in 0..cm {
                    output[pp * n * cm + nn * cm + mm] = data[pp * n * m + nn * m + mm];
                }
            }
        }

        Ok(output)
    }

    fn inverse(&self, input: &[Complex<f32>]) -> Result<Vec<f32>> {
        if input.len() != self.csize {
            return Err(FftError::InvalidDimensions(format!(
                "Expected {} complex elements, got {}",
                self.csize,
                input.len()
            )));
        }

        let (m, n, p) = (self.m, self.n, self.p);
        let cm = m / 2 + 1;

        // Expand back to full complex domain using Hermitian symmetry
        let mut data = vec![Complex::new(0.0, 0.0); m * n * p];
        for pp in 0..p {
            for nn in 0..n {
                for mm in 0..cm {
                    data[pp * n * m + nn * m + mm] = input[pp * n * cm + nn * cm + mm];
                }
                // Fill Hermitian conjugates
                for mm in cm..m {
                    let conj_pp = if pp == 0 { 0 } else { p - pp };
                    let conj_nn = if nn == 0 { 0 } else { n - nn };
                    let conj_mm = m - mm;
                    data[pp * n * m + nn * m + mm] =
                        input[conj_pp * n * cm + conj_nn * cm + conj_mm].conj();
                }
            }
        }

        let mut planner = FftPlanner::new();

        // Inverse 1D FFT along P
        if p > 1 {
            let ifft_p = planner.plan_fft(p, FftDirection::Inverse);
            let mut buf = vec![Complex::new(0.0, 0.0); p];
            for nn in 0..n {
                for mm in 0..m {
                    for pp in 0..p {
                        buf[pp] = data[pp * n * m + nn * m + mm];
                    }
                    ifft_p.process(&mut buf);
                    for pp in 0..p {
                        data[pp * n * m + nn * m + mm] = buf[pp];
                    }
                }
            }
        }

        // Inverse 1D FFT along N
        let ifft_n = planner.plan_fft(n, FftDirection::Inverse);
        let mut buf = vec![Complex::new(0.0, 0.0); n];
        for pp in 0..p {
            for mm in 0..m {
                for nn in 0..n {
                    buf[nn] = data[pp * n * m + nn * m + mm];
                }
                ifft_n.process(&mut buf);
                for nn in 0..n {
                    data[pp * n * m + nn * m + mm] = buf[nn];
                }
            }
        }

        // Inverse 1D FFT along M
        let ifft_m = planner.plan_fft(m, FftDirection::Inverse);
        for pp in 0..p {
            for nn in 0..n {
                let offset = pp * n * m + nn * m;
                ifft_m.process(&mut data[offset..offset + m]);
            }
        }

        // Normalize and extract real part
        let norm = self.norm;
        Ok(data.iter().map(|c| c.re * norm).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_1d() {
        let backend = RustFftBackend::new_context(8, 1, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let freq = backend.forward(&input).unwrap();
        let output = backend.inverse(&freq).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-4, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_roundtrip_3d() {
        let backend = RustFftBackend::new_context(4, 3, 2, 1).unwrap();
        let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let freq = backend.forward(&input).unwrap();
        let output = backend.inverse(&freq).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-3, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_complex_size() {
        let backend = RustFftBackend::new_context(8, 4, 2, 1).unwrap();
        assert_eq!(backend.complex_size(), 5 * 4 * 2); // (8/2+1) * 4 * 2
    }
}
