use num_complex::Complex;
use rayon::prelude::*;

/// Element-wise complex multiplication: C[k] = A[k] * B[k].
pub fn complex_mul(a: &[Complex<f32>], b: &[Complex<f32>]) -> Vec<Complex<f32>> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| a * b)
        .collect()
}

/// Element-wise multiplication with conjugate: C[k] = conj(A[k]) * B[k].
pub fn complex_mul_conj(a: &[Complex<f32>], b: &[Complex<f32>]) -> Vec<Complex<f32>> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| a.conj() * b)
        .collect()
}

/// In-place multiplication: B[k] *= A[k].
pub fn complex_mul_inplace(a: &[Complex<f32>], b: &mut [Complex<f32>]) {
    assert_eq!(a.len(), b.len());
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(b, a)| *b *= a);
}

/// In-place conjugate multiplication: B[k] *= conj(A[k]).
pub fn complex_mul_conj_inplace(a: &[Complex<f32>], b: &mut [Complex<f32>]) {
    assert_eq!(a.len(), b.len());
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(b, a)| *b *= a.conj());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_mul() {
        let a = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let c = complex_mul(&a, &b);
        // (1+2i)(5+6i) = 5+6i+10i+12i² = 5-12+16i = -7+16i
        assert!((c[0].re - (-7.0)).abs() < 1e-6);
        assert!((c[0].im - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_mul_conj() {
        let a = vec![Complex::new(1.0, 2.0)];
        let b = vec![Complex::new(3.0, 4.0)];
        let c = complex_mul_conj(&a, &b);
        // conj(1+2i) * (3+4i) = (1-2i)(3+4i) = 3+4i-6i-8i² = 11-2i
        assert!((c[0].re - 11.0).abs() < 1e-6);
        assert!((c[0].im - (-2.0)).abs() < 1e-6);
    }
}
