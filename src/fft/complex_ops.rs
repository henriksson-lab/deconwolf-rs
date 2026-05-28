use num_complex::Complex;
use rayon::prelude::*;

use super::backend::{FftError, Result};

/// Element-wise complex multiplication: C[k] = A[k] * B[k].
pub fn complex_mul(a: &[Complex<f32>], b: &[Complex<f32>]) -> Result<Vec<Complex<f32>>> {
    validate_same_len(a.len(), b.len(), "complex multiplication")?;
    Ok(a.par_iter().zip(b.par_iter()).map(|(a, b)| a * b).collect())
}

/// Element-wise multiplication with conjugate: C[k] = conj(A[k]) * B[k].
pub fn complex_mul_conj(a: &[Complex<f32>], b: &[Complex<f32>]) -> Result<Vec<Complex<f32>>> {
    validate_same_len(a.len(), b.len(), "conjugate complex multiplication")?;
    Ok(a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| a.conj() * b)
        .collect())
}

/// In-place multiplication: B[k] *= A[k].
pub fn complex_mul_inplace(a: &[Complex<f32>], b: &mut [Complex<f32>]) -> Result<()> {
    validate_same_len(a.len(), b.len(), "in-place complex multiplication")?;
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(b, a)| *b *= a);
    Ok(())
}

/// In-place conjugate multiplication: B[k] *= conj(A[k]).
pub fn complex_mul_conj_inplace(a: &[Complex<f32>], b: &mut [Complex<f32>]) -> Result<()> {
    validate_same_len(
        a.len(),
        b.len(),
        "in-place conjugate complex multiplication",
    )?;
    b.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(b, a)| *b *= a.conj());
    Ok(())
}

fn validate_same_len(a_len: usize, b_len: usize, context: &str) -> Result<()> {
    if a_len != b_len {
        return Err(FftError::InvalidDimensions(format!(
            "{} requires equal lengths, got {} and {}",
            context, a_len, b_len
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_mul() {
        let a = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let c = complex_mul(&a, &b).unwrap();
        // (1+2i)(5+6i) = 5+6i+10i+12i² = 5-12+16i = -7+16i
        assert!((c[0].re - (-7.0)).abs() < 1e-6);
        assert!((c[0].im - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_mul_conj() {
        let a = vec![Complex::new(1.0, 2.0)];
        let b = vec![Complex::new(3.0, 4.0)];
        let c = complex_mul_conj(&a, &b).unwrap();
        // conj(1+2i) * (3+4i) = (1-2i)(3+4i) = 3+4i-6i-8i² = 11-2i
        assert!((c[0].re - 11.0).abs() < 1e-6);
        assert!((c[0].im - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn complex_ops_reject_length_mismatch() {
        let a = vec![Complex::new(1.0, 0.0)];
        let mut b = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];

        assert!(complex_mul(&a, &b).is_err());
        assert!(complex_mul_conj(&a, &b).is_err());
        assert!(complex_mul_inplace(&a, &mut b).is_err());
        assert!(complex_mul_conj_inplace(&a, &mut b).is_err());
    }
}
