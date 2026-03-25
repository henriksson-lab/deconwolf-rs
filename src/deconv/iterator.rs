use super::config::IterType;

/// Controls iteration stopping for deconvolution methods.
pub struct DwIterator {
    pub iter: usize,
    pub max_iter: usize,
    pub error: f64,
    pub last_error: f64,
    pub rel_error_threshold: f64,
    pub abs_error_threshold: f64,
    pub iter_type: IterType,
}

impl DwIterator {
    pub fn new(
        iter_type: IterType,
        n_iter: usize,
        max_iter: usize,
        rel_error: f64,
        abs_error: f64,
    ) -> Self {
        Self {
            iter: 0,
            max_iter: match iter_type {
                IterType::Fixed => n_iter,
                _ => max_iter,
            },
            error: f64::INFINITY,
            last_error: f64::INFINITY,
            rel_error_threshold: rel_error,
            abs_error_threshold: abs_error,
            iter_type,
        }
    }

    /// Set the error for the current iteration.
    pub fn set_error(&mut self, error: f64) {
        self.last_error = self.error;
        self.error = error;
    }

    /// Advance to the next iteration.
    /// Returns `true` if iteration should continue.
    pub fn next(&mut self) -> bool {
        self.iter += 1;

        if self.iter > self.max_iter {
            return false;
        }

        match self.iter_type {
            IterType::Fixed => self.iter <= self.max_iter,
            IterType::Relative => {
                if self.iter <= 2 {
                    return true;
                }
                let rel_change = if self.error != 0.0 {
                    ((self.error - self.last_error) / self.error).abs()
                } else {
                    0.0
                };
                rel_change >= self.rel_error_threshold
            }
            IterType::Absolute => self.error >= self.abs_error_threshold,
        }
    }

    /// Current iteration number.
    pub fn current(&self) -> usize {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_iterations() {
        let mut it = DwIterator::new(IterType::Fixed, 5, 250, 0.02, 0.0);
        for _ in 0..5 {
            assert!(it.next());
        }
        assert!(!it.next());
    }

    #[test]
    fn test_relative_convergence() {
        let mut it = DwIterator::new(IterType::Relative, 50, 100, 0.01, 0.0);
        it.set_error(100.0);
        assert!(it.next()); // iter 1

        it.set_error(50.0);
        assert!(it.next()); // iter 2, always continue

        it.set_error(49.9);
        assert!(!it.next()); // relative change < 0.01, should stop
    }
}
