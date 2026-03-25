use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::error::{DwError, Result};

/// Row-major floating-point table with named columns.
///
/// Equivalent to the C `ftab_t` struct in deconwolf.  Data is stored in a
/// flat `Vec<f32>` in row-major order: element `(r, c)` lives at index
/// `r * ncol + c`.
#[derive(Debug, Clone)]
pub struct FTab {
    data: Vec<f32>,
    nrow: usize,
    ncol: usize,
    colnames: Vec<String>,
}

impl FTab {
    // ------------------------------------------------------------------ //
    //  Construction
    // ------------------------------------------------------------------ //

    /// Create an empty table with the given number of columns.
    pub fn new(ncol: usize) -> Self {
        Self {
            data: Vec::new(),
            nrow: 0,
            ncol,
            colnames: Vec::new(),
        }
    }

    /// Create a table from existing data.
    ///
    /// Returns an error if `data.len() != nrow * ncol`.
    pub fn from_data(nrow: usize, ncol: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != nrow * ncol {
            return Err(DwError::InvalidDimensions(format!(
                "FTab::from_data: expected {} elements ({}x{}), got {}",
                nrow * ncol,
                nrow,
                ncol,
                data.len()
            )));
        }
        Ok(Self {
            data,
            nrow,
            ncol,
            colnames: Vec::new(),
        })
    }

    /// Set column names (builder pattern).
    ///
    /// Only the first `min(names.len(), ncol)` names are used.
    pub fn with_colnames(mut self, names: &[&str]) -> Self {
        self.colnames = names.iter().take(self.ncol).map(|s| s.to_string()).collect();
        self
    }

    // ------------------------------------------------------------------ //
    //  Accessors
    // ------------------------------------------------------------------ //

    /// Number of rows.
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    /// Number of columns.
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Get a single element.  Panics on out-of-bounds.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.nrow && col < self.ncol, "FTab::get out of bounds");
        self.data[row * self.ncol + col]
    }

    /// Set a single element.  Panics on out-of-bounds.
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(row < self.nrow && col < self.ncol, "FTab::set out of bounds");
        self.data[row * self.ncol + col] = val;
    }

    /// Look up the index of a column by name, or `None` if not found.
    pub fn get_col_index(&self, name: &str) -> Option<usize> {
        self.colnames.iter().position(|n| n == name)
    }

    /// Extract all values in a column as a new `Vec<f32>`.
    pub fn col_data(&self, col: usize) -> Vec<f32> {
        assert!(col < self.ncol, "FTab::col_data: column index out of bounds");
        (0..self.nrow).map(|r| self.data[r * self.ncol + col]).collect()
    }

    // ------------------------------------------------------------------ //
    //  Row operations
    // ------------------------------------------------------------------ //

    /// Append a row.  Panics if `row.len() != ncol`.
    pub fn insert_row(&mut self, row: &[f32]) {
        assert_eq!(
            row.len(),
            self.ncol,
            "FTab::insert_row: expected {} values, got {}",
            self.ncol,
            row.len()
        );
        self.data.extend_from_slice(row);
        self.nrow += 1;
    }

    /// Keep only the first `n` rows.  If `n >= nrow`, this is a no-op.
    pub fn head(&mut self, n: usize) {
        if n < self.nrow {
            self.data.truncate(n * self.ncol);
            self.nrow = n;
        }
    }

    /// Keep only the rows where the corresponding entry in `mask` is `true`.
    ///
    /// Panics if `mask.len() != nrow`.
    pub fn subselect_rows(&mut self, mask: &[bool]) {
        assert_eq!(
            mask.len(),
            self.nrow,
            "FTab::subselect_rows: mask length mismatch"
        );
        let mut new_data = Vec::with_capacity(self.ncol * mask.iter().filter(|&&b| b).count());
        for (r, &keep) in mask.iter().enumerate() {
            if keep {
                let start = r * self.ncol;
                new_data.extend_from_slice(&self.data[start..start + self.ncol]);
            }
        }
        self.nrow = new_data.len() / self.ncol;
        self.data = new_data;
    }

    /// Sort rows by the values in the given column.
    pub fn sort_by_col(&mut self, col: usize, descending: bool) {
        assert!(col < self.ncol, "FTab::sort_by_col: column index out of bounds");

        // Build an index array and sort it.
        let mut indices: Vec<usize> = (0..self.nrow).collect();
        indices.sort_by(|&a, &b| {
            let va = self.data[a * self.ncol + col];
            let vb = self.data[b * self.ncol + col];
            if descending {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Reorder data according to the sorted indices.
        let old = self.data.clone();
        for (new_r, &old_r) in indices.iter().enumerate() {
            let src = old_r * self.ncol;
            let dst = new_r * self.ncol;
            self.data[dst..dst + self.ncol].copy_from_slice(&old[src..src + self.ncol]);
        }
    }

    // ------------------------------------------------------------------ //
    //  Column operations
    // ------------------------------------------------------------------ //

    /// Overwrite all values in a column.  Panics if `data.len() != nrow`.
    pub fn set_col_data(&mut self, col: usize, data: &[f32]) {
        assert!(col < self.ncol, "FTab::set_col_data: column index out of bounds");
        assert_eq!(
            data.len(),
            self.nrow,
            "FTab::set_col_data: data length mismatch"
        );
        for (r, &val) in data.iter().enumerate() {
            self.data[r * self.ncol + col] = val;
        }
    }

    /// Append a new column with the given name and data.
    ///
    /// Panics if `data.len() != nrow`.
    pub fn insert_col(&mut self, name: &str, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.nrow,
            "FTab::insert_col: data length mismatch"
        );
        let new_ncol = self.ncol + 1;
        let mut new_data = Vec::with_capacity(self.nrow * new_ncol);
        for r in 0..self.nrow {
            let start = r * self.ncol;
            new_data.extend_from_slice(&self.data[start..start + self.ncol]);
            new_data.push(data[r]);
        }
        self.data = new_data;
        self.ncol = new_ncol;
        self.colnames.push(name.to_string());
    }

    /// Create a new table by horizontally concatenating columns from `self`
    /// and `other`.  Both tables must have the same number of rows.
    pub fn concatenate_columns(&self, other: &FTab) -> Result<FTab> {
        if self.nrow != other.nrow {
            return Err(DwError::InvalidDimensions(format!(
                "FTab::concatenate_columns: row count mismatch ({} vs {})",
                self.nrow, other.nrow
            )));
        }
        let new_ncol = self.ncol + other.ncol;
        let mut new_data = Vec::with_capacity(self.nrow * new_ncol);
        for r in 0..self.nrow {
            let s1 = r * self.ncol;
            new_data.extend_from_slice(&self.data[s1..s1 + self.ncol]);
            let s2 = r * other.ncol;
            new_data.extend_from_slice(&other.data[s2..s2 + other.ncol]);
        }
        let mut colnames = self.colnames.clone();
        colnames.extend(other.colnames.iter().cloned());
        Ok(FTab {
            data: new_data,
            nrow: self.nrow,
            ncol: new_ncol,
            colnames,
        })
    }

    // ------------------------------------------------------------------ //
    //  I/O
    // ------------------------------------------------------------------ //

    /// Read a table from a TSV (tab-separated values) file.
    pub fn from_tsv(path: &Path) -> Result<Self> {
        Self::from_delimited(path, '\t')
    }

    /// Read a table from a CSV (comma-separated values) file.
    pub fn from_csv(path: &Path) -> Result<Self> {
        Self::from_delimited(path, ',')
    }

    /// Write the table to a TSV file.
    pub fn write_tsv(&self, path: &Path) -> Result<()> {
        self.write_delimited(path, '\t')
    }

    /// Write the table to a CSV file.
    pub fn write_csv(&self, path: &Path) -> Result<()> {
        self.write_delimited(path, ',')
    }

    fn from_delimited(path: &Path, delim: char) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // First line: column names.
        let header = lines
            .next()
            .ok_or_else(|| DwError::InvalidDimensions("FTab: empty file".into()))??;
        let colnames: Vec<String> = header.split(delim).map(|s| s.trim().to_string()).collect();
        let ncol = colnames.len();

        let mut data = Vec::new();
        let mut nrow = 0usize;
        for line_result in lines {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let fields: Vec<&str> = trimmed.split(delim).collect();
            if fields.len() != ncol {
                return Err(DwError::InvalidDimensions(format!(
                    "FTab: row {} has {} fields, expected {}",
                    nrow, fields.len(), ncol
                )));
            }
            for f in &fields {
                let val: f32 = f.trim().parse().map_err(|e| {
                    DwError::InvalidDimensions(format!("FTab: parse error: {}", e))
                })?;
                data.push(val);
            }
            nrow += 1;
        }

        Ok(Self {
            data,
            nrow,
            ncol,
            colnames,
        })
    }

    fn write_delimited(&self, path: &Path, delim: char) -> Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Header
        if !self.colnames.is_empty() {
            let header: Vec<&str> = self.colnames.iter().map(|s| s.as_str()).collect();
            writeln!(file, "{}", header.join(&delim.to_string()))?;
        } else {
            // Generate placeholder names: col0, col1, ...
            let names: Vec<String> = (0..self.ncol).map(|i| format!("col{}", i)).collect();
            writeln!(file, "{}", names.join(&delim.to_string()))?;
        }

        // Data rows
        for r in 0..self.nrow {
            let start = r * self.ncol;
            let vals: Vec<String> = self.data[start..start + self.ncol]
                .iter()
                .map(|v| format!("{}", v))
                .collect();
            writeln!(file, "{}", vals.join(&delim.to_string()))?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------ //
    //  Conversion
    // ------------------------------------------------------------------ //

    /// Return all data promoted to `f64`.
    pub fn data_as_f64(&self) -> Vec<f64> {
        self.data.iter().map(|&v| v as f64).collect()
    }

    /// Borrow the underlying flat data slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// ====================================================================== //
//  Tests
// ====================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_insert_sort() {
        let mut t = FTab::new(3).with_colnames(&["x", "y", "val"]);
        t.insert_row(&[1.0, 2.0, 30.0]);
        t.insert_row(&[4.0, 5.0, 10.0]);
        t.insert_row(&[7.0, 8.0, 20.0]);

        assert_eq!(t.nrow(), 3);
        assert_eq!(t.ncol(), 3);
        assert_eq!(t.get(0, 2), 30.0);

        // Sort ascending by "val" (column 2).
        t.sort_by_col(2, false);
        assert_eq!(t.get(0, 2), 10.0);
        assert_eq!(t.get(1, 2), 20.0);
        assert_eq!(t.get(2, 2), 30.0);

        // Sort descending.
        t.sort_by_col(2, true);
        assert_eq!(t.get(0, 2), 30.0);
        assert_eq!(t.get(2, 2), 10.0);
    }

    #[test]
    fn test_head_and_subselect() {
        let mut t = FTab::new(2);
        t.insert_row(&[1.0, 2.0]);
        t.insert_row(&[3.0, 4.0]);
        t.insert_row(&[5.0, 6.0]);

        let mut t2 = t.clone();
        t2.head(2);
        assert_eq!(t2.nrow(), 2);
        assert_eq!(t2.get(1, 0), 3.0);

        t.subselect_rows(&[true, false, true]);
        assert_eq!(t.nrow(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 5.0);
    }

    #[test]
    fn test_tsv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tsv");

        let mut t = FTab::new(2).with_colnames(&["a", "b"]);
        t.insert_row(&[1.5, 2.5]);
        t.insert_row(&[3.0, 4.0]);
        t.write_tsv(&path).unwrap();

        let t2 = FTab::from_tsv(&path).unwrap();
        assert_eq!(t2.nrow(), 2);
        assert_eq!(t2.ncol(), 2);
        assert_eq!(t2.get_col_index("a"), Some(0));
        assert_eq!(t2.get_col_index("b"), Some(1));
        assert_eq!(t2.get(0, 0), 1.5);
        assert_eq!(t2.get(1, 1), 4.0);
    }

    #[test]
    fn test_csv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let mut t = FTab::new(3).with_colnames(&["x", "y", "z"]);
        t.insert_row(&[10.0, 20.0, 30.0]);
        t.write_csv(&path).unwrap();

        let t2 = FTab::from_csv(&path).unwrap();
        assert_eq!(t2.nrow(), 1);
        assert_eq!(t2.get(0, 1), 20.0);
    }

    #[test]
    fn test_column_operations() {
        let mut t = FTab::new(2).with_colnames(&["a", "b"]);
        t.insert_row(&[1.0, 2.0]);
        t.insert_row(&[3.0, 4.0]);

        // col_data
        assert_eq!(t.col_data(0), vec![1.0, 3.0]);
        assert_eq!(t.col_data(1), vec![2.0, 4.0]);

        // set_col_data
        t.set_col_data(1, &[20.0, 40.0]);
        assert_eq!(t.get(0, 1), 20.0);
        assert_eq!(t.get(1, 1), 40.0);

        // insert_col
        t.insert_col("c", &[100.0, 200.0]);
        assert_eq!(t.ncol(), 3);
        assert_eq!(t.get(0, 2), 100.0);
        assert_eq!(t.get(1, 2), 200.0);
        assert_eq!(t.get_col_index("c"), Some(2));

        // concatenate_columns
        let mut other = FTab::new(1).with_colnames(&["d"]);
        other.insert_row(&[0.5]);
        other.insert_row(&[0.6]);
        let merged = t.concatenate_columns(&other).unwrap();
        assert_eq!(merged.ncol(), 4);
        assert_eq!(merged.get(0, 3), 0.5);
        assert_eq!(merged.get(1, 3), 0.6);
    }

    #[test]
    fn test_from_data() {
        let t = FTab::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.get(1, 2), 6.0);

        // Wrong length should fail.
        assert!(FTab::from_data(2, 3, vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_data_as_f64_and_as_slice() {
        let t = FTab::from_data(1, 2, vec![1.0, 2.0]).unwrap();
        assert_eq!(t.data_as_f64(), vec![1.0f64, 2.0f64]);
        assert_eq!(t.as_slice(), &[1.0f32, 2.0f32]);
    }
}
