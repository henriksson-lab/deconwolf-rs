use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{DwError, Result};
use crate::image::FimImage;

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Read a .npy file as a FimImage.
/// Supports f32, f64, u8, u16, i16, i32 dtypes.
/// Arrays with 2 or 3 dimensions are supported.
pub fn npy_read(path: &Path) -> Result<FimImage> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|_| DwError::Npy("Not a valid NPY file".into()))?;
    if magic != NPY_MAGIC {
        return Err(DwError::Npy("Invalid NPY magic".into()));
    }

    // Read version
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    // Read header length
    let header_len = if version[0] == 1 {
        reader.read_u16::<LittleEndian>()? as usize
    } else {
        reader.read_u32::<LittleEndian>()? as usize
    };

    // Read header
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);

    // Parse header dict
    let (dtype, fortran_order, shape) = parse_npy_header(&header)?;

    // Convert shape to (M, N, P)
    let (m, n, p) = match shape.len() {
        1 => (shape[0], 1, 1),
        2 => (shape[1], shape[0], 1),
        3 => (shape[2], shape[1], shape[0]),
        _ => {
            return Err(DwError::Npy(format!(
                "Unsupported number of dimensions: {}",
                shape.len()
            )))
        }
    };

    let nel = m * n * p;

    // Read data based on dtype
    let data = match dtype.as_str() {
        "<f4" | "=f4" | "f4" => {
            let mut buf = vec![0.0f32; nel];
            for v in buf.iter_mut() {
                *v = reader.read_f32::<LittleEndian>()?;
            }
            buf
        }
        "<f8" | "=f8" | "f8" => {
            let mut buf = Vec::with_capacity(nel);
            for _ in 0..nel {
                buf.push(reader.read_f64::<LittleEndian>()? as f32);
            }
            buf
        }
        "<u1" | "=u1" | "|u1" | "u1" => {
            let mut raw = vec![0u8; nel];
            reader.read_exact(&mut raw)?;
            raw.iter().map(|&v| v as f32).collect()
        }
        "<u2" | "=u2" | "u2" => {
            let mut buf = Vec::with_capacity(nel);
            for _ in 0..nel {
                buf.push(reader.read_u16::<LittleEndian>()? as f32);
            }
            buf
        }
        "<i2" | "=i2" | "i2" => {
            let mut buf = Vec::with_capacity(nel);
            for _ in 0..nel {
                buf.push(reader.read_i16::<LittleEndian>()? as f32);
            }
            buf
        }
        "<i4" | "=i4" | "i4" => {
            let mut buf = Vec::with_capacity(nel);
            for _ in 0..nel {
                buf.push(reader.read_i32::<LittleEndian>()? as f32);
            }
            buf
        }
        _ => return Err(DwError::Npy(format!("Unsupported dtype: {}", dtype))),
    };

    // Handle Fortran order (column-major) by transposing
    if fortran_order {
        // For now, just load as-is (matching C behavior)
        log::warn!("Fortran-order NPY file, data may need transposition");
    }

    FimImage::from_vec(m, n, p, data)
}

/// Write a FimImage to a .npy file as float32.
pub fn npy_write(path: &Path, img: &FimImage) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (m, n, p) = img.dims();

    // Build shape tuple string
    let shape_str = if p == 1 && n == 1 {
        format!("({},)", m)
    } else if p == 1 {
        format!("({}, {})", n, m)
    } else {
        format!("({}, {}, {})", p, n, m)
    };

    // Build header dict
    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );

    // Pad header to multiple of 64 bytes (including magic + version + header_len)
    let prefix_len = 6 + 2 + 2; // magic + version + header_len (v1)
    let total = prefix_len + header_dict.len() + 1; // +1 for newline
    let padding = (64 - (total % 64)) % 64;
    let padded_header = format!(
        "{}{}{}",
        header_dict,
        " ".repeat(padding),
        "\n"
    );

    // Write magic
    writer.write_all(NPY_MAGIC)?;
    // Version 1.0
    writer.write_all(&[1, 0])?;
    // Header length (u16 LE)
    writer.write_u16::<LittleEndian>(padded_header.len() as u16)?;
    // Header
    writer.write_all(padded_header.as_bytes())?;
    // Data
    for &v in img.as_slice() {
        writer.write_f32::<LittleEndian>(v)?;
    }

    writer.flush()?;
    Ok(())
}

/// Check if a filename has a .npy extension.
pub fn is_npy_file(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("npy"))
        .unwrap_or(false)
}

fn parse_npy_header(header: &str) -> Result<(String, bool, Vec<usize>)> {
    let header = header.trim();

    let mut dtype = String::new();
    let mut fortran_order = false;
    let mut shape: Vec<usize> = Vec::new();

    // Extract 'descr' value
    if let Some(pos) = header.find("'descr'") {
        let rest = &header[pos + 7..];
        if let Some(colon) = rest.find(':') {
            let after = rest[colon + 1..].trim();
            // Find the quoted value
            if let Some(q1) = after.find('\'') {
                if let Some(q2) = after[q1 + 1..].find('\'') {
                    dtype = after[q1 + 1..q1 + 1 + q2].to_string();
                }
            }
        }
    }

    // Extract 'fortran_order' value
    if let Some(pos) = header.find("'fortran_order'") {
        let rest = &header[pos..];
        fortran_order = rest.contains("True");
    }

    // Extract 'shape' value
    if let Some(pos) = header.find("'shape'") {
        let rest = &header[pos..];
        if let Some(paren_start) = rest.find('(') {
            if let Some(paren_end) = rest.find(')') {
                let shape_str = &rest[paren_start + 1..paren_end];
                for dim in shape_str.split(',') {
                    let dim = dim.trim();
                    if !dim.is_empty() {
                        shape.push(
                            dim.parse::<usize>()
                                .map_err(|_| DwError::Npy(format!("Invalid shape dimension: {}", dim)))?,
                        );
                    }
                }
            }
        }
    }

    if dtype.is_empty() {
        return Err(DwError::Npy("Missing descr in NPY header".into()));
    }
    if shape.is_empty() {
        return Err(DwError::Npy("Missing shape in NPY header".into()));
    }

    Ok((dtype, fortran_order, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4, 5), }";
        let (dtype, fortran, shape) = parse_npy_header(header).unwrap();
        assert_eq!(dtype, "<f4");
        assert!(!fortran);
        assert_eq!(shape, vec![3, 4, 5]);
    }

    #[test]
    fn test_is_npy() {
        assert!(is_npy_file(Path::new("test.npy")));
        assert!(is_npy_file(Path::new("test.NPY")));
        assert!(!is_npy_file(Path::new("test.tif")));
    }

    #[test]
    fn test_npy_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_dw_roundtrip.npy");

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = FimImage::from_slice(4, 3, 2, &data).unwrap();

        npy_write(&path, &img).unwrap();
        let img2 = npy_read(&path).unwrap();

        assert_eq!(img.dims(), img2.dims());
        for (a, b) in img.as_slice().iter().zip(img2.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(&path).ok();
    }
}
