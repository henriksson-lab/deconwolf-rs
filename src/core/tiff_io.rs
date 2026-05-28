use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::colortype::{Gray16, Gray32Float};
use tiff::encoder::TiffEncoder;
use tiff::tags::Tag;

use super::error::{DwError, Result};
use super::image::FimImage;

/// TIFF metadata (pixel size, software info, ImageJ metadata).
#[derive(Debug, Clone, Default)]
pub struct TiffMeta {
    pub x_resolution: Option<f64>,
    pub y_resolution: Option<f64>,
    pub z_spacing: Option<f64>,
    pub resolution_unit: Option<u16>,
    pub image_description: Option<String>,
    pub software: Option<String>,
}

impl TiffMeta {
    /// Create ImageJ-compatible image description for a 3D stack.
    pub fn imagej_description(p: usize, z_spacing: Option<f64>) -> String {
        let mut desc = format!("ImageJ=1.52r\nimages={}\nslices={}\n", p, p);
        if let Some(zs) = z_spacing {
            desc.push_str(&format!("unit=nm\nspacing={}\n", zs));
        }
        desc.push_str("loop=false\n");
        desc
    }

    /// Set pixel size in the metadata.
    pub fn set_pixel_size(&mut self, xres: f64, yres: f64, zres: f64) {
        self.x_resolution = Some(xres);
        self.y_resolution = Some(yres);
        self.z_spacing = Some(zres);
    }
}

/// Read a 3D TIFF stack as a FimImage (float32).
/// Supports uint8, uint16, and float32 input formats.
pub fn tiff_read(path: &Path) -> Result<(FimImage, TiffMeta)> {
    tiff_read_with_meta(path)
}

fn tiff_read_with_meta(path: &Path) -> Result<(FimImage, TiffMeta)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)?;

    // Read metadata from first directory
    let mut meta = TiffMeta::default();
    if let Ok(val) = decoder.get_tag_ascii_string(Tag::ImageDescription) {
        meta.image_description = Some(val);
    }
    if let Ok(val) = decoder.get_tag_ascii_string(Tag::Software) {
        meta.software = Some(val);
    }
    // Try to read resolution
    if let Ok(val) = decoder.get_tag_f64(Tag::XResolution) {
        if val > 0.0 {
            meta.x_resolution = Some(1.0 / val);
        }
    }
    if let Ok(val) = decoder.get_tag_f64(Tag::YResolution) {
        if val > 0.0 {
            meta.y_resolution = Some(1.0 / val);
        }
    }
    // Parse z-spacing from ImageJ metadata if present
    if let Some(ref desc) = meta.image_description {
        for line in desc.lines() {
            if let Some(val) = line.strip_prefix("spacing=") {
                if let Ok(z) = val.trim().parse::<f64>() {
                    if z.is_finite() && z > 0.0 {
                        meta.z_spacing = Some(z);
                    }
                }
            }
        }
    }

    let (width, height) = decoder.dimensions()?;
    let m = width as usize;
    let n = height as usize;
    let plane_len = checked_plane_len(m, n)?;

    // Read all planes
    let mut planes: Vec<Vec<f32>> = Vec::new();

    loop {
        let (plane_width, plane_height) = decoder.dimensions()?;
        if plane_width != width || plane_height != height {
            return Err(DwError::InvalidDimensions(format!(
                "TIFF plane has dimensions {}x{}, expected {}x{}",
                plane_width, plane_height, width, height,
            )));
        }
        let result = decoder.read_image()?;
        let plane = decode_to_f32(result, plane_len)?;
        planes.push(plane);

        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    let p = planes.len();
    let mut data = Vec::with_capacity(checked_stack_len(m, n, p)?);
    for plane in planes {
        data.extend_from_slice(&plane);
    }

    let img = FimImage::from_vec(m, n, p, data)?;
    Ok((img, meta))
}

fn decode_to_f32(result: DecodingResult, expected_len: usize) -> Result<Vec<f32>> {
    match result {
        DecodingResult::U8(buf) => {
            if buf.len() != expected_len {
                return Err(DwError::InvalidDimensions(format!(
                    "Expected {} pixels, got {}",
                    expected_len,
                    buf.len()
                )));
            }
            Ok(buf.iter().map(|&v| v as f32).collect())
        }
        DecodingResult::U16(buf) => {
            if buf.len() != expected_len {
                return Err(DwError::InvalidDimensions(format!(
                    "Expected {} pixels, got {}",
                    expected_len,
                    buf.len()
                )));
            }
            Ok(buf.iter().map(|&v| v as f32).collect())
        }
        DecodingResult::F32(buf) => {
            if buf.len() != expected_len {
                return Err(DwError::InvalidDimensions(format!(
                    "Expected {} pixels, got {}",
                    expected_len,
                    buf.len()
                )));
            }
            Ok(buf)
        }
        DecodingResult::U32(buf) => {
            if buf.len() != expected_len {
                return Err(DwError::InvalidDimensions(format!(
                    "Expected {} pixels, got {}",
                    expected_len,
                    buf.len()
                )));
            }
            Ok(buf.iter().map(|&v| v as f32).collect())
        }
        DecodingResult::F64(buf) => {
            if buf.len() != expected_len {
                return Err(DwError::InvalidDimensions(format!(
                    "Expected {} pixels, got {}",
                    expected_len,
                    buf.len()
                )));
            }
            Ok(buf.iter().map(|&v| v as f32).collect())
        }
        _ => Err(DwError::UnsupportedFormat(
            "Unsupported TIFF sample format".into(),
        )),
    }
}

/// Write a FimImage as a 16-bit TIFF, auto-scaling to [0, 65535].
/// Returns the scaling factor used.
pub fn tiff_write_u16(
    path: &Path,
    img: &FimImage,
    meta: Option<&TiffMeta>,
    scaling: Option<f32>,
) -> Result<f32> {
    let (m, n, p) = img.dims();
    let (width, height) = tiff_output_dimensions(m, n)?;
    let plane_len = checked_plane_len(m, n)?;
    let max_val = img.max();
    let scale = scaling.unwrap_or_else(|| {
        if max_val > 0.0 {
            65535.0 / max_val
        } else {
            1.0
        }
    });
    if !scale.is_finite() || scale < 0.0 {
        return Err(DwError::Config(
            "TIFF u16 scaling must be a non-negative finite value".into(),
        ));
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;

    for pp in 0..p {
        let mut plane = vec![0u16; plane_len];
        for nn in 0..n {
            for mm in 0..m {
                let v = (img.get(mm, nn, pp) * scale).round();
                plane[nn * m + mm] = v.clamp(0.0, 65535.0) as u16;
            }
        }

        let mut dir = encoder.new_image::<Gray16>(width, height)?;

        // Write metadata on first plane
        if pp == 0 {
            if let Some(meta) = meta {
                if let Some(ref desc) = meta.image_description {
                    dir.encoder()
                        .write_tag(Tag::ImageDescription, desc.as_str())?;
                } else if p > 1 {
                    let desc = TiffMeta::imagej_description(p, meta.z_spacing);
                    dir.encoder()
                        .write_tag(Tag::ImageDescription, desc.as_str())?;
                }
                if let Some(ref sw) = meta.software {
                    dir.encoder().write_tag(Tag::Software, sw.as_str())?;
                }
            } else if p > 1 {
                let desc = TiffMeta::imagej_description(p, None);
                dir.encoder()
                    .write_tag(Tag::ImageDescription, desc.as_str())?;
            }
        }

        dir.write_data(&plane)?;
    }

    Ok(scale)
}

/// Write a FimImage as a 32-bit float TIFF.
pub fn tiff_write_f32(path: &Path, img: &FimImage, meta: Option<&TiffMeta>) -> Result<()> {
    let (m, n, p) = img.dims();
    let (width, height) = tiff_output_dimensions(m, n)?;
    let plane_len = checked_plane_len(m, n)?;
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;

    for pp in 0..p {
        let mut plane = vec![0.0f32; plane_len];
        for nn in 0..n {
            for mm in 0..m {
                plane[nn * m + mm] = img.get(mm, nn, pp);
            }
        }

        let mut dir = encoder.new_image::<Gray32Float>(width, height)?;

        if pp == 0 {
            if let Some(meta) = meta {
                if let Some(ref desc) = meta.image_description {
                    dir.encoder()
                        .write_tag(Tag::ImageDescription, desc.as_str())?;
                } else if p > 1 {
                    let desc = TiffMeta::imagej_description(p, meta.z_spacing);
                    dir.encoder()
                        .write_tag(Tag::ImageDescription, desc.as_str())?;
                }
                if let Some(ref sw) = meta.software {
                    dir.encoder().write_tag(Tag::Software, sw.as_str())?;
                }
            } else if p > 1 {
                let desc = TiffMeta::imagej_description(p, None);
                dir.encoder()
                    .write_tag(Tag::ImageDescription, desc.as_str())?;
            }
        }

        dir.write_data(&plane)?;
    }

    Ok(())
}

fn tiff_output_dimensions(m: usize, n: usize) -> Result<(u32, u32)> {
    let width = u32::try_from(m)
        .map_err(|_| DwError::InvalidDimensions(format!("TIFF width {} exceeds u32::MAX", m)))?;
    let height = u32::try_from(n)
        .map_err(|_| DwError::InvalidDimensions(format!("TIFF height {} exceeds u32::MAX", n)))?;
    Ok((width, height))
}

fn checked_plane_len(m: usize, n: usize) -> Result<usize> {
    m.checked_mul(n)
        .ok_or_else(|| DwError::InvalidDimensions("TIFF plane size overflow".into()))
}

fn checked_stack_len(m: usize, n: usize, p: usize) -> Result<usize> {
    checked_plane_len(m, n)?
        .checked_mul(p)
        .ok_or_else(|| DwError::InvalidDimensions("TIFF stack size overflow".into()))
}

/// Get image dimensions from a TIFF file without loading data.
pub fn tiff_get_size(path: &Path) -> Result<(usize, usize, usize)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)?;

    let (width, height) = decoder.dimensions()?;
    let m = width as usize;
    let n = height as usize;

    let mut p: usize = 1;
    while decoder.more_images() {
        decoder.next_image()?;
        let (plane_width, plane_height) = decoder.dimensions()?;
        if plane_width != width || plane_height != height {
            return Err(DwError::InvalidDimensions(format!(
                "TIFF plane has dimensions {}x{}, expected {}x{}",
                plane_width, plane_height, width, height,
            )));
        }
        p = p
            .checked_add(1)
            .ok_or_else(|| DwError::InvalidDimensions("TIFF plane count overflow".into()))?;
    }

    Ok((m, n, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiff_output_dimensions_reject_oversized_images() {
        assert_eq!(tiff_output_dimensions(10, 20).unwrap(), (10, 20));
        assert!(tiff_output_dimensions(u32::MAX as usize + 1, 20).is_err());
        assert!(tiff_output_dimensions(10, u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn tiff_size_helpers_reject_overflow() {
        assert_eq!(checked_plane_len(4, 5).unwrap(), 20);
        assert_eq!(checked_stack_len(4, 5, 3).unwrap(), 60);
        assert!(checked_plane_len(usize::MAX, 2).is_err());
        assert!(checked_stack_len(usize::MAX, 2, 1).is_err());
        assert!(checked_stack_len(usize::MAX / 2 + 1, 2, 2).is_err());
    }

    #[test]
    fn tiff_write_u16_rejects_invalid_scaling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tif");
        let img = FimImage::zeros(1, 1, 1);
        assert!(tiff_write_u16(&path, &img, None, Some(f32::NAN)).is_err());
        assert!(tiff_write_u16(&path, &img, None, Some(-1.0)).is_err());
    }

    #[test]
    fn tiff_read_rejects_mixed_plane_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.tif");
        let file = File::create(&path).unwrap();
        let writer = BufWriter::new(file);
        let mut encoder = TiffEncoder::new(writer).unwrap();

        encoder
            .new_image::<Gray16>(2, 2)
            .unwrap()
            .write_data(&[1u16; 4])
            .unwrap();
        encoder
            .new_image::<Gray16>(3, 2)
            .unwrap()
            .write_data(&[1u16; 6])
            .unwrap();

        assert!(tiff_read(&path).is_err());
        assert!(tiff_get_size(&path).is_err());
    }
}
