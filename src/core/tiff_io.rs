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
        let mut desc = format!(
            "ImageJ=1.52r\nimages={}\nslices={}\n",
            p, p
        );
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
                    meta.z_spacing = Some(z);
                }
            }
        }
    }

    let (width, height) = decoder.dimensions()?;
    let m = width as usize;
    let n = height as usize;

    // Read all planes
    let mut planes: Vec<Vec<f32>> = Vec::new();

    loop {
        let result = decoder.read_image()?;
        let plane = decode_to_f32(result, m * n)?;
        planes.push(plane);

        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    let p = planes.len();
    let mut data = Vec::with_capacity(m * n * p);
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
    let max_val = img.max();
    let scale = scaling.unwrap_or_else(|| {
        if max_val > 0.0 {
            65535.0 / max_val
        } else {
            1.0
        }
    });

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;

    let (m, n, p) = img.dims();
    for pp in 0..p {
        let mut plane = vec![0u16; m * n];
        for nn in 0..n {
            for mm in 0..m {
                let v = (img.get(mm, nn, pp) * scale).round();
                plane[nn * m + mm] = v.clamp(0.0, 65535.0) as u16;
            }
        }

        let mut dir = encoder.new_image::<Gray16>(m as u32, n as u32)?;

        // Write metadata on first plane
        if pp == 0 {
            if let Some(meta) = meta {
                if let Some(ref desc) = meta.image_description {
                    dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
                } else if p > 1 {
                    let desc = TiffMeta::imagej_description(p, meta.z_spacing);
                    dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
                }
                if let Some(ref sw) = meta.software {
                    dir.encoder().write_tag(Tag::Software, sw.as_str())?;
                }
            } else if p > 1 {
                let desc = TiffMeta::imagej_description(p, None);
                dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
            }
        }

        dir.write_data(&plane)?;
    }

    Ok(scale)
}

/// Write a FimImage as a 32-bit float TIFF.
pub fn tiff_write_f32(
    path: &Path,
    img: &FimImage,
    meta: Option<&TiffMeta>,
) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;

    let (m, n, p) = img.dims();
    for pp in 0..p {
        let mut plane = vec![0.0f32; m * n];
        for nn in 0..n {
            for mm in 0..m {
                plane[nn * m + mm] = img.get(mm, nn, pp);
            }
        }

        let mut dir = encoder.new_image::<Gray32Float>(m as u32, n as u32)?;

        if pp == 0 {
            if let Some(meta) = meta {
                if let Some(ref desc) = meta.image_description {
                    dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
                } else if p > 1 {
                    let desc = TiffMeta::imagej_description(p, meta.z_spacing);
                    dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
                }
                if let Some(ref sw) = meta.software {
                    dir.encoder().write_tag(Tag::Software, sw.as_str())?;
                }
            } else if p > 1 {
                let desc = TiffMeta::imagej_description(p, None);
                dir.encoder().write_tag(Tag::ImageDescription, desc.as_str())?;
            }
        }

        dir.write_data(&plane)?;
    }

    Ok(())
}

/// Get image dimensions from a TIFF file without loading data.
pub fn tiff_get_size(path: &Path) -> Result<(usize, usize, usize)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)?;

    let (width, height) = decoder.dimensions()?;
    let m = width as usize;
    let n = height as usize;

    let mut p = 1;
    while decoder.more_images() {
        decoder.next_image()?;
        p += 1;
    }

    Ok((m, n, p))
}
