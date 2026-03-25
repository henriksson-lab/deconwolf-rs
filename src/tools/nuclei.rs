//! Nuclei pixel classification using random forest on multiscale image features.

use std::path::Path;

use crate::core::tiff_io::{tiff_read, tiff_write_f32};
use crate::core::trafo::{RandomForest, TrafoSettings};
use crate::core::{DwError, FTab, FimImage};

/// Default set of Gaussian sigmas for multiscale feature extraction.
pub fn default_sigmas() -> Vec<f32> {
    vec![1.0, 2.0, 4.0]
}

/// Extract multiscale image features from a 2D image (P=1).
///
/// For each sigma, computes three features per pixel:
///   1. Gaussian-smoothed value
///   2. Gradient magnitude
///   3. Laplacian of Gaussian (LoG) response
///
/// Returns an FTab with `nrow = M * N` and `ncol = 3 * sigmas.len()`.
fn extract_features_2d(image: &FimImage, sigmas: &[f32]) -> FTab {
    let m = image.m();
    let n = image.n();
    let n_pixels = m * n;
    let n_features = 3 * sigmas.len();

    let mut data = vec![0.0f32; n_pixels * n_features];

    for (si, &sigma) in sigmas.iter().enumerate() {
        // Feature 1: Gaussian smoothed
        let mut smoothed = image.clone();
        smoothed.gsmooth(sigma);

        // Feature 2: Gradient magnitude
        let grad = image.gradient_magnitude(sigma);

        // Feature 3: Laplacian of Gaussian
        let log = image.log_filter(sigma, sigma);

        let smoothed_s = smoothed.as_slice();
        let grad_s = grad.as_slice();
        let log_s = log.as_slice();

        let col_base = si * 3;
        for px in 0..n_pixels {
            data[px * n_features + col_base] = smoothed_s[px];
            data[px * n_features + col_base + 1] = grad_s[px];
            data[px * n_features + col_base + 2] = log_s[px];
        }
    }

    // Build column names
    let mut colnames: Vec<String> = Vec::with_capacity(n_features);
    for sigma in sigmas {
        colnames.push(format!("smooth_{}", sigma));
        colnames.push(format!("grad_{}", sigma));
        colnames.push(format!("log_{}", sigma));
    }
    let colname_refs: Vec<&str> = colnames.iter().map(|s| s.as_str()).collect();

    FTab::from_data(n_pixels, n_features, data)
        .expect("feature table dimensions must match")
        .with_colnames(&colname_refs)
}

/// Train a random forest model for nuclei pixel classification.
///
/// # Arguments
/// * `image_path` - Path to the input image (3D TIFF, will be max-projected to 2D).
/// * `annotation_path` - Path to a 2D TIFF with class labels (0 = unlabelled/background,
///   1, 2, ... = class labels). Only pixels with annotation > 0 are used for training.
/// * `model_output` - Path to write the trained random forest model.
/// * `n_trees` - Number of trees in the random forest.
/// * `sigmas` - Gaussian sigmas for multiscale feature extraction.
pub fn run_nuclei_fit(
    image_path: &Path,
    annotation_path: &Path,
    model_output: &Path,
    n_trees: usize,
    sigmas: &[f32],
) -> Result<(), DwError> {
    // Load image and max-project to 2D
    let (image, _meta) = tiff_read(image_path)?;
    let proj = image.max_projection();

    // Extract features
    let features = extract_features_2d(&proj, sigmas);

    // Load annotation image as class labels
    let (annot_img, _) = tiff_read(annotation_path)?;
    let annot_slice = annot_img.as_slice();

    let m = proj.m();
    let n = proj.n();
    let n_pixels = m * n;

    if annot_slice.len() != n_pixels {
        return Err(DwError::InvalidDimensions(format!(
            "Annotation image has {} pixels, but projected image has {}",
            annot_slice.len(),
            n_pixels
        )));
    }

    // Build training set: only pixels where annotation > 0
    let n_features = features.ncol();
    let mut train_features: Vec<f64> = Vec::new();
    let mut train_labels: Vec<u32> = Vec::new();

    for px in 0..n_pixels {
        let label = annot_slice[px].round() as u32;
        if label > 0 {
            // Convert feature row from f32 to f64
            for col in 0..n_features {
                train_features.push(features.get(px, col) as f64);
            }
            // Labels are 0-based internally, so subtract 1 from annotation labels
            train_labels.push(label - 1);
        }
    }

    let n_samples = train_labels.len();
    if n_samples == 0 {
        return Err(DwError::Config(
            "No annotated pixels found (all annotation values are 0)".into(),
        ));
    }

    println!(
        "Training random forest: {} samples, {} features, {} trees",
        n_samples, n_features, n_trees
    );

    // Train random forest
    let settings = TrafoSettings {
        n_trees,
        ..TrafoSettings::default()
    };
    let forest = RandomForest::fit(
        &train_features,
        &train_labels,
        n_samples,
        n_features,
        &settings,
    )?;

    // Save model
    forest.save(model_output)?;
    println!("Model saved to {}", model_output.display());

    Ok(())
}

/// Classify image pixels using a trained random forest model.
///
/// # Arguments
/// * `image_path` - Path to the input image (3D TIFF, will be max-projected to 2D).
/// * `model_path` - Path to the trained random forest model.
/// * `output_path` - Path to write the classified image as TIFF.
/// * `sigmas` - Gaussian sigmas for multiscale feature extraction (must match training).
pub fn run_nuclei_classify(
    image_path: &Path,
    model_path: &Path,
    output_path: &Path,
    sigmas: &[f32],
) -> Result<(), DwError> {
    // Load image and max-project to 2D
    let (image, _meta) = tiff_read(image_path)?;
    let proj = image.max_projection();

    // Extract features
    let features = extract_features_2d(&proj, sigmas);

    let m = proj.m();
    let n = proj.n();
    let n_pixels = m * n;
    let n_features = features.ncol();

    // Convert features to f64 for prediction
    let mut features_f64 = vec![0.0f64; n_pixels * n_features];
    for px in 0..n_pixels {
        for col in 0..n_features {
            features_f64[px * n_features + col] = features.get(px, col) as f64;
        }
    }

    // Load model
    let forest = RandomForest::load(model_path)?;

    println!(
        "Classifying {} pixels with {} features",
        n_pixels, n_features
    );

    // Predict class for each pixel
    let predictions = forest.predict(&features_f64, n_pixels);

    // Build output image: class labels stored as f32 (add 1 to restore 1-based labels)
    let out_data: Vec<f32> = predictions.iter().map(|&c| (c + 1) as f32).collect();
    let out_image = FimImage::from_vec(m, n, 1, out_data)?;

    // Write classified image
    tiff_write_f32(output_path, &out_image, None)?;
    println!("Classified image saved to {}", output_path.display());

    Ok(())
}
