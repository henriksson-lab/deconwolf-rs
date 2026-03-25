//! Align 3D point clouds between two images using KDE-based optimization.
//!
//! Finds the optimal rigid translation (dx, dy, dz) that aligns one set of
//! detected dots onto another, using kernel density estimation on a KD-tree.

use std::path::Path;

use crate::core::kdtree::KdTree;
use crate::core::{DwError, FTab};

/// Align two 3D dot sets by finding the translation that maximizes KDE overlap.
///
/// # Arguments
/// * `dots1_path` - Path to TSV file for the first dot set (columns: x, y, z, intensity).
/// * `dots2_path` - Path to TSV file for the second dot set (columns: x, y, z, intensity).
/// * `sigma` - Gaussian kernel bandwidth for KDE scoring.
/// * `capture_distance` - Search range in each dimension: `[-capture_distance, +capture_distance]`.
/// * `max_points` - Maximum number of dots to use from each set (sorted by intensity).
/// * `output_path` - Path to write the result TSV (columns: dx, dy, dz, score).
pub fn run_align_dots(
    dots1_path: &Path,
    dots2_path: &Path,
    sigma: f64,
    capture_distance: f64,
    max_points: usize,
    output_path: &Path,
) -> Result<(), DwError> {
    // Load dot files
    let mut dots1 = FTab::from_tsv(dots1_path)?;
    let mut dots2 = FTab::from_tsv(dots2_path)?;

    // Resolve column indices (x, y, z, intensity)
    let col_x = 0;
    let col_y = 1;
    let col_z = 2;
    let col_int = 3;

    if dots1.ncol() < 4 || dots2.ncol() < 4 {
        return Err(DwError::Config(
            "Dot TSV files must have at least 4 columns (x, y, z, intensity)".into(),
        ));
    }

    // Sort by intensity (descending) and keep top max_points
    dots1.sort_by_col(col_int, true);
    dots2.sort_by_col(col_int, true);
    dots1.head(max_points);
    dots2.head(max_points);

    println!(
        "Using {} dots from set 1, {} dots from set 2",
        dots1.nrow(),
        dots2.nrow()
    );

    // Extract dots2 as 3D points and build KD-tree
    let points2: Vec<[f64; 3]> = (0..dots2.nrow())
        .map(|r| {
            [
                dots2.get(r, col_x) as f64,
                dots2.get(r, col_y) as f64,
                dots2.get(r, col_z) as f64,
            ]
        })
        .collect();
    let tree = KdTree::new(&points2, 16);

    // Extract dots1 as 3D points
    let points1: Vec<[f64; 3]> = (0..dots1.nrow())
        .map(|r| {
            [
                dots1.get(r, col_x) as f64,
                dots1.get(r, col_y) as f64,
                dots1.get(r, col_z) as f64,
            ]
        })
        .collect();

    // Coarse grid search: step = sigma / 2
    let coarse_step = sigma / 2.0;
    let (best_dx, best_dy, best_dz, _best_score) =
        grid_search(&points1, &tree, sigma, capture_distance, coarse_step);

    println!(
        "Coarse alignment: dx={:.3}, dy={:.3}, dz={:.3}, score={:.6}",
        best_dx, best_dy, best_dz, _best_score
    );

    // Fine grid search around coarse result: step = sigma / 8
    let fine_step = sigma / 8.0;
    let fine_range = coarse_step; // search +/- one coarse step around best
    let (refined_dx, refined_dy, refined_dz, refined_score) = grid_search_around(
        &points1,
        &tree,
        sigma,
        best_dx,
        best_dy,
        best_dz,
        fine_range,
        fine_step,
    );

    println!(
        "Refined alignment: dx={:.3}, dy={:.3}, dz={:.3}, score={:.6}",
        refined_dx, refined_dy, refined_dz, refined_score
    );

    // Write result
    let mut result = FTab::new(4);
    result = result.with_colnames(&["dx", "dy", "dz", "score"]);
    result.insert_row(&[
        refined_dx as f32,
        refined_dy as f32,
        refined_dz as f32,
        refined_score as f32,
    ]);
    result.write_tsv(output_path)?;

    println!("Result saved to {}", output_path.display());

    Ok(())
}

/// Compute the KDE score for a given shift: sum of kde(dot1 + shift) for all dots1.
fn kde_score(
    points1: &[[f64; 3]],
    tree: &KdTree,
    sigma: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) -> f64 {
    let cutoff = 3.0;
    points1
        .iter()
        .map(|p| {
            let shifted = [p[0] + dx, p[1] + dy, p[2] + dz];
            tree.kde(&shifted, sigma, cutoff)
        })
        .sum()
}

/// Grid search over [-range, +range] in each dimension.
/// Returns (best_dx, best_dy, best_dz, best_score).
fn grid_search(
    points1: &[[f64; 3]],
    tree: &KdTree,
    sigma: f64,
    range: f64,
    step: f64,
) -> (f64, f64, f64, f64) {
    grid_search_around(points1, tree, sigma, 0.0, 0.0, 0.0, range, step)
}

/// Grid search around a center point (cx, cy, cz) +/- range in each dimension.
/// Returns (best_dx, best_dy, best_dz, best_score).
fn grid_search_around(
    points1: &[[f64; 3]],
    tree: &KdTree,
    sigma: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    range: f64,
    step: f64,
) -> (f64, f64, f64, f64) {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_dx = cx;
    let mut best_dy = cy;
    let mut best_dz = cz;

    let n_steps = (range / step).ceil() as i64;

    for ix in -n_steps..=n_steps {
        let dx = cx + ix as f64 * step;
        for iy in -n_steps..=n_steps {
            let dy = cy + iy as f64 * step;
            for iz in -n_steps..=n_steps {
                let dz = cz + iz as f64 * step;
                let score = kde_score(points1, tree, sigma, dx, dy, dz);
                if score > best_score {
                    best_score = score;
                    best_dx = dx;
                    best_dy = dy;
                    best_dz = dz;
                }
            }
        }
    }

    (best_dx, best_dy, best_dz, best_score)
}
