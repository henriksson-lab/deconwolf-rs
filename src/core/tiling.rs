use super::error::Result;
use super::image::FimImage;

/// A single tile within a tiling scheme.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile position in the original image [m0, m1, n0, n1, p0, p1].
    pub pos: [usize; 6],
    /// Extended position including overlap [m0, m1, n0, n1, p0, p1].
    pub xpos: [usize; 6],
    /// Tile size (m1-m0, n1-n0, p1-p0).
    pub size: [usize; 3],
    /// Extended tile size.
    pub xsize: [usize; 3],
}

/// Tiling scheme for processing large images in overlapping tiles.
///
/// Only tiles in M and N dimensions; P (depth) is kept whole.
#[derive(Debug)]
pub struct Tiling {
    /// Original image dimensions.
    pub m: usize,
    pub n: usize,
    pub p: usize,
    /// Maximum tile size per dimension.
    pub max_size: usize,
    /// Overlap in pixels.
    pub overlap: usize,
    /// The tiles.
    pub tiles: Vec<Tile>,
}

impl Tiling {
    /// Create a tiling scheme for an MxNxP image.
    ///
    /// Tiles only M and N; P is kept whole.
    pub fn new(m: usize, n: usize, p: usize, max_size: usize, overlap: usize) -> Self {
        let m_divs = get_divisions(m, max_size);
        let n_divs = get_divisions(n, max_size);

        let mut tiles = Vec::with_capacity(m_divs.len() * n_divs.len());

        for &(n0, n1) in &n_divs {
            for &(m0, m1) in &m_divs {
                // Extended position with overlap (clamped to image bounds)
                let xm0 = m0.saturating_sub(overlap);
                let xm1 = (m1 + overlap).min(m);
                let xn0 = n0.saturating_sub(overlap);
                let xn1 = (n1 + overlap).min(n);

                tiles.push(Tile {
                    pos: [m0, m1, n0, n1, 0, p],
                    xpos: [xm0, xm1, xn0, xn1, 0, p],
                    size: [m1 - m0, n1 - n0, p],
                    xsize: [xm1 - xm0, xn1 - xn0, p],
                });
            }
        }

        Tiling {
            m,
            n,
            p,
            max_size,
            overlap,
            tiles,
        }
    }

    /// Number of tiles.
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Extract a tile (extended region) from an image.
    pub fn extract_tile(&self, img: &FimImage, tile_idx: usize) -> Result<FimImage> {
        let tile = &self.tiles[tile_idx];
        img.get_cuboid(
            tile.xpos[0],
            tile.xpos[1],
            tile.xpos[2],
            tile.xpos[3],
            tile.xpos[4],
            tile.xpos[5],
        )
    }

    /// Blend a processed tile back into the output image using triangular weights.
    pub fn blend_tile(&self, output: &mut FimImage, weights: &mut FimImage, tile_idx: usize, tile_data: &FimImage) {
        let tile = &self.tiles[tile_idx];
        let [xm0, xm1, xn0, xn1, _xp0, _xp1] = tile.xpos;
        let [m0, m1, n0, n1, _, _] = tile.pos;
        let xm = xm1 - xm0;
        let xn = xn1 - xn0;

        for pp in 0..self.p {
            for nn in 0..xn {
                for mm in 0..xm {
                    let global_m = xm0 + mm;
                    let global_n = xn0 + nn;

                    // Weight: triangular ramp, 1.0 inside core, ramp in overlap
                    let wm = weight_1d(mm, xm, global_m.saturating_sub(m0), m1 - m0);
                    let wn = weight_1d(nn, xn, global_n.saturating_sub(n0), n1 - n0);
                    let w = wm.min(wn);

                    let val = tile_data.get(mm, nn, pp);
                    let cur = output.get(global_m, global_n, pp);
                    let cur_w = weights.get(global_m, global_n, pp);
                    output.set(global_m, global_n, pp, cur + val * w);
                    weights.set(global_m, global_n, pp, cur_w + w);
                }
            }
        }
    }

    /// Finalize blending by dividing by accumulated weights.
    pub fn finalize(output: &mut FimImage, weights: &FimImage) {
        let out = output.as_slice_mut();
        let w = weights.as_slice();
        for i in 0..out.len() {
            if w[i] > 0.0 {
                out[i] /= w[i];
            }
        }
    }
}

/// Compute a 1D triangular weight.
fn weight_1d(local_pos: usize, _extended_size: usize, offset_from_core: usize, core_size: usize) -> f32 {
    if core_size == 0 {
        return 1.0;
    }
    // Inside core region: weight = 1.0
    // In overlap region: linear ramp from 0 to 1
    let _ = offset_from_core;
    let center = _extended_size as f32 / 2.0;
    let half_core = core_size as f32 / 2.0;
    let dist_from_center = (local_pos as f32 - center).abs();
    if dist_from_center <= half_core {
        1.0
    } else {
        let overlap_extent = ((_extended_size as f32 - core_size as f32) / 2.0).max(1.0);
        ((overlap_extent - (dist_from_center - half_core)) / overlap_extent).max(0.0)
    }
}

/// Compute how to divide a dimension of size `total` into chunks of at most `max_size`.
fn get_divisions(total: usize, max_size: usize) -> Vec<(usize, usize)> {
    if total <= max_size {
        return vec![(0, total)];
    }
    let n_tiles = (total + max_size - 1) / max_size;
    let tile_size = total / n_tiles;
    let remainder = total % n_tiles;

    let mut divs = Vec::with_capacity(n_tiles);
    let mut pos = 0;
    for i in 0..n_tiles {
        let size = tile_size + if i < remainder { 1 } else { 0 };
        divs.push((pos, pos + size));
        pos += size;
    }
    divs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divisions() {
        let divs = get_divisions(100, 40);
        assert_eq!(divs.len(), 3);
        assert_eq!(divs[0].0, 0);
        assert_eq!(divs.last().unwrap().1, 100);
    }

    #[test]
    fn test_single_tile() {
        let divs = get_divisions(30, 50);
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0], (0, 30));
    }

    #[test]
    fn test_tiling_creation() {
        let tiling = Tiling::new(200, 200, 10, 100, 20);
        assert_eq!(tiling.num_tiles(), 4); // 2x2
        for tile in &tiling.tiles {
            assert_eq!(tile.size[2], 10); // P is always full
        }
    }
}
