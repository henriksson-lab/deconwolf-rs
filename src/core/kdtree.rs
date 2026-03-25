//! 3D K-d tree for spatial nearest-neighbor queries.
//!
//! This is a Rust port of the C kdtree library used for dot alignment
//! and image shifting operations. The tree is hard-coded for 3D data
//! (set unused dimensions to 0 for 1D/2D use).

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// 3D K-d tree for spatial queries.
#[allow(dead_code)]
pub struct KdTree {
    nodes: Vec<KdNode>,
    /// Stored points (may be reordered during construction).
    points: Vec<[f64; 3]>,
    /// Original indices of points (tracks reordering).
    indices: Vec<usize>,
    max_leaf_size: usize,
}

struct KdNode {
    /// Bounding box: [min_x, max_x, min_y, max_y, min_z, max_z]
    bbx: [f64; 6],
    /// Start index into `points`/`indices` arrays.
    data_start: usize,
    /// Number of points owned by this node.
    n_points: usize,
    /// Split dimension (0, 1, 2) for internal nodes; 3 for leaf.
    split_dim: u8,
    /// Pivot value along `split_dim`.
    pivot: f64,
    /// Left child index (Eytzinger: 2*i+1).
    left: usize,
    /// Right child index (Eytzinger: 2*i+2).
    right: usize,
}

/// Entry in the max-heap used during k-NN search.
/// We want a max-heap by distance so we can evict the farthest candidate.
#[derive(Clone, Copy)]
struct HeapEntry {
    dist_sq: f64,
    index: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance (largest on top).
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
    }
}

impl KdNode {
    fn is_leaf(&self) -> bool {
        self.split_dim == 3
    }

    fn new_empty() -> Self {
        KdNode {
            bbx: [0.0; 6],
            data_start: 0,
            n_points: 0,
            split_dim: 3,
            pivot: 0.0,
            left: 0,
            right: 0,
        }
    }
}

impl KdTree {
    /// Build a new 3D K-d tree from a slice of points.
    ///
    /// `bin_size` controls the maximum number of points per leaf node.
    /// Values of 4-32 are typically optimal regardless of data size.
    pub fn new(points: &[[f64; 3]], bin_size: usize) -> Self {
        let bin_size = bin_size.max(1);
        let n = points.len();
        assert!(n > 0, "KdTree requires at least one point");

        let mut tree_points: Vec<[f64; 3]> = points.to_vec();
        let mut indices: Vec<usize> = (0..n).collect();

        // Allocate nodes for a complete binary tree.
        // Estimate leaf count assuming ~50% fill, round up to power of two.
        let n_leafs_est = (2.0 * n as f64 / bin_size as f64).max(1.0);
        let n_leafs = (2.0_f64).powf(n_leafs_est.log2().ceil()) as usize;
        let n_nodes = (n_leafs * 2 - 1).max(3);

        let mut nodes: Vec<KdNode> = (0..n_nodes).map(|_| KdNode::new_empty()).collect();

        // Set up root node.
        let bbx = bounding_box(&tree_points);
        nodes[0].bbx = bbx;
        nodes[0].data_start = 0;
        nodes[0].n_points = n;

        // Recursively split.
        split_node(
            &mut nodes,
            &mut tree_points,
            &mut indices,
            0,
            bin_size,
        );

        KdTree {
            nodes,
            points: tree_points,
            indices,
            max_leaf_size: bin_size,
        }
    }

    /// Find the k nearest neighbors of `query`.
    ///
    /// Returns a `Vec` of `(original_index, distance)` pairs sorted by
    /// distance (closest first). Distance is Euclidean (not squared).
    pub fn query_knn(&self, query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        assert!(k > 0 && k <= self.points.len());

        // Max-heap of size k: we keep the k smallest distances seen so far.
        let mut heap = BinaryHeap::with_capacity(k + 1);
        // Seed with a very large sentinel.
        heap.push(HeapEntry {
            dist_sq: f64::MAX,
            index: 0,
        });

        self.knn_search(0, query, k, &mut heap);

        // Extract results, sorted closest-first.
        let mut results: Vec<(usize, f64)> = heap
            .into_sorted_vec()
            .into_iter()
            .filter(|e| e.dist_sq < f64::MAX)
            .map(|e| (e.index, e.dist_sq.sqrt()))
            .collect();
        results.truncate(k);
        results
    }

    /// Find the single closest point to `query`.
    ///
    /// Returns `(original_index, distance)`.
    pub fn query_closest(&self, query: &[f64; 3]) -> (usize, f64) {
        let results = self.query_knn(query, 1);
        results[0]
    }

    /// Find all points within Euclidean `radius` of `query`.
    ///
    /// Returns `(original_index, distance)` pairs (unordered).
    pub fn query_radius(&self, query: &[f64; 3], radius: f64) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        let r2 = radius * radius;
        self.radius_search(0, query, r2, &mut results);
        results
    }

    /// Kernel Density Estimation using a Gaussian kernel.
    ///
    /// Computes `sum of exp(-d^2 / (2 * sigma^2))` for all points within
    /// `cutoff * sigma` of the query point. If `cutoff <= 0`, a default
    /// of 2.5 is used.
    pub fn kde(&self, query: &[f64; 3], sigma: f64, cutoff: f64) -> f64 {
        let r = if cutoff > 0.0 {
            cutoff * sigma
        } else {
            2.5 * sigma
        };
        let r2 = r * r;
        let sigma22 = 2.0 * sigma * sigma;
        self.kde_recursive(0, query, r2, sigma22)
    }

    // --- Internal recursive methods ---

    fn knn_search(
        &self,
        node_id: usize,
        query: &[f64; 3],
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        if node_id >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[node_id];
        if node.n_points == 0 {
            return;
        }

        // Prune: if the bounding box is farther than current k-th best, skip.
        let box_dist_sq = min_dist_sq_to_box(query, &node.bbx);
        let current_max = heap.peek().map_or(f64::MAX, |e| e.dist_sq);
        if heap.len() >= k && box_dist_sq > current_max {
            return;
        }

        if node.is_leaf() {
            // Check all points in this leaf.
            for i in 0..node.n_points {
                let idx = node.data_start + i;
                let d2 = dist_sq(query, &self.points[idx]);
                let orig_idx = self.indices[idx];

                if heap.len() < k {
                    heap.push(HeapEntry {
                        dist_sq: d2,
                        index: orig_idx,
                    });
                } else if d2 < heap.peek().unwrap().dist_sq {
                    // Replace the farthest candidate.
                    heap.pop();
                    heap.push(HeapEntry {
                        dist_sq: d2,
                        index: orig_idx,
                    });
                }
            }
            return;
        }

        // Descend into the child closer to the query first.
        let split_dim = node.split_dim as usize;
        let (first, second) = if query[split_dim] > node.pivot {
            (node.right, node.left)
        } else {
            (node.left, node.right)
        };

        self.knn_search(first, query, k, heap);
        self.knn_search(second, query, k, heap);
    }

    fn radius_search(
        &self,
        node_id: usize,
        query: &[f64; 3],
        r2: f64,
        results: &mut Vec<(usize, f64)>,
    ) {
        if node_id >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[node_id];
        if node.n_points == 0 {
            return;
        }

        // Prune: if bounding box does not overlap the query ball, skip.
        if !bounds_overlap_ball(&node.bbx, query, r2) {
            return;
        }

        if node.is_leaf() {
            for i in 0..node.n_points {
                let idx = node.data_start + i;
                let d2 = dist_sq(query, &self.points[idx]);
                if d2 < r2 {
                    results.push((self.indices[idx], d2.sqrt()));
                }
            }
            return;
        }

        let left = node.left;
        let right = node.right;
        self.radius_search(left, query, r2, results);
        self.radius_search(right, query, r2, results);
    }

    fn kde_recursive(
        &self,
        node_id: usize,
        query: &[f64; 3],
        r2: f64,
        sigma22: f64,
    ) -> f64 {
        if node_id >= self.nodes.len() {
            return 0.0;
        }
        let node = &self.nodes[node_id];
        if node.n_points == 0 {
            return 0.0;
        }

        if !bounds_overlap_ball(&node.bbx, query, r2) {
            return 0.0;
        }

        if node.is_leaf() {
            let mut kde = 0.0;
            for i in 0..node.n_points {
                let idx = node.data_start + i;
                let d2 = dist_sq(query, &self.points[idx]);
                if d2 < r2 {
                    kde += (-d2 / sigma22).exp();
                }
            }
            return kde;
        }

        let left = node.left;
        let right = node.right;
        self.kde_recursive(left, query, r2, sigma22)
            + self.kde_recursive(right, query, r2, sigma22)
    }
}

// --- Free functions ---

/// Squared Euclidean distance between two 3D points.
#[inline]
fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Minimum squared distance from a point to an axis-aligned bounding box.
#[inline]
fn min_dist_sq_to_box(q: &[f64; 3], bbx: &[f64; 6]) -> f64 {
    let mut d2 = 0.0;
    for dim in 0..3 {
        let lo = bbx[2 * dim];
        let hi = bbx[2 * dim + 1];
        if q[dim] < lo {
            let d = lo - q[dim];
            d2 += d * d;
        } else if q[dim] > hi {
            let d = q[dim] - hi;
            d2 += d * d;
        }
    }
    d2
}

/// Check whether a ball (center `q`, radius^2 = `r2`) overlaps a bounding box.
#[inline]
fn bounds_overlap_ball(bbx: &[f64; 6], q: &[f64; 3], r2: f64) -> bool {
    min_dist_sq_to_box(q, bbx) < r2
}

/// Compute the axis-aligned bounding box of a set of points.
fn bounding_box(points: &[[f64; 3]]) -> [f64; 6] {
    let mut bbx = [0.0_f64; 6];
    if points.is_empty() {
        return bbx;
    }
    for dim in 0..3 {
        bbx[2 * dim] = points[0][dim];
        bbx[2 * dim + 1] = points[0][dim];
    }
    for p in points.iter().skip(1) {
        for dim in 0..3 {
            if p[dim] < bbx[2 * dim] {
                bbx[2 * dim] = p[dim];
            }
            if p[dim] > bbx[2 * dim + 1] {
                bbx[2 * dim + 1] = p[dim];
            }
        }
    }
    bbx
}

/// Find the median of `values` using quickselect (modifies the slice).
fn quickselect_median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    quickselect(values, mid)
}

/// Quickselect: find the k-th smallest element. Partially reorders the slice.
fn quickselect(arr: &mut [f64], k: usize) -> f64 {
    let n = arr.len();
    if n <= 1 {
        return arr[0];
    }
    if n == 2 {
        return if k == 0 {
            arr[0].min(arr[1])
        } else {
            arr[0].max(arr[1])
        };
    }

    // Median-of-five pivot selection.
    let pivot = med5(
        arr[0],
        arr[(n - 1) / 4],
        arr[(n - 1) / 2],
        arr[3 * (n - 1) / 4],
        arr[n - 1],
    );

    // Partition: elements <= pivot go left, elements > pivot go right.
    let mut lo: isize = -1;
    let mut hi: isize = n as isize;
    loop {
        lo += 1;
        while (lo as usize) < n && arr[lo as usize] <= pivot {
            lo += 1;
        }
        hi -= 1;
        while hi > 0 && arr[hi as usize] > pivot {
            hi -= 1;
        }
        if lo >= hi {
            break;
        }
        arr.swap(lo as usize, hi as usize);
    }
    let n_low = lo as usize;

    if n_low == 0 || n_low == n {
        // All elements are equal (or pivot selection degenerate).
        return arr[0];
    }

    if k < n_low {
        quickselect(&mut arr[..n_low], k)
    } else {
        quickselect(&mut arr[n_low..], k - n_low)
    }
}

fn med5(a: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
    let f = a.min(b).max(c.min(d));
    let g = a.max(b).min(c.max(d));
    e.min(f).max(g.min(e.max(f)))
}

/// Partition points[start..start+count] (and corresponding indices) so that
/// elements with `points[i][dim] <= pivot` come first.
/// Returns the number of elements in the "low" partition.
fn partition_points(
    points: &mut [[f64; 3]],
    indices: &mut [usize],
    start: usize,
    count: usize,
    dim: usize,
    pivot: f64,
) -> usize {
    let end = start + count;
    let mut lo: isize = start as isize - 1;
    let mut hi: isize = end as isize;
    loop {
        lo += 1;
        while (lo as usize) < end && points[lo as usize][dim] <= pivot {
            lo += 1;
        }
        hi -= 1;
        while (hi as usize) > start && points[hi as usize][dim] > pivot {
            hi -= 1;
        }
        if lo >= hi {
            break;
        }
        points.swap(lo as usize, hi as usize);
        indices.swap(lo as usize, hi as usize);
    }
    (lo as usize) - start
}

/// Recursively split a node to build the tree.
fn split_node(
    nodes: &mut Vec<KdNode>,
    points: &mut Vec<[f64; 3]>,
    indices: &mut Vec<usize>,
    node_id: usize,
    max_leaf_size: usize,
) {
    let left_id = 2 * node_id + 1;
    let right_id = 2 * node_id + 2;

    // Make this a leaf if we cannot fit children or have few enough points.
    if right_id >= nodes.len() || nodes[node_id].n_points <= max_leaf_size {
        nodes[node_id].split_dim = 3; // leaf
        return;
    }

    let bbx = nodes[node_id].bbx;
    let data_start = nodes[node_id].data_start;
    let n_points = nodes[node_id].n_points;

    // Choose split dimension: the one with the largest bounding-box extent.
    let mut split_dim = 0usize;
    let mut max_extent = bbx[1] - bbx[0];
    for dim in 1..3 {
        let extent = bbx[2 * dim + 1] - bbx[2 * dim];
        if extent > max_extent {
            split_dim = dim;
            max_extent = extent;
        }
    }

    // Find median along the split dimension.
    let mut values: Vec<f64> = (0..n_points)
        .map(|i| points[data_start + i][split_dim])
        .collect();
    let pivot = quickselect_median(&mut values);

    // Guard against infinite recursion when points are flat in this dimension.
    if pivot == bbx[2 * split_dim] || pivot == bbx[2 * split_dim + 1] {
        nodes[node_id].split_dim = 3; // leaf
        return;
    }

    nodes[node_id].split_dim = split_dim as u8;
    nodes[node_id].pivot = pivot;
    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;

    // Partition points around the pivot.
    let n_low = partition_points(points, indices, data_start, n_points, split_dim, pivot);
    let n_high = n_points - n_low;

    // Set up left child.
    let mut left_bbx = bbx;
    left_bbx[2 * split_dim + 1] = pivot;
    nodes[left_id] = KdNode {
        bbx: left_bbx,
        data_start,
        n_points: n_low,
        split_dim: 3,
        pivot: 0.0,
        left: 0,
        right: 0,
    };

    // Set up right child.
    let mut right_bbx = bbx;
    right_bbx[2 * split_dim] = pivot;
    nodes[right_id] = KdNode {
        bbx: right_bbx,
        data_start: data_start + n_low,
        n_points: n_high,
        split_dim: 3,
        pivot: 0.0,
        left: 0,
        right: 0,
    };

    // Recurse.
    split_node(nodes, points, indices, left_id, max_leaf_size);
    split_node(nodes, points, indices, right_id, max_leaf_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force k-NN for correctness comparison.
    fn brute_force_knn(points: &[[f64; 3]], query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut dists: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dist_sq(query, p).sqrt()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        dists
    }

    /// Brute-force radius query.
    fn brute_force_radius(
        points: &[[f64; 3]],
        query: &[f64; 3],
        radius: f64,
    ) -> Vec<(usize, f64)> {
        points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let d = dist_sq(query, p).sqrt();
                if d < radius {
                    Some((i, d))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Simple deterministic pseudo-random number generator (xorshift64).
    fn pseudo_random_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let mut coords = [0.0; 3];
            for c in coords.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *c = (state as f64) / (u64::MAX as f64) * 100.0;
            }
            points.push(coords);
        }
        points
    }

    #[test]
    fn test_build_tree() {
        let points = pseudo_random_points(100, 42);
        let tree = KdTree::new(&points, 10);
        assert!(!tree.nodes.is_empty());
        assert_eq!(tree.points.len(), 100);
        assert_eq!(tree.indices.len(), 100);
    }

    #[test]
    fn test_knn_small() {
        let points = pseudo_random_points(50, 123);
        let tree = KdTree::new(&points, 4);

        let query = [50.0, 50.0, 50.0];
        for k in [1, 3, 5, 10] {
            let tree_result = tree.query_knn(&query, k);
            let brute_result = brute_force_knn(&points, &query, k);

            assert_eq!(tree_result.len(), brute_result.len(), "k={}", k);
            for i in 0..k {
                assert!(
                    (tree_result[i].1 - brute_result[i].1).abs() < 1e-9,
                    "k={}, i={}: tree dist={}, brute dist={}",
                    k,
                    i,
                    tree_result[i].1,
                    brute_result[i].1,
                );
            }
        }
    }

    #[test]
    fn test_knn_larger() {
        let points = pseudo_random_points(500, 999);
        let tree = KdTree::new(&points, 8);

        let queries = pseudo_random_points(20, 777);
        for query in &queries {
            let k = 5;
            let tree_result = tree.query_knn(query, k);
            let brute_result = brute_force_knn(&points, query, k);

            for i in 0..k {
                assert!(
                    (tree_result[i].1 - brute_result[i].1).abs() < 1e-9,
                    "Mismatch at rank {}: tree={}, brute={}",
                    i,
                    tree_result[i].1,
                    brute_result[i].1,
                );
            }
        }
    }

    #[test]
    fn test_closest() {
        let points = pseudo_random_points(200, 55);
        let tree = KdTree::new(&points, 8);

        let query = [25.0, 75.0, 50.0];
        let (idx, dist) = tree.query_closest(&query);
        let brute = brute_force_knn(&points, &query, 1);
        assert_eq!(idx, brute[0].0);
        assert!((dist - brute[0].1).abs() < 1e-9);
    }

    #[test]
    fn test_radius_query() {
        let points = pseudo_random_points(200, 77);
        let tree = KdTree::new(&points, 8);

        let query = [50.0, 50.0, 50.0];
        let radius = 20.0;

        let mut tree_result = tree.query_radius(&query, radius);
        let mut brute_result = brute_force_radius(&points, &query, radius);

        // Sort both by index for comparison.
        tree_result.sort_by_key(|&(i, _)| i);
        brute_result.sort_by_key(|&(i, _)| i);

        assert_eq!(
            tree_result.len(),
            brute_result.len(),
            "Different number of results"
        );
        for (t, b) in tree_result.iter().zip(brute_result.iter()) {
            assert_eq!(t.0, b.0, "Index mismatch");
            assert!((t.1 - b.1).abs() < 1e-9, "Distance mismatch");
        }
    }

    #[test]
    fn test_kde() {
        let points = pseudo_random_points(100, 33);
        let tree = KdTree::new(&points, 8);

        let query = [50.0, 50.0, 50.0];
        let sigma = 10.0;
        let cutoff = 3.0;
        let r = cutoff * sigma;

        // Brute-force KDE.
        let sigma22 = 2.0 * sigma * sigma;
        let brute_kde: f64 = points
            .iter()
            .map(|p| {
                let d2 = dist_sq(&query, p);
                if d2 < r * r {
                    (-d2 / sigma22).exp()
                } else {
                    0.0
                }
            })
            .sum();

        let tree_kde = tree.kde(&query, sigma, cutoff);
        assert!(
            (tree_kde - brute_kde).abs() < 1e-9,
            "KDE mismatch: tree={}, brute={}",
            tree_kde,
            brute_kde,
        );
    }

    #[test]
    fn test_single_point() {
        let points = vec![[1.0, 2.0, 3.0]];
        let tree = KdTree::new(&points, 10);
        let (idx, dist) = tree.query_closest(&[1.0, 2.0, 3.0]);
        assert_eq!(idx, 0);
        assert!(dist < 1e-12);
    }

    #[test]
    fn test_identical_points() {
        let points = vec![[5.0, 5.0, 5.0]; 20];
        let tree = KdTree::new(&points, 4);
        let results = tree.query_knn(&[5.0, 5.0, 5.0], 5);
        assert_eq!(results.len(), 5);
        for (_, d) in &results {
            assert!(*d < 1e-12);
        }
    }
}
