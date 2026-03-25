//! Random Forest classifier for nuclei pixel classification.
//!
//! Port of the C `trafo` library. Supports training, prediction,
//! feature importance, and binary save/load.

use rayon::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

use super::error::{DwError, Result};

// ---------------------------------------------------------------------------
// Magic bytes for the binary format
// ---------------------------------------------------------------------------
const TRAFO_MAGIC: u32 = 0x5452_4146; // "TRAF"
const TRAFO_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A single decision tree node.
#[derive(Debug, Clone)]
struct TreeNode {
    /// Feature index to split on (-1 for leaf).
    feature: i32,
    /// Split threshold.
    threshold: f32,
    /// Left child index (or class label for leaves).
    left: u32,
    /// Right child index (unused for leaves).
    right: u32,
}

impl TreeNode {
    fn is_leaf(&self) -> bool {
        self.feature < 0
    }

    fn class_label(&self) -> u32 {
        self.left // class stored in left for leaves
    }
}

// ---------------------------------------------------------------------------
// Decision tree
// ---------------------------------------------------------------------------

/// A single decision tree.
#[derive(Debug, Clone)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

impl DecisionTree {
    /// Predict the class for a single sample (row-major feature slice).
    fn predict_one(&self, sample: &[f64]) -> u32 {
        let mut idx: usize = 0;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf() {
                return node.class_label();
            }
            let val = sample[node.feature as usize];
            idx = if val <= node.threshold as f64 {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }
}

// ---------------------------------------------------------------------------
// RandomForest
// ---------------------------------------------------------------------------

/// Random Forest classifier.
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_features: usize,
    n_classes: u32,
}

/// Settings for training.
#[derive(Debug, Clone)]
pub struct TrafoSettings {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split. `None` means sqrt(n_features).
    pub max_features: Option<usize>,
    /// Fraction of samples in each bootstrap (default 0.632).
    pub sample_fraction: f64,
    /// If `true` use entropy; otherwise use Gini impurity (default).
    pub use_entropy: bool,
}

impl Default for TrafoSettings {
    fn default() -> Self {
        Self {
            n_trees: 50,
            min_samples_leaf: 1,
            max_features: None,
            sample_fraction: 0.632,
            use_entropy: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Splitting criteria
// ---------------------------------------------------------------------------

/// Gini impurity: 1 - sum(p_i^2).
fn gini_impurity(class_counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let sum_sq: f64 = class_counts
        .iter()
        .map(|&c| {
            let p = c as f64 / t;
            p * p
        })
        .sum();
    1.0 - sum_sq
}

/// Entropy: -sum(p_i * log2(p_i)).
fn entropy(class_counts: &[u32], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let e: f64 = class_counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                let p = c as f64 / t;
                -p * p.log2()
            }
        })
        .sum();
    e
}

/// Compute impurity using the selected criterion.
#[inline]
fn impurity(class_counts: &[u32], total: u32, use_entropy: bool) -> f64 {
    if use_entropy {
        entropy(class_counts, total)
    } else {
        gini_impurity(class_counts, total)
    }
}

/// Find the best split across the given feature subset.
///
/// Returns `(feature_idx, threshold, impurity_reduction)` or `None` if no
/// valid split exists.
fn find_best_split(
    features: &[f64],
    labels: &[u32],
    indices: &[usize],
    feature_subset: &[usize],
    n_features: usize,
    n_classes: u32,
    min_leaf: usize,
    use_entropy: bool,
) -> Option<(usize, f64, f64)> {
    let n = indices.len();
    if n < 2 * min_leaf {
        return None;
    }

    let nc = n_classes as usize;

    // Parent counts.
    let mut parent_counts = vec![0u32; nc];
    for &i in indices {
        parent_counts[labels[i] as usize] += 1;
    }
    let parent_imp = impurity(&parent_counts, n as u32, use_entropy);

    let mut best: Option<(usize, f64, f64)> = None;

    // Sorted index buffer reused per feature.
    let mut sorted_idx: Vec<usize> = indices.to_vec();

    for &feat in feature_subset {
        // Sort indices by feature value.
        sorted_idx.copy_from_slice(indices);
        sorted_idx.sort_by(|&a, &b| {
            let va = features[a * n_features + feat];
            let vb = features[b * n_features + feat];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut left_counts = vec![0u32; nc];
        let mut right_counts = parent_counts.clone();

        for split_pos in 0..n - 1 {
            let idx = sorted_idx[split_pos];
            let cls = labels[idx] as usize;
            left_counts[cls] += 1;
            right_counts[cls] -= 1;

            let left_total = (split_pos + 1) as u32;
            let right_total = (n - split_pos - 1) as u32;

            // Enforce min_leaf.
            if (left_total as usize) < min_leaf || (right_total as usize) < min_leaf {
                continue;
            }

            // Skip if feature values are equal (no real split).
            let v_cur = features[idx * n_features + feat];
            let v_next = features[sorted_idx[split_pos + 1] * n_features + feat];
            if (v_cur - v_next).abs() < f64::EPSILON {
                continue;
            }

            let left_imp = impurity(&left_counts, left_total, use_entropy);
            let right_imp = impurity(&right_counts, right_total, use_entropy);

            let wl = left_total as f64 / n as f64;
            let wr = right_total as f64 / n as f64;
            let reduction = parent_imp - wl * left_imp - wr * right_imp;

            let threshold = (v_cur + v_next) / 2.0;

            if let Some((_, _, best_red)) = best {
                if reduction > best_red {
                    best = Some((feat, threshold, reduction));
                }
            } else {
                best = Some((feat, threshold, reduction));
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Tree building
// ---------------------------------------------------------------------------

/// Determine majority class among `indices`.
fn majority_class(labels: &[u32], indices: &[usize], n_classes: u32) -> u32 {
    let mut counts = vec![0u32; n_classes as usize];
    for &i in indices {
        counts[labels[i] as usize] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(cls, _)| cls as u32)
        .unwrap_or(0)
}

/// Check whether all labels in `indices` are the same class.
fn all_same_class(labels: &[u32], indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }
    let first = labels[indices[0]];
    indices.iter().all(|&i| labels[i] == first)
}

/// Recursively build a decision tree, appending nodes to `nodes`.
/// Returns the index of the root node of this subtree.
fn build_tree_recursive(
    nodes: &mut Vec<TreeNode>,
    features: &[f64],
    labels: &[u32],
    indices: &[usize],
    n_features: usize,
    feature_subset: &[usize],
    n_classes: u32,
    min_leaf: usize,
    use_entropy: bool,
) -> usize {
    // Leaf conditions.
    if indices.len() <= min_leaf || all_same_class(labels, indices) {
        let cls = majority_class(labels, indices, n_classes);
        let idx = nodes.len();
        nodes.push(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: cls,
            right: 0,
        });
        return idx;
    }

    // Try to find a split.
    let split = find_best_split(
        features,
        labels,
        indices,
        feature_subset,
        n_features,
        n_classes,
        min_leaf,
        use_entropy,
    );

    let (feat, thresh, _) = match split {
        Some(s) => s,
        None => {
            // Cannot split further -> leaf.
            let cls = majority_class(labels, indices, n_classes);
            let idx = nodes.len();
            nodes.push(TreeNode {
                feature: -1,
                threshold: 0.0,
                left: cls,
                right: 0,
            });
            return idx;
        }
    };

    // Partition indices.
    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| features[i * n_features + feat] <= thresh);

    // Reserve a slot for this node.
    let node_idx = nodes.len();
    nodes.push(TreeNode {
        feature: feat as i32,
        threshold: thresh as f32,
        left: 0,
        right: 0,
    });

    // Build children.
    let left_child = build_tree_recursive(
        nodes,
        features,
        labels,
        &left_idx,
        n_features,
        feature_subset,
        n_classes,
        min_leaf,
        use_entropy,
    );
    let right_child = build_tree_recursive(
        nodes,
        features,
        labels,
        &right_idx,
        n_features,
        feature_subset,
        n_classes,
        min_leaf,
        use_entropy,
    );

    nodes[node_idx].left = left_child as u32;
    nodes[node_idx].right = right_child as u32;

    node_idx
}

/// Build a single decision tree from the provided sample indices.
fn build_tree(
    features: &[f64],
    labels: &[u32],
    indices: &[usize],
    n_features: usize,
    feature_subset: &[usize],
    n_classes: u32,
    min_leaf: usize,
    use_entropy: bool,
) -> DecisionTree {
    let mut nodes = Vec::new();
    build_tree_recursive(
        &mut nodes,
        features,
        labels,
        indices,
        n_features,
        feature_subset,
        n_classes,
        min_leaf,
        use_entropy,
    );
    DecisionTree { nodes }
}

// ---------------------------------------------------------------------------
// RandomForest implementation
// ---------------------------------------------------------------------------

impl RandomForest {
    /// Train a random forest.
    ///
    /// `features` is a row-major `n_samples x n_features` slice.
    /// `labels` contains one class label per sample (0-based).
    pub fn fit(
        features: &[f64],
        labels: &[u32],
        n_samples: usize,
        n_features: usize,
        settings: &TrafoSettings,
    ) -> Result<Self> {
        if features.len() != n_samples * n_features {
            return Err(DwError::Config(format!(
                "features length {} != n_samples({}) * n_features({})",
                features.len(),
                n_samples,
                n_features
            )));
        }
        if labels.len() != n_samples {
            return Err(DwError::Config(
                "labels length does not match n_samples".into(),
            ));
        }
        if n_samples == 0 || n_features == 0 {
            return Err(DwError::Config("empty training set".into()));
        }

        let n_classes = labels.iter().copied().max().unwrap_or(0) + 1;

        let max_feat = settings
            .max_features
            .unwrap_or_else(|| (n_features as f64).sqrt().ceil() as usize)
            .min(n_features)
            .max(1);

        let bootstrap_size =
            ((n_samples as f64 * settings.sample_fraction).round() as usize).max(1);

        // Build trees in parallel.
        let trees: Vec<DecisionTree> = (0..settings.n_trees)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = StdRng::seed_from_u64(tree_idx as u64 ^ 0xCAFE_BABE);

                // Bootstrap sample.
                let indices: Vec<usize> = (0..bootstrap_size)
                    .map(|_| rng.gen_range(0..n_samples))
                    .collect();

                // Random feature subset.
                let mut all_feats: Vec<usize> = (0..n_features).collect();
                all_feats.shuffle(&mut rng);
                let feature_subset: Vec<usize> =
                    all_feats.into_iter().take(max_feat).collect();

                build_tree(
                    features,
                    labels,
                    &indices,
                    n_features,
                    &feature_subset,
                    n_classes,
                    settings.min_samples_leaf,
                    settings.use_entropy,
                )
            })
            .collect();

        Ok(Self {
            trees,
            n_features,
            n_classes,
        })
    }

    /// Predict class labels for `n_samples` samples (row-major features).
    pub fn predict(&self, features: &[f64], n_samples: usize) -> Vec<u32> {
        let nf = self.n_features;
        let nc = self.n_classes as usize;

        (0..n_samples)
            .into_par_iter()
            .map(|s| {
                let sample = &features[s * nf..(s + 1) * nf];
                // Tally votes.
                let mut votes = vec![0u32; nc];
                for tree in &self.trees {
                    let cls = tree.predict_one(sample) as usize;
                    if cls < nc {
                        votes[cls] += 1;
                    }
                }
                votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(cls, _)| cls as u32)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute feature importance as the fraction of splits using each feature.
    pub fn feature_importance(&self) -> Vec<f64> {
        let mut counts = vec![0u64; self.n_features];
        let mut total: u64 = 0;
        for tree in &self.trees {
            for node in &tree.nodes {
                if !node.is_leaf() {
                    let f = node.feature as usize;
                    if f < self.n_features {
                        counts[f] += 1;
                        total += 1;
                    }
                }
            }
        }
        if total == 0 {
            return vec![0.0; self.n_features];
        }
        counts.iter().map(|&c| c as f64 / total as f64).collect()
    }

    /// Save the forest in a compact binary format.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&TRAFO_MAGIC.to_le_bytes())?;
        w.write_all(&TRAFO_VERSION.to_le_bytes())?;
        w.write_all(&(self.n_features as u32).to_le_bytes())?;
        w.write_all(&self.n_classes.to_le_bytes())?;
        w.write_all(&(self.trees.len() as u32).to_le_bytes())?;

        for tree in &self.trees {
            w.write_all(&(tree.nodes.len() as u32).to_le_bytes())?;
            for node in &tree.nodes {
                w.write_all(&node.feature.to_le_bytes())?;
                w.write_all(&node.threshold.to_le_bytes())?;
                w.write_all(&node.left.to_le_bytes())?;
                w.write_all(&node.right.to_le_bytes())?;
            }
        }

        w.flush()?;
        Ok(())
    }

    /// Load a forest from the binary format.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != TRAFO_MAGIC {
            return Err(DwError::Config("not a trafo file (bad magic)".into()));
        }

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != TRAFO_VERSION {
            return Err(DwError::Config(format!(
                "unsupported trafo version {}",
                version
            )));
        }

        r.read_exact(&mut buf4)?;
        let n_features = u32::from_le_bytes(buf4) as usize;

        r.read_exact(&mut buf4)?;
        let n_classes = u32::from_le_bytes(buf4);

        r.read_exact(&mut buf4)?;
        let n_trees = u32::from_le_bytes(buf4) as usize;

        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            r.read_exact(&mut buf4)?;
            let n_nodes = u32::from_le_bytes(buf4) as usize;

            let mut nodes = Vec::with_capacity(n_nodes);
            for _ in 0..n_nodes {
                r.read_exact(&mut buf4)?;
                let feature = i32::from_le_bytes(buf4);

                r.read_exact(&mut buf4)?;
                let threshold = f32::from_le_bytes(buf4);

                r.read_exact(&mut buf4)?;
                let left = u32::from_le_bytes(buf4);

                r.read_exact(&mut buf4)?;
                let right = u32::from_le_bytes(buf4);

                nodes.push(TreeNode {
                    feature,
                    threshold,
                    left,
                    right,
                });
            }
            trees.push(DecisionTree { nodes });
        }

        Ok(Self {
            trees,
            n_features,
            n_classes,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate XOR-like 2D data: class 1 when (x>0.5) XOR (y>0.5).
    fn xor_dataset() -> (Vec<f64>, Vec<u32>, usize, usize) {
        let n = 200;
        let n_features = 2;
        let mut features = Vec::with_capacity(n * n_features);
        let mut labels = Vec::with_capacity(n);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..n {
            let x: f64 = rng.gen();
            let y: f64 = rng.gen();
            features.push(x);
            features.push(y);
            let cls = if (x > 0.5) ^ (y > 0.5) { 1u32 } else { 0u32 };
            labels.push(cls);
        }
        (features, labels, n, n_features)
    }

    #[test]
    fn test_xor_accuracy() {
        let (features, labels, n, nf) = xor_dataset();
        let settings = TrafoSettings {
            n_trees: 30,
            min_samples_leaf: 2,
            ..Default::default()
        };
        let forest = RandomForest::fit(&features, &labels, n, nf, &settings).unwrap();
        let preds = forest.predict(&features, n);

        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p == l)
            .count();
        let acc = correct as f64 / n as f64;
        assert!(
            acc > 0.85,
            "XOR accuracy too low: {:.2}% ({}/{})",
            acc * 100.0,
            correct,
            n
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let (features, labels, n, nf) = xor_dataset();
        let settings = TrafoSettings {
            n_trees: 10,
            ..Default::default()
        };
        let forest = RandomForest::fit(&features, &labels, n, nf, &settings).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("trafo_test_roundtrip.bin");

        forest.save(&path).unwrap();
        let loaded = RandomForest::load(&path).unwrap();

        assert_eq!(forest.n_features, loaded.n_features);
        assert_eq!(forest.n_classes, loaded.n_classes);
        assert_eq!(forest.trees.len(), loaded.trees.len());

        // Predictions should be identical.
        let p1 = forest.predict(&features, n);
        let p2 = loaded.predict(&features, n);
        assert_eq!(p1, p2);

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_simple_separable() {
        // Two clearly separable clusters along feature 0.
        let n = 100;
        let nf = 1;
        let mut features = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            if i < n / 2 {
                features.push(i as f64);
                labels.push(0);
            } else {
                features.push((i + 1000) as f64);
                labels.push(1);
            }
        }

        let settings = TrafoSettings {
            n_trees: 10,
            ..Default::default()
        };
        let forest = RandomForest::fit(&features, &labels, n, nf, &settings).unwrap();
        let preds = forest.predict(&features, n);
        assert_eq!(
            preds, labels,
            "perfectly separable data should be classified perfectly"
        );
    }

    #[test]
    fn test_feature_importance() {
        // Feature 0 is informative, feature 1 is noise.
        let n = 200;
        let nf = 2;
        let mut features = Vec::with_capacity(n * nf);
        let mut labels = Vec::with_capacity(n);
        let mut rng = StdRng::seed_from_u64(99);

        for _ in 0..n {
            let x: f64 = rng.gen();
            let noise: f64 = rng.gen();
            features.push(x);
            features.push(noise);
            labels.push(if x > 0.5 { 1 } else { 0 });
        }

        let settings = TrafoSettings {
            n_trees: 30,
            max_features: Some(2),
            ..Default::default()
        };
        let forest = RandomForest::fit(&features, &labels, n, nf, &settings).unwrap();
        let imp = forest.feature_importance();
        assert_eq!(imp.len(), 2);
        assert!(
            imp[0] > imp[1],
            "informative feature should be more important: {:?}",
            imp
        );
    }

    #[test]
    fn test_gini_and_entropy() {
        // Pure node.
        assert!((gini_impurity(&[10, 0], 10) - 0.0).abs() < 1e-9);
        assert!((entropy(&[10, 0], 10) - 0.0).abs() < 1e-9);

        // Balanced binary.
        assert!((gini_impurity(&[5, 5], 10) - 0.5).abs() < 1e-9);
        assert!((entropy(&[5, 5], 10) - 1.0).abs() < 1e-9);
    }
}
