# deconwolf-rs

A Rust port of [deconwolf](https://github.com/elgw/deconwolf), a software for
3D deconvolution of fluorescent wide-field microscopy images[^1].

The original C implementation is preserved in `deconwolf/` as a reference.

## Building

Requires Rust 1.70+. No external C libraries needed — the default build is pure Rust.

```bash
cargo build --release
```

The binary is at `target/release/dw`.

## Subcommands

| Command | Description |
|---------|-------------|
| `deconvolve` | 3D deconvolution (Richardson-Lucy or Scaled Heavy Ball) |
| `psf` | Generate PSF (widefield, confocal, or STED) |
| `dots` | Detect diffraction-limited dots via Laplacian of Gaussian |
| `maxproj` | Maximum Z-projection, XYZ collage, or best-focus slice |
| `merge` | Merge 2D TIFF slices into a 3D volume |
| `imshift` | Translate a 3D image with sub-pixel interpolation |
| `background` | Estimate background/vignetting from multiple images |
| `noise1` | Noise reduction (L1 + total variation regularization) |
| `tif2npy` | Convert TIFF to NumPy `.npy` |
| `npy2tif` | Convert NumPy `.npy` to TIFF |

## Usage examples

### Deconvolution

```bash
dw deconvolve image.tif psf.tif                           # SHB, adaptive stopping
dw deconvolve -m rl -n 50 image.tif psf.tif               # Richardson-Lucy, 50 iterations
dw deconvolve -m shb -o result.tif image.tif psf.tif      # explicit output path
dw deconvolve -F f32 image.tif psf.tif                    # 32-bit float output
dw deconvolve -s 512 image.tif psf.tif                    # tile large images
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `shb` | `shb` (Scaled Heavy Ball), `rl` (Richardson-Lucy), `id` (identity) |
| `-n, --iter` | — | Fixed iteration count |
| `-N, --maxiter` | 250 | Max iterations for adaptive stopping |
| `-j, --relerror` | 0.02 | Relative error threshold |
| `-M, --metric` | `idiv` | `idiv` (I-divergence) or `mse` |
| `-F, --format` | `u16` | Output: `u16` (16-bit, auto-scaled) or `f32` |
| `-B, --bq` | 2 | Border quality: 0=periodic, 1=compromise, 2=full |
| `-b, --bg` | 0.01 | Background level |
| `-q, --offset` | 5 | Offset to reduce Gaussian noise |
| `-c, --threads` | auto | Thread count |
| `-s, --tilesize` | — | Max tile size (enables tiling) |
| `--start` | `flat` | Initial guess: `flat`, `identity`, `low-pass` |

### PSF generation

```bash
dw psf psf.tif --na 1.4 --ni 1.515 --lambda 525 --dx 65 --dz 200
dw psf psf.tif --psf-type confocal --na 1.4 --ni 1.515 --lambda 525 --lambda2 488 --dx 65 --dz 200
dw psf psf.tif --psf-type sted --lateral 2.0 --axial 4.0 --size 81 --nslice 81
```

### Dot detection

```bash
dw dots image.tif dots.tsv --na 1.4 --ni 1.515 --lambda 525 --dx 65 --dz 200
dw dots image.tif dots.csv --na 1.4 --ni 1.515 --lambda 525 --dx 65 --dz 200 --csv --ndots 100
```

### Image utilities

```bash
dw maxproj input.tif output.tif                             # max Z-projection
dw maxproj --xyz input.tif output.tif                       # XY/XZ/YZ collage
dw maxproj --gm input.tif output.tif                        # most in-focus slice
dw maxproj --slice 10 input.tif output.tif                  # extract z-slice
dw merge output.tif slice1.tif slice2.tif slice3.tif        # stack slices
dw imshift input.tif output.tif --dx 5.5 --dy -3.2 --dz 1  # translate
dw background --out bg.tif img1.tif img2.tif --sigma 100    # vignetting estimate
dw noise1 input.tif output.tif --lambda 0.1 --lambda-s 0.1  # denoise
dw tif2npy input.tif output.npy                             # format conversion
dw npy2tif input.npy output.tif
```

## Source structure

```
src/
  core/           image type, operations, TIFF/NPY I/O, tiling, K-d tree, random forest, data tables
  fft/            FFT backend (pure-Rust 3D real-to-complex via composed 1D FFTs)
  deconv/         deconvolution methods (RL, SHB), config, Bertero boundaries, runner
  tools/          PSF generation, dot detection, maxproj, merge, imshift, background, denoising
  main.rs         CLI (clap) → `dw` binary
```

## Tests

```bash
cargo test    # 57 tests
```

## License

GPL-3.0-or-later

[^1]: E. Wernersson et al. "Deconwolf enables high-performance deconvolution of
    widefield fluorescence microscopy images", Nature Methods, 2024,
    [doi:10.1038/s41592-024-02294-7](https://doi.org/10.1038/s41592-024-02294-7)
