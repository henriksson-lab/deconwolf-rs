# deconwolf-rs

A Rust port of [deconwolf](https://github.com/elgw/deconwolf), a software for
3D deconvolution of fluorescent wide-field microscopy images[^1].

The original C implementation is preserved in `deconwolf/` as a reference.

## Building

Requires Rust 1.70+.

```bash
cargo build --release
```

## Usage

### Deconvolution

```bash
dw deconvolve image.tif psf.tif
dw deconvolve -m rl -n 50 image.tif psf.tif              # Richardson-Lucy, 50 iterations
dw deconvolve -m shb -o result.tif image.tif psf.tif      # Scaled Heavy Ball (default)
dw deconvolve -F f32 image.tif psf.tif                    # 32-bit float output
dw deconvolve -s 512 image.tif psf.tif                    # tile-based processing
```

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
dw maxproj input.tif output.tif             # maximum Z-projection
dw maxproj --xyz input.tif output.tif       # XY/XZ/YZ collage
dw maxproj --gm input.tif output.tif        # most in-focus slice
dw maxproj --slice 10 input.tif output.tif  # extract slice 10
dw merge output.tif slice1.tif slice2.tif   # merge slices into 3D stack
dw imshift input.tif output.tif --dx 5.5 --dy -3.2 --dz 1.0
dw background --out bg.tif img1.tif img2.tif img3.tif --sigma 100
dw noise1 input.tif output.tif --lambda 0.1 --lambda-s 0.1 -n 10
dw tif2npy input.tif output.npy
dw npy2tif input.npy output.tif
```

### Deconvolution options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --method` | `shb` | Algorithm: `shb` (Scaled Heavy Ball), `rl` (Richardson-Lucy), `id` (identity) |
| `-n, --iter` | — | Fixed number of iterations |
| `-N, --maxiter` | 250 | Maximum iterations for adaptive stopping |
| `-j, --relerror` | 0.02 | Relative error threshold for stopping |
| `-M, --metric` | `idiv` | Error metric: `idiv` (I-divergence) or `mse` |
| `-F, --format` | `u16` | Output format: `u16` (16-bit, auto-scaled) or `f32` (float) |
| `-B, --bq` | 2 | Border quality: 0=periodic, 1=compromise, 2=full |
| `-b, --bg` | 0.01 | Background level |
| `-q, --offset` | 5 | Offset to reduce Gaussian noise |
| `-c, --threads` | auto | Number of threads |
| `-s, --tilesize` | — | Enable tiling with max tile size |
| `--start` | `flat` | Initial guess: `flat`, `identity`, or `low-pass` |

## Source structure

| Module | Description |
|--------|-------------|
| `src/core/` | Image type, operations, TIFF/NPY I/O, tiling, K-d tree, random forest, data tables |
| `src/fft/` | FFT abstraction with `RustFftBackend` (pure Rust, 3D via 1D decomposition) |
| `src/deconv/` | Deconvolution methods (RL, SHB), configuration, runner |
| `src/tools/` | PSF generation, dot detection, maxproj, merge, imshift, background, denoising |
| `src/main.rs` | CLI entry point (clap), produces `dw` binary |

## Running tests

```bash
cargo test    # 57 tests
```

## License

GPL-3.0-or-later

[^1]: E. Wernersson et al. "Deconwolf enables high-performance deconvolution of
    widefield fluorescence microscopy images", Nature Methods, 2024,
    [doi:10.1038/s41592-024-02294-7](https://doi.org/10.1038/s41592-024-02294-7)
