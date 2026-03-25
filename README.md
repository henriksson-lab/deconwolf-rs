# deconwolf-rs

A Rust port of [deconwolf](https://github.com/elgw/deconwolf), a software for
3D deconvolution of fluorescent wide-field microscopy images[^1].

The original C implementation is preserved in `deconwolf/` as a reference.

## Building

Requires Rust 1.70+.

```bash
cargo build --release
```

### Optional features

| Feature | Description |
|---------|-------------|
| `fftw`  | Use FFTW3 instead of the default pure-Rust FFT backend (requires FFTW3 installed) |
| `gpu`   | Enable OpenCL GPU acceleration (work in progress) |

```bash
cargo build --release --features fftw
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

### Utilities

```bash
dw maxproj input.tif output.tif             # maximum Z-projection
dw maxproj --slice 10 input.tif output.tif  # extract slice 10
dw tif2npy input.tif output.npy             # convert TIFF to NumPy
dw npy2tif input.npy output.tif             # convert NumPy to TIFF
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
| `src/core/` | Image type (`FimImage`), image operations, TIFF/NPY I/O, tiling |
| `src/fft/` | FFT abstraction with `RustFftBackend` (pure Rust, 3D via 1D decomposition) |
| `src/deconv/` | Deconvolution methods (RL, SHB), configuration, runner |
| `src/tools/` | Utility subcommands (maxproj, etc.) |
| `src/main.rs` | CLI entry point (clap), produces `dw` binary |

## Running tests

```bash
cargo test
```

## License

GPL-3.0-or-later

[^1]: E. Wernersson et al. "Deconwolf enables high-performance deconvolution of
    widefield fluorescence microscopy images", Nature Methods, 2024,
    [doi:10.1038/s41592-024-02294-7](https://doi.org/10.1038/s41592-024-02294-7)
