# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deconwolf is a deconvolution tool for 3D fluorescence microscopy images, published in Nature Methods 2024. This repository contains both:
- **Rust port** (active development) — single crate at the repo root
- **C reference** (`deconwolf/`) — Original ~31K LOC C implementation, kept for reference

## Build Commands

```bash
cargo build              # debug build
cargo build --release    # release build
cargo test               # run all 27 tests
cargo run -- --help      # run the CLI
cargo run -- deconvolve image.tif psf.tif  # run deconvolution
```

## Source Layout

```
src/
  lib.rs              # crate root: pub mod core, fft, deconv, tools
  main.rs             # CLI entry point (clap), produces `dw` binary
  core/
    mod.rs            # re-exports FimImage, DwError, Result
    image.rs          # FimImage: wraps ndarray::Array3<f32>, shape [P,N,M]
    image_ops.rs      # min/max/sum/mean/std/percentile, projections, Anscombe, etc.
    tiff_io.rs        # TIFF read/write (u8/u16/f32), ImageJ metadata
    npy_io.rs         # NumPy .npy read/write
    tiling.rs         # tile decomposition with weighted blending
    error.rs          # DwError enum, Result type alias
  fft/
    mod.rs            # re-exports FftBackend, FftContext
    backend.rs        # FftBackend trait, FftContext<B> with convolve/convolve_conj
    rustfft_backend.rs # 3D real-to-complex via composed 1D FFTs (default)
    complex_ops.rs    # element-wise complex multiply/conj
  deconv/
    mod.rs            # re-exports DwOpts, DeconvMethod
    config.rs         # DwOpts config struct, Method/Metric/IterType enums
    method.rs         # DeconvMethod trait
    identity.rs       # pass-through method (testing)
    rl.rs             # Richardson-Lucy
    shb.rs            # Scaled Heavy Ball (default, Nesterov momentum)
    boundary.rs       # Bertero boundary weights
    iterator.rs       # convergence control (fixed/relative/absolute)
    runner.rs         # dw_run() orchestration: load → preprocess → deconvolve → write
  tools/
    maxproj.rs        # max Z-projection, slice extraction
```

## Architecture

### Core image type
`FimImage` wraps `ndarray::Array3<f32>` with shape `[P, N, M]` (z, y, x). M is stride-1, matching the C memory layout. Defined in `src/core/image.rs`.

### FFT backend trait
`FftBackend` trait in `src/fft/backend.rs` with `RustFftBackend` (3D via 1D decomposition). `FftContext<B>` provides `convolve()` and `convolve_conj()`.

### Deconvolution method trait
`DeconvMethod` trait in `src/deconv/method.rs`. Implemented by `IdentityMethod`, `RlMethod`, `ShbMethod`. Method dispatch in `runner.rs`.

### Parallelism
`rayon` replaces OpenMP. Elementwise operations use `par_iter_mut()`.

## C Reference Build (in `deconwolf/`)

```bash
cd deconwolf && mkdir build && cd build && cmake .. && make -j$(nproc)
```

Dependencies: FFTW3, GSL, LibTIFF, LibPNG, OpenMP. C11/C99 standard.
