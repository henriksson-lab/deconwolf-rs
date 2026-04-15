# deconwolf-rs

A Rust port of [deconwolf](https://github.com/elgw/deconwolf), a software for
3D deconvolution of fluorescent wide-field microscopy images[^1].


## This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this page](https://rewrites.bio/)
but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back to the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information


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
