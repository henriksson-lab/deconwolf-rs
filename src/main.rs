use std::path::PathBuf;

use clap::{Parser, Subcommand};

use deconwolf::deconv::config;
use deconwolf::fft::rustfft_backend::RustFftBackend;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = "dw", version = VERSION, about = "Deconwolf - 3D deconvolution for fluorescence microscopy")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Deconvolve a 3D image
    Deconvolve {
        /// Input image (TIFF or NPY)
        image: PathBuf,

        /// Point Spread Function (TIFF or NPY)
        psf: PathBuf,

        /// Output file
        #[arg(short, long)]
        out: Option<PathBuf>,

        /// Deconvolution method
        #[arg(short, long, value_enum, default_value = "shb")]
        method: config::Method,

        /// Number of iterations (fixed mode)
        #[arg(short = 'n', long)]
        iter: Option<usize>,

        /// Maximum iterations (for adaptive stopping)
        #[arg(short = 'N', long, default_value = "250")]
        maxiter: usize,

        /// Relative error threshold for stopping
        #[arg(short = 'j', long, default_value = "0.02")]
        relerror: f64,

        /// Error metric
        #[arg(short = 'M', long, value_enum, default_value = "idiv")]
        metric: config::Metric,

        /// Output format
        #[arg(short = 'F', long, value_enum, default_value = "u16")]
        format: config::OutputFormat,

        /// Manual scaling factor for 16-bit output
        #[arg(short = 'S', long)]
        scaling: Option<f32>,

        /// Border quality: 0=periodic, 1=compromise, 2=normal
        #[arg(short = 'B', long, default_value = "2")]
        bq: u8,

        /// Background level
        #[arg(short = 'b', long, default_value = "0.01")]
        bg: f32,

        /// Offset added to reduce Gaussian noise
        #[arg(short = 'q', long, default_value = "5")]
        offset: f32,

        /// Number of threads
        #[arg(short = 'c', long)]
        threads: Option<usize>,

        /// Verbosity level
        #[arg(short = 'l', long, default_value = "1")]
        verbose: u32,

        /// Remove N z-planes from top and bottom
        #[arg(short = 'Z', long, default_value = "0")]
        cz: usize,

        /// Auto-crop to N z-planes
        #[arg(short = 'A', long, default_value = "0")]
        az: usize,

        /// Output file prefix
        #[arg(short = 'f', long, default_value = "dw")]
        prefix: String,

        /// Overwrite existing output
        #[arg(short = 'w', long)]
        overwrite: bool,

        /// Initial guess: flat, identity, or lowpass
        #[arg(long, value_enum, default_value = "flat")]
        start: config::StartCondition,

        /// Maximum tiling size (enables tiling)
        #[arg(short = 's', long)]
        tilesize: Option<usize>,

        /// Tile overlap padding
        #[arg(short = 'p', long, default_value = "20")]
        tilepad: usize,
    },

    /// Maximum Z-projection
    Maxproj {
        /// Input TIFF file
        input: PathBuf,

        /// Output TIFF file
        output: PathBuf,

        /// Extract specific slice instead of max projection
        #[arg(long)]
        slice: Option<usize>,

        /// Create XY/XZ/YZ collage
        #[arg(long)]
        xyz: bool,

        /// Extract most in-focus slice (gradient magnitude)
        #[arg(long)]
        gm: bool,
    },

    /// Merge TIFF slices into a 3D volume
    Merge {
        /// Output TIFF file
        output: PathBuf,

        /// Input TIFF files (one per z-slice)
        inputs: Vec<PathBuf>,
    },

    /// Estimate background/vignetting from multiple images
    Background {
        /// Output background TIFF
        #[arg(short, long)]
        out: PathBuf,

        /// Gaussian smoothing sigma (pixels)
        #[arg(long, default_value = "100")]
        sigma: f32,

        /// Input TIFF files
        inputs: Vec<PathBuf>,
    },

    /// Shift/translate a 3D image
    Imshift {
        /// Input TIFF file
        input: PathBuf,

        /// Output TIFF file
        output: PathBuf,

        /// Shift in X (pixels)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        dx: f32,

        /// Shift in Y (pixels)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        dy: f32,

        /// Shift in Z (pixels)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        dz: f32,
    },

    /// Detect dots using Laplacian of Gaussian
    Dots {
        /// Input TIFF file
        input: PathBuf,

        /// Output TSV/CSV file
        output: PathBuf,

        /// Numerical aperture
        #[arg(long)]
        na: f64,

        /// Refractive index
        #[arg(long)]
        ni: f64,

        /// Emission wavelength (nm)
        #[arg(long)]
        lambda: f64,

        /// Lateral pixel size (nm)
        #[arg(long)]
        dx: f64,

        /// Axial pixel size (nm)
        #[arg(long)]
        dz: f64,

        /// Number of dots to export
        #[arg(long)]
        ndots: Option<usize>,

        /// Output CSV instead of TSV
        #[arg(long)]
        csv: bool,
    },

    /// Generate a PSF (Point Spread Function)
    Psf {
        /// Output TIFF file
        output: PathBuf,

        /// PSF type: widefield, confocal, or sted
        #[arg(long, default_value = "widefield")]
        psf_type: String,

        /// Numerical aperture
        #[arg(long)]
        na: Option<f64>,

        /// Refractive index
        #[arg(long)]
        ni: Option<f64>,

        /// Emission wavelength (nm)
        #[arg(long)]
        lambda: Option<f64>,

        /// Excitation wavelength (nm, for confocal)
        #[arg(long)]
        lambda2: Option<f64>,

        /// Lateral pixel size (nm)
        #[arg(long)]
        dx: Option<f64>,

        /// Axial pixel size (nm)
        #[arg(long)]
        dz: Option<f64>,

        /// Lateral size (pixels, must be odd)
        #[arg(long, default_value = "181")]
        size: usize,

        /// Number of Z-planes (must be odd)
        #[arg(long, default_value = "181")]
        nslice: usize,

        /// Lateral FWHM in pixels (STED mode)
        #[arg(long)]
        lateral: Option<f64>,

        /// Axial FWHM in pixels (STED mode)
        #[arg(long)]
        axial: Option<f64>,

        /// Pinhole size in Airy Units (confocal)
        #[arg(long, default_value = "1.0")]
        pinhole: f64,
    },

    /// Sparse preprocessing (noise reduction)
    Noise1 {
        /// Input TIFF file
        input: PathBuf,

        /// Output TIFF file
        output: PathBuf,

        /// L1 sparsity penalty
        #[arg(long, default_value = "0.1")]
        lambda: f64,

        /// Smoothness penalty
        #[arg(long, default_value = "0.1")]
        lambda_s: f64,

        /// Number of iterations
        #[arg(short = 'n', long, default_value = "10")]
        iter: usize,
    },

    /// Convert TIFF to NumPy .npy format
    Tif2npy {
        /// Input TIFF file
        input: PathBuf,

        /// Output NPY file
        output: PathBuf,
    },

    /// Convert NumPy .npy to TIFF format
    Npy2tif {
        /// Input NPY file
        input: PathBuf,

        /// Output TIFF file
        output: PathBuf,
    },
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Deconvolve {
            image, psf, out, method, iter, maxiter, relerror, metric, format,
            scaling, bq, bg, offset, threads, verbose, cz, az, prefix,
            overwrite, start, tilesize, tilepad,
        } => {
            let mut opts = config::DwOpts::default();
            opts.image_file = image;
            opts.psf_file = psf;
            opts.out_file = out;
            opts.method = method;
            opts.metric = metric;
            opts.start_condition = start;
            opts.output_format = format;
            opts.border_quality = bq;
            opts.bg = bg;
            opts.offset = offset;
            opts.verbosity = verbose;
            opts.zcrop = cz;
            opts.auto_zcrop = az;
            opts.prefix = prefix;
            opts.overwrite = overwrite;
            opts.max_iter = maxiter;
            opts.rel_error = relerror;
            opts.tiling_padding = tilepad;

            if let Some(n) = iter {
                opts.n_iter = n;
                opts.iter_type = config::IterType::Fixed;
            }
            if let Some(s) = scaling {
                opts.scaling = s;
            }
            if let Some(t) = threads {
                opts.n_threads_omp = t;
                opts.n_threads_fft = t;
            }
            if let Some(ts) = tilesize {
                opts.tiling_max_size = ts as i64;
            }

            deconwolf::deconv::runner::dw_run::<RustFftBackend>(&opts)
        }

        Commands::Maxproj { input, output, slice, xyz, gm } => {
            let mode = if let Some(z) = slice {
                deconwolf::tools::maxproj::MaxProjMode::Slice(z)
            } else if xyz {
                deconwolf::tools::maxproj::MaxProjMode::Xyz
            } else if gm {
                deconwolf::tools::maxproj::MaxProjMode::GradientMagnitude
            } else {
                deconwolf::tools::maxproj::MaxProjMode::Max
            };
            deconwolf::tools::maxproj::run_maxproj(&input, &output, mode)
        }

        Commands::Merge { output, inputs } => {
            deconwolf::tools::merge::run_merge(&output, &inputs)
        }

        Commands::Background { out, sigma, inputs } => {
            deconwolf::tools::background::run_background(&out, &inputs, sigma)
        }

        Commands::Imshift { input, output, dx, dy, dz } => {
            deconwolf::tools::imshift::run_imshift(&input, &output, dx, dy, dz)
        }

        Commands::Dots { input, output, na, ni, lambda, dx, dz, ndots, csv } => {
            deconwolf::tools::dots::run_dots(&input, &output, na, ni, lambda, dx, dz, ndots, csv)
        }

        Commands::Psf {
            output, psf_type, na, ni, lambda, lambda2, dx, dz,
            size, nslice, lateral, axial, pinhole,
        } => {
            match psf_type.as_str() {
                "sted" => {
                    let lat = lateral.unwrap_or(2.0);
                    let ax = axial.unwrap_or(4.0);
                    match deconwolf::tools::psf::generate_sted_psf(lat, ax, size, nslice) {
                        Ok(img) => deconwolf::core::tiff_io::tiff_write_f32(&output, &img, None),
                        Err(e) => Err(e),
                    }
                }
                "confocal" => {
                    let na = na.unwrap_or(1.4);
                    let ni = ni.unwrap_or(1.515);
                    let lam = lambda.unwrap_or(525.0);
                    let lam2 = lambda2.unwrap_or(488.0);
                    let pixel_dx = dx.unwrap_or(65.0);
                    let pixel_dz = dz.unwrap_or(200.0);
                    match deconwolf::tools::psf::generate_confocal_psf(
                        na, ni, lam, lam2, pixel_dx, pixel_dz, size, nslice, pinhole,
                    ) {
                        Ok(img) => deconwolf::core::tiff_io::tiff_write_f32(&output, &img, None),
                        Err(e) => Err(e),
                    }
                }
                _ => {
                    // widefield (default)
                    let na = na.unwrap_or(1.4);
                    let ni = ni.unwrap_or(1.515);
                    let lam = lambda.unwrap_or(525.0);
                    let pixel_dx = dx.unwrap_or(65.0);
                    let pixel_dz = dz.unwrap_or(200.0);
                    match deconwolf::tools::psf::generate_widefield_psf(
                        na, ni, lam, pixel_dx, pixel_dz, size, nslice,
                    ) {
                        Ok(img) => deconwolf::core::tiff_io::tiff_write_f32(&output, &img, None),
                        Err(e) => Err(e),
                    }
                }
            }
        }

        Commands::Noise1 { input, output, lambda, lambda_s, iter } => {
            deconwolf::tools::sparse::run_sparse(&input, &output, lambda, lambda_s, iter)
        }

        Commands::Tif2npy { input, output } => {
            deconwolf::core::tiff_io::tiff_read(&input).and_then(|(img, _)| {
                deconwolf::core::npy_io::npy_write(&output, &img)
            })
        }

        Commands::Npy2tif { input, output } => {
            deconwolf::core::npy_io::npy_read(&input).and_then(|img| {
                deconwolf::core::tiff_io::tiff_write_f32(&output, &img, None)
            })
        }
    };

    if let Err(e) = result {
        log::error!("{}", e);
        std::process::exit(1);
    }
}
