use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
        method: dw_deconv::config::Method,

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
        metric: dw_deconv::config::Metric,

        /// Output format
        #[arg(short = 'F', long, value_enum, default_value = "u16")]
        format: dw_deconv::config::OutputFormat,

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
        start: dw_deconv::config::StartCondition,

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
            image,
            psf,
            out,
            method,
            iter,
            maxiter,
            relerror,
            metric,
            format,
            scaling,
            bq,
            bg,
            offset,
            threads,
            verbose,
            cz,
            az,
            prefix,
            overwrite,
            start,
            tilesize,
            tilepad,
        } => {
            let mut opts = dw_deconv::config::DwOpts::default();
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
                opts.iter_type = dw_deconv::config::IterType::Fixed;
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

            #[cfg(feature = "fftw")]
            {
                dw_deconv::runner::dw_run::<dw_fft::fftw_backend::FftwBackend>(&opts)
            }
            #[cfg(not(feature = "fftw"))]
            {
                dw_deconv::runner::dw_run::<dw_fft::rustfft_backend::RustFftBackend>(&opts)
            }
        }

        Commands::Maxproj {
            input,
            output,
            slice,
        } => {
            let mode = if let Some(z) = slice {
                dw_tools::maxproj::MaxProjMode::Slice(z)
            } else {
                dw_tools::maxproj::MaxProjMode::Max
            };
            dw_tools::maxproj::run_maxproj(&input, &output, mode)
        }

        Commands::Tif2npy { input, output } => {
            dw_core::tiff_io::tiff_read(&input).and_then(|(img, _meta)| {
                dw_core::npy_io::npy_write(&output, &img)
            })
        }

        Commands::Npy2tif { input, output } => {
            dw_core::npy_io::npy_read(&input).and_then(|img| {
                dw_core::tiff_io::tiff_write_f32(&output, &img, None)
            })
        }
    };

    if let Err(e) = result {
        log::error!("{}", e);
        std::process::exit(1);
    }
}
