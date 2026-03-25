use std::path::PathBuf;

/// Deconvolution method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Method {
    /// Richardson-Lucy
    Rl,
    /// Scaled Heavy Ball (default)
    Shb,
    /// Identity / pass-through (testing)
    Id,
}

/// Error metric for convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Metric {
    /// I-Divergence (default)
    Idiv,
    /// Mean Squared Error
    Mse,
}

/// Initial guess strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum StartCondition {
    /// Flat: average of input image
    Flat,
    /// Use input image as initial guess
    Identity,
    /// Low-pass filtered input
    LowPass,
}

/// Iteration stopping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterType {
    /// Stop after exactly N iterations.
    Fixed,
    /// Stop when relative error change < threshold.
    Relative,
    /// Stop when absolute error < threshold.
    Absolute,
}

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// 16-bit unsigned integer (auto-scaled)
    U16,
    /// 32-bit float
    F32,
}

/// Full deconvolution configuration.
#[derive(Debug, Clone)]
pub struct DwOpts {
    // Input/Output
    pub image_file: PathBuf,
    pub psf_file: PathBuf,
    pub out_file: Option<PathBuf>,
    pub prefix: String,
    pub overwrite: bool,

    // Method
    pub method: Method,
    pub metric: Metric,
    pub start_condition: StartCondition,

    // Iteration control
    pub iter_type: IterType,
    pub n_iter: usize,
    pub max_iter: usize,
    pub rel_error: f64,
    pub abs_error: f64,

    // Threading
    pub n_threads_fft: usize,
    pub n_threads_omp: usize,

    // Image preprocessing
    pub bg: f32,
    pub bg_auto: bool,
    pub offset: f32,
    pub psigma: f32,
    pub flatfield_file: Option<PathBuf>,
    pub zcrop: usize,
    pub auto_zcrop: usize,
    pub xy_crop_factor: f64,

    // Tiling
    pub tiling_max_size: i64,
    pub tiling_padding: usize,

    // Output
    pub output_format: OutputFormat,
    pub scaling: f32,
    pub border_quality: u8,

    // FFT
    pub fft_inplace: bool,
    pub fftw_planning: u32,

    // Advanced
    pub lookahead: usize,
    pub verbosity: u32,
    pub show_time: bool,
    pub iter_dump: bool,
    pub n_iter_dump: usize,
    pub ref_file: Option<PathBuf>,
    pub tsv_file: Option<PathBuf>,
    pub temp_folder: Option<PathBuf>,
    pub full_dump: bool,
    pub one_tile: bool,
    pub alpha_max: f32,

    // GPU
    pub gpu: bool,
    pub cl_device: usize,
}

impl Default for DwOpts {
    fn default() -> Self {
        Self {
            image_file: PathBuf::new(),
            psf_file: PathBuf::new(),
            out_file: None,
            prefix: "dw".to_string(),
            overwrite: false,

            method: Method::Shb,
            metric: Metric::Idiv,
            start_condition: StartCondition::Flat,

            iter_type: IterType::Relative,
            n_iter: 50,
            max_iter: 250,
            rel_error: 0.02,
            abs_error: 0.0,

            n_threads_fft: 0,
            n_threads_omp: 0,

            bg: 0.01,
            bg_auto: true,
            offset: 5.0,
            psigma: 0.0,
            flatfield_file: None,
            zcrop: 0,
            auto_zcrop: 0,
            xy_crop_factor: 0.001,

            tiling_max_size: -1,
            tiling_padding: 20,

            output_format: OutputFormat::U16,
            scaling: -1.0,
            border_quality: 2,

            fft_inplace: true,
            fftw_planning: 0,

            lookahead: 0,
            verbosity: 1,
            show_time: false,
            iter_dump: false,
            n_iter_dump: 0,
            ref_file: None,
            tsv_file: None,
            temp_folder: None,
            full_dump: false,
            one_tile: false,
            alpha_max: 1.0,

            gpu: false,
            cl_device: 0,
        }
    }
}

impl DwOpts {
    pub fn threads(&self) -> usize {
        if self.n_threads_omp > 0 {
            self.n_threads_omp
        } else {
            rayon::current_num_threads()
        }
    }

    pub fn fft_threads(&self) -> usize {
        if self.n_threads_fft > 0 {
            self.n_threads_fft
        } else {
            self.threads()
        }
    }
}
