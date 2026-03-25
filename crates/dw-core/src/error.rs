use thiserror::Error;

#[derive(Debug, Error)]
pub enum DwError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TIFF error: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("Invalid image dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Image has negative values")]
    NegativeValues,

    #[error("PSF not centered")]
    PsfNotCentered,

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("NPY error: {0}")]
    Npy(String),
}

pub type Result<T> = std::result::Result<T, DwError>;
