//! GPU acceleration via OpenCL.
//!
//! This module provides GPU-accelerated deconvolution methods using OpenCL.
//! Enable with the `gpu` feature flag and ensure OpenCL drivers are installed.
//!
//! Two methods are available:
//! - `ShbCl`: SHB with GPU FFT offloading (CPU handles ratio division)
//! - `ShbCl2`: Fully GPU-resident SHB (minimal CPU-GPU transfers)

pub mod kernels;

#[cfg(feature = "gpu")]
pub mod cl_env;
#[cfg(feature = "gpu")]
pub mod gpu_image;
#[cfg(feature = "gpu")]
pub mod shb_cl;
#[cfg(feature = "gpu")]
pub mod shb_cl2;
