//! OpenCL kernel source code embedded as string constants.
//!
//! These are the same kernels used in the C implementation,
//! compiled at runtime by the OpenCL driver.

/// Complex multiplication: C = A * B (interleaved complex format).
pub const COMPLEX_MUL: &str = r#"
__kernel void cl_complex_mul(
    const __global float *A,
    const __global float *B,
    __global float *C)
{
    size_t idx = get_global_id(0) * 2;
    float ar = A[idx], ai = A[idx+1];
    float br = B[idx], bi = B[idx+1];
    C[idx]   = ar*br - ai*bi;
    C[idx+1] = ar*bi + ai*br;
}
"#;

/// Complex conjugate multiplication: C = conj(A) * B.
pub const COMPLEX_MUL_CONJ: &str = r#"
__kernel void cl_complex_mul_conj(
    const __global float *A,
    const __global float *B,
    __global float *C)
{
    size_t idx = get_global_id(0) * 2;
    float ar = A[idx], ai = A[idx+1];
    float br = B[idx], bi = B[idx+1];
    C[idx]   = ar*br + ai*bi;
    C[idx+1] = ar*bi - ai*br;
}
"#;

/// In-place complex multiplication: B *= A.
pub const COMPLEX_MUL_INPLACE: &str = r#"
__kernel void cl_complex_mul_inplace(
    const __global float *A,
    __global float *B)
{
    size_t idx = get_global_id(0) * 2;
    float ar = A[idx], ai = A[idx+1];
    float br = B[idx], bi = B[idx+1];
    B[idx]   = ar*br - ai*bi;
    B[idx+1] = ar*bi + ai*br;
}
"#;

/// In-place conjugate multiplication: B *= conj(A).
pub const COMPLEX_MUL_CONJ_INPLACE: &str = r#"
__kernel void cl_complex_mul_conj_inplace(
    const __global float *A,
    __global float *B)
{
    size_t idx = get_global_id(0) * 2;
    float ar = A[idx], ai = A[idx+1];
    float br = B[idx], bi = B[idx+1];
    B[idx]   = ar*br + ai*bi;
    B[idx+1] = ar*bi - ai*br;
}
"#;

/// In-place real multiplication: B *= A (element-wise on real arrays).
pub const REAL_MUL_INPLACE: &str = r#"
__kernel void cl_real_mul_inplace(
    const __global float *A,
    __global float *B)
{
    size_t idx = get_global_id(0);
    B[idx] = A[idx] * B[idx];
}
"#;

/// Positivity constraint: A[i] = max(A[i], threshold).
pub const POSITIVITY: &str = r#"
__kernel void cl_positivity(
    __global float *A,
    const __global float *threshold)
{
    size_t idx = get_global_id(0);
    if (A[idx] < threshold[0]) {
        A[idx] = threshold[0];
    }
}
"#;

/// SHB momentum update: P = x + alpha * (x - xp).
pub const SHB_UPDATE: &str = r#"
__kernel void cl_shb_update(
    __global float *P,
    const __global float *x,
    const __global float *xp,
    const __global float *alpha)
{
    size_t idx = get_global_id(0);
    P[idx] = x[idx] + alpha[0] * (x[idx] - xp[idx]);
}
"#;

/// Zero out image frequencies where PSF magnitude is below threshold.
pub const PREPROCESS_IMAGE: &str = r#"
__kernel void cl_preprocess_image(
    __global float *fft_image,
    const __global float *fft_PSF,
    const __global float *value)
{
    size_t idx = get_global_id(0) * 2;
    float pr = fft_PSF[idx], pi = fft_PSF[idx+1];
    float el_norm = sqrt(pr*pr + pi*pi);
    if (el_norm < value[0]) {
        fft_image[idx]   = 0.0f;
        fft_image[idx+1] = 0.0f;
    }
}
"#;

/// I-divergence error computation with parallel reduction.
/// Compile-time defines: nPerItem, NELEMENTS, M0, M, N, P, wM, wN, wP.
pub const IDIV_KERNEL: &str = r#"
kernel void idiv_kernel(
    global const float *forward,
    global const float *image,
    local float *wgBuff,
    global float *partialSums)
{
    size_t lid = get_local_id(0);
    size_t wgSize = get_local_size(0);
    size_t wgNum = get_group_id(0);

    float mySum = 0.0f;
    size_t startIdx = (wgNum * wgSize + lid) * nPerItem;

    for (size_t kk = startIdx; kk < startIdx + nPerItem && kk < NELEMENTS; kk++) {
        size_t m = kk % M0;
        size_t n = (kk / M0) % wN;
        size_t p = kk / (M0 * wN);
        if (m < M && n < N && p < P) {
            size_t imIdx = p * N * M + n * M + m;
            float y = forward[kk];
            float z = image[imIdx];
            if (z > 0.0f && y > 0.0f) {
                mySum += z * log(z / y) - z + y;
            } else {
                mySum += y;
            }
        }
    }

    wgBuff[lid] = mySum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t s = wgSize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            wgBuff[lid] += wgBuff[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSums[wgNum] = wgBuff[0];
    }
}
"#;

/// Likelihood ratio: y = image / y, with boundary zeroing.
/// Compile-time defines: M0, M, N, P, wN.
pub const UPDATE_Y_KERNEL: &str = r#"
kernel void update_y_kernel(
    global float *y,
    global const float *image)
{
    size_t gid = get_global_id(0);
    size_t m = gid % M0;
    size_t n = (gid / M0) % wN;
    size_t p = gid / (M0 * wN);

    if (m < M && n < N && p < P) {
        size_t imIdx = p * N * M + n * M + m;
        float yval = y[gid];
        if (yval < 1e-6f) yval = 1e-6f;
        y[gid] = image[imIdx] / yval;
    } else {
        y[gid] = 0.0f;
    }
}
"#;
