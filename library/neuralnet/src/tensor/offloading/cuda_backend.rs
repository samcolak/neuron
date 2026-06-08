use crate::tensor::backend::{
    cpu_conv_block_backward_gradients,
    ConvBlockBackwardGradients,
    TensorBackend,
};
use crate::tensor::device::BackendTrainingCapabilities;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[cfg(feature = "offloading-cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "offloading-cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "offloading-cuda")]
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "offloading-cuda")]
use std::collections::HashMap;
#[cfg(feature = "offloading-cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "offloading-cuda")]
static CUDA_ALLOW_CPU_FALLBACK: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "offloading-cuda")]
fn cuda_allow_cpu_fallback() -> bool {
    *CUDA_ALLOW_CPU_FALLBACK.get_or_init(|| {
        std::env::var("NEURALNET_ALLOW_CPU_FALLBACK")
            .or_else(|_| std::env::var("NEURALNET_CUDA_ALLOW_CPU_FALLBACK"))
            .ok()
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on"
            })
            .unwrap_or(true)
    })
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CudaTensorBackend;

impl TensorBackend for CudaTensorBackend {

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn training_capabilities(&self) -> BackendTrainingCapabilities {
        BackendTrainingCapabilities::native_compute_host_training()
    }

    fn conv2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        cuda_conv2d_valid_fallback(input, kernels, bias, stride_h, stride_w)
    }

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        cuda_max_pool2d_fallback(input, window_h, window_w, stride_h, stride_w)
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        cuda_global_average_pool2d_fallback(input)
    }

    fn relu_inplace(&self, input: &mut Tensor4D) {
        cuda_relu_inplace_fallback(input)
    }

    fn conv_relu_max_pool2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        cuda_conv_relu_max_pool2d_valid_fallback(
            input,
            kernels,
            bias,
            conv_stride_h,
            conv_stride_w,
            pool_window_h,
            pool_window_w,
            pool_stride_h,
            pool_stride_w,
        )
    }

    fn conv_blocks_to_feature_vec(
        &self,
        input: &Tensor4D,
        block1_kernels: &Tensor4D,
        block1_bias: &[f32],
        block2: Option<(&Tensor4D, &[f32])>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    ) -> Result<Vec<f32>, TensorError> {
        cuda_conv_blocks_to_feature_vec_fallback(
            input,
            block1_kernels,
            block1_bias,
            block2,
            conv_stride_h,
            conv_stride_w,
            pool_window_h,
            pool_window_w,
            pool_stride_h,
            pool_stride_w,
        )
    }

    fn conv_block_backward_gradients(
        &self,
        kernels: &Tensor4D,
        input: &Tensor4D,
        conv_pre_activation: &Tensor4D,
        pool_indices: &[(usize, usize)],
        pooled_shape: (usize, usize, usize, usize),
        pooled_grad: &Tensor4D,
        compute_input_grad: bool,
    ) -> Result<ConvBlockBackwardGradients, TensorError> {
        cuda_conv_block_backward_gradients_fallback(
            kernels,
            input,
            conv_pre_activation,
            pool_indices,
            pooled_shape,
            pooled_grad,
            compute_input_grad,
        )
    }

}

#[allow(clippy::too_many_arguments)]
fn cuda_conv_block_backward_gradients_fallback(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_pre_activation: &Tensor4D,
    pool_indices: &[(usize, usize)],
    pooled_shape: (usize, usize, usize, usize),
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackwardGradients, TensorError> {
    #[cfg(feature = "offloading-cuda")]
    {
        let native_result = if pooled_shape.0 > 1 {
            cuda_conv_block_backward_gradients_native_batched_by_sample(
                kernels,
                input,
                conv_pre_activation,
                pool_indices,
                pooled_shape,
                pooled_grad,
                compute_input_grad,
            )
        } else {
            cuda_conv_block_backward_gradients_kernel(
                kernels,
                input,
                conv_pre_activation,
                pool_indices,
                pooled_shape,
                pooled_grad,
                compute_input_grad,
            )
        };

        if let Ok(result) = native_result {
            return Ok(result);
        }

        if !cuda_allow_cpu_fallback() {
            return native_result;
        }
    }

    cpu_conv_block_backward_gradients(
        kernels,
        input,
        conv_pre_activation,
        pool_indices,
        pooled_shape,
        pooled_grad,
        compute_input_grad,
    )
}

#[cfg(feature = "offloading-cuda")]
fn tensor4d_sample(input: &Tensor4D, sample_idx: usize) -> Result<Tensor4D, TensorError> {
    let (n, c, h, w) = input.shape();
    if sample_idx >= n {
        return Err(TensorError::InvalidArgument(
            "sample index out of bounds while splitting batched tensor",
        ));
    }

    let per_sample = c
        .checked_mul(h)
        .and_then(|v| v.checked_mul(w))
        .ok_or(TensorError::InvalidArgument("sample shape overflow"))?;
    let start = sample_idx
        .checked_mul(per_sample)
        .ok_or(TensorError::InvalidArgument("sample offset overflow"))?;
    let end = start
        .checked_add(per_sample)
        .ok_or(TensorError::InvalidArgument("sample end overflow"))?;

    Tensor4D::from_vec(1, c, h, w, input.as_slice()[start..end].to_vec())
}

#[cfg(feature = "offloading-cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_conv_block_backward_gradients_native_batched_by_sample(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_pre_activation: &Tensor4D,
    pool_indices: &[(usize, usize)],
    pooled_shape: (usize, usize, usize, usize),
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackwardGradients, TensorError> {
    let (batch, channels, pooled_h, pooled_w) = pooled_shape;
    let pooled_grad_shape = pooled_grad.shape();
    if pooled_grad_shape != pooled_shape {
        return Err(TensorError::IncompatibleShapes {
            left: pooled_shape,
            right: pooled_grad_shape,
        });
    }

    let (input_batch, input_channels, in_h, in_w) = input.shape();
    let (conv_batch, conv_channels, relu_h, relu_w) = conv_pre_activation.shape();
    if input_batch != batch || conv_batch != batch || conv_channels != channels {
        return Err(TensorError::IncompatibleShapes {
            left: pooled_shape,
            right: conv_pre_activation.shape(),
        });
    }

    let per_sample_indices = channels
        .checked_mul(pooled_h)
        .and_then(|v| v.checked_mul(pooled_w))
        .ok_or(TensorError::InvalidArgument("pool index shape overflow"))?;
    let expected_pool_indices = batch
        .checked_mul(per_sample_indices)
        .ok_or(TensorError::InvalidArgument("pool index shape overflow"))?;
    if pool_indices.len() != expected_pool_indices {
        return Err(TensorError::ShapeMismatch {
            expected: expected_pool_indices,
            actual: pool_indices.len(),
        });
    }

    let per_sample_input = input_channels
        .checked_mul(in_h)
        .and_then(|v| v.checked_mul(in_w))
        .ok_or(TensorError::InvalidArgument("input sample shape overflow"))?;

    let mut kernel_grad = Tensor4D::zeros(
        kernels.shape().0,
        kernels.shape().1,
        kernels.shape().2,
        kernels.shape().3,
    );
    let mut bias_grad = vec![0.0f32; channels];
    let mut input_grad = if compute_input_grad {
        Some(Tensor4D::zeros(batch, input_channels, in_h, in_w))
    } else {
        None
    };

    for sample_idx in 0..batch {
        let input_sample = tensor4d_sample(input, sample_idx)?;
        let conv_pre_sample = tensor4d_sample(conv_pre_activation, sample_idx)?;
        let pooled_grad_sample = tensor4d_sample(pooled_grad, sample_idx)?;

        let idx_start = sample_idx
            .checked_mul(per_sample_indices)
            .ok_or(TensorError::InvalidArgument("pool index offset overflow"))?;
        let idx_end = idx_start
            .checked_add(per_sample_indices)
            .ok_or(TensorError::InvalidArgument("pool index end overflow"))?;
        let sample_pool_indices = &pool_indices[idx_start..idx_end];

        let sample_backward = cuda_conv_block_backward_gradients_kernel(
            kernels,
            &input_sample,
            &conv_pre_sample,
            sample_pool_indices,
            (1, channels, pooled_h, pooled_w),
            &pooled_grad_sample,
            compute_input_grad,
        )?;

        kernel_grad.add_inplace(&sample_backward.kernel_grad)?;
        for (accum, grad) in bias_grad.iter_mut().zip(sample_backward.bias_grad.iter()) {
            *accum += *grad;
        }

        if let (Some(total_input_grad), Some(sample_input_grad)) =
            (input_grad.as_mut(), sample_backward.input_grad.as_ref())
        {
            let start = sample_idx
                .checked_mul(per_sample_input)
                .ok_or(TensorError::InvalidArgument("input grad offset overflow"))?;
            let end = start
                .checked_add(per_sample_input)
                .ok_or(TensorError::InvalidArgument("input grad end overflow"))?;
            total_input_grad.as_mut_slice()[start..end]
                .copy_from_slice(sample_input_grad.as_slice());
        }
    }

    let _ = (relu_h, relu_w);

    Ok(ConvBlockBackwardGradients {
        kernel_grad,
        bias_grad,
        input_grad,
    })
}

pub fn cuda_backend() -> CudaTensorBackend {
    CudaTensorBackend
}

pub fn cuda_backend_available() -> bool {
    cfg!(feature = "offloading-cuda")
}

// Staged CUDA support:
// - Current implementation keeps behavior correct by using CPU kernels.
// - Real CUDA kernels can replace these call sites without touching trait users.
fn cuda_conv2d_valid_fallback(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-cuda")]
    {
        if let Ok(result) = cuda_conv2d_valid_kernel(input, kernels, bias, stride_h, stride_w) {
            return Ok(result);
        }
    }

    input.conv2d_valid_cpu(kernels, bias, stride_h, stride_w)
}

fn cuda_max_pool2d_fallback(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-cuda")]
    {
        if let Ok(result) = cuda_max_pool2d_kernel(input, window_h, window_w, stride_h, stride_w) {
            return Ok(result);
        }
    }

    input.max_pool2d_cpu(window_h, window_w, stride_h, stride_w)
}

fn cuda_global_average_pool2d_fallback(input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-cuda")]
    {
        if let Ok(result) = cuda_global_average_pool2d_kernel(input) {
            return Ok(result);
        }
    }

    input.global_average_pool2d_cpu()
}

fn cuda_relu_inplace_fallback(input: &mut Tensor4D) {
    #[cfg(feature = "offloading-cuda")]
    {
        if cuda_relu_inplace_kernel(input).is_ok() {
            return;
        }
    }

    input.relu_inplace_cpu();
}

#[cfg(feature = "offloading-cuda")]
struct CudaKernelContext {
    device: Arc<CudaDevice>,
    buffer_pool: Mutex<CudaDeviceBufferPool>,
}

#[cfg(feature = "offloading-cuda")]
#[derive(Default)]
struct CudaDeviceBufferPool {
    f32_buffers: HashMap<&'static str, CudaSlice<f32>>,
    u32_buffers: HashMap<&'static str, CudaSlice<u32>>,
}

#[cfg(feature = "offloading-cuda")]
impl CudaDeviceBufferPool {
    fn take_f32(
        &mut self,
        device: &Arc<CudaDevice>,
        key: &'static str,
        len: usize,
    ) -> Result<CudaSlice<f32>, TensorError> {
        if let Some(existing) = self.f32_buffers.remove(key)
            && existing.len() == len
        {
            return Ok(existing);
        }

        device
            .alloc_zeros::<f32>(len)
            .map_err(|_| TensorError::InvalidArgument("failed to allocate pooled CUDA f32 buffer"))
    }

    fn take_u32(
        &mut self,
        device: &Arc<CudaDevice>,
        key: &'static str,
        len: usize,
    ) -> Result<CudaSlice<u32>, TensorError> {
        if let Some(existing) = self.u32_buffers.remove(key)
            && existing.len() == len
        {
            return Ok(existing);
        }

        device
            .alloc_zeros::<u32>(len)
            .map_err(|_| TensorError::InvalidArgument("failed to allocate pooled CUDA u32 buffer"))
    }

    fn put_f32(&mut self, key: &'static str, buffer: CudaSlice<f32>) {
        self.f32_buffers.insert(key, buffer);
    }

    fn put_u32(&mut self, key: &'static str, buffer: CudaSlice<u32>) {
        self.u32_buffers.insert(key, buffer);
    }
}

#[cfg(feature = "offloading-cuda")]
static CUDA_KERNEL_CONTEXT: OnceLock<Result<CudaKernelContext, String>> = OnceLock::new();

#[cfg(feature = "offloading-cuda")]
fn cuda_runtime_supported_on_platform() -> bool {
    cfg!(any(target_os = "linux", target_os = "windows"))
}

#[cfg(feature = "offloading-cuda")]
const CUDA_KERNEL_MODULE: &str = "neuralnet_tensor_module";

#[cfg(feature = "offloading-cuda")]
const CUDA_RELU_KERNEL: &str = "relu_inplace_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_CONV_VALID_KERNEL: &str = "conv2d_valid_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_MAX_POOL2D_VALID_KERNEL: &str = "max_pool2d_valid_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_GAP2D_KERNEL: &str = "global_average_pool2d_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_UNPOOL_RELU_GRAD_KERNEL: &str = "unpool_relu_grad_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_BIAS_GRAD_KERNEL: &str = "conv_bias_grad_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_KERNEL_GRAD_KERNEL: &str = "conv_kernel_grad_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_INPUT_GRAD_KERNEL: &str = "conv_input_grad_nchw_kernel";

#[cfg(feature = "offloading-cuda")]
const CUDA_RELU_SRC: &str = r#"
extern "C" __global__
void relu_inplace_kernel(float* input, unsigned int len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float value = input[idx];
        input[idx] = value > 0.0f ? value : 0.0f;
    }
}

extern "C" __global__
void conv2d_valid_nchw_kernel(
    const float* input,
    const float* kernels,
    const float* bias,
    const unsigned int* params,
    float* output
) {
    unsigned int has_bias = params[0];
    unsigned int batch = params[1];
    unsigned int in_channels = params[2];
    unsigned int in_h = params[3];
    unsigned int in_w = params[4];
    unsigned int out_channels = params[5];
    unsigned int kernel_h = params[6];
    unsigned int kernel_w = params[7];
    unsigned int stride_h = params[8];
    unsigned int stride_w = params[9];
    unsigned int out_h = params[10];
    unsigned int out_w = params[11];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * out_channels * out_h * out_w;
    if (idx >= total) {
        return;
    }

    unsigned int ox = idx % out_w;
    unsigned int oy = (idx / out_w) % out_h;
    unsigned int oc = (idx / (out_w * out_h)) % out_channels;
    unsigned int b = idx / (out_w * out_h * out_channels);

    unsigned int in_y = oy * stride_h;
    unsigned int in_x = ox * stride_w;

    float acc = has_bias ? bias[oc] : 0.0f;

    for (unsigned int ic = 0; ic < in_channels; ++ic) {
        for (unsigned int ky = 0; ky < kernel_h; ++ky) {
            for (unsigned int kx = 0; kx < kernel_w; ++kx) {
                unsigned int input_idx =
                    (((b * in_channels + ic) * in_h + (in_y + ky)) * in_w) + (in_x + kx);
                unsigned int kernel_idx =
                    (((oc * in_channels + ic) * kernel_h + ky) * kernel_w) + kx;
                acc += input[input_idx] * kernels[kernel_idx];
            }
        }
    }

    output[idx] = acc;
}

extern "C" __global__
void max_pool2d_valid_nchw_kernel(
    const float* input,
    const unsigned int* params,
    float* output
) {
    unsigned int batch = params[0];
    unsigned int channels = params[1];
    unsigned int in_h = params[2];
    unsigned int in_w = params[3];
    unsigned int window_h = params[4];
    unsigned int window_w = params[5];
    unsigned int stride_h = params[6];
    unsigned int stride_w = params[7];
    unsigned int out_h = params[8];
    unsigned int out_w = params[9];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * channels * out_h * out_w;
    if (idx >= total) {
        return;
    }

    unsigned int ox = idx % out_w;
    unsigned int oy = (idx / out_w) % out_h;
    unsigned int c = (idx / (out_w * out_h)) % channels;
    unsigned int b = idx / (out_w * out_h * channels);

    unsigned int in_y = oy * stride_h;
    unsigned int in_x = ox * stride_w;

    float max_value = -3.402823466e+38f;

    for (unsigned int wy = 0; wy < window_h; ++wy) {
        for (unsigned int wx = 0; wx < window_w; ++wx) {
            unsigned int input_idx =
                (((b * channels + c) * in_h + (in_y + wy)) * in_w) + (in_x + wx);
            float value = input[input_idx];
            if (value > max_value) {
                max_value = value;
            }
        }
    }

    output[idx] = max_value;
}

extern "C" __global__
void global_average_pool2d_nchw_kernel(
    const float* input,
    const unsigned int* params,
    float* output
) {
    unsigned int batch = params[0];
    unsigned int channels = params[1];
    unsigned int h = params[2];
    unsigned int w = params[3];
    unsigned int spatial_area = params[4];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * channels;
    if (idx >= total) {
        return;
    }

    unsigned int c = idx % channels;
    unsigned int b = idx / channels;
    unsigned int base = ((b * channels + c) * h) * w;

    float sum = 0.0f;
    for (unsigned int i = 0; i < spatial_area; ++i) {
        sum += input[base + i];
    }

    output[idx] = sum / (float)spatial_area;
}

extern "C" __global__
void unpool_relu_grad_nchw_kernel(
    const float* pooled_grad,
    const float* conv_pre_activation,
    const unsigned int* pool_indices,
    const unsigned int* params,
    float* conv_grad
) {
    unsigned int channels = params[0];
    unsigned int pooled_h = params[1];
    unsigned int pooled_w = params[2];
    unsigned int relu_h = params[3];
    unsigned int relu_w = params[4];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = channels * pooled_h * pooled_w;
    if (idx >= total) {
        return;
    }

    unsigned int packed = pool_indices[idx];
    unsigned int src_y = packed / relu_w;
    unsigned int src_x = packed % relu_w;
    if (src_y >= relu_h || src_x >= relu_w) {
        return;
    }

    unsigned int conv_idx = ((c * relu_h) + src_y) * relu_w + src_x;
    float grad = pooled_grad[idx];
    if (conv_pre_activation[conv_idx] <= 0.0f) {
        return;
    }

    atomicAdd(&conv_grad[conv_idx], grad);
}

extern "C" __global__
void conv_bias_grad_nchw_kernel(
    const float* conv_grad,
    const unsigned int* params,
    float* bias_grad
) {
    unsigned int channels = params[0];
    unsigned int conv_h = params[1];
    unsigned int conv_w = params[2];

    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    unsigned int plane = conv_h * conv_w;
    unsigned int base = c * plane;
    float sum = 0.0f;
    for (unsigned int i = 0; i < plane; ++i) {
        sum += conv_grad[base + i];
    }
    bias_grad[c] = sum;
}

extern "C" __global__
void conv_kernel_grad_nchw_kernel(
    const float* conv_grad,
    const float* input,
    const unsigned int* params,
    float* kernel_grad
) {
    unsigned int channels = params[0];
    unsigned int in_channels = params[1];
    unsigned int conv_h = params[2];
    unsigned int conv_w = params[3];
    unsigned int in_h = params[4];
    unsigned int in_w = params[5];
    unsigned int kernel_h = params[6];
    unsigned int kernel_w = params[7];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = channels * in_channels * kernel_h * kernel_w;
    if (idx >= total) {
        return;
    }

    unsigned int kx = idx % kernel_w;
    unsigned int ky = (idx / kernel_w) % kernel_h;
    unsigned int ic = (idx / (kernel_w * kernel_h)) % in_channels;
    unsigned int oc = idx / (kernel_w * kernel_h * in_channels);

    float sum = 0.0f;
    for (unsigned int oy = 0; oy < conv_h; ++oy) {
        unsigned int conv_row_base = (oc * conv_h + oy) * conv_w;
        unsigned int input_row_base = (ic * in_h + (oy + ky)) * in_w + kx;
        for (unsigned int ox = 0; ox < conv_w; ++ox) {
            float grad = conv_grad[conv_row_base + ox];
            float inp = input[input_row_base + ox];
            sum += grad * inp;
        }
    }

    kernel_grad[idx] = sum;
}

extern "C" __global__
void conv_input_grad_nchw_kernel(
    const float* conv_grad,
    const float* kernels,
    const unsigned int* params,
    float* input_grad
) {
    unsigned int channels = params[0];
    unsigned int in_channels = params[1];
    unsigned int conv_h = params[2];
    unsigned int conv_w = params[3];
    unsigned int in_h = params[4];
    unsigned int in_w = params[5];
    unsigned int kernel_h = params[6];
    unsigned int kernel_w = params[7];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = in_channels * in_h * in_w;
    if (idx >= total) {
        return;
    }

    unsigned int ix = idx % in_w;
    unsigned int iy = (idx / in_w) % in_h;
    unsigned int ic = idx / (in_w * in_h);

    float sum = 0.0f;
    for (unsigned int oc = 0; oc < channels; ++oc) {
        unsigned int kernel_channel_base = (oc * in_channels + ic) * kernel_h * kernel_w;
        for (unsigned int ky = 0; ky < kernel_h; ++ky) {
            if (iy < ky) {
                continue;
            }
            unsigned int oy = iy - ky;
            if (oy >= conv_h) {
                continue;
            }
            unsigned int conv_row_base = (oc * conv_h + oy) * conv_w;

            for (unsigned int kx = 0; kx < kernel_w; ++kx) {
                if (ix < kx) {
                    continue;
                }
                unsigned int ox = ix - kx;
                if (ox >= conv_w) {
                    continue;
                }
                float grad = conv_grad[conv_row_base + ox];
                float weight = kernels[kernel_channel_base + ky * kernel_w + kx];
                sum += grad * weight;
            }
        }
    }

    input_grad[idx] = sum;
}
"#;

#[cfg(feature = "offloading-cuda")]
fn cuda_kernel_context() -> Result<&'static CudaKernelContext, TensorError> {

    let context_result = CUDA_KERNEL_CONTEXT.get_or_init(|| {
        if !cuda_runtime_supported_on_platform() {
            return Err("CUDA runtime is not supported on this platform".to_string());
        }

        let device = panic::catch_unwind(AssertUnwindSafe(|| CudaDevice::new(0)))
            .map_err(|_| "CUDA runtime initialization panicked".to_string())?
            .map_err(|err| {
                format!("failed to initialize CUDA device 0 for neuralnet kernels: {err:?}")
            })?;

        let ptx = compile_ptx(CUDA_RELU_SRC)
            .map_err(|err| format!("failed to compile CUDA relu kernel PTX: {err:?}"))?;

        device
            .load_ptx(
                ptx,
                CUDA_KERNEL_MODULE,
                &[
                    CUDA_RELU_KERNEL,
                    CUDA_CONV_VALID_KERNEL,
                    CUDA_MAX_POOL2D_VALID_KERNEL,
                    CUDA_GAP2D_KERNEL,
                    CUDA_UNPOOL_RELU_GRAD_KERNEL,
                    CUDA_BIAS_GRAD_KERNEL,
                    CUDA_KERNEL_GRAD_KERNEL,
                    CUDA_INPUT_GRAD_KERNEL,
                ],
            )
            .map_err(|err| format!("failed to load CUDA tensor kernels: {err:?}"))?;

        Ok(CudaKernelContext {
            device,
            buffer_pool: Mutex::new(CudaDeviceBufferPool::default()),
        })
    });

    context_result
        .as_ref()
        .map_err(|_| TensorError::InvalidArgument("failed to initialize CUDA kernel context"))

}

#[cfg(feature = "offloading-cuda")]
fn cuda_relu_inplace_kernel(input: &mut Tensor4D) -> Result<(), TensorError> {

    if input.is_empty() {
        return Ok(());
    }

    let context = cuda_kernel_context()?;
    let host = input.as_slice().to_vec();
    let mut device_input = context
        .device
        .htod_copy(host)
        .map_err(|_| TensorError::InvalidArgument("failed to copy tensor to CUDA device"))?;

    let launch = LaunchConfig::for_num_elems(input.len() as u32);
    let kernel = context
        .device
        .get_func(CUDA_KERNEL_MODULE, CUDA_RELU_KERNEL)
        .ok_or(TensorError::InvalidArgument(
            "failed to get CUDA relu kernel function",
        ))?;

    unsafe {
        kernel
            .launch(launch, (&mut device_input, input.len() as u32))
            .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA relu kernel"))?;
    }

    let output: Vec<f32> = context
        .device
        .dtoh_sync_copy(&device_input)
        .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA tensor to host"))?;
    input.as_mut_slice().copy_from_slice(output.as_slice());
    Ok(())

}

#[cfg(feature = "offloading-cuda")]
fn cuda_conv2d_valid_kernel(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("stride must be greater than zero"));
    }

    let (batch, in_channels, in_h, in_w) = input.shape();
    let (out_channels, kernel_in_channels, kernel_h, kernel_w) = kernels.shape();

    if in_channels != kernel_in_channels {
        return Err(TensorError::IncompatibleShapes {
            left: input.shape(),
            right: kernels.shape(),
        });
    }

    if out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
        return Err(TensorError::InvalidArgument(
            "kernel shape must have non-zero output channels and spatial size",
        ));
    }

    if in_h < kernel_h || in_w < kernel_w {
        return Err(TensorError::InvalidArgument(
            "kernel spatial size cannot exceed input spatial size",
        ));
    }

    if let Some(bias_values) = bias
        && bias_values.len() != out_channels
    {
        return Err(TensorError::ShapeMismatch {
            expected: out_channels,
            actual: bias_values.len(),
        });
    }

    let out_h = ((in_h - kernel_h) / stride_h) + 1;
    let out_w = ((in_w - kernel_w) / stride_w) + 1;
    let output_len = batch
        .checked_mul(out_channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or(TensorError::InvalidArgument("output shape overflow"))?;

    if output_len == 0 {
        return Ok(Tensor4D::zeros(batch, out_channels, out_h, out_w));
    }

    let context = cuda_kernel_context()?;

    let device_input = context
        .device
        .htod_copy(input.as_slice().to_vec())
        .map_err(|_| TensorError::InvalidArgument("failed to copy input tensor to CUDA device"))?;

    let device_kernels = context
        .device
        .htod_copy(kernels.as_slice().to_vec())
        .map_err(|_| TensorError::InvalidArgument("failed to copy kernel tensor to CUDA device"))?;

    let (host_bias, has_bias) = if let Some(values) = bias {
        (values.to_vec(), 1u32)
    } else {
        (vec![0.0f32; out_channels], 0u32)
    };

    let device_bias = context
        .device
        .htod_copy(host_bias)
        .map_err(|_| TensorError::InvalidArgument("failed to copy bias tensor to CUDA device"))?;

    let device_params = context
        .device
        .htod_copy(vec![
            has_bias,
            batch as u32,
            in_channels as u32,
            in_h as u32,
            in_w as u32,
            out_channels as u32,
            kernel_h as u32,
            kernel_w as u32,
            stride_h as u32,
            stride_w as u32,
            out_h as u32,
            out_w as u32,
        ])
        .map_err(|_| TensorError::InvalidArgument("failed to copy conv params to CUDA device"))?;

    let mut device_output = context
        .device
        .htod_copy(vec![0.0f32; output_len])
        .map_err(|_| TensorError::InvalidArgument("failed to allocate CUDA output tensor"))?;

    let kernel = context
        .device
        .get_func(CUDA_KERNEL_MODULE, CUDA_CONV_VALID_KERNEL)
        .ok_or(TensorError::InvalidArgument(
            "failed to get CUDA conv2d kernel function",
        ))?;

    let launch = LaunchConfig::for_num_elems(output_len as u32);
    unsafe {
        kernel
            .launch(
                launch,
                (
                    &device_input,
                    &device_kernels,
                    &device_bias,
                    &device_params,
                    &mut device_output,
                ),
            )
            .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA conv2d kernel"))?;
    }

    let output: Vec<f32> = context
        .device
        .dtoh_sync_copy(&device_output)
        .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA output tensor to host"))?;

    Tensor4D::from_vec(batch, out_channels, out_h, out_w, output)
    
}

#[cfg(feature = "offloading-cuda")]
fn cuda_max_pool2d_kernel(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {

    if window_h == 0 || window_w == 0 {
        return Err(TensorError::InvalidArgument(
            "pooling window must be greater than zero",
        ));
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("stride must be greater than zero"));
    }

    let (batch, channels, in_h, in_w) = input.shape();
    if in_h < window_h || in_w < window_w {
        return Err(TensorError::InvalidArgument(
            "pooling window cannot exceed input spatial size",
        ));
    }

    let out_h = ((in_h - window_h) / stride_h) + 1;
    let out_w = ((in_w - window_w) / stride_w) + 1;
    let output_len = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or(TensorError::InvalidArgument("output shape overflow"))?;

    if output_len == 0 {
        return Ok(Tensor4D::zeros(batch, channels, out_h, out_w));
    }

    let context = cuda_kernel_context()?;

    let device_input = context
        .device
        .htod_copy(input.as_slice().to_vec())
        .map_err(|_| TensorError::InvalidArgument("failed to copy input tensor to CUDA device"))?;

    let device_params = context
        .device
        .htod_copy(vec![
            batch as u32,
            channels as u32,
            in_h as u32,
            in_w as u32,
            window_h as u32,
            window_w as u32,
            stride_h as u32,
            stride_w as u32,
            out_h as u32,
            out_w as u32,
        ])
        .map_err(|_| TensorError::InvalidArgument("failed to copy max_pool params to CUDA device"))?;

    let mut device_output = context
        .device
        .htod_copy(vec![0.0f32; output_len])
        .map_err(|_| TensorError::InvalidArgument("failed to allocate CUDA max_pool output"))?;

    let kernel = context
        .device
        .get_func(CUDA_KERNEL_MODULE, CUDA_MAX_POOL2D_VALID_KERNEL)
        .ok_or(TensorError::InvalidArgument(
            "failed to get CUDA max_pool2d kernel function",
        ))?;

    let launch = LaunchConfig::for_num_elems(output_len as u32);
    unsafe {
        kernel
            .launch(launch, (&device_input, &device_params, &mut device_output))
            .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA max_pool2d kernel"))?;
    }

    let output: Vec<f32> = context
        .device
        .dtoh_sync_copy(&device_output)
        .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA max_pool output to host"))?;

    Tensor4D::from_vec(batch, channels, out_h, out_w, output)

}

#[cfg(feature = "offloading-cuda")]
fn cuda_global_average_pool2d_kernel(input: &Tensor4D) -> Result<Tensor4D, TensorError> {

    let (batch, channels, h, w) = input.shape();
    if h == 0 || w == 0 {
        return Err(TensorError::InvalidArgument(
            "global average pooling requires non-zero spatial dimensions",
        ));
    }

    let output_len = batch
        .checked_mul(channels)
        .ok_or(TensorError::InvalidArgument("output shape overflow"))?;

    if output_len == 0 {
        return Ok(Tensor4D::zeros(batch, channels, 1, 1));
    }

    let context = cuda_kernel_context()?;

    let device_input = context
        .device
        .htod_copy(input.as_slice().to_vec())
        .map_err(|_| TensorError::InvalidArgument("failed to copy input tensor to CUDA device"))?;

    let device_params = context
        .device
        .htod_copy(vec![batch as u32, channels as u32, h as u32, w as u32, (h * w) as u32])
        .map_err(|_| TensorError::InvalidArgument("failed to copy GAP params to CUDA device"))?;

    let mut device_output = context
        .device
        .htod_copy(vec![0.0f32; output_len])
        .map_err(|_| TensorError::InvalidArgument("failed to allocate CUDA GAP output"))?;

    let kernel = context
        .device
        .get_func(CUDA_KERNEL_MODULE, CUDA_GAP2D_KERNEL)
        .ok_or(TensorError::InvalidArgument(
            "failed to get CUDA global average pooling kernel function",
        ))?;

    let launch = LaunchConfig::for_num_elems(output_len as u32);
    unsafe {
        kernel
            .launch(launch, (&device_input, &device_params, &mut device_output))
            .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA GAP kernel"))?;
    }

    let output: Vec<f32> = context
        .device
        .dtoh_sync_copy(&device_output)
        .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA GAP output to host"))?;

    Tensor4D::from_vec(batch, channels, 1, 1, output)

}

#[allow(clippy::too_many_arguments)]
fn cuda_conv_relu_max_pool2d_valid_fallback(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    conv_stride_h: usize,
    conv_stride_w: usize,
    pool_window_h: usize,
    pool_window_w: usize,
    pool_stride_h: usize,
    pool_stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    let mut conv = cuda_conv2d_valid_fallback(input, kernels, bias, conv_stride_h, conv_stride_w)?;
    cuda_relu_inplace_fallback(&mut conv);
    cuda_max_pool2d_fallback(&conv, pool_window_h, pool_window_w, pool_stride_h, pool_stride_w)
}

#[allow(clippy::too_many_arguments)]
fn cuda_conv_blocks_to_feature_vec_fallback(
    input: &Tensor4D,
    block1_kernels: &Tensor4D,
    block1_bias: &[f32],
    block2: Option<(&Tensor4D, &[f32])>,
    conv_stride_h: usize,
    conv_stride_w: usize,
    pool_window_h: usize,
    pool_window_w: usize,
    pool_stride_h: usize,
    pool_stride_w: usize,
) -> Result<Vec<f32>, TensorError> {
    let block1_out = cuda_conv_relu_max_pool2d_valid_fallback(
        input,
        block1_kernels,
        Some(block1_bias),
        conv_stride_h,
        conv_stride_w,
        pool_window_h,
        pool_window_w,
        pool_stride_h,
        pool_stride_w,
    )?;

    let final_out = if let Some((k2, b2)) = block2 {
        cuda_conv_relu_max_pool2d_valid_fallback(
            &block1_out,
            k2,
            Some(b2),
            conv_stride_h,
            conv_stride_w,
            pool_window_h,
            pool_window_w,
            pool_stride_h,
            pool_stride_w,
        )?
    } else {
        block1_out
    };

    let gap = cuda_global_average_pool2d_fallback(&final_out)?;
    Ok(gap.first_sample_features())

}

#[cfg(feature = "offloading-cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_conv_block_backward_gradients_kernel(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_pre_activation: &Tensor4D,
    pool_indices: &[(usize, usize)],
    pooled_shape: (usize, usize, usize, usize),
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackwardGradients, TensorError> {
    let pooled_grad_shape = pooled_grad.shape();
    if pooled_grad_shape != pooled_shape {
        return Err(TensorError::IncompatibleShapes {
            left: pooled_shape,
            right: pooled_grad_shape,
        });
    }

    if pooled_shape.0 != 1 {
        return Err(TensorError::InvalidArgument(
            "cuda conv block backward currently supports batch size 1",
        ));
    }

    let (_, channels, pooled_h, pooled_w) = pooled_shape;
    let (_, _, relu_h, relu_w) = conv_pre_activation.shape();

    let expected_pool_indices = channels
        .checked_mul(pooled_h)
        .and_then(|v| v.checked_mul(pooled_w))
        .ok_or(TensorError::InvalidArgument("pool index shape overflow"))?;

    if pool_indices.len() != expected_pool_indices {
        return Err(TensorError::ShapeMismatch {
            expected: expected_pool_indices,
            actual: pool_indices.len(),
        });
    }

    let conv_grad_len = channels
        .checked_mul(relu_h)
        .and_then(|v| v.checked_mul(relu_w))
        .ok_or(TensorError::InvalidArgument("conv grad shape overflow"))?;

    let context = cuda_kernel_context()?;

    let mut pooled_grad_device;
    let mut conv_pre_device;
    let mut conv_grad_device;
    let mut pool_indices_device;

    {
        let mut pool = context
            .buffer_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;

        pooled_grad_device = pool.take_f32(
            &context.device,
            "backprop_pooled_grad",
            pooled_grad.len(),
        )?;
        conv_pre_device = pool.take_f32(
            &context.device,
            "backprop_conv_pre",
            conv_pre_activation.len(),
        )?;
        conv_grad_device = pool.take_f32(&context.device, "backprop_conv_grad", conv_grad_len)?;
        pool_indices_device = pool.take_u32(
            &context.device,
            "backprop_pool_indices",
            expected_pool_indices,
        )?;
    }

    context
        .device
        .htod_copy_into(pooled_grad.as_slice().to_vec(), &mut pooled_grad_device)
        .map_err(|_| TensorError::InvalidArgument("failed to copy pooled_grad to CUDA device"))?;

    context
        .device
        .htod_copy_into(conv_pre_activation.as_slice().to_vec(), &mut conv_pre_device)
        .map_err(|_| TensorError::InvalidArgument("failed to copy conv_pre_activation to CUDA device"))?;

    let mut packed_pool_indices = Vec::with_capacity(pool_indices.len());
    for &(y, x) in pool_indices {
        if y >= relu_h || x >= relu_w {
            return Err(TensorError::InvalidArgument(
                "pool indices exceed relu feature map bounds",
            ));
        }
        packed_pool_indices.push((y * relu_w + x) as u32);
    }

    context
        .device
        .htod_copy_into(packed_pool_indices, &mut pool_indices_device)
        .map_err(|_| TensorError::InvalidArgument("failed to copy packed pool indices to CUDA device"))?;

    let mut params_device = {
        let mut pool = context
            .buffer_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
        pool.take_u32(&context.device, "backprop_unpool_params", 5)?
    };

    context
        .device
        .htod_copy_into(
            vec![
                channels as u32,
                pooled_h as u32,
                pooled_w as u32,
                relu_h as u32,
                relu_w as u32,
            ],
            &mut params_device,
        )
        .map_err(|_| TensorError::InvalidArgument("failed to copy unpool params to CUDA device"))?;

    context
        .device
        .memset_zeros(&mut conv_grad_device)
        .map_err(|_| TensorError::InvalidArgument("failed to reset CUDA conv_grad buffer"))?;

    let kernel = context
        .device
        .get_func(CUDA_KERNEL_MODULE, CUDA_UNPOOL_RELU_GRAD_KERNEL)
        .ok_or(TensorError::InvalidArgument(
            "failed to get CUDA unpool relu-grad kernel function",
        ))?;

    let launch = LaunchConfig::for_num_elems(expected_pool_indices as u32);
    unsafe {
        kernel
            .launch(
                launch,
                (
                    &pooled_grad_device,
                    &conv_pre_device,
                    &pool_indices_device,
                    &params_device,
                    &mut conv_grad_device,
                ),
            )
            .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA unpool relu-grad kernel"))?;
    }

    let (_, in_channels, kernel_h, kernel_w) = kernels.shape();
    let (_, _, conv_h, conv_w) = conv_pre_activation.shape();
    let (_, _, in_h, in_w) = input.shape();

    let bias_grad = if channels == 0 {
        Vec::new()
    } else {
        let (mut bias_params_device, mut bias_grad_device) = {
            let mut pool = context
                .buffer_pool
                .lock()
                .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
            (
                pool.take_u32(&context.device, "backprop_bias_params", 3)?,
                pool.take_f32(&context.device, "backprop_bias_grad", channels)?,
            )
        };

        context
            .device
            .htod_copy_into(
                vec![channels as u32, conv_h as u32, conv_w as u32],
                &mut bias_params_device,
            )
            .map_err(|_| TensorError::InvalidArgument("failed to copy bias-grad params to CUDA device"))?;

        let bias_kernel = context
            .device
            .get_func(CUDA_KERNEL_MODULE, CUDA_BIAS_GRAD_KERNEL)
            .ok_or(TensorError::InvalidArgument(
                "failed to get CUDA conv bias-grad kernel function",
            ))?;

        let bias_launch = LaunchConfig::for_num_elems(channels as u32);
        unsafe {
            bias_kernel
                .launch(
                    bias_launch,
                    (&conv_grad_device, &bias_params_device, &mut bias_grad_device),
                )
                .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA conv bias-grad kernel"))?;
        }

        let bias_grad = context
            .device
            .dtoh_sync_copy(&bias_grad_device)
            .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA bias_grad to host"))?;

        {
            let mut pool = context
                .buffer_pool
                .lock()
                .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
            pool.put_u32("backprop_bias_params", bias_params_device);
            pool.put_f32("backprop_bias_grad", bias_grad_device);
        }

        bias_grad
    };

    let has_kernel_grad = channels > 0 && in_channels > 0 && kernel_h > 0 && kernel_w > 0;
    let has_input_grad = compute_input_grad && in_channels > 0 && in_h > 0 && in_w > 0;

    let mut grad_params_device = if has_kernel_grad || has_input_grad {
        let mut params = {
            let mut pool = context
                .buffer_pool
                .lock()
                .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
            pool.take_u32(&context.device, "backprop_conv_grad_params", 8)?
        };

        context
            .device
            .htod_copy_into(
                vec![
                    channels as u32,
                    in_channels as u32,
                    conv_h as u32,
                    conv_w as u32,
                    in_h as u32,
                    in_w as u32,
                    kernel_h as u32,
                    kernel_w as u32,
                ],
                &mut params,
            )
            .map_err(|_| TensorError::InvalidArgument("failed to copy conv grad params to CUDA device"))?;

        Some(params)
    } else {
        None
    };

    let kernel_grad = if !has_kernel_grad {
        Tensor4D::zeros(channels, in_channels, kernel_h, kernel_w)
    } else {
        let kernel_grad_len = channels
            .checked_mul(in_channels)
            .and_then(|v| v.checked_mul(kernel_h))
            .and_then(|v| v.checked_mul(kernel_w))
            .ok_or(TensorError::InvalidArgument("kernel grad shape overflow"))?;

        let (mut input_device, mut kernel_grad_device) = {
            let mut pool = context
                .buffer_pool
                .lock()
                .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
            (
                pool.take_f32(&context.device, "backprop_input_tensor", input.len())?,
                pool.take_f32(&context.device, "backprop_kernel_grad", kernel_grad_len)?,
            )
        };

        context
            .device
            .htod_copy_into(input.as_slice().to_vec(), &mut input_device)
            .map_err(|_| TensorError::InvalidArgument("failed to copy input tensor to CUDA device"))?;

        let kernel_grad_kernel = context
            .device
            .get_func(CUDA_KERNEL_MODULE, CUDA_KERNEL_GRAD_KERNEL)
            .ok_or(TensorError::InvalidArgument(
                "failed to get CUDA conv kernel-grad kernel function",
            ))?;

        let kernel_grad_launch = LaunchConfig::for_num_elems(kernel_grad_len as u32);
        unsafe {
            kernel_grad_kernel
                .launch(
                    kernel_grad_launch,
                    (
                        &conv_grad_device,
                        &input_device,
                        grad_params_device
                            .as_ref()
                            .ok_or(TensorError::InvalidArgument("missing conv grad params device buffer"))?,
                        &mut kernel_grad_device,
                    ),
                )
                .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA conv kernel-grad kernel"))?;
        }

        let kernel_grad_values: Vec<f32> = context
            .device
            .dtoh_sync_copy(&kernel_grad_device)
            .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA kernel_grad to host"))?;

        {
            let mut pool = context
                .buffer_pool
                .lock()
                .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
            pool.put_f32("backprop_input_tensor", input_device);
            pool.put_f32("backprop_kernel_grad", kernel_grad_device);
        }

        Tensor4D::from_vec(channels, in_channels, kernel_h, kernel_w, kernel_grad_values)?
    };

    let input_grad = if compute_input_grad {
        if !has_input_grad {
            Some(Tensor4D::zeros(1, in_channels, in_h, in_w))
        } else {
            let input_grad_len = in_channels
                .checked_mul(in_h)
                .and_then(|v| v.checked_mul(in_w))
                .ok_or(TensorError::InvalidArgument("input grad shape overflow"))?;

            let (mut kernels_device, mut input_grad_device) = {
                let mut pool = context
                    .buffer_pool
                    .lock()
                    .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
                (
                    pool.take_f32(&context.device, "backprop_kernels_tensor", kernels.len())?,
                    pool.take_f32(&context.device, "backprop_input_grad", input_grad_len)?,
                )
            };

            context
                .device
                .htod_copy_into(kernels.as_slice().to_vec(), &mut kernels_device)
                .map_err(|_| TensorError::InvalidArgument("failed to copy kernels tensor to CUDA device"))?;

            let input_grad_kernel = context
                .device
                .get_func(CUDA_KERNEL_MODULE, CUDA_INPUT_GRAD_KERNEL)
                .ok_or(TensorError::InvalidArgument(
                    "failed to get CUDA conv input-grad kernel function",
                ))?;

            let input_grad_launch = LaunchConfig::for_num_elems(input_grad_len as u32);
            unsafe {
                input_grad_kernel
                    .launch(
                        input_grad_launch,
                        (
                            &conv_grad_device,
                            &kernels_device,
                            grad_params_device
                                .as_ref()
                                .ok_or(TensorError::InvalidArgument("missing conv grad params device buffer"))?,
                            &mut input_grad_device,
                        ),
                    )
                    .map_err(|_| TensorError::InvalidArgument("failed to launch CUDA conv input-grad kernel"))?;
            }

            let input_grad_values: Vec<f32> = context
                .device
                .dtoh_sync_copy(&input_grad_device)
                .map_err(|_| TensorError::InvalidArgument("failed to copy CUDA input_grad to host"))?;

            {
                let mut pool = context
                    .buffer_pool
                    .lock()
                    .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
                pool.put_f32("backprop_kernels_tensor", kernels_device);
                pool.put_f32("backprop_input_grad", input_grad_device);
            }

            Some(Tensor4D::from_vec(1, in_channels, in_h, in_w, input_grad_values)?)
        }
    } else {
        None
    };

    {
        let mut pool = context
            .buffer_pool
            .lock()
            .map_err(|_| TensorError::InvalidArgument("cuda device buffer pool lock poisoned"))?;
        pool.put_f32("backprop_pooled_grad", pooled_grad_device);
        pool.put_f32("backprop_conv_pre", conv_pre_device);
        pool.put_f32("backprop_conv_grad", conv_grad_device);
        pool.put_u32("backprop_pool_indices", pool_indices_device);
        pool.put_u32("backprop_unpool_params", params_device);
        if let Some(params) = grad_params_device.take() {
            pool.put_u32("backprop_conv_grad_params", params);
        }
    }

    Ok(ConvBlockBackwardGradients {
        kernel_grad,
        bias_grad,
        input_grad,
    })
}
