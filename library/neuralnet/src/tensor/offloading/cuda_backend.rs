use crate::tensor::backend::TensorBackend;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[cfg(feature = "offloading-cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "offloading-cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "offloading-cuda")]
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "offloading-cuda")]
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, Default)]
pub struct CudaTensorBackend;

impl TensorBackend for CudaTensorBackend {

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn conv2d_valid(
        &self,
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
        #[cfg(feature = "offloading-cuda")]
        {
            if let Ok(result) = cuda_max_pool2d_kernel(input, window_h, window_w, stride_h, stride_w) {
                return Ok(result);
            }
        }

        cuda_max_pool2d_fallback(input, window_h, window_w, stride_h, stride_w)
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        #[cfg(feature = "offloading-cuda")]
        {
            if let Ok(result) = cuda_global_average_pool2d_kernel(input) {
                return Ok(result);
            }
        }

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
                ],
            )
            .map_err(|err| format!("failed to load CUDA tensor kernels: {err:?}"))?;

        Ok(CudaKernelContext { device })
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
    Ok(gap.flatten_batch_features().first().cloned().unwrap_or_default())

}
