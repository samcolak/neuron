use crate::tensor::backend::TensorBackend;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[cfg(feature = "offloading-mlx")]
use apple_mlx::raw;
#[cfg(feature = "offloading-mlx")]
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, Default)]
pub struct MlxTensorBackend;

impl TensorBackend for MlxTensorBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn conv2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        mlx_conv2d_valid(input, kernels, bias, stride_h, stride_w)
    }

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        mlx_max_pool2d(input, window_h, window_w, stride_h, stride_w)
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        mlx_global_average_pool2d(input)
    }

    fn relu_inplace(&self, input: &mut Tensor4D) {
        mlx_relu_inplace(input)
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
        mlx_conv_relu_max_pool2d_valid(
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
        mlx_conv_blocks_to_feature_vec(
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

pub fn mlx_backend() -> MlxTensorBackend {
    MlxTensorBackend
}

pub fn mlx_backend_available() -> bool {
    #[cfg(feature = "offloading-mlx")]
    {
        true
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        false
    }
}

pub fn mlx_backend_uses_gpu() -> bool {
    #[cfg(feature = "offloading-mlx")]
    unsafe {
        let mut count = 0;
        raw::mlx_device_count(&mut count, raw::mlx_device_type__MLX_GPU) == 0 && count > 0
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        false
    }
}

pub fn mlx_backend_label() -> &'static str {
    if mlx_backend_uses_gpu() {
        "mlx(gpu)"
    } else {
        "mlx(cpu)"
    }
}

fn mlx_conv2d_valid(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-mlx")]
    {
        mlx_conv2d_valid_native(input, kernels, bias, stride_h, stride_w)
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        input.conv2d_valid_cpu(kernels, bias, stride_h, stride_w)
    }
}

#[allow(clippy::too_many_arguments)]
fn mlx_conv_relu_max_pool2d_valid(
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
    #[cfg(feature = "offloading-mlx")]
    {
        mlx_conv_relu_max_pool2d_valid_native(
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

    #[cfg(not(feature = "offloading-mlx"))]
    {
        let mut conv = input.conv2d_valid_cpu(kernels, bias, conv_stride_h, conv_stride_w)?;
        conv.relu_inplace_cpu();
        conv.max_pool2d_cpu(pool_window_h, pool_window_w, pool_stride_h, pool_stride_w)
    }
}

#[allow(clippy::too_many_arguments)]
fn mlx_conv_blocks_to_feature_vec(
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
    #[cfg(feature = "offloading-mlx")]
    {
        mlx_conv_blocks_to_feature_vec_native(
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

    #[cfg(not(feature = "offloading-mlx"))]
    {
        let b1_out = {
            let mut conv = input.conv2d_valid_cpu(block1_kernels, Some(block1_bias), conv_stride_h, conv_stride_w)?;
            conv.relu_inplace_cpu();
            conv.max_pool2d_cpu(pool_window_h, pool_window_w, pool_stride_h, pool_stride_w)?
        };
        let final_out = if let Some((k2, b2)) = block2 {
            let mut conv = b1_out.conv2d_valid_cpu(k2, Some(b2), conv_stride_h, conv_stride_w)?;
            conv.relu_inplace_cpu();
            conv.max_pool2d_cpu(pool_window_h, pool_window_w, pool_stride_h, pool_stride_w)?
        } else {
            b1_out
        };
        let gap = final_out.global_average_pool2d_cpu()?;
        Ok(gap.flatten_batch_features().first().cloned().unwrap_or_default())
    }
}

fn mlx_max_pool2d(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-mlx")]
    {
        mlx_max_pool2d_native(input, window_h, window_w, stride_h, stride_w)
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        input.max_pool2d_cpu(window_h, window_w, stride_h, stride_w)
    }
}

fn mlx_global_average_pool2d(input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    #[cfg(feature = "offloading-mlx")]
    {
        mlx_global_average_pool2d_native(input)
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        input.global_average_pool2d_cpu()
    }
}

fn mlx_relu_inplace(input: &mut Tensor4D) {
    #[cfg(feature = "offloading-mlx")]
    {
        if let Ok(result) = mlx_relu_native(input) {
            *input = result;
            return;
        }
    }

    input.relu_inplace_cpu()
}

#[cfg(feature = "offloading-mlx")]
fn mlx_check(code: i32, context: &'static str) -> Result<(), TensorError> {
    if code == 0 {
        Ok(())
    } else {
        Err(TensorError::InvalidArgument(context))
    }
}

#[cfg(feature = "offloading-mlx")]
fn mlx_stream_new() -> raw::mlx_stream {
    unsafe {
        if mlx_backend_uses_gpu() {
            return raw::mlx_default_gpu_stream_new();
        }
        raw::mlx_default_cpu_stream_new()
    }
}

#[cfg(feature = "offloading-mlx")]
fn mlx_array_new() -> raw::mlx_array {
    unsafe { raw::mlx_array_new() }
}

#[cfg(feature = "offloading-mlx")]
fn mlx_runtime_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(feature = "offloading-mlx")]
fn mlx_array_from_tensor(tensor: &Tensor4D) -> raw::mlx_array {
    let (n, c, h, w) = tensor.shape();
    let shape = [n as i32, c as i32, h as i32, w as i32];
    unsafe {
        raw::mlx_array_new_data(
            tensor.as_slice().as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            raw::mlx_dtype__MLX_FLOAT32,
        )
    }
}

#[cfg(feature = "offloading-mlx")]
fn mlx_array_to_tensor(array: raw::mlx_array, stream: raw::mlx_stream) -> Result<Tensor4D, TensorError> {
    unsafe {
        mlx_check(raw::mlx_array_eval(array), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;

        let ndim = raw::mlx_array_ndim(array);
        if ndim != 4 {
            return Err(TensorError::InvalidArgument(
                "mlx backend expected 4D array output",
            ));
        }

        let shape_ptr = raw::mlx_array_shape(array);
        if shape_ptr.is_null() {
            return Err(TensorError::InvalidArgument("mlx_array_shape returned null"));
        }

        let dims = std::slice::from_raw_parts(shape_ptr, ndim)
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<usize>>();
        let data_ptr = raw::mlx_array_data_float32(array);
        if data_ptr.is_null() {
            return Err(TensorError::InvalidArgument(
                "mlx_array_data_float32 returned null",
            ));
        }

        let len = raw::mlx_array_size(array);
        let values = std::slice::from_raw_parts(data_ptr, len).to_vec();
        Tensor4D::from_vec(dims[0], dims[1], dims[2], dims[3], values)
    }
}

#[cfg(feature = "offloading-mlx")]
fn mlx_transpose_axes(
    input: raw::mlx_array,
    axes: &[i32],
    stream: raw::mlx_stream,
) -> Result<raw::mlx_array, TensorError> {
    let mut out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_transpose_axes(&mut out, input, axes.as_ptr(), axes.len(), stream),
            "mlx_transpose_axes failed",
        )?;
    }
    Ok(out)
}

#[cfg(feature = "offloading-mlx")]
fn mlx_reshape(
    input: raw::mlx_array,
    shape: &[i32],
    stream: raw::mlx_stream,
) -> Result<raw::mlx_array, TensorError> {
    let mut out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_reshape(&mut out, input, shape.as_ptr(), shape.len(), stream),
            "mlx_reshape failed",
        )?;
    }
    Ok(out)
}

#[cfg(feature = "offloading-mlx")]
fn mlx_max_pool2d_array_nchw(
    input: raw::mlx_array,
    input_shape: (usize, usize, usize, usize),
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
    stream: raw::mlx_stream,
) -> Result<raw::mlx_array, TensorError> {
    if window_h == 0 || window_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("pooling window/stride must be > 0"));
    }

    let (n, c, h, w) = input_shape;
    let non_overlapping = window_h == stride_h
        && window_w == stride_w
        && h % window_h == 0
        && w % window_w == 0;

    if !non_overlapping {
        return Err(TensorError::InvalidArgument(
            "mlx backend only fuses non-overlapping pooling windows",
        ));
    }

    let out_h = h / window_h;
    let out_w = w / window_w;
    let reshaped = mlx_reshape(
        input,
        &[
            n as i32,
            c as i32,
            out_h as i32,
            window_h as i32,
            out_w as i32,
            window_w as i32,
        ],
        stream,
    )?;

    let mut reduced_w = mlx_array_new();
    let mut reduced_hw = mlx_array_new();
    let axes_w = [5i32];
    let axes_h = [3i32];
    unsafe {
        mlx_check(
            raw::mlx_max_axes(
                &mut reduced_w,
                reshaped,
                axes_w.as_ptr(),
                axes_w.len(),
                false,
                stream,
            ),
            "mlx_max_axes (width) failed",
        )?;
        mlx_check(
            raw::mlx_max_axes(
                &mut reduced_hw,
                reduced_w,
                axes_h.as_ptr(),
                axes_h.len(),
                false,
                stream,
            ),
            "mlx_max_axes (height) failed",
        )?;
        let _ = raw::mlx_array_free(reshaped);
        let _ = raw::mlx_array_free(reduced_w);
    }

    Ok(reduced_hw)
}

#[cfg(feature = "offloading-mlx")]
fn mlx_conv2d_valid_native(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let (n, in_c, in_h, in_w) = input.shape();
    let (out_c, kernel_in_c, kh, kw) = kernels.shape();

    if in_c != kernel_in_c {
        return Err(TensorError::IncompatibleShapes {
            left: input.shape(),
            right: kernels.shape(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("stride must be > 0"));
    }
    if in_h < kh || in_w < kw {
        return Err(TensorError::InvalidArgument(
            "kernel cannot exceed input dimensions",
        ));
    }

    let stream = mlx_stream_new();
    let input_arr = mlx_array_from_tensor(input);
    let kernels_arr = mlx_array_from_tensor(kernels);

    let input_nhwc = mlx_transpose_axes(input_arr, &[0, 2, 3, 1], stream)?;
    let kernels_ohwi = mlx_transpose_axes(kernels_arr, &[0, 2, 3, 1], stream)?;

    let mut conv_out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_conv2d(
                &mut conv_out,
                input_nhwc,
                kernels_ohwi,
                stride_h as i32,
                stride_w as i32,
                0,
                0,
                1,
                1,
                1,
                stream,
            ),
            "mlx_conv2d failed",
        )?;
    }

    let conv_nchw = mlx_transpose_axes(conv_out, &[0, 3, 1, 2], stream)?;

    let final_arr = if let Some(bias_values) = bias {
        if bias_values.len() != out_c {
            return Err(TensorError::ShapeMismatch {
                expected: out_c,
                actual: bias_values.len(),
            });
        }

        let bias_shape = [1, out_c as i32, 1, 1];
        let bias_arr = unsafe {
            raw::mlx_array_new_data(
                bias_values.as_ptr().cast(),
                bias_shape.as_ptr(),
                bias_shape.len() as i32,
                raw::mlx_dtype__MLX_FLOAT32,
            )
        };

        let mut biased = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_add(&mut biased, conv_nchw, bias_arr, stream),
                "mlx_add failed",
            )?;
            let _ = raw::mlx_array_free(bias_arr);
        }
        biased
    } else {
        conv_nchw
    };

    let result = mlx_array_to_tensor(final_arr, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(kernels_arr);
        let _ = raw::mlx_array_free(input_nhwc);
        let _ = raw::mlx_array_free(kernels_ohwi);
        let _ = raw::mlx_array_free(conv_out);
        let _ = raw::mlx_array_free(conv_nchw);
        if final_arr.ctx != conv_nchw.ctx {
            let _ = raw::mlx_array_free(final_arr);
        }
        let _ = raw::mlx_stream_free(stream);
    }

    // Avoid unused variable lint for shape values where only validation needs them.
    let _ = n;
    result
}

#[cfg(feature = "offloading-mlx")]
fn mlx_conv_relu_max_pool2d_valid_native(
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
    let (n, in_c, in_h, in_w) = input.shape();
    let (out_c, kernel_in_c, kernel_h, kernel_w) = kernels.shape();

    if in_c != kernel_in_c {
        return Err(TensorError::IncompatibleShapes {
            left: input.shape(),
            right: kernels.shape(),
        });
    }
    if conv_stride_h == 0 || conv_stride_w == 0 {
        return Err(TensorError::InvalidArgument("stride must be > 0"));
    }
    if in_h < kernel_h || in_w < kernel_w {
        return Err(TensorError::InvalidArgument(
            "kernel cannot exceed input dimensions",
        ));
    }

    let conv_h = ((in_h - kernel_h) / conv_stride_h) + 1;
    let conv_w = ((in_w - kernel_w) / conv_stride_w) + 1;

    let non_overlapping = pool_window_h == pool_stride_h
        && pool_window_w == pool_stride_w
        && conv_h % pool_window_h == 0
        && conv_w % pool_window_w == 0;

    if !non_overlapping {
        let mut conv = mlx_conv2d_valid_native(input, kernels, bias, conv_stride_h, conv_stride_w)?;
        conv.relu_inplace_cpu();
        return conv.max_pool2d_cpu(pool_window_h, pool_window_w, pool_stride_h, pool_stride_w);
    }

    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let stream = mlx_stream_new();
    let input_arr = mlx_array_from_tensor(input);
    let kernels_arr = mlx_array_from_tensor(kernels);
    let input_nhwc = mlx_transpose_axes(input_arr, &[0, 2, 3, 1], stream)?;
    let kernels_ohwi = mlx_transpose_axes(kernels_arr, &[0, 2, 3, 1], stream)?;

    let mut conv_out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_conv2d(
                &mut conv_out,
                input_nhwc,
                kernels_ohwi,
                conv_stride_h as i32,
                conv_stride_w as i32,
                0,
                0,
                1,
                1,
                1,
                stream,
            ),
            "mlx_conv2d failed",
        )?;
    }

    let conv_nchw = mlx_transpose_axes(conv_out, &[0, 3, 1, 2], stream)?;
    let biased_nchw = if let Some(bias_values) = bias {
        if bias_values.len() != out_c {
            return Err(TensorError::ShapeMismatch {
                expected: out_c,
                actual: bias_values.len(),
            });
        }

        let bias_shape = [1, out_c as i32, 1, 1];
        let bias_arr = unsafe {
            raw::mlx_array_new_data(
                bias_values.as_ptr().cast(),
                bias_shape.as_ptr(),
                bias_shape.len() as i32,
                raw::mlx_dtype__MLX_FLOAT32,
            )
        };
        let mut biased = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_add(&mut biased, conv_nchw, bias_arr, stream),
                "mlx_add failed",
            )?;
            let _ = raw::mlx_array_free(bias_arr);
        }
        Some(biased)
    } else {
        None
    };

    let relu_input = biased_nchw.unwrap_or(conv_nchw);
    let zero = unsafe { raw::mlx_array_new_float32(0.0) };
    let mut relu_out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_maximum(&mut relu_out, relu_input, zero, stream),
            "mlx_maximum failed",
        )?;
        let _ = raw::mlx_array_free(zero);
    }

    let pooled = mlx_max_pool2d_array_nchw(
        relu_out,
        (n, out_c, conv_h, conv_w),
        pool_window_h,
        pool_window_w,
        pool_stride_h,
        pool_stride_w,
        stream,
    )?;

    let result = mlx_array_to_tensor(pooled, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(kernels_arr);
        let _ = raw::mlx_array_free(input_nhwc);
        let _ = raw::mlx_array_free(kernels_ohwi);
        let _ = raw::mlx_array_free(conv_out);
        let _ = raw::mlx_array_free(conv_nchw);
        if let Some(biased) = biased_nchw {
            let _ = raw::mlx_array_free(biased);
        }
        let _ = raw::mlx_array_free(relu_out);
        let _ = raw::mlx_array_free(pooled);
        let _ = raw::mlx_stream_free(stream);
    }
    result
}

/// Fused conv→relu→pool (up to 2 blocks) → GAP for inference.
/// All intermediate arrays stay on-device; only the tiny feature vec is
/// copied back to the host, eliminating 2-4 intermediate synchronisations.
#[cfg(feature = "offloading-mlx")]
fn mlx_conv_blocks_to_feature_vec_native(
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
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let stream = mlx_stream_new();

    // Helper: run one fused conv→relu→pool block, return pooled NCHW array
    // and the analytical output spatial dims.  No host copy.
    let run_block = |inp_arr: raw::mlx_array,
                     inp_shape: (usize, usize, usize, usize),
                     kernels: &Tensor4D,
                     bias: &[f32]|
     -> Result<(raw::mlx_array, (usize, usize, usize, usize)), TensorError> {
        let (n, in_c, in_h, in_w) = inp_shape;
        let (out_c, kernel_in_c, kh, kw) = kernels.shape();
        if in_c != kernel_in_c {
            return Err(TensorError::IncompatibleShapes {
                left: inp_shape,
                right: kernels.shape(),
            });
        }
        if in_h < kh || in_w < kw {
            return Err(TensorError::InvalidArgument(
                "kernel cannot exceed input dimensions",
            ));
        }
        let conv_h = ((in_h - kh) / conv_stride_h) + 1;
        let conv_w = ((in_w - kw) / conv_stride_w) + 1;
        let non_overlapping = pool_window_h == pool_stride_h
            && pool_window_w == pool_stride_w
            && conv_h % pool_window_h == 0
            && conv_w % pool_window_w == 0;

        let kernels_arr = mlx_array_from_tensor(kernels);
        let inp_nhwc = mlx_transpose_axes(inp_arr, &[0, 2, 3, 1], stream)?;
        let kernels_ohwi = mlx_transpose_axes(kernels_arr, &[0, 2, 3, 1], stream)?;

        let mut conv_out = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_conv2d(
                    &mut conv_out,
                    inp_nhwc,
                    kernels_ohwi,
                    conv_stride_h as i32,
                    conv_stride_w as i32,
                    0, 0, 1, 1, 1,
                    stream,
                ),
                "mlx_conv2d failed",
            )?;
            let _ = raw::mlx_array_free(kernels_arr);
            let _ = raw::mlx_array_free(inp_nhwc);
            let _ = raw::mlx_array_free(kernels_ohwi);
        }

        let conv_nchw = mlx_transpose_axes(conv_out, &[0, 3, 1, 2], stream)?;
        unsafe { let _ = raw::mlx_array_free(conv_out); }

        let bias_shape = [1, out_c as i32, 1, 1];
        let bias_arr = unsafe {
            raw::mlx_array_new_data(
                bias.as_ptr().cast(),
                bias_shape.as_ptr(),
                bias_shape.len() as i32,
                raw::mlx_dtype__MLX_FLOAT32,
            )
        };
        let mut biased = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_add(&mut biased, conv_nchw, bias_arr, stream),
                "mlx_add bias failed",
            )?;
            let _ = raw::mlx_array_free(bias_arr);
            let _ = raw::mlx_array_free(conv_nchw);
        }

        let zero = unsafe { raw::mlx_array_new_float32(0.0) };
        let mut relu_out = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_maximum(&mut relu_out, biased, zero, stream),
                "mlx_maximum relu failed",
            )?;
            let _ = raw::mlx_array_free(zero);
            let _ = raw::mlx_array_free(biased);
        }

        let pooled = if non_overlapping {
            let p = mlx_max_pool2d_array_nchw(
                relu_out,
                (n, out_c, conv_h, conv_w),
                pool_window_h, pool_window_w,
                pool_stride_h, pool_stride_w,
                stream,
            )?;
            unsafe { let _ = raw::mlx_array_free(relu_out); }
            p
        } else {
            relu_out
        };

        let out_h = if non_overlapping { conv_h / pool_window_h } else { conv_h };
        let out_w = if non_overlapping { conv_w / pool_window_w } else { conv_w };
        Ok((pooled, (n, out_c, out_h, out_w)))
    };

    let input_arr = mlx_array_from_tensor(input);
    let input_shape = input.shape();
    let (b1_pooled, b1_shape) = run_block(input_arr, input_shape, block1_kernels, block1_bias)?;

    let final_pooled = if let Some((k2, b2)) = block2 {
        let (b2_pooled, _) = run_block(b1_pooled, b1_shape, k2, b2)?;
        b2_pooled
    } else {
        b1_pooled
    };

    // GAP: mean over spatial axes [2,3], keep dims=true gives shape [N,C,1,1]
    let mut gap_out = mlx_array_new();
    let gap_axes = [2i32, 3i32];
    unsafe {
        mlx_check(
            raw::mlx_mean_axes(
                &mut gap_out,
                final_pooled,
                gap_axes.as_ptr(),
                gap_axes.len(),
                true,
                stream,
            ),
            "mlx_mean_axes gap failed",
        )?;
        let _ = raw::mlx_array_free(final_pooled);
    }

    // One synchronise for the entire chain.
    unsafe {
        mlx_check(raw::mlx_array_eval(gap_out), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
    }

    let (n, c, _, _) = input_shape;
    let _ = n;
    let feature_len = c;
    let (_, out_c, _, _) = if block2.is_some() { (0usize, 0usize, 0usize, 0usize) } else { (0, 0, 0, 0) };
    let _ = out_c;
    let feature_len = unsafe { raw::mlx_array_size(gap_out) };
    let data_ptr = unsafe { raw::mlx_array_data_float32(gap_out) };
    let features = if data_ptr.is_null() || feature_len == 0 {
        vec![0.0f32]
    } else {
        unsafe { std::slice::from_raw_parts(data_ptr, feature_len).to_vec() }
    };

    unsafe {
        let _ = raw::mlx_array_free(gap_out);
        let _ = raw::mlx_stream_free(stream);
    }

    Ok(features)
}

#[cfg(feature = "offloading-mlx")]
fn mlx_max_pool2d_native(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    // MLX C API currently has no dedicated max_pool2d function; use reshape+max
    // for the common non-overlapping case and fallback to CPU otherwise.
    if window_h == 0 || window_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("pooling window/stride must be > 0"));
    }

    let input_shape = input.shape();
    let (_, _, h, w) = input_shape;
    if !(window_h == stride_h
        && window_w == stride_w
        && h % window_h == 0
        && w % window_w == 0)
    {
        return input.max_pool2d_cpu(window_h, window_w, stride_h, stride_w);
    }

    let stream = mlx_stream_new();
    let input_arr = mlx_array_from_tensor(input);
    let reduced_hw = mlx_max_pool2d_array_nchw(
        input_arr,
        input_shape,
        window_h,
        window_w,
        stride_h,
        stride_w,
        stream,
    )?;

    let result = mlx_array_to_tensor(reduced_hw, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(reduced_hw);
        let _ = raw::mlx_stream_free(stream);
    }
    result
}

#[cfg(feature = "offloading-mlx")]
fn mlx_global_average_pool2d_native(input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let stream = mlx_stream_new();
    let input_arr = mlx_array_from_tensor(input);
    let mut out = mlx_array_new();
    let axes = [2i32, 3i32];
    unsafe {
        mlx_check(
            raw::mlx_mean_axes(&mut out, input_arr, axes.as_ptr(), axes.len(), true, stream),
            "mlx_mean_axes failed",
        )?;
    }

    let result = mlx_array_to_tensor(out, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(out);
        let _ = raw::mlx_stream_free(stream);
    }
    result
}

#[cfg(feature = "offloading-mlx")]
fn mlx_relu_native(input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let stream = mlx_stream_new();
    let input_arr = mlx_array_from_tensor(input);
    let zero = unsafe { raw::mlx_array_new_float32(0.0) };
    let mut out = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_maximum(&mut out, input_arr, zero, stream),
            "mlx_maximum failed",
        )?;
    }

    let result = mlx_array_to_tensor(out, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(zero);
        let _ = raw::mlx_array_free(out);
        let _ = raw::mlx_stream_free(stream);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_backend_conv_matches_cpu_core() {
        let backend = mlx_backend();

        let input = Tensor4D::from_vec(
            1,
            1,
            3,
            3,
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap_or_else(|_| panic!("input tensor should be valid"));

        let kernels = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 0.0, 0.0, 1.0])
            .unwrap_or_else(|_| panic!("kernel tensor should be valid"));

        let cpu = input
            .conv2d_valid_cpu(&kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("cpu core conv should succeed"));

        let mlx = backend
            .conv2d_valid(&input, &kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("mlx conv should succeed"));

        assert_eq!(cpu, mlx);
    }

    #[test]
    fn mlx_backend_pool_gap_and_relu_match_cpu_core() {
        let backend = mlx_backend();

        let input = Tensor4D::from_vec(
            1,
            1,
            4,
            4,
            vec![
                -1.0, 3.0, 2.0, 0.0,
                5.0, -6.0, 1.0, 4.0,
                2.0, 8.0, -7.0, 3.0,
                9.0, 1.0, 5.0, -2.0,
            ],
        )
        .unwrap_or_else(|_| panic!("input tensor should be valid"));

        let cpu_pool = input
            .max_pool2d_cpu(2, 2, 2, 2)
            .unwrap_or_else(|_| panic!("cpu core pool should succeed"));
        let mlx_pool = backend
            .max_pool2d(&input, 2, 2, 2, 2)
            .unwrap_or_else(|_| panic!("mlx pool should succeed"));
        assert_eq!(cpu_pool, mlx_pool);

        let cpu_gap = cpu_pool
            .global_average_pool2d_cpu()
            .unwrap_or_else(|_| panic!("cpu core gap should succeed"));
        let mlx_gap = mlx_pool
            .global_average_pool2d_with_backend(&backend)
            .unwrap_or_else(|_| panic!("mlx gap should succeed"));
        assert_eq!(cpu_gap, mlx_gap);

        let mut cpu_relu = input.clone();
        let mut mlx_relu = input.clone();
        cpu_relu.relu_inplace_cpu();
        backend.relu_inplace(&mut mlx_relu);
        assert_eq!(cpu_relu, mlx_relu);
    }

    #[test]
    fn mlx_backend_feature_flag_reflects_build_configuration() {
        assert_eq!(mlx_backend_available(), cfg!(feature = "offloading-mlx"));
    }

    #[test]
    fn mlx_backend_label_includes_runtime_mode() {
        let label = mlx_backend_label();
        assert!(label == "mlx(cpu)" || label == "mlx(gpu)");
    }
    
}
