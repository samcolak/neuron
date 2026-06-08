use crate::tensor::backend::{
    cpu_conv_block_backward_gradients,
    ConvBlockBackwardGradients,
    TensorBackend,
};
use crate::tensor::device::BackendTrainingCapabilities;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[cfg(feature = "offloading-mlx")]
use apple_mlx::raw;
#[cfg(feature = "offloading-mlx")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "offloading-mlx")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "offloading-mlx")]
use std::time::Instant;

#[cfg(feature = "offloading-mlx")]
fn mlx_allow_cpu_fallback() -> bool {
    crate::tensor::backend::cpu_fallback_enabled()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MlxTensorBackend;

#[cfg(feature = "offloading-mlx")]
#[derive(Debug)]
pub struct MlxOwnedArray {
    array: raw::mlx_array,
}

#[cfg(feature = "offloading-mlx")]
impl Drop for MlxOwnedArray {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::mlx_array_free(self.array);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxBackpropPathSnapshot {
    pub total_calls: u64,
    pub full_native_success: u64,
    pub full_cpu_fallback: u64,
    pub fallback_incompatible_shapes: u64,
    pub fallback_shape_mismatch: u64,
    pub fallback_invalid_argument: u64,
    pub fallback_other: u64,
    pub input_grad_requested: u64,
    pub input_grad_skipped: u64,
    pub intended_native_dw: u64,
    pub executed_native_dw: u64,
    pub fallback_dw: u64,
    pub intended_native_dinput: u64,
    pub executed_native_dinput: u64,
    pub fallback_dinput: u64,
    pub native_dw_time_ns: u64,
    pub native_dinput_time_ns: u64,
    pub native_dw_transpose_time_ns: u64,
    pub native_dw_conv_time_ns: u64,
    pub native_dw_materialize_time_ns: u64,
}

impl MlxBackpropPathSnapshot {
    pub fn full_native_success_ratio(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.full_native_success as f64 / self.total_calls as f64
        }
    }

    pub fn full_cpu_fallback_ratio(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.full_cpu_fallback as f64 / self.total_calls as f64
        }
    }

    pub fn dw_native_realization_ratio(&self) -> f64 {
        if self.intended_native_dw == 0 {
            0.0
        } else {
            self.executed_native_dw as f64 / self.intended_native_dw as f64
        }
    }

    pub fn dinput_native_realization_ratio(&self) -> f64 {
        if self.intended_native_dinput == 0 {
            0.0
        } else {
            self.executed_native_dinput as f64 / self.intended_native_dinput as f64
        }
    }
}

#[cfg(feature = "offloading-mlx")]
#[derive(Default)]
struct MlxBackpropPathCounters {
    total_calls: AtomicU64,
    full_native_success: AtomicU64,
    full_cpu_fallback: AtomicU64,
    fallback_incompatible_shapes: AtomicU64,
    fallback_shape_mismatch: AtomicU64,
    fallback_invalid_argument: AtomicU64,
    fallback_other: AtomicU64,
    input_grad_requested: AtomicU64,
    input_grad_skipped: AtomicU64,
    intended_native_dw: AtomicU64,
    executed_native_dw: AtomicU64,
    fallback_dw: AtomicU64,
    intended_native_dinput: AtomicU64,
    executed_native_dinput: AtomicU64,
    fallback_dinput: AtomicU64,
    native_dw_time_ns: AtomicU64,
    native_dinput_time_ns: AtomicU64,
    native_dw_transpose_time_ns: AtomicU64,
    native_dw_conv_time_ns: AtomicU64,
    native_dw_materialize_time_ns: AtomicU64,
}

#[cfg(feature = "offloading-mlx")]
fn mlx_backprop_path_counters() -> &'static MlxBackpropPathCounters {
    static COUNTERS: OnceLock<MlxBackpropPathCounters> = OnceLock::new();
    COUNTERS.get_or_init(MlxBackpropPathCounters::default)
}

pub fn mlx_backprop_path_snapshot() -> MlxBackpropPathSnapshot {
    #[cfg(feature = "offloading-mlx")]
    {
        let counters = mlx_backprop_path_counters();
        return MlxBackpropPathSnapshot {
            total_calls: counters.total_calls.load(Ordering::Relaxed),
            full_native_success: counters.full_native_success.load(Ordering::Relaxed),
            full_cpu_fallback: counters.full_cpu_fallback.load(Ordering::Relaxed),
            fallback_incompatible_shapes: counters
                .fallback_incompatible_shapes
                .load(Ordering::Relaxed),
            fallback_shape_mismatch: counters
                .fallback_shape_mismatch
                .load(Ordering::Relaxed),
            fallback_invalid_argument: counters
                .fallback_invalid_argument
                .load(Ordering::Relaxed),
            fallback_other: counters.fallback_other.load(Ordering::Relaxed),
            input_grad_requested: counters.input_grad_requested.load(Ordering::Relaxed),
            input_grad_skipped: counters.input_grad_skipped.load(Ordering::Relaxed),
            intended_native_dw: counters.intended_native_dw.load(Ordering::Relaxed),
            executed_native_dw: counters.executed_native_dw.load(Ordering::Relaxed),
            fallback_dw: counters.fallback_dw.load(Ordering::Relaxed),
            intended_native_dinput: counters.intended_native_dinput.load(Ordering::Relaxed),
            executed_native_dinput: counters.executed_native_dinput.load(Ordering::Relaxed),
            fallback_dinput: counters.fallback_dinput.load(Ordering::Relaxed),
            native_dw_time_ns: counters.native_dw_time_ns.load(Ordering::Relaxed),
            native_dinput_time_ns: counters.native_dinput_time_ns.load(Ordering::Relaxed),
            native_dw_transpose_time_ns: counters.native_dw_transpose_time_ns.load(Ordering::Relaxed),
            native_dw_conv_time_ns: counters.native_dw_conv_time_ns.load(Ordering::Relaxed),
            native_dw_materialize_time_ns: counters.native_dw_materialize_time_ns.load(Ordering::Relaxed),
        };
    }

    #[cfg(not(feature = "offloading-mlx"))]
    {
        MlxBackpropPathSnapshot::default()
    }
}

pub fn mlx_backprop_path_reset() {
    #[cfg(feature = "offloading-mlx")]
    {
        let counters = mlx_backprop_path_counters();
        counters.total_calls.store(0, Ordering::Relaxed);
        counters.full_native_success.store(0, Ordering::Relaxed);
        counters.full_cpu_fallback.store(0, Ordering::Relaxed);
        counters
            .fallback_incompatible_shapes
            .store(0, Ordering::Relaxed);
        counters
            .fallback_shape_mismatch
            .store(0, Ordering::Relaxed);
        counters
            .fallback_invalid_argument
            .store(0, Ordering::Relaxed);
        counters.fallback_other.store(0, Ordering::Relaxed);
        counters.input_grad_requested.store(0, Ordering::Relaxed);
        counters.input_grad_skipped.store(0, Ordering::Relaxed);
        counters.intended_native_dw.store(0, Ordering::Relaxed);
        counters.executed_native_dw.store(0, Ordering::Relaxed);
        counters.fallback_dw.store(0, Ordering::Relaxed);
        counters.intended_native_dinput.store(0, Ordering::Relaxed);
        counters.executed_native_dinput.store(0, Ordering::Relaxed);
        counters.fallback_dinput.store(0, Ordering::Relaxed);
        counters.native_dw_time_ns.store(0, Ordering::Relaxed);
        counters.native_dinput_time_ns.store(0, Ordering::Relaxed);
        counters.native_dw_transpose_time_ns.store(0, Ordering::Relaxed);
        counters.native_dw_conv_time_ns.store(0, Ordering::Relaxed);
        counters.native_dw_materialize_time_ns.store(0, Ordering::Relaxed);
    }
}

impl TensorBackend for MlxTensorBackend {
    fn name(&self) -> &'static str {
        "mlx"
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
        mlx_conv_block_backward_gradients_fallback(
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
fn mlx_conv_block_backward_gradients_fallback(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_pre_activation: &Tensor4D,
    pool_indices: &[(usize, usize)],
    pooled_shape: (usize, usize, usize, usize),
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackwardGradients, TensorError> {
    
    #[cfg(feature = "offloading-mlx")]
    {
        let counters = mlx_backprop_path_counters();
        counters.total_calls.fetch_add(1, Ordering::Relaxed);

        let native_result = if pooled_shape.0 > 1 {
            mlx_conv_block_backward_gradients_native_batched_by_sample(
                kernels,
                input,
                conv_pre_activation,
                pool_indices,
                pooled_shape,
                pooled_grad,
                compute_input_grad,
            )
        } else {
            mlx_conv_block_backward_gradients_native(
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
            counters.full_native_success.fetch_add(1, Ordering::Relaxed);
            return Ok(result);
        }
        let Err(ref err) = native_result else {
            unreachable!()
        };
        match err {
            TensorError::IncompatibleShapes { .. } => {
                counters
                    .fallback_incompatible_shapes
                    .fetch_add(1, Ordering::Relaxed);
            }
            TensorError::ShapeMismatch { .. } => {
                counters
                    .fallback_shape_mismatch
                    .fetch_add(1, Ordering::Relaxed);
            }
            TensorError::InvalidArgument(_) => {
                counters
                    .fallback_invalid_argument
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                counters.fallback_other.fetch_add(1, Ordering::Relaxed);
            }
        }

        if !mlx_allow_cpu_fallback() {
            return native_result;
        }

        counters.full_cpu_fallback.fetch_add(1, Ordering::Relaxed);
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

#[cfg(feature = "offloading-mlx")]
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

#[cfg(feature = "offloading-mlx")]
#[allow(clippy::too_many_arguments)]
fn mlx_conv_block_backward_gradients_native_batched_by_sample(
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
    let (conv_batch, conv_channels, _, _) = conv_pre_activation.shape();
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

        let sample_backward = mlx_conv_block_backward_gradients_native(
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

    Ok(ConvBlockBackwardGradients {
        kernel_grad,
        bias_grad,
        input_grad,
    })
}

#[cfg(feature = "offloading-mlx")]
#[allow(clippy::too_many_arguments)]
fn mlx_conv_block_backward_gradients_native(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_pre_activation: &Tensor4D,
    pool_indices: &[(usize, usize)],
    pooled_shape: (usize, usize, usize, usize),
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackwardGradients, TensorError> {

    let counters = mlx_backprop_path_counters();
    if compute_input_grad {
        counters.input_grad_requested.fetch_add(1, Ordering::Relaxed);
    } else {
        counters.input_grad_skipped.fetch_add(1, Ordering::Relaxed);
    }

    let pooled_grad_shape = pooled_grad.shape();
    if pooled_grad_shape != pooled_shape {
        return Err(TensorError::IncompatibleShapes {
            left: pooled_shape,
            right: pooled_grad_shape,
        });
    }

    if pooled_shape.0 != 1 {
        return Err(TensorError::InvalidArgument(
            "mlx conv block backward currently supports batch size 1",
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

    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let stream = mlx_stream_new();
    let pooled_plane = pooled_h * pooled_w;
    let relu_plane = relu_h * relu_w;

    let pooled_grad_shape_arr = [1, channels as i32, pooled_plane as i32];
    let pooled_grad_arr = unsafe {
        raw::mlx_array_new_data(
            pooled_grad.as_slice().as_ptr().cast(),
            pooled_grad_shape_arr.as_ptr(),
            pooled_grad_shape_arr.len() as i32,
            raw::mlx_dtype__MLX_FLOAT32,
        )
    };

    let mut packed_indices = Vec::with_capacity(pool_indices.len());
    for &(y, x) in pool_indices {
        if y >= relu_h || x >= relu_w {
            return Err(TensorError::InvalidArgument(
                "pool indices exceed relu feature map bounds",
            ));
        }
        packed_indices.push((y * relu_w + x) as i32);
    }

    let packed_indices_shape_arr = [1, channels as i32, pooled_plane as i32];
    let packed_indices_arr = unsafe {
        raw::mlx_array_new_data(
            packed_indices.as_ptr().cast(),
            packed_indices_shape_arr.as_ptr(),
            packed_indices_shape_arr.len() as i32,
            raw::mlx_dtype__MLX_INT32,
        )
    };

    let conv_grad_flat_shape = [1, channels as i32, relu_plane as i32];
    let mut conv_grad_flat = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_zeros(
                &mut conv_grad_flat,
                conv_grad_flat_shape.as_ptr(),
                conv_grad_flat_shape.len(),
                raw::mlx_dtype__MLX_FLOAT32,
                stream,
            ),
            "mlx_zeros conv_grad failed",
        )?;
    }

    let mut conv_grad_scattered = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_scatter_add_axis(
                &mut conv_grad_scattered,
                conv_grad_flat,
                packed_indices_arr,
                pooled_grad_arr,
                2,
                stream,
            ),
            "mlx_scatter_add_axis conv_grad failed",
        )?;
    }

    let conv_grad = mlx_reshape(
        conv_grad_scattered,
        &[1, channels as i32, relu_h as i32, relu_w as i32],
        stream,
    )?;

    let conv_pre_arr = mlx_array_from_tensor(conv_pre_activation);
    let zero = unsafe { raw::mlx_array_new_float32(0.0) };
    let mut relu_mask = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_greater(&mut relu_mask, conv_pre_arr, zero, stream),
            "mlx_greater relu mask failed",
        )?;
    }

    let mut conv_grad_zeros = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_zeros_like(&mut conv_grad_zeros, conv_grad, stream),
            "mlx_zeros_like conv_grad failed",
        )?;
    }

    let mut conv_grad_masked = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_where(
                &mut conv_grad_masked,
                relu_mask,
                conv_grad,
                conv_grad_zeros,
                stream,
            ),
            "mlx_where relu mask application failed",
        )?;
    }

    let intended_native_dw = mlx_enable_experimental_native_dw();
    if intended_native_dw {
        counters.intended_native_dw.fetch_add(1, Ordering::Relaxed);
    }

    let native_kernel_grad = if intended_native_dw {
        let native_kernel_grad_started = Instant::now();
        let native_kernel_grad = mlx_kernel_grad_native_from_conv_grad_array(conv_grad_masked, input, stream);
        counters.native_dw_time_ns.fetch_add(
            native_kernel_grad_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        Some(native_kernel_grad)
    } else {
        None
    };

    if matches!(native_kernel_grad, Some(Ok(_))) {
        counters.executed_native_dw.fetch_add(1, Ordering::Relaxed);
    } else if intended_native_dw {
        counters.fallback_dw.fetch_add(1, Ordering::Relaxed);
    }

    let native_input_grad: Option<Result<Tensor4D, TensorError>> = if compute_input_grad {
        let intended_native_dinput = mlx_enable_experimental_native_dinput();
        if intended_native_dinput {
            counters
                .intended_native_dinput
                .fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            let result = mlx_input_grad_native_from_conv_grad_array(conv_grad_masked, kernels, stream);
            counters.native_dinput_time_ns.fetch_add(
                started.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            Some(result)
        } else {
            None
        }
    } else {
        None
    };

    if matches!(native_input_grad, Some(Ok(_))) {
        counters
            .executed_native_dinput
            .fetch_add(1, Ordering::Relaxed);
    } else if compute_input_grad && mlx_enable_experimental_native_dinput() {
        counters.fallback_dinput.fetch_add(1, Ordering::Relaxed);
    }

    let mut bias_arr = mlx_array_new();
    let bias_axes = [0i32, 2i32, 3i32];
    unsafe {
        mlx_check(
            raw::mlx_sum_axes(
                &mut bias_arr,
                conv_grad_masked,
                bias_axes.as_ptr(),
                bias_axes.len(),
                true,
                stream,
            ),
            "mlx_sum_axes bias grad failed",
        )?;
    }

    let bias_tensor = mlx_array_to_tensor(bias_arr, stream)?;
    let bias_grad = bias_tensor.first_sample_features();

    let needs_cpu_kernel_grad = matches!(native_kernel_grad, Some(Err(_)) | None);
    let needs_cpu_input_grad = compute_input_grad && matches!(native_input_grad, Some(Err(_)) | None);
    let conv_grad_tensor = if needs_cpu_kernel_grad || needs_cpu_input_grad {
        Some(mlx_array_to_tensor(conv_grad_masked, stream)?)
    } else {
        None
    };

    unsafe {
        let _ = raw::mlx_array_free(pooled_grad_arr);
        let _ = raw::mlx_array_free(packed_indices_arr);
        let _ = raw::mlx_array_free(conv_grad_flat);
        let _ = raw::mlx_array_free(conv_grad_scattered);
        let _ = raw::mlx_array_free(conv_grad);
        let _ = raw::mlx_array_free(conv_pre_arr);
        let _ = raw::mlx_array_free(zero);
        let _ = raw::mlx_array_free(relu_mask);
        let _ = raw::mlx_array_free(conv_grad_zeros);
        let _ = raw::mlx_array_free(conv_grad_masked);
        let _ = raw::mlx_array_free(bias_arr);
        let _ = raw::mlx_stream_free(stream);
    }

    let kernel_grad = match native_kernel_grad {
        Some(Ok(native)) => native,
        Some(Err(err)) => {
            if !mlx_allow_cpu_fallback() {
                return Err(err);
            }
            cpu_kernel_grad_from_conv_grad(
                kernels,
                input,
                conv_grad_tensor
                    .as_ref()
                    .expect("conv_grad tensor should be materialized for CPU fallback"),
            )?
        }
        None => cpu_kernel_grad_from_conv_grad(
            kernels,
            input,
            conv_grad_tensor
                .as_ref()
                .expect("conv_grad tensor should be materialized for CPU gradient path"),
        )?,
    };

    let input_grad = if compute_input_grad {
        match native_input_grad {
            Some(Ok(native)) => Some(native),
            Some(Err(err)) => {
                if !mlx_allow_cpu_fallback() {
                    return Err(err);
                }
                Some(cpu_input_grad_from_conv_grad(
                    kernels,
                    conv_grad_tensor
                        .as_ref()
                        .expect("conv_grad tensor should be materialized for CPU fallback"),
                )?)
            }
            None => Some(cpu_input_grad_from_conv_grad(
                kernels,
                conv_grad_tensor
                    .as_ref()
                    .expect("conv_grad tensor should be materialized for CPU gradient path"),
            )?),
        }
    } else {
        None
    };

    Ok(ConvBlockBackwardGradients {
        kernel_grad,
        bias_grad,
        input_grad,
    })
}

#[cfg(feature = "offloading-mlx")]
fn mlx_enable_experimental_native_dw() -> bool {
    crate::tensor::backend::native_dw_enabled()
}

#[cfg(feature = "offloading-mlx")]
fn mlx_enable_experimental_native_dinput() -> bool {
    crate::tensor::backend::native_dinput_enabled()
}

#[cfg(feature = "offloading-mlx")]
fn mlx_kernel_grad_native_from_conv_grad_array(
    conv_grad_nchw: raw::mlx_array,
    input: &Tensor4D,
    stream: raw::mlx_stream,
) -> Result<Tensor4D, TensorError> {
    let counters = mlx_backprop_path_counters();
    let (_, in_channels, in_h, in_w) = input.shape();
    let (out_channels, conv_h, conv_w) = unsafe {
        let ndim = raw::mlx_array_ndim(conv_grad_nchw);
        if ndim != 4 {
            return Err(TensorError::InvalidArgument(
                "mlx kernel grad expected 4D conv-grad tensor",
            ));
        }
        let shape_ptr = raw::mlx_array_shape(conv_grad_nchw);
        if shape_ptr.is_null() {
            return Err(TensorError::InvalidArgument(
                "mlx_array_shape returned null for conv-grad",
            ));
        }
        let dims = std::slice::from_raw_parts(shape_ptr, ndim);
        (dims[1] as usize, dims[2] as usize, dims[3] as usize)
    };

    if in_h < conv_h || in_w < conv_w {
        return Err(TensorError::InvalidArgument(
            "conv-grad spatial dimensions cannot exceed input dimensions",
        ));
    }

    let kernel_h = in_h - conv_h + 1;
    let kernel_w = in_w - conv_w + 1;

    let transpose_start = Instant::now();
    let input_arr = mlx_array_from_tensor(input);
    let input_nhwc = mlx_transpose_axes(input_arr, &[0, 2, 3, 1], stream)?;
    let input_as_batches = mlx_transpose_axes(input_nhwc, &[3, 1, 2, 0], stream)?;

    let grad_nhwc = mlx_transpose_axes(conv_grad_nchw, &[0, 2, 3, 1], stream)?;
    let grad_as_kernels = mlx_transpose_axes(grad_nhwc, &[3, 1, 2, 0], stream)?;
    counters
        .native_dw_transpose_time_ns
        .fetch_add(transpose_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let conv_start = Instant::now();
    let mut kernel_grad_ic_h_w_oc = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_conv2d(
                &mut kernel_grad_ic_h_w_oc,
                input_as_batches,
                grad_as_kernels,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                stream,
            ),
            "mlx_conv2d kernel grad failed",
        )?;
    }
    counters
        .native_dw_conv_time_ns
        .fetch_add(conv_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let materialize_start = Instant::now();
    let kernel_grad_oc_ic_h_w = mlx_transpose_axes(kernel_grad_ic_h_w_oc, &[3, 0, 1, 2], stream)?;
    let kernel_grad = mlx_array_to_tensor(kernel_grad_oc_ic_h_w, stream)?;
    counters
        .native_dw_materialize_time_ns
        .fetch_add(materialize_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if kernel_grad.shape() != (out_channels, in_channels, kernel_h, kernel_w) {
        return Err(TensorError::IncompatibleShapes {
            left: kernel_grad.shape(),
            right: (out_channels, in_channels, kernel_h, kernel_w),
        });
    }

    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(input_nhwc);
        let _ = raw::mlx_array_free(input_as_batches);
        let _ = raw::mlx_array_free(grad_nhwc);
        let _ = raw::mlx_array_free(grad_as_kernels);
        let _ = raw::mlx_array_free(kernel_grad_ic_h_w_oc);
        let _ = raw::mlx_array_free(kernel_grad_oc_ic_h_w);
    }

    Ok(kernel_grad)
}

#[cfg(feature = "offloading-mlx")]
fn mlx_input_grad_native_from_conv_grad_array(
    conv_grad_nchw: raw::mlx_array,
    kernels: &Tensor4D,
    stream: raw::mlx_stream,
) -> Result<Tensor4D, TensorError> {
    let conv_grad_nhwc = mlx_transpose_axes(conv_grad_nchw, &[0, 2, 3, 1], stream)?;

    let kernels_arr = mlx_array_from_tensor(kernels);
    let kernels_ohwi = mlx_transpose_axes(kernels_arr, &[0, 2, 3, 1], stream)?;

    let mut input_grad_nhwc = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_conv_transpose2d(
                &mut input_grad_nhwc,
                conv_grad_nhwc,
                kernels_ohwi,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                stream,
            ),
            "mlx_conv_transpose2d input grad failed",
        )?;
    }

    let input_grad_nchw = mlx_transpose_axes(input_grad_nhwc, &[0, 3, 1, 2], stream)?;
    let input_grad = mlx_array_to_tensor(input_grad_nchw, stream)?;

    unsafe {
        let _ = raw::mlx_array_free(conv_grad_nhwc);
        let _ = raw::mlx_array_free(kernels_arr);
        let _ = raw::mlx_array_free(kernels_ohwi);
        let _ = raw::mlx_array_free(input_grad_nhwc);
        let _ = raw::mlx_array_free(input_grad_nchw);
    }

    Ok(input_grad)
}

#[cfg(feature = "offloading-mlx")]
fn cpu_kernel_grad_from_conv_grad(
    kernels: &Tensor4D,
    input: &Tensor4D,
    conv_grad: &Tensor4D,
) -> Result<Tensor4D, TensorError> {
    let (_, channels, conv_h, conv_w) = conv_grad.shape();
    let (_, in_channels, kernel_h, kernel_w) = kernels.shape();

    let mut kernel_grad = Tensor4D::zeros(channels, in_channels, kernel_h, kernel_w);
    let conv_plane = conv_h * conv_w;
    let (_, _, in_h, in_w) = input.shape();
    let input_plane = in_h * in_w;
    let kernel_plane = kernel_h * kernel_w;

    for out_c in 0..channels {
        let conv_channel_base = out_c * conv_plane;

        for in_c in 0..in_channels {
            let input_channel_base = in_c * input_plane;
            let kernel_channel_base = (out_c * in_channels + in_c) * kernel_plane;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let mut accum = 0.0f32;
                    for oy in 0..conv_h {
                        let conv_row_base = conv_channel_base + oy * conv_w;
                        let input_row_base = input_channel_base + (oy + ky) * in_w + kx;
                        for ox in 0..conv_w {
                            let grad = conv_grad.as_slice()[conv_row_base + ox];
                            let inp = input.as_slice()[input_row_base + ox];
                            accum += grad * inp;
                        }
                    }
                    kernel_grad.as_mut_slice()[kernel_channel_base + ky * kernel_w + kx] = accum;
                }
            }
        }
    }

    Ok(kernel_grad)
}

#[cfg(feature = "offloading-mlx")]
fn cpu_input_grad_from_conv_grad(
    kernels: &Tensor4D,
    conv_grad: &Tensor4D,
) -> Result<Tensor4D, TensorError> {
    let (_, channels, conv_h, conv_w) = conv_grad.shape();
    let (_, in_channels, kernel_h, kernel_w) = kernels.shape();
    let in_h = conv_h + kernel_h - 1;
    let in_w = conv_w + kernel_w - 1;

    let mut input_grad = Tensor4D::zeros(1, in_channels, in_h, in_w);
    let input_plane = in_h * in_w;
    let conv_plane = conv_h * conv_w;
    let kernel_plane = kernel_h * kernel_w;

    for in_c in 0..in_channels {
        let input_channel_base = in_c * input_plane;
        for iy in 0..in_h {
            for ix in 0..in_w {
                let mut accum = 0.0f32;
                for out_c in 0..channels {
                    let conv_channel_base = out_c * conv_plane;
                    let kernel_channel_base = (out_c * in_channels + in_c) * kernel_plane;
                    for ky in 0..kernel_h {
                        if iy < ky {
                            continue;
                        }
                        let oy = iy - ky;
                        if oy >= conv_h {
                            continue;
                        }
                        let conv_row_base = conv_channel_base + oy * conv_w;
                        for kx in 0..kernel_w {
                            if ix < kx {
                                continue;
                            }
                            let ox = ix - kx;
                            if ox >= conv_w {
                                continue;
                            }
                            let grad = conv_grad.as_slice()[conv_row_base + ox];
                            let weight = kernels.as_slice()[kernel_channel_base + ky * kernel_w + kx];
                            accum += grad * weight;
                        }
                    }
                }
                input_grad.as_mut_slice()[input_channel_base + iy * in_w + ix] = accum;
            }
        }
    }

    Ok(input_grad)
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

#[cfg(feature = "offloading-mlx")]
pub fn mlx_owned_array_from_tensor(tensor: &Tensor4D) -> Result<MlxOwnedArray, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let array = mlx_array_from_tensor(tensor);

    unsafe {
        mlx_check(raw::mlx_array_eval(array), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
        let _ = raw::mlx_stream_free(stream);
    }

    Ok(MlxOwnedArray { array })
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_owned_array_to_tensor(array: &MlxOwnedArray) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let result = mlx_array_to_tensor(array.array, stream);
    unsafe {
        let _ = raw::mlx_stream_free(stream);
    }
    result
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_apply_sgd_update_in_place(
    target: &mut MlxOwnedArray,
    gradient: &Tensor4D,
    learning_rate: f32,
    batch_size: f32,
) -> Result<(), TensorError> {
    let scale = if batch_size > 0.0 {
        -learning_rate / batch_size
    } else {
        -learning_rate
    };

    let mut scaled = gradient.clone();
    scaled.map_inplace(|value| value * scale);

    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let grad_arr = mlx_array_from_tensor(&scaled);
    let mut updated = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_add(&mut updated, target.array, grad_arr, stream),
            "mlx_add sgd update failed",
        )?;
        mlx_check(raw::mlx_array_eval(updated), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
        let _ = raw::mlx_array_free(grad_arr);
        let _ = raw::mlx_array_free(target.array);
        let _ = raw::mlx_stream_free(stream);
    }
    target.array = updated;
    Ok(())
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_add_tensor_in_place(
    target: &mut MlxOwnedArray,
    value: &Tensor4D,
) -> Result<(), TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let value_arr = mlx_array_from_tensor(value);
    let mut updated = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_add(&mut updated, target.array, value_arr, stream),
            "mlx_add accumulator failed",
        )?;
        mlx_check(raw::mlx_array_eval(updated), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
        let _ = raw::mlx_array_free(value_arr);
        let _ = raw::mlx_array_free(target.array);
        let _ = raw::mlx_stream_free(stream);
    }
    target.array = updated;
    Ok(())
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_zero_in_place(target: &mut MlxOwnedArray) -> Result<(), TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let mut zeros = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_zeros_like(&mut zeros, target.array, stream),
            "mlx_zeros_like failed",
        )?;
        mlx_check(raw::mlx_array_eval(zeros), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
        let _ = raw::mlx_array_free(target.array);
        let _ = raw::mlx_stream_free(stream);
    }
    target.array = zeros;
    Ok(())
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_apply_sgd_update_from_array_in_place(
    target: &mut MlxOwnedArray,
    gradient: &MlxOwnedArray,
    learning_rate: f32,
    batch_size: f32,
) -> Result<(), TensorError> {
    let scale = if batch_size > 0.0 {
        -learning_rate / batch_size
    } else {
        -learning_rate
    };

    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;
    let stream = mlx_stream_new();
    let scale_arr = unsafe { raw::mlx_array_new_float32(scale) };
    let mut scaled_grad = mlx_array_new();
    let mut updated = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_multiply(&mut scaled_grad, gradient.array, scale_arr, stream),
            "mlx_multiply sgd scale failed",
        )?;
        mlx_check(
            raw::mlx_add(&mut updated, target.array, scaled_grad, stream),
            "mlx_add sgd update failed",
        )?;
        mlx_check(raw::mlx_array_eval(updated), "mlx_array_eval failed")?;
        mlx_check(raw::mlx_synchronize(stream), "mlx_synchronize failed")?;
        let _ = raw::mlx_array_free(scale_arr);
        let _ = raw::mlx_array_free(scaled_grad);
        let _ = raw::mlx_array_free(target.array);
        let _ = raw::mlx_stream_free(stream);
    }
    target.array = updated;
    Ok(())
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_conv_blocks_to_feature_vec_with_mirrored_params(
    input: &Tensor4D,
    block1: (&MlxOwnedArray, (usize, usize, usize, usize), &MlxOwnedArray),
    block2: Option<(&MlxOwnedArray, (usize, usize, usize, usize), &MlxOwnedArray)>,
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

    let run_block = |inp_arr: raw::mlx_array,
                     inp_shape: (usize, usize, usize, usize),
                     kernels_arr: raw::mlx_array,
                     kernels_shape: (usize, usize, usize, usize),
                     bias_arr: raw::mlx_array|
     -> Result<(raw::mlx_array, (usize, usize, usize, usize)), TensorError> {
        let (n, in_c, in_h, in_w) = inp_shape;
        let (out_c, kernel_in_c, kh, kw) = kernels_shape;
        if in_c != kernel_in_c {
            return Err(TensorError::IncompatibleShapes {
                left: inp_shape,
                right: kernels_shape,
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
            let _ = raw::mlx_array_free(inp_nhwc);
            let _ = raw::mlx_array_free(kernels_ohwi);
        }

        let conv_nchw = mlx_transpose_axes(conv_out, &[0, 3, 1, 2], stream)?;
        unsafe { let _ = raw::mlx_array_free(conv_out); }

        let mut biased = mlx_array_new();
        unsafe {
            mlx_check(
                raw::mlx_add(&mut biased, conv_nchw, bias_arr, stream),
                "mlx_add bias failed",
            )?;
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
    let (b1_arr, b1_shape_meta, b1_bias_arr) = block1;
    let (b1_pooled, b1_shape) = run_block(input_arr, input_shape, b1_arr.array, b1_shape_meta, b1_bias_arr.array)?;

    let final_pooled = if let Some((k2_arr, k2_shape, b2_arr)) = block2 {
        let (b2_pooled, _) = run_block(b1_pooled, b1_shape, k2_arr.array, k2_shape, b2_arr.array)?;
        b2_pooled
    } else {
        b1_pooled
    };

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

    let gap_tensor = mlx_array_to_tensor(gap_out, stream);
    unsafe {
        let _ = raw::mlx_array_free(gap_out);
        let _ = raw::mlx_stream_free(stream);
    }
    Ok(gap_tensor?.first_sample_features())
}

#[cfg(feature = "offloading-mlx")]
pub fn mlx_conv2d_valid_with_mirrored_params(
    input: &Tensor4D,
    kernels: &MlxOwnedArray,
    kernels_shape: (usize, usize, usize, usize),
    bias: &MlxOwnedArray,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    let _guard = mlx_runtime_lock()
        .lock()
        .map_err(|_| TensorError::InvalidArgument("mlx runtime lock poisoned"))?;

    let (n, in_c, in_h, in_w) = input.shape();
    let (out_c, kernel_in_c, kh, kw) = kernels_shape;

    if in_c != kernel_in_c {
        return Err(TensorError::IncompatibleShapes {
            left: input.shape(),
            right: kernels_shape,
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
    let input_nhwc = mlx_transpose_axes(input_arr, &[0, 2, 3, 1], stream)?;
    let kernels_ohwi = mlx_transpose_axes(kernels.array, &[0, 2, 3, 1], stream)?;

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

    let mut biased = mlx_array_new();
    unsafe {
        mlx_check(
            raw::mlx_add(&mut biased, conv_nchw, bias.array, stream),
            "mlx_add failed",
        )?;
    }

    let result = mlx_array_to_tensor(biased, stream);
    unsafe {
        let _ = raw::mlx_array_free(input_arr);
        let _ = raw::mlx_array_free(input_nhwc);
        let _ = raw::mlx_array_free(kernels_ohwi);
        let _ = raw::mlx_array_free(conv_out);
        let _ = raw::mlx_array_free(conv_nchw);
        let _ = raw::mlx_array_free(biased);
        let _ = raw::mlx_stream_free(stream);
    }

    let _ = (n, out_c);
    result
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
        Ok(gap.first_sample_features())
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
