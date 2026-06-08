use std::env;

use crate::tensor::device::BackendTrainingCapabilities;
use crate::tensor::offloading::{cpu_backend, cuda_backend, mlx_backend};
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorBackendKind {
    Cpu,
    Cuda,
    Mlx,
}

impl TensorBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Mlx => "mlx",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" | "gpu" => Some(Self::Cuda),
            "mlx" => Some(Self::Mlx),
            "auto" | "default" | "" => None,
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct ActiveBackendSelection {
    kind: TensorBackendKind,
    backend: &'static dyn TensorBackend,
}

pub trait TensorBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn training_capabilities(&self) -> BackendTrainingCapabilities {
        BackendTrainingCapabilities::host_only()
    }

    fn conv2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError>;

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError>;

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError>;

    fn relu_inplace(&self, input: &mut Tensor4D);

    #[allow(clippy::too_many_arguments)]
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
        let mut conv = self.conv2d_valid(input, kernels, bias, conv_stride_h, conv_stride_w)?;
        self.relu_inplace(&mut conv);
        self.max_pool2d(
            &conv,
            pool_window_h,
            pool_window_w,
            pool_stride_h,
            pool_stride_w,
        )
    }

    /// Fused forward: conv→relu→pool (→ optional second block) → global average pool.
    /// Returns the flat feature vector directly. Backends that support on-device
    /// lazy evaluation (MLX) override this to avoid intermediate host round-trips.
    /// `block2` is `Some((kernels, bias))` when a second conv block is present.
    #[allow(clippy::too_many_arguments)]
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
        let block1_out = self.conv_relu_max_pool2d_valid(
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
            self.conv_relu_max_pool2d_valid(
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

        let gap = self.global_average_pool2d(&final_out)?;
        Ok(gap.first_sample_features())
    }

    #[allow(clippy::too_many_arguments)]
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
}

#[derive(Debug, Clone)]
pub struct ConvBlockBackwardGradients {
    pub kernel_grad: Tensor4D,
    pub bias_grad: Vec<f32>,
    pub input_grad: Option<Tensor4D>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_conv_block_backward_gradients(
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

    let (batch, channels, pooled_h, pooled_w) = pooled_shape;
    let (input_batch, _, in_h, in_w) = input.shape();
    let (conv_batch, _, relu_h, relu_w) = conv_pre_activation.shape();

    if input_batch != batch || conv_batch != batch {
        return Err(TensorError::IncompatibleShapes {
            left: pooled_shape,
            right: conv_pre_activation.shape(),
        });
    }

    let expected_pool_indices = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(pooled_h))
        .and_then(|v| v.checked_mul(pooled_w))
        .ok_or(TensorError::InvalidArgument("pool index shape overflow"))?;

    if pool_indices.len() != expected_pool_indices {
        return Err(TensorError::ShapeMismatch {
            expected: expected_pool_indices,
            actual: pool_indices.len(),
        });
    }

    let mut conv_grad = Tensor4D::zeros(batch, channels, relu_h, relu_w);
    let relu_batch_plane = channels * relu_h * relu_w;
    let relu_plane = relu_h * relu_w;
    let pooled_batch_plane = channels * pooled_h * pooled_w;
    let pooled_plane = pooled_h * pooled_w;

    for sample in 0..batch {
        let conv_sample_base = sample * relu_batch_plane;
        let pooled_sample_base = sample * pooled_batch_plane;

        for channel in 0..channels {
            let conv_channel_base = conv_sample_base + channel * relu_plane;
            let pooled_channel_base = pooled_sample_base + channel * pooled_plane;

            for py in 0..pooled_h {
                for px in 0..pooled_w {
                    let pooled_idx = pooled_channel_base + py * pooled_w + px;
                    let (src_y, src_x) = pool_indices[pooled_idx];
                    if src_y >= relu_h || src_x >= relu_w {
                        return Err(TensorError::InvalidArgument(
                            "pool indices exceed relu feature map bounds",
                        ));
                    }
                    let conv_idx = conv_channel_base + src_y * relu_w + src_x;
                    conv_grad.as_mut_slice()[conv_idx] += pooled_grad.as_slice()[pooled_idx];
                }
            }
        }
    }

    for (grad, pre) in conv_grad
        .as_mut_slice()
        .iter_mut()
        .zip(conv_pre_activation.as_slice().iter())
    {
        if *pre <= 0.0 {
            *grad = 0.0;
        }
    }

    let (_, in_channels, kernel_h, kernel_w) = kernels.shape();
    let (_, _, conv_h, conv_w) = conv_pre_activation.shape();

    let mut kernel_grad = Tensor4D::zeros(channels, in_channels, kernel_h, kernel_w);
    let mut bias_grad = vec![0.0f32; channels];
    let conv_plane = conv_h * conv_w;
    let input_plane = in_h * in_w;
    let input_batch_plane = in_channels * input_plane;
    let kernel_plane = kernel_h * kernel_w;

    for (out_c, bias_slot) in bias_grad.iter_mut().enumerate() {
        let mut bias_accum = 0.0f32;
        for sample in 0..batch {
            let conv_channel_base = sample * relu_batch_plane + out_c * conv_plane;
            let conv_channel =
                &conv_grad.as_slice()[conv_channel_base..conv_channel_base + conv_plane];
            bias_accum += conv_channel.iter().copied().sum::<f32>();
        }
        *bias_slot = bias_accum;

        for in_c in 0..in_channels {
            let kernel_channel_base = (out_c * in_channels + in_c) * kernel_plane;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let mut accum = 0.0f32;
                    for sample in 0..batch {
                        let conv_channel_base = sample * relu_batch_plane + out_c * conv_plane;
                        let input_channel_base = sample * input_batch_plane + in_c * input_plane;
                        for oy in 0..conv_h {
                            let conv_row_base = conv_channel_base + oy * conv_w;
                            let input_row_base = input_channel_base + (oy + ky) * in_w + kx;
                            for ox in 0..conv_w {
                                let grad = conv_grad.as_slice()[conv_row_base + ox];
                                let inp = input.as_slice()[input_row_base + ox];
                                accum += grad * inp;
                            }
                        }
                    }
                    kernel_grad.as_mut_slice()[kernel_channel_base + ky * kernel_w + kx] = accum;
                }
            }
        }

    }

    let input_grad = if compute_input_grad {

        let mut input_grad = Tensor4D::zeros(batch, in_channels, in_h, in_w);

        for sample in 0..batch {
            for in_c in 0..in_channels {
                let input_channel_base = sample * input_batch_plane + in_c * input_plane;
                for iy in 0..in_h {
                    for ix in 0..in_w {
                        let mut accum = 0.0f32;
                        for out_c in 0..channels {
                            let conv_channel_base = sample * relu_batch_plane + out_c * conv_plane;
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
                                    let weight =
                                        kernels.as_slice()[kernel_channel_base + ky * kernel_w + kx];
                                    accum += grad * weight;
                                }
                            }
                        }
                        input_grad.as_mut_slice()[input_channel_base + iy * in_w + ix] = accum;
                    }
                }
            }
        }

        Some(input_grad)

    } else {
        None
    };

    Ok(ConvBlockBackwardGradients {
        kernel_grad,
        bias_grad,
        input_grad,
    })

}

pub use cpu_backend::CpuTensorBackend;
pub use cuda_backend::CudaTensorBackend;
pub use mlx_backend::{
    MlxBackpropPathSnapshot,
    MlxTensorBackend,
    mlx_backprop_path_reset,
    mlx_backprop_path_snapshot,
};

pub fn cpu_backend() -> CpuTensorBackend {
    cpu_backend::cpu_backend()
}

pub fn cuda_backend() -> CudaTensorBackend {
    cuda_backend::cuda_backend()
}

pub fn mlx_backend() -> MlxTensorBackend {
    mlx_backend::mlx_backend()
}

pub fn cuda_backend_available() -> bool {
    cuda_backend::cuda_backend_available()
}

pub fn mlx_backend_available() -> bool {
    mlx_backend::mlx_backend_available()
}

pub fn preferred_backend_kind() -> Option<TensorBackendKind> {
    env::var("NEURALNET_TENSOR_BACKEND")
        .ok()
        .or_else(|| env::var("NEURALNET_BACKEND").ok())
        .and_then(|value| TensorBackendKind::parse(value.as_str()))
}

fn backend_for_kind(kind: TensorBackendKind) -> Option<ActiveBackendSelection> {
    match kind {
        TensorBackendKind::Cpu => {
            static CPU: CpuTensorBackend = CpuTensorBackend;
            Some(ActiveBackendSelection {
                kind: TensorBackendKind::Cpu,
                backend: &CPU,
            })
        }
        TensorBackendKind::Cuda => {
            #[cfg(feature = "offloading-cuda")]
            {
                if cuda_backend_available() {
                    static CUDA: CudaTensorBackend = CudaTensorBackend;
                    return Some(ActiveBackendSelection {
                        kind: TensorBackendKind::Cuda,
                        backend: &CUDA,
                    });
                }
            }
            None
        }
        TensorBackendKind::Mlx => {
            #[cfg(feature = "offloading-mlx")]
            {
                if mlx_backend_available() {
                    static MLX: MlxTensorBackend = MlxTensorBackend;
                    return Some(ActiveBackendSelection {
                        kind: TensorBackendKind::Mlx,
                        backend: &MLX,
                    });
                }
            }
            None
        }
    }
}

fn resolve_active_backend() -> ActiveBackendSelection {
    if let Some(preferred) = preferred_backend_kind()
        && let Some(selection) = backend_for_kind(preferred)
    {
        return selection;
    }

    for fallback in [
        TensorBackendKind::Cuda,
        TensorBackendKind::Mlx,
        TensorBackendKind::Cpu,
    ] {
        if let Some(selection) = backend_for_kind(fallback) {
            return selection;
        }
    }

    unreachable!("cpu backend must always be available")
}

pub fn active_backend() -> &'static dyn TensorBackend {
    resolve_active_backend().backend
}

pub fn active_backend_name() -> &'static str {
    active_backend().name()
}

pub fn active_backend_training_capabilities() -> BackendTrainingCapabilities {
    active_backend().training_capabilities()
}

pub fn active_backend_label() -> &'static str {
    match resolve_active_backend().kind {
        TensorBackendKind::Cpu => "cpu",
        TensorBackendKind::Cuda => "cuda",
        TensorBackendKind::Mlx => {
            #[cfg(feature = "offloading-mlx")]
            {
                mlx_backend::mlx_backend_label()
            }
            #[cfg(not(feature = "offloading-mlx"))]
            {
                "mlx"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_slices(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - b).abs();
            assert!(
                delta <= tol,
                "slice mismatch at index {idx}: actual={a}, expected={b}, delta={delta}, tol={tol}"
            );
        }
    }

    fn assert_close_tensor(actual: &Tensor4D, expected: &Tensor4D, tol: f32) {
        assert_eq!(actual.shape(), expected.shape());
        assert_close_slices(actual.as_slice(), expected.as_slice(), tol);
    }

    #[test]
    fn cpu_backend_reports_host_only_training_capabilities() {
        let caps = cpu_backend().training_capabilities();
        assert!(!caps.native_forward_execution);
        assert!(!caps.native_backward_execution);
        assert!(caps.host_materializes_gradients);
        assert!(!caps.supports_device_resident_training());
    }

    #[test]
    fn active_backend_training_capabilities_match_backend_mode() {
        let caps = active_backend_training_capabilities();

        #[cfg(all(not(feature = "offloading-cuda"), not(feature = "offloading-mlx")))]
        {
            assert!(!caps.native_forward_execution);
            assert!(!caps.native_backward_execution);
        }

        #[cfg(any(feature = "offloading-cuda", feature = "offloading-mlx"))]
        {
            assert!(caps.native_forward_execution);
            assert!(caps.native_backward_execution);
            assert!(caps.host_materializes_gradients);
            assert!(!caps.supports_device_resident_training());
        }
    }

    #[test]
    fn cpu_backend_conv_matches_tensor4d_conv() {
        let backend = cpu_backend();

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

        let direct = input
            .conv2d_valid(&kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("direct conv should succeed"));

        let via_backend = backend
            .conv2d_valid(&input, &kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("backend conv should succeed"));

        assert_eq!(direct, via_backend);
        assert_eq!(backend.name(), "cpu");
        
    }

    #[test]
    fn cuda_backend_matches_cpu_ops_in_staged_mode() {
        let backend = cuda_backend();
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

        let conv_via_cpu = input
            .conv2d_valid(&kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("cpu convolution should succeed"));
        let conv_via_cuda_backend = backend
            .conv2d_valid(&input, &kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("cuda staged convolution should succeed"));

        let pool_via_cpu = conv_via_cpu
            .max_pool2d(2, 2, 1, 1)
            .unwrap_or_else(|_| panic!("cpu max pooling should succeed"));
        let pool_via_cuda_backend = backend
            .max_pool2d(&conv_via_cuda_backend, 2, 2, 1, 1)
            .unwrap_or_else(|_| panic!("cuda staged max pooling should succeed"));

        let gap_via_cpu = pool_via_cpu
            .global_average_pool2d()
            .unwrap_or_else(|_| panic!("cpu global average pooling should succeed"));
        let gap_via_cuda_backend = backend
            .global_average_pool2d(&pool_via_cuda_backend)
            .unwrap_or_else(|_| panic!("cuda staged global average pooling should succeed"));

        assert_eq!(conv_via_cuda_backend, conv_via_cpu);
        assert_eq!(pool_via_cuda_backend, pool_via_cpu);
        assert_eq!(gap_via_cuda_backend, gap_via_cpu);

        assert_eq!(backend.name(), "cuda");
        assert_eq!(cuda_backend_available(), cfg!(feature = "offloading-cuda"));
    }

    #[test]
    fn mlx_backend_executes_tensor_ops() {
        let backend = mlx_backend();
        let input = Tensor4D::zeros(1, 1, 3, 3);
        let kernels = Tensor4D::zeros(1, 1, 2, 2);

        let conv = backend.conv2d_valid(&input, &kernels, None, 1, 1);
        assert!(conv.is_ok());

        assert_eq!(backend.name(), "mlx");
        assert_eq!(mlx_backend_available(), cfg!(feature = "offloading-mlx"));
    }

    #[cfg(feature = "offloading-mlx")]
    #[test]
    fn mlx_conv_block_backward_matches_cpu_reference() {
        let backend = mlx_backend();

        let kernels = Tensor4D::from_vec(
            2,
            2,
            3,
            3,
            vec![
                0.10, -0.20, 0.30, 0.00, 0.15, -0.05, 0.20, 0.10, -0.25,
                -0.10, 0.05, 0.15, 0.25, -0.30, 0.20, 0.10, -0.05, 0.35,
                0.05, 0.10, -0.15, 0.20, 0.25, -0.05, -0.10, 0.30, 0.10,
                -0.20, 0.15, 0.05, -0.10, 0.00, 0.20, 0.25, -0.15, 0.10,
            ],
        )
        .unwrap_or_else(|_| panic!("kernels should be valid"));

        let input = Tensor4D::from_vec(
            1,
            2,
            6,
            6,
            vec![
                0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                0.35, 0.45, 0.55, 0.65, 0.75, 0.85,
                0.60, 0.50, 0.40, 0.30, 0.20, 0.10,
                0.55, 0.45, 0.35, 0.25, 0.15, 0.05,
                0.50, 0.40, 0.30, 0.20, 0.10, 0.00,
                0.45, 0.35, 0.25, 0.15, 0.05, -0.05,
                0.40, 0.30, 0.20, 0.10, 0.00, -0.10,
                0.35, 0.25, 0.15, 0.05, -0.05, -0.15,
            ],
        )
        .unwrap_or_else(|_| panic!("input should be valid"));

        let conv_pre_activation = Tensor4D::from_vec(
            1,
            2,
            4,
            4,
            vec![
                0.8, -0.2, 0.5, 0.1,
                0.4, 0.6, -0.1, 0.3,
                -0.4, 0.7, 0.2, -0.6,
                0.5, -0.3, 0.9, 0.0,
                -0.1, 0.2, 0.3, -0.4,
                0.6, -0.5, 0.7, 0.8,
                0.9, 0.1, -0.2, 0.4,
                -0.6, 0.5, 0.0, 0.2,
            ],
        )
        .unwrap_or_else(|_| panic!("conv pre-activation should be valid"));

        let pooled_shape = (1, 2, 2, 2);
        let pool_indices = vec![
            (0, 0), (0, 2),
            (2, 1), (3, 2),
            (1, 1), (1, 3),
            (2, 0), (3, 3),
        ];

        let pooled_grad = Tensor4D::from_vec(
            1,
            2,
            2,
            2,
            vec![
                0.20, -0.10,
                0.35, 0.40,
                -0.15, 0.25,
                0.05, -0.30,
            ],
        )
        .unwrap_or_else(|_| panic!("pooled gradient should be valid"));

        let cpu = cpu_conv_block_backward_gradients(
            &kernels,
            &input,
            &conv_pre_activation,
            &pool_indices,
            pooled_shape,
            &pooled_grad,
            true,
        )
        .unwrap_or_else(|_| panic!("cpu reference gradients should succeed"));

        let mlx = backend
            .conv_block_backward_gradients(
                &kernels,
                &input,
                &conv_pre_activation,
                &pool_indices,
                pooled_shape,
                &pooled_grad,
                true,
            )
            .unwrap_or_else(|_| panic!("mlx gradients should succeed"));

        assert_close_tensor(&mlx.kernel_grad, &cpu.kernel_grad, 1e-4);
        assert_close_slices(&mlx.bias_grad, &cpu.bias_grad, 1e-5);
        let mlx_input_grad = mlx
            .input_grad
            .as_ref()
            .unwrap_or_else(|| panic!("mlx input grad should be present"));
        let cpu_input_grad = cpu
            .input_grad
            .as_ref()
            .unwrap_or_else(|| panic!("cpu input grad should be present"));
        assert_close_tensor(mlx_input_grad, cpu_input_grad, 1e-4);
    }

    #[cfg(feature = "offloading-mlx")]
    #[test]
    fn mlx_batched_conv_block_backward_matches_cpu_reference() {
        let backend = mlx_backend();

        let kernels = Tensor4D::from_vec(
            2,
            2,
            3,
            3,
            vec![
                0.10, -0.20, 0.30, 0.00, 0.15, -0.05, 0.20, 0.10, -0.25,
                -0.10, 0.05, 0.15, 0.25, -0.30, 0.20, 0.10, -0.05, 0.35,
                0.05, 0.10, -0.15, 0.20, 0.25, -0.05, -0.10, 0.30, 0.10,
                -0.20, 0.15, 0.05, -0.10, 0.00, 0.20, 0.25, -0.15, 0.10,
            ],
        )
        .unwrap_or_else(|_| panic!("kernels should be valid"));

        let input = Tensor4D::from_vec(
            2,
            2,
            6,
            6,
            vec![
                0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                0.35, 0.45, 0.55, 0.65, 0.75, 0.85,
                0.60, 0.50, 0.40, 0.30, 0.20, 0.10,
                0.55, 0.45, 0.35, 0.25, 0.15, 0.05,
                0.50, 0.40, 0.30, 0.20, 0.10, 0.00,
                0.45, 0.35, 0.25, 0.15, 0.05, -0.05,
                0.40, 0.30, 0.20, 0.10, 0.00, -0.10,
                0.35, 0.25, 0.15, 0.05, -0.05, -0.15,
                0.12, 0.18, 0.24, 0.30, 0.36, 0.42,
                0.16, 0.22, 0.28, 0.34, 0.40, 0.46,
                0.20, 0.26, 0.32, 0.38, 0.44, 0.50,
                0.24, 0.30, 0.36, 0.42, 0.48, 0.54,
                0.28, 0.34, 0.40, 0.46, 0.52, 0.58,
                0.32, 0.38, 0.44, 0.50, 0.56, 0.62,
                0.48, 0.42, 0.36, 0.30, 0.24, 0.18,
                0.44, 0.38, 0.32, 0.26, 0.20, 0.14,
                0.40, 0.34, 0.28, 0.22, 0.16, 0.10,
                0.36, 0.30, 0.24, 0.18, 0.12, 0.06,
                0.32, 0.26, 0.20, 0.14, 0.08, 0.02,
                0.28, 0.22, 0.16, 0.10, 0.04, -0.02,
            ],
        )
        .unwrap_or_else(|_| panic!("input should be valid"));

        let conv_pre_activation = Tensor4D::from_vec(
            2,
            2,
            4,
            4,
            vec![
                0.8, -0.2, 0.5, 0.1,
                0.4, 0.6, -0.1, 0.3,
                -0.4, 0.7, 0.2, -0.6,
                0.5, -0.3, 0.9, 0.0,
                -0.1, 0.2, 0.3, -0.4,
                0.6, -0.5, 0.7, 0.8,
                0.9, 0.1, -0.2, 0.4,
                -0.6, 0.5, 0.0, 0.2,
                0.7, -0.1, 0.4, 0.2,
                0.3, 0.5, -0.2, 0.6,
                -0.3, 0.8, 0.1, -0.5,
                0.4, -0.2, 1.0, 0.1,
                -0.2, 0.1, 0.4, -0.3,
                0.5, -0.4, 0.6, 0.7,
                0.8, 0.2, -0.1, 0.5,
                -0.5, 0.4, 0.1, 0.3,
            ],
        )
        .unwrap_or_else(|_| panic!("conv pre-activation should be valid"));

        let pooled_shape = (2, 2, 2, 2);
        let pool_indices = vec![
            (0, 0), (0, 2),
            (2, 1), (3, 2),
            (1, 1), (1, 3),
            (2, 0), (3, 3),
            (0, 1), (0, 3),
            (2, 2), (3, 2),
            (1, 0), (1, 2),
            (2, 1), (3, 3),
        ];

        let pooled_grad = Tensor4D::from_vec(
            2,
            2,
            2,
            2,
            vec![
                0.20, -0.10,
                0.35, 0.40,
                -0.15, 0.25,
                0.05, -0.30,
                0.12, -0.08,
                0.28, 0.33,
                -0.10, 0.21,
                0.07, -0.24,
            ],
        )
        .unwrap_or_else(|_| panic!("pooled gradient should be valid"));

        let cpu = cpu_conv_block_backward_gradients(
            &kernels,
            &input,
            &conv_pre_activation,
            &pool_indices,
            pooled_shape,
            &pooled_grad,
            true,
        )
        .unwrap_or_else(|_| panic!("cpu reference gradients should succeed"));

        let mlx = backend
            .conv_block_backward_gradients(
                &kernels,
                &input,
                &conv_pre_activation,
                &pool_indices,
                pooled_shape,
                &pooled_grad,
                true,
            )
            .unwrap_or_else(|_| panic!("mlx gradients should succeed"));

        assert_close_tensor(&mlx.kernel_grad, &cpu.kernel_grad, 1e-4);
        assert_close_slices(&mlx.bias_grad, &cpu.bias_grad, 1e-5);
        let mlx_input_grad = mlx
            .input_grad
            .as_ref()
            .unwrap_or_else(|| panic!("mlx input grad should be present"));
        let cpu_input_grad = cpu
            .input_grad
            .as_ref()
            .unwrap_or_else(|| panic!("cpu input grad should be present"));
        assert_close_tensor(mlx_input_grad, cpu_input_grad, 1e-4);
    }

    #[cfg(all(not(feature = "offloading-cuda"), not(feature = "offloading-mlx")))]
    #[test]
    fn active_backend_defaults_to_cpu_without_offloading_features() {
        assert_eq!(active_backend_name(), "cpu");
    }
}
