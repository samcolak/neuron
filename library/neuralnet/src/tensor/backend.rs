use crate::tensor::tensor4d::{Tensor4D, TensorError};
use crate::tensor::offloading::{cpu_backend, cuda_backend, mlx_backend};

#[cfg(all(feature = "offloading-cuda", feature = "offloading-mlx"))]
compile_error!("features 'offloading-cuda' and 'offloading-mlx' are mutually exclusive for active runtime backend selection");

pub trait TensorBackend: Send + Sync {
    fn name(&self) -> &'static str;

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
        Ok(gap.flatten_batch_features().first().cloned().unwrap_or_default())
    }
}

pub use cpu_backend::CpuTensorBackend;
pub use cuda_backend::CudaTensorBackend;
pub use mlx_backend::MlxTensorBackend;

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

pub fn active_backend() -> &'static dyn TensorBackend {
    #[cfg(feature = "offloading-cuda")]
    {
        static CUDA: CudaTensorBackend = CudaTensorBackend;
        return &CUDA;
    }

    #[cfg(all(not(feature = "offloading-cuda"), feature = "offloading-mlx"))]
    {
        static MLX: MlxTensorBackend = MlxTensorBackend;
        return &MLX;
    }

    #[cfg(all(not(feature = "offloading-cuda"), not(feature = "offloading-mlx")))]
    {
        static CPU: CpuTensorBackend = CpuTensorBackend;
        &CPU
    }
}

pub fn active_backend_name() -> &'static str {
    active_backend().name()
}

pub fn active_backend_label() -> &'static str {
    #[cfg(feature = "offloading-cuda")]
    {
        return "cuda";
    }

    #[cfg(all(not(feature = "offloading-cuda"), feature = "offloading-mlx"))]
    {
        return mlx_backend::mlx_backend_label();
    }

    #[cfg(all(not(feature = "offloading-cuda"), not(feature = "offloading-mlx")))]
    {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn cuda_backend_stub_reports_unavailable_ops() {
        let backend = cuda_backend();
        let input = Tensor4D::zeros(1, 1, 3, 3);
        let kernels = Tensor4D::zeros(1, 1, 2, 2);

        let conv = backend.conv2d_valid(&input, &kernels, None, 1, 1);
        assert!(matches!(
            conv,
            Err(TensorError::InvalidArgument(
                "cuda conv2d stub is not implemented yet"
            ))
        ));

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

    #[cfg(all(not(feature = "offloading-cuda"), not(feature = "offloading-mlx")))]
    #[test]
    fn active_backend_defaults_to_cpu_without_offloading_features() {
        assert_eq!(active_backend_name(), "cpu");
    }
}
