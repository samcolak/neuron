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
