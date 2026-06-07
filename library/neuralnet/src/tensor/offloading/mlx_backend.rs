use crate::tensor::backend::TensorBackend;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

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
    
}

pub fn mlx_backend() -> MlxTensorBackend {
    MlxTensorBackend
}

pub fn mlx_backend_available() -> bool {
    cfg!(feature = "offloading-mlx")
}

fn mlx_conv2d_valid(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: Option<&[f32]>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    // Phase 1 MLX path: delegate to validated CPU tensor kernels while keeping
    // the backend wiring and feature-gated runtime selection in place.
    input.conv2d_valid_cpu(kernels, bias, stride_h, stride_w)
}

fn mlx_max_pool2d(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    input.max_pool2d_cpu(window_h, window_w, stride_h, stride_w)
}

fn mlx_global_average_pool2d(input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    input.global_average_pool2d_cpu()
}

fn mlx_relu_inplace(input: &mut Tensor4D) {
    input.relu_inplace_cpu()
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
    
}
