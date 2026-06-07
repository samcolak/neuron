use crate::tensor::backend::TensorBackend;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuTensorBackend;

impl TensorBackend for CpuTensorBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn conv2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        input.conv2d_valid_cpu(kernels, bias, stride_h, stride_w)
    }

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        input.max_pool2d_cpu(window_h, window_w, stride_h, stride_w)
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        input.global_average_pool2d_cpu()
    }

    fn relu_inplace(&self, input: &mut Tensor4D) {
        input.relu_inplace_cpu()
    }
}

pub fn cpu_backend() -> CpuTensorBackend {
    CpuTensorBackend
}
