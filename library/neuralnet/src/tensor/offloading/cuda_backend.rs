use crate::tensor::backend::TensorBackend;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

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
        cuda_conv2d_valid_stub(input, kernels, bias, stride_h, stride_w)
    }

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        cuda_max_pool2d_stub(input, window_h, window_w, stride_h, stride_w)
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        cuda_global_average_pool2d_stub(input)
    }

    fn relu_inplace(&self, input: &mut Tensor4D) {
        cuda_relu_inplace_stub(input)
    }
}

pub fn cuda_backend() -> CudaTensorBackend {
    CudaTensorBackend
}

pub fn cuda_backend_available() -> bool {
    cfg!(feature = "offloading-cuda")
}

fn cuda_conv2d_valid_stub(
    _input: &Tensor4D,
    _kernels: &Tensor4D,
    _bias: Option<&[f32]>,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    Err(TensorError::InvalidArgument(
        "cuda conv2d stub is not implemented yet",
    ))
}

fn cuda_max_pool2d_stub(
    _input: &Tensor4D,
    _window_h: usize,
    _window_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<Tensor4D, TensorError> {
    Err(TensorError::InvalidArgument(
        "cuda max_pool2d stub is not implemented yet",
    ))
}

fn cuda_global_average_pool2d_stub(_input: &Tensor4D) -> Result<Tensor4D, TensorError> {
    Err(TensorError::InvalidArgument(
        "cuda global_average_pool2d stub is not implemented yet",
    ))
}

fn cuda_relu_inplace_stub(_input: &mut Tensor4D) {}
