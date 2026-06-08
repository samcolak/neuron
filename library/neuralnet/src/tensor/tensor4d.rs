use std::error::Error;
use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

use crate::tensor::backend::{active_backend, TensorBackend};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {

    ShapeMismatch {
        expected: usize,
        actual: usize,
    },

    IncompatibleShapes {
        left: (usize, usize, usize, usize),
        right: (usize, usize, usize, usize),
    },

    OutOfBounds {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    },

    InvalidArgument(&'static str),
}

impl Display for TensorError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor shape mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
            Self::IncompatibleShapes { left, right } => {
                write!(
                    f,
                    "incompatible tensor shapes: left={:?}, right={:?}",
                    left, right
                )
            }
            Self::OutOfBounds { n, c, h, w } => {
                write!(
                    f,
                    "tensor index out of bounds at n={}, c={}, h={}, w={}",
                    n, c, h, w
                )
            }
            Self::InvalidArgument(message) => write!(f, "invalid tensor argument: {}", message),
        }
    }
    
}

impl Error for TensorError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor4D {
    data: Vec<f32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

impl Tensor4D {
    pub fn new(n: usize, c: usize, h: usize, w: usize, value: f32) -> Self {
        let len = n.saturating_mul(c).saturating_mul(h).saturating_mul(w);
        Self {
            data: vec![value; len],
            n,
            c,
            h,
            w,
        }
    }

    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self::new(n, c, h, w, 0.0)
    }

    pub fn from_vec(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        data: Vec<f32>,
    ) -> Result<Self, TensorError> {
        let expected = n
            .checked_mul(c)
            .and_then(|x| x.checked_mul(h))
            .and_then(|x| x.checked_mul(w))
            .unwrap_or(usize::MAX);

        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self { data, n, c, h, w })
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.n, self.c, self.h, self.w)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }

    pub fn flatten_batch_features(&self) -> Vec<Vec<f32>> {
        if self.n == 0 {
            return Vec::new();
        }

        let per_sample = self.c.saturating_mul(self.h).saturating_mul(self.w);
        if per_sample == 0 {
            return vec![Vec::new(); self.n];
        }

        self.data
            .chunks_exact(per_sample)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Returns the feature vector for the first sample without allocating
    /// the outer `Vec<Vec<f32>>` that `flatten_batch_features` requires.
    /// Use this wherever only the first batch sample is needed.
    pub fn first_sample_features(&self) -> Vec<f32> {
        if self.n == 0 {
            return Vec::new();
        }
        let per_sample = self.c.saturating_mul(self.h).saturating_mul(self.w);
        if per_sample == 0 {
            return Vec::new();
        }
        self.data[..per_sample.min(self.data.len())].to_vec()
    }

    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> Result<f32, TensorError> {
        let idx = self.offset(n, c, h, w)?;
        Ok(self.data[idx])
    }

    pub fn set(
        &mut self,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        value: f32,
    ) -> Result<(), TensorError> {
        let idx = self.offset(n, c, h, w)?;
        self.data[idx] = value;
        Ok(())
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(f32) -> f32,
    {
        for item in &mut self.data {
            *item = f(*item);
        }
    }

    pub fn relu_inplace(&mut self) {
        self.relu_inplace_with_backend(active_backend());
    }

    pub(crate) fn relu_inplace_cpu(&mut self) {
        self.map_inplace(|value| value.max(0.0));
    }

    pub fn add_inplace(&mut self, other: &Self) -> Result<(), TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape(),
                right: other.shape(),
            });
        }

        for (left, right) in self.data.iter_mut().zip(other.data.iter()) {
            *left += *right;
        }

        Ok(())
    }

    pub fn conv2d_valid(
        &self,
        kernels: &Self,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {
        self.conv2d_valid_with_backend(active_backend(), kernels, bias, stride_h, stride_w)
    }

    pub(crate) fn conv2d_valid_cpu(
        &self,
        kernels: &Self,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {

        if stride_h == 0 || stride_w == 0 {
            return Err(TensorError::InvalidArgument("stride must be greater than zero"));
        }

        if self.c != kernels.c {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape(),
                right: kernels.shape(),
            });
        }

        if kernels.n == 0 || kernels.h == 0 || kernels.w == 0 {
            return Err(TensorError::InvalidArgument(
                "kernel shape must have non-zero output channels and spatial size",
            ));
        }

        if self.h < kernels.h || self.w < kernels.w {
            return Err(TensorError::InvalidArgument(
                "kernel spatial size cannot exceed input spatial size",
            ));
        }

        if let Some(bias_values) = bias
            && bias_values.len() != kernels.n
        {
            return Err(TensorError::ShapeMismatch {
                expected: kernels.n,
                actual: bias_values.len(),
            });
        }

        let out_h = ((self.h - kernels.h) / stride_h) + 1;
        let out_w = ((self.w - kernels.w) / stride_w) + 1;
        let mut output = Tensor4D::zeros(self.n, kernels.n, out_h, out_w);

        let input_channel_stride = self.h * self.w;
        let input_batch_stride = self.c * input_channel_stride;
        let kernel_channel_stride = kernels.h * kernels.w;
        let kernel_out_stride = kernels.c * kernel_channel_stride;
        let output_channel_stride = out_h * out_w;
        let output_batch_stride = kernels.n * output_channel_stride;
        let bias_values = bias.unwrap_or(&[]);

        for batch in 0..self.n {
            let input_batch_base = batch * input_batch_stride;
            let output_batch_base = batch * output_batch_stride;

            for out_c in 0..kernels.n {
                let bias_value = bias_values.get(out_c).copied().unwrap_or(0.0);
                let kernel_out_base = out_c * kernel_out_stride;
                let output_channel_base = output_batch_base + out_c * output_channel_stride;

                for out_y in 0..out_h {
                    let in_y = out_y * stride_h;
                    let output_row_base = output_channel_base + out_y * out_w;

                    for out_x in 0..out_w {
                        let in_x = out_x * stride_w;
                        let mut acc = bias_value;

                        for in_c in 0..self.c {
                            let input_channel_base = input_batch_base + in_c * input_channel_stride;
                            let kernel_channel_base = kernel_out_base + in_c * kernel_channel_stride;

                            for ky in 0..kernels.h {
                                let input_row_base = input_channel_base + (in_y + ky) * self.w + in_x;
                                let kernel_row_base = kernel_channel_base + ky * kernels.w;
                                let input_row = &self.data[input_row_base..input_row_base + kernels.w];
                                let kernel_row = &kernels.data[kernel_row_base..kernel_row_base + kernels.w];

                                for (input_value, kernel_value) in
                                    input_row.iter().zip(kernel_row.iter())
                                {
                                    acc += *input_value * *kernel_value;
                                }
                            }
                        }

                        output.data[output_row_base + out_x] = acc;
                    }
                }
            }
        }

        Ok(output)

    }

    pub fn conv2d_valid_with_backend<B: TensorBackend + ?Sized>(
        &self,
        backend: &B,
        kernels: &Self,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {
        backend.conv2d_valid(self, kernels, bias, stride_h, stride_w)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_relu_max_pool2d_valid(
        &self,
        kernels: &Self,
        bias: Option<&[f32]>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    ) -> Result<Self, TensorError> {
        self.conv_relu_max_pool2d_valid_with_backend(
            active_backend(),
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

    #[allow(clippy::too_many_arguments)]
    pub fn conv_relu_max_pool2d_valid_with_backend<B: TensorBackend + ?Sized>(
        &self,
        backend: &B,
        kernels: &Self,
        bias: Option<&[f32]>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    ) -> Result<Self, TensorError> {
        backend.conv_relu_max_pool2d_valid(
            self,
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

    pub fn max_pool2d(
        &self,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {
        self.max_pool2d_with_backend(active_backend(), window_h, window_w, stride_h, stride_w)
    }

    pub(crate) fn max_pool2d_cpu(
        &self,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {
        if window_h == 0 || window_w == 0 {
            return Err(TensorError::InvalidArgument(
                "pooling window must be greater than zero",
            ));
        }
        if stride_h == 0 || stride_w == 0 {
            return Err(TensorError::InvalidArgument("stride must be greater than zero"));
        }
        if self.h < window_h || self.w < window_w {
            return Err(TensorError::InvalidArgument(
                "pooling window cannot exceed input spatial size",
            ));
        }

        let out_h = ((self.h - window_h) / stride_h) + 1;
        let out_w = ((self.w - window_w) / stride_w) + 1;
        let mut output = Tensor4D::zeros(self.n, self.c, out_h, out_w);

        let input_channel_stride = self.h * self.w;
        let input_batch_stride = self.c * input_channel_stride;
        let output_channel_stride = out_h * out_w;
        let output_batch_stride = self.c * output_channel_stride;

        for batch in 0..self.n {
            let input_batch_base = batch * input_batch_stride;
            let output_batch_base = batch * output_batch_stride;

            for channel in 0..self.c {
                let input_channel_base = input_batch_base + channel * input_channel_stride;
                let output_channel_base = output_batch_base + channel * output_channel_stride;

                for out_y in 0..out_h {
                    let in_y = out_y * stride_h;
                    let output_row_base = output_channel_base + out_y * out_w;

                    for out_x in 0..out_w {
                        let in_x = out_x * stride_w;
                        let mut max_value = f32::NEG_INFINITY;

                        for wy in 0..window_h {
                            let input_row_base = input_channel_base + (in_y + wy) * self.w + in_x;
                            let input_row = &self.data[input_row_base..input_row_base + window_w];

                            for value in input_row.iter().copied() {
                                if value > max_value {
                                    max_value = value;
                                }
                            }
                        }

                        output.data[output_row_base + out_x] = max_value;
                    }
                }
            }
        }

        Ok(output)
    }

    pub fn max_pool2d_with_backend<B: TensorBackend + ?Sized>(
        &self,
        backend: &B,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Self, TensorError> {
        backend.max_pool2d(self, window_h, window_w, stride_h, stride_w)
    }

    pub fn global_average_pool2d(&self) -> Result<Self, TensorError> {
        self.global_average_pool2d_with_backend(active_backend())
    }

    pub(crate) fn global_average_pool2d_cpu(&self) -> Result<Self, TensorError> {
        if self.h == 0 || self.w == 0 {
            return Err(TensorError::InvalidArgument(
                "global average pooling requires non-zero spatial dimensions",
            ));
        }

        let mut output = Tensor4D::zeros(self.n, self.c, 1, 1);
        let spatial_area = self.h * self.w;
        let batch_stride = self.c * spatial_area;

        for batch in 0..self.n {
            let batch_base = batch * batch_stride;
            for channel in 0..self.c {
                let channel_base = batch_base + channel * spatial_area;
                let channel_slice = &self.data[channel_base..channel_base + spatial_area];
                let sum: f32 = channel_slice.iter().copied().sum();
                let denom = spatial_area as f32;
                output.data[batch * self.c + channel] = sum / denom;
            }
        }

        Ok(output)
    }

    pub fn global_average_pool2d_with_backend<B: TensorBackend + ?Sized>(
        &self,
        backend: &B,
    ) -> Result<Self, TensorError> {
        backend.global_average_pool2d(self)
    }

    pub fn relu_inplace_with_backend<B: TensorBackend + ?Sized>(&mut self, backend: &B) {
        backend.relu_inplace(self)
    }

    fn offset(&self, n: usize, c: usize, h: usize, w: usize) -> Result<usize, TensorError> {
        if n >= self.n || c >= self.c || h >= self.h || w >= self.w {
            return Err(TensorError::OutOfBounds { n, c, h, w });
        }

        Ok((((n * self.c) + c) * self.h + h) * self.w + w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::backend::cpu_backend;

    #[test]
    fn tensor4d_round_trip_get_set() {
        let mut tensor = Tensor4D::zeros(2, 1, 2, 2);

        assert!(tensor.set(1, 0, 1, 1, 3.5).is_ok());
        assert_eq!(tensor.get(1, 0, 1, 1), Ok(3.5));
    }

    #[test]
    fn tensor4d_from_vec_validates_shape() {
        let result = Tensor4D::from_vec(1, 1, 2, 2, vec![0.0, 1.0, 2.0]);
        assert!(matches!(
            result,
            Err(TensorError::ShapeMismatch {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn tensor4d_add_inplace_requires_same_shape() {
        let mut left = Tensor4D::zeros(1, 1, 2, 2);
        let right = Tensor4D::zeros(1, 2, 2, 2);

        let result = left.add_inplace(&right);
        assert!(matches!(result, Err(TensorError::IncompatibleShapes { .. })));
    }

    #[test]
    fn tensor4d_map_and_add_work() {
        let mut left = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap_or_else(|_| panic!("left tensor should be valid"));
        let right = Tensor4D::from_vec(1, 1, 2, 2, vec![0.5, 0.5, 0.5, 0.5])
            .unwrap_or_else(|_| panic!("right tensor should be valid"));

        left.map_inplace(|x| x * 2.0);
        assert!(left.add_inplace(&right).is_ok());

        assert_eq!(left.as_slice(), &[2.5, 4.5, 6.5, 8.5]);
    }

    #[test]
    fn tensor4d_conv2d_valid_applies_kernel_and_bias() {
        let input = Tensor4D::from_vec(
            1,
            1,
            3,
            3,
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                7.0, 8.0, 9.0, // row 2
            ],
        )
        .unwrap_or_else(|_| panic!("input tensor should be valid"));

        let kernels = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 0.0, 0.0, 1.0])
            .unwrap_or_else(|_| panic!("kernel tensor should be valid"));

        let output = input
            .conv2d_valid(&kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("convolution should succeed"));

        assert_eq!(output.shape(), (1, 1, 2, 2));
        assert_eq!(output.as_slice(), &[7.0, 9.0, 13.0, 15.0]);
    }

    #[test]
    fn tensor4d_conv2d_valid_rejects_zero_stride() {
        let input = Tensor4D::zeros(1, 1, 3, 3);
        let kernels = Tensor4D::zeros(1, 1, 2, 2);

        let result = input.conv2d_valid(&kernels, None, 0, 1);
        assert!(matches!(
            result,
            Err(TensorError::InvalidArgument("stride must be greater than zero"))
        ));
    }

    #[test]
    fn tensor4d_max_pool2d_computes_window_maxima() {
        let input = Tensor4D::from_vec(
            1,
            1,
            4,
            4,
            vec![
                1.0, 3.0, 2.0, 0.0, // row 0
                5.0, 6.0, 1.0, 4.0, // row 1
                2.0, 8.0, 7.0, 3.0, // row 2
                9.0, 1.0, 5.0, 2.0, // row 3
            ],
        )
        .unwrap_or_else(|_| panic!("input tensor should be valid"));

        let output = input
            .max_pool2d(2, 2, 2, 2)
            .unwrap_or_else(|_| panic!("max pooling should succeed"));

        assert_eq!(output.shape(), (1, 1, 2, 2));
        assert_eq!(output.as_slice(), &[6.0, 4.0, 9.0, 7.0]);
    }

    #[test]
    fn tensor4d_max_pool2d_rejects_invalid_window() {
        let input = Tensor4D::zeros(1, 1, 2, 2);

        let result = input.max_pool2d(0, 2, 1, 1);
        assert!(matches!(
            result,
            Err(TensorError::InvalidArgument(
                "pooling window must be greater than zero"
            ))
        ));
    }

    #[test]
    fn tensor4d_relu_inplace_clamps_negative_values() {
        let mut tensor = Tensor4D::from_vec(1, 1, 2, 2, vec![-2.0, -0.1, 0.0, 3.5])
            .unwrap_or_else(|_| panic!("tensor should be valid"));

        tensor.relu_inplace();

        assert_eq!(tensor.as_slice(), &[0.0, 0.0, 0.0, 3.5]);
    }

    #[test]
    fn tensor4d_global_average_pool2d_reduces_spatial_dimensions() {
        let input = Tensor4D::from_vec(
            1,
            2,
            2,
            2,
            vec![
                1.0, 3.0, 5.0, 7.0, // channel 0
                2.0, 4.0, 6.0, 8.0, // channel 1
            ],
        )
        .unwrap_or_else(|_| panic!("input tensor should be valid"));

        let pooled = input
            .global_average_pool2d()
            .unwrap_or_else(|_| panic!("global average pooling should succeed"));

        assert_eq!(pooled.shape(), (1, 2, 1, 1));
        assert_eq!(pooled.get(0, 0, 0, 0), Ok(4.0));
        assert_eq!(pooled.get(0, 1, 0, 0), Ok(5.0));
    }

    #[test]
    fn tensor4d_flatten_batch_features_returns_per_sample_vectors() {
        let tensor = Tensor4D::from_vec(
            2,
            1,
            2,
            2,
            vec![
                1.0, 2.0, 3.0, 4.0, // sample 0
                5.0, 6.0, 7.0, 8.0, // sample 1
            ],
        )
        .unwrap_or_else(|_| panic!("tensor should be valid"));

        let flat = tensor.flatten_batch_features();

        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(flat[1], vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn tensor4d_backend_wrappers_match_cpu_operations() {
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

        let conv = input
            .conv2d_valid_with_backend(&backend, &kernels, Some(&[1.0]), 1, 1)
            .unwrap_or_else(|_| panic!("backend convolution should succeed"));
        let pooled = conv
            .max_pool2d_with_backend(&backend, 2, 2, 1, 1)
            .unwrap_or_else(|_| panic!("backend pooling should succeed"));
        let gap = pooled
            .global_average_pool2d_with_backend(&backend)
            .unwrap_or_else(|_| panic!("backend global average pooling should succeed"));

        assert_eq!(conv.shape(), (1, 1, 2, 2));
        assert_eq!(pooled.shape(), (1, 1, 1, 1));
        assert_eq!(gap.shape(), (1, 1, 1, 1));

        let mut relu_tensor = Tensor4D::from_vec(1, 1, 1, 2, vec![-1.0, 3.0])
            .unwrap_or_else(|_| panic!("relu tensor should be valid"));
        relu_tensor.relu_inplace_with_backend(&backend);
        assert_eq!(relu_tensor.as_slice(), &[0.0, 3.0]);
    }
}
