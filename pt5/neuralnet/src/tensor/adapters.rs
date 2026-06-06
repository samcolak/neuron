use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::helpers::multimodal_controller::MultiModalInput;
use crate::tensor::image_utils::{
    validate_image_dimensions,
    ImageTensorUtilityError,
};
use crate::tensor::tensor4d::Tensor4D;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorAdapterError {
    InvalidDimensions {
        height: usize,
        width: usize,
    },
    InvalidImageSize {
        expected: usize,
        actual: usize,
    },
    EmptyBatch,
    InconsistentImageSize {
        index: usize,
        expected: usize,
        actual: usize,
    },
    UnsupportedInputType,
}

impl Display for TensorAdapterError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        match self {
            Self::InvalidDimensions { height, width } => {
                write!(
                    f,
                    "invalid image dimensions: height={}, width={} (must both be > 0)",
                    height, width
                )
            }
            Self::InvalidImageSize { expected, actual } => {
                write!(
                    f,
                    "invalid image size for tensor conversion: expected {}, got {}",
                    expected, actual
                )
            }
            Self::EmptyBatch => write!(f, "cannot build tensor from empty image batch"),
            Self::InconsistentImageSize {
                index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "inconsistent image size at batch index {}: expected {}, got {}",
                    index, expected, actual
                )
            }
            Self::UnsupportedInputType => {
                write!(f, "only MultiModalInput::ImageBytes can be adapted to NCHW tensor")
            }
        }

    }

}

impl Error for TensorAdapterError {}

impl From<ImageTensorUtilityError> for TensorAdapterError {
    fn from(value: ImageTensorUtilityError) -> Self {
        match value {
            ImageTensorUtilityError::InvalidDimensions { height, width } => {
                Self::InvalidDimensions { height, width }
            }
            ImageTensorUtilityError::InvalidImageSize { expected, actual } => {
                Self::InvalidImageSize { expected, actual }
            }
            ImageTensorUtilityError::CropOutOfBounds {
                in_height,
                in_width,
                crop_height,
                crop_width,
            } => Self::InvalidImageSize {
                expected: in_height.saturating_mul(in_width),
                actual: crop_height.saturating_mul(crop_width),
            },
        }
    }
}

fn normalize(value: u8, normalize_pixels: bool) -> f32 {
    if normalize_pixels {
        value as f32 / 255.0
    } else {
        value as f32
    }
}

pub fn image_bytes_to_tensor_nchw(
    image: &[u8],
    height: usize,
    width: usize,
    normalize_pixels: bool,
) -> Result<Tensor4D, TensorAdapterError> {

    validate_image_dimensions(height, width)?;

    let expected = height.saturating_mul(width);

    if image.len() != expected {
        return Err(TensorAdapterError::InvalidImageSize {
            expected,
            actual: image.len(),
        });
    }

    let data: Vec<f32> = image
        .iter()
        .map(|value| normalize(*value, normalize_pixels))
        .collect();

    // Shape has already been validated against expected length.
    Tensor4D::from_vec(1, 1, height, width, data).map_err(|_| {
        TensorAdapterError::InvalidImageSize {
            expected,
            actual: image.len(),
        }
    })
}

pub fn image_batch_to_tensor_nchw(
    images: &[Vec<u8>],
    height: usize,
    width: usize,
    normalize_pixels: bool,
) -> Result<Tensor4D, TensorAdapterError> {

    validate_image_dimensions(height, width)?;

    if images.is_empty() {
        return Err(TensorAdapterError::EmptyBatch);
    }

    let expected = height.saturating_mul(width);
    let mut data: Vec<f32> = Vec::with_capacity(images.len().saturating_mul(expected));

    for (index, image) in images.iter().enumerate() {
        if image.len() != expected {
            return Err(TensorAdapterError::InconsistentImageSize {
                index,
                expected,
                actual: image.len(),
            });
        }

        data.extend(
            image
                .iter()
                .map(|value| normalize(*value, normalize_pixels)),
        );
    }

    // Batch images are guaranteed to be same validated size.
    Tensor4D::from_vec(images.len(), 1, height, width, data).map_err(|_| {
        TensorAdapterError::InvalidImageSize {
            expected: images.len().saturating_mul(expected),
            actual: images.len().saturating_mul(expected),
        }
    })

}

pub fn multimodal_input_to_tensor_nchw(
    input: &MultiModalInput,
    height: usize,
    width: usize,
    normalize_pixels: bool,
) -> Result<Tensor4D, TensorAdapterError> {

    match input {
        MultiModalInput::ImageBytes(image) => {
            image_bytes_to_tensor_nchw(image.as_slice(), height, width, normalize_pixels)
        }
        _ => Err(TensorAdapterError::UnsupportedInputType),
    }
    
}

pub fn image_bytes_to_tensor_nchw_resized(
    image: &[u8],
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
    normalize_pixels: bool,
) -> Result<Tensor4D, TensorAdapterError> {

    let resized = crate::tensor::image_utils::resize_grayscale_nearest(
        image,
        in_height,
        in_width,
        out_height,
        out_width,
    )
    .map_err(TensorAdapterError::from)?;

    image_bytes_to_tensor_nchw(
        resized.as_slice(),
        out_height,
        out_width,
        normalize_pixels,
    )
    
}

pub fn multimodal_input_to_tensor_nchw_resized(
    input: &MultiModalInput,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
    normalize_pixels: bool,
) -> Result<Tensor4D, TensorAdapterError> {

    match input {
        MultiModalInput::ImageBytes(image) => image_bytes_to_tensor_nchw_resized(
            image.as_slice(),
            in_height,
            in_width,
            out_height,
            out_width,
            normalize_pixels,
        ),
        _ => Err(TensorAdapterError::UnsupportedInputType),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_adapter_builds_nchw_tensor_with_normalization() {
        let image = vec![0u8, 64u8, 128u8, 255u8];
        let tensor = image_bytes_to_tensor_nchw(image.as_slice(), 2, 2, true)
            .unwrap_or_else(|_| panic!("image tensor conversion should succeed"));

        assert_eq!(tensor.shape(), (1, 1, 2, 2));
        assert_eq!(tensor.get(0, 0, 0, 0), Ok(0.0));

        let mid = tensor
            .get(0, 0, 0, 1)
            .unwrap_or_else(|_| panic!("tensor index should be valid"));
        assert!((mid - (64.0 / 255.0)).abs() < f32::EPSILON);

        assert_eq!(tensor.get(0, 0, 1, 1), Ok(1.0));
    }

    #[test]
    fn image_adapter_rejects_wrong_pixel_count() {
        let image = vec![1u8, 2u8, 3u8];
        let result = image_bytes_to_tensor_nchw(image.as_slice(), 2, 2, false);

        assert!(matches!(
            result,
            Err(TensorAdapterError::InvalidImageSize {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn batch_adapter_stacks_images_in_batch_dimension() {
        let images = vec![vec![1u8, 2u8, 3u8, 4u8], vec![5u8, 6u8, 7u8, 8u8]];
        let tensor = image_batch_to_tensor_nchw(images.as_slice(), 2, 2, false)
            .unwrap_or_else(|_| panic!("batch conversion should succeed"));

        assert_eq!(tensor.shape(), (2, 1, 2, 2));
        assert_eq!(tensor.get(0, 0, 0, 0), Ok(1.0));
        assert_eq!(tensor.get(1, 0, 1, 1), Ok(8.0));
    }

    #[test]
    fn multimodal_adapter_rejects_non_image_input() {
        let input = MultiModalInput::Text("cat".to_string());
        let result = multimodal_input_to_tensor_nchw(&input, 1, 1, false);

        assert_eq!(result, Err(TensorAdapterError::UnsupportedInputType));
    }

    #[test]
    fn resized_adapter_builds_target_shape_tensor() {
        let image = vec![0u8, 64u8, 128u8, 255u8];
        let tensor = image_bytes_to_tensor_nchw_resized(
            image.as_slice(),
            2,
            2,
            4,
            4,
            true,
        )
        .unwrap_or_else(|_| panic!("resized tensor conversion should succeed"));

        assert_eq!(tensor.shape(), (1, 1, 4, 4));
        assert_eq!(tensor.get(0, 0, 0, 0), Ok(0.0));
        assert_eq!(tensor.get(0, 0, 3, 3), Ok(1.0));
    }
}
