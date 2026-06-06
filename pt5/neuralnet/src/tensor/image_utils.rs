use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageTensorUtilityError {
    InvalidDimensions {
        height: usize,
        width: usize,
    },
    InvalidImageSize {
        expected: usize,
        actual: usize,
    },
    CropOutOfBounds {
        in_height: usize,
        in_width: usize,
        crop_height: usize,
        crop_width: usize,
    },
}

impl Display for ImageTensorUtilityError {
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
                    "invalid image size for utility operation: expected {}, got {}",
                    expected, actual
                )
            }
            Self::CropOutOfBounds {
                in_height,
                in_width,
                crop_height,
                crop_width,
            } => {
                write!(
                    f,
                    "crop {}x{} exceeds source {}x{}",
                    crop_height, crop_width, in_height, in_width
                )
            }
        }
    }
}

impl Error for ImageTensorUtilityError {}

pub(crate) fn validate_image_dimensions(
    height: usize,
    width: usize,
) -> Result<(), ImageTensorUtilityError> {
    if height == 0 || width == 0 {
        return Err(ImageTensorUtilityError::InvalidDimensions { height, width });
    }

    Ok(())
}

pub fn resize_grayscale_nearest(
    image: &[u8],
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
) -> Result<Vec<u8>, ImageTensorUtilityError> {
    validate_image_dimensions(in_height, in_width)?;
    validate_image_dimensions(out_height, out_width)?;

    let expected = in_height.saturating_mul(in_width);
    if image.len() != expected {
        return Err(ImageTensorUtilityError::InvalidImageSize {
            expected,
            actual: image.len(),
        });
    }

    let mut resized = vec![0u8; out_height.saturating_mul(out_width)];

    for out_y in 0..out_height {
        let in_y = out_y.saturating_mul(in_height) / out_height;
        for out_x in 0..out_width {
            let in_x = out_x.saturating_mul(in_width) / out_width;
            let src_idx = in_y.saturating_mul(in_width) + in_x;
            let dst_idx = out_y.saturating_mul(out_width) + out_x;
            resized[dst_idx] = image[src_idx];
        }
    }

    Ok(resized)
}

pub fn center_crop_grayscale(
    image: &[u8],
    in_height: usize,
    in_width: usize,
    crop_height: usize,
    crop_width: usize,
) -> Result<Vec<u8>, ImageTensorUtilityError> {
    validate_image_dimensions(in_height, in_width)?;
    validate_image_dimensions(crop_height, crop_width)?;

    let expected = in_height.saturating_mul(in_width);
    if image.len() != expected {
        return Err(ImageTensorUtilityError::InvalidImageSize {
            expected,
            actual: image.len(),
        });
    }

    if crop_height > in_height || crop_width > in_width {
        return Err(ImageTensorUtilityError::CropOutOfBounds {
            in_height,
            in_width,
            crop_height,
            crop_width,
        });
    }

    let start_y = (in_height - crop_height) / 2;
    let start_x = (in_width - crop_width) / 2;
    let mut cropped = vec![0u8; crop_height.saturating_mul(crop_width)];

    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_y = start_y + y;
            let src_x = start_x + x;
            let src_idx = src_y.saturating_mul(in_width) + src_x;
            let dst_idx = y.saturating_mul(crop_width) + x;
            cropped[dst_idx] = image[src_idx];
        }
    }

    Ok(cropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_grayscale_nearest_upscales_by_replication() {
        let image = vec![1u8, 2u8, 3u8, 4u8];
        let resized = resize_grayscale_nearest(image.as_slice(), 2, 2, 4, 4)
            .unwrap_or_else(|_| panic!("resize should succeed"));

        assert_eq!(resized.len(), 16);
        assert_eq!(resized[0], 1);
        assert_eq!(resized[1], 1);
        assert_eq!(resized[2], 2);
        assert_eq!(resized[3], 2);
        assert_eq!(resized[8], 3);
        assert_eq!(resized[15], 4);
    }

    #[test]
    fn center_crop_grayscale_extracts_middle_region() {
        let image = vec![
            0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8,
            15u8,
        ];

        let cropped = center_crop_grayscale(image.as_slice(), 4, 4, 2, 2)
            .unwrap_or_else(|_| panic!("crop should succeed"));

        assert_eq!(cropped, vec![5u8, 6u8, 9u8, 10u8]);
    }
}
