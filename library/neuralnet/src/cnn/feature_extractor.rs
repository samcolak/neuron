use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::cnn::data_pipeline::ImageTensorShape;
use crate::tensor::adapters::{
    image_bytes_to_tensor_nchw_resized_with_channels,
    TensorAdapterError,
};
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub struct CnnFeatureExtractor {
    input_height: usize,
    input_width: usize,
    kernels: Tensor4D,
    bias: Vec<f32>,
}

#[derive(Debug)]
pub enum CnnFeatureExtractorError {
    UnsupportedImageShape {
        byte_len: usize,
    },
    TensorAdapter(TensorAdapterError),
    Tensor(TensorError),
}

impl Display for CnnFeatureExtractorError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        match self {

            Self::UnsupportedImageShape { byte_len } => {
                write!(
                    f,
                    "unsupported image shape for CNN extractor (expected square bytes with 1, 3, or 4 channels): {} bytes",
                    byte_len
                )
            },

            Self::TensorAdapter(err) => write!(f, "cnn extractor adapter error: {err}"),
            Self::Tensor(err) => write!(f, "cnn extractor tensor error: {err}"),
        }

    }

}

impl Error for CnnFeatureExtractorError {}

impl From<TensorAdapterError> for CnnFeatureExtractorError {
    fn from(value: TensorAdapterError) -> Self {
        Self::TensorAdapter(value)
    }
}

impl From<TensorError> for CnnFeatureExtractorError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

impl Default for CnnFeatureExtractor {
    fn default() -> Self {
        Self::new(16, 16)
    }
}

impl CnnFeatureExtractor {

    pub fn new(input_height: usize, input_width: usize) -> Self {

        let kernels = Tensor4D::from_vec(
            2,
            1,
            3,
            3,
            vec![
                -1.0, 0.0, 1.0,
                -1.0, 0.0, 1.0,
                -1.0, 0.0, 1.0,
                -1.0, -1.0, -1.0,
                0.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
            ],
        )
        .unwrap_or_else(|_| panic!("fixed CNN kernels should be valid"));

        Self {
            input_height,
            input_width,
            kernels,
            bias: vec![0.0, 0.0],
        }

    }

    pub fn extract_feature_tokens(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<String>, CnnFeatureExtractorError> {

        let (in_height, in_width, in_channels) = infer_square_dimensions_and_channels(image_bytes).ok_or(
            CnnFeatureExtractorError::UnsupportedImageShape {
                byte_len: image_bytes.len(),
            },
        )?;

        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            in_height,
            in_width,
            in_channels,
            self.input_height,
            self.input_width,
            true,
        )?;

        let input = if in_channels == 1 {
            input
        } else {
            collapse_to_single_channel(&input)?
        };

        let pooled = input.conv_relu_max_pool2d_valid(
            &self.kernels,
            Some(self.bias.as_slice()),
            1,
            1,
            2,
            2,
            2,
            2,
        )?;
        let mut tokens = quantize_channels(&pooled)?;

        let global_pooled = pooled.global_average_pool2d()?;
        let mean_activation = if global_pooled.is_empty() {
            0.0
        } else {
            global_pooled.as_slice().iter().sum::<f32>() / global_pooled.len() as f32
        };
        let mean_bucket = (mean_activation.clamp(0.0, 3.0) * 10.0) as usize;
        tokens.push(format!("g{}", mean_bucket.min(30)));

        Ok(tokens)

    }

    pub fn extract_feature_tokens_with_dimensions(
        &self,
        image_bytes: &[u8],
        shape: ImageTensorShape,
    ) -> Result<Vec<String>, CnnFeatureExtractorError> {

        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            shape.height,
            shape.width,
            shape.channels,
            self.input_height,
            self.input_width,
            true,
        )?;

        let input = if shape.channels == 1 {
            input
        } else {
            collapse_to_single_channel(&input)?
        };

        let pooled = input.conv_relu_max_pool2d_valid(
            &self.kernels,
            Some(self.bias.as_slice()),
            1,
            1,
            2,
            2,
            2,
            2,
        )?;
        let mut tokens = quantize_channels(&pooled)?;

        let global_pooled = pooled.global_average_pool2d()?;
        let mean_activation = if global_pooled.is_empty() {
            0.0
        } else {
            global_pooled.as_slice().iter().sum::<f32>() / global_pooled.len() as f32
        };
        let mean_bucket = (mean_activation.clamp(0.0, 3.0) * 10.0) as usize;
        tokens.push(format!("g{}", mean_bucket.min(30)));

        Ok(tokens)

    }

}

fn infer_square_dimensions_and_channels(image_bytes: &[u8]) -> Option<(usize, usize, usize)> {

    if image_bytes.is_empty() {
        return None;
    }

    let len = image_bytes.len();

    for channels in [1usize, 3usize, 4usize] {
        if !len.is_multiple_of(channels) {
            continue;
        }

        let pixels = len / channels;
        let side = (pixels as f64).sqrt() as usize;

        if side.saturating_mul(side) == pixels {
            return Some((side, side, channels));
        }
    }

    None

}

fn collapse_to_single_channel(input: &Tensor4D) -> Result<Tensor4D, TensorError> {

    let (n, c, h, w) = input.shape();
    let mut collapsed = Tensor4D::zeros(n, 1, h, w);

    for batch in 0..n {
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                for channel in 0..c {
                    sum += input.get(batch, channel, y, x)?;
                }
                collapsed.set(batch, 0, y, x, sum / c as f32)?;
            }
        }
    }

    Ok(collapsed)

}

fn quantize_channels(pooled: &Tensor4D) -> Result<Vec<String>, TensorError> {

    let (_, channels, height, width) = pooled.shape();
    let mut tokens = Vec::with_capacity(channels.saturating_mul(4));

    for channel in 0..channels {
        let mut sum = 0.0f32;
        let mut peak = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let value = pooled.get(0, channel, y, x)?;
                sum += value;
                if value > peak {
                    peak = value;
                }
            }
        }

        let count = (height.saturating_mul(width)).max(1) as f32;
        let mean = sum / count;
        let mean_bucket = (mean.clamp(0.0, 3.0) * 10.0) as usize;
        let peak_bucket = (peak.clamp(0.0, 3.0) * 10.0) as usize;

        tokens.push(format!("ch{}m{}", channel, mean_bucket.min(30)));
        tokens.push(format!("ch{}p{}", channel, peak_bucket.min(30)));
    }

    Ok(tokens)

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_produces_tokens_for_square_image() {
        let extractor = CnnFeatureExtractor::new(16, 16);
        let image = vec![32u8; 64];

        let tokens = extractor
            .extract_feature_tokens(image.as_slice())
            .unwrap_or_else(|_| panic!("extractor should produce tokens"));

        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|token| !token.contains(':')));
    }

    #[test]
    fn extractor_rejects_non_square_image() {
        let extractor = CnnFeatureExtractor::default();
        let image = vec![10u8; 1000];

        let result = extractor.extract_feature_tokens(image.as_slice());

        assert!(matches!(
            result,
            Err(CnnFeatureExtractorError::UnsupportedImageShape { byte_len: 1000 })
        ));
    }

    #[test]
    fn extractor_accepts_rgb_square_image_bytes() {
        let extractor = CnnFeatureExtractor::new(16, 16);
        let grayscale = [32u8; 64];
        let rgb: Vec<u8> = grayscale
            .iter()
            .flat_map(|value| [*value, *value, *value])
            .collect();

        let tokens = extractor
            .extract_feature_tokens(rgb.as_slice())
            .unwrap_or_else(|_| panic!("extractor should produce tokens for rgb square image"));

        assert!(!tokens.is_empty());
    }

    #[test]
    fn extractor_tokens_remain_stable_on_fused_path() {
        let extractor = CnnFeatureExtractor::new(16, 16);
        let image = vec![64u8; 64];

        let tokens = extractor
            .extract_feature_tokens(image.as_slice())
            .unwrap_or_else(|_| panic!("extractor should produce stable fused tokens"));

        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|token| token.starts_with("g")));
    }

    #[test]
    fn extractor_supports_rectangular_images_with_explicit_dimensions() {
        
        let extractor = CnnFeatureExtractor::new(16, 16);
        let image = vec![32u8; 48];
        let shape = crate::cnn::data_pipeline::ImageTensorShape::new(4, 12, 1)
            .unwrap_or_else(|| panic!("shape should be valid"));

        let tokens = extractor
            .extract_feature_tokens_with_dimensions(image.as_slice(), shape)
            .unwrap_or_else(|_| panic!("extractor should support rectangular inputs"));

        assert!(!tokens.is_empty());

    }
    
}
