use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::tensor::adapters::{
    image_bytes_to_tensor_nchw_resized_with_channels,
    TensorAdapterError,
};
use crate::tensor::tensor4d::{Tensor4D, TensorError};
use crate::training::linear_head::{LinearHead, LinearHeadError, LinearOptimizer};

const CNN_CLASSIFIER_BIN_MAGIC: [u8; 4] = *b"CNN1";

#[derive(Debug)]
pub enum CnnImageClassifierError {
    InvalidConfiguration(&'static str),
    UnsupportedImageShape {
        byte_len: usize,
    },
    InputChannelMismatch {
        expected: usize,
        actual: usize,
    },
    UnknownLabel(String),
    GradientShapeMismatch {
        expected: usize,
        actual: usize,
    },
    GradientTensorShapeMismatch {
        expected: (usize, usize, usize, usize),
        actual: (usize, usize, usize, usize),
    },
    TensorAdapter(TensorAdapterError),
    Tensor(TensorError),
    LinearHead(LinearHeadError),
}

impl Display for CnnImageClassifierError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid cnn classifier configuration: {}", message)
            }
            Self::UnsupportedImageShape { byte_len } => write!(
                f,
                "unsupported image shape for CNN classifier (expected square bytes with 1, 3, or 4 channels): {} bytes",
                byte_len
            ),
            Self::InputChannelMismatch { expected, actual } => write!(
                f,
                "cnn classifier input channel mismatch: expected {}, got {}",
                expected, actual
            ),
            Self::UnknownLabel(label) => write!(f, "unknown CNN classifier label: {}", label),
            Self::GradientShapeMismatch { expected, actual } => write!(
                f,
                "cnn classifier gradient shape mismatch: expected {}, got {}",
                expected, actual
            ),
            Self::GradientTensorShapeMismatch { expected, actual } => write!(
                f,
                "cnn classifier tensor gradient shape mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::TensorAdapter(err) => write!(f, "cnn classifier adapter error: {err}"),
            Self::Tensor(err) => write!(f, "cnn classifier tensor error: {err}"),
            Self::LinearHead(err) => write!(f, "cnn classifier linear head error: {err}"),
        }
    }
}

impl Error for CnnImageClassifierError {}

impl From<TensorAdapterError> for CnnImageClassifierError {
    fn from(value: TensorAdapterError) -> Self {
        Self::TensorAdapter(value)
    }
}

impl From<TensorError> for CnnImageClassifierError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

impl From<LinearHeadError> for CnnImageClassifierError {
    fn from(value: LinearHeadError) -> Self {
        Self::LinearHead(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnImageClassifier {
    input_height: usize,
    input_width: usize,
    input_channels: usize,
    class_labels: Vec<String>,
    label_to_index: BTreeMap<String, usize>,
    conv1_kernels: Tensor4D,
    conv1_bias: Vec<f32>,
    conv2_kernels: Option<Tensor4D>,
    conv2_bias: Option<Vec<f32>>,
    feature_learning_rate: f32,
    head: LinearHead,
    min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CnnImageClassifierSnapshot {
    input_height: usize,
    input_width: usize,
    #[serde(default = "default_input_channels")]
    input_channels: usize,
    class_labels: Vec<String>,
    label_to_index: BTreeMap<String, usize>,
    conv1_kernels: Tensor4D,
    conv1_bias: Vec<f32>,
    conv2_kernels: Option<Tensor4D>,
    conv2_bias: Option<Vec<f32>>,
    feature_learning_rate: f32,
    head: LinearHead,
    min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyCnnImageClassifierSnapshot {
    input_height: usize,
    input_width: usize,
    class_labels: Vec<String>,
    label_to_index: BTreeMap<String, usize>,
    feature_kernels: Tensor4D,
    feature_bias: Vec<f32>,
    feature_learning_rate: f32,
    head: LinearHead,
    min_confidence: f32,
}

impl From<LegacyCnnImageClassifierSnapshot> for CnnImageClassifierSnapshot {
    fn from(value: LegacyCnnImageClassifierSnapshot) -> Self {
        Self {
            input_height: value.input_height,
            input_width: value.input_width,
            input_channels: 1,
            class_labels: value.class_labels,
            label_to_index: value.label_to_index,
            conv1_kernels: value.feature_kernels,
            conv1_bias: value.feature_bias,
            conv2_kernels: None,
            conv2_bias: None,
            feature_learning_rate: value.feature_learning_rate,
            head: value.head,
            min_confidence: value.min_confidence,
        }
    }
}

#[derive(Debug, Clone)]
struct ConvBlockCache {
    input: Tensor4D,
    conv_pre_activation: Tensor4D,
    relu_output: Tensor4D,
    pool_indices: Vec<(usize, usize)>,
    pooled_shape: (usize, usize, usize, usize),
}

#[derive(Debug, Clone)]
struct ForwardCache {
    block1: ConvBlockCache,
    block2: Option<ConvBlockCache>,
}

#[derive(Debug, Clone)]
struct ConvBlockBackward {
    kernel_grad: Tensor4D,
    bias_grad: Vec<f32>,
    input_grad: Option<Tensor4D>,
}

impl CnnImageClassifier {
    pub fn new(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        learning_rate: f32,
    ) -> Result<Self, CnnImageClassifierError> {
        Self::new_with_feature_channels(
            class_labels,
            input_height,
            input_width,
            &[2],
            learning_rate,
        )
    }

    pub fn new_two_layer(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        conv1_channels: usize,
        conv2_channels: usize,
        learning_rate: f32,
    ) -> Result<Self, CnnImageClassifierError> {
        Self::new_with_feature_channels(
            class_labels,
            input_height,
            input_width,
            &[conv1_channels, conv2_channels],
            learning_rate,
        )
    }

    pub fn new_with_feature_channels(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        feature_channels: &[usize],
        learning_rate: f32,
    ) -> Result<Self, CnnImageClassifierError> {
        Self::new_with_feature_channels_and_input_channels(
            class_labels,
            input_height,
            input_width,
            1,
            feature_channels,
            learning_rate,
        )
    }

    pub fn new_with_feature_channels_and_input_channels(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        input_channels: usize,
        feature_channels: &[usize],
        learning_rate: f32,
    ) -> Result<Self, CnnImageClassifierError> {
        if input_height == 0 || input_width == 0 {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "input height and width must be greater than zero",
            ));
        }

        if input_channels == 0 {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "input channels must be greater than zero",
            ));
        }

        if feature_channels.is_empty() || feature_channels.len() > 2 {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "feature channel configuration must contain one or two layers",
            ));
        }
        if feature_channels.contains(&0) {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "feature channel counts must be greater than zero",
            ));
        }

        let mut normalized_labels = Vec::new();
        let mut label_to_index = BTreeMap::new();

        for raw in class_labels {
            let normalized = raw.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            if label_to_index.contains_key(&normalized) {
                continue;
            }

            let idx = normalized_labels.len();
            label_to_index.insert(normalized.clone(), idx);
            normalized_labels.push(normalized);
        }

        if normalized_labels.len() < 2 {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "at least two distinct class labels are required",
            ));
        }

        let conv1_out = feature_channels[0];
        let conv1_kernels = initialize_conv_kernels(conv1_out, input_channels, true)?;
        let conv1_bias = vec![0.0; conv1_out];

        let (conv2_kernels, conv2_bias, feature_dim) = if feature_channels.len() == 2 {
            let conv2_out = feature_channels[1];
            (
                Some(initialize_conv_kernels(conv2_out, conv1_out, false)?),
                Some(vec![0.0; conv2_out]),
                conv2_out,
            )
        } else {
            (None, None, conv1_out)
        };

        let head = LinearHead::new(feature_dim, normalized_labels.len(), learning_rate)?;

        Ok(Self {
            input_height,
            input_width,
            input_channels,
            class_labels: normalized_labels,
            label_to_index,
            conv1_kernels,
            conv1_bias,
            conv2_kernels,
            conv2_bias,
            feature_learning_rate: learning_rate.max(0.0),
            head,
            min_confidence: 0.5,
        })
    }

    pub fn class_labels(&self) -> &[String] {
        self.class_labels.as_slice()
    }

    pub fn set_min_confidence(&mut self, min_confidence: f32) {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
    }

    pub fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    pub fn set_head_optimizer(&mut self, optimizer: LinearOptimizer) {
        self.head.set_optimizer(optimizer);
    }

    pub fn head_optimizer(&self) -> LinearOptimizer {
        self.head.optimizer()
    }

    pub fn head_learning_rate(&self) -> f32 {
        self.head.learning_rate()
    }

    pub fn set_head_weight_decay(&mut self, weight_decay: f32) {
        self.head.set_weight_decay(weight_decay);
    }

    #[cfg(feature = "optimizer-adam")]
    pub fn configure_head_adam(&mut self, beta1: f32, beta2: f32, epsilon: f32) {
        self.head.configure_adam(beta1, beta2, epsilon);
    }

    pub fn extract_features(&self, image_bytes: &[u8]) -> Result<Vec<f32>, CnnImageClassifierError> {
        self.forward_for_inference(image_bytes)
    }

    fn forward_for_inference(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<f32>, CnnImageClassifierError> {
        let (in_height, in_width, in_channels) = infer_square_dimensions_and_channels(image_bytes).ok_or(
            CnnImageClassifierError::UnsupportedImageShape {
                byte_len: image_bytes.len(),
            },
        )?;

        if in_channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: in_channels,
            });
        }

        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            in_height,
            in_width,
            in_channels,
            self.input_height,
            self.input_width,
            true,
        )?;

        let block1_output = input.conv_relu_max_pool2d_valid(
            &self.conv1_kernels,
            Some(self.conv1_bias.as_slice()),
            1,
            1,
            2,
            2,
            2,
            2,
        )?;

        let final_output = if let (Some(conv2_kernels), Some(conv2_bias)) =
            (&self.conv2_kernels, &self.conv2_bias)
        {
            block1_output.conv_relu_max_pool2d_valid(
                conv2_kernels,
                Some(conv2_bias.as_slice()),
                1,
                1,
                2,
                2,
                2,
                2,
            )?
        } else {
            block1_output
        };

        let global = final_output.global_average_pool2d()?;
        let flat = global.flatten_batch_features();
        Ok(flat.first().cloned().unwrap_or_default())
    }

    fn forward_with_cache(
        &self,
        image_bytes: &[u8],
    ) -> Result<(Vec<f32>, ForwardCache), CnnImageClassifierError> {
        let (in_height, in_width, in_channels) = infer_square_dimensions_and_channels(image_bytes).ok_or(
            CnnImageClassifierError::UnsupportedImageShape {
                byte_len: image_bytes.len(),
            },
        )?;

        if in_channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: in_channels,
            });
        }

        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            in_height,
            in_width,
            in_channels,
            self.input_height,
            self.input_width,
            true,
        )?;

        let (block1_output, block1_cache) =
            forward_conv_block(&input, &self.conv1_kernels, self.conv1_bias.as_slice())?;

        let (final_output, block2_cache) = if let (Some(conv2_kernels), Some(conv2_bias)) =
            (&self.conv2_kernels, &self.conv2_bias)
        {
            let (block2_output, block2_cache) =
                forward_conv_block(&block1_output, conv2_kernels, conv2_bias.as_slice())?;
            (block2_output, Some(block2_cache))
        } else {
            (block1_output, None)
        };

        let global = final_output.global_average_pool2d()?;
        let flat = global.flatten_batch_features();
        let first = flat.first().cloned().unwrap_or_default();

        Ok((
            first,
            ForwardCache {
                block1: block1_cache,
                block2: block2_cache,
            },
        ))
    }

    pub fn train_image(&mut self, label: &str, image_bytes: &[u8]) -> Result<f32, CnnImageClassifierError> {
        let normalized = label.trim().to_ascii_lowercase();
        let class_index = self
            .label_to_index
            .get(&normalized)
            .copied()
            .ok_or_else(|| CnnImageClassifierError::UnknownLabel(normalized.clone()))?;

        let (features, cache) = self.forward_with_cache(image_bytes)?;
        let (loss, feature_grad) = self
            .head
            .train_step_with_input_gradient(features.as_slice(), class_index)?;

        self.backward_feature_extractor(&cache, feature_grad.as_slice())?;
        Ok(loss)
    }

    pub fn train_image_batch(
        &mut self,
        label: &str,
        images: &[Vec<u8>],
    ) -> Result<f32, CnnImageClassifierError> {
        if images.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "cannot train on an empty image batch",
            ));
        }

        let normalized = label.trim().to_ascii_lowercase();
        let class_index = self
            .label_to_index
            .get(&normalized)
            .copied()
            .ok_or_else(|| CnnImageClassifierError::UnknownLabel(normalized.clone()))?;

        let mut feature_batch: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        let mut caches: Vec<ForwardCache> = Vec::with_capacity(images.len());

        for image in images {
            let (features, cache) = self.forward_with_cache(image.as_slice())?;
            feature_batch.push(features);
            caches.push(cache);
        }

        let targets = vec![class_index; images.len()];
        let (loss, feature_grads) = self
            .head
            .train_batch_with_input_gradients(feature_batch.as_slice(), targets.as_slice())?;

        let batch_size = images.len() as f32;
        let conv1_snapshot = self.conv1_kernels.clone();
        let (conv1_out, conv1_in, conv1_h, conv1_w) = conv1_snapshot.shape();
        let mut conv1_kernel_grad_accum = Tensor4D::zeros(conv1_out, conv1_in, conv1_h, conv1_w);
        let mut conv1_bias_grad_accum = vec![0.0f32; self.conv1_bias.len()];

        if let (Some(conv2_kernels), Some(conv2_bias)) = (&self.conv2_kernels, &self.conv2_bias) {
            let conv2_snapshot = conv2_kernels.clone();
            let (conv2_out, conv2_in, conv2_h, conv2_w) = conv2_snapshot.shape();
            let mut conv2_kernel_grad_accum =
                Tensor4D::zeros(conv2_out, conv2_in, conv2_h, conv2_w);
            let mut conv2_bias_grad_accum = vec![0.0f32; conv2_bias.len()];

            for (cache, feature_grad) in caches.iter().zip(feature_grads.iter()) {
                let block2_cache = cache.block2.as_ref().ok_or(
                    CnnImageClassifierError::InvalidConfiguration(
                        "second convolution block cache missing during batch backprop",
                    ),
                )?;

                let block2_pooled_grad = pooled_grad_from_feature_gradient(
                    block2_cache.pooled_shape,
                    feature_grad.as_slice(),
                )?;

                let block2_backward =
                    backward_conv_block_gradients(
                        &conv2_snapshot,
                        block2_cache,
                        &block2_pooled_grad,
                        true,
                    )?;

                conv2_kernel_grad_accum.add_inplace(&block2_backward.kernel_grad)?;
                add_bias_grad(
                    conv2_bias_grad_accum.as_mut_slice(),
                    block2_backward.bias_grad.as_slice(),
                )?;

                let block2_input_grad = block2_backward.input_grad.as_ref().ok_or(
                    CnnImageClassifierError::InvalidConfiguration(
                        "second convolution block gradient is missing pooled input gradient",
                    ),
                )?;

                if block2_input_grad.shape() != cache.block1.pooled_shape {
                    return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
                        expected: cache.block1.pooled_shape,
                        actual: block2_input_grad.shape(),
                    });
                }

                let block1_backward = backward_conv_block_gradients(
                    &conv1_snapshot,
                    &cache.block1,
                    block2_input_grad,
                    false,
                )?;

                conv1_kernel_grad_accum.add_inplace(&block1_backward.kernel_grad)?;
                add_bias_grad(
                    conv1_bias_grad_accum.as_mut_slice(),
                    block1_backward.bias_grad.as_slice(),
                )?;
            }

            apply_conv_gradients(
                &mut self.conv2_kernels,
                &mut self.conv2_bias,
                &conv2_kernel_grad_accum,
                conv2_bias_grad_accum.as_slice(),
                self.feature_learning_rate,
                batch_size,
            )?;
        } else {
            for (cache, feature_grad) in caches.iter().zip(feature_grads.iter()) {
                let block1_pooled_grad = pooled_grad_from_feature_gradient(
                    cache.block1.pooled_shape,
                    feature_grad.as_slice(),
                )?;

                let block1_backward = backward_conv_block_gradients(
                    &conv1_snapshot,
                    &cache.block1,
                    &block1_pooled_grad,
                    false,
                )?;

                conv1_kernel_grad_accum.add_inplace(&block1_backward.kernel_grad)?;
                add_bias_grad(
                    conv1_bias_grad_accum.as_mut_slice(),
                    block1_backward.bias_grad.as_slice(),
                )?;
            }
        }

        apply_conv_gradients_single(
            &mut self.conv1_kernels,
            self.conv1_bias.as_mut_slice(),
            &conv1_kernel_grad_accum,
            conv1_bias_grad_accum.as_slice(),
            self.feature_learning_rate,
            batch_size,
        )?;

        Ok(loss)
    }

    pub fn predict_with_confidence(
        &self,
        image_bytes: &[u8],
    ) -> Result<Option<(String, f32)>, CnnImageClassifierError> {
        let features = self.extract_features(image_bytes)?;
        let probs = self.head.probabilities(features.as_slice())?;

        let mut best_idx = 0usize;
        let mut best_prob = f32::NEG_INFINITY;
        for (idx, prob) in probs.iter().enumerate() {
            if *prob > best_prob {
                best_prob = *prob;
                best_idx = idx;
            }
        }

        if best_prob < self.min_confidence {
            return Ok(None);
        }

        let label = self
            .class_labels
            .get(best_idx)
            .cloned()
            .ok_or(CnnImageClassifierError::InvalidConfiguration(
                "predicted class index out of range",
            ))?;

        Ok(Some((label, best_prob)))
    }

    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let snapshot = CnnImageClassifierSnapshot {
            input_height: self.input_height,
            input_width: self.input_width,
            input_channels: self.input_channels,
            class_labels: self.class_labels.clone(),
            label_to_index: self.label_to_index.clone(),
            conv1_kernels: self.conv1_kernels.clone(),
            conv1_bias: self.conv1_bias.clone(),
            conv2_kernels: self.conv2_kernels.clone(),
            conv2_bias: self.conv2_bias.clone(),
            feature_learning_rate: self.feature_learning_rate,
            head: self.head.clone(),
            min_confidence: self.min_confidence,
        };

        let encoded = bincode::serialize(&snapshot).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to serialize CNN classifier snapshot '{}': {err}",
                    path.display()
                ),
            )
        })?;

        let mut bytes = Vec::with_capacity(CNN_CLASSIFIER_BIN_MAGIC.len() + encoded.len());
        bytes.extend_from_slice(&CNN_CLASSIFIER_BIN_MAGIC);
        bytes.extend_from_slice(&encoded);

        let tmp_path = path.with_extension(format!(
            "{}.tmp",
            path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("cnn")
        ));

        fs::write(&tmp_path, bytes)?;
        fs::rename(&tmp_path, path)?;

        Ok(())
    }

    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;

        if bytes.len() < CNN_CLASSIFIER_BIN_MAGIC.len()
            || bytes[0..CNN_CLASSIFIER_BIN_MAGIC.len()] != CNN_CLASSIFIER_BIN_MAGIC
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot magic in '{}'",
                    path.display()
                ),
            ));
        }

        let payload = &bytes[CNN_CLASSIFIER_BIN_MAGIC.len()..];

        let snapshot: CnnImageClassifierSnapshot = match bincode::deserialize(payload) {
            Ok(snapshot) => snapshot,
            Err(_) => {
                let legacy: LegacyCnnImageClassifierSnapshot =
                    bincode::deserialize(payload).map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "failed to deserialize CNN classifier snapshot '{}': {err}",
                                path.display()
                            ),
                        )
                    })?;
                legacy.into()
            }
        };

        if snapshot.class_labels.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (requires at least two class labels)",
                    path.display()
                ),
            ));
        }

        if snapshot.label_to_index.len() != snapshot.class_labels.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (label map and labels mismatch)",
                    path.display()
                ),
            ));
        }

        if snapshot.conv1_kernels.shape().0 != snapshot.conv1_bias.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (conv1 channels and bias mismatch)",
                    path.display()
                ),
            ));
        }

        if snapshot.conv1_kernels.shape().1 != snapshot.input_channels {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (conv1 kernel input channels and model input channels mismatch)",
                    path.display()
                ),
            ));
        }

        if (snapshot.conv2_kernels.is_some()) != (snapshot.conv2_bias.is_some()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (conv2 kernels/bias must both be present or absent)",
                    path.display()
                ),
            ));
        }

        if let (Some(conv2_kernels), Some(conv2_bias)) = (&snapshot.conv2_kernels, &snapshot.conv2_bias)
            && conv2_kernels.shape().0 != conv2_bias.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (conv2 channels and bias mismatch)",
                    path.display()
                ),
            ));
        }

        let (head_input_dim, head_output_dim) = snapshot.head.dimensions();
        let feature_dim = snapshot
            .conv2_kernels
            .as_ref()
            .map(|kernels| kernels.shape().0)
            .unwrap_or_else(|| snapshot.conv1_kernels.shape().0);

        if head_input_dim != feature_dim || head_output_dim != snapshot.class_labels.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (head dimensions do not match model state)",
                    path.display()
                ),
            ));
        }

        Ok(Self {
            input_height: snapshot.input_height,
            input_width: snapshot.input_width,
            input_channels: snapshot.input_channels,
            class_labels: snapshot.class_labels,
            label_to_index: snapshot.label_to_index,
            conv1_kernels: snapshot.conv1_kernels,
            conv1_bias: snapshot.conv1_bias,
            conv2_kernels: snapshot.conv2_kernels,
            conv2_bias: snapshot.conv2_bias,
            feature_learning_rate: snapshot.feature_learning_rate,
            head: snapshot.head,
            min_confidence: snapshot.min_confidence,
        })
    }

    fn backward_feature_extractor(
        &mut self,
        cache: &ForwardCache,
        feature_grad: &[f32],
    ) -> Result<(), CnnImageClassifierError> {
        self.backward_feature_extractor_with_scale(cache, feature_grad, 1.0)
    }

    fn backward_feature_extractor_with_scale(
        &mut self,
        cache: &ForwardCache,
        feature_grad: &[f32],
        grad_scale: f32,
    ) -> Result<(), CnnImageClassifierError> {
        let effective_learning_rate = self.feature_learning_rate * grad_scale.max(0.0);

        if let Some(block2_cache) = cache.block2.as_ref() {
            let block2_pooled_grad = pooled_grad_from_feature_gradient(
                block2_cache.pooled_shape,
                feature_grad,
            )?;

            let (conv2_kernels, conv2_bias) = match (&mut self.conv2_kernels, &mut self.conv2_bias) {
                (Some(kernels), Some(bias)) => (kernels, bias.as_mut_slice()),
                _ => {
                    return Err(CnnImageClassifierError::InvalidConfiguration(
                        "second convolution block is missing parameters",
                    ))
                }
            };

            let grad_to_block1_pooled = backward_conv_block(
                conv2_kernels,
                conv2_bias,
                block2_cache,
                &block2_pooled_grad,
                effective_learning_rate,
                true,
            )?;

            if grad_to_block1_pooled.shape() != cache.block1.pooled_shape {
                return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
                    expected: cache.block1.pooled_shape,
                    actual: grad_to_block1_pooled.shape(),
                });
            }

            let _ = backward_conv_block(
                &mut self.conv1_kernels,
                self.conv1_bias.as_mut_slice(),
                &cache.block1,
                &grad_to_block1_pooled,
                effective_learning_rate,
                false,
            )?;

            return Ok(());
        }

        let block1_pooled_grad =
            pooled_grad_from_feature_gradient(cache.block1.pooled_shape, feature_grad)?;
        let _ = backward_conv_block(
            &mut self.conv1_kernels,
            self.conv1_bias.as_mut_slice(),
            &cache.block1,
            &block1_pooled_grad,
            effective_learning_rate,
            false,
        )?;
        Ok(())
    }
}

fn default_input_channels() -> usize {
    1
}

fn initialize_conv_kernels(
    out_channels: usize,
    in_channels: usize,
    oriented_first_layer: bool,
) -> Result<Tensor4D, CnnImageClassifierError> {
    let mut values = Vec::with_capacity(out_channels * in_channels * 9);

    for out_c in 0..out_channels {
        for in_c in 0..in_channels {
            let kernel = if oriented_first_layer && in_channels == 1 && out_c == 0 {
                vec![
                    -1.0, 0.0, 1.0,
                    -1.0, 0.0, 1.0,
                    -1.0, 0.0, 1.0,
                ]
            } else if oriented_first_layer && in_channels == 1 && out_c == 1 {
                vec![
                    -1.0, -1.0, -1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0,
                ]
            } else {
                let center_weight = 1.0f32 / in_channels as f32;
                let sign = if (out_c + in_c) % 2 == 0 { 1.0 } else { -1.0 };
                vec![
                    0.0, 0.0, 0.0,
                    0.0, sign * center_weight, 0.0,
                    0.0, 0.0, 0.0,
                ]
            };
            values.extend(kernel);
        }
    }

    Tensor4D::from_vec(out_channels, in_channels, 3, 3, values)
        .map_err(CnnImageClassifierError::Tensor)
}

fn forward_conv_block(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: &[f32],
) -> Result<(Tensor4D, ConvBlockCache), CnnImageClassifierError> {
    let conv_pre = input.conv2d_valid(kernels, Some(bias), 1, 1)?;
    let mut relu = conv_pre.clone();
    relu.relu_inplace();

    let (pooled, pool_indices) = max_pool2d_with_indices(&relu, 2, 2, 2, 2)?;
    let pooled_shape = pooled.shape();

    Ok((
        pooled,
        ConvBlockCache {
            input: input.clone(),
            conv_pre_activation: conv_pre,
            relu_output: relu,
            pool_indices,
            pooled_shape,
        },
    ))
}

fn pooled_grad_from_feature_gradient(
    pooled_shape: (usize, usize, usize, usize),
    feature_grad: &[f32],
) -> Result<Tensor4D, CnnImageClassifierError> {
    let (_, channels, pooled_h, pooled_w) = pooled_shape;

    if feature_grad.len() != channels {
        return Err(CnnImageClassifierError::GradientShapeMismatch {
            expected: channels,
            actual: feature_grad.len(),
        });
    }

    let mut pooled_grad = Tensor4D::zeros(1, channels, pooled_h, pooled_w);
    let pooled_area = (pooled_h * pooled_w).max(1) as f32;

    for (channel, grad_value) in feature_grad.iter().enumerate() {
        let per_cell = *grad_value / pooled_area;
        for py in 0..pooled_h {
            for px in 0..pooled_w {
                pooled_grad.set(0, channel, py, px, per_cell)?;
            }
        }
    }

    Ok(pooled_grad)
}

fn backward_conv_block(
    kernels: &mut Tensor4D,
    bias: &mut [f32],
    cache: &ConvBlockCache,
    pooled_grad: &Tensor4D,
    learning_rate: f32,
    compute_input_grad: bool,
) -> Result<Tensor4D, CnnImageClassifierError> {
    let backward = backward_conv_block_gradients(kernels, cache, pooled_grad, compute_input_grad)?;
    apply_conv_gradients_single(
        kernels,
        bias,
        &backward.kernel_grad,
        backward.bias_grad.as_slice(),
        learning_rate,
        1.0,
    )?;

    Ok(backward.input_grad.unwrap_or_else(|| Tensor4D::zeros(0, 0, 0, 0)))
}

fn backward_conv_block_gradients(
    kernels: &Tensor4D,
    cache: &ConvBlockCache,
    pooled_grad: &Tensor4D,
    compute_input_grad: bool,
) -> Result<ConvBlockBackward, CnnImageClassifierError> {
    let pooled_shape = pooled_grad.shape();
    if pooled_shape != cache.pooled_shape {
        return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
            expected: cache.pooled_shape,
            actual: pooled_shape,
        });
    }

    let (_, channels, pooled_h, pooled_w) = cache.pooled_shape;
    let (_, _, relu_h, relu_w) = cache.relu_output.shape();

    let mut conv_grad = Tensor4D::zeros(1, channels, relu_h, relu_w);
    let relu_plane = relu_h * relu_w;
    let pooled_plane = pooled_h * pooled_w;

    for channel in 0..channels {
        let conv_channel_base = channel * relu_plane;
        let pooled_channel_base = channel * pooled_plane;

        for py in 0..pooled_h {
            for px in 0..pooled_w {
                let pooled_idx = pooled_channel_base + py * pooled_w + px;
                let (src_y, src_x) = cache.pool_indices[pooled_idx];
                let conv_idx = conv_channel_base + src_y * relu_w + src_x;
                conv_grad.as_mut_slice()[conv_idx] += pooled_grad.as_slice()[pooled_idx];
            }
        }
    }

    for (grad, pre) in conv_grad
        .as_mut_slice()
        .iter_mut()
        .zip(cache.conv_pre_activation.as_slice().iter())
    {
        if *pre <= 0.0 {
            *grad = 0.0;
        }
    }

    let (_, in_channels, kernel_h, kernel_w) = kernels.shape();
    let (_, _, conv_h, conv_w) = cache.conv_pre_activation.shape();

    let mut kernel_grad = Tensor4D::zeros(channels, in_channels, kernel_h, kernel_w);
    let mut bias_grad = vec![0.0f32; channels];
    let conv_plane = conv_h * conv_w;
    let (_, _, in_h, in_w) = cache.input.shape();
    let input_plane = in_h * in_w;
    let kernel_plane = kernel_h * kernel_w;

    for (out_c, bias_slot) in bias_grad.iter_mut().enumerate() {
        let conv_channel_base = out_c * conv_plane;
        let conv_channel = &conv_grad.as_slice()[conv_channel_base..conv_channel_base + conv_plane];
        *bias_slot = conv_channel.iter().copied().sum();

        for in_c in 0..in_channels {
            let input_channel_base = in_c * input_plane;
            let kernel_channel_base = (out_c * in_channels + in_c) * kernel_plane;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let mut accum = 0.0f32;
                    for oy in 0..conv_h {
                        let conv_row_base = conv_channel_base + oy * conv_w;
                        let input_row_base = input_channel_base + (oy + ky) * in_w + kx;
                        for ox in 0..conv_w {
                            let grad = conv_grad.as_slice()[conv_row_base + ox];
                            let inp = cache.input.as_slice()[input_row_base + ox];
                            accum += grad * inp;
                        }
                    }
                    kernel_grad.as_mut_slice()[kernel_channel_base + ky * kernel_w + kx] = accum;
                }
            }
        }
    }

    let input_grad = if compute_input_grad {
        let mut input_grad = Tensor4D::zeros(1, in_channels, in_h, in_w);

        for in_c in 0..in_channels {
            let input_channel_base = in_c * input_plane;
            for iy in 0..in_h {
                for ix in 0..in_w {
                    let mut accum = 0.0f32;
                    for out_c in 0..channels {
                        let conv_channel_base = out_c * conv_plane;
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
                                let weight = kernels.as_slice()[kernel_channel_base + ky * kernel_w + kx];
                                accum += grad * weight;
                            }
                        }
                    }
                    input_grad.as_mut_slice()[input_channel_base + iy * in_w + ix] = accum;
                }
            }
        }

        Some(input_grad)
    } else {
        None
    };

    Ok(ConvBlockBackward {
        kernel_grad,
        bias_grad,
        input_grad,
    })
}

fn add_bias_grad(accum: &mut [f32], grad: &[f32]) -> Result<(), CnnImageClassifierError> {
    if accum.len() != grad.len() {
        return Err(CnnImageClassifierError::GradientShapeMismatch {
            expected: accum.len(),
            actual: grad.len(),
        });
    }

    for (left, right) in accum.iter_mut().zip(grad.iter()) {
        *left += *right;
    }

    Ok(())
}

fn apply_conv_gradients(
    kernels: &mut Option<Tensor4D>,
    bias: &mut Option<Vec<f32>>,
    kernel_grad: &Tensor4D,
    bias_grad: &[f32],
    learning_rate: f32,
    batch_size: f32,
) -> Result<(), CnnImageClassifierError> {
    let (kernels, bias) = match (kernels.as_mut(), bias.as_mut()) {
        (Some(kernels), Some(bias)) => (kernels, bias.as_mut_slice()),
        _ => {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "second convolution block is missing parameters",
            ))
        }
    };

    apply_conv_gradients_single(
        kernels,
        bias,
        kernel_grad,
        bias_grad,
        learning_rate,
        batch_size,
    )
}

fn apply_conv_gradients_single(
    kernels: &mut Tensor4D,
    bias: &mut [f32],
    kernel_grad: &Tensor4D,
    bias_grad: &[f32],
    learning_rate: f32,
    batch_size: f32,
) -> Result<(), CnnImageClassifierError> {
    if kernels.shape() != kernel_grad.shape() {
        return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
            expected: kernels.shape(),
            actual: kernel_grad.shape(),
        });
    }

    if bias.len() != bias_grad.len() {
        return Err(CnnImageClassifierError::GradientShapeMismatch {
            expected: bias.len(),
            actual: bias_grad.len(),
        });
    }

    let (out_channels, in_channels, kernel_h, kernel_w) = kernels.shape();
    let scale = if batch_size > 0.0 { 1.0 / batch_size } else { 1.0 };
    let kernel_plane = kernel_h * kernel_w;

    for out_c in 0..out_channels {
        for in_c in 0..in_channels {
            let kernel_channel_base = (out_c * in_channels + in_c) * kernel_plane;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let idx = kernel_channel_base + ky * kernel_w + kx;
                    kernels.as_mut_slice()[idx] -= learning_rate * kernel_grad.as_slice()[idx] * scale;
                }
            }
        }

        bias[out_c] -= learning_rate * bias_grad[out_c] * scale;
    }

    Ok(())
}

fn max_pool2d_with_indices(
    input: &Tensor4D,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(Tensor4D, Vec<(usize, usize)>), TensorError> {
    if window_h == 0 || window_w == 0 {
        return Err(TensorError::InvalidArgument(
            "pooling window must be greater than zero",
        ));
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorError::InvalidArgument("stride must be greater than zero"));
    }

    let (n, c, h, w) = input.shape();
    if h < window_h || w < window_w {
        return Err(TensorError::InvalidArgument(
            "pooling window cannot exceed input spatial size",
        ));
    }

    let out_h = ((h - window_h) / stride_h) + 1;
    let out_w = ((w - window_w) / stride_w) + 1;
    let mut pooled = Tensor4D::zeros(n, c, out_h, out_w);
    let mut indices = vec![(0usize, 0usize); c * out_h * out_w];

    let input_channel_stride = h * w;
    let output_channel_stride = out_h * out_w;

    for channel in 0..c {
        let input_channel_base = channel * input_channel_stride;
        let output_channel_base = channel * output_channel_stride;

        for oy in 0..out_h {
            let in_y = oy * stride_h;
            let output_row_base = output_channel_base + oy * out_w;

            for ox in 0..out_w {
                let in_x = ox * stride_w;
                let mut max_value = f32::NEG_INFINITY;
                let mut max_idx = (in_y, in_x);

                for wy in 0..window_h {
                    let row_base = input_channel_base + (in_y + wy) * w + in_x;
                    let row = &input.as_slice()[row_base..row_base + window_w];

                    for (wx, value) in row.iter().copied().enumerate() {
                        let src_y = in_y + wy;
                        let src_x = in_x + wx;
                        if value > max_value {
                            max_value = value;
                            max_idx = (src_y, src_x);
                        }
                    }
                }

                pooled.as_mut_slice()[output_row_base + ox] = max_value;
                let idx = ((channel * out_h) + oy) * out_w + ox;
                indices[idx] = max_idx;
            }
        }
    }

    Ok((pooled, indices))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn classifier_test_dir(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("./target/cnn_classifier_snapshots");
        path.push(test_name);
        path
    }

    fn cleanup_classifier_test_dir(test_name: &str) {
        let _ = fs::remove_dir_all(classifier_test_dir(test_name));
    }

    fn vertical_stripes_image_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for _y in 0..8 {
            for x in 0..8 {
                if x % 2 == 0 {
                    bytes.push(220);
                } else {
                    bytes.push(20);
                }
            }
        }
        bytes
    }

    fn horizontal_stripes_image_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for y in 0..8 {
            for _x in 0..8 {
                if y % 2 == 0 {
                    bytes.push(220);
                } else {
                    bytes.push(20);
                }
            }
        }
        bytes
    }

    #[test]
    fn cnn_classifier_trains_and_predicts_on_simple_patterns() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.4);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..40 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let cat_prediction = classifier
            .predict_with_confidence(cat.as_slice())
            .unwrap_or_else(|_| panic!("prediction should succeed"));
        let dog_prediction = classifier
            .predict_with_confidence(dog.as_slice())
            .unwrap_or_else(|_| panic!("prediction should succeed"));

        assert_eq!(cat_prediction.map(|value| value.0), Some("animal_cat".to_string()));
        assert_eq!(dog_prediction.map(|value| value.0), Some("animal_dog".to_string()));
    }

    #[test]
    fn cnn_classifier_two_layer_trains_and_predicts_on_simple_patterns() {
        let mut classifier = CnnImageClassifier::new_two_layer(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            4,
            4,
            0.15,
        )
        .unwrap_or_else(|_| panic!("two-layer cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..80 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let cat_prediction = classifier
            .predict_with_confidence(cat.as_slice())
            .unwrap_or_else(|_| panic!("prediction should succeed"));
        let dog_prediction = classifier
            .predict_with_confidence(dog.as_slice())
            .unwrap_or_else(|_| panic!("prediction should succeed"));

        assert_eq!(cat_prediction.map(|value| value.0), Some("animal_cat".to_string()));
        assert_eq!(dog_prediction.map(|value| value.0), Some("animal_dog".to_string()));
    }

    #[test]
    fn cnn_classifier_rejects_non_square_image() {
        let classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.1,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));

        let result = classifier.extract_features(&vec![1u8; 1000]);
        assert!(matches!(
            result,
            Err(CnnImageClassifierError::UnsupportedImageShape { byte_len: 1000 })
        ));
    }

    #[test]
    fn cnn_classifier_supports_rgb_square_images_when_configured_for_three_channels() {
        let mut classifier = CnnImageClassifier::new_with_feature_channels_and_input_channels(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            3,
            &[2],
            0.2,
        )
        .unwrap_or_else(|_| panic!("rgb cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let grayscale_cat = vertical_stripes_image_8x8();
        let grayscale_dog = horizontal_stripes_image_8x8();

        let cat: Vec<u8> = grayscale_cat
            .iter()
            .flat_map(|value| [*value, *value, *value])
            .collect();
        let dog: Vec<u8> = grayscale_dog
            .iter()
            .flat_map(|value| [*value, *value, *value])
            .collect();

        for _ in 0..40 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let predicted_cat = classifier
            .predict_with_confidence(cat.as_slice())
            .unwrap_or_else(|_| panic!("rgb cat prediction should succeed"));
        let predicted_dog = classifier
            .predict_with_confidence(dog.as_slice())
            .unwrap_or_else(|_| panic!("rgb dog prediction should succeed"));

        assert!(predicted_cat.is_some());
        assert!(predicted_dog.is_some());
    }

    #[test]
    fn cnn_classifier_snapshot_round_trip_preserves_predictions() {
        let test_name = "cnn_classifier_snapshot_round_trip";
        let test_dir = classifier_test_dir(test_name);
        cleanup_classifier_test_dir(test_name);

        let snapshot_path = test_dir.join("classifier.cnn");

        let mut classifier = CnnImageClassifier::new_two_layer(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            4,
            4,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..40 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let before_cat = classifier
            .predict_with_confidence(cat.as_slice())
            .unwrap_or_else(|_| panic!("prediction before save should succeed"))
            .map(|value| value.0);
        let before_dog = classifier
            .predict_with_confidence(dog.as_slice())
            .unwrap_or_else(|_| panic!("prediction before save should succeed"))
            .map(|value| value.0);

        assert!(classifier.save_to_file(snapshot_path.as_path()).is_ok());

        let restored = CnnImageClassifier::load_from_file(snapshot_path.as_path())
            .unwrap_or_else(|_| panic!("restored classifier should load"));

        let after_cat = restored
            .predict_with_confidence(cat.as_slice())
            .unwrap_or_else(|_| panic!("prediction after load should succeed"))
            .map(|value| value.0);
        let after_dog = restored
            .predict_with_confidence(dog.as_slice())
            .unwrap_or_else(|_| panic!("prediction after load should succeed"))
            .map(|value| value.0);

        assert_eq!(before_cat, after_cat);
        assert_eq!(before_dog, after_dog);

        cleanup_classifier_test_dir(test_name);
    }

    #[test]
    fn cnn_classifier_inference_path_matches_cached_forward_features() {
        let classifier = CnnImageClassifier::new_two_layer(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            4,
            4,
            0.15,
        )
        .unwrap_or_else(|_| panic!("two-layer classifier should initialize"));

        let image = vertical_stripes_image_8x8();
        let (cached_features, _) = classifier
            .forward_with_cache(image.as_slice())
            .unwrap_or_else(|_| panic!("cached forward should succeed"));
        let inference_features = classifier
            .forward_for_inference(image.as_slice())
            .unwrap_or_else(|_| panic!("inference forward should succeed"));

        // Allow small floating-point difference from fused path vs chained ops.
        assert_eq!(cached_features.len(), inference_features.len());
        for (a, b) in cached_features.iter().zip(inference_features.iter()) {
            assert!((a - b).abs() < 1e-4, "feature mismatch: {a} vs {b}");
        }
    }
}
