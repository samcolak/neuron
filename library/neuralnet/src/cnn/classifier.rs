use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::tensor::backend::{
    active_backend,
    ConvBlockBackwardGradients,
};
use crate::tensor::adapters::{
    image_batch_to_tensor_nchw_resized_with_channels,
    image_bytes_to_tensor_nchw_resized_with_channels,
    TensorAdapterError,
};
use crate::cnn::data_pipeline::ImageTensorShape;
#[cfg(feature = "offloading-mlx")]
use crate::tensor::offloading::mlx_backend::{
    mlx_conv2d_valid_with_mirrored_params,
};
use crate::tensor::parameters::ConvParameterState;
use crate::tensor::tensor4d::{Tensor4D, TensorError};
use crate::training::linear_head::{LinearHead, LinearHeadError, LinearOptimizer};

const CNN_CLASSIFIER_BIN_MAGIC: [u8; 4] = *b"CNN1";
static CNN_BATCH_PREPROCESS_ENABLED: OnceLock<bool> = OnceLock::new();

fn cnn_batch_preprocess_enabled() -> bool {
    *CNN_BATCH_PREPROCESS_ENABLED.get_or_init(|| {
        env::var("NEURALNET_CNN_BATCH_PREPROCESS")
            .ok()
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !(normalized == "0"
                    || normalized == "false"
                    || normalized == "no"
                    || normalized == "off"
                    || normalized == "legacy")
            })
            .unwrap_or(true)
    })
}

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
    feature_blocks: Vec<ConvParameterState>,
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
    feature_block_kernels: Vec<Tensor4D>,
    feature_block_biases: Vec<Vec<f32>>,
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
            feature_block_kernels: vec![value.feature_kernels],
            feature_block_biases: vec![value.feature_bias],
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
    pool_indices: Vec<(usize, usize)>,
    pooled_shape: (usize, usize, usize, usize),
}

#[derive(Debug, Clone)]
struct ForwardCache {
    blocks: Vec<ConvBlockCache>,
}

#[derive(Debug, Clone)]
struct BatchForwardCache {
    blocks: Vec<ConvBlockCache>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnBatchPredictOptions {
    pub max_micro_batch_size: usize,
    pub enable_batch_preprocess: bool,
}

impl Default for CnnBatchPredictOptions {
    fn default() -> Self {
        Self {
            max_micro_batch_size: 32,
            enable_batch_preprocess: cnn_batch_preprocess_enabled(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnBatchPredictReport {
    pub predictions: Vec<Option<(String, f32)>>,
    pub total_images: usize,
    pub micro_batch_count: usize,
    pub max_micro_batch_size: usize,
    pub preprocessing_elapsed_ms: f64,
    pub model_elapsed_ms: f64,
    pub total_elapsed_ms: f64,
    pub throughput_images_per_sec: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnCoalescingPredictOptions {
    pub max_micro_batch_size: usize,
    pub max_queue_size: usize,
    pub max_queue_delay_ms: u64,
    pub enable_batch_preprocess: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CnnForwardStageMetrics {
    pub conv_elapsed_ms: f64,
    pub pool_elapsed_ms: f64,
    pub global_pool_elapsed_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CnnBackpropStageMetrics {
    pub pooled_grad_elapsed_ms: f64,
    pub gradients_elapsed_ms: f64,
    pub apply_update_elapsed_ms: f64,
    pub input_grad_transfer_elapsed_ms: f64,
    pub unpool_relu_elapsed_ms: f64,
    pub dw_elapsed_ms: f64,
    pub dinput_elapsed_ms: f64,
    pub bias_grad_elapsed_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CnnTrainTimingReport {
    pub input_elapsed_ms: f64,
    pub feature_elapsed_ms: f64,
    pub head_elapsed_ms: f64,
    pub backprop_elapsed_ms: f64,
    pub total_elapsed_ms: f64,
    pub forward_stage_metrics: CnnForwardStageMetrics,
    pub backprop_stage_metrics: CnnBackpropStageMetrics,
}

impl Default for CnnCoalescingPredictOptions {
    fn default() -> Self {
        Self {
            max_micro_batch_size: 32,
            max_queue_size: 64,
            max_queue_delay_ms: 2,
            enable_batch_preprocess: cnn_batch_preprocess_enabled(),
        }
    }
}

pub struct CnnCoalescingBatchPredictor<'a> {
    classifier: &'a CnnImageClassifier,
    options: CnnCoalescingPredictOptions,
    image_shape: Option<ImageTensorShape>,
    pending: Vec<Vec<u8>>,
    ready_predictions: Vec<Option<(String, f32)>>,
    queue_started_at: Option<Instant>,
    flushed_images: usize,
    flush_count: usize,
    preprocessing_elapsed_ms: f64,
    model_elapsed_ms: f64,
    total_elapsed_ms: f64,
}

type CnnPrediction = Option<(String, f32)>;
type CnnPredictionResult = Result<CnnPrediction, CnnImageClassifierError>;
type CnnPredictionSender = mpsc::Sender<CnnPredictionResult>;

#[derive(Debug)]
pub struct CnnCoalescedPredictionHandle {
    response_rx: mpsc::Receiver<CnnPredictionResult>,
}

impl CnnCoalescedPredictionHandle {

    pub fn wait(self) -> CnnPredictionResult {
        self.response_rx.recv().map_err(|_| {
            CnnImageClassifierError::InvalidConfiguration(
                "coalescing scheduler response channel closed before delivering prediction",
            )
        })?
    }

}

pub struct CnnCoalescingScheduler<'a> {
    predictor: CnnCoalescingBatchPredictor<'a>,
    responders: VecDeque<CnnPredictionSender>,
    is_shutdown: bool,
}

impl<'a> CnnCoalescingScheduler<'a> {

    pub fn submit(
        &mut self,
        image: Vec<u8>,
    ) -> Result<CnnCoalescedPredictionHandle, CnnImageClassifierError> {
        if self.is_shutdown {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "coalescing scheduler is not accepting new requests",
            ));
        }

        let (response_tx, response_rx) = mpsc::channel();
        self.responders.push_back(response_tx);

        if let Err(err) = self.predictor.enqueue(image) {
            fail_all_pending_responders(&mut self.responders);
            return Err(err);
        }

        dispatch_ready_predictions(&mut self.predictor, &mut self.responders);
        Ok(CnnCoalescedPredictionHandle { response_rx })
    }

    pub fn flush_if_due(&mut self) -> Result<bool, CnnImageClassifierError> {
        if self.is_shutdown {
            return Ok(false);
        }

        let flushed = self.predictor.flush_if_due()?;
        if flushed {
            dispatch_ready_predictions(&mut self.predictor, &mut self.responders);
        }
        Ok(flushed)
    }

    pub fn flush(&mut self) -> Result<(), CnnImageClassifierError> {
        if self.is_shutdown {
            return Ok(());
        }

        let _ = self.predictor.flush()?;
        dispatch_ready_predictions(&mut self.predictor, &mut self.responders);
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<(), CnnImageClassifierError> {
        self.flush()?;
        self.is_shutdown = true;
        Ok(())
    }

}

impl Drop for CnnCoalescingScheduler<'_> {

    fn drop(&mut self) {
        if !self.is_shutdown {
            let _ = self.predictor.flush();
            dispatch_ready_predictions(&mut self.predictor, &mut self.responders);
            self.is_shutdown = true;
        }
    }
    
}

fn dispatch_ready_predictions(
    predictor: &mut CnnCoalescingBatchPredictor<'_>,
    responders: &mut VecDeque<CnnPredictionSender>,
) {
    for prediction in predictor.take_ready() {
        if let Some(response_tx) = responders.pop_front() {
            let _ = response_tx.send(Ok(prediction));
        }
    }
}

fn fail_all_pending_responders(
    responders: &mut VecDeque<CnnPredictionSender>,
) {
    while let Some(response_tx) = responders.pop_front() {
        let _ = response_tx.send(Err(CnnImageClassifierError::InvalidConfiguration(
            "coalescing scheduler failed while processing pending requests",
        )));
    }
}

impl<'a> CnnCoalescingBatchPredictor<'a> {

    pub fn new(classifier: &'a CnnImageClassifier, options: CnnCoalescingPredictOptions) -> Self {
        Self {
            classifier,
            options,
            image_shape: None,
            pending: Vec::new(),
            ready_predictions: Vec::new(),
            queue_started_at: None,
            flushed_images: 0,
            flush_count: 0,
            preprocessing_elapsed_ms: 0.0,
            model_elapsed_ms: 0.0,
            total_elapsed_ms: 0.0,
        }
    }

    pub fn new_with_shape(
        classifier: &'a CnnImageClassifier,
        options: CnnCoalescingPredictOptions,
        image_shape: ImageTensorShape,
    ) -> Self {
        Self {
            classifier,
            options,
            image_shape: Some(image_shape),
            pending: Vec::new(),
            ready_predictions: Vec::new(),
            queue_started_at: None,
            flushed_images: 0,
            flush_count: 0,
            preprocessing_elapsed_ms: 0.0,
            model_elapsed_ms: 0.0,
            total_elapsed_ms: 0.0,
        }
    }

    pub fn enqueue(&mut self, image: Vec<u8>) -> Result<(), CnnImageClassifierError> {
        if self.pending.is_empty() {
            self.queue_started_at = Some(Instant::now());
        }

        self.pending.push(image);

        let max_queue_size = self.options.max_queue_size.max(1);
        if self.pending.len() >= max_queue_size {
            let _ = self.flush()?;
        }

        Ok(())
    }

    pub fn flush_if_due(&mut self) -> Result<bool, CnnImageClassifierError> {

        if self.pending.is_empty() {
            return Ok(false);
        }

        let should_flush = self
            .queue_started_at
            .is_some_and(|started| started.elapsed().as_millis() >= self.options.max_queue_delay_ms as u128);

        if should_flush {
            return self.flush();
        }

        Ok(false)
    }

    pub fn flush(&mut self) -> Result<bool, CnnImageClassifierError> {

        if self.pending.is_empty() {
            return Ok(false);
        }

        let report = if let Some(shape) = self.image_shape {
            self.classifier.predict_batch_with_confidence_report_with_dimensions(
                self.pending.as_slice(),
                shape.height,
                shape.width,
                shape.channels,
                CnnBatchPredictOptions {
                    max_micro_batch_size: self.options.max_micro_batch_size,
                    enable_batch_preprocess: self.options.enable_batch_preprocess,
                },
            )?
        } else {
            self.classifier.predict_batch_with_confidence_report(
                self.pending.as_slice(),
                CnnBatchPredictOptions {
                    max_micro_batch_size: self.options.max_micro_batch_size,
                    enable_batch_preprocess: self.options.enable_batch_preprocess,
                },
            )?
        };

        self.flushed_images += report.total_images;
        self.flush_count += 1;
        self.preprocessing_elapsed_ms += report.preprocessing_elapsed_ms;
        self.model_elapsed_ms += report.model_elapsed_ms;
        self.total_elapsed_ms += report.total_elapsed_ms;
        self.ready_predictions.extend(report.predictions);

        self.pending.clear();
        self.queue_started_at = None;
        Ok(true)

    }

    pub fn take_ready(&mut self) -> Vec<Option<(String, f32)>> {
        std::mem::take(&mut self.ready_predictions)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn finish(mut self) -> Result<CnnBatchPredictReport, CnnImageClassifierError> {

        let _ = self.flush()?;

        let throughput = if self.total_elapsed_ms > 0.0 {
            self.flushed_images as f64 / (self.total_elapsed_ms / 1_000.0)
        } else {
            0.0
        };

        Ok(CnnBatchPredictReport {
            predictions: self.ready_predictions,
            total_images: self.flushed_images,
            micro_batch_count: self.flush_count,
            max_micro_batch_size: self.options.max_micro_batch_size.max(1),
            preprocessing_elapsed_ms: self.preprocessing_elapsed_ms,
            model_elapsed_ms: self.model_elapsed_ms,
            total_elapsed_ms: self.total_elapsed_ms,
            throughput_images_per_sec: throughput,
        })
        
    }

}

impl CnnImageClassifier {

    pub fn coalescing_batch_predictor(
        &self,
        options: CnnCoalescingPredictOptions,
    ) -> CnnCoalescingBatchPredictor<'_> {
        CnnCoalescingBatchPredictor::new(self, options)
    }

    pub fn coalescing_batch_predictor_with_dimensions(
        &self,
        options: CnnCoalescingPredictOptions,
        image_shape: ImageTensorShape,
    ) -> CnnCoalescingBatchPredictor<'_> {
        CnnCoalescingBatchPredictor::new_with_shape(self, options, image_shape)
    }

    pub fn start_coalescing_scheduler(
        &self,
        options: CnnCoalescingPredictOptions,
    ) -> CnnCoalescingScheduler<'_> {
        CnnCoalescingScheduler {
            predictor: CnnCoalescingBatchPredictor::new(self, options),
            responders: VecDeque::new(),
            is_shutdown: false,
        }
    }

    pub fn start_coalescing_scheduler_with_dimensions(
        &self,
        options: CnnCoalescingPredictOptions,
        image_shape: ImageTensorShape,
    ) -> CnnCoalescingScheduler<'_> {
        CnnCoalescingScheduler {
            predictor: CnnCoalescingBatchPredictor::new_with_shape(self, options, image_shape),
            responders: VecDeque::new(),
            is_shutdown: false,
        }
    }

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

    pub fn new_with_feature_channels_and_kernel_size(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        feature_channels: &[usize],
        kernel_height: usize,
        kernel_width: usize,
        learning_rate: f32,
    ) -> Result<Self, CnnImageClassifierError> {
        Self::new_with_feature_channels_and_input_channels_and_kernel_size(
            class_labels,
            input_height,
            input_width,
            1,
            feature_channels,
            kernel_height,
            kernel_width,
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
        Self::new_with_feature_channels_and_input_channels_and_kernel_size(
            class_labels,
            input_height,
            input_width,
            input_channels,
            feature_channels,
            3,
            3,
            learning_rate,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_feature_channels_and_input_channels_and_kernel_size(
        class_labels: Vec<String>,
        input_height: usize,
        input_width: usize,
        input_channels: usize,
        feature_channels: &[usize],
        kernel_height: usize,
        kernel_width: usize,
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

        if kernel_height == 0 || kernel_width == 0 {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "kernel height and width must be greater than zero",
            ));
        }

        if feature_channels.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "feature channel configuration must contain at least one layer",
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

        let mut feature_blocks = Vec::with_capacity(feature_channels.len());
        let mut in_channels = input_channels;
        for (layer_idx, &out_channels) in feature_channels.iter().enumerate() {
            let kernels = initialize_conv_kernels(
                out_channels,
                in_channels,
                kernel_height,
                kernel_width,
                layer_idx == 0,
            )?;
            let bias = vec![0.0; out_channels];
            feature_blocks.push(ConvParameterState::new(kernels, bias));
            in_channels = out_channels;
        }

        let feature_dim = feature_channels[feature_channels.len() - 1];

        let head = LinearHead::new(feature_dim, normalized_labels.len(), learning_rate)?;

        Ok(Self {
            input_height,
            input_width,
            input_channels,
            class_labels: normalized_labels,
            label_to_index,
            feature_blocks,
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

    fn forward_feature_stack_with_cache_from_tensor(
        &self,
        input: &Tensor4D,
    ) -> Result<(Tensor4D, ForwardCache), CnnImageClassifierError> {

        let mut current = input.clone();
        let mut blocks = Vec::with_capacity(self.feature_blocks.len());

        for block in &self.feature_blocks {
            let (next, cache) = forward_feature_block_with_cache(&current, block)?;
            current = next;
            blocks.push(cache);
        }

        Ok((current, ForwardCache { blocks }))
    }

    fn forward_feature_stack_with_cache_from_tensor_timed(
        &self,
        input: &Tensor4D,
    ) -> Result<(Tensor4D, ForwardCache, CnnForwardStageMetrics), CnnImageClassifierError> {

        let mut current = input.clone();
        let mut blocks = Vec::with_capacity(self.feature_blocks.len());
        let mut metrics = CnnForwardStageMetrics::default();

        for block in &self.feature_blocks {
            let (next, cache, block_metrics) = forward_feature_block_with_cache_timed(&current, block)?;
            current = next;
            blocks.push(cache);
            metrics.conv_elapsed_ms += block_metrics.conv_elapsed_ms;
            metrics.pool_elapsed_ms += block_metrics.pool_elapsed_ms;
        }

        Ok((current, ForwardCache { blocks }, metrics))
    }

    fn forward_feature_stack_no_cache_from_tensor(
        &self,
        input: &Tensor4D,
    ) -> Result<Tensor4D, CnnImageClassifierError> {

        let mut current = input.clone();

        for block in &self.feature_blocks {
            current = forward_feature_block_no_cache(&current, block)?;
        }

        Ok(current)
    }

    fn sync_feature_blocks_backend_mirror(&mut self) {
        for block in &mut self.feature_blocks {
            block.sync_backend_mirror();
        }
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
        self.forward_feature_stack_no_cache_from_tensor(&input)?
            .global_average_pool2d()
            .map(|global| global.first_sample_features())
            .map_err(CnnImageClassifierError::Tensor)
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

        self.forward_with_cache_from_tensor(&input)
    }

    fn forward_with_cache_timed(
        &self,
        image_bytes: &[u8],
    ) -> Result<(Vec<f32>, ForwardCache, CnnForwardStageMetrics), CnnImageClassifierError> {
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

        self.forward_with_cache_from_tensor_timed(&input)
    }

    pub fn extract_features_with_dimensions(
        &self,
        image_bytes: &[u8],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Vec<f32>, CnnImageClassifierError> {
        if channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: channels,
            });
        }

        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            height,
            width,
            channels,
            self.input_height,
            self.input_width,
            true,
        )?;
        self.forward_feature_stack_no_cache_from_tensor(&input)?
            .global_average_pool2d()
            .map(|global| global.first_sample_features())
            .map_err(CnnImageClassifierError::Tensor)
    }

    fn forward_with_cache_from_tensor(
        &self,
        input: &Tensor4D,
    ) -> Result<(Vec<f32>, ForwardCache), CnnImageClassifierError> {

        let (n, c, h, w) = input.shape();
        if n != 1
            || c != self.input_channels
            || h != self.input_height
            || w != self.input_width
        {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "forward input tensor shape mismatch",
            ));
        }

        let (final_output, cache) = self.forward_feature_stack_with_cache_from_tensor(input)?;
        let global = final_output.global_average_pool2d()?;
        let first = global.first_sample_features();

        Ok((
            first,
            cache,
        ))
    }

    fn forward_with_cache_from_tensor_timed(
        &self,
        input: &Tensor4D,
    ) -> Result<(Vec<f32>, ForwardCache, CnnForwardStageMetrics), CnnImageClassifierError> {

        let (n, c, h, w) = input.shape();
        if n != 1
            || c != self.input_channels
            || h != self.input_height
            || w != self.input_width
        {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "forward input tensor shape mismatch",
            ));
        }

        let (final_output, cache, mut metrics) = self.forward_feature_stack_with_cache_from_tensor_timed(input)?;
        let global_start = Instant::now();
        let global = final_output.global_average_pool2d()?;
        metrics.global_pool_elapsed_ms = global_start.elapsed().as_secs_f64() * 1000.0;
        let first = global.first_sample_features();

        Ok((
            first,
            cache,
            metrics,
        ))
    }

    fn forward_batch_with_cache_from_tensor(
        &self,
        input: &Tensor4D,
    ) -> Result<(Vec<Vec<f32>>, BatchForwardCache), CnnImageClassifierError> {
        let (n, c, h, w) = input.shape();
        if n == 0
            || c != self.input_channels
            || h != self.input_height
            || w != self.input_width
        {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "forward batch input tensor shape mismatch",
            ));
        }

        let (final_output, cache) = self.forward_feature_stack_with_cache_from_tensor(input)?;

        let global = final_output.global_average_pool2d()?;
        let feature_batch = global.flatten_batch_features();

        Ok((
            feature_batch,
            BatchForwardCache { blocks: cache.blocks },
        ))
        
    }

    fn forward_batch_with_cache_from_tensor_timed(
        &self,
        input: &Tensor4D,
    ) -> Result<(Vec<Vec<f32>>, BatchForwardCache, CnnForwardStageMetrics), CnnImageClassifierError> {
        let (n, c, h, w) = input.shape();
        if n == 0
            || c != self.input_channels
            || h != self.input_height
            || w != self.input_width
        {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "forward batch input tensor shape mismatch",
            ));
        }

        let (final_output, cache, mut metrics) = self.forward_feature_stack_with_cache_from_tensor_timed(input)?;

        let global_start = Instant::now();
        let global = final_output.global_average_pool2d()?;
        metrics.global_pool_elapsed_ms = global_start.elapsed().as_secs_f64() * 1000.0;
        let feature_batch = global.flatten_batch_features();

        Ok((
            feature_batch,
            BatchForwardCache { blocks: cache.blocks },
            metrics,
        ))

    }

    fn forward_batch_features_from_tensor(
        &self,
        input: &Tensor4D,
    ) -> Result<Vec<Vec<f32>>, CnnImageClassifierError> {
        let (n, c, h, w) = input.shape();
        if n == 0
            || c != self.input_channels
            || h != self.input_height
            || w != self.input_width
        {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "forward batch input tensor shape mismatch",
            ));
        }

        let final_output = self.forward_feature_stack_no_cache_from_tensor(input)?;

        let global = final_output.global_average_pool2d()?;
        Ok(global.flatten_batch_features())
    }

    pub fn train_image(&mut self, label: &str, image_bytes: &[u8]) -> Result<f32, CnnImageClassifierError> {
        self
            .train_image_timed(label, image_bytes)
            .map(|(loss, _timing)| loss)

    }

    pub fn train_image_timed(
        &mut self,
        label: &str,
        image_bytes: &[u8],
    ) -> Result<(f32, CnnTrainTimingReport), CnnImageClassifierError> {
        let total_start = Instant::now();
        let normalized = label.trim().to_ascii_lowercase();
        let class_index = match self.label_to_index.get(&normalized).copied() {
            Some(index) => index,
            None => return Err(CnnImageClassifierError::UnknownLabel(normalized)),
        };

        let feature_start = Instant::now();
        let (features, cache, forward_stage_metrics) = self.forward_with_cache_timed(image_bytes)?;
        let feature_elapsed_ms = feature_start.elapsed().as_secs_f64() * 1000.0;

        let head_start = Instant::now();
        let (loss, feature_grad) = self
            .head
            .train_step_with_input_gradient(features.as_slice(), class_index)?;
        let head_elapsed_ms = head_start.elapsed().as_secs_f64() * 1000.0;

        let backprop_start = Instant::now();
        let backprop_stage_metrics = self.backward_feature_extractor_with_scale_timed(
            &cache,
            feature_grad.as_slice(),
            1.0,
        )?;
        let backprop_elapsed_ms = backprop_start.elapsed().as_secs_f64() * 1000.0;

        let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            loss,
            CnnTrainTimingReport {
                input_elapsed_ms: 0.0,
                feature_elapsed_ms,
                head_elapsed_ms,
                backprop_elapsed_ms,
                total_elapsed_ms,
                forward_stage_metrics,
                backprop_stage_metrics,
            },
        ))

    }

    pub fn train_image_with_dimensions(
        &mut self,
        label: &str,
        image_bytes: &[u8],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<f32, CnnImageClassifierError> {
        self
            .train_image_with_dimensions_timed(label, image_bytes, height, width, channels)
            .map(|(loss, _timing)| loss)
    }

    pub fn train_image_with_dimensions_timed(
        &mut self,
        label: &str,
        image_bytes: &[u8],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<(f32, CnnTrainTimingReport), CnnImageClassifierError> {
        let total_start = Instant::now();
        let normalized = label.trim().to_ascii_lowercase();
        let class_index = match self.label_to_index.get(&normalized).copied() {
            Some(index) => index,
            None => return Err(CnnImageClassifierError::UnknownLabel(normalized)),
        };

        if channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: channels,
            });
        }

        let input_start = Instant::now();
        let input = image_bytes_to_tensor_nchw_resized_with_channels(
            image_bytes,
            height,
            width,
            channels,
            self.input_height,
            self.input_width,
            true,
        )?;
        let input_elapsed_ms = input_start.elapsed().as_secs_f64() * 1000.0;

        let feature_start = Instant::now();
        let (features, cache, forward_stage_metrics) = self.forward_with_cache_from_tensor_timed(&input)?;
        let feature_elapsed_ms = feature_start.elapsed().as_secs_f64() * 1000.0;

        let head_start = Instant::now();
        let (loss, feature_grad) = self
            .head
            .train_step_with_input_gradient(features.as_slice(), class_index)?;
        let head_elapsed_ms = head_start.elapsed().as_secs_f64() * 1000.0;

        let backprop_start = Instant::now();
        let backprop_stage_metrics = self.backward_feature_extractor_with_scale_timed(
            &cache,
            feature_grad.as_slice(),
            1.0,
        )?;
        let backprop_elapsed_ms = backprop_start.elapsed().as_secs_f64() * 1000.0;

        let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            loss,
            CnnTrainTimingReport {
                input_elapsed_ms,
                feature_elapsed_ms,
                head_elapsed_ms,
                backprop_elapsed_ms,
                total_elapsed_ms,
                forward_stage_metrics,
                backprop_stage_metrics,
            },
        ))
    }

    pub fn train_image_batch(
        &mut self,
        label: &str,
        images: &[Vec<u8>],
    ) -> Result<f32, CnnImageClassifierError> {
        self
            .train_image_batch_timed(label, images)
            .map(|(loss, _timing)| loss)

    }

    pub fn train_image_batch_timed(
        &mut self,
        label: &str,
        images: &[Vec<u8>],
    ) -> Result<(f32, CnnTrainTimingReport), CnnImageClassifierError> {
        let total_start = Instant::now();

        if images.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "cannot train on an empty image batch",
            ));
        }

        let normalized = label.trim().to_ascii_lowercase();
        let class_index = match self.label_to_index.get(&normalized).copied() {
            Some(index) => index,
            None => return Err(CnnImageClassifierError::UnknownLabel(normalized)),
        };

        let mut feature_batch: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        let mut caches: Vec<ForwardCache> = Vec::with_capacity(images.len());
        let mut batch_cache: Option<BatchForwardCache> = None;
        let mut forward_stage_metrics = CnnForwardStageMetrics::default();
        let mut backprop_stage_metrics = CnnBackpropStageMetrics::default();

        let common_shape = images
            .first()
            .and_then(|first| infer_square_dimensions_and_channels(first.as_slice()))
            .filter(|(_, _, channels)| *channels == self.input_channels)
            .and_then(|(in_h, in_w, in_c)| {
                let all_match = images.iter().all(|image| {
                    infer_square_dimensions_and_channels(image.as_slice())
                        .is_some_and(|(h, w, c)| h == in_h && w == in_w && c == in_c)
                });
                if all_match {
                    Some((in_h, in_w, in_c))
                } else {
                    None
                }
            });

        let mut input_elapsed_ms = 0.0f64;
        let feature_start = Instant::now();
        if cnn_batch_preprocess_enabled() {
            if let Some((in_h, in_w, in_c)) = common_shape {
                let input_start = Instant::now();
                let batch_inputs = image_batch_to_tensor_nchw_resized_with_channels(
                    images,
                    in_h,
                    in_w,
                    in_c,
                    self.input_height,
                    self.input_width,
                    true,
                )?;
                input_elapsed_ms += input_start.elapsed().as_secs_f64() * 1000.0;

                let (features, cache_item, metrics) = self.forward_batch_with_cache_from_tensor_timed(&batch_inputs)?;
                feature_batch = features;
                batch_cache = Some(cache_item);
                forward_stage_metrics.conv_elapsed_ms += metrics.conv_elapsed_ms;
                forward_stage_metrics.pool_elapsed_ms += metrics.pool_elapsed_ms;
                forward_stage_metrics.global_pool_elapsed_ms += metrics.global_pool_elapsed_ms;
            } else {
                for image in images {
                    let (features, cache, metrics) = self.forward_with_cache_timed(image.as_slice())?;
                    feature_batch.push(features);
                    caches.push(cache);
                    forward_stage_metrics.conv_elapsed_ms += metrics.conv_elapsed_ms;
                    forward_stage_metrics.pool_elapsed_ms += metrics.pool_elapsed_ms;
                    forward_stage_metrics.global_pool_elapsed_ms += metrics.global_pool_elapsed_ms;
                }
            }
        } else {
            for image in images {
                let (features, cache, metrics) = self.forward_with_cache_timed(image.as_slice())?;
                feature_batch.push(features);
                caches.push(cache);
                forward_stage_metrics.conv_elapsed_ms += metrics.conv_elapsed_ms;
                forward_stage_metrics.pool_elapsed_ms += metrics.pool_elapsed_ms;
                forward_stage_metrics.global_pool_elapsed_ms += metrics.global_pool_elapsed_ms;
            }
        }
        let feature_elapsed_ms = feature_start.elapsed().as_secs_f64() * 1000.0;

        let head_start = Instant::now();
        let targets = vec![class_index; images.len()];
        let (loss, feature_grads) = self
            .head
            .train_batch_with_input_gradients(feature_batch.as_slice(), targets.as_slice())?;
        let head_elapsed_ms = head_start.elapsed().as_secs_f64() * 1000.0;

        let batch_size = images.len() as f32;

        for block in &mut self.feature_blocks {
            block.reset_accumulated_gradients();
        }

        let backprop_start = Instant::now();
        if let Some(batch_cache) = batch_cache.as_ref() {
            let metrics = self.backward_feature_extractor_batch_with_scale_timed(
                batch_cache,
                feature_grads.as_slice(),
                batch_size,
                1.0,
            )?;
            backprop_stage_metrics.pooled_grad_elapsed_ms += metrics.pooled_grad_elapsed_ms;
            backprop_stage_metrics.gradients_elapsed_ms += metrics.gradients_elapsed_ms;
            backprop_stage_metrics.apply_update_elapsed_ms += metrics.apply_update_elapsed_ms;
            backprop_stage_metrics.input_grad_transfer_elapsed_ms += metrics.input_grad_transfer_elapsed_ms;
            backprop_stage_metrics.unpool_relu_elapsed_ms += metrics.unpool_relu_elapsed_ms;
            backprop_stage_metrics.dw_elapsed_ms += metrics.dw_elapsed_ms;
            backprop_stage_metrics.dinput_elapsed_ms += metrics.dinput_elapsed_ms;
            backprop_stage_metrics.bias_grad_elapsed_ms += metrics.bias_grad_elapsed_ms;
        } else {
            for (cache, feature_grad) in caches.iter().zip(feature_grads.iter()) {
                let metrics = self.backward_feature_extractor_with_scale_timed(
                    cache,
                    feature_grad.as_slice(),
                    1.0 / batch_size,
                )?;
                backprop_stage_metrics.pooled_grad_elapsed_ms += metrics.pooled_grad_elapsed_ms;
                backprop_stage_metrics.gradients_elapsed_ms += metrics.gradients_elapsed_ms;
                backprop_stage_metrics.apply_update_elapsed_ms += metrics.apply_update_elapsed_ms;
                backprop_stage_metrics.input_grad_transfer_elapsed_ms += metrics.input_grad_transfer_elapsed_ms;
                backprop_stage_metrics.unpool_relu_elapsed_ms += metrics.unpool_relu_elapsed_ms;
                backprop_stage_metrics.dw_elapsed_ms += metrics.dw_elapsed_ms;
                backprop_stage_metrics.dinput_elapsed_ms += metrics.dinput_elapsed_ms;
                backprop_stage_metrics.bias_grad_elapsed_ms += metrics.bias_grad_elapsed_ms;
            }
        }
        let backprop_elapsed_ms = backprop_start.elapsed().as_secs_f64() * 1000.0;

        let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            loss,
            CnnTrainTimingReport {
                input_elapsed_ms,
                feature_elapsed_ms,
                head_elapsed_ms,
                backprop_elapsed_ms,
                total_elapsed_ms,
                forward_stage_metrics,
                backprop_stage_metrics,
            },
        ))

    }

    pub fn train_image_batch_with_dimensions(
        &mut self,
        label: &str,
        images: &[Vec<u8>],
        shape: ImageTensorShape,
    ) -> Result<f32, CnnImageClassifierError> {
        self
            .train_image_batch_with_dimensions_timed(label, images, shape)
            .map(|(loss, _timing)| loss)
    }

    pub fn train_image_batch_with_dimensions_timed(
        &mut self,
        label: &str,
        images: &[Vec<u8>],
        shape: ImageTensorShape,
    ) -> Result<(f32, CnnTrainTimingReport), CnnImageClassifierError> {
        let total_start = Instant::now();

        if images.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "cannot train on an empty image batch",
            ));
        }

        let normalized = label.trim().to_ascii_lowercase();
        let class_index = match self.label_to_index.get(&normalized).copied() {
            Some(index) => index,
            None => return Err(CnnImageClassifierError::UnknownLabel(normalized)),
        };

        if shape.channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: shape.channels,
            });
        }

        let input_start = Instant::now();
        let batch_inputs = image_batch_to_tensor_nchw_resized_with_channels(
            images,
            shape.height,
            shape.width,
            shape.channels,
            self.input_height,
            self.input_width,
            true,
        )?;
        let input_elapsed_ms = input_start.elapsed().as_secs_f64() * 1000.0;

        let feature_start = Instant::now();
        let (feature_batch, batch_cache, forward_stage_metrics) = self.forward_batch_with_cache_from_tensor_timed(&batch_inputs)?;
        let feature_elapsed_ms = feature_start.elapsed().as_secs_f64() * 1000.0;

        let head_start = Instant::now();
        let targets = vec![class_index; images.len()];
        let (loss, feature_grads) = self
            .head
            .train_batch_with_input_gradients(feature_batch.as_slice(), targets.as_slice())?;
        let head_elapsed_ms = head_start.elapsed().as_secs_f64() * 1000.0;

        let batch_size = images.len() as f32;

        for block in &mut self.feature_blocks {
            block.reset_accumulated_gradients();
        }

        let backprop_start = Instant::now();
        let backprop_stage_metrics = self.backward_feature_extractor_batch_with_scale_timed(
            &batch_cache,
            feature_grads.as_slice(),
            batch_size,
            1.0,
        )?;
        let backprop_elapsed_ms = backprop_start.elapsed().as_secs_f64() * 1000.0;

        let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            loss,
            CnnTrainTimingReport {
                input_elapsed_ms,
                feature_elapsed_ms,
                head_elapsed_ms,
                backprop_elapsed_ms,
                total_elapsed_ms,
                forward_stage_metrics,
                backprop_stage_metrics,
            },
        ))
    }

    pub fn predict_with_confidence(
        &self,
        image_bytes: &[u8],
    ) -> Result<Option<(String, f32)>, CnnImageClassifierError> {
        let features = self.extract_features(image_bytes)?;
        self.classify_feature_vector(features.as_slice())
    }

    pub fn predict_with_confidence_with_dimensions(
        &self,
        image_bytes: &[u8],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Option<(String, f32)>, CnnImageClassifierError> {
        let features = self.extract_features_with_dimensions(image_bytes, height, width, channels)?;
        self.classify_feature_vector(features.as_slice())
    }

    pub fn predict_batch_with_confidence(
        &self,
        images: &[Vec<u8>],
    ) -> Result<Vec<Option<(String, f32)>>, CnnImageClassifierError> {
        let report = self.predict_batch_with_confidence_report(images, CnnBatchPredictOptions::default())?;
        Ok(report.predictions)
    }

    pub fn predict_batch_with_confidence_report(
        &self,
        images: &[Vec<u8>],
        options: CnnBatchPredictOptions,
    ) -> Result<CnnBatchPredictReport, CnnImageClassifierError> {
        if images.is_empty() {
            return Ok(CnnBatchPredictReport {
                predictions: Vec::new(),
                total_images: 0,
                micro_batch_count: 0,
                max_micro_batch_size: options.max_micro_batch_size.max(1),
                preprocessing_elapsed_ms: 0.0,
                model_elapsed_ms: 0.0,
                total_elapsed_ms: 0.0,
                throughput_images_per_sec: 0.0,
            });
        }

        let total_start = Instant::now();
        let max_micro_batch_size = options.max_micro_batch_size.max(1);
        let mut predictions = Vec::with_capacity(images.len());
        let mut preprocess_elapsed_sec = 0.0f64;
        let mut model_elapsed_sec = 0.0f64;
        let mut micro_batch_count = 0usize;

        for chunk in images.chunks(max_micro_batch_size) {
            micro_batch_count += 1;

            let common_shape = chunk
                .first()
                .and_then(|first| infer_square_dimensions_and_channels(first.as_slice()))
                .filter(|(_, _, channels)| *channels == self.input_channels)
                .and_then(|(in_h, in_w, in_c)| {
                    let all_match = chunk.iter().all(|image| {
                        infer_square_dimensions_and_channels(image.as_slice())
                            .is_some_and(|(h, w, c)| h == in_h && w == in_w && c == in_c)
                    });
                    if all_match {
                        Some((in_h, in_w, in_c))
                    } else {
                        None
                    }
                });

            if options.enable_batch_preprocess
                && let Some((in_h, in_w, in_c)) = common_shape
            {
                let preprocess_start = Instant::now();
                let batch_inputs = image_batch_to_tensor_nchw_resized_with_channels(
                    chunk,
                    in_h,
                    in_w,
                    in_c,
                    self.input_height,
                    self.input_width,
                    true,
                )?;
                preprocess_elapsed_sec += preprocess_start.elapsed().as_secs_f64();

                let model_start = Instant::now();
                let feature_batch = self.forward_batch_features_from_tensor(&batch_inputs)?;
                for features in feature_batch {
                    predictions.push(self.classify_feature_vector(features.as_slice())?);
                }
                model_elapsed_sec += model_start.elapsed().as_secs_f64();
                continue;
            }

            for image in chunk {
                let model_start = Instant::now();
                let features = self.forward_for_inference(image.as_slice())?;
                predictions.push(self.classify_feature_vector(features.as_slice())?);
                model_elapsed_sec += model_start.elapsed().as_secs_f64();
            }
        }

        let total_elapsed_sec = total_start.elapsed().as_secs_f64();
        Ok(CnnBatchPredictReport {
            predictions,
            total_images: images.len(),
            micro_batch_count,
            max_micro_batch_size,
            preprocessing_elapsed_ms: preprocess_elapsed_sec * 1_000.0,
            model_elapsed_ms: model_elapsed_sec * 1_000.0,
            total_elapsed_ms: total_elapsed_sec * 1_000.0,
            throughput_images_per_sec: images.len() as f64 / total_elapsed_sec.max(1e-9),
        })
    }

    pub fn predict_batch_with_confidence_report_with_dimensions(
        &self,
        images: &[Vec<u8>],
        height: usize,
        width: usize,
        channels: usize,
        options: CnnBatchPredictOptions,
    ) -> Result<CnnBatchPredictReport, CnnImageClassifierError> {
        if images.is_empty() {
            return Ok(CnnBatchPredictReport {
                predictions: Vec::new(),
                total_images: 0,
                micro_batch_count: 0,
                max_micro_batch_size: options.max_micro_batch_size.max(1),
                preprocessing_elapsed_ms: 0.0,
                model_elapsed_ms: 0.0,
                total_elapsed_ms: 0.0,
                throughput_images_per_sec: 0.0,
            });
        }

        if channels != self.input_channels {
            return Err(CnnImageClassifierError::InputChannelMismatch {
                expected: self.input_channels,
                actual: channels,
            });
        }

        let total_start = Instant::now();
        let max_micro_batch_size = options.max_micro_batch_size.max(1);
        let mut predictions = Vec::with_capacity(images.len());
        let mut preprocess_elapsed_sec = 0.0f64;
        let mut model_elapsed_sec = 0.0f64;
        let mut micro_batch_count = 0usize;

        for chunk in images.chunks(max_micro_batch_size) {
            micro_batch_count += 1;

            if options.enable_batch_preprocess {
                let preprocess_start = Instant::now();
                let batch_inputs = image_batch_to_tensor_nchw_resized_with_channels(
                    chunk,
                    height,
                    width,
                    channels,
                    self.input_height,
                    self.input_width,
                    true,
                )?;
                preprocess_elapsed_sec += preprocess_start.elapsed().as_secs_f64();

                let model_start = Instant::now();
                let feature_batch = self.forward_batch_features_from_tensor(&batch_inputs)?;
                for features in feature_batch {
                    predictions.push(self.classify_feature_vector(features.as_slice())?);
                }
                model_elapsed_sec += model_start.elapsed().as_secs_f64();
                continue;
            }

            for image in chunk {
                let model_start = Instant::now();
                let features = self.extract_features_with_dimensions(image.as_slice(), height, width, channels)?;
                predictions.push(self.classify_feature_vector(features.as_slice())?);
                model_elapsed_sec += model_start.elapsed().as_secs_f64();
            }
        }

        let total_elapsed_sec = total_start.elapsed().as_secs_f64();
        Ok(CnnBatchPredictReport {
            predictions,
            total_images: images.len(),
            micro_batch_count,
            max_micro_batch_size,
            preprocessing_elapsed_ms: preprocess_elapsed_sec * 1_000.0,
            model_elapsed_ms: model_elapsed_sec * 1_000.0,
            total_elapsed_ms: total_elapsed_sec * 1_000.0,
            throughput_images_per_sec: images.len() as f64 / total_elapsed_sec.max(1e-9),
        })
    }

    pub fn predict_batch_with_confidence_with_dimensions(
        &self,
        images: &[Vec<u8>],
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Vec<Option<(String, f32)>>, CnnImageClassifierError> {
        let report = self.predict_batch_with_confidence_report_with_dimensions(
            images,
            height,
            width,
            channels,
            CnnBatchPredictOptions::default(),
        )?;
        Ok(report.predictions)
    }

    fn classify_feature_vector(
        &self,
        features: &[f32],
    ) -> Result<Option<(String, f32)>, CnnImageClassifierError> {
        let probs = self.head.probabilities(features)?;

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

        let mut feature_blocks = self.feature_blocks.clone();
        for block in feature_blocks.iter_mut() {
            block.refresh_host_from_backend();
        }

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
            feature_block_kernels: feature_blocks.iter().map(|state| state.snapshot().0).collect(),
            feature_block_biases: feature_blocks.iter().map(|state| state.snapshot().1).collect(),
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

        if snapshot.feature_block_kernels.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (feature block stack is empty)",
                    path.display()
                ),
            ));
        }

        if snapshot.feature_block_kernels.len() != snapshot.feature_block_biases.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (feature block kernels/biases length mismatch)",
                    path.display()
                ),
            ));
        }

        for (index, (kernels, bias)) in snapshot
            .feature_block_kernels
            .iter()
            .zip(snapshot.feature_block_biases.iter())
            .enumerate()
        {
            if kernels.shape().0 != bias.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid CNN classifier snapshot '{}' (feature block {} channels and bias mismatch)",
                        path.display(),
                        index,
                    ),
                ));
            }
        }

        if snapshot.feature_block_kernels[0].shape().1 != snapshot.input_channels {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid CNN classifier snapshot '{}' (first feature block input channels and model input channels mismatch)",
                    path.display()
                ),
            ));
        }

        let (head_input_dim, head_output_dim) = snapshot.head.dimensions();
        let feature_dim = snapshot
            .feature_block_kernels
            .last()
            .map(|kernels| kernels.shape().0)
            .unwrap_or(0);

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
            feature_blocks: snapshot
                .feature_block_kernels
                .into_iter()
                .zip(snapshot.feature_block_biases)
                .map(|(kernels, bias)| ConvParameterState::new(kernels, bias))
                .collect(),
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
        self
            .backward_feature_extractor_with_scale_timed(cache, feature_grad, 1.0)
            .map(|_metrics| ())
    }

    fn backward_feature_extractor_with_scale(
        &mut self,
        cache: &ForwardCache,
        feature_grad: &[f32],
        grad_scale: f32,
    ) -> Result<(), CnnImageClassifierError> {
        self
            .backward_feature_extractor_with_scale_timed(cache, feature_grad, grad_scale)
            .map(|_metrics| ())
    }

    fn backward_feature_extractor_with_scale_timed(
        &mut self,
        cache: &ForwardCache,
        feature_grad: &[f32],
        grad_scale: f32,
    ) -> Result<CnnBackpropStageMetrics, CnnImageClassifierError> {

        if cache.blocks.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "feature block cache stack is empty",
            ));
        }

        let mut metrics = CnnBackpropStageMetrics::default();

        let effective_learning_rate = self.feature_learning_rate * grad_scale.max(0.0);
        let pooled_grad_start = Instant::now();
        let mut pooled_grad = pooled_grad_from_feature_gradient(
            cache.blocks.last().unwrap().pooled_shape,
            feature_grad,
        )?;
        metrics.pooled_grad_elapsed_ms = pooled_grad_start.elapsed().as_secs_f64() * 1000.0;

        for block_index in (0..self.feature_blocks.len()).rev() {
            let block_cache = cache.blocks.get(block_index).ok_or(
                CnnImageClassifierError::InvalidConfiguration(
                    "feature block cache missing during backprop",
                ),
            )?;
            let compute_input_grad = block_index > 0;
            let block = self.feature_blocks.get_mut(block_index).ok_or(
                CnnImageClassifierError::InvalidConfiguration(
                    "feature block parameters missing during backprop",
                ),
            )?;
            let (kernels, bias) = block.parameter_views_mut();
            let grad_start = Instant::now();
            let backward = backward_conv_block_gradients(
                kernels,
                block_cache,
                &pooled_grad,
                compute_input_grad,
            )?;
            metrics.gradients_elapsed_ms += grad_start.elapsed().as_secs_f64() * 1000.0;

            let update_start = Instant::now();
            apply_conv_gradients_single(
                kernels,
                bias,
                &backward.kernel_grad,
                backward.bias_grad.as_slice(),
                effective_learning_rate,
                1.0,
            )?;
            metrics.apply_update_elapsed_ms += update_start.elapsed().as_secs_f64() * 1000.0;

            if compute_input_grad {
                let transfer_start = Instant::now();
                let next_grad = backward.input_grad.as_ref().ok_or(
                    CnnImageClassifierError::InvalidConfiguration(
                        "feature block gradient is missing pooled input gradient",
                    ),
                )?;
                let expected_shape = cache.blocks[block_index - 1].pooled_shape;
                if next_grad.shape() != expected_shape {
                    return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
                        expected: expected_shape,
                        actual: next_grad.shape(),
                    });
                }
                pooled_grad = next_grad.clone();
                metrics.input_grad_transfer_elapsed_ms +=
                    transfer_start.elapsed().as_secs_f64() * 1000.0;
            }
        }

        self.sync_feature_blocks_backend_mirror();
        Ok(metrics)

    }

    fn backward_feature_extractor_batch_with_scale(
        &mut self,
        cache: &BatchForwardCache,
        feature_grads: &[Vec<f32>],
        batch_size: f32,
        grad_scale: f32,
    ) -> Result<(), CnnImageClassifierError> {
        self
            .backward_feature_extractor_batch_with_scale_timed(cache, feature_grads, batch_size, grad_scale)
            .map(|_metrics| ())
    }

    fn backward_feature_extractor_batch_with_scale_timed(
        &mut self,
        cache: &BatchForwardCache,
        feature_grads: &[Vec<f32>],
        batch_size: f32,
        grad_scale: f32,
    ) -> Result<CnnBackpropStageMetrics, CnnImageClassifierError> {

        if cache.blocks.is_empty() {
            return Err(CnnImageClassifierError::InvalidConfiguration(
                "feature block cache stack is empty",
            ));
        }

        let mut metrics = CnnBackpropStageMetrics::default();

        let effective_learning_rate = self.feature_learning_rate * grad_scale.max(0.0);
        let pooled_grad_start = Instant::now();
        let mut pooled_grad = pooled_grad_batch_from_feature_gradients(
            cache.blocks.last().unwrap().pooled_shape,
            feature_grads,
        )?;
        metrics.pooled_grad_elapsed_ms = pooled_grad_start.elapsed().as_secs_f64() * 1000.0;

        for block_index in (0..self.feature_blocks.len()).rev() {
            let block_cache = cache.blocks.get(block_index).ok_or(
                CnnImageClassifierError::InvalidConfiguration(
                    "feature block cache missing during batch backprop",
                ),
            )?;
            let compute_input_grad = block_index > 0;
            let block = self.feature_blocks.get_mut(block_index).ok_or(
                CnnImageClassifierError::InvalidConfiguration(
                    "feature block parameters missing during batch backprop",
                ),
            )?;
            let (kernels, _bias) = block.parameter_views_mut();
            let grad_start = Instant::now();
            let backward = backward_conv_block_gradients(
                kernels,
                block_cache,
                &pooled_grad,
                compute_input_grad,
            )?;
            metrics.gradients_elapsed_ms += grad_start.elapsed().as_secs_f64() * 1000.0;

            let update_start = Instant::now();
            block.accumulate_gradients(
                &backward.kernel_grad,
                backward.bias_grad.as_slice(),
            )?;
            metrics.apply_update_elapsed_ms += update_start.elapsed().as_secs_f64() * 1000.0;

            if compute_input_grad {
                let transfer_start = Instant::now();
                let next_grad = backward.input_grad.as_ref().ok_or(
                    CnnImageClassifierError::InvalidConfiguration(
                        "feature block gradient is missing pooled input gradient",
                    ),
                )?;
                let expected_shape = cache.blocks[block_index - 1].pooled_shape;
                if next_grad.shape() != expected_shape {
                    return Err(CnnImageClassifierError::GradientTensorShapeMismatch {
                        expected: expected_shape,
                        actual: next_grad.shape(),
                    });
                }
                pooled_grad = next_grad.clone();
                metrics.input_grad_transfer_elapsed_ms +=
                    transfer_start.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let apply_start = Instant::now();
        for block in &mut self.feature_blocks {
            block.apply_sgd_update(effective_learning_rate, batch_size)?;
        }
        metrics.apply_update_elapsed_ms += apply_start.elapsed().as_secs_f64() * 1000.0;

        self.sync_feature_blocks_backend_mirror();
        Ok(metrics)

    }

}

fn default_input_channels() -> usize {
    1
}

fn initialize_conv_kernels(
    out_channels: usize,
    in_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    oriented_first_layer: bool,
) -> Result<Tensor4D, CnnImageClassifierError> {

    let mut values = Vec::with_capacity(out_channels * in_channels * kernel_height * kernel_width);

    let center_y = kernel_height / 2;
    let center_x = kernel_width / 2;

    for out_c in 0..out_channels {
        for in_c in 0..in_channels {
            let kernel = if oriented_first_layer && in_channels == 1 && kernel_height == 3 && kernel_width == 3 && out_c == 0 {
                vec![
                    -1.0, 0.0, 1.0,
                    -1.0, 0.0, 1.0,
                    -1.0, 0.0, 1.0,
                ]
            } else if oriented_first_layer && in_channels == 1 && kernel_height == 3 && kernel_width == 3 && out_c == 1 {
                vec![
                    -1.0, -1.0, -1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0,
                ]
            } else {
                let center_weight = 1.0f32 / in_channels as f32;
                let sign = if (out_c + in_c) % 2 == 0 { 1.0 } else { -1.0 };
                let mut kernel = vec![0.0; kernel_height * kernel_width];
                kernel[center_y * kernel_width + center_x] = sign * center_weight;
                kernel
            };
            values.extend(kernel);
        }
    }

    Tensor4D::from_vec(out_channels, in_channels, kernel_height, kernel_width, values)
        .map_err(CnnImageClassifierError::Tensor)

}

fn forward_feature_block_with_cache(
    input: &Tensor4D,
    block: &ConvParameterState,
) -> Result<(Tensor4D, ConvBlockCache), CnnImageClassifierError> {

    #[cfg(feature = "offloading-mlx")]
    if active_backend().name() == "mlx"
        && let Some((kernels, kernels_shape, bias)) = block.mlx_mirror_views()
    {
        return forward_conv_block_with_mlx_mirror(input, kernels, kernels_shape, bias);
    }

    forward_conv_block(input, block.kernels(), block.bias())
}

fn forward_feature_block_with_cache_timed(
    input: &Tensor4D,
    block: &ConvParameterState,
) -> Result<(Tensor4D, ConvBlockCache, CnnForwardStageMetrics), CnnImageClassifierError> {

    #[cfg(feature = "offloading-mlx")]
    if active_backend().name() == "mlx"
        && let Some((kernels, kernels_shape, bias)) = block.mlx_mirror_views()
    {
        return forward_conv_block_with_mlx_mirror_timed(input, kernels, kernels_shape, bias);
    }

    forward_conv_block_timed(input, block.kernels(), block.bias())
}

fn forward_feature_block_no_cache(
    input: &Tensor4D,
    block: &ConvParameterState,
) -> Result<Tensor4D, CnnImageClassifierError> {

    #[cfg(feature = "offloading-mlx")]
    if active_backend().name() == "mlx"
        && let Some((kernels, kernels_shape, bias)) = block.mlx_mirror_views()
    {
        return forward_conv_block_with_mlx_mirror_no_cache(input, kernels, kernels_shape, bias);
    }

    forward_conv_block_no_cache(input, block.kernels(), block.bias())
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
            pool_indices,
            pooled_shape,
        },
    ))

}

fn forward_conv_block_timed(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: &[f32],
) -> Result<(Tensor4D, ConvBlockCache, CnnForwardStageMetrics), CnnImageClassifierError> {

    let conv_start = Instant::now();
    let conv_pre = input.conv2d_valid(kernels, Some(bias), 1, 1)?;
    let conv_elapsed_ms = conv_start.elapsed().as_secs_f64() * 1000.0;

    let mut relu = conv_pre.clone();
    relu.relu_inplace();

    let pool_start = Instant::now();
    let (pooled, pool_indices) = max_pool2d_with_indices(&relu, 2, 2, 2, 2)?;
    let pool_elapsed_ms = pool_start.elapsed().as_secs_f64() * 1000.0;
    let pooled_shape = pooled.shape();

    Ok((
        pooled,
        ConvBlockCache {
            input: input.clone(),
            conv_pre_activation: conv_pre,
            pool_indices,
            pooled_shape,
        },
        CnnForwardStageMetrics {
            conv_elapsed_ms,
            pool_elapsed_ms,
            global_pool_elapsed_ms: 0.0,
        },
    ))

}

fn forward_conv_block_no_cache(
    input: &Tensor4D,
    kernels: &Tensor4D,
    bias: &[f32],
) -> Result<Tensor4D, CnnImageClassifierError> {
    let mut conv = input.conv2d_valid(kernels, Some(bias), 1, 1)?;
    conv.relu_inplace();
    let (pooled, _pool_indices) = max_pool2d_with_indices(&conv, 2, 2, 2, 2)?;
    Ok(pooled)
}

#[cfg(feature = "offloading-mlx")]
fn forward_conv_block_with_mlx_mirror(
    input: &Tensor4D,
    kernels: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
    kernels_shape: (usize, usize, usize, usize),
    bias: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
) -> Result<(Tensor4D, ConvBlockCache), CnnImageClassifierError> {

    let conv_pre = mlx_conv2d_valid_with_mirrored_params(input, kernels, kernels_shape, bias, 1, 1)?;
    let mut relu = conv_pre.clone();
    relu.relu_inplace();

    let (pooled, pool_indices) = max_pool2d_with_indices(&relu, 2, 2, 2, 2)?;
    let pooled_shape = pooled.shape();

    Ok((
        pooled,
        ConvBlockCache {
            input: input.clone(),
            conv_pre_activation: conv_pre,
            pool_indices,
            pooled_shape,
        },
    ))
}

#[cfg(feature = "offloading-mlx")]
fn forward_conv_block_with_mlx_mirror_timed(
    input: &Tensor4D,
    kernels: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
    kernels_shape: (usize, usize, usize, usize),
    bias: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
) -> Result<(Tensor4D, ConvBlockCache, CnnForwardStageMetrics), CnnImageClassifierError> {

    let conv_start = Instant::now();
    let conv_pre = mlx_conv2d_valid_with_mirrored_params(input, kernels, kernels_shape, bias, 1, 1)?;
    let conv_elapsed_ms = conv_start.elapsed().as_secs_f64() * 1000.0;

    let mut relu = conv_pre.clone();
    relu.relu_inplace();

    let pool_start = Instant::now();
    let (pooled, pool_indices) = max_pool2d_with_indices(&relu, 2, 2, 2, 2)?;
    let pool_elapsed_ms = pool_start.elapsed().as_secs_f64() * 1000.0;
    let pooled_shape = pooled.shape();

    Ok((
        pooled,
        ConvBlockCache {
            input: input.clone(),
            conv_pre_activation: conv_pre,
            pool_indices,
            pooled_shape,
        },
        CnnForwardStageMetrics {
            conv_elapsed_ms,
            pool_elapsed_ms,
            global_pool_elapsed_ms: 0.0,
        },
    ))
}

#[cfg(feature = "offloading-mlx")]
fn forward_conv_block_with_mlx_mirror_no_cache(
    input: &Tensor4D,
    kernels: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
    kernels_shape: (usize, usize, usize, usize),
    bias: &crate::tensor::offloading::mlx_backend::MlxOwnedArray,
) -> Result<Tensor4D, CnnImageClassifierError> {
    let mut conv = mlx_conv2d_valid_with_mirrored_params(input, kernels, kernels_shape, bias, 1, 1)?;
    conv.relu_inplace();
    let (pooled, _pool_indices) = max_pool2d_with_indices(&conv, 2, 2, 2, 2)?;
    Ok(pooled)
}

#[cfg(not(feature = "offloading-mlx"))]
fn forward_conv_block_with_mlx_mirror_no_cache(
    _input: &Tensor4D,
    _kernels: &(),
    _kernels_shape: (usize, usize, usize, usize),
    _bias: &(),
) -> Result<Tensor4D, CnnImageClassifierError> {
    Err(CnnImageClassifierError::InvalidConfiguration(
        "mlx mirror path requires offloading-mlx feature",
    ))
}

#[cfg(not(feature = "offloading-mlx"))]
fn forward_conv_block_with_mlx_mirror(
    _input: &Tensor4D,
    _kernels: &(),
    _kernels_shape: (usize, usize, usize, usize),
    _bias: &(),
) -> Result<(Tensor4D, ConvBlockCache), CnnImageClassifierError> {
    Err(CnnImageClassifierError::InvalidConfiguration(
        "mlx mirror path requires offloading-mlx feature",
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

fn tensor_shape_element_count(shape: (usize, usize, usize, usize)) -> usize {
    shape
        .0
        .saturating_mul(shape.1)
        .saturating_mul(shape.2)
        .saturating_mul(shape.3)
}

fn pooled_grad_batch_from_feature_gradients(
    pooled_shape: (usize, usize, usize, usize),
    feature_grads: &[Vec<f32>],
) -> Result<Tensor4D, CnnImageClassifierError> {

    let (batch, channels, pooled_h, pooled_w) = pooled_shape;

    if feature_grads.len() != batch {
        return Err(CnnImageClassifierError::GradientShapeMismatch {
            expected: batch,
            actual: feature_grads.len(),
        });
    }

    let mut pooled_grad = Tensor4D::zeros(batch, channels, pooled_h, pooled_w);
    let pooled_area = (pooled_h * pooled_w).max(1) as f32;
    let channel_stride = pooled_h * pooled_w;
    let sample_stride = channels * channel_stride;

    for (sample_idx, sample_grad) in feature_grads.iter().enumerate() {
        if sample_grad.len() != channels {
            return Err(CnnImageClassifierError::GradientShapeMismatch {
                expected: channels,
                actual: sample_grad.len(),
            });
        }

        let sample_base = sample_idx * sample_stride;
        for (channel, grad_value) in sample_grad.iter().enumerate() {
            let per_cell = *grad_value / pooled_area;
            let channel_base = sample_base + channel * channel_stride;
            for offset in 0..channel_stride {
                pooled_grad.as_mut_slice()[channel_base + offset] = per_cell;
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
) -> Result<ConvBlockBackwardGradients, CnnImageClassifierError> {

    active_backend()
        .conv_block_backward_gradients(
            kernels,
            &cache.input,
            &cache.conv_pre_activation,
            cache.pool_indices.as_slice(),
            cache.pooled_shape,
            pooled_grad,
            compute_input_grad,
        )
        .map_err(CnnImageClassifierError::Tensor)
        
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
    let mut indices = vec![(0usize, 0usize); n * c * out_h * out_w];

    let input_batch_stride = c * h * w;
    let input_channel_stride = h * w;
    let output_batch_stride = c * out_h * out_w;
    let output_channel_stride = out_h * out_w;

    for batch in 0..n {
        let input_batch_base = batch * input_batch_stride;
        let output_batch_base = batch * output_batch_stride;

        for channel in 0..c {
            let input_channel_base = input_batch_base + channel * input_channel_stride;
            let output_channel_base = output_batch_base + channel * output_channel_stride;

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
                    let idx = (((batch * c + channel) * out_h) + oy) * out_w + ox;
                    indices[idx] = max_idx;
                }
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
    fn cnn_classifier_supports_non_3x3_kernel_initialization() {
        let classifier = CnnImageClassifier::new_with_feature_channels_and_kernel_size(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            &[2],
            5,
            5,
            0.2,
        )
        .unwrap_or_else(|_| panic!("kernel-size aware classifier should initialize"));

        let image = vertical_stripes_image_8x8();
        let features = classifier
            .extract_features(image.as_slice())
            .unwrap_or_else(|_| panic!("feature extraction should succeed"));

        assert!(!features.is_empty());
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

    #[test]
    fn cnn_classifier_batch_prediction_matches_single_prediction() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
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

        let images = vec![cat.clone(), dog.clone(), cat.clone(), dog.clone()];
        let singles: Vec<Option<(String, f32)>> = images
            .iter()
            .map(|image| {
                classifier
                    .predict_with_confidence(image.as_slice())
                    .unwrap_or_else(|_| panic!("single prediction should succeed"))
            })
            .collect();

        let batch = classifier
            .predict_batch_with_confidence(images.as_slice())
            .unwrap_or_else(|_| panic!("batch prediction should succeed"));

        assert_eq!(batch, singles);
    }

    #[test]
    fn cnn_classifier_batch_prediction_report_has_expected_metrics() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..20 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let images = vec![cat.clone(), dog.clone(), cat, dog];
        let report = classifier
            .predict_batch_with_confidence_report(
                images.as_slice(),
                CnnBatchPredictOptions {
                    max_micro_batch_size: 2,
                    enable_batch_preprocess: true,
                },
            )
            .unwrap_or_else(|_| panic!("batch report prediction should succeed"));

        assert_eq!(report.total_images, 4);
        assert_eq!(report.max_micro_batch_size, 2);
        assert_eq!(report.micro_batch_count, 2);
        assert_eq!(report.predictions.len(), 4);
        assert!(report.total_elapsed_ms >= 0.0);
        assert!(report.throughput_images_per_sec >= 0.0);
    }

    #[test]
    fn cnn_classifier_coalescing_predictor_flushes_by_queue_size() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..20 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let mut predictor = classifier.coalescing_batch_predictor(CnnCoalescingPredictOptions {
            max_micro_batch_size: 2,
            max_queue_size: 2,
            max_queue_delay_ms: 50,
            enable_batch_preprocess: true,
        });

        predictor
            .enqueue(cat.clone())
            .unwrap_or_else(|_| panic!("enqueue should succeed"));
        assert_eq!(predictor.pending_len(), 1);

        predictor
            .enqueue(dog.clone())
            .unwrap_or_else(|_| panic!("enqueue should flush when queue is full"));
        assert_eq!(predictor.pending_len(), 0);

        let ready = predictor.take_ready();
        assert_eq!(ready.len(), 2);

        let singles = vec![
            classifier
                .predict_with_confidence(cat.as_slice())
                .unwrap_or_else(|_| panic!("single prediction should succeed")),
            classifier
                .predict_with_confidence(dog.as_slice())
                .unwrap_or_else(|_| panic!("single prediction should succeed")),
        ];
        assert_eq!(ready, singles);
    }

    #[test]
    fn cnn_classifier_coalescing_predictor_flushes_when_due() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..20 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let mut predictor = classifier.coalescing_batch_predictor(CnnCoalescingPredictOptions {
            max_micro_batch_size: 8,
            max_queue_size: 64,
            max_queue_delay_ms: 0,
            enable_batch_preprocess: true,
        });

        predictor
            .enqueue(cat)
            .unwrap_or_else(|_| panic!("enqueue should succeed"));
        predictor
            .enqueue(dog)
            .unwrap_or_else(|_| panic!("enqueue should succeed"));

        let due_flushed = predictor
            .flush_if_due()
            .unwrap_or_else(|_| panic!("flush-if-due should succeed"));
        assert!(due_flushed);

        let report = predictor
            .finish()
            .unwrap_or_else(|_| panic!("finish should succeed"));

        assert_eq!(report.total_images, 2);
        assert_eq!(report.micro_batch_count, 1);
        assert_eq!(report.predictions.len(), 2);
    }

    #[test]
    fn cnn_classifier_coalescing_scheduler_matches_single_prediction_order() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..20 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let images = vec![cat.clone(), dog.clone(), cat.clone(), dog.clone()];
        let expected: Vec<Option<(String, f32)>> = images
            .iter()
            .map(|image| {
                classifier
                    .predict_with_confidence(image.as_slice())
                    .unwrap_or_else(|_| panic!("single prediction should succeed"))
            })
            .collect();

        let mut scheduler = classifier.start_coalescing_scheduler(CnnCoalescingPredictOptions {
            max_micro_batch_size: 4,
            max_queue_size: 64,
            max_queue_delay_ms: 0,
            enable_batch_preprocess: true,
        });

        let handles: Vec<CnnCoalescedPredictionHandle> = images
            .into_iter()
            .map(|image| {
                scheduler
                    .submit(image)
                    .unwrap_or_else(|_| panic!("scheduler submit should succeed"))
            })
            .collect();

        scheduler
            .flush()
            .unwrap_or_else(|_| panic!("scheduler flush should succeed"));

        let actual: Vec<Option<(String, f32)>> = handles
            .into_iter()
            .map(|handle| {
                handle
                    .wait()
                    .unwrap_or_else(|_| panic!("scheduler wait should succeed"))
            })
            .collect();

        assert_eq!(actual, expected);

        scheduler
            .shutdown()
            .unwrap_or_else(|_| panic!("scheduler shutdown should succeed"));
    }

    #[test]
    fn cnn_classifier_coalescing_scheduler_flush_unblocks_pending_request() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();

        for _ in 0..20 {
            let _ = classifier.train_image("animal_cat", cat.as_slice());
            let _ = classifier.train_image("animal_dog", dog.as_slice());
        }

        let mut scheduler = classifier.start_coalescing_scheduler(CnnCoalescingPredictOptions {
            max_micro_batch_size: 8,
            max_queue_size: 64,
            max_queue_delay_ms: 5_000,
            enable_batch_preprocess: true,
        });

        let handle = scheduler
            .submit(cat)
            .unwrap_or_else(|_| panic!("scheduler submit should succeed"));

        scheduler
            .flush()
            .unwrap_or_else(|_| panic!("scheduler flush should succeed"));

        let prediction = handle
            .wait()
            .unwrap_or_else(|_| panic!("scheduler wait should succeed"));
        assert!(prediction.is_some());

        scheduler
            .shutdown()
            .unwrap_or_else(|_| panic!("scheduler shutdown should succeed"));
    }
}
