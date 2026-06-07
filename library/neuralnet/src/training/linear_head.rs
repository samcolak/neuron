use std::error::Error;
use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

use crate::training::optimizers::OptimizerState;

fn default_optimizer() -> LinearOptimizer {
    LinearOptimizer::Sgd
}
pub use crate::training::optimizers::types::LinearOptimizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearHeadError {
    InvalidDimensions {
        input_dim: usize,
        output_dim: usize,
    },
    FeatureLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidTargetClass {
        class_index: usize,
        class_count: usize,
    },
    BatchSizeMismatch {
        feature_count: usize,
        target_count: usize,
    },
    EmptyBatch,
}

impl Display for LinearHeadError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions {
                input_dim,
                output_dim,
            } => write!(
                f,
                "invalid linear head dimensions: input_dim={}, output_dim={} (must both be > 0)",
                input_dim, output_dim
            ),
            Self::FeatureLengthMismatch { expected, actual } => write!(
                f,
                "feature vector length mismatch: expected {}, got {}",
                expected, actual
            ),
            Self::InvalidTargetClass {
                class_index,
                class_count,
            } => write!(
                f,
                "invalid target class {} for class count {}",
                class_index, class_count
            ),
            Self::BatchSizeMismatch {
                feature_count,
                target_count,
            } => write!(
                f,
                "batch size mismatch: features={}, targets={}",
                feature_count, target_count
            ),
            Self::EmptyBatch => write!(f, "cannot train linear head on an empty batch"),
        }
    }

}

impl Error for LinearHeadError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearHead {
    input_dim: usize,
    output_dim: usize,
    learning_rate: f32,
    weights: Vec<f32>,
    bias: Vec<f32>,
    #[serde(default = "default_optimizer")]
    optimizer: LinearOptimizer,
    #[serde(default)]
    weight_decay: f32,
    #[serde(default)]
    weight_optimizer_state: OptimizerState,
    #[serde(default)]
    bias_optimizer_state: OptimizerState,
    // Scratch buffers reused across train calls — excluded from serialisation.
    // All sizes are fixed by input_dim / output_dim for the lifetime of the head.
    #[serde(skip)]
    scratch_logits: Vec<f32>,
    #[serde(skip)]
    scratch_probs: Vec<f32>,
    #[serde(skip)]
    scratch_weight_grad: Vec<f32>,
    #[serde(skip)]
    scratch_bias_grad: Vec<f32>,
    #[serde(skip)]
    scratch_input_grad: Vec<f32>,
    #[serde(skip)]
    scratch_scaled_weight_grad: Vec<f32>,
    #[serde(skip)]
    scratch_scaled_bias_grad: Vec<f32>,
}

impl LinearHead {

    pub fn new(
        input_dim: usize,
        output_dim: usize,
        learning_rate: f32,
    ) -> Result<Self, LinearHeadError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(LinearHeadError::InvalidDimensions {
                input_dim,
                output_dim,
            });
        }

        let scale = 0.01f32;
        let mut weights = Vec::with_capacity(output_dim.saturating_mul(input_dim));
        for out in 0..output_dim {
            for in_idx in 0..input_dim {
                let centered = (in_idx as f32 / input_dim as f32) - 0.5;
                let signed = if out % 2 == 0 { centered } else { -centered };
                weights.push(signed * scale);
            }
        }

        let weight_len = output_dim * input_dim;
        Ok(Self {
            input_dim,
            output_dim,
            learning_rate: learning_rate.max(0.0),
            weights,
            bias: vec![0.0; output_dim],
            optimizer: LinearOptimizer::Sgd,
            weight_decay: 0.0,
            weight_optimizer_state: OptimizerState::from_kind(LinearOptimizer::Sgd),
            bias_optimizer_state: OptimizerState::from_kind(LinearOptimizer::Sgd),
            scratch_logits: vec![0.0f32; output_dim],
            scratch_probs: vec![0.0f32; output_dim],
            scratch_weight_grad: vec![0.0f32; weight_len],
            scratch_bias_grad: vec![0.0f32; output_dim],
            scratch_input_grad: vec![0.0f32; input_dim],
            scratch_scaled_weight_grad: vec![0.0f32; weight_len],
            scratch_scaled_bias_grad: vec![0.0f32; output_dim],
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub fn optimizer(&self) -> LinearOptimizer {
        self.optimizer
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_optimizer(&mut self, optimizer: LinearOptimizer) {
        self.optimizer = optimizer;
        self.weight_optimizer_state = OptimizerState::from_kind(optimizer);
        self.bias_optimizer_state = OptimizerState::from_kind(optimizer);
    }

    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay.max(0.0);
    }

    #[cfg(feature = "optimizer-adam")]
    pub fn configure_adam(&mut self, beta1: f32, beta2: f32, epsilon: f32) {
        self.weight_optimizer_state
            .configure_adam(beta1, beta2, epsilon);
        self.bias_optimizer_state
            .configure_adam(beta1, beta2, epsilon);
    }

    pub fn logits(&self, features: &[f32]) -> Result<Vec<f32>, LinearHeadError> {
        validate_features(features, self.input_dim)?;
        let mut logits = vec![0.0f32; self.output_dim];
        Self::compute_logits_into(&self.weights, &self.bias, self.input_dim, features, &mut logits);
        Ok(logits)
    }

    fn compute_logits_into(weights: &[f32], bias: &[f32], input_dim: usize, features: &[f32], logits: &mut Vec<f32>) {
        logits.resize(bias.len(), 0.0);
        for (out, out_logit) in logits.iter_mut().enumerate() {
            let mut value = bias[out];
            let row_offset = out * input_dim;
            for (in_idx, feature) in features.iter().enumerate() {
                value += weights[row_offset + in_idx] * *feature;
            }
            *out_logit = value;
        }
    }

    pub fn probabilities(&self, features: &[f32]) -> Result<Vec<f32>, LinearHeadError> {
        let logits = self.logits(features)?;
        Ok(softmax(&logits))
    }

    pub fn predict_class(&self, features: &[f32]) -> Result<usize, LinearHeadError> {

        let logits = self.logits(features)?;
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;

        for (idx, value) in logits.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best_idx = idx;
            }
        }

        Ok(best_idx)
        
    }

    pub fn train_step(
        &mut self,
        features: &[f32],
        target_class: usize,
    ) -> Result<f32, LinearHeadError> {
        let (loss, _input_grad) = self.train_step_with_input_gradient(features, target_class)?;
        Ok(loss)
    }

    pub fn train_step_with_input_gradient(
        &mut self,
        features: &[f32],
        target_class: usize,
    ) -> Result<(f32, Vec<f32>), LinearHeadError> {
        validate_features(features, self.input_dim)?;
        validate_target(target_class, self.output_dim)?;
        self.ensure_scratch_initialized();
        Self::compute_logits_into(&self.weights, &self.bias, self.input_dim, features, &mut self.scratch_logits);
        softmax_into(&self.scratch_logits.clone(), &mut self.scratch_probs);

        let target_prob = self.scratch_probs[target_class].max(1e-9);
        let loss = -target_prob.ln();

        self.scratch_probs[target_class] -= 1.0;

        self.scratch_input_grad.fill(0.0);
        for (in_idx, grad) in self.scratch_input_grad.iter_mut().enumerate() {
            let mut accum = 0.0f32;
            for (out, delta) in self.scratch_probs.iter().enumerate() {
                accum += *delta * self.weights[out * self.input_dim + in_idx];
            }
            *grad = accum;
        }

        self.scratch_weight_grad.fill(0.0);
        self.scratch_bias_grad.fill(0.0);
        for (out, delta) in self.scratch_probs.iter().enumerate() {
            let row_offset = out * self.input_dim;
            for (in_idx, feature) in features.iter().enumerate() {
                self.scratch_weight_grad[row_offset + in_idx] += *delta * *feature;
            }
            self.scratch_bias_grad[out] += *delta;
        }

        self.apply_gradients_from_scratch(1.0);

        Ok((loss, self.scratch_input_grad.clone()))
    }

    pub fn train_batch(
        &mut self,
        feature_batch: &[Vec<f32>],
        target_classes: &[usize],
    ) -> Result<f32, LinearHeadError> {
        let (loss, _input_grads) =
            self.train_batch_with_input_gradients(feature_batch, target_classes)?;
        Ok(loss)
    }

    pub fn train_batch_with_input_gradients(
        &mut self,
        feature_batch: &[Vec<f32>],
        target_classes: &[usize],
    ) -> Result<(f32, Vec<Vec<f32>>), LinearHeadError> {
        if feature_batch.is_empty() {
            return Err(LinearHeadError::EmptyBatch);
        }
        if feature_batch.len() != target_classes.len() {
            return Err(LinearHeadError::BatchSizeMismatch {
                feature_count: feature_batch.len(),
                target_count: target_classes.len(),
            });
        }

        for (features, target) in feature_batch.iter().zip(target_classes.iter()) {
            validate_features(features.as_slice(), self.input_dim)?;
            validate_target(*target, self.output_dim)?;
        }

        let batch_size = feature_batch.len() as f32;
        let mut total_loss = 0.0f32;
        self.ensure_scratch_initialized();
        self.scratch_weight_grad.fill(0.0);
        self.scratch_bias_grad.fill(0.0);
        let mut input_grads: Vec<Vec<f32>> = Vec::with_capacity(feature_batch.len());

        for (features, target) in feature_batch.iter().zip(target_classes.iter()) {
            Self::compute_logits_into(&self.weights, &self.bias, self.input_dim, features.as_slice(), &mut self.scratch_logits);
            softmax_into(&self.scratch_logits.clone(), &mut self.scratch_probs);

            let target_prob = self.scratch_probs[*target].max(1e-9);
            total_loss += -target_prob.ln();

            self.scratch_probs[*target] -= 1.0;

            self.scratch_input_grad.fill(0.0);
            for (in_idx, grad) in self.scratch_input_grad.iter_mut().enumerate() {
                let mut accum = 0.0f32;
                for (out, delta) in self.scratch_probs.iter().enumerate() {
                    accum += *delta * self.weights[out * self.input_dim + in_idx];
                }
                *grad = accum;
            }
            input_grads.push(self.scratch_input_grad.clone());

            for (out, delta) in self.scratch_probs.iter().enumerate() {
                let row_offset = out * self.input_dim;
                for (in_idx, feature) in features.iter().enumerate() {
                    self.scratch_weight_grad[row_offset + in_idx] += *delta * *feature;
                }
                self.scratch_bias_grad[out] += *delta;
            }
        }

        let scale = 1.0 / batch_size;
        self.apply_gradients_from_scratch(scale);

        Ok((total_loss * scale, input_grads))
    }

    fn synchronize_optimizer_state(&mut self) {
        if self.weight_optimizer_state.kind() != self.optimizer {
            self.weight_optimizer_state = OptimizerState::from_kind(self.optimizer);
        }
        if self.bias_optimizer_state.kind() != self.optimizer {
            self.bias_optimizer_state = OptimizerState::from_kind(self.optimizer);
        }
    }

    /// Ensures scratch buffers are the correct size after deserialisation,
    /// where serde(skip) leaves them as empty Vecs.
    fn ensure_scratch_initialized(&mut self) {
        let weight_len = self.output_dim * self.input_dim;
        if self.scratch_logits.len() != self.output_dim {
            self.scratch_logits.resize(self.output_dim, 0.0);
        }
        if self.scratch_probs.len() != self.output_dim {
            self.scratch_probs.resize(self.output_dim, 0.0);
        }
        if self.scratch_weight_grad.len() != weight_len {
            self.scratch_weight_grad.resize(weight_len, 0.0);
        }
        if self.scratch_bias_grad.len() != self.output_dim {
            self.scratch_bias_grad.resize(self.output_dim, 0.0);
        }
        if self.scratch_input_grad.len() != self.input_dim {
            self.scratch_input_grad.resize(self.input_dim, 0.0);
        }
        if self.scratch_scaled_weight_grad.len() != weight_len {
            self.scratch_scaled_weight_grad.resize(weight_len, 0.0);
        }
        if self.scratch_scaled_bias_grad.len() != self.output_dim {
            self.scratch_scaled_bias_grad.resize(self.output_dim, 0.0);
        }
    }

    fn apply_gradients(&mut self, weight_grad: &[f32], bias_grad: &[f32], scale: f32) {
        self.synchronize_optimizer_state();

        self.scratch_scaled_weight_grad.resize(weight_grad.len(), 0.0);
        self.scratch_scaled_bias_grad.resize(bias_grad.len(), 0.0);

        for (dst, src) in self.scratch_scaled_weight_grad.iter_mut().zip(weight_grad.iter()) {
            *dst = *src * scale;
        }
        for (dst, src) in self.scratch_scaled_bias_grad.iter_mut().zip(bias_grad.iter()) {
            *dst = *src * scale;
        }

        self.weight_optimizer_state.apply(
            self.weights.as_mut_slice(),
            self.scratch_scaled_weight_grad.as_slice(),
            self.learning_rate,
            self.weight_decay,
        );
        self.bias_optimizer_state.apply(
            self.bias.as_mut_slice(),
            self.scratch_scaled_bias_grad.as_slice(),
            self.learning_rate,
            0.0,
        );
    }

    /// Variant used by train_step/train_batch — reads directly from scratch_weight_grad
    /// and scratch_bias_grad, avoiding a redundant borrow of self.
    fn apply_gradients_from_scratch(&mut self, scale: f32) {
        self.synchronize_optimizer_state();

        self.scratch_scaled_weight_grad.resize(self.scratch_weight_grad.len(), 0.0);
        self.scratch_scaled_bias_grad.resize(self.scratch_bias_grad.len(), 0.0);

        for (dst, src) in self.scratch_scaled_weight_grad.iter_mut().zip(self.scratch_weight_grad.iter()) {
            *dst = *src * scale;
        }
        for (dst, src) in self.scratch_scaled_bias_grad.iter_mut().zip(self.scratch_bias_grad.iter()) {
            *dst = *src * scale;
        }

        self.weight_optimizer_state.apply(
            self.weights.as_mut_slice(),
            self.scratch_scaled_weight_grad.as_slice(),
            self.learning_rate,
            self.weight_decay,
        );
        self.bias_optimizer_state.apply(
            self.bias.as_mut_slice(),
            self.scratch_scaled_bias_grad.as_slice(),
            self.learning_rate,
            0.0,
        );
    }
}

fn validate_features(features: &[f32], expected_len: usize) -> Result<(), LinearHeadError> {
    if features.len() != expected_len {
        return Err(LinearHeadError::FeatureLengthMismatch {
            expected: expected_len,
            actual: features.len(),
        });
    }

    Ok(())
}

fn validate_target(target_class: usize, class_count: usize) -> Result<(), LinearHeadError> {
    if target_class >= class_count {
        return Err(LinearHeadError::InvalidTargetClass {
            class_index: target_class,
            class_count,
        });
    }

    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.iter().map(|logit| (*logit - max_logit).exp()).collect();
    let denom: f32 = exps.iter().copied().sum();

    if denom <= 0.0 {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }

    exps.into_iter().map(|value| value / denom).collect()
}

/// In-place softmax — writes into an existing buffer to avoid allocation on hot paths.
fn softmax_into(logits: &[f32], out: &mut Vec<f32>) {
    out.resize(logits.len(), 0.0);

    if logits.is_empty() {
        return;
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut denom = 0.0f32;
    for (dst, src) in out.iter_mut().zip(logits.iter()) {
        *dst = (*src - max_logit).exp();
        denom += *dst;
    }

    if denom <= 0.0 {
        let uniform = 1.0 / logits.len() as f32;
        out.fill(uniform);
    } else {
        for v in out.iter_mut() {
            *v /= denom;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_head_probabilities_sum_to_one() {
        let head = LinearHead::new(4, 3, 0.1)
            .unwrap_or_else(|_| panic!("linear head should initialize"));

        let probs = head
            .probabilities(&[1.0, 0.0, 0.5, -0.5])
            .unwrap_or_else(|_| panic!("probabilities should compute"));

        let sum: f32 = probs.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_head_train_step_reduces_loss_on_repeated_example() {
        let mut head = LinearHead::new(4, 2, 0.2)
            .unwrap_or_else(|_| panic!("linear head should initialize"));
        let features = [1.0, 1.0, 0.0, 0.0];

        let initial = head
            .train_step(&features, 0)
            .unwrap_or_else(|_| panic!("first training step should succeed"));

        let mut final_loss = initial;
        for _ in 0..50 {
            final_loss = head
                .train_step(&features, 0)
                .unwrap_or_else(|_| panic!("repeated training step should succeed"));
        }

        assert!(final_loss < initial);
        assert_eq!(
            head.predict_class(&features)
                .unwrap_or_else(|_| panic!("prediction should succeed")),
            0
        );
    }

    #[test]
    fn linear_head_train_batch_rejects_mismatched_batch_sizes() {
        let mut head = LinearHead::new(2, 2, 0.1)
            .unwrap_or_else(|_| panic!("linear head should initialize"));

        let result = head.train_batch(&[vec![1.0, 0.0]], &[0, 1]);
        assert!(matches!(
            result,
            Err(LinearHeadError::BatchSizeMismatch {
                feature_count: 1,
                target_count: 2
            })
        ));
    }

    #[test]
    fn linear_head_train_step_with_input_gradient_matches_input_dim() {
        let mut head = LinearHead::new(3, 2, 0.1)
            .unwrap_or_else(|_| panic!("linear head should initialize"));

        let (_loss, input_grad) = head
            .train_step_with_input_gradient(&[1.0, -0.5, 0.25], 1)
            .unwrap_or_else(|_| panic!("train step with input gradient should succeed"));

        assert_eq!(input_grad.len(), 3);
        assert!(input_grad.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn linear_head_train_batch_with_input_gradients_returns_per_sample_grads() {
        let mut head = LinearHead::new(3, 2, 0.1)
            .unwrap_or_else(|_| panic!("linear head should initialize"));

        let (loss, grads) = head
            .train_batch_with_input_gradients(
                &[vec![1.0, 0.0, 0.5], vec![0.5, 1.0, 0.0]],
                &[0, 1],
            )
            .unwrap_or_else(|_| panic!("batch train with input gradients should succeed"));

        assert!(loss.is_finite());
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].len(), 3);
        assert_eq!(grads[1].len(), 3);
        assert!(grads.iter().flatten().all(|value| value.is_finite()));
    }

    #[cfg(feature = "optimizer-adam")]
    #[test]
    fn linear_head_adam_optimizer_reduces_loss_on_repeated_example() {
        let mut head = LinearHead::new(4, 2, 0.05)
            .unwrap_or_else(|_| panic!("linear head should initialize"));
        head.set_optimizer(LinearOptimizer::Adam);
        let features = [1.0, 1.0, 0.0, 0.0];

        let initial = head
            .train_step(&features, 0)
            .unwrap_or_else(|_| panic!("first training step should succeed"));

        let mut final_loss = initial;
        for _ in 0..80 {
            final_loss = head
                .train_step(&features, 0)
                .unwrap_or_else(|_| panic!("repeated training step should succeed"));
        }

        assert!(final_loss < initial);
        assert_eq!(
            head.predict_class(&features)
                .unwrap_or_else(|_| panic!("prediction should succeed")),
            0
        );
    }
}
