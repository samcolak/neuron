pub mod sgd;
pub mod types;

#[cfg(feature = "optimizer-adam")]
pub mod adam;

use serde::{Deserialize, Serialize};

use self::sgd::SgdOptimizer;
use self::types::LinearOptimizer;

#[cfg(feature = "optimizer-adam")]
use self::adam::AdamOptimizer;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerState {
    Sgd(SgdOptimizer),
    #[cfg(feature = "optimizer-adam")]
    Adam(AdamOptimizer),
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::Sgd(SgdOptimizer)
    }
}

impl OptimizerState {
    pub fn from_kind(kind: LinearOptimizer) -> Self {
        match kind {
            LinearOptimizer::Sgd => Self::Sgd(SgdOptimizer),
            #[cfg(feature = "optimizer-adam")]
            LinearOptimizer::Adam => Self::Adam(AdamOptimizer::default()),
        }
    }

    pub fn kind(&self) -> LinearOptimizer {
        match self {
            Self::Sgd(_) => LinearOptimizer::Sgd,
            #[cfg(feature = "optimizer-adam")]
            Self::Adam(_) => LinearOptimizer::Adam,
        }
    }

    pub fn apply(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) {
        match self {
            Self::Sgd(optimizer) => {
                optimizer.apply(parameters, gradients, learning_rate, weight_decay)
            }
            #[cfg(feature = "optimizer-adam")]
            Self::Adam(optimizer) => {
                optimizer.apply(parameters, gradients, learning_rate, weight_decay)
            }
        }
    }

    #[cfg(feature = "optimizer-adam")]
    pub fn configure_adam(&mut self, beta1: f32, beta2: f32, epsilon: f32) {
        if let Self::Adam(optimizer) = self {
            optimizer.configure(beta1, beta2, epsilon);
        }
    }
}
