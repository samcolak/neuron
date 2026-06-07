use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SgdOptimizer;

impl SgdOptimizer {

    pub fn apply(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) {
        for (parameter, gradient) in parameters.iter_mut().zip(gradients.iter()) {
            let regularized = *gradient + weight_decay * *parameter;
            *parameter -= learning_rate * regularized;
        }
    }
    
}
