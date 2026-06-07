use serde::{Deserialize, Serialize};

const DEFAULT_BETA1: f32 = 0.9;
const DEFAULT_BETA2: f32 = 0.999;
const DEFAULT_EPSILON: f32 = 1e-8;

fn default_beta1() -> f32 {
    DEFAULT_BETA1
}

fn default_beta2() -> f32 {
    DEFAULT_BETA2
}

fn default_epsilon() -> f32 {
    DEFAULT_EPSILON
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamOptimizer {
    #[serde(default = "default_beta1")]
    beta1: f32,
    #[serde(default = "default_beta2")]
    beta2: f32,
    #[serde(default = "default_epsilon")]
    epsilon: f32,
    #[serde(default)]
    step: u64,
    #[serde(default)]
    m: Vec<f32>,
    #[serde(default)]
    v: Vec<f32>,
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self {
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_EPSILON,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl AdamOptimizer {

    pub fn configure(&mut self, beta1: f32, beta2: f32, epsilon: f32) {
        self.beta1 = beta1.clamp(0.0, 0.9999);
        self.beta2 = beta2.clamp(0.0, 0.9999);
        self.epsilon = epsilon.max(1e-12);
    }

    pub fn hyperparameters(&self) -> (f32, f32, f32) {
        (self.beta1, self.beta2, self.epsilon)
    }
    
    fn ensure_state_len(&mut self, len: usize) {
        if self.m.len() != len {
            self.m = vec![0.0; len];
            self.v = vec![0.0; len];
        }
    }

    pub fn apply(
        &mut self,
        parameters: &mut [f32],
        gradients: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) {
        self.ensure_state_len(parameters.len());
        self.step = self.step.saturating_add(1);

        let step = self.step as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(step);
        let bias_correction2 = 1.0 - self.beta2.powf(step);
        let epsilon = self.epsilon.max(1e-12);

        for (idx, parameter) in parameters.iter_mut().enumerate() {
            let grad = gradients[idx] + weight_decay * *parameter;

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad;
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * grad * grad;

            let m_hat = self.m[idx] / bias_correction1.max(1e-12);
            let v_hat = self.v[idx] / bias_correction2.max(1e-12);

            *parameter -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

}
