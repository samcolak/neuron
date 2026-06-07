use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LinearOptimizer {
    #[default]
    Sgd,
    #[cfg(feature = "optimizer-adam")]
    Adam,
}
