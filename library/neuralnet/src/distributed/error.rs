use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::distributed::DistributedWorkUnitKind;
use crate::tensor::tensor4d::TensorError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedTensorError {
    Timeout {
        peer_id: String,
        timeout_ms: u64,
    },
    PeerUnavailable {
        peer_id: String,
    },
    UnsupportedWorkUnit {
        work_unit: DistributedWorkUnitKind,
    },
    InvalidPolicy(&'static str),
    Transport(String),
    Tensor(TensorError),
}

impl Display for DistributedTensorError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout {
                peer_id,
                timeout_ms,
            } => write!(f, "distributed tensor timeout on peer {peer_id} after {timeout_ms}ms"),
            Self::PeerUnavailable { peer_id } => {
                write!(f, "distributed tensor peer unavailable: {peer_id}")
            }
            Self::UnsupportedWorkUnit { work_unit } => {
                write!(f, "distributed tensor work unit is not supported: {work_unit:?}")
            }
            Self::InvalidPolicy(message) => write!(f, "invalid distributed execution policy: {message}"),
            Self::Transport(message) => write!(f, "distributed transport error: {message}"),
            Self::Tensor(err) => write!(f, "distributed tensor error: {err}"),
        }
    }
    
}

impl Error for DistributedTensorError {}

impl From<TensorError> for DistributedTensorError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}