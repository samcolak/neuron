use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedWorkUnitKind {
    TensorOp,
    ConvBlock,
    FeatureStackForward,
    TrainingShard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedSyncMode {
    SynchronousDeterministic,
    BoundedStaleness { max_version_lag: u64 },
    AsynchronousEventual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedFallbackPolicy {
    RetryLocalOnTimeout,
    RetryPeerThenLocal { max_peer_attempts: usize },
    Abort,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorResidency {
    LocalOnly,
    LocalMirroredRemote { peer_id: String, tensor_id: String },
    RemoteOwned { peer_id: String, tensor_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePeerDescriptor {
    pub peer_id: String,
    pub platform: String,
    pub accelerator: Option<String>,
    pub transport: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedExecutionPolicy {
    pub preferred_work_unit: DistributedWorkUnitKind,
    pub sync_mode: DistributedSyncMode,
    pub fallback_policy: DistributedFallbackPolicy,
    pub timeout_ms: u64,
    pub min_remote_batch_size: usize,
    pub min_remote_tensor_elements: usize,
}

impl DistributedExecutionPolicy {

    pub fn coordinator_default() -> Self {
        Self {
            preferred_work_unit: DistributedWorkUnitKind::FeatureStackForward,
            sync_mode: DistributedSyncMode::AsynchronousEventual,
            fallback_policy: DistributedFallbackPolicy::RetryLocalOnTimeout,
            timeout_ms: 2_000,
            min_remote_batch_size: 8,
            min_remote_tensor_elements: 65_536,
        }
    }

    pub fn should_offload(&self, batch_size: usize, tensor_elements: usize) -> bool {
        batch_size >= self.min_remote_batch_size
            || tensor_elements >= self.min_remote_tensor_elements
    }

}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedExecutorCapabilities {
    pub supported_work_units: Vec<DistributedWorkUnitKind>,
    pub supports_parameter_push: bool,
    pub supports_parameter_pull: bool,
    pub supports_result_streaming: bool,
    pub supports_eventual_sync: bool,
    pub max_tensor_elements: Option<usize>,
}

impl Default for DistributedExecutorCapabilities {

    fn default() -> Self {
        Self {
            supported_work_units: vec![
                DistributedWorkUnitKind::TensorOp,
                DistributedWorkUnitKind::FeatureStackForward,
            ],
            supports_parameter_push: true,
            supports_parameter_pull: false,
            supports_result_streaming: false,
            supports_eventual_sync: true,
            max_tensor_elements: None,
        }
    }
    
}