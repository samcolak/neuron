mod error;
mod executor;
mod job;
mod libp2p;
mod policy;
mod server;
mod transport_executor;
mod transport;

pub use error::DistributedTensorError;
pub use executor::DistributedTensorExecutor;
pub use job::{
    DistributedTensorJob,
    DistributedTensorJobResult,
    RemoteConvBlockDescriptor,
    RemoteFeatureStackForwardRequest,
    RemoteTensorOp,
    RemoteTensorOpRequest,
    RemoteTensorRef,
};
pub use policy::{
    DistributedExecutionPolicy,
    DistributedExecutorCapabilities,
    DistributedFallbackPolicy,
    DistributedSyncMode,
    DistributedWorkUnitKind,
    RemotePeerDescriptor,
    TensorResidency,
};
pub use transport_executor::TransportBackedDistributedExecutor;
pub use transport::{
    DistributedJobRequestEnvelope,
    DistributedJobResultEnvelope,
    DistributedTransport,
    DistributedTransportKind,
    DistributedTransportMessage,
    DistributedTransportPayload,
    TransportCapabilitiesAnnouncement,
    TransportPeerRecord,
    DISTRIBUTED_TRANSPORT_PROTOCOL_NAME,
    DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
};

pub use libp2p::{Libp2pBootstrapPeer, Libp2pTransport, Libp2pTransportConfig};

pub use server::{
    DistributedServerConfig,
    DistributedServerRuntime,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_default_prefers_eventual_feature_stack_execution() {
        let policy = DistributedExecutionPolicy::coordinator_default();

        assert_eq!(
            policy.preferred_work_unit,
            DistributedWorkUnitKind::FeatureStackForward
        );
        assert_eq!(policy.sync_mode, DistributedSyncMode::AsynchronousEventual);
        assert_eq!(
            policy.fallback_policy,
            DistributedFallbackPolicy::RetryLocalOnTimeout
        );
    }

    #[test]
    fn policy_requires_batch_or_tensor_scale_to_offload() {
        let policy = DistributedExecutionPolicy {
            min_remote_batch_size: 4,
            min_remote_tensor_elements: 1_024,
            ..DistributedExecutionPolicy::coordinator_default()
        };

        assert!(!policy.should_offload(1, 256));
        assert!(policy.should_offload(4, 256));
        assert!(policy.should_offload(1, 1_024));
    }
}