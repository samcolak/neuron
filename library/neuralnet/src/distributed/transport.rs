use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::DistributedTensorError;
use super::job::{DistributedTensorJob, DistributedTensorJobResult};
use super::policy::{DistributedExecutorCapabilities, RemotePeerDescriptor};

pub const DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION: u16 = 1;
pub const DISTRIBUTED_TRANSPORT_PROTOCOL_NAME: &str = "/neuralnet/distributed/1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedTransportKind {
    InProcess,
    Libp2p,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportPeerRecord {
    pub descriptor: RemotePeerDescriptor,
    pub addresses: Vec<String>,
    pub average_rtt_ms: Option<u64>,
    pub last_seen_unix_ms: Option<u64>,
    pub is_reachable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportCapabilitiesAnnouncement {
    pub peer: RemotePeerDescriptor,
    pub executor: DistributedExecutorCapabilities,
    pub protocol_name: String,
    pub protocol_version: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedJobRequestEnvelope {
    pub job: DistributedTensorJob,
    pub timeout_ms: Option<u64>,
    pub require_ack: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedJobResultEnvelope {
    pub result: DistributedTensorJobResult,
    pub worker_peer_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributedTransportPayload {
    HealthCheck,
    HealthStatus {
        available: bool,
    },
    PeerDiscoveryRequest,
    PeerDiscoveryResponse {
        peers: Vec<TransportPeerRecord>,
    },
    CapabilitiesAnnouncement(TransportCapabilitiesAnnouncement),
    JobRequest(DistributedJobRequestEnvelope),
    JobResult(DistributedJobResultEnvelope),
    Ack {
        correlation_id: Uuid,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedTransportMessage {
    pub message_id: Uuid,
    pub correlation_id: Option<Uuid>,
    pub source_peer_id: String,
    pub target_peer_id: Option<String>,
    pub protocol_version: u16,
    pub payload: DistributedTransportPayload,
}

impl DistributedTransportMessage {
    pub fn new(
        source_peer_id: impl Into<String>,
        target_peer_id: Option<String>,
        payload: DistributedTransportPayload,
    ) -> Self {
        Self {
            message_id: Uuid::now_v7(),
            correlation_id: None,
            source_peer_id: source_peer_id.into(),
            target_peer_id,
            protocol_version: DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
            payload,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
}

pub trait DistributedTransport: Send + Sync {
    fn kind(&self) -> DistributedTransportKind;

    fn protocol_name(&self) -> &str;

    fn local_peer(&self) -> &RemotePeerDescriptor;

    fn announce_capabilities(
        &self,
        capabilities: DistributedExecutorCapabilities,
    ) -> Result<(), DistributedTensorError>;

    fn discover_peers(&self) -> Result<Vec<TransportPeerRecord>, DistributedTensorError>;

    fn send_message(
        &self,
        peer: &RemotePeerDescriptor,
        message: DistributedTransportMessage,
    ) -> Result<DistributedTransportMessage, DistributedTensorError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        DistributedExecutionPolicy,
        DistributedFallbackPolicy,
        DistributedSyncMode,
        DistributedWorkUnitKind,
        RemoteFeatureStackForwardRequest,
        TensorResidency,
    };
    use crate::tensor::tensor4d::Tensor4D;

    #[test]
    fn transport_message_roundtrips_with_job_payload() {
        let input = Tensor4D::zeros(1, 1, 2, 2);
        let job = DistributedTensorJob::FeatureStackForward(RemoteFeatureStackForwardRequest {
            input,
            blocks: Vec::new(),
            parameter_version: Some(7),
        });
        let message = DistributedTransportMessage::new(
            "peer-local",
            Some("peer-remote".to_string()),
            DistributedTransportPayload::JobRequest(DistributedJobRequestEnvelope {
                job,
                timeout_ms: Some(2_000),
                require_ack: true,
            }),
        );

        let encoded = serde_json::to_string(&message).expect("message should serialize");
        let decoded: DistributedTransportMessage =
            serde_json::from_str(&encoded).expect("message should deserialize");

        assert_eq!(decoded.protocol_version, DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION);
        assert_eq!(decoded.target_peer_id.as_deref(), Some("peer-remote"));
        assert!(matches!(
            decoded.payload,
            DistributedTransportPayload::JobRequest(DistributedJobRequestEnvelope {
                timeout_ms: Some(2_000),
                require_ack: true,
                ..
            })
        ));
    }

    #[test]
    fn capabilities_announcement_carries_executor_shape() {
        let announcement = TransportCapabilitiesAnnouncement {
            peer: RemotePeerDescriptor {
                peer_id: "peer-a".to_string(),
                platform: "macos".to_string(),
                accelerator: Some("metal".to_string()),
                transport: "libp2p".to_string(),
            },
            executor: DistributedExecutorCapabilities {
                supported_work_units: vec![DistributedWorkUnitKind::FeatureStackForward],
                supports_parameter_push: true,
                supports_parameter_pull: false,
                supports_result_streaming: false,
                supports_eventual_sync: true,
                max_tensor_elements: Some(1_000_000),
            },
            protocol_name: DISTRIBUTED_TRANSPORT_PROTOCOL_NAME.to_string(),
            protocol_version: DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
        };

        assert_eq!(announcement.protocol_name, "/neuralnet/distributed/1.0.0");
        assert_eq!(announcement.peer.transport, "libp2p");
    }

    #[test]
    fn peer_records_capture_distributed_reachability() {
        let policy = DistributedExecutionPolicy {
            preferred_work_unit: DistributedWorkUnitKind::FeatureStackForward,
            sync_mode: DistributedSyncMode::AsynchronousEventual,
            fallback_policy: DistributedFallbackPolicy::RetryLocalOnTimeout,
            timeout_ms: 2_000,
            min_remote_batch_size: 8,
            min_remote_tensor_elements: 65_536,
        };
        let record = TransportPeerRecord {
            descriptor: RemotePeerDescriptor {
                peer_id: "peer-b".to_string(),
                platform: "linux".to_string(),
                accelerator: None,
                transport: "libp2p".to_string(),
            },
            addresses: vec!["/ip4/127.0.0.1/udp/9000/quic-v1".to_string()],
            average_rtt_ms: Some(12),
            last_seen_unix_ms: Some(100),
            is_reachable: policy.should_offload(8, 128),
        };

        assert!(record.is_reachable);
        assert_eq!(record.descriptor.transport, "libp2p");
        assert_eq!(TensorResidency::LocalOnly, TensorResidency::LocalOnly);
    }
}