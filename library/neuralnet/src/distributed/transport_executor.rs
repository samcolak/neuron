use super::error::DistributedTensorError;
use super::executor::DistributedTensorExecutor;
use super::job::{DistributedTensorJob, DistributedTensorJobResult};
use super::policy::{
    DistributedExecutionPolicy,
    DistributedExecutorCapabilities,
    RemotePeerDescriptor,
};
use super::transport::{
    DistributedJobRequestEnvelope,
    DistributedTransport,
    DistributedTransportMessage,
    DistributedTransportPayload,
};

pub struct TransportBackedDistributedExecutor<T: DistributedTransport> {
    policy: DistributedExecutionPolicy,
    capabilities: DistributedExecutorCapabilities,
    transport: T,
}

impl<T: DistributedTransport> TransportBackedDistributedExecutor<T> {
    pub fn new(
        policy: DistributedExecutionPolicy,
        capabilities: DistributedExecutorCapabilities,
        transport: T,
    ) -> Self {
        Self {
            policy,
            capabilities,
            transport,
        }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }
}

impl<T: DistributedTransport> DistributedTensorExecutor for TransportBackedDistributedExecutor<T> {
    fn name(&self) -> &'static str {
        "transport-backed-distributed-executor"
    }

    fn policy(&self) -> &DistributedExecutionPolicy {
        &self.policy
    }

    fn capabilities(&self) -> DistributedExecutorCapabilities {
        self.capabilities.clone()
    }

    fn execute(
        &self,
        peer: &RemotePeerDescriptor,
        job: DistributedTensorJob,
    ) -> Result<DistributedTensorJobResult, DistributedTensorError> {
        let request = DistributedTransportMessage::new(
            self.transport.local_peer().peer_id.clone(),
            Some(peer.peer_id.clone()),
            DistributedTransportPayload::JobRequest(DistributedJobRequestEnvelope {
                job,
                timeout_ms: Some(self.policy.timeout_ms),
                require_ack: true,
            }),
        );

        let response = self.transport.send_message(peer, request)?;
        match response.payload {
            DistributedTransportPayload::JobResult(result) => Ok(result.result),
            DistributedTransportPayload::Error { message } => {
                Err(DistributedTensorError::Transport(message))
            }
            other => Err(DistributedTensorError::Transport(format!(
                "unexpected transport response payload: {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        DistributedFallbackPolicy,
        DistributedSyncMode,
        DistributedTransportKind,
        DistributedWorkUnitKind,
        RemoteFeatureStackForwardRequest,
        TensorResidency,
    };
    use crate::tensor::tensor4d::Tensor4D;

    #[derive(Clone)]
    struct FakeTransport {
        local: RemotePeerDescriptor,
        response: DistributedTransportPayload,
    }

    impl DistributedTransport for FakeTransport {
        fn kind(&self) -> DistributedTransportKind {
            DistributedTransportKind::InProcess
        }

        fn protocol_name(&self) -> &str {
            "in-process"
        }

        fn local_peer(&self) -> &RemotePeerDescriptor {
            &self.local
        }

        fn announce_capabilities(
            &self,
            _capabilities: DistributedExecutorCapabilities,
        ) -> Result<(), DistributedTensorError> {
            Ok(())
        }

        fn discover_peers(&self) -> Result<Vec<crate::distributed::TransportPeerRecord>, DistributedTensorError> {
            Ok(Vec::new())
        }

        fn send_message(
            &self,
            _peer: &RemotePeerDescriptor,
            _message: DistributedTransportMessage,
        ) -> Result<DistributedTransportMessage, DistributedTensorError> {
            Ok(DistributedTransportMessage::new(
                self.local.peer_id.clone(),
                None,
                self.response.clone(),
            ))
        }
    }

    fn test_policy() -> DistributedExecutionPolicy {
        DistributedExecutionPolicy {
            preferred_work_unit: DistributedWorkUnitKind::FeatureStackForward,
            sync_mode: DistributedSyncMode::AsynchronousEventual,
            fallback_policy: DistributedFallbackPolicy::RetryLocalOnTimeout,
            timeout_ms: 1_000,
            min_remote_batch_size: 8,
            min_remote_tensor_elements: 65_536,
        }
    }

    fn test_caps() -> DistributedExecutorCapabilities {
        DistributedExecutorCapabilities {
            supported_work_units: vec![DistributedWorkUnitKind::FeatureStackForward],
            supports_parameter_push: true,
            supports_parameter_pull: false,
            supports_result_streaming: false,
            supports_eventual_sync: true,
            max_tensor_elements: Some(1_000_000),
        }
    }

    #[test]
    fn executor_returns_job_result_from_transport() {
        let local = RemotePeerDescriptor {
            peer_id: "local".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "in-process".to_string(),
        };
        let remote = RemotePeerDescriptor {
            peer_id: "remote".to_string(),
            platform: "linux".to_string(),
            accelerator: None,
            transport: "in-process".to_string(),
        };
        let response_payload = DistributedTransportPayload::JobResult(
            super::super::transport::DistributedJobResultEnvelope {
                result: DistributedTensorJobResult::RemoteTensor(crate::distributed::RemoteTensorRef {
                    peer_id: "remote".to_string(),
                    tensor_id: "t-1".to_string(),
                    shape: (1, 1, 1, 1),
                    residency: TensorResidency::RemoteOwned {
                        peer_id: "remote".to_string(),
                        tensor_id: "t-1".to_string(),
                    },
                    version: Some(1),
                }),
                worker_peer_id: "remote".to_string(),
            },
        );

        let executor = TransportBackedDistributedExecutor::new(
            test_policy(),
            test_caps(),
            FakeTransport {
                local,
                response: response_payload,
            },
        );

        let job = DistributedTensorJob::FeatureStackForward(RemoteFeatureStackForwardRequest {
            input: Tensor4D::zeros(1, 1, 2, 2),
            blocks: Vec::new(),
            parameter_version: Some(42),
        });
        let result = executor.execute(&remote, job).expect("execution should succeed");

        assert!(matches!(result, DistributedTensorJobResult::RemoteTensor(_)));
    }

    #[test]
    fn executor_maps_transport_error_payload() {
        let local = RemotePeerDescriptor {
            peer_id: "local".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "in-process".to_string(),
        };
        let remote = RemotePeerDescriptor {
            peer_id: "remote".to_string(),
            platform: "linux".to_string(),
            accelerator: None,
            transport: "in-process".to_string(),
        };

        let executor = TransportBackedDistributedExecutor::new(
            test_policy(),
            test_caps(),
            FakeTransport {
                local,
                response: DistributedTransportPayload::Error {
                    message: "peer execution failed".to_string(),
                },
            },
        );

        let job = DistributedTensorJob::FeatureStackForward(RemoteFeatureStackForwardRequest {
            input: Tensor4D::zeros(1, 1, 2, 2),
            blocks: Vec::new(),
            parameter_version: Some(42),
        });

        let error = executor
            .execute(&remote, job)
            .expect_err("execution should surface transport error");
        assert!(matches!(error, DistributedTensorError::Transport(_)));
    }
}