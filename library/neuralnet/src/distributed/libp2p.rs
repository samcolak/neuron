use std::collections::HashMap;
use std::env;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use libp2p::{Multiaddr, PeerId};

use super::error::DistributedTensorError;
use super::policy::{DistributedExecutorCapabilities, RemotePeerDescriptor};
use super::transport::{
    DistributedJobResultEnvelope,
    DistributedTransport,
    DistributedTransportKind,
    DistributedTransportMessage,
    DistributedTransportPayload,
    TransportPeerRecord,
    DISTRIBUTED_TRANSPORT_PROTOCOL_NAME,
    DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
};
use crate::distributed::{DistributedTensorJobResult, RemoteTensorOp};
use crate::tensor::backend::{
    cpu_backend,
    cuda_backend,
    cuda_backend_available,
    mlx_backend,
    mlx_backend_available,
    TensorBackend,
    TensorBackendKind,
};

const DEFAULT_SWARM_NAME: &str = "neuralnet-distributed";
const DEFAULT_SWARM_VERSION: &str = "v1";

enum ReactorCommand {
    Message(Box<ReactorMessageCommand>),
    Stop,
}

struct ReactorMessageCommand {
    source: RemotePeerDescriptor,
    incoming: DistributedTransportMessage,
    response_tx: Sender<Result<DistributedTransportMessage, DistributedTensorError>>,
}

type SwarmRegistry = HashMap<String, HashMap<String, Sender<ReactorCommand>>>;
static LOCAL_SWARM_REGISTRY: OnceLock<Mutex<SwarmRegistry>> = OnceLock::new();

fn local_swarm_registry() -> &'static Mutex<SwarmRegistry> {
    LOCAL_SWARM_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Libp2pBootstrapPeer {
    pub peer_id: String,
    pub address: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Libp2pTransportConfig {
    pub listen_addresses: Vec<String>,
    pub bootstrap_peers: Vec<Libp2pBootstrapPeer>,
    pub swarm_name: String,
    pub swarm_version: String,
    pub protocol_name: String,
    pub request_timeout_ms: u64,
}

impl Default for Libp2pTransportConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec!["/ip4/0.0.0.0/udp/9000/quic-v1".to_string()],
            bootstrap_peers: Vec::new(),
            swarm_name: DEFAULT_SWARM_NAME.to_string(),
            swarm_version: DEFAULT_SWARM_VERSION.to_string(),
            protocol_name: DISTRIBUTED_TRANSPORT_PROTOCOL_NAME.to_string(),
            request_timeout_ms: 2_000,
        }
    }
}

impl Libp2pTransportConfig {
    pub fn swarm_identifier(&self) -> String {
        format!("{}@{}", self.swarm_name.trim(), self.swarm_version.trim())
    }

    pub fn scoped_protocol_name(&self) -> String {
        format!(
            "{}/{}/{}",
            self.protocol_name.trim_end_matches('/'),
            self.swarm_name.trim(),
            self.swarm_version.trim()
        )
    }

    pub fn local_discovery_namespace(&self) -> String {
        format!(
            "/neuralnet/swarm/{}/{}",
            self.swarm_name.trim(),
            self.swarm_version.trim()
        )
    }

    pub fn validate(&self) -> Result<(), DistributedTensorError> {
        if self.swarm_name.trim().is_empty() {
            return Err(DistributedTensorError::Transport(
                "libp2p swarm_name must not be empty".to_string(),
            ));
        }

        if self.swarm_version.trim().is_empty() {
            return Err(DistributedTensorError::Transport(
                "libp2p swarm_version must not be empty".to_string(),
            ));
        }

        if self.protocol_name.trim().is_empty() {
            return Err(DistributedTensorError::Transport(
                "libp2p protocol name must not be empty".to_string(),
            ));
        }

        for address in &self.listen_addresses {
            address.parse::<Multiaddr>().map_err(|err| {
                DistributedTensorError::Transport(format!(
                    "invalid libp2p listen address {address}: {err}"
                ))
            })?;
        }

        for peer in &self.bootstrap_peers {
            peer.address.parse::<Multiaddr>().map_err(|err| {
                DistributedTensorError::Transport(format!(
                    "invalid libp2p bootstrap address {}: {err}",
                    peer.address
                ))
            })?;

            peer.peer_id.parse::<PeerId>().map_err(|err| {
                DistributedTensorError::Transport(format!(
                    "invalid libp2p bootstrap peer id {}: {err}",
                    peer.peer_id
                ))
            })?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct Libp2pTransport {
    local_peer: RemotePeerDescriptor,
    config: Libp2pTransportConfig,
    known_peers: Arc<Mutex<Vec<TransportPeerRecord>>>,
    reactor_tx: Sender<ReactorCommand>,
    reactor_join: Mutex<Option<JoinHandle<()>>>,
}

fn selected_server_backend_kind() -> TensorBackendKind {
    let configured = env::var("NEURALNET_DISTRIBUTED_SERVER_BACKEND")
        .ok()
        .or_else(|| env::var("NEURALNET_TENSOR_BACKEND").ok())
        .or_else(|| env::var("NEURALNET_BACKEND").ok())
        .and_then(|value| TensorBackendKind::parse(value.as_str()));

    let candidate = configured.unwrap_or_else(default_server_backend_kind);
    if candidate == TensorBackendKind::Distributed {
        return default_server_backend_kind();
    }

    if is_backend_available(candidate) {
        candidate
    } else {
        default_server_backend_kind()
    }
}

fn default_server_backend_kind() -> TensorBackendKind {
    if cuda_backend_available() {
        TensorBackendKind::Cuda
    } else if mlx_backend_available() {
        TensorBackendKind::Mlx
    } else {
        TensorBackendKind::Cpu
    }
}

fn is_backend_available(kind: TensorBackendKind) -> bool {
    match kind {
        TensorBackendKind::Cpu => true,
        TensorBackendKind::Cuda => cuda_backend_available(),
        TensorBackendKind::Mlx => mlx_backend_available(),
        TensorBackendKind::Distributed => false,
    }
}

fn with_server_backend<T, F>(kind: TensorBackendKind, op: F) -> Result<T, DistributedTensorError>
where
    F: FnOnce(&dyn TensorBackend) -> Result<T, DistributedTensorError>,
{
    match kind {
        TensorBackendKind::Cpu => {
            let backend = cpu_backend();
            op(&backend)
        }
        TensorBackendKind::Cuda => {
            let backend = cuda_backend();
            op(&backend)
        }
        TensorBackendKind::Mlx => {
            let backend = mlx_backend();
            op(&backend)
        }
        TensorBackendKind::Distributed => {
            let backend = cpu_backend();
            op(&backend)
        }
    }
}

fn execute_feature_stack_forward(
    kind: TensorBackendKind,
    req: super::job::RemoteFeatureStackForwardRequest,
) -> Result<Vec<Vec<f32>>, DistributedTensorError> {
    with_server_backend(kind, move |backend| {
        let mut current = req.input;

        for block in req.blocks {
            current = backend.conv_relu_max_pool2d_valid(
                &current,
                &block.kernels,
                Some(block.bias.as_slice()),
                1,
                1,
                2,
                2,
                2,
                2,
            )?;
        }

        let pooled = backend.global_average_pool2d(&current)?;
        Ok(pooled.flatten_batch_features())
    })
}

fn execute_remote_tensor_op(
    kind: TensorBackendKind,
    operation: RemoteTensorOp,
) -> Result<crate::tensor::tensor4d::Tensor4D, DistributedTensorError> {
    with_server_backend(kind, move |backend| match operation {
        RemoteTensorOp::Conv2dValid {
            input,
            kernels,
            bias,
            stride_h,
            stride_w,
        } => backend
            .conv2d_valid(&input, &kernels, bias.as_deref(), stride_h, stride_w)
            .map_err(DistributedTensorError::from),
        RemoteTensorOp::MaxPool2d {
            input,
            window_h,
            window_w,
            stride_h,
            stride_w,
        } => backend
            .max_pool2d(&input, window_h, window_w, stride_h, stride_w)
            .map_err(DistributedTensorError::from),
        RemoteTensorOp::GlobalAveragePool2d { input } => backend
            .global_average_pool2d(&input)
            .map_err(DistributedTensorError::from),
        RemoteTensorOp::Relu { mut input } => {
            backend.relu_inplace(&mut input);
            Ok(input)
        }
        RemoteTensorOp::ConvReluMaxPool2dValid {
            input,
            kernels,
            bias,
            conv_stride_h,
            conv_stride_w,
            pool_window_h,
            pool_window_w,
            pool_stride_h,
            pool_stride_w,
        } => backend
            .conv_relu_max_pool2d_valid(
                &input,
                &kernels,
                bias.as_deref(),
                conv_stride_h,
                conv_stride_w,
                pool_window_h,
                pool_window_w,
                pool_stride_h,
                pool_stride_w,
            )
            .map_err(DistributedTensorError::from),
    })
}

impl Libp2pTransport {
    pub fn new(
        local_peer: RemotePeerDescriptor,
        config: Libp2pTransportConfig,
    ) -> Result<Self, DistributedTensorError> {
        config.validate()?;
        let known_peers = Arc::new(Mutex::new(Vec::new()));
        let (reactor_tx, reactor_join) = Self::spawn_swarm_reactor(
            local_peer.clone(),
            config.clone(),
            Arc::clone(&known_peers),
        );

        let swarm_id = config.swarm_identifier();
        Self::register_local_reactor(&swarm_id, &local_peer.peer_id, reactor_tx.clone())?;

        Ok(Self {
            local_peer,
            config,
            known_peers,
            reactor_tx,
            reactor_join: Mutex::new(Some(reactor_join)),
        })
    }

    fn register_local_reactor(
        swarm_identifier: &str,
        peer_id: &str,
        sender: Sender<ReactorCommand>,
    ) -> Result<(), DistributedTensorError> {
        let mut registry = local_swarm_registry().lock().map_err(|_| {
            DistributedTensorError::Transport(
                "failed to lock local swarm registry for reactor registration".to_string(),
            )
        })?;

        registry
            .entry(swarm_identifier.to_string())
            .or_default()
            .insert(peer_id.to_string(), sender);
        Ok(())
    }

    fn unregister_local_reactor(swarm_identifier: &str, peer_id: &str) {
        if let Ok(mut registry) = local_swarm_registry().lock()
            && let Some(peers) = registry.get_mut(swarm_identifier)
        {
            peers.remove(peer_id);
            if peers.is_empty() {
                registry.remove(swarm_identifier);
            }
        }
    }

    fn resolve_local_reactor(
        &self,
        peer_id: &str,
    ) -> Result<Option<Sender<ReactorCommand>>, DistributedTensorError> {
        let registry = local_swarm_registry().lock().map_err(|_| {
            DistributedTensorError::Transport(
                "failed to lock local swarm registry for reactor lookup".to_string(),
            )
        })?;

        Ok(registry
            .get(&self.swarm_identifier())
            .and_then(|peers| peers.get(peer_id))
            .cloned())
    }

    fn dispatch_message_to_reactor(
        &self,
        sender: Sender<ReactorCommand>,
        peer: &RemotePeerDescriptor,
        message: DistributedTransportMessage,
    ) -> Result<DistributedTransportMessage, DistributedTensorError> {
        let (response_tx, response_rx) = mpsc::channel();
        let source = RemotePeerDescriptor {
            peer_id: message.source_peer_id.clone(),
            platform: "unknown".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };

        sender
            .send(ReactorCommand::Message(Box::new(ReactorMessageCommand {
                source,
                incoming: message,
                response_tx,
            })))
            .map_err(|err| {
                DistributedTensorError::Transport(format!(
                    "libp2p swarm reactor command send failed for {}: {err}",
                    self.swarm_identifier()
                ))
            })?;

        response_rx
            .recv_timeout(Duration::from_millis(self.config.request_timeout_ms))
            .map_err(|_| DistributedTensorError::Timeout {
                peer_id: peer.peer_id.clone(),
                timeout_ms: self.config.request_timeout_ms,
            })?
    }

    fn spawn_swarm_reactor(
        local_peer: RemotePeerDescriptor,
        config: Libp2pTransportConfig,
        known_peers: Arc<Mutex<Vec<TransportPeerRecord>>>,
    ) -> (Sender<ReactorCommand>, JoinHandle<()>) {
        let (command_tx, command_rx) = mpsc::channel::<ReactorCommand>();
        let join = thread::Builder::new()
            .name(format!("{}-reactor", config.swarm_identifier()))
            .spawn(move || {
                while let Ok(command) = command_rx.recv() {
                    match command {
                        ReactorCommand::Message(message) => {
                            let ReactorMessageCommand {
                            source,
                            incoming,
                            response_tx,
                        } = *message;
                            let response = Self::process_swarm_message(
                                &local_peer,
                                &config,
                                &known_peers,
                                &source,
                                incoming,
                            );
                            let _ = response_tx.send(response);
                        }
                        ReactorCommand::Stop => {
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn libp2p swarm reactor thread");

        (command_tx, join)
    }

    fn process_swarm_message(
        local_peer: &RemotePeerDescriptor,
        config: &Libp2pTransportConfig,
        known_peers: &Arc<Mutex<Vec<TransportPeerRecord>>>,
        source: &RemotePeerDescriptor,
        incoming: DistributedTransportMessage,
    ) -> Result<DistributedTransportMessage, DistributedTensorError> {
        let swarm_identifier = config.swarm_identifier();
        if incoming.protocol_version != DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION {
            return Ok(
                DistributedTransportMessage::new(
                    local_peer.peer_id.clone(),
                    Some(source.peer_id.clone()),
                    DistributedTransportPayload::Error {
                        message: format!(
                            "protocol version mismatch in swarm {}: expected {}, got {}",
                            swarm_identifier,
                            DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
                            incoming.protocol_version
                        ),
                    },
                )
                .with_correlation_id(incoming.message_id),
            );
        }

        let payload = match incoming.payload {
            DistributedTransportPayload::HealthCheck => {
                DistributedTransportPayload::HealthStatus { available: true }
            }
            DistributedTransportPayload::PeerDiscoveryRequest => {
                let mut peers = config
                    .bootstrap_peers
                    .iter()
                    .map(|peer| TransportPeerRecord {
                        descriptor: RemotePeerDescriptor {
                            peer_id: peer.peer_id.clone(),
                            platform: "unknown".to_string(),
                            accelerator: None,
                            transport: "libp2p".to_string(),
                        },
                        addresses: vec![peer.address.clone()],
                        average_rtt_ms: None,
                        last_seen_unix_ms: None,
                        is_reachable: false,
                    })
                    .collect::<Vec<_>>();
                peers.extend(
                    known_peers
                        .lock()
                        .map(|p| p.clone())
                        .unwrap_or_default(),
                );
                DistributedTransportPayload::PeerDiscoveryResponse { peers }
            }
            DistributedTransportPayload::CapabilitiesAnnouncement(announcement) => {
                if announcement.protocol_name != config.scoped_protocol_name() {
                    DistributedTransportPayload::Error {
                        message: format!(
                            "capabilities announcement ignored: swarm scope mismatch for {}",
                            swarm_identifier
                        ),
                    }
                } else {
                    if let Ok(mut peers) = known_peers.lock() {
                        let record = TransportPeerRecord {
                            descriptor: announcement.peer,
                            addresses: Vec::new(),
                            average_rtt_ms: None,
                            last_seen_unix_ms: None,
                            is_reachable: true,
                        };

                        if let Some(existing) = peers
                            .iter_mut()
                            .find(|existing| {
                                existing.descriptor.peer_id == record.descriptor.peer_id
                            })
                        {
                            *existing = record;
                        } else {
                            peers.push(record);
                        }
                    }

                    DistributedTransportPayload::Ack {
                        correlation_id: incoming.message_id,
                    }
                }
            }
            DistributedTransportPayload::JobRequest(request) => {
                match execute_distributed_job_request(local_peer, request) {
                    Ok(result) => DistributedTransportPayload::JobResult(DistributedJobResultEnvelope {
                        result,
                        worker_peer_id: local_peer.peer_id.clone(),
                    }),
                    Err(err) => DistributedTransportPayload::Error {
                        message: format!("distributed job execution failed: {err}"),
                    },
                }
            }
            DistributedTransportPayload::Ack { correlation_id } => {
                DistributedTransportPayload::Ack { correlation_id }
            }
            DistributedTransportPayload::Error { message } => {
                DistributedTransportPayload::Error { message }
            }
            other => DistributedTransportPayload::Error {
                message: format!(
                    "swarm {} cannot handle payload in sync loop: {other:?}",
                    swarm_identifier
                ),
            },
        };

        Ok(
            DistributedTransportMessage::new(
                local_peer.peer_id.clone(),
                Some(source.peer_id.clone()),
                payload,
            )
            .with_correlation_id(incoming.message_id),
        )
    }

    pub fn config(&self) -> &Libp2pTransportConfig {
        &self.config
    }

    pub fn bootstrap_peer_records(&self) -> Vec<TransportPeerRecord> {
        self.config
            .bootstrap_peers
            .iter()
            .map(|peer| TransportPeerRecord {
                descriptor: RemotePeerDescriptor {
                    peer_id: peer.peer_id.clone(),
                    platform: "unknown".to_string(),
                    accelerator: None,
                    transport: "libp2p".to_string(),
                },
                addresses: vec![peer.address.clone()],
                average_rtt_ms: None,
                last_seen_unix_ms: None,
                is_reachable: false,
            })
            .collect()
    }

    pub fn swarm_identifier(&self) -> String {
        self.config.swarm_identifier()
    }

    pub fn discovery_namespace(&self) -> String {
        self.config.local_discovery_namespace()
    }

    pub fn protocol_scope(&self) -> String {
        self.config.scoped_protocol_name()
    }

    fn upsert_known_peer(&self, record: TransportPeerRecord) {
        if let Ok(mut peers) = self.known_peers.lock() {
            if let Some(existing) = peers
                .iter_mut()
                .find(|existing| existing.descriptor.peer_id == record.descriptor.peer_id)
            {
                *existing = record;
            } else {
                peers.push(record);
            }
        }
    }

    fn snapshot_known_peers(&self) -> Vec<TransportPeerRecord> {
        self.known_peers
            .lock()
            .map(|peers| peers.clone())
            .unwrap_or_default()
    }

    /// Handles one transport message as a swarm-loop iteration.
    ///
    /// This method is intentionally synchronous so it can be invoked by unit tests
    /// and by the current transport implementation before the async swarm reactor
    /// is fully wired.
    pub fn run_swarm_loop_once(
        &self,
        source: &RemotePeerDescriptor,
        incoming: DistributedTransportMessage,
    ) -> Result<DistributedTransportMessage, DistributedTensorError> {
        Self::process_swarm_message(
            &self.local_peer,
            &self.config,
            &self.known_peers,
            source,
            incoming,
        )
    }
}

fn execute_distributed_job_request(
    _local_peer: &RemotePeerDescriptor,
    request: super::transport::DistributedJobRequestEnvelope,
) -> Result<DistributedTensorJobResult, DistributedTensorError> {
    let backend_kind = selected_server_backend_kind();

    match request.job {
        super::job::DistributedTensorJob::TensorOp(op_request) => {
            let tensor = execute_remote_tensor_op(backend_kind, op_request.operation)?;
            Ok(DistributedTensorJobResult::Tensor(tensor))
        }
        super::job::DistributedTensorJob::FeatureStackForward(req) => {
            let features = execute_feature_stack_forward(backend_kind, req)?;
            Ok(DistributedTensorJobResult::FeatureBatch(features))
        }
    }
}

trait JobRequestEnvelopeVersionExt {
    fn parameter_version_for_stub(&self) -> Option<u64>;
}

impl JobRequestEnvelopeVersionExt for super::transport::DistributedJobRequestEnvelope {

    fn parameter_version_for_stub(&self) -> Option<u64> {
        match &self.job {
            super::job::DistributedTensorJob::TensorOp(request) => request.parameter_version,
            super::job::DistributedTensorJob::FeatureStackForward(request) => request.parameter_version,
        }
    }

}

impl DistributedTransport for Libp2pTransport {

    fn kind(&self) -> DistributedTransportKind {
        DistributedTransportKind::Libp2p
    }

    fn protocol_name(&self) -> &str {
        self.config.protocol_name.as_str()
    }

    fn local_peer(&self) -> &RemotePeerDescriptor {
        &self.local_peer
    }

    fn announce_capabilities(
        &self,
        capabilities: DistributedExecutorCapabilities,
    ) -> Result<(), DistributedTensorError> {
        let announcement = super::transport::TransportCapabilitiesAnnouncement {
            peer: self.local_peer.clone(),
            executor: capabilities,
            protocol_name: self.config.scoped_protocol_name(),
            protocol_version: DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
        };

        self.upsert_known_peer(TransportPeerRecord {
            descriptor: self.local_peer.clone(),
            addresses: self.config.listen_addresses.clone(),
            average_rtt_ms: Some(0),
            last_seen_unix_ms: None,
            is_reachable: true,
        });

        let self_peer = self.local_peer.clone();
        let msg = DistributedTransportMessage::new(
            self.local_peer.peer_id.clone(),
            Some(self.local_peer.peer_id.clone()),
            DistributedTransportPayload::CapabilitiesAnnouncement(announcement),
        );
        let _ = self.send_message(&self_peer, msg)?;
        Ok(())
    }

    fn discover_peers(&self) -> Result<Vec<TransportPeerRecord>, DistributedTensorError> {

        let mut peers = self.bootstrap_peer_records();

        if let Ok(registry) = local_swarm_registry().lock()
            && let Some(swarm_peers) = registry.get(&self.swarm_identifier())
        {
            peers.extend(swarm_peers.keys().filter(|id| *id != &self.local_peer.peer_id).map(|peer_id| {
                TransportPeerRecord {
                    descriptor: RemotePeerDescriptor {
                        peer_id: peer_id.clone(),
                        platform: "local".to_string(),
                        accelerator: None,
                        transport: "libp2p-local".to_string(),
                    },
                    addresses: vec![self.discovery_namespace()],
                    average_rtt_ms: Some(0),
                    last_seen_unix_ms: None,
                    is_reachable: true,
                }
            }));
        }
        
        peers.extend(self.snapshot_known_peers());
        Ok(peers)

    }

    fn send_message(
        &self,
        peer: &RemotePeerDescriptor,
        message: DistributedTransportMessage,
    ) -> Result<DistributedTransportMessage, DistributedTensorError> {
        if peer.peer_id == self.local_peer.peer_id {
            return self.dispatch_message_to_reactor(self.reactor_tx.clone(), peer, message);
        }

        if let Some(remote_reactor) = self.resolve_local_reactor(&peer.peer_id)? {
            return self.dispatch_message_to_reactor(remote_reactor, peer, message);
        }

        Err(DistributedTensorError::Transport(format!(
            "libp2p swarm loop {} has no remote reactor attached for peer {}",
            self.swarm_identifier(),
            peer.peer_id
        )))
    }

}

impl Drop for Libp2pTransport {

    fn drop(&mut self) {
        Self::unregister_local_reactor(&self.swarm_identifier(), &self.local_peer.peer_id);
        let _ = self.reactor_tx.send(ReactorCommand::Stop);
        if let Ok(mut join) = self.reactor_join.lock()
            && let Some(handle) = join.take()
        {
            let _ = handle.join();
        }
    }

}

#[cfg(test)]
mod tests {
    
    use super::*;
    use crate::distributed::{
        DistributedJobRequestEnvelope,
        DistributedTensorJob,
        DistributedTransportPayload,
        RemoteFeatureStackForwardRequest,
    };
    use crate::tensor::tensor4d::Tensor4D;

    #[test]
    fn libp2p_config_rejects_invalid_address() {
        let config = Libp2pTransportConfig {
            listen_addresses: vec!["not-a-multiaddr".to_string()],
            ..Libp2pTransportConfig::default()
        };

        let error = config.validate().expect_err("invalid address should fail");
        assert!(matches!(error, DistributedTensorError::Transport(_)));
    }

    #[test]
    fn libp2p_config_includes_named_versioned_swarm_scope() {
        let config = Libp2pTransportConfig {
            swarm_name: "training-cluster".to_string(),
            swarm_version: "v2026.06".to_string(),
            ..Libp2pTransportConfig::default()
        };

        assert_eq!(config.swarm_identifier(), "training-cluster@v2026.06");
        assert_eq!(
            config.local_discovery_namespace(),
            "/neuralnet/swarm/training-cluster/v2026.06"
        );
        assert_eq!(
            config.scoped_protocol_name(),
            "/neuralnet/distributed/1.0.0/training-cluster/v2026.06"
        );
    }

    #[test]
    fn loopback_swarm_loop_responds_to_health_and_job_requests() {
        let peer = RemotePeerDescriptor {
            peer_id: "peer-local".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };
        let transport = Libp2pTransport::new(peer.clone(), Libp2pTransportConfig::default())
            .expect("config should be valid");

        let health_request = DistributedTransportMessage::new(
            peer.peer_id.clone(),
            Some(peer.peer_id.clone()),
            DistributedTransportPayload::HealthCheck,
        );
        let health_response = transport
            .run_swarm_loop_once(&peer, health_request)
            .expect("health response should be returned");
        assert!(matches!(
            health_response.payload,
            DistributedTransportPayload::HealthStatus { available: true }
        ));

        let job = DistributedTensorJob::FeatureStackForward(RemoteFeatureStackForwardRequest {
            input: Tensor4D::zeros(1, 1, 2, 2),
            blocks: Vec::new(),
            parameter_version: Some(99),
        });
        let job_request = DistributedTransportMessage::new(
            peer.peer_id.clone(),
            Some(peer.peer_id.clone()),
            DistributedTransportPayload::JobRequest(DistributedJobRequestEnvelope {
                job,
                timeout_ms: Some(500),
                require_ack: true,
            }),
        );
        let job_response = transport
            .run_swarm_loop_once(&peer, job_request)
            .expect("job response should be returned");

        assert!(matches!(
            job_response.payload,
            DistributedTransportPayload::JobResult(_)
        ));
    }

    #[test]
    fn send_message_routes_through_background_reactor() {
        let peer = RemotePeerDescriptor {
            peer_id: "peer-local".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };
        let transport = Libp2pTransport::new(peer.clone(), Libp2pTransportConfig::default())
            .expect("config should be valid");

        let message = DistributedTransportMessage::new(
            peer.peer_id.clone(),
            Some(peer.peer_id.clone()),
            DistributedTransportPayload::HealthCheck,
        );
        let response = transport
            .send_message(&peer, message)
            .expect("reactor should return health response");

        assert!(matches!(
            response.payload,
            DistributedTransportPayload::HealthStatus { available: true }
        ));
    }

    #[test]
    fn capabilities_announcement_is_scoped_to_named_swarm() {
        let peer = RemotePeerDescriptor {
            peer_id: "peer-local".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };
        let transport = Libp2pTransport::new(peer.clone(), Libp2pTransportConfig::default())
            .expect("config should be valid");

        let mismatched_announcement = crate::distributed::TransportCapabilitiesAnnouncement {
            peer: RemotePeerDescriptor {
                peer_id: "peer-remote".to_string(),
                platform: "linux".to_string(),
                accelerator: None,
                transport: "libp2p".to_string(),
            },
            executor: DistributedExecutorCapabilities {
                supported_work_units: vec![crate::distributed::DistributedWorkUnitKind::FeatureStackForward],
                supports_parameter_push: true,
                supports_parameter_pull: false,
                supports_result_streaming: false,
                supports_eventual_sync: true,
                max_tensor_elements: Some(1024),
            },
            protocol_name: "/neuralnet/distributed/1.0.0/other-swarm/v9".to_string(),
            protocol_version: DISTRIBUTED_TRANSPORT_PROTOCOL_VERSION,
        };

        let response = transport
            .run_swarm_loop_once(
                &peer,
                DistributedTransportMessage::new(
                    peer.peer_id.clone(),
                    Some(peer.peer_id.clone()),
                    DistributedTransportPayload::CapabilitiesAnnouncement(mismatched_announcement),
                ),
            )
            .expect("loop should respond");

        assert!(matches!(
            response.payload,
            DistributedTransportPayload::Error { .. }
        ));
    }

    #[test]
    fn local_swarm_event_loop_routes_between_two_peers() {
        let config = Libp2pTransportConfig {
            swarm_name: "cluster-a".to_string(),
            swarm_version: "v1".to_string(),
            ..Libp2pTransportConfig::default()
        };

        let peer_a = RemotePeerDescriptor {
            peer_id: "peer-a".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };
        let peer_b = RemotePeerDescriptor {
            peer_id: "peer-b".to_string(),
            platform: "linux".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };

        let transport_a = Libp2pTransport::new(peer_a.clone(), config.clone())
            .expect("config should be valid");
        let _transport_b = Libp2pTransport::new(peer_b.clone(), config)
            .expect("config should be valid");

        let response = transport_a
            .send_message(
                &peer_b,
                DistributedTransportMessage::new(
                    peer_a.peer_id.clone(),
                    Some(peer_b.peer_id.clone()),
                    DistributedTransportPayload::HealthCheck,
                ),
            )
            .expect("message should route through local swarm registry");

        assert!(matches!(
            response.payload,
            DistributedTransportPayload::HealthStatus { available: true }
        ));
    }

    #[test]
    fn local_discovery_is_isolated_by_swarm_version() {
        let peer_a = RemotePeerDescriptor {
            peer_id: "peer-a2".to_string(),
            platform: "macos".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };
        let peer_b = RemotePeerDescriptor {
            peer_id: "peer-b2".to_string(),
            platform: "linux".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };

        let transport_a = Libp2pTransport::new(
            peer_a.clone(),
            Libp2pTransportConfig {
                swarm_name: "cluster-b".to_string(),
                swarm_version: "v1".to_string(),
                ..Libp2pTransportConfig::default()
            },
        )
        .expect("config should be valid");
        let _transport_b = Libp2pTransport::new(
            peer_b.clone(),
            Libp2pTransportConfig {
                swarm_name: "cluster-b".to_string(),
                swarm_version: "v2".to_string(),
                ..Libp2pTransportConfig::default()
            },
        )
        .expect("config should be valid");

        let error = transport_a
            .send_message(
                &peer_b,
                DistributedTransportMessage::new(
                    peer_a.peer_id.clone(),
                    Some(peer_b.peer_id.clone()),
                    DistributedTransportPayload::HealthCheck,
                ),
            )
            .expect_err("different swarm versions should not route locally");

        assert!(matches!(error, DistributedTensorError::Transport(_)));
    }
}