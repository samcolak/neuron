use std::env;
use std::thread;
use std::time::Duration;
use std::sync::OnceLock;
use uuid::Uuid;

use crate::distributed::{
    DistributedExecutionPolicy,
    DistributedExecutorCapabilities,
    DistributedTensorExecutor,
    DistributedTensorJob,
    DistributedTensorJobResult,
    DistributedTransport,
    DistributedWorkUnitKind,
    Libp2pTransport,
    Libp2pTransportConfig,
    RemoteFeatureStackForwardRequest,
    RemotePeerDescriptor,
    RemoteTensorOp,
    RemoteTensorOpRequest,
    TransportBackedDistributedExecutor,
};
use crate::tensor::backend::TensorBackend;
use crate::tensor::device::BackendTrainingCapabilities;
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[derive(Debug, Clone, Copy)]
pub struct DistributedTensorBackend;

pub fn distributed_backend() -> DistributedTensorBackend {
    DistributedTensorBackend
}

pub fn distributed_backend_available() -> bool {
    distributed_executor().is_some()
}

impl TensorBackend for DistributedTensorBackend {
    fn name(&self) -> &'static str {
        "distributed"
    }

    fn training_capabilities(&self) -> BackendTrainingCapabilities {
        BackendTrainingCapabilities::native_compute_host_training()
    }

    fn conv2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        execute_tensor_job(DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::Conv2dValid {
                input: input.clone(),
                kernels: kernels.clone(),
                bias: bias.map(|b| b.to_vec()),
                stride_h,
                stride_w,
            },
            parameter_version: None,
        }))
    }

    fn max_pool2d(
        &self,
        input: &Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        execute_tensor_job(DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::MaxPool2d {
                input: input.clone(),
                window_h,
                window_w,
                stride_h,
                stride_w,
            },
            parameter_version: None,
        }))
    }

    fn global_average_pool2d(&self, input: &Tensor4D) -> Result<Tensor4D, TensorError> {
        execute_tensor_job(DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::GlobalAveragePool2d {
                input: input.clone(),
            },
            parameter_version: None,
        }))
    }

    fn relu_inplace(&self, input: &mut Tensor4D) {
        let result = execute_tensor_job(DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::Relu {
                input: input.clone(),
            },
            parameter_version: None,
        }));

        if let Ok(next) = result {
            *input = next;
        } else {
            input.relu_inplace_cpu();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_relu_max_pool2d_valid(
        &self,
        input: &Tensor4D,
        kernels: &Tensor4D,
        bias: Option<&[f32]>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    ) -> Result<Tensor4D, TensorError> {
        execute_tensor_job(DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::ConvReluMaxPool2dValid {
                input: input.clone(),
                kernels: kernels.clone(),
                bias: bias.map(|b| b.to_vec()),
                conv_stride_h,
                conv_stride_w,
                pool_window_h,
                pool_window_w,
                pool_stride_h,
                pool_stride_w,
            },
            parameter_version: None,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_blocks_to_feature_vec(
        &self,
        input: &Tensor4D,
        block1_kernels: &Tensor4D,
        block1_bias: &[f32],
        block2: Option<(&Tensor4D, &[f32])>,
        _conv_stride_h: usize,
        _conv_stride_w: usize,
        _pool_window_h: usize,
        _pool_window_w: usize,
        _pool_stride_h: usize,
        _pool_stride_w: usize,
    ) -> Result<Vec<f32>, TensorError> {
        let mut blocks = vec![crate::distributed::RemoteConvBlockDescriptor {
            kernels: block1_kernels.clone(),
            bias: block1_bias.to_vec(),
        }];
        if let Some((k2, b2)) = block2 {
            blocks.push(crate::distributed::RemoteConvBlockDescriptor {
                kernels: k2.clone(),
                bias: b2.to_vec(),
            });
        }

        let result = execute_distributed_job(DistributedTensorJob::FeatureStackForward(
            RemoteFeatureStackForwardRequest {
                input: input.clone(),
                blocks,
                parameter_version: None,
            },
        ))?;

        match result {
            DistributedTensorJobResult::FeatureBatch(mut batch) => {
                Ok(batch.pop().unwrap_or_default())
            }
            DistributedTensorJobResult::Tensor(tensor) => Ok(tensor.first_sample_features()),
            DistributedTensorJobResult::RemoteTensor(_) => Err(TensorError::InvalidArgument(
                "distributed feature vector request returned remote tensor ref",
            )),
        }
    }
}

type DistributedExecutor = TransportBackedDistributedExecutor<Libp2pTransport>;

fn first_non_empty_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn distributed_executor() -> Option<&'static DistributedExecutor> {
    static EXECUTOR: OnceLock<Option<DistributedExecutor>> = OnceLock::new();
    EXECUTOR.get_or_init(build_distributed_executor).as_ref()
}

fn distributed_target_peer() -> RemotePeerDescriptor {
    let peer_id = first_non_empty_env(&["NEURALNET_DISTRIBUTED_TARGET_PEER"])
        .or_else(|| {
            first_non_empty_env(&["NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS"])
                .and_then(|value| {
                    let peers = parse_bootstrap_peers(value.as_str());
                    if peers.len() == 1 {
                        peers.first().map(|peer| peer.peer_id.clone())
                    } else {
                        None
                    }
                })
        })
        .unwrap_or_else(|| "auto-discover-target".to_string());

    RemotePeerDescriptor {
        peer_id,
        platform: env::var("NEURALNET_DISTRIBUTED_TARGET_PLATFORM")
            .ok()
            .unwrap_or_else(|| "unknown".to_string()),
        accelerator: env::var("NEURALNET_DISTRIBUTED_TARGET_ACCELERATOR").ok(),
        transport: "libp2p".to_string(),
    }
}

fn distributed_local_peer() -> RemotePeerDescriptor {
    RemotePeerDescriptor {
        peer_id: env::var("NEURALNET_DISTRIBUTED_LOCAL_PEER")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(default_local_peer_id),
        platform: env::var("NEURALNET_DISTRIBUTED_LOCAL_PLATFORM")
            .ok()
            .unwrap_or_else(|| std::env::consts::OS.to_string()),
        accelerator: env::var("NEURALNET_DISTRIBUTED_LOCAL_ACCELERATOR").ok(),
        transport: "libp2p".to_string(),
    }
}

fn default_local_peer_id() -> String {
    static GENERATED_ID: OnceLock<String> = OnceLock::new();
    GENERATED_ID
        .get_or_init(|| {
            let compact = Uuid::now_v7().to_string().replace('-', "");
            format!("neuralnet-client-{compact}")
        })
        .clone()
}

fn build_distributed_executor() -> Option<DistributedExecutor> {
    let local_peer = distributed_local_peer();
    let mut config = Libp2pTransportConfig::default();

    if let Some(name) = first_non_empty_env(&[
        "NEURALNET_DISTRIBUTED_SWARM_NAME",
        "NEURALNET_SWARM_NAME",
    ]) {
        config.swarm_name = name;
    }
    if let Some(version) = first_non_empty_env(&[
        "NEURALNET_DISTRIBUTED_SWARM_VERSION",
        "NEURALNET_SWARM_VERSION",
    ]) {
        config.swarm_version = version;
    }
    if let Ok(timeout) = env::var("NEURALNET_DISTRIBUTED_TIMEOUT_MS")
        && let Ok(parsed) = timeout.parse::<u64>()
    {
        config.request_timeout_ms = parsed.max(1);
    }
    if let Ok(bootstrap) = env::var("NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS") {
        config.bootstrap_peers = parse_bootstrap_peers(bootstrap.as_str());
    }

    let transport = Libp2pTransport::new(local_peer, config).ok()?;

    // Give mDNS/identify a short window to discover remote peers before the
    // first distributed tensor op, which avoids immediate CPU failover.
    warm_up_peer_discovery(&transport);

    let mut policy = DistributedExecutionPolicy::coordinator_default();
    policy.timeout_ms = transport.config().request_timeout_ms;

    let capabilities = DistributedExecutorCapabilities {
        supported_work_units: vec![
            DistributedWorkUnitKind::TensorOp,
            DistributedWorkUnitKind::FeatureStackForward,
        ],
        supports_parameter_push: true,
        supports_parameter_pull: false,
        supports_result_streaming: false,
        supports_eventual_sync: true,
        max_tensor_elements: None,
    };

    Some(TransportBackedDistributedExecutor::new(
        policy,
        capabilities,
        transport,
    ))
}

fn warm_up_peer_discovery(transport: &Libp2pTransport) {
    let wait_ms = env::var("NEURALNET_DISTRIBUTED_DISCOVERY_WAIT_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(1_500);

    if wait_ms == 0 {
        return;
    }

    let local_peer_id = transport.local_peer().peer_id.clone();
    let step_ms = 100u64;
    let mut waited = 0u64;
    while waited < wait_ms {
        if let Ok(peers) = transport.discover_peers() {
            let found_remote = peers
                .iter()
                .any(|peer| peer.descriptor.peer_id != local_peer_id);
            if found_remote {
                return;
            }
        }

        thread::sleep(Duration::from_millis(step_ms));
        waited = waited.saturating_add(step_ms);
    }
}

fn parse_bootstrap_peers(value: &str) -> Vec<crate::distributed::Libp2pBootstrapPeer> {
    value
        .split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return None;
            }

            let (peer_id, address) = trimmed.split_once('@')?;
            let peer_id = peer_id.trim();
            let address = address.trim();
            if peer_id.is_empty() || address.is_empty() {
                return None;
            }

            Some(crate::distributed::Libp2pBootstrapPeer {
                peer_id: peer_id.to_string(),
                address: address.to_string(),
            })
        })
        .collect()
}

fn execute_distributed_job(job: DistributedTensorJob) -> Result<DistributedTensorJobResult, TensorError> {
    let executor = distributed_executor().ok_or(TensorError::InvalidArgument(
        "distributed backend is not initialized",
    ))?;

    let target_peer = resolve_distributed_target_peer(executor.transport())
        .map_err(|err| TensorError::InvalidArgument(Box::leak(format!(
            "distributed backend target resolution failed: {err}"
        ).into_boxed_str())))?;

    // eprintln!(
    //     "[distributed][client] target peer={}",
    //     target_peer.peer_id
    // );

    executor
        .execute(&target_peer, job)
        .map_err(|err| TensorError::InvalidArgument(Box::leak(format!("distributed backend error: {err}").into_boxed_str())))
}

fn resolve_distributed_target_peer(
    transport: &Libp2pTransport,
) -> Result<RemotePeerDescriptor, String> {
    let explicit_target = first_non_empty_env(&["NEURALNET_DISTRIBUTED_TARGET_PEER"]);
    let local_peer_id = transport.local_peer().peer_id.clone();

    if let Some(peer_id) = explicit_target {
        if peer_id != "auto-discover-target" {
            return Ok(RemotePeerDescriptor {
                peer_id,
                platform: env::var("NEURALNET_DISTRIBUTED_TARGET_PLATFORM")
                    .ok()
                    .unwrap_or_else(|| "unknown".to_string()),
                accelerator: env::var("NEURALNET_DISTRIBUTED_TARGET_ACCELERATOR").ok(),
                transport: "libp2p".to_string(),
            });
        }
    }

    let discovered = transport
        .discover_peers()
        .map_err(|err| format!("peer discovery failed: {err}"))?;

    let candidates = discovered
        .into_iter()
        .filter(|peer| peer.descriptor.peer_id != local_peer_id)
        .collect::<Vec<_>>();

    if candidates.len() == 1 {
        let peer = &candidates[0].descriptor;
        return Ok(peer.clone());
    }

    if candidates.is_empty() {
        eprintln!("[distributed][client] waiting for peer discovery");
        return Err("no remote peers discovered; set NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS or wait for mDNS discovery".to_string());
    }

    Err(format!(
        "multiple remote peers discovered ({count}); set NEURALNET_DISTRIBUTED_TARGET_PEER explicitly",
        count = candidates.len()
    ))
}

fn execute_tensor_job(job: DistributedTensorJob) -> Result<Tensor4D, TensorError> {
    match execute_distributed_job(job)? {
        DistributedTensorJobResult::Tensor(tensor) => Ok(tensor),
        DistributedTensorJobResult::FeatureBatch(_) => Err(TensorError::InvalidArgument(
            "distributed tensor op returned feature batch",
        )),
        DistributedTensorJobResult::RemoteTensor(_) => Err(TensorError::InvalidArgument(
            "distributed tensor op returned remote tensor ref",
        )),
    }
}
