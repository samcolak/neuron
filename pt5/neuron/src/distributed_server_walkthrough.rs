use std::env;
use std::time::Duration;

use neuralnet::distributed::{
    DistributedExecutionPolicy,
    DistributedExecutorCapabilities,
    DistributedServerConfig,
    DistributedServerRuntime,
    DistributedTensorExecutor,
    DistributedTensorJob,
    Libp2pTransport,
    Libp2pTransportConfig,
    RemotePeerDescriptor,
    RemoteTensorOp,
    RemoteTensorOpRequest,
    TransportBackedDistributedExecutor,
};
use neuralnet::tensor::tensor4d::Tensor4D;

fn parse_u64_env(var_name: &str, default: u64) -> u64 {
    env::var(var_name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_bool_env(var_name: &str, default: bool) -> bool {
    env::var(var_name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn parse_bootstrap_peers(value: &str) -> Vec<neuralnet::distributed::Libp2pBootstrapPeer> {
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

            Some(neuralnet::distributed::Libp2pBootstrapPeer {
                peer_id: peer_id.to_string(),
                address: address.to_string(),
            })
        })
        .collect()
}

fn run_impl() -> Result<(), String> {

    println!("\nDistributed server walkthrough");

    let mut transport = Libp2pTransportConfig::default();
    if let Some(name) = env_first(&["NEURALNET_DISTRIBUTED_SWARM_NAME", "NEURALNET_SWARM_NAME"]) {
        transport.swarm_name = name;
    }
    if let Some(version) = env_first(&["NEURALNET_DISTRIBUTED_SWARM_VERSION", "NEURALNET_SWARM_VERSION"]) {
        transport.swarm_version = version;
    }
    let swarm_name = transport.swarm_name.clone();
    let swarm_version = transport.swarm_version.clone();
    let transport = if let Some(bootstrap) = env_first(&["NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS"]) {
        Libp2pTransportConfig {
            bootstrap_peers: parse_bootstrap_peers(bootstrap.as_str()),
            ..transport
        }
    } else {
        transport
    };

    let server_peer = RemotePeerDescriptor {
        peer_id: env_first(&[
            "NEURALNET_DISTRIBUTED_LOCAL_PEER",
            "NEURON_DISTRIBUTED_SERVER_PEER_ID",
        ])
        .unwrap_or_else(|| "neuron-server-1".to_string()),
        platform: env_first(&["NEURALNET_DISTRIBUTED_LOCAL_PLATFORM"])
            .unwrap_or_else(|| "app-server".to_string()),
        accelerator: env_first(&["NEURALNET_DISTRIBUTED_LOCAL_ACCELERATOR"]),
        transport: "libp2p".to_string(),
    };

    let runtime = DistributedServerRuntime::start(DistributedServerConfig {
        local_peer: server_peer.clone(),
        transport: transport.clone(),
        capabilities: DistributedExecutorCapabilities::default(),
        announce_capabilities: true,
    })
    .map_err(|error| format!("failed to start distributed server runtime: {error}"))?;

    println!(
        "  server peer={} swarm={} version={}",
        runtime.local_peer().peer_id,
        swarm_name,
        swarm_version
    );
    println!("  discovered peers: {:?}", runtime.discovered_peers());

    let run_self_test = parse_bool_env("NEURON_DISTRIBUTED_SERVER_SELF_TEST", false);
    if run_self_test {
        let client_peer = RemotePeerDescriptor {
            peer_id: env_first(&[
                "NEURALNET_DISTRIBUTED_CLIENT_PEER",
                "NEURON_DISTRIBUTED_CLIENT_PEER_ID",
            ])
            .unwrap_or_else(|| "neuron-client-1".to_string()),
            platform: "app-client".to_string(),
            accelerator: None,
            transport: "libp2p".to_string(),
        };

        let client_transport = Libp2pTransport::new(client_peer, transport)
            .map_err(|error| format!("failed to start distributed client transport: {error}"))?;

        let executor = TransportBackedDistributedExecutor::new(
            DistributedExecutionPolicy::coordinator_default(),
            DistributedExecutorCapabilities::default(),
            client_transport,
        );

        let input = Tensor4D::from_vec(1, 1, 1, 4, vec![-2.0, -0.5, 0.25, 3.0])
            .map_err(|error| format!("failed to build sample tensor: {error}"))?;
        let job = DistributedTensorJob::TensorOp(RemoteTensorOpRequest {
            operation: RemoteTensorOp::Relu { input },
            parameter_version: Some(1),
        });

        let result = executor
            .execute(&server_peer, job)
            .map_err(|error| format!("distributed self-test execution failed: {error}"))?;
        println!("  self-test result={result:?}");
    }

    let wait_secs = parse_u64_env("NEURON_DISTRIBUTED_SERVER_WAIT_SECS", 0);
    if wait_secs == 0 {
        println!("  server is running (press Ctrl+C to stop)");
        loop {
            runtime.await_events_for(Duration::from_secs(1));
        }
    }

    println!("  awaiting external events for {wait_secs}s...");
    runtime.await_events_for(Duration::from_secs(wait_secs));
    println!("  server uptime: {:?}", runtime.uptime());

    Ok(())
    
}

pub fn run_distributed_server_walkthrough() {
    if let Err(error) = run_impl() {
        eprintln!("Distributed server walkthrough failed: {error}");
    }
}

pub fn has_distributed_server_flag(args: &[String]) -> bool {
    args
        .iter()
        .any(|arg| arg == "--distributed-server" || arg == "--p2p")
}

pub fn has_help_flag(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--help" || arg == "-h")
}

pub fn print_distributed_server_help() {
    println!("  --p2p                 Start distributed server walkthrough mode");
    println!("  --distributed-server  Alias for --p2p");
}