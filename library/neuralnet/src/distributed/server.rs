use std::thread;
use std::time::{Duration, Instant};

use super::error::DistributedTensorError;
use super::libp2p::{Libp2pTransport, Libp2pTransportConfig};
use super::policy::{
    DistributedExecutorCapabilities,
    RemotePeerDescriptor,
};
use super::transport::{DistributedTransport, TransportPeerRecord};

#[derive(Debug, Clone)]
pub struct DistributedServerConfig {
    pub local_peer: RemotePeerDescriptor,
    pub transport: Libp2pTransportConfig,
    pub capabilities: DistributedExecutorCapabilities,
    pub announce_capabilities: bool,
}

impl DistributedServerConfig {
    pub fn with_defaults(local_peer: RemotePeerDescriptor) -> Self {
        Self {
            local_peer,
            transport: Libp2pTransportConfig::default(),
            capabilities: DistributedExecutorCapabilities::default(),
            announce_capabilities: true,
        }
    }
}

#[derive(Debug)]
pub struct DistributedServerRuntime {
    transport: Libp2pTransport,
    local_peer: RemotePeerDescriptor,
    started_at: Instant,
}

impl DistributedServerRuntime {

    pub fn start(config: DistributedServerConfig) -> Result<Self, DistributedTensorError> {
        let transport = Libp2pTransport::new(config.local_peer.clone(), config.transport)?;

        if config.announce_capabilities {
            transport.announce_capabilities(config.capabilities)?;
        }

        Ok(Self {
            transport,
            local_peer: config.local_peer,
            started_at: Instant::now(),
        })
    }

    pub fn local_peer(&self) -> &RemotePeerDescriptor {
        &self.local_peer
    }

    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn discovered_peers(&self) -> Result<Vec<TransportPeerRecord>, DistributedTensorError> {
        self.transport.discover_peers()
    }

    pub fn await_events_for(&self, duration: Duration) {
        thread::sleep(duration);
    }

    pub fn run_until<F>(&self, poll_interval: Duration, should_stop: F)
    where
        F: Fn() -> bool,
    {
        while !should_stop() {
            thread::sleep(poll_interval);
        }
    }
    
}