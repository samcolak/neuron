use crate::helpers::controllers::textnode_controller::TextNodeController;
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::text_dendrite::TextDendrite;

use serde::{de::DeserializeOwned, Serialize};

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

type TextNeuralNetwork = NeuralNetwork<TextNodeController, TextDendrite>;

const NEURON_BIN_MAGIC: [u8; 4] = *b"NRN4";

fn load_network_from_file<C, N>(network: &mut NeuralNetwork<C, N>, filename: &str) -> bool
where
    C: crate::helpers::nodenet::NodeNetworkController,
    N: crate::helpers::nodenet::NetworkNode + Clone + Serialize + DeserializeOwned,
{
    if let Ok(bytes) = fs::read(filename) {
        if bytes.len() >= 4 && bytes[0..4] == NEURON_BIN_MAGIC
            && let Ok(loaded) = bincode::deserialize::<NeuralNetwork<C, N>>(&bytes[4..])
        {
            *network = loaded;
            network.rebuild_connection_indexes();
            network.rebuild_token_index();
            return true;
        }

        if let Ok(loaded) = serde_json::from_slice::<NeuralNetwork<C, N>>(&bytes) {
            *network = loaded;
            network.rebuild_connection_indexes();
            network.rebuild_token_index();
            return true;
        }
    }

    false
}

fn save_network_to_file<C, N>(network: &NeuralNetwork<C, N>, filename: &str)
where
    C: crate::helpers::nodenet::NodeNetworkController,
    N: crate::helpers::nodenet::NetworkNode + Clone + Serialize + DeserializeOwned,
{
    if let Ok(encoded) = bincode::serialize(network) {
        let mut bytes = Vec::with_capacity(NEURON_BIN_MAGIC.len() + encoded.len());
        bytes.extend_from_slice(&NEURON_BIN_MAGIC);
        bytes.extend_from_slice(&encoded);
        let _ = fs::write(filename, bytes);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TextNetworkStoreMetrics {
    pub buffered_mode: bool,
    pub flush_every: usize,
    pub flush_interval_ms: u64,
    pub persist_requests: u64,
    pub flush_writes: u64,
    pub forced_flush_writes: u64,
    pub load_attempts: u64,
    pub load_successes: u64,
    pub total_flush_bytes: u64,
    pub last_flush_bytes: u64,
    pub last_flush_latency_micros: u64,
    pub pending_writes: usize,
}

pub trait TextNetworkStore: Send + Sync {
    fn load_into(&self, network: &mut TextNeuralNetwork) -> bool;

    fn persist(&self, network: &TextNeuralNetwork);

    fn persist_force(&self, network: &TextNeuralNetwork) {
        self.persist(network);
    }

    fn metrics(&self) -> TextNetworkStoreMetrics {
        TextNetworkStoreMetrics::default()
    }
}

pub struct FileTextNetworkStore {
    filename: String,
    persist_requests: AtomicU64,
    flush_writes: AtomicU64,
    forced_flush_writes: AtomicU64,
    load_attempts: AtomicU64,
    load_successes: AtomicU64,
    total_flush_bytes: AtomicU64,
    last_flush_bytes: AtomicU64,
    last_flush_latency_micros: AtomicU64,
}

impl FileTextNetworkStore {
    pub fn from_filename(filename: String) -> Self {
        Self {
            filename,
            persist_requests: AtomicU64::new(0),
            flush_writes: AtomicU64::new(0),
            forced_flush_writes: AtomicU64::new(0),
            load_attempts: AtomicU64::new(0),
            load_successes: AtomicU64::new(0),
            total_flush_bytes: AtomicU64::new(0),
            last_flush_bytes: AtomicU64::new(0),
            last_flush_latency_micros: AtomicU64::new(0),
        }
    }

    fn persist_internal(&self, network: &TextNeuralNetwork, forced: bool) {
        self.persist_requests.fetch_add(1, Ordering::Relaxed);

        let start = Instant::now();
        save_network_to_file(network, &self.filename);
        let latency_micros = start.elapsed().as_micros() as u64;

        let bytes = fs::metadata(&self.filename)
            .map(|metadata| metadata.len())
            .unwrap_or(0);

        self.last_flush_latency_micros
            .store(latency_micros, Ordering::Relaxed);
        self.last_flush_bytes.store(bytes, Ordering::Relaxed);
        self.total_flush_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.flush_writes.fetch_add(1, Ordering::Relaxed);

        if forced {
            self.forced_flush_writes.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl TextNetworkStore for FileTextNetworkStore {
    fn load_into(&self, network: &mut TextNeuralNetwork) -> bool {
        self.load_attempts.fetch_add(1, Ordering::Relaxed);
        let loaded = load_network_from_file(network, &self.filename);
        if loaded {
            self.load_successes.fetch_add(1, Ordering::Relaxed);
        }
        loaded
    }

    fn persist(&self, network: &TextNeuralNetwork) {
        self.persist_internal(network, false);
    }

    fn persist_force(&self, network: &TextNeuralNetwork) {
        self.persist_internal(network, true);
    }

    fn metrics(&self) -> TextNetworkStoreMetrics {
        TextNetworkStoreMetrics {
            buffered_mode: false,
            flush_every: 1,
            flush_interval_ms: 0,
            persist_requests: self.persist_requests.load(Ordering::Relaxed),
            flush_writes: self.flush_writes.load(Ordering::Relaxed),
            forced_flush_writes: self.forced_flush_writes.load(Ordering::Relaxed),
            load_attempts: self.load_attempts.load(Ordering::Relaxed),
            load_successes: self.load_successes.load(Ordering::Relaxed),
            total_flush_bytes: self.total_flush_bytes.load(Ordering::Relaxed),
            last_flush_bytes: self.last_flush_bytes.load(Ordering::Relaxed),
            last_flush_latency_micros: self.last_flush_latency_micros.load(Ordering::Relaxed),
            pending_writes: 0,
        }
    }
}

pub struct BufferedFileTextNetworkStore {
    inner: FileTextNetworkStore,
    flush_every: usize,
    flush_interval: Duration,
    persist_requests: AtomicU64,
    pending_writes: AtomicUsize,
    last_flush: Mutex<Instant>,
}

impl BufferedFileTextNetworkStore {
    pub fn from_filename(filename: String) -> Self {
        let flush_every = env::var("NEURON_STORE_FLUSH_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(32);

        let flush_interval_ms = env::var("NEURON_STORE_FLUSH_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2_000);

        Self {
            inner: FileTextNetworkStore::from_filename(filename),
            flush_every,
            flush_interval: Duration::from_millis(flush_interval_ms),
            persist_requests: AtomicU64::new(0),
            pending_writes: AtomicUsize::new(0),
            last_flush: Mutex::new(Instant::now()),
        }
    }

    fn mark_flush_complete(&self) {
        self.pending_writes.store(0, Ordering::Relaxed);
        if let Ok(mut last) = self.last_flush.lock() {
            *last = Instant::now();
        }
    }
}

impl TextNetworkStore for BufferedFileTextNetworkStore {
    fn load_into(&self, network: &mut TextNeuralNetwork) -> bool {
        self.inner.load_into(network)
    }

    fn persist(&self, network: &TextNeuralNetwork) {
        self.persist_requests.fetch_add(1, Ordering::Relaxed);

        let pending = self.pending_writes.fetch_add(1, Ordering::Relaxed) + 1;
        let should_flush_by_count = pending >= self.flush_every;
        let should_flush_by_time = self
            .last_flush
            .lock()
            .map(|last| last.elapsed() >= self.flush_interval)
            .unwrap_or(false);

        if should_flush_by_count || should_flush_by_time {
            self.inner.persist(network);
            self.mark_flush_complete();
        }
    }

    fn persist_force(&self, network: &TextNeuralNetwork) {
        self.persist_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.persist_force(network);
        self.mark_flush_complete();
    }

    fn metrics(&self) -> TextNetworkStoreMetrics {
        let inner_metrics = self.inner.metrics();
        TextNetworkStoreMetrics {
            buffered_mode: true,
            flush_every: self.flush_every,
            flush_interval_ms: self.flush_interval.as_millis() as u64,
            persist_requests: self.persist_requests.load(Ordering::Relaxed),
            flush_writes: inner_metrics.flush_writes,
            forced_flush_writes: inner_metrics.forced_flush_writes,
            load_attempts: inner_metrics.load_attempts,
            load_successes: inner_metrics.load_successes,
            total_flush_bytes: inner_metrics.total_flush_bytes,
            last_flush_bytes: inner_metrics.last_flush_bytes,
            last_flush_latency_micros: inner_metrics.last_flush_latency_micros,
            pending_writes: self.pending_writes.load(Ordering::Relaxed),
        }
    }
}

fn sanitize_network_id(network_id: &str) -> String {
    network_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn store_filename_for_network(network_id: &str) -> String {
    let base = env::var("NEURON_STORE_FILE").unwrap_or_else(|_| "./neuron_store.nrn".to_string());

    if network_id == "default" {
        return base;
    }

    let sanitized_id = sanitize_network_id(network_id);

    if let Ok(dir) = env::var("NEURON_STORE_DIR") {
        let mut path = PathBuf::from(dir);
        path.push(format!("{}.nrn", sanitized_id));
        return path.to_string_lossy().to_string();
    }

    let base_path = Path::new(&base);
    let parent = base_path.parent().map(PathBuf::from).unwrap_or_default();
    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("neuron_store");
    let extension = base_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("nrn");

    let mut path = parent;
    path.push(format!("{}_{}.{}", stem, sanitized_id, extension));
    path.to_string_lossy().to_string()
}

pub fn build_text_network_store_for_network(network_id: &str) -> Box<dyn TextNetworkStore> {
    let filename = store_filename_for_network(network_id);

    match env::var("NEURON_STORE_MODE")
        .unwrap_or_else(|_| "sync".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "buffered" => Box::new(BufferedFileTextNetworkStore::from_filename(filename)),
        _ => Box::new(FileTextNetworkStore::from_filename(filename)),
    }
}