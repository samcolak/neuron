use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::collections::{HashMap, VecDeque};
use std::cell::RefCell;

use crate::cnn::feature_extractor::CnnFeatureExtractor;
use crate::core::nodenet::{NodeMetadata, NodeNetwork};
use crate::core::snapshot_writer::{
    latest_snapshot_write_error,
    submit_snapshot_write,
    wait_for_snapshot_write,
    write_snapshot_bytes_to_path,
};
use crate::core::transaction_log::{
    append_transaction,
    flush_transactions,
    latest_transaction_error,
    load_and_sanitize_transaction_log,
    truncate_transaction_log,
    TransactionOperation,
    TransactionTarget,
};
use crate::helpers::image_io::{ImageIoError, load_png_or_jpeg_from_path};
use crate::helpers::multimodal_controller::MultiModalInput;
use crate::helpers::multimodal_neuralnet::MultiModalSubNetwork;
use crate::helpers::pattern_classifier::{ClassificationResult, PatternClassifier};
use crate::dendrites::text_dendrite::DendriteType;
use crate::dendrites::multimodal_dendrite::MultimodalDendrite;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionStoreDecision {
    Store,
    DoNotStore,
    Defer,
}

#[derive(Debug, Clone)]
pub struct NetworkDump {
    pub name: &'static str,
    pub nodes: Vec<MultimodalDendrite>,
}

#[derive(Debug, Clone)]
pub struct BrainDump {
    pub cognitive: NetworkDump,
    pub memory: NetworkDump,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotLoadStatus {
    pub cognitive_loaded: bool,
    pub memory_loaded: bool,
    pub classifier_loaded: bool,
}

const BRAIN_SNAPSHOT_BIN_MAGIC_V1: [u8; 4] = *b"BSP1";

#[derive(Debug, Clone, Default)]
struct AutoSnapshotPolicy {
    instance_id: Option<String>,
    dir: Option<PathBuf>,
    every_n_inserts: usize,
    pending_inserts: usize,
    last_enqueued_generation: u64,
    worker_key: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct AutoTransactionLogPolicy {
    instance_id: Option<String>,
    dir: Option<PathBuf>,
    last_enqueued_sequence: u64,
    worker_key: Option<String>,
    last_error: Option<String>,
    replay_in_progress: bool,
}

#[derive(Debug, Clone, Default)]
struct QueryScoreCache {
    capacity: usize,
    map: HashMap<String, f64>,
    order: VecDeque<String>,
}

impl QueryScoreCache {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<f64> {
        let value = self.map.get(key).copied()?;
        self.touch_key(key);
        Some(value)
    }

    fn insert(&mut self, key: String, value: f64) {
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), value);
            self.touch_key(key.as_str());
            return;
        }

        self.map.insert(key.clone(), value);
        self.order.push_back(key);

        while self.map.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(oldest.as_str());
            }
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn touch_key(&mut self, key: &str) {
        if let Some(index) = self.order.iter().position(|existing| existing == key) {
            let _ = self.order.remove(index);
        }
        self.order.push_back(key.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainSnapshotBundleV1 {
    cognitive_net: MultiModalSubNetwork,
    memory_net: MultiModalSubNetwork,
    classifier: PatternClassifier,
}

#[derive(Debug, Clone)]
pub struct MultiModalBrain {
    cognitive_net: MultiModalSubNetwork,
    memory_net: MultiModalSubNetwork,
    classifier: PatternClassifier,
    image_feature_extractor: Option<CnnFeatureExtractor>,
    auto_snapshot: AutoSnapshotPolicy,
    transaction_log: AutoTransactionLogPolicy,
    query_cache_enabled: bool,
    query_score_cache: RefCell<QueryScoreCache>,
}

pub type MultiModalNeuralNetwork = MultiModalBrain;

impl MultiModalBrain {
    
    pub fn new_multimodal() -> Self {
        Self {
            cognitive_net: MultiModalSubNetwork::new_multimodal(),
            memory_net: MultiModalSubNetwork::new_multimodal(),
            classifier: PatternClassifier::new(),
            image_feature_extractor: None,
            auto_snapshot: AutoSnapshotPolicy::default(),
            transaction_log: AutoTransactionLogPolicy::default(),
            query_cache_enabled: false,
            query_score_cache: RefCell::new(QueryScoreCache::with_capacity(4_096)),
        }
    }

    pub fn enable_query_score_cache(&mut self, capacity: usize) {
        self.query_cache_enabled = true;
        self.query_score_cache = RefCell::new(QueryScoreCache::with_capacity(capacity));
    }

    pub fn disable_query_score_cache(&mut self) {
        self.query_cache_enabled = false;
        self.invalidate_query_cache();
    }

    pub fn enable_auto_snapshot(&mut self, instance_id: &str, every_n_inserts: usize) {
        self.auto_snapshot.instance_id = Some(instance_id.trim().to_string());
        self.auto_snapshot.dir = None;
        self.auto_snapshot.every_n_inserts = every_n_inserts.max(1);
        self.auto_snapshot.pending_inserts = 0;
        self.auto_snapshot.last_enqueued_generation = 0;
        self.auto_snapshot.worker_key = None;
        self.auto_snapshot.last_error = None;
        self.configure_transaction_log_for_instance(instance_id, None);
    }

    pub fn enable_auto_snapshot_in_dir(
        &mut self,
        instance_id: &str,
        dir: &Path,
        every_n_inserts: usize,
    ) {

        self.auto_snapshot.instance_id = Some(instance_id.trim().to_string());
        self.auto_snapshot.dir = Some(dir.to_path_buf());
        self.auto_snapshot.every_n_inserts = every_n_inserts.max(1);
        self.auto_snapshot.pending_inserts = 0;
        self.auto_snapshot.last_enqueued_generation = 0;
        self.auto_snapshot.worker_key = None;
        self.auto_snapshot.last_error = None;
        self.configure_transaction_log_for_instance(instance_id, Some(dir.to_path_buf()));
    }

    pub fn disable_auto_snapshot(&mut self) {
        self.auto_snapshot = AutoSnapshotPolicy::default();
        self.transaction_log = AutoTransactionLogPolicy::default();
    }

    fn invalidate_query_cache(&self) {
        if !self.query_cache_enabled {
            return;
        }
        let mut cache = self.query_score_cache.borrow_mut();
        cache.clear();
    }

    pub fn flush_auto_snapshot(&mut self) -> io::Result<bool> {
        if self.auto_snapshot.instance_id.is_none()
            || (self.auto_snapshot.pending_inserts == 0
                && self.auto_snapshot.last_enqueued_generation == 0)
        {
            return Ok(false);
        }

        self.flush_transaction_log()?;

        let Some((worker_key, bundle_path)) = self.auto_snapshot_target_paths() else {
            return Ok(false);
        };

        let generation = self.submit_snapshot_bundle_write(worker_key.clone(), bundle_path)?;
        let Some(active_worker_key) = self.auto_snapshot.worker_key.as_deref() else {
            return Err(io::Error::other(
                "snapshot worker key missing after enqueue",
            ));
        };

        wait_for_snapshot_write(active_worker_key, generation)?;
        self.auto_snapshot.last_error = latest_snapshot_write_error(active_worker_key);

        self.auto_snapshot.pending_inserts = 0;
        if self.auto_snapshot.last_error.is_none() {
            self.auto_snapshot.last_enqueued_generation = generation;
            if let Some((_, log_path)) = self.transaction_log_target_paths() {
                truncate_transaction_log(log_path.as_path())?;
                self.transaction_log.last_enqueued_sequence = 0;
            }
        }

        Ok(true)
    }

    pub fn auto_snapshot_pending_inserts(&self) -> usize {
        self.auto_snapshot.pending_inserts
    }

    pub fn auto_snapshot_last_error(&self) -> Option<&str> {
        self.auto_snapshot.last_error.as_deref()
    }

    fn configure_transaction_log_for_instance(&mut self, instance_id: &str, dir: Option<PathBuf>) {
        self.transaction_log.instance_id = Some(instance_id.trim().to_string());
        self.transaction_log.dir = dir;
        self.transaction_log.last_enqueued_sequence = 0;
        self.transaction_log.worker_key = None;
        self.transaction_log.last_error = None;
        self.transaction_log.replay_in_progress = false;
    }

    fn flush_transaction_log(&mut self) -> io::Result<bool> {

        if self.transaction_log.replay_in_progress {
            return Ok(false);
        }

        let Some(worker_key) = self.transaction_log.worker_key.as_deref() else {
            return Ok(false);
        };

        let target_sequence = self.transaction_log.last_enqueued_sequence;
        if target_sequence == 0 {
            return Ok(false);
        }

        flush_transactions(worker_key, target_sequence)?;
        self.transaction_log.last_error = latest_transaction_error(worker_key);
        Ok(true)

    }

    fn auto_snapshot_target_paths(&self) -> Option<(String, PathBuf)> {
        
        let instance_id = self.auto_snapshot.instance_id.as_deref()?;

        let bundle_path = if let Some(dir) = self.auto_snapshot.dir.as_deref() {
            snapshot_bundle_path_for_instance(instance_id, Some(dir))
        } else {
            snapshot_bundle_path_for_instance(instance_id, None)
        };

        let worker_key = bundle_path.to_string_lossy().to_string();
        Some((worker_key, bundle_path))

    }

    fn transaction_log_target_paths(&self) -> Option<(String, PathBuf)> {

        let instance_id = self.transaction_log.instance_id.as_deref()?;

        let log_path = if let Some(dir) = self.transaction_log.dir.as_deref() {
            transaction_log_path_for_instance(instance_id, Some(dir))
        } else {
            transaction_log_path_for_instance(instance_id, None)
        };

        let worker_key = log_path.to_string_lossy().to_string();
        Some((worker_key, log_path))

    }

    fn append_transaction_operation(&mut self, operation: TransactionOperation) {

        if self.transaction_log.replay_in_progress {
            return;
        }

        let Some((worker_key, log_path)) = self.transaction_log_target_paths() else {
            return;
        };

        let sequence = append_transaction(worker_key.clone(), log_path, operation);
        self.transaction_log.worker_key = Some(worker_key.clone());
        self.transaction_log.last_enqueued_sequence = sequence;
        self.transaction_log.last_error = latest_transaction_error(worker_key.as_str());

    }

    fn query_cache_key(content: &MultiModalInput) -> Option<String> {

        match content {
            
            MultiModalInput::Text(text) => {
                let normalized = text.trim().to_ascii_lowercase();
                if normalized.is_empty() {
                    None
                } else {
                    Some(format!("t:{normalized}"))
                }
            }

            MultiModalInput::FeatureTokens { modality, tokens } => {
                if tokens.is_empty() {
                    return None;
                }
                let mut key = String::from("f:");
                key.push_str(modality.trim().to_ascii_lowercase().as_str());
                key.push('|');
                for token in tokens {
                    key.push_str(token.trim().to_ascii_lowercase().as_str());
                    key.push('\u{1f}');
                }
                Some(key)
            }
            MultiModalInput::ImageBytes(_) => None,

        }

    }

    fn submit_snapshot_bundle_write(
        &mut self,
        worker_key: String,
        bundle_path: PathBuf,
    ) -> io::Result<u64> {

        let bytes = encode_brain_snapshot_bundle(self)?;
        let generation = submit_snapshot_write(worker_key.clone(), bundle_path, bytes);

        self.auto_snapshot.worker_key = Some(worker_key);
        self.auto_snapshot.last_enqueued_generation = generation;

        Ok(generation)
    }

    fn maybe_auto_snapshot_after_insert(&mut self) {
        if self.auto_snapshot.instance_id.is_none() {
            return;
        }

        self.auto_snapshot.pending_inserts = self.auto_snapshot.pending_inserts.saturating_add(1);

        // Inserts are WAL-only on the hotpath. Full-state snapshots are reserved for
        // explicit checkpoint/flush calls (periodic compaction or non-append updates).
        if let Some(worker_key) = self.transaction_log.worker_key.as_deref() {
            self.auto_snapshot.last_error = latest_transaction_error(worker_key);
        }
    }

    pub fn enable_default_cnn_image_path(&mut self) {
        self.image_feature_extractor = Some(CnnFeatureExtractor::default());
    }

    pub fn enable_cnn_image_path(&mut self, extractor: CnnFeatureExtractor) {
        self.image_feature_extractor = Some(extractor);
    }

    pub fn disable_cnn_image_path(&mut self) {
        self.image_feature_extractor = None;
    }

    pub fn is_cnn_image_path_enabled(&self) -> bool {
        self.image_feature_extractor.is_some()
    }

    pub(crate) fn classifier_ready_input(&self, content: &MultiModalInput) -> MultiModalInput {
        if let (Some(extractor), MultiModalInput::ImageBytes(bytes)) =
            (&self.image_feature_extractor, content)
            && let Ok(tokens) = extractor.extract_feature_tokens(bytes)
        {
            return MultiModalInput::FeatureTokens {
                modality: "cnnimg".to_string(),
                tokens,
            };
        }

        content.clone()
    }

    pub fn cognitive_network(&self) -> &MultiModalSubNetwork {
        &self.cognitive_net
    }

    pub fn memory_network(&self) -> &MultiModalSubNetwork {
        &self.memory_net
    }

    pub fn classifier(&self) -> &PatternClassifier {
        &self.classifier
    }

    pub fn classifier_mut(&mut self) -> &mut PatternClassifier {
        &mut self.classifier
    }

    pub fn classify_pattern(&self, content: &MultiModalInput) -> Option<ClassificationResult> {
        let prepared = self.classifier_ready_input(content);
        self.classifier.classify(&prepared)
    }

    pub fn classify_pattern_top_k(
        &self,
        content: &MultiModalInput,
        k: usize,
    ) -> Vec<ClassificationResult> {
        let prepared = self.classifier_ready_input(content);
        self.classifier.classify_top_k(&prepared, k)
    }

    pub fn absorb_truth(
        &mut self,
        content: &MultiModalInput,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.memory_net
            .insert_content(content, metadata, dendrite_type);

        self.append_transaction_operation(TransactionOperation::Upsert {
            target: TransactionTarget::Memory,
            content: content.clone(),
            metadata: metadata.clone(),
            dendrite_type,
        });

        self.invalidate_query_cache();

        self.maybe_auto_snapshot_after_insert();
    }

    pub fn absorb_true_text(
        &mut self,
        text: &str,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.absorb_truth(
            &MultiModalInput::Text(text.to_string()),
            metadata,
            dendrite_type,
        );
    }

    pub fn absorb_true_image_bytes(
        &mut self,
        bytes: &[u8],
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.absorb_truth(
            &MultiModalInput::ImageBytes(bytes.to_vec()),
            metadata,
            dendrite_type,
        );
    }

    pub fn absorb_true_image_from_file(
        &mut self,
        path: &Path,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) -> Result<(), ImageIoError> {
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        self.absorb_true_image_bytes(image_buffer.as_slice(), metadata, dendrite_type);
        Ok(())
    }

    pub fn all_dendrites_sorted(&self) -> Vec<MultimodalDendrite> {
        let mut dendrites = self.cognitive_net.all_dendrites_sorted();
        dendrites.extend(self.memory_net.all_dendrites_sorted());
        dendrites.sort_by(|left, right| left.uid.cmp(&right.uid));
        dendrites
    }

    pub fn snapshot_instance(&self, instance_id: &str) -> io::Result<()> {
        let bundle_path = snapshot_bundle_path_for_instance(instance_id, None);
        save_brain_snapshot_bundle_to_file(self, &bundle_path)?;
        Ok(())
    }

    pub fn snapshot_instance_in_dir(&self, instance_id: &str, dir: &Path) -> io::Result<()> {
        let bundle_path = snapshot_bundle_path_for_instance(instance_id, Some(dir));
        save_brain_snapshot_bundle_to_file(self, &bundle_path)?;
        Ok(())
    }

    pub fn load_snapshot_instance(&mut self, instance_id: &str) -> io::Result<SnapshotLoadStatus> {
        self.configure_transaction_log_for_instance(instance_id, None);
        load_snapshot_status_with_transaction_replay(self, instance_id, None)
    }

    pub fn load_snapshot_instance_in_dir(
        &mut self,
        instance_id: &str,
        dir: &Path,
    ) -> io::Result<SnapshotLoadStatus> {
        self.configure_transaction_log_for_instance(instance_id, Some(dir.to_path_buf()));
        load_snapshot_status_with_transaction_replay(self, instance_id, Some(dir))
    }

    pub fn snapshot_bundle_path_for_instance_in_dir(
        &self,
        instance_id: &str,
        dir: &Path,
    ) -> PathBuf {
        snapshot_bundle_path_for_instance(instance_id, Some(dir))
    }

    pub fn snapshot_paths_for_instance_in_dir(
        &self,
        instance_id: &str,
        dir: &Path,
    ) -> (PathBuf, PathBuf, PathBuf) {
        snapshot_paths_for_instance(instance_id, Some(dir))
    }

    pub fn dump_cognitive_network(&self) -> NetworkDump {
        NetworkDump {
            name: "cognitive",
            nodes: self.cognitive_net.all_dendrites_sorted(),
        }
    }

    pub fn dump_memory_network(&self) -> NetworkDump {
        NetworkDump {
            name: "memory",
            nodes: self.memory_net.all_dendrites_sorted(),
        }
    }

    pub fn dump_brain(&self) -> BrainDump {
        BrainDump {
            cognitive: self.dump_cognitive_network(),
            memory: self.dump_memory_network(),
        }
    }

    pub fn format_brain_dump(&self) -> String {
        let dump = self.dump_brain();
        let mut lines = Vec::new();

        lines.push(format!(
            "Brain dump: cognitive={} memory={}",
            dump.cognitive.nodes.len(),
            dump.memory.nodes.len()
        ));
        lines.push(Self::format_network_dump(&dump.cognitive));
        lines.push(Self::format_network_dump(&dump.memory));

        lines.join("\n")
    }

    fn format_network_dump(network: &NetworkDump) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "  {} network: {} nodes",
            network.name,
            network.nodes.len()
        ));

        for node in &network.nodes {
            let connections: Vec<String> = node
                .connections
                .iter()
                .map(|axon| format!("{}({})", axon.to, axon.weight))
                .collect();

            lines.push(format!(
                "    [{}] data='{}' modality='{}' lang='{}' type={:?} normalized='{}' links=[{}]",
                node.uid,
                node.data,
                node.modality,
                node.metadata.get("lang").unwrap_or(""),
                node.dendrite_type,
                node.normalized_key,
                connections.join(", ")
            ));
        }

        lines.join("\n")
    }

    pub fn insert_multimodal(
        &mut self,
        content: &MultiModalInput,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.cognitive_net
            .insert_content(content, metadata, dendrite_type);

        self.append_transaction_operation(TransactionOperation::Upsert {
            target: TransactionTarget::Cognitive,
            content: content.clone(),
            metadata: metadata.clone(),
            dendrite_type,
        });

        self.invalidate_query_cache();

        self.maybe_auto_snapshot_after_insert();
    }

    pub fn enumerate_multimodal_path(
        &self,
        content: &MultiModalInput,
    ) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
        let cognitive_result = self.cognitive_net.enumerate_path_content(content);

        if cognitive_result.0.is_some() {
            return cognitive_result;
        }

        self.memory_net.enumerate_path_content(content)
    }

    pub fn insert_text(
        &mut self,
        text: &str,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.insert_multimodal(
            &MultiModalInput::Text(text.to_string()),
            metadata,
            dendrite_type,
        );
    }

    pub fn enumerate_text_path(
        &self,
        text: &str,
    ) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
        self.enumerate_multimodal_path(&MultiModalInput::Text(text.to_string()))
    }

    pub fn insert_image_bytes(
        &mut self,
        bytes: &[u8],
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.insert_multimodal(
            &MultiModalInput::ImageBytes(bytes.to_vec()),
            metadata,
            dendrite_type,
        );
    }

    pub fn enumerate_image_bytes_path(
        &self,
        bytes: &[u8],
    ) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
        self.enumerate_multimodal_path(&MultiModalInput::ImageBytes(bytes.to_vec()))
    }

    pub fn evaluate_question_fuzziness(&self, content: &MultiModalInput) -> f64 {
        if !self.query_cache_enabled {
            let cognitive_score = self.cognitive_net.fuzzy_success_score_for_content(content);

            if cognitive_score >= 0.0 {
                return cognitive_score;
            }

            return self.memory_net.fuzzy_success_score_for_content(content);
        }

        let cache_key = Self::query_cache_key(content);
        if let Some(cache_key) = cache_key.as_deref() {
            let mut cache = self.query_score_cache.borrow_mut();
            if let Some(score) = cache.get(cache_key) {
                return score;
            }
        }

        let cognitive_score = self.cognitive_net.fuzzy_success_score_for_content(content);

        if cognitive_score >= 0.0 {
            if let Some(cache_key) = cache_key {
                self.query_score_cache
                    .borrow_mut()
                    .insert(cache_key, cognitive_score);
            }
            return cognitive_score;
        }

        let memory_score = self.memory_net.fuzzy_success_score_for_content(content);
        if let Some(cache_key) = cache_key {
            self.query_score_cache
                .borrow_mut()
                .insert(cache_key, memory_score);
        }
        memory_score

    }

    pub fn decide_question_storage(&self, content: &MultiModalInput) -> QuestionStoreDecision {

        let score = self.evaluate_question_fuzziness(content);

        if score < 0.0 {
            return QuestionStoreDecision::Defer;
        }

        if score >= 0.60 {
            QuestionStoreDecision::Store
        } else {
            QuestionStoreDecision::DoNotStore
        }

    }

    pub fn insert_image_from_file(
        &mut self,
        path: &Path,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) -> Result<(), ImageIoError> {
        
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        self.insert_image_bytes(image_buffer.as_slice(), metadata, dendrite_type);

        Ok(())

    }
    
}

fn save_brain_snapshot_bundle_to_file(brain: &MultiModalBrain, path: &Path) -> io::Result<()> {

    let bytes = encode_brain_snapshot_bundle(brain)?;
    write_snapshot_bytes_to_path(path, bytes.as_slice())

}

fn encode_brain_snapshot_bundle(brain: &MultiModalBrain) -> io::Result<Vec<u8>> {

    let bundle = BrainSnapshotBundleV1 {
        cognitive_net: brain.cognitive_net.clone(),
        memory_net: brain.memory_net.clone(),
        classifier: brain.classifier.clone(),
    };

    let encoded = bincode::serialize(&bundle).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to encode brain snapshot bundle: {err}"),
        )
    })?;

    let mut bytes = Vec::with_capacity(BRAIN_SNAPSHOT_BIN_MAGIC_V1.len() + encoded.len());
    bytes.extend_from_slice(&BRAIN_SNAPSHOT_BIN_MAGIC_V1);
    bytes.extend_from_slice(&encoded);
    Ok(bytes)

}

fn load_brain_snapshot_bundle_from_file(
    brain: &mut MultiModalBrain,
    path: &Path,
) -> io::Result<bool> {

    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err),
    };

    if bytes.len() < BRAIN_SNAPSHOT_BIN_MAGIC_V1.len()
        || bytes[0..BRAIN_SNAPSHOT_BIN_MAGIC_V1.len()] != BRAIN_SNAPSHOT_BIN_MAGIC_V1
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to decode brain snapshot bundle '{}': missing BSP1 magic header",
                path.display()
            ),
        ));
    }

    let payload = &bytes[BRAIN_SNAPSHOT_BIN_MAGIC_V1.len()..];
    let decoded = bincode::deserialize::<BrainSnapshotBundleV1>(payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to decode brain snapshot bundle '{}': {err}",
                path.display()
            ),
        )
    })?;

    brain.cognitive_net = decoded.cognitive_net;
    brain.memory_net = decoded.memory_net;
    brain.classifier = decoded.classifier;

    brain.cognitive_net.rebuild_connection_indexes();
    brain.cognitive_net.ensure_token_index();
    brain.memory_net.rebuild_connection_indexes();
    brain.memory_net.ensure_token_index();
    brain.invalidate_query_cache();

    Ok(true)

}

fn load_snapshot_status_from_bundle(
    brain: &mut MultiModalBrain,
    instance_id: &str,
    dir_override: Option<&Path>,
) -> io::Result<SnapshotLoadStatus> {

    let bundle_path = snapshot_bundle_path_for_instance(instance_id, dir_override);
    match load_brain_snapshot_bundle_from_file(brain, &bundle_path) {
        Ok(true) => Ok(SnapshotLoadStatus {
            cognitive_loaded: true,
            memory_loaded: true,
            classifier_loaded: true,
        }),
        Ok(false) => Ok(SnapshotLoadStatus {
            cognitive_loaded: false,
            memory_loaded: false,
            classifier_loaded: false,
        }),
        Err(err) => Err(err),
    }

}

fn load_snapshot_status_with_transaction_replay(
    brain: &mut MultiModalBrain,
    instance_id: &str,
    dir_override: Option<&Path>,
) -> io::Result<SnapshotLoadStatus> {
    let status = load_snapshot_status_from_bundle(brain, instance_id, dir_override)?;

    let log_path = transaction_log_path_for_instance(instance_id, dir_override);
    let operations = load_and_sanitize_transaction_log(log_path.as_path())?;

    brain.transaction_log.replay_in_progress = true;
    for operation in operations {
        apply_transaction_operation(brain, operation);
    }
    brain.transaction_log.replay_in_progress = false;

    Ok(status)
}

fn apply_transaction_operation(brain: &mut MultiModalBrain, operation: TransactionOperation) {
    match operation {
        TransactionOperation::Upsert {
            target,
            content,
            metadata,
            dendrite_type,
        } => match target {
            TransactionTarget::Cognitive => {
                brain
                    .cognitive_net
                    .insert_content(&content, &metadata, dendrite_type);
            }
            TransactionTarget::Memory => {
                brain
                    .memory_net
                    .insert_content(&content, &metadata, dendrite_type);
            }
        },
    }
    brain.invalidate_query_cache();
}

fn sanitize_instance_id(instance_id: &str) -> String {

    instance_id
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

fn fingerprint_instance_id(instance_id: &str) -> String {

    // Deterministic FNV-1a 64-bit digest to keep filenames short while avoiding collisions.
    let mut hash: u64 = 0xcbf29ce484222325;

    for byte in instance_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    format!("{hash:016x}")
}

fn snapshot_paths_for_instance(
    instance_id: &str,
    dir_override: Option<&Path>,
) -> (PathBuf, PathBuf, PathBuf) {

    let bundle = snapshot_bundle_path_for_instance(instance_id, dir_override);
    (bundle.clone(), bundle.clone(), bundle)

}

fn snapshot_bundle_path_for_instance(instance_id: &str, dir_override: Option<&Path>) -> PathBuf {

    let (cognitive, _, _) = snapshot_component_paths_for_instance(instance_id, dir_override);
    let parent = cognitive.parent().map(PathBuf::from).unwrap_or_default();
    let cognitive_name = cognitive
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("multimodal_snapshot_cognitive.nrn");

    let bundle_name = if cognitive_name.contains("_cognitive.") {
        cognitive_name.replacen("_cognitive.", "_bundle.", 1)
    } else {
        format!("{}_bundle.nrn", cognitive_name)
    };

    let mut bundle = parent;
    bundle.push(bundle_name);
    bundle

}

fn transaction_log_path_for_instance(instance_id: &str, dir_override: Option<&Path>) -> PathBuf {

    let (cognitive, _, _) = snapshot_component_paths_for_instance(instance_id, dir_override);
    let parent = cognitive.parent().map(PathBuf::from).unwrap_or_default();
    let cognitive_name = cognitive
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("multimodal_snapshot_cognitive.nrn");

    let log_name = if cognitive_name.contains("_cognitive.") {
        cognitive_name.replacen("_cognitive.", "_transactions.", 1)
    } else {
        format!("{}_transactions.wal", cognitive_name)
    };

    let mut log_path = parent;
    log_path.push(log_name);
    log_path

}

fn snapshot_component_paths_for_instance(
    instance_id: &str,
    dir_override: Option<&Path>,
) -> (PathBuf, PathBuf, PathBuf) {

    let trimmed = instance_id.trim();
    let resolved_id = if trimmed.is_empty() {
        "default"
    } else {
        trimmed
    };

    let base = env::var("NEURON_MULTIMODAL_STORE_FILE")
        .unwrap_or_else(|_| "./multimodal_store.nrn".to_string());

    let mut sanitized_id = sanitize_instance_id(resolved_id);
    if sanitized_id.is_empty() {
        sanitized_id = "default".to_string();
    }
    if sanitized_id.len() > 48 {
        sanitized_id.truncate(48);
    }

    let fingerprint = fingerprint_instance_id(resolved_id);
    let id_segment = format!("{}_{}", sanitized_id, fingerprint);

    if let Some(dir) = dir_override {
        let mut cognitive = PathBuf::from(dir);
        cognitive.push(format!("{}_cognitive.nrn", id_segment));

        let mut memory = PathBuf::from(dir);
        memory.push(format!("{}_memory.nrn", id_segment));

        let mut classifier = PathBuf::from(dir);
        classifier.push(format!("{}_classifier.json", id_segment));

        return (cognitive, memory, classifier);
    }

    if let Ok(dir) = env::var("NEURON_MULTIMODAL_STORE_DIR") {
        let mut cognitive = PathBuf::from(&dir);
        cognitive.push(format!("{}_cognitive.nrn", id_segment));

        let mut memory = PathBuf::from(&dir);
        memory.push(format!("{}_memory.nrn", id_segment));

        let mut classifier = PathBuf::from(&dir);
        classifier.push(format!("{}_classifier.json", id_segment));

        return (cognitive, memory, classifier);
    }

    let base_path = Path::new(&base);
    let parent = base_path.parent().map(PathBuf::from).unwrap_or_default();
    let stem = base_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("multimodal_store");

    let extension = base_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("nrn");

    let mut cognitive = parent.clone();
    cognitive.push(format!("{}_{}_cognitive.{}", stem, id_segment, extension));

    let mut memory = parent;
    memory.push(format!("{}_{}_memory.{}", stem, id_segment, extension));

    let mut classifier = base_path.parent().map(PathBuf::from).unwrap_or_default();
    classifier.push(format!("{}_{}_classifier.json", stem, id_segment));

    (cognitive, memory, classifier)
    
}



#[cfg(test)]
mod tests {

    use super::*;
    use std::path::PathBuf;

    fn snapshot_test_dir(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("./target/multimodal_test_snapshots");
        path.push(test_name);
        path
    }

    fn cleanup_snapshot_test_dir(test_name: &str) {
        let _ = fs::remove_dir_all(snapshot_test_dir(test_name));
    }

    #[test]
    fn brain_routes_learning_to_cognitive_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let text_metadata = NodeMetadata::with_lang("en");
        let image_metadata = NodeMetadata::with_lang("img");
        let image_bytes = [12u8; 128];

        network.insert_text("neuron learns", &text_metadata, DendriteType::Statement);
        network.insert_image_bytes(&image_bytes, &image_metadata, DendriteType::Token);

        let text_query = network.enumerate_text_path("neuron learns");
        let image_query = network.enumerate_image_bytes_path(&image_bytes);

        assert!(text_query.0.is_some());
        assert!(image_query.0.is_some());
        assert!(
            !network
                .cognitive_network()
                .all_dendrites_sorted()
                .is_empty()
        );
        assert!(network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_absorbs_truth_into_memory_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        network.absorb_true_text("earth orbits sun", &metadata, DendriteType::Statement);

        let truth_query = network.enumerate_text_path("earth orbits sun");

        assert!(truth_query.0.is_some());
        assert!(
            network
                .cognitive_network()
                .all_dendrites_sorted()
                .is_empty()
        );
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_absorbs_truth_image_into_memory_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let truth_image = vec![64u8; 128];
        let metadata = NodeMetadata::with_lang("img");

        network.absorb_true_image_bytes(&truth_image, &metadata, DendriteType::Token);

        let image_query = network.enumerate_image_bytes_path(&truth_image);
        let score =
            network.evaluate_question_fuzziness(&MultiModalInput::ImageBytes(truth_image.clone()));

        assert!(image_query.0.is_some());
        assert_eq!(score, 1.0);
        assert!(
            network
                .cognitive_network()
                .all_dendrites_sorted()
                .is_empty()
        );
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_rejects_empty_questions_as_invalid() {
        let network = MultiModalNeuralNetwork::new_multimodal();

        let empty_text = network.evaluate_question_fuzziness(&MultiModalInput::Text(String::new()));
        let empty_features = network.evaluate_question_fuzziness(&MultiModalInput::FeatureTokens {
            modality: "sensor".to_string(),
            tokens: Vec::new(),
        });

        assert_eq!(empty_text, -1.0);
        assert_eq!(empty_features, -1.0);
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::Text(String::new())),
            QuestionStoreDecision::Defer
        );
    }

    #[test]
    fn brain_rejects_missing_truth_image_file() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let missing_path = Path::new("/definitely/not/a/real/image.png");
        let metadata = NodeMetadata::with_lang("img");

        let result =
            network.absorb_true_image_from_file(missing_path, &metadata, DendriteType::Token);

        assert!(result.is_err());
        assert!(
            network
                .cognitive_network()
                .all_dendrites_sorted()
                .is_empty()
        );
        assert!(network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_returns_negative_score_when_network_cannot_answer() {
        let network = MultiModalNeuralNetwork::new_multimodal();
        let score = network
            .evaluate_question_fuzziness(&MultiModalInput::Text("unknown question".to_string()));

        assert_eq!(score, -1.0);
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::Text("unknown question".to_string())),
            QuestionStoreDecision::Defer
        );
    }

    #[test]
    fn brain_recommends_store_for_confident_matches() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        network.insert_text("neuron learns", &metadata, DendriteType::Statement);

        let decision =
            network.decide_question_storage(&MultiModalInput::Text("neuron learns".to_string()));

        assert_eq!(decision, QuestionStoreDecision::Store);
    }

    #[test]
    fn brain_dump_separates_cognitive_and_memory_state() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        network.insert_text("neuron learns", &metadata, DendriteType::Statement);
        network.absorb_true_text("earth orbits sun", &metadata, DendriteType::Statement);

        let dump = network.dump_brain();

        assert_eq!(dump.cognitive.name, "cognitive");
        assert_eq!(dump.memory.name, "memory");
        assert!(
            dump.cognitive
                .nodes
                .iter()
                .any(|node| node.data == "txt:neuron")
        );
        assert!(
            dump.cognitive
                .nodes
                .iter()
                .any(|node| node.data == "txt:learns")
        );
        assert!(
            dump.memory
                .nodes
                .iter()
                .any(|node| node.data == "txt:earth")
        );
        assert!(
            dump.memory
                .nodes
                .iter()
                .any(|node| node.data == "txt:orbits")
        );
        assert!(dump.memory.nodes.iter().any(|node| node.data == "txt:sun"));
        assert!(network.format_brain_dump().contains("Brain dump:"));
    }

    #[test]
    fn brain_classifier_ready_input_uses_cnn_feature_tokens_for_square_images() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let image = vec![42u8; 64];

        network.enable_default_cnn_image_path();
        let prepared = network.classifier_ready_input(&MultiModalInput::ImageBytes(image));

        match prepared {
            MultiModalInput::FeatureTokens { modality, tokens } => {
                assert_eq!(modality, "cnnimg");
                assert!(!tokens.is_empty());
            }
            _ => panic!("expected CNN feature tokens"),
        }
    }

    #[test]
    fn brain_classifier_ready_input_falls_back_when_cnn_extractor_cannot_process_image() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let image = vec![42u8; 1000];

        network.enable_default_cnn_image_path();
        let prepared = network.classifier_ready_input(&MultiModalInput::ImageBytes(image.clone()));

        assert_eq!(prepared, MultiModalInput::ImageBytes(image));
    }

    #[test]
    fn brain_snapshot_round_trip_restores_cognitive_and_memory_networks() {
        let test_name = "brain_snapshot_roundtrip";
        let test_dir = snapshot_test_dir(test_name);
        cleanup_snapshot_test_dir(test_name);
        let mut original = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        original.insert_text("neuron learns", &metadata, DendriteType::Statement);
        original.absorb_true_text("earth orbits sun", &metadata, DendriteType::Statement);
        original
            .classifier_mut()
            .train_text("animal_cat", "cat on mat");

        let snapshot_id = "brain_snapshot_roundtrip";
        original
            .snapshot_instance_in_dir(snapshot_id, &test_dir)
            .expect("snapshot should persist");

        let mut restored = MultiModalNeuralNetwork::new_multimodal();
        let status = restored
            .load_snapshot_instance_in_dir(snapshot_id, &test_dir)
            .expect("snapshot should load");

        assert!(status.cognitive_loaded);
        assert!(status.memory_loaded);
        assert!(status.classifier_loaded);
        assert!(restored.enumerate_text_path("neuron learns").0.is_some());
        assert!(restored.enumerate_text_path("earth orbits sun").0.is_some());
        assert_eq!(
            restored
                .classify_pattern(&MultiModalInput::Text("cat on mat".to_string()))
                .map(|prediction| prediction.label),
            Some("animal_cat".to_string())
        );
        cleanup_snapshot_test_dir(test_name);
    }

    #[test]
    fn brain_snapshot_writes_v2_magic_header() {
        let test_name = "brain_snapshot_v2_magic";
        let test_dir = snapshot_test_dir(test_name);
        cleanup_snapshot_test_dir(test_name);

        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");
        network.insert_text("header check", &metadata, DendriteType::Statement);

        let snapshot_id = "brain_snapshot_v2_magic";
        network
            .snapshot_instance_in_dir(snapshot_id, &test_dir)
            .expect("snapshot should persist");

        let bundle_path = network.snapshot_bundle_path_for_instance_in_dir(snapshot_id, &test_dir);
        let bytes = fs::read(bundle_path).expect("bundle snapshot should be readable");

        assert!(bytes.len() >= BRAIN_SNAPSHOT_BIN_MAGIC_V1.len());
        assert_eq!(bytes[0..4], BRAIN_SNAPSHOT_BIN_MAGIC_V1);

        cleanup_snapshot_test_dir(test_name);
    }

    #[test]
    fn brain_auto_snapshot_batches_by_insert_count() {
        let test_name = "brain_auto_snapshot_batches";
        let test_dir = snapshot_test_dir(test_name);
        cleanup_snapshot_test_dir(test_name);

        let mut network = MultiModalNeuralNetwork::new_multimodal();
        network.enable_auto_snapshot_in_dir("autosnap", &test_dir, 2);

        let metadata = NodeMetadata::with_lang("en");
        let bundle_path = network.snapshot_bundle_path_for_instance_in_dir("autosnap", &test_dir);

        network.insert_text("first insert", &metadata, DendriteType::Statement);

        assert!(!bundle_path.exists());
        assert_eq!(network.auto_snapshot_pending_inserts(), 1);

        network.insert_text("second insert", &metadata, DendriteType::Statement);

        let flushed = network
            .flush_auto_snapshot()
            .expect("flush should persist batched autosnapshot write");
        assert!(flushed);
        assert!(bundle_path.exists());
        assert_eq!(network.auto_snapshot_pending_inserts(), 0);
        assert!(network.auto_snapshot_last_error().is_none());

        cleanup_snapshot_test_dir(test_name);
    }

    #[test]
    fn brain_auto_snapshot_flush_persists_dirty_state() {
        let test_name = "brain_auto_snapshot_flush";
        let test_dir = snapshot_test_dir(test_name);
        cleanup_snapshot_test_dir(test_name);

        let mut network = MultiModalNeuralNetwork::new_multimodal();
        network.enable_auto_snapshot_in_dir("autosnap_flush", &test_dir, 10);

        let metadata = NodeMetadata::with_lang("en");
        let bundle_path = network.snapshot_bundle_path_for_instance_in_dir("autosnap_flush", &test_dir);

        network.insert_text("pending insert", &metadata, DendriteType::Statement);
        assert_eq!(network.auto_snapshot_pending_inserts(), 1);
        assert!(!bundle_path.exists());

        let flushed = network
            .flush_auto_snapshot()
            .expect("flush should persist pending inserts");

        assert!(flushed);
    assert!(bundle_path.exists());
        assert_eq!(network.auto_snapshot_pending_inserts(), 0);
        assert!(network.auto_snapshot_last_error().is_none());

        cleanup_snapshot_test_dir(test_name);
    }

    #[test]
    fn brain_snapshot_instances_do_not_collide() {
        let test_name = "brain_snapshot_isolation";
        let test_dir = snapshot_test_dir(test_name);
        cleanup_snapshot_test_dir(test_name);
        let id_a = "brain_snapshot_a";
        let id_b = "brain_snapshot_b";
        let metadata = NodeMetadata::with_lang("en");

        let mut net_a = MultiModalNeuralNetwork::new_multimodal();
        net_a.insert_text("alpha signal", &metadata, DendriteType::Statement);
        net_a.absorb_true_text("alpha truth", &metadata, DendriteType::Statement);
        net_a
            .snapshot_instance_in_dir(id_a, &test_dir)
            .expect("snapshot A should persist");

        let mut net_b = MultiModalNeuralNetwork::new_multimodal();
        net_b.insert_text("beta signal", &metadata, DendriteType::Statement);
        net_b.absorb_true_text("beta truth", &metadata, DendriteType::Statement);
        net_b
            .snapshot_instance_in_dir(id_b, &test_dir)
            .expect("snapshot B should persist");

        let mut loaded_a = MultiModalNeuralNetwork::new_multimodal();
        loaded_a
            .load_snapshot_instance_in_dir(id_a, &test_dir)
            .expect("snapshot A should load");

        let mut loaded_b = MultiModalNeuralNetwork::new_multimodal();
        loaded_b
            .load_snapshot_instance_in_dir(id_b, &test_dir)
            .expect("snapshot B should load");

        assert!(loaded_a.enumerate_text_path("alpha signal").0.is_some());
        assert!(loaded_a.enumerate_text_path("alpha truth").0.is_some());
        assert!(loaded_a.enumerate_text_path("beta signal").0.is_none());

        assert!(loaded_b.enumerate_text_path("beta signal").0.is_some());
        assert!(loaded_b.enumerate_text_path("beta truth").0.is_some());
        assert!(loaded_b.enumerate_text_path("alpha signal").0.is_none());
        cleanup_snapshot_test_dir(test_name);
    }
    
}
