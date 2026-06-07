use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::core::cnn_feature_extractor::CnnFeatureExtractor;
use crate::core::nodenet::{NodeMetadata, NodeNetwork};
use crate::helpers::image_io::{ImageIoError, load_png_or_jpeg_from_path};
use crate::helpers::multimodal_controller::MultiModalInput;
use crate::helpers::multimodal_neuralnet::MultiModalSubNetwork;
use crate::helpers::pattern_classifier::{ClassificationResult, PatternClassifier};
use crate::dendrites::text_dendrite::DendriteType;
use crate::dendrites::multimodal_dendrite::MultimodalDendrite;

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

const MULTIMODAL_BIN_MAGIC: [u8; 4] = *b"MMB1";

#[derive(Debug, Clone)]
pub struct MultiModalBrain {
    cognitive_net: MultiModalSubNetwork,
    memory_net: MultiModalSubNetwork,
    classifier: PatternClassifier,
    image_feature_extractor: Option<CnnFeatureExtractor>,
}

pub type MultiModalNeuralNetwork = MultiModalBrain;

impl MultiModalBrain {
    pub fn new_multimodal() -> Self {
        Self {
            cognitive_net: MultiModalSubNetwork::new_multimodal(),
            memory_net: MultiModalSubNetwork::new_multimodal(),
            classifier: PatternClassifier::new(),
            image_feature_extractor: None,
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
        let (cognitive_path, memory_path, classifier_path) =
            snapshot_paths_for_instance(instance_id, None);
        save_multimodal_network_to_file(&self.cognitive_net, &cognitive_path)?;
        save_multimodal_network_to_file(&self.memory_net, &memory_path)?;
        save_classifier_to_file(&self.classifier, &classifier_path)?;
        Ok(())
    }

    pub fn snapshot_instance_in_dir(&self, instance_id: &str, dir: &Path) -> io::Result<()> {
        let (cognitive_path, memory_path, classifier_path) =
            snapshot_paths_for_instance(instance_id, Some(dir));
        save_multimodal_network_to_file(&self.cognitive_net, &cognitive_path)?;
        save_multimodal_network_to_file(&self.memory_net, &memory_path)?;
        save_classifier_to_file(&self.classifier, &classifier_path)?;
        Ok(())
    }

    pub fn load_snapshot_instance(&mut self, instance_id: &str) -> io::Result<SnapshotLoadStatus> {
        let (cognitive_path, memory_path, classifier_path) =
            snapshot_paths_for_instance(instance_id, None);
        let cognitive_loaded =
            load_multimodal_network_from_file(&mut self.cognitive_net, &cognitive_path)?;
        let memory_loaded = load_multimodal_network_from_file(&mut self.memory_net, &memory_path)?;
        let classifier_loaded = load_classifier_from_file(&mut self.classifier, &classifier_path)?;
        Ok(SnapshotLoadStatus {
            cognitive_loaded,
            memory_loaded,
            classifier_loaded,
        })
    }

    pub fn load_snapshot_instance_in_dir(
        &mut self,
        instance_id: &str,
        dir: &Path,
    ) -> io::Result<SnapshotLoadStatus> {
        let (cognitive_path, memory_path, classifier_path) =
            snapshot_paths_for_instance(instance_id, Some(dir));
        let cognitive_loaded =
            load_multimodal_network_from_file(&mut self.cognitive_net, &cognitive_path)?;
        let memory_loaded = load_multimodal_network_from_file(&mut self.memory_net, &memory_path)?;
        let classifier_loaded = load_classifier_from_file(&mut self.classifier, &classifier_path)?;
        Ok(SnapshotLoadStatus {
            cognitive_loaded,
            memory_loaded,
            classifier_loaded,
        })
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
        let cognitive_score = self.cognitive_net.fuzzy_success_score_for_content(content);

        if cognitive_score >= 0.0 {
            return cognitive_score;
        }

        self.memory_net.fuzzy_success_score_for_content(content)
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

fn load_multimodal_network_from_file(
    network: &mut MultiModalSubNetwork,
    path: &Path,
) -> io::Result<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err),
    };

    if bytes.len() >= 4 && bytes[0..4] == MULTIMODAL_BIN_MAGIC {
        let loaded = bincode::deserialize::<MultiModalSubNetwork>(&bytes[4..]).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to decode multimodal snapshot '{}': {err}",
                    path.display()
                ),
            )
        })?;

        *network = loaded;
        network.rebuild_connection_indexes();
        network.rebuild_token_index();

        return Ok(true);
    }

    let loaded = serde_json::from_slice::<MultiModalSubNetwork>(&bytes).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to decode multimodal snapshot '{}': {err}",
                path.display()
            ),
        )
    })?;

    *network = loaded;
    network.rebuild_connection_indexes();
    network.rebuild_token_index();

    Ok(true)
}

fn save_multimodal_network_to_file(network: &MultiModalSubNetwork, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let encoded = bincode::serialize(network).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to encode multimodal snapshot '{}': {err}",
                path.display()
            ),
        )
    })?;

    let mut bytes = Vec::with_capacity(MULTIMODAL_BIN_MAGIC.len() + encoded.len());
    bytes.extend_from_slice(&MULTIMODAL_BIN_MAGIC);
    bytes.extend_from_slice(&encoded);

    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("nrn")
    ));

    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, path)?;

    Ok(())
}

fn load_classifier_from_file(classifier: &mut PatternClassifier, path: &Path) -> io::Result<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err),
    };

    let loaded = serde_json::from_slice::<PatternClassifier>(&bytes).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to decode classifier snapshot '{}': {err}",
                path.display()
            ),
        )
    })?;

    *classifier = loaded;
    Ok(true)
}

fn save_classifier_to_file(classifier: &PatternClassifier, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let encoded = serde_json::to_vec(classifier).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to encode classifier snapshot '{}': {err}",
                path.display()
            ),
        )
    })?;

    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));

    fs::write(&tmp_path, encoded)?;
    fs::rename(&tmp_path, path)?;

    Ok(())
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
