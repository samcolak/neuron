use crate::helpers::controllers::multimodal_controller::MultiModalInput;
use crate::helpers::image_io::{load_png_or_jpeg_from_path, ImageIoError};
use crate::helpers::multimodal_dendrite::MultimodalDendrite;
use crate::helpers::multimodal_neuralnet::MultiModalSubNetwork;
use crate::helpers::nodenet::NodeNetwork;
use crate::helpers::text_dendrite::DendriteType;

use std::path::Path;

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

#[derive(Debug, Clone)]
pub struct MultiModalBrain {
    cognitive_net: MultiModalSubNetwork,
    memory_net: MultiModalSubNetwork,
}

pub type MultiModalNeuralNetwork = MultiModalBrain;

impl MultiModalBrain {
    
    pub fn new_multimodal() -> Self {
        Self {
            cognitive_net: MultiModalSubNetwork::new_multimodal(),
            memory_net: MultiModalSubNetwork::new_multimodal(),
        }
    }

    pub fn cognitive_network(&self) -> &MultiModalSubNetwork {
        &self.cognitive_net
    }

    pub fn memory_network(&self) -> &MultiModalSubNetwork {
        &self.memory_net
    }

    pub fn absorb_truth(&mut self, content: &MultiModalInput, language: &str, dendrite_type: DendriteType) {
        self.memory_net.insert_content(content, language, dendrite_type);
    }

    pub fn absorb_true_text(&mut self, text: &str, language: &str, dendrite_type: DendriteType) {
        self.absorb_truth(&MultiModalInput::Text(text.to_string()), language, dendrite_type);
    }

    pub fn absorb_true_image_bytes(&mut self, bytes: &[u8], language: &str, dendrite_type: DendriteType) {
        self.absorb_truth(&MultiModalInput::ImageBytes(bytes.to_vec()), language, dendrite_type);
    }

    pub fn absorb_true_image_from_file(
        &mut self,
        path: &Path,
        language: &str,
        dendrite_type: DendriteType,
    ) -> Result<(), ImageIoError> {
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        self.absorb_true_image_bytes(image_buffer.as_slice(), language, dendrite_type);
        Ok(())
    }

    pub fn all_dendrites_sorted(&self) -> Vec<MultimodalDendrite> {
        let mut dendrites = self.cognitive_net.all_dendrites_sorted();
        dendrites.extend(self.memory_net.all_dendrites_sorted());
        dendrites.sort_by(|left, right| left.uid.cmp(&right.uid));
        dendrites
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

        lines.push(format!("Brain dump: cognitive={} memory={}", dump.cognitive.nodes.len(), dump.memory.nodes.len()));
        lines.push(Self::format_network_dump(&dump.cognitive));
        lines.push(Self::format_network_dump(&dump.memory));

        lines.join("\n")
    }

    fn format_network_dump(network: &NetworkDump) -> String {
        let mut lines = Vec::new();
        lines.push(format!("  {} network: {} nodes", network.name, network.nodes.len()));

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
                node.lang,
                node.dendrite_type,
                node.normalized_key,
                connections.join(", ")
            ));
        }

        lines.join("\n")
    }

    pub fn insert_multimodal(&mut self, content: &MultiModalInput, language: &str, dendrite_type: DendriteType) {
        self.cognitive_net.insert_content(content, language, dendrite_type);
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

    pub fn insert_text(&mut self, text: &str, language: &str, dendrite_type: DendriteType) {
        self.insert_multimodal(&MultiModalInput::Text(text.to_string()), language, dendrite_type);
    }

    pub fn enumerate_text_path(&self, text: &str) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
        self.enumerate_multimodal_path(&MultiModalInput::Text(text.to_string()))
    }

    pub fn insert_image_bytes(&mut self, bytes: &[u8], language: &str, dendrite_type: DendriteType) {
        self.insert_multimodal(&MultiModalInput::ImageBytes(bytes.to_vec()), language, dendrite_type);
    }

    pub fn enumerate_image_bytes_path(&self, bytes: &[u8]) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
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
        language: &str,
        dendrite_type: DendriteType,
    ) -> Result<(), ImageIoError> {
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        self.insert_image_bytes(image_buffer.as_slice(), language, dendrite_type);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_routes_learning_to_cognitive_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();

        network.insert_text("neuron learns", "en", DendriteType::Statement);
        network.insert_image_bytes(&vec![12u8; 128], "img", DendriteType::Token);

        let text_query = network.enumerate_text_path("neuron learns");
        let image_query = network.enumerate_image_bytes_path(&vec![12u8; 128]);

        assert!(text_query.0.is_some());
        assert!(image_query.0.is_some());
        assert!(!network.cognitive_network().all_dendrites_sorted().is_empty());
        assert!(network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_absorbs_truth_into_memory_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();

        network.absorb_true_text("earth orbits sun", "en", DendriteType::Statement);

        let truth_query = network.enumerate_text_path("earth orbits sun");

        assert!(truth_query.0.is_some());
        assert!(network.cognitive_network().all_dendrites_sorted().is_empty());
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_absorbs_truth_image_into_memory_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let truth_image = vec![64u8; 128];

        network.absorb_true_image_bytes(&truth_image, "img", DendriteType::Token);

        let image_query = network.enumerate_image_bytes_path(&truth_image);
        let score = network.evaluate_question_fuzziness(&MultiModalInput::ImageBytes(truth_image.clone()));

        assert!(image_query.0.is_some());
        assert_eq!(score, 1.0);
        assert!(network.cognitive_network().all_dendrites_sorted().is_empty());
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
        assert_eq!(network.decide_question_storage(&MultiModalInput::Text(String::new())), QuestionStoreDecision::Defer);
    }

    #[test]
    fn brain_rejects_missing_truth_image_file() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let missing_path = Path::new("/definitely/not/a/real/image.png");

        let result = network.absorb_true_image_from_file(missing_path, "img", DendriteType::Token);

        assert!(result.is_err());
        assert!(network.cognitive_network().all_dendrites_sorted().is_empty());
        assert!(network.memory_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn brain_returns_negative_score_when_network_cannot_answer() {
        let network = MultiModalNeuralNetwork::new_multimodal();
        let score = network.evaluate_question_fuzziness(&MultiModalInput::Text("unknown question".to_string()));

        assert_eq!(score, -1.0);
        assert_eq!(network.decide_question_storage(&MultiModalInput::Text("unknown question".to_string())), QuestionStoreDecision::Defer);
    }

    #[test]
    fn brain_recommends_store_for_confident_matches() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();

        network.insert_text("neuron learns", "en", DendriteType::Statement);

        let decision = network.decide_question_storage(&MultiModalInput::Text("neuron learns".to_string()));

        assert_eq!(decision, QuestionStoreDecision::Store);
    }

    #[test]
    fn brain_dump_separates_cognitive_and_memory_state() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();

        network.insert_text("neuron learns", "en", DendriteType::Statement);
        network.absorb_true_text("earth orbits sun", "en", DendriteType::Statement);

        let dump = network.dump_brain();

        assert_eq!(dump.cognitive.name, "cognitive");
        assert_eq!(dump.memory.name, "memory");
        assert!(dump.cognitive.nodes.iter().any(|node| node.data == "txt:neuron"));
        assert!(dump.cognitive.nodes.iter().any(|node| node.data == "txt:learns"));
        assert!(dump.memory.nodes.iter().any(|node| node.data == "txt:earth"));
        assert!(dump.memory.nodes.iter().any(|node| node.data == "txt:orbits"));
        assert!(dump.memory.nodes.iter().any(|node| node.data == "txt:sun"));
        assert!(network.format_brain_dump().contains("Brain dump:"));
    }
}