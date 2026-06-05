use crate::helpers::controllers::multimodal_controller::MultiModalInput;
use crate::helpers::image_io::{load_png_or_jpeg_from_path, ImageIoError};
use crate::helpers::multimodal_dendrite::MultimodalDendrite;
use crate::helpers::multimodal_neuralnet::MultiModalSubNetwork;
use crate::helpers::nodenet::NodeNetwork;
use crate::helpers::text_dendrite::DendriteType;

use std::path::Path;

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
}