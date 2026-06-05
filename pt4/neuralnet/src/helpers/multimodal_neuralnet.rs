
use crate::helpers::image_io::{ImageIoError, load_png_or_jpeg_from_path};
use crate::helpers::multimodal_controller::{MultiModalController, MultiModalInput};
use crate::helpers::multimodal_dendrite::MultimodalDendrite;
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::{NetworkNode, NodeMetadata, NodeNetwork};
use crate::helpers::text_dendrite::DendriteType;

use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::path::Path;

pub type MultiModalSubNetwork = NeuralNetwork<MultiModalController, MultimodalDendrite>;

impl<N> NodeNetwork<MultiModalController> for NeuralNetwork<MultiModalController, N>
where
    N: NetworkNode + Clone + Serialize + DeserializeOwned,
{
    type Node = N;

    fn enumerate_path_content(&self, content: &MultiModalInput) -> (Option<N>, Vec<N>) {
        let path_tokens: Vec<String> = self
            .tokenize_content(content)
            .into_iter()
            .map(|token| self.token_key_for_index(&token))
            .filter(|token| !token.is_empty())
            .collect();

        if path_tokens.is_empty() {
            return (None, Vec::new());
        }

        let mut current_uids = self.candidate_uids_for_token_vec(&path_tokens[0]);

        for segment_key in &path_tokens[1..] {
            let target_uids = self.candidate_uids_for_token_vec(segment_key);

            if target_uids.is_empty() {
                return (None, Vec::new());
            }

            let target_uid_set: HashSet<&str> = target_uids.iter().map(String::as_str).collect();
            let mut next_uids = Vec::new();

            for uid in &current_uids {
                let Some(node) = self.dendrites().get(uid) else {
                    continue;
                };

                for connection in node.connections() {
                    if target_uid_set.contains(connection.to.as_str()) {
                        next_uids.push(connection.to.clone());
                    }
                }
            }

            current_uids = next_uids;
            if current_uids.is_empty() {
                return (None, Vec::new());
            }
        }

        if let Some(last_uid) = current_uids.last()
            && let Some(last) = self.dendrites().get(last_uid)
        {
            let mut optional_path = Vec::new();
            for connection in last.connections() {
                if let Some(next) = self.dendrites().get(&connection.to) {
                    optional_path.push(next.clone());
                }
            }
            return (Some(last.clone()), optional_path);
        }

        (None, Vec::new())
    }

    fn insert_content(
        &mut self,
        content: &MultiModalInput,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.ensure_token_index();

        let tokens = self.tokenize_content(content);
        if tokens.is_empty() {
            return;
        }

        let mut chosen_path = Vec::new();

        for token in tokens {
            let token_key = self.token_key_for_index(&token);
            if token_key.is_empty() {
                continue;
            }

            let uid = self
                .candidate_uids_for_token_vec(&token_key)
                .into_iter()
                .next()
                .unwrap_or_else(|| self.insert_dendrite_and_index(&token, metadata, dendrite_type));

            chosen_path.push(uid);
        }

        for pair in chosen_path.windows(2) {
            self.connect_dendrites(&pair[0], &pair[1], 1);
        }
    }
}

impl NeuralNetwork<MultiModalController, MultimodalDendrite> {
    pub fn new_multimodal() -> Self {
        Self::with_controller(MultiModalController)
    }

    pub fn insert_multimodal(
        &mut self,
        content: &MultiModalInput,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.insert_content(content, metadata, dendrite_type);
    }

    pub fn enumerate_multimodal_path(
        &self,
        content: &MultiModalInput,
    ) -> (Option<MultimodalDendrite>, Vec<MultimodalDendrite>) {
        self.enumerate_path_content(content)
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn multimodal_subnetwork_accepts_text_and_image_content() {
        let mut network = MultiModalSubNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");
        let image_bytes = [12u8; 128];

        network.insert_text("neuron learns", &metadata, DendriteType::Statement);
        network.insert_image_bytes(
            &image_bytes,
            &NodeMetadata::with_lang("img"),
            DendriteType::Token,
        );

        let text_query = network.enumerate_text_path("neuron learns");
        let image_query = network.enumerate_image_bytes_path(&image_bytes);

        assert!(text_query.0.is_some());
        assert!(image_query.0.is_some());
    }

    #[test]
    fn multimodal_subnetwork_supports_future_modalities() {
        let mut network = MultiModalSubNetwork::new_multimodal();

        let audio = MultiModalInput::FeatureTokens {
            modality: "audio".to_string(),
            tokens: vec!["mfcc0:0a".to_string(), "mfcc1:1f".to_string()],
        };

        network.insert_multimodal(
            &audio,
            &NodeMetadata::with_lang("audio"),
            DendriteType::Token,
        );
        let result = network.enumerate_multimodal_path(&audio);

        assert!(result.0.is_some());
    }
}
