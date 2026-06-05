use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::{NetworkNode, NodeNetwork, NodeNetworkController, TokenClusterKeyStrategy};
use crate::helpers::text_dendrite::DendriteType;

use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct ImageNodeController;

fn mean_bucket(chunk: &[u8]) -> u8 {

    if chunk.is_empty() {
        return 0;
    }

    let sum: u32 = chunk.iter().map(|v| *v as u32).sum();
    let mean = sum / chunk.len() as u32;
    (mean / 16) as u8

}

fn edge_bucket(chunk: &[u8]) -> u8 {

    if chunk.len() < 2 {
        return 0;
    }

    let mut total_delta: u32 = 0;

    for pair in chunk.windows(2) {
        let left = pair[0] as i16;
        let right = pair[1] as i16;
        total_delta += (left - right).unsigned_abs() as u32;
    }

    let avg_delta = total_delta / (chunk.len() as u32 - 1);
    (avg_delta / 16) as u8

}

impl TokenClusterKeyStrategy for ImageNodeController {

    fn cluster_key_for_token(&self, token_key: &str) -> Option<String> {
        if token_key.is_empty() {
            return None;
        }

        let mut parts = token_key.split(':');
        let family = parts.next().unwrap_or_default();
        let value = parts.next().unwrap_or_default();

        if family.is_empty() || value.is_empty() {
            return None;
        }

        Some(format!("{}:{}", family, value))
    }

}

impl NodeNetworkController for ImageNodeController {

    type Content = [u8];

    fn tokenize(&self, content: &Self::Content) -> Vec<String> {
        if content.is_empty() {
            return Vec::new();
        }

        let chunk_count = 8;
        let chunk_size = (content.len() / chunk_count).max(1);
        let mut tokens = Vec::new();

        for (idx, chunk) in content.chunks(chunk_size).take(chunk_count).enumerate() {
            let mean = mean_bucket(chunk);
            let edge = edge_bucket(chunk);
            tokens.push(format!("m{}:{:02x}", idx, mean));
            tokens.push(format!("e{}:{:02x}", idx, edge));
        }

        let length_bucket = (content.len() / 128).min(255);
        tokens.push(format!("len:{:02x}", length_bucket));

        tokens
    }

    fn normalize_token(&self, token: &str) -> String {
        token.trim().to_ascii_lowercase()
    }

    fn evaluate_match(&self, left: &str, right: &str) -> (f64, Vec<String>) {
        if left == right {
            return (1.0, Vec::new());
        }

        let left_family = left.split(':').next().unwrap_or_default();
        let right_family = right.split(':').next().unwrap_or_default();

        if left_family == right_family {
            (0.65, vec!["same_feature_family".to_string()])
        } else {
            (0.0, vec![left.to_string()])
        }
    }

    fn stop_words(&self, _language: &str) -> Vec<&'static str> {
        Vec::new()
    }

}

impl<N> NodeNetwork<ImageNodeController> for NeuralNetwork<ImageNodeController, N>
where
    N: NetworkNode + Clone + Serialize + DeserializeOwned,
{

    type Node = N;

    fn enumerate_path_content(&self, content: &[u8]) -> (Option<N>, Vec<N>) {
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

    fn insert_content(&mut self, content: &[u8], language: &str, dendrite_type: DendriteType) {
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
                .unwrap_or_else(|| self.insert_dendrite_and_index(&token, language, dendrite_type));

            chosen_path.push(uid);
        }

        for pair in chosen_path.windows(2) {
            self.connect_dendrites(&pair[0], &pair[1], 1);
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::image_dendrite::ImageDendrite;
    use crate::helpers::nodenet::NodeNetwork;

    #[test]
    fn image_tokenization_produces_stable_feature_families() {
        let controller = ImageNodeController;
        let data = vec![0u8; 64];
        let tokens = controller.tokenize(&data);

        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|t| t.starts_with("m0:")));
        assert!(tokens.iter().any(|t| t.starts_with("e0:")));
        assert!(tokens.iter().any(|t| t.starts_with("len:")));
    }

    #[test]
    fn image_insert_and_enumerate_path_work() {
        let mut network: NeuralNetwork<ImageNodeController, ImageDendrite> =
            NeuralNetwork::with_controller(ImageNodeController);

        let mut image = Vec::new();
        for i in 0..256u16 {
            image.push((i % 255) as u8);
        }

        network.insert_content(&image, "img", DendriteType::Token);

        let all = network.all_dendrites_sorted();
        assert!(!all.is_empty());

        let (last, _) = network.enumerate_path_content(&image);
        assert!(last.is_some());
    }
}