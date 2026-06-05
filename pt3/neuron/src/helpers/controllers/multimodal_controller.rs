use crate::helpers::controllers::textnode_controller::{
    evaluate_fuzziness,
    normalize_for_fuzzy_comparison,
};
use crate::helpers::nodenet::{NodeNetworkController, TokenClusterKeyStrategy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiModalInput {
    Text(String),
    ImageBytes(Vec<u8>),
    FeatureTokens { modality: String, tokens: Vec<String> },
}

#[derive(Debug, Clone, Default)]
pub struct MultiModalController;

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

fn tokenize_text(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| format!("txt:{}", normalize_for_fuzzy_comparison(token)))
        .filter(|token| token != "txt:")
        .collect()
}

fn tokenize_image(content: &[u8]) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }

    let chunk_count = 8;
    let chunk_size = (content.len() / chunk_count).max(1);
    let mut tokens = Vec::new();

    for (idx, chunk) in content.chunks(chunk_size).take(chunk_count).enumerate() {
        let mean = mean_bucket(chunk);
        let edge = edge_bucket(chunk);
        tokens.push(format!("img:m{}:{:02x}", idx, mean));
        tokens.push(format!("img:e{}:{:02x}", idx, edge));
    }

    let length_bucket = (content.len() / 128).min(255);
    tokens.push(format!("img:len:{:02x}", length_bucket));
    tokens
}

impl TokenClusterKeyStrategy for MultiModalController {
    fn cluster_key_for_token(&self, token_key: &str) -> Option<String> {
        if token_key.is_empty() {
            return None;
        }

        let mut parts = token_key.split(':');
        let modality = parts.next().unwrap_or_default();
        let family = parts.next().unwrap_or_default();

        if modality.is_empty() || family.is_empty() {
            return None;
        }

        Some(format!("{}:{}", modality, family))
    }
}

impl NodeNetworkController for MultiModalController {
    type Content = MultiModalInput;

    fn tokenize(&self, content: &Self::Content) -> Vec<String> {
        match content {
            MultiModalInput::Text(text) => tokenize_text(text),
            MultiModalInput::ImageBytes(bytes) => tokenize_image(bytes),
            MultiModalInput::FeatureTokens { modality, tokens } => tokens
                .iter()
                .map(|token| {
                    let normalized_modality = modality.trim().to_ascii_lowercase();
                    let normalized_token = token.trim().to_ascii_lowercase();
                    format!("{}:{}", normalized_modality, normalized_token)
                })
                .filter(|token| token != ":")
                .collect(),
        }
    }

    fn normalize_token(&self, token: &str) -> String {
        token.trim().to_ascii_lowercase()
    }

    fn evaluate_match(&self, left: &str, right: &str) -> (f64, Vec<String>) {
        if left == right {
            return (1.0, Vec::new());
        }

        let (left_modality, left_rest) = left.split_once(':').unwrap_or(("", ""));
        let (right_modality, right_rest) = right.split_once(':').unwrap_or(("", ""));

        if left_modality != right_modality {
            return (0.0, vec![left.to_string()]);
        }

        if left_modality == "txt" {
            return evaluate_fuzziness(left_rest, right_rest);
        }

        if left_modality == "img" {
            let left_family = left_rest.split(':').next().unwrap_or_default();
            let right_family = right_rest.split(':').next().unwrap_or_default();

            if left_family == right_family {
                return (0.65, vec!["same_image_feature_family".to_string()]);
            }

            return (0.0, vec![left.to_string()]);
        }

        let left_family = left_rest.split(':').next().unwrap_or_default();
        let right_family = right_rest.split(':').next().unwrap_or_default();

        if left_family == right_family {
            (0.60, vec!["same_modality_feature_family".to_string()])
        } else {
            (0.0, vec![left.to_string()])
        }
    }

    fn stop_words(&self, _language: &str) -> Vec<&'static str> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_text_with_modality_prefix() {
        let controller = MultiModalController;
        let tokens = controller.tokenize(&MultiModalInput::Text("Hello, Neural Net".to_string()));

        assert_eq!(tokens, vec!["txt:hello", "txt:neural", "txt:net"]);
    }

    #[test]
    fn tokenizes_image_with_modality_prefix() {
        let controller = MultiModalController;
        let tokens = controller.tokenize(&MultiModalInput::ImageBytes(vec![10u8; 128]));

        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|token| token.starts_with("img:")));
    }

    #[test]
    fn evaluate_match_rejects_cross_modality() {
        let controller = MultiModalController;
        let (score, _) = controller.evaluate_match("txt:hello", "img:m0:0a");
        assert_eq!(score, 0.0);
    }
}