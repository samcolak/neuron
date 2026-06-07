use crate::core::brain::MultiModalBrain;

use crate::helpers::multimodal_controller::MultiModalInput;
use crate::helpers::pattern_classifier::ClassificationResult;

pub struct BrainRagBridge;

impl BrainRagBridge {
    pub fn query_from_input(input: &MultiModalInput) -> String {
        match input {
            MultiModalInput::Text(text) => text.trim().to_string(),
            MultiModalInput::ImageBytes(bytes) => {
                format!("image-bytes:{}", bytes.len())
            }
            MultiModalInput::FeatureTokens { modality, tokens } => {
                if tokens.is_empty() {
                    modality.trim().to_ascii_lowercase()
                } else {
                    format!(
                        "{} {}",
                        modality.trim().to_ascii_lowercase(),
                        tokens.join(" ")
                    )
                }
            }
        }
    }

    pub fn classify_hint(
        brain: &MultiModalBrain,
        input: &MultiModalInput,
    ) -> Option<ClassificationResult> {
        brain.classify_pattern(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_from_text_input_returns_trimmed_text() {
        let input = MultiModalInput::Text("  cat on mat  ".to_string());
        let query = BrainRagBridge::query_from_input(&input);
        assert_eq!(query, "cat on mat");
    }

    #[test]
    fn query_from_feature_tokens_includes_modality_and_tokens() {
        let input = MultiModalInput::FeatureTokens {
            modality: "vision".to_string(),
            tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
        };
        let query = BrainRagBridge::query_from_input(&input);
        assert_eq!(query, "vision edge:04 shape:09");
    }
}
