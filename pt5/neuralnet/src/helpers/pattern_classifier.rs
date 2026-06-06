use crate::helpers::multimodal_controller::{MultiModalController, MultiModalInput};
use crate::core::nodenet::NodeNetworkController;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabeledPattern {
    pub label: String,
    pub content: MultiModalInput,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternClassifier {
    patterns: Vec<LabeledPattern>,
    min_confidence: f64,
}

impl PatternClassifier {
    
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            min_confidence: 0.60,
        }
    }

    pub fn with_min_confidence(min_confidence: f64) -> Self {
        let mut classifier = Self::new();
        classifier.set_min_confidence(min_confidence);
        classifier
    }

    pub fn min_confidence(&self) -> f64 {
        self.min_confidence
    }

    pub fn set_min_confidence(&mut self, min_confidence: f64) {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    pub fn train_example(&mut self, label: &str, content: MultiModalInput) {

        let normalized_label = label.trim().to_ascii_lowercase();

        if normalized_label.is_empty() {
            return;
        }

        if let Some(existing) = self
            .patterns
            .iter_mut()
            .find(|pattern| pattern.label == normalized_label && pattern.content == content)
        {
            existing.content = content;
            return;
        }

        self.patterns.push(LabeledPattern {
            label: normalized_label,
            content,
        });

    }

    pub fn train_text(&mut self, label: &str, text: &str) {
        self.train_example(label, MultiModalInput::Text(text.to_string()));
    }

    pub fn train_image_bytes(&mut self, label: &str, bytes: &[u8]) {
        self.train_example(label, MultiModalInput::ImageBytes(bytes.to_vec()));
    }

    pub fn train_feature_tokens(&mut self, label: &str, modality: &str, tokens: Vec<String>) {
        self.train_example(
            label,
            MultiModalInput::FeatureTokens {
                modality: modality.to_string(),
                tokens,
            },
        );
    }

    pub fn classify(&self, input: &MultiModalInput) -> Option<ClassificationResult> {

        let mut best: Option<ClassificationResult> = None;

        for pattern in &self.patterns {
            let score = score_inputs(input, &pattern.content);

            if let Some(current_best) = &best
                && score <= current_best.score
            {
                continue;
            }

            best = Some(ClassificationResult {
                label: pattern.label.clone(),
                score,
            });
        }

        match best {
            Some(result) if result.score >= self.min_confidence => Some(result),
            _ => None,
        }

    }

    pub fn classify_top_k(&self, input: &MultiModalInput, k: usize) -> Vec<ClassificationResult> {
        if k == 0 || self.patterns.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<ClassificationResult> = self
            .patterns
            .iter()
            .map(|pattern| ClassificationResult {
                label: pattern.label.clone(),
                score: score_inputs(input, &pattern.content),
            })
            .filter(|result| result.score >= self.min_confidence)
            .collect();

        ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
        ranked.truncate(k);
        ranked
    }
}

fn score_inputs(left: &MultiModalInput, right: &MultiModalInput) -> f64 {

    let controller = MultiModalController;

    let left_tokens: Vec<String> = controller
        .tokenize(left)
        .into_iter()
        .map(|token| controller.normalize_token(&token))
        .filter(|token| !token.is_empty())
        .collect();

    let right_tokens: Vec<String> = controller
        .tokenize(right)
        .into_iter()
        .map(|token| controller.normalize_token(&token))
        .filter(|token| !token.is_empty())
        .collect();

    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }

    let left_to_right = directional_score(&controller, &left_tokens, &right_tokens);
    let right_to_left = directional_score(&controller, &right_tokens, &left_tokens);

    ((left_to_right + right_to_left) / 2.0).clamp(0.0, 1.0)

}

fn directional_score(
    controller: &MultiModalController,
    source_tokens: &[String],
    target_tokens: &[String],
) -> f64 {

    if source_tokens.is_empty() || target_tokens.is_empty() {
        return 0.0;
    }

    let mut token_scores = Vec::with_capacity(source_tokens.len());

    for source in source_tokens {
        let mut best = 0.0;

        for target in target_tokens {
            let (score, _) = controller.evaluate_match(source, target);
            if score > best {
                best = score;
            }
        }

        token_scores.push(best);
    }

    token_scores.iter().copied().sum::<f64>() / token_scores.len() as f64

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_predicts_text_label_from_trained_examples() {
        let mut classifier = PatternClassifier::new();
        classifier.train_text("animal_cat", "cat on mat");
        classifier.train_text("animal_dog", "dog in park");

        let query = MultiModalInput::Text("cat on mat".to_string());
        let result = classifier.classify(&query);

        assert!(result.is_some());
        let output = result.unwrap_or(ClassificationResult {
            label: String::new(),
            score: 0.0,
        });
        assert_eq!(output.label, "animal_cat");
        assert!(output.score >= 0.90);
    }

    #[test]
    fn classifier_rejects_when_below_confidence_threshold() {
        let mut classifier = PatternClassifier::with_min_confidence(0.95);
        classifier.train_text("animal_cat", "cat on mat");

        let query = MultiModalInput::Text("cat mat".to_string());
        let result = classifier.classify(&query);

        assert!(result.is_none());
    }

    #[test]
    fn classifier_supports_multimodal_feature_tokens() {
        let mut classifier = PatternClassifier::new();
        classifier.train_feature_tokens(
            "sensor_hot",
            "sensor",
            vec!["temp:1f".to_string(), "humidity:08".to_string()],
        );
        classifier.train_feature_tokens(
            "sensor_cold",
            "sensor",
            vec!["temp:05".to_string(), "humidity:20".to_string()],
        );

        let query = MultiModalInput::FeatureTokens {
            modality: "sensor".to_string(),
            tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
        };

        let result = classifier.classify(&query);
        assert!(result.is_some());
        assert_eq!(
            result
                .unwrap_or(ClassificationResult {
                    label: String::new(),
                    score: 0.0,
                })
                .label,
            "sensor_hot"
        );
    }

    #[test]
    fn classifier_top_k_returns_ranked_candidates() {
        let mut classifier = PatternClassifier::new();
        classifier.train_text("alpha", "cat on mat");
        classifier.train_text("beta", "cat on sofa");
        classifier.train_text("gamma", "dog in park");

        let query = MultiModalInput::Text("cat on".to_string());
        let top = classifier.classify_top_k(&query, 2);

        assert_eq!(top.len(), 2);
        assert!(top[0].score >= top[1].score);
    }
}