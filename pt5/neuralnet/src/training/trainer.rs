use std::collections::{BTreeMap, BTreeSet};

use crate::core::brain::MultiModalBrain;
use crate::core::nodenet::NodeMetadata;
use crate::dendrites::text_dendrite::DendriteType;
use crate::helpers::multimodal_controller::MultiModalInput;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainerBridgeTarget {
    Cognitive,
    Memory,
    Both,
}

#[derive(Debug, Clone)]
pub struct TrainerBatch {
    pub label: String,
    pub samples: Vec<MultiModalInput>,
}

impl TrainerBatch {
    pub fn new(label: &str, samples: Vec<MultiModalInput>) -> Self {
        Self {
            label: label.to_string(),
            samples,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TrainerBatchReport {
    pub total_examples: usize,
    pub trained_examples: usize,
    pub skipped_examples: usize,
    pub per_label_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct TrainerEvaluationSample {
    pub expected_label: String,
    pub content: MultiModalInput,
}

impl TrainerEvaluationSample {
    pub fn new(expected_label: &str, content: MultiModalInput) -> Self {
        Self {
            expected_label: expected_label.to_string(),
            content,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TrainerEvaluationReport {
    pub total_samples: usize,
    pub evaluated_samples: usize,
    pub correct_predictions: usize,
    pub unknown_predictions: usize,
    pub skipped_samples: usize,
    pub accuracy: f64,
    pub per_label_total: BTreeMap<String, usize>,
    pub per_label_correct: BTreeMap<String, usize>,
    pub confusion_matrix: BTreeMap<String, BTreeMap<String, usize>>,
    pub per_label_metrics: BTreeMap<String, LabelQualityMetrics>,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LabelQualityMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl MultiModalBrain {

    pub fn train_labeled_pattern(
        &mut self,
        label: &str,
        content: &MultiModalInput,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
        target: TrainerBridgeTarget,
    ) {
        let prepared = self.classifier_ready_input(content);
        self.classifier_mut().train_example(label, prepared);

        match target {
            TrainerBridgeTarget::Cognitive => {
                self.insert_multimodal(content, metadata, dendrite_type);
            }
            TrainerBridgeTarget::Memory => {
                self.absorb_truth(content, metadata, dendrite_type);
            }
            TrainerBridgeTarget::Both => {
                self.insert_multimodal(content, metadata, dendrite_type);
                self.absorb_truth(content, metadata, dendrite_type);
            }
        }
    }

    pub fn train_labeled_batches(
        &mut self,
        batches: &[TrainerBatch],
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
        target: TrainerBridgeTarget,
    ) -> TrainerBatchReport {
        let mut report = TrainerBatchReport::default();

        for batch in batches {
            report.total_examples += batch.samples.len();
            let normalized_label = batch.label.trim().to_ascii_lowercase();

            if normalized_label.is_empty() {
                continue;
            }

            for sample in &batch.samples {
                self.train_labeled_pattern(
                    &normalized_label,
                    sample,
                    metadata,
                    dendrite_type,
                    target,
                );

                report.trained_examples += 1;
                *report
                    .per_label_counts
                    .entry(normalized_label.clone())
                    .or_insert(0) += 1;
            }
        }

        report.skipped_examples += report.total_examples.saturating_sub(report.trained_examples);
        report

    }

    pub fn evaluate_labeled_samples(
        &self,
        samples: &[TrainerEvaluationSample],
    ) -> TrainerEvaluationReport {

        let mut report = TrainerEvaluationReport {
            total_samples: samples.len(),
            ..TrainerEvaluationReport::default()
        };

        for sample in samples {
            let expected = sample.expected_label.trim().to_ascii_lowercase();

            if expected.is_empty() {
                report.skipped_samples += 1;
                continue;
            }

            report.evaluated_samples += 1;
            *report.per_label_total.entry(expected.clone()).or_insert(0) += 1;

            match self.classify_pattern(&sample.content) {
                Some(predicted) if predicted.label == expected => {
                    report.correct_predictions += 1;
                    *report.per_label_correct.entry(expected).or_insert(0) += 1;
                    increment_confusion_count(
                        &mut report.confusion_matrix,
                        sample.expected_label.trim().to_ascii_lowercase(),
                        predicted.label,
                    );
                }
                Some(predicted) => {
                    increment_confusion_count(
                        &mut report.confusion_matrix,
                        sample.expected_label.trim().to_ascii_lowercase(),
                        predicted.label,
                    );
                }
                None => {
                    report.unknown_predictions += 1;
                    increment_confusion_count(
                        &mut report.confusion_matrix,
                        sample.expected_label.trim().to_ascii_lowercase(),
                        "<unknown>".to_string(),
                    );
                }
            }
        }

        if report.evaluated_samples > 0 {
            report.accuracy = report.correct_predictions as f64 / report.evaluated_samples as f64;
        }

        compute_quality_metrics(&mut report);

        report

    }

}

fn increment_confusion_count(
    matrix: &mut BTreeMap<String, BTreeMap<String, usize>>,
    expected: String,
    predicted: String,
) {
    *matrix
        .entry(expected)
        .or_default()
        .entry(predicted)
        .or_insert(0) += 1;
}

fn compute_quality_metrics(report: &mut TrainerEvaluationReport) {
    let mut labels: BTreeSet<String> = BTreeSet::new();

    labels.extend(report.per_label_total.keys().cloned());
    for predicted_map in report.confusion_matrix.values() {
        for predicted in predicted_map.keys() {
            if predicted != "<unknown>" {
                labels.insert(predicted.clone());
            }
        }
    }

    if labels.is_empty() {
        return;
    }

    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;
    let mut tp_total: usize = 0;
    let mut fp_total: usize = 0;
    let mut fn_total: usize = 0;

    for label in labels {
        let tp = report
            .confusion_matrix
            .get(&label)
            .and_then(|row| row.get(&label))
            .copied()
            .unwrap_or(0);

        let fp = report
            .confusion_matrix
            .iter()
            .filter(|(expected, _)| *expected != &label)
            .map(|(_, row)| row.get(&label).copied().unwrap_or(0))
            .sum::<usize>();

        let fn_count = report
            .confusion_matrix
            .get(&label)
            .map(|row| {
                row.iter()
                    .filter(|(predicted, _)| *predicted != &label)
                    .map(|(_, count)| *count)
                    .sum::<usize>()
            })
            .unwrap_or(0);

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let f1 = if (precision + recall) > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let support = report.per_label_total.get(&label).copied().unwrap_or(0);

        report.per_label_metrics.insert(
            label,
            LabelQualityMetrics {
                precision,
                recall,
                f1,
                support,
            },
        );

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
        tp_total += tp;
        fp_total += fp;
        fn_total += fn_count;
    }

    let label_count = report.per_label_metrics.len() as f64;
    if label_count > 0.0 {
        report.macro_precision = precision_sum / label_count;
        report.macro_recall = recall_sum / label_count;
        report.macro_f1 = f1_sum / label_count;
    }

    report.micro_precision = if tp_total + fp_total > 0 {
        tp_total as f64 / (tp_total + fp_total) as f64
    } else {
        0.0
    };

    report.micro_recall = if tp_total + fn_total > 0 {
        tp_total as f64 / (tp_total + fn_total) as f64
    } else {
        0.0
    };

    report.micro_f1 = if (report.micro_precision + report.micro_recall) > 0.0 {
        2.0 * report.micro_precision * report.micro_recall
            / (report.micro_precision + report.micro_recall)
    } else {
        0.0
    };
}

#[cfg(test)]
mod tests {
    
    use super::*;
    use crate::core::brain::MultiModalNeuralNetwork;

    fn vertical_stripes_image_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for _y in 0..8 {
            for x in 0..8 {
                if x % 2 == 0 {
                    bytes.push(220);
                } else {
                    bytes.push(20);
                }
            }
        }
        bytes
    }

    fn horizontal_stripes_image_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for y in 0..8 {
            for _x in 0..8 {
                if y % 2 == 0 {
                    bytes.push(220);
                } else {
                    bytes.push(20);
                }
            }
        }
        bytes
    }

    fn diagonal_gradient_image_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                bytes.push(((x + y) * 16) as u8);
            }
        }
        bytes
    }

    #[test]
    fn trainer_bridge_trains_classifier_and_cognitive_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");
        let content = MultiModalInput::Text("cat on mat".to_string());

        network.train_labeled_pattern(
            "animal_cat",
            &content,
            &metadata,
            DendriteType::Statement,
            TrainerBridgeTarget::Cognitive,
        );

        assert_eq!(network.classifier().pattern_count(), 1);
        assert!(network.enumerate_text_path("cat on mat").0.is_some());
        assert!(network.memory_network().all_dendrites_sorted().is_empty());

        let prediction = network.classify_pattern(&content);
        assert!(prediction.is_some());
        assert_eq!(
            prediction
                .unwrap_or_else(|| panic!("classifier should return a prediction"))
                .label,
            "animal_cat"
        );
    }

    #[test]
    fn trainer_bridge_can_route_to_memory_network() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("img");
        let content = MultiModalInput::ImageBytes(vec![64u8; 128]);

        network.train_labeled_pattern(
            "truth_pattern",
            &content,
            &metadata,
            DendriteType::Token,
            TrainerBridgeTarget::Memory,
        );

        assert_eq!(network.classifier().pattern_count(), 1);
        assert!(network.cognitive_network().all_dendrites_sorted().is_empty());
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());
        assert!(network.enumerate_image_bytes_path(&[64u8; 128]).0.is_some());
    }

    #[test]
    fn trainer_bridge_can_mirror_to_both_networks() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("sensor");
        let content = MultiModalInput::FeatureTokens {
            modality: "sensor".to_string(),
            tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
        };

        network.train_labeled_pattern(
            "sensor_hot",
            &content,
            &metadata,
            DendriteType::Token,
            TrainerBridgeTarget::Both,
        );

        assert_eq!(network.classifier().pattern_count(), 1);
        assert!(!network.cognitive_network().all_dendrites_sorted().is_empty());
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());

        let ranked = network.classify_pattern_top_k(&content, 3);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].label, "sensor_hot");
    }

    #[test]
    fn trainer_bridge_predictions_include_expected_confidence_and_unknown_behavior() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        network.train_labeled_pattern(
            "animal_cat",
            &MultiModalInput::Text("cat on mat".to_string()),
            &metadata,
            DendriteType::Statement,
            TrainerBridgeTarget::Cognitive,
        );

        let predicted = network.classify_pattern(&MultiModalInput::Text("cat on mat".to_string()));
        assert!(predicted.is_some());
        let prediction = predicted
            .unwrap_or_else(|| panic!("classifier should return prediction for known sample"));
        assert_eq!(prediction.label, "animal_cat");
        assert!((prediction.score - 1.0).abs() < f64::EPSILON);

        let unknown = network.classify_pattern(&MultiModalInput::Text("bird in sky".to_string()));
        assert!(unknown.is_none());

        let ranked = network.classify_pattern_top_k(&MultiModalInput::Text("cat on mat".to_string()), 3);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].label, "animal_cat");
        assert!((ranked[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn batch_trainer_reports_counts_and_routes() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        let batches = vec![
            TrainerBatch::new(
                "animal_cat",
                vec![
                    MultiModalInput::Text("cat on mat".to_string()),
                    MultiModalInput::Text("cat on sofa".to_string()),
                ],
            ),
            TrainerBatch::new(
                "animal_dog",
                vec![MultiModalInput::Text("dog in park".to_string())],
            ),
        ];

        let report = network.train_labeled_batches(
            &batches,
            &metadata,
            DendriteType::Statement,
            TrainerBridgeTarget::Cognitive,
        );

        assert_eq!(report.total_examples, 3);
        assert_eq!(report.trained_examples, 3);
        assert_eq!(report.skipped_examples, 0);
        assert_eq!(report.per_label_counts.get("animal_cat"), Some(&2usize));
        assert_eq!(report.per_label_counts.get("animal_dog"), Some(&1usize));
        assert!(!network.cognitive_network().all_dendrites_sorted().is_empty());
    }

    #[test]
    fn batch_evaluation_returns_accuracy_and_unknown_counts() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        let batches = vec![
            TrainerBatch::new(
                "animal_cat",
                vec![MultiModalInput::Text("cat on mat".to_string())],
            ),
            TrainerBatch::new(
                "animal_dog",
                vec![MultiModalInput::Text("dog in park".to_string())],
            ),
        ];

        let _ = network.train_labeled_batches(
            &batches,
            &metadata,
            DendriteType::Statement,
            TrainerBridgeTarget::Cognitive,
        );

        let eval_samples = vec![
            TrainerEvaluationSample::new(
                "animal_cat",
                MultiModalInput::Text("cat on mat".to_string()),
            ),
            TrainerEvaluationSample::new(
                "animal_dog",
                MultiModalInput::Text("dog in park".to_string()),
            ),
            TrainerEvaluationSample::new(
                "animal_bird",
                MultiModalInput::Text("bird in sky".to_string()),
            ),
        ];

        let report = network.evaluate_labeled_samples(&eval_samples);
        assert_eq!(report.total_samples, 3);
        assert_eq!(report.evaluated_samples, 3);
        assert_eq!(report.correct_predictions, 2);
        assert_eq!(report.unknown_predictions, 1);
        assert_eq!(report.skipped_samples, 0);
        assert!((report.accuracy - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert_eq!(report.per_label_total.get("animal_cat"), Some(&1usize));
        assert_eq!(report.per_label_correct.get("animal_cat"), Some(&1usize));
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_cat")
                .and_then(|row| row.get("animal_cat")),
            Some(&1usize)
        );
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_dog")
                .and_then(|row| row.get("animal_dog")),
            Some(&1usize)
        );
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_bird")
                .and_then(|row| row.get("<unknown>")),
            Some(&1usize)
        );
        assert!(
            report
                .per_label_metrics
                .get("animal_cat")
                .map(|metric| {
                    (metric.precision - 1.0).abs() < f64::EPSILON
                        && (metric.recall - 1.0).abs() < f64::EPSILON
                        && (metric.f1 - 1.0).abs() < f64::EPSILON
                        && metric.support == 1
                })
                .unwrap_or(false)
        );
        assert!(
            report
                .per_label_metrics
                .get("animal_bird")
                .map(|metric| {
                    (metric.precision - 0.0).abs() < f64::EPSILON
                        && (metric.recall - 0.0).abs() < f64::EPSILON
                        && (metric.f1 - 0.0).abs() < f64::EPSILON
                        && metric.support == 1
                })
                .unwrap_or(false)
        );
        assert!((report.macro_precision - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((report.macro_recall - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((report.macro_f1 - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((report.micro_precision - 1.0).abs() < f64::EPSILON);
        assert!((report.micro_recall - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((report.micro_f1 - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn image_batch_evaluation_uses_cnn_path_and_reports_micro_metrics() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        network.enable_default_cnn_image_path();
        network.classifier_mut().set_min_confidence(0.95);
        let metadata = NodeMetadata::with_lang("img");

        let cat_a = vertical_stripes_image_8x8();
        let cat_b = vertical_stripes_image_8x8();
        let dog_a = horizontal_stripes_image_8x8();
        let dog_b = horizontal_stripes_image_8x8();

        let batches = vec![
            TrainerBatch::new(
                "animal_cat",
                vec![
                    MultiModalInput::ImageBytes(cat_a.clone()),
                    MultiModalInput::ImageBytes(cat_b.clone()),
                ],
            ),
            TrainerBatch::new(
                "animal_dog",
                vec![
                    MultiModalInput::ImageBytes(dog_a.clone()),
                    MultiModalInput::ImageBytes(dog_b.clone()),
                ],
            ),
        ];

        let train_report = network.train_labeled_batches(
            &batches,
            &metadata,
            DendriteType::Token,
            TrainerBridgeTarget::Cognitive,
        );

        assert_eq!(train_report.trained_examples, 4);

        let eval = vec![
            TrainerEvaluationSample::new("animal_cat", MultiModalInput::ImageBytes(cat_a)),
            TrainerEvaluationSample::new("animal_cat", MultiModalInput::ImageBytes(cat_b)),
            TrainerEvaluationSample::new("animal_dog", MultiModalInput::ImageBytes(dog_a)),
            TrainerEvaluationSample::new("animal_dog", MultiModalInput::ImageBytes(dog_b)),
            TrainerEvaluationSample::new(
                "animal_bird",
                MultiModalInput::ImageBytes(diagonal_gradient_image_8x8()),
            ),
        ];

        let report = network.evaluate_labeled_samples(&eval);

        assert_eq!(report.total_samples, 5);
        assert_eq!(report.evaluated_samples, 5);
        assert_eq!(report.correct_predictions, 4);
        assert_eq!(report.unknown_predictions, 1);
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_cat")
                .and_then(|row| row.get("animal_cat")),
            Some(&2usize)
        );
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_dog")
                .and_then(|row| row.get("animal_dog")),
            Some(&2usize)
        );
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_bird")
                .and_then(|row| row.get("<unknown>")),
            Some(&1usize)
        );
        assert!((report.accuracy - 0.8).abs() < f64::EPSILON);
        assert!((report.micro_precision - 1.0).abs() < f64::EPSILON);
        assert!((report.micro_recall - 0.8).abs() < f64::EPSILON);
        assert!((report.micro_f1 - (8.0 / 9.0)).abs() < 1e-12);
    }
}
