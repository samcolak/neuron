use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::core::model::{SupervisedSample, TrainableModel};
use crate::core::nodenet::{NodeMetadata, NodeNetworkController};
use crate::training::trainer::{
    LabelQualityMetrics,
    TrainerBatchReport,
    TrainerBridgeTarget,
    TrainerEvaluationReport,
};
use crate::dendrites::text_dendrite::DendriteType;
use crate::helpers::multimodal_controller::MultiModalController;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DenseTokenBaseline {
    label_token_counts: BTreeMap<String, BTreeMap<String, usize>>,
    label_total_tokens: BTreeMap<String, usize>,
}

impl DenseTokenBaseline {

    pub fn new() -> Self {
        Self::default()
    }

    fn predict_label(&self, sample: &SupervisedSample) -> Option<String> {
        let controller = MultiModalController;
        let tokens = controller.tokenize(&sample.content);

        if tokens.is_empty() || self.label_token_counts.is_empty() {
            return None;
        }

        let mut best_label: Option<String> = None;
        let mut best_score = 0.0;

        for (label, token_counts) in &self.label_token_counts {
            let total = self.label_total_tokens.get(label).copied().unwrap_or(0);
            if total == 0 {
                continue;
            }

            let overlap = tokens
                .iter()
                .map(|token| token_counts.get(token).copied().unwrap_or(0))
                .sum::<usize>();

            let score = overlap as f64 / total as f64;

            if score > best_score {
                best_score = score;
                best_label = Some(label.clone());
            }
        }

        if best_score <= 0.0 {
            None
        } else {
            best_label
        }
    
    }

}

impl TrainableModel for DenseTokenBaseline {

    fn train_epoch(
        &mut self,
        samples: &[SupervisedSample],
        _metadata: &NodeMetadata,
        _target: TrainerBridgeTarget,
        _dendrite_type: DendriteType,
    ) -> TrainerBatchReport {

        let controller = MultiModalController;
        let mut report = TrainerBatchReport::default();

        for sample in samples {
            report.total_examples += 1;

            let label = sample.label.trim().to_ascii_lowercase();
            if label.is_empty() {
                report.skipped_examples += 1;
                continue;
            }

            let tokens = controller.tokenize(&sample.content);
            if tokens.is_empty() {
                report.skipped_examples += 1;
                continue;
            }

            let label_counts = self.label_token_counts.entry(label.clone()).or_default();
            let total_tokens = self.label_total_tokens.entry(label.clone()).or_insert(0);

            for token in tokens {
                *label_counts.entry(token).or_insert(0) += 1;
                *total_tokens += 1;
            }

            report.trained_examples += 1;
            *report.per_label_counts.entry(label).or_insert(0) += 1;

        }

        report

    }

    fn evaluate_dataset(&self, samples: &[SupervisedSample]) -> TrainerEvaluationReport {

        let mut report = TrainerEvaluationReport {
            total_samples: samples.len(),
            ..TrainerEvaluationReport::default()
        };

        for sample in samples {
            let expected = sample.label.trim().to_ascii_lowercase();
            if expected.is_empty() {
                report.skipped_samples += 1;
                continue;
            }

            report.evaluated_samples += 1;
            *report.per_label_total.entry(expected.clone()).or_insert(0) += 1;

            match self.predict_label(sample) {
                Some(predicted) if predicted == expected => {
                    report.correct_predictions += 1;
                    *report.per_label_correct.entry(expected.clone()).or_insert(0) += 1;
                    increment_confusion(&mut report.confusion_matrix, expected, predicted);
                }
                Some(predicted) => {
                    increment_confusion(&mut report.confusion_matrix, expected, predicted);
                }
                None => {
                    report.unknown_predictions += 1;
                    increment_confusion(
                        &mut report.confusion_matrix,
                        expected,
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

    fn save_checkpoint(&self, checkpoint_id: &str, dir: &Path) -> io::Result<Vec<PathBuf>> {
        fs::create_dir_all(dir)?;
        let checkpoint_path = dir.join(format!("{}.dense.json", checkpoint_id));
        let encoded = serde_json::to_vec(self).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to serialize dense baseline checkpoint '{}': {err}",
                    checkpoint_path.display()
                ),
            )
        })?;

        fs::write(&checkpoint_path, encoded)?;
        Ok(vec![checkpoint_path])
    }

    fn load_checkpoint(&mut self, checkpoint_id: &str, dir: &Path) -> io::Result<()> {
        let checkpoint_path = dir.join(format!("{}.dense.json", checkpoint_id));
        let bytes = fs::read(&checkpoint_path)?;
        let decoded: DenseTokenBaseline = serde_json::from_slice(&bytes).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to deserialize dense baseline checkpoint '{}': {err}",
                    checkpoint_path.display()
                ),
            )
        })?;

        *self = decoded;
        Ok(())
    }
}

fn increment_confusion(
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
    for row in report.confusion_matrix.values() {
        for predicted in row.keys() {
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
            .and_then(|r| r.get(&label))
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
    use crate::helpers::multimodal_controller::MultiModalInput;

    #[test]
    fn dense_baseline_trains_and_predicts_known_text() {
        let mut model = DenseTokenBaseline::new();
        let metadata = NodeMetadata::with_lang("en");
        let samples = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let report = model.train_epoch(
            samples.as_slice(),
            &metadata,
            TrainerBridgeTarget::Cognitive,
            DendriteType::Statement,
        );

        assert_eq!(report.trained_examples, 2);

        let eval = model.evaluate_dataset(samples.as_slice());
        assert!(eval.accuracy >= 1.0);
    }
}