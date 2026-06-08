use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::core::brain::MultiModalBrain;
use crate::core::nodenet::NodeMetadata;
use crate::training::metrics::compute_quality_metrics;
use crate::training::trainer::{
    TrainerBatchReport,
    TrainerBridgeTarget,
    TrainerEvaluationReport,
};
use crate::dendrites::text_dendrite::DendriteType;
use crate::helpers::multimodal_controller::MultiModalInput;

#[derive(Debug, Clone)]
pub struct SupervisedSample {
    pub label: String,
    pub content: MultiModalInput,
}

impl SupervisedSample {
    pub fn new(label: &str, content: MultiModalInput) -> Self {
        Self {
            label: label.to_string(),
            content,
        }
    }
}

pub trait TrainableModel {
    fn train_epoch(
        &mut self,
        samples: &[SupervisedSample],
        metadata: &NodeMetadata,
        target: TrainerBridgeTarget,
        dendrite_type: DendriteType,
    ) -> TrainerBatchReport;

    fn evaluate_dataset(&self, samples: &[SupervisedSample]) -> TrainerEvaluationReport;

    fn save_checkpoint(&self, checkpoint_id: &str, dir: &Path) -> io::Result<Vec<PathBuf>>;

    fn load_checkpoint(&mut self, checkpoint_id: &str, dir: &Path) -> io::Result<()>;
}

impl TrainableModel for MultiModalBrain {
    fn train_epoch(
        &mut self,
        samples: &[SupervisedSample],
        metadata: &NodeMetadata,
        target: TrainerBridgeTarget,
        dendrite_type: DendriteType,
    ) -> TrainerBatchReport {
        let mut grouped_indices: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, sample) in samples.iter().enumerate() {
            let normalized_label = sample.label.trim().to_ascii_lowercase();
            if normalized_label.is_empty() {
                continue;
            }
            grouped_indices.entry(normalized_label).or_default().push(index);
        }

        let mut report = TrainerBatchReport {
            total_examples: samples.len(),
            ..TrainerBatchReport::default()
        };

        for (label, sample_indices) in grouped_indices {
            let mut trained_for_label = 0usize;
            for sample_index in sample_indices {
                self.train_labeled_pattern(
                    &label,
                    &samples[sample_index].content,
                    metadata,
                    dendrite_type,
                    target,
                );
                report.trained_examples += 1;
                trained_for_label += 1;
            }

            if trained_for_label > 0 {
                report.per_label_counts.insert(label, trained_for_label);
            }
        }

        report.skipped_examples = report.total_examples.saturating_sub(report.trained_examples);
        report
    }

    fn evaluate_dataset(&self, samples: &[SupervisedSample]) -> TrainerEvaluationReport {
        if samples.is_empty() {
            TrainerEvaluationReport::default()
        } else {
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
                if let Some(count) = report.per_label_total.get_mut(&expected) {
                    *count += 1;
                } else {
                    report.per_label_total.insert(expected.clone(), 1);
                }

                match self.classify_pattern(&sample.content) {
                    Some(predicted) if predicted.label == expected => {
                        report.correct_predictions += 1;
                        *report.per_label_correct.entry(expected.clone()).or_insert(0) += 1;
                        increment_confusion_count_borrowed(
                            &mut report.confusion_matrix,
                            &expected,
                            predicted.label,
                        );
                    }
                    Some(predicted) => {
                        increment_confusion_count_borrowed(
                            &mut report.confusion_matrix,
                            &expected,
                            predicted.label,
                        );
                    }
                    None => {
                        report.unknown_predictions += 1;
                        increment_confusion_count_borrowed(
                            &mut report.confusion_matrix,
                            &expected,
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

    fn save_checkpoint(&self, checkpoint_id: &str, dir: &Path) -> io::Result<Vec<PathBuf>> {
        self.snapshot_instance_in_dir(checkpoint_id, dir)?;
        let bundle = self.snapshot_bundle_path_for_instance_in_dir(checkpoint_id, dir);
        Ok(vec![bundle])
    }

    fn load_checkpoint(&mut self, checkpoint_id: &str, dir: &Path) -> io::Result<()> {
        let status = self.load_snapshot_instance_in_dir(checkpoint_id, dir)?;

        match (
            status.cognitive_loaded,
            status.memory_loaded,
            status.classifier_loaded,
        ) {
            (true, true, true) => Ok(()),
            (false, false, false) => Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "checkpoint '{}' not found in '{}'",
                    checkpoint_id,
                    dir.display()
                ),
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "checkpoint '{}' is incomplete (expected cognitive, memory, classifier) in '{}'",
                    checkpoint_id,
                    dir.display()
                ),
            )),
        }
    }
}

fn increment_confusion_count_borrowed(
    matrix: &mut BTreeMap<String, BTreeMap<String, usize>>,
    expected: &str,
    predicted: String,
) {
    if let Some(row) = matrix.get_mut(expected) {
        *row.entry(predicted).or_insert(0) += 1;
    } else {
        let mut row = BTreeMap::new();
        row.insert(predicted, 1);
        matrix.insert(expected.to_string(), row);
    }
}