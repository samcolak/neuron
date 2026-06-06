use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::core::brain::MultiModalBrain;
use crate::core::nodenet::NodeMetadata;
use crate::training::trainer::{
    TrainerBatch,
    TrainerBatchReport,
    TrainerBridgeTarget,
    TrainerEvaluationReport,
    TrainerEvaluationSample,
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
        let batches = build_batches_from_samples(samples);
        self.train_labeled_batches(batches.as_slice(), metadata, dendrite_type, target)
    }

    fn evaluate_dataset(&self, samples: &[SupervisedSample]) -> TrainerEvaluationReport {
        let eval_samples = build_eval_samples(samples);
        if eval_samples.is_empty() {
            TrainerEvaluationReport::default()
        } else {
            self.evaluate_labeled_samples(eval_samples.as_slice())
        }
    }

    fn save_checkpoint(&self, checkpoint_id: &str, dir: &Path) -> io::Result<Vec<PathBuf>> {
        self.snapshot_instance_in_dir(checkpoint_id, dir)?;
        let (cognitive, memory, classifier) =
            self.snapshot_paths_for_instance_in_dir(checkpoint_id, dir);
        Ok(vec![cognitive, memory, classifier])
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

fn build_batches_from_samples(samples: &[SupervisedSample]) -> Vec<TrainerBatch> {
    let mut grouped: BTreeMap<String, Vec<MultiModalInput>> = BTreeMap::new();

    for sample in samples {
        let normalized_label = sample.label.trim().to_ascii_lowercase();
        if normalized_label.is_empty() {
            continue;
        }

        grouped
            .entry(normalized_label)
            .or_default()
            .push(sample.content.clone());
    }

    grouped
        .into_iter()
        .map(|(label, grouped_samples)| TrainerBatch::new(&label, grouped_samples))
        .collect()
}

fn build_eval_samples(samples: &[SupervisedSample]) -> Vec<TrainerEvaluationSample> {
    samples
        .iter()
        .map(|sample| TrainerEvaluationSample::new(&sample.label, sample.content.clone()))
        .collect()
}