use crate::core::brain::MultiModalBrain;
use crate::core::model::{SupervisedSample, TrainableModel};
use crate::core::nodenet::NodeMetadata;
use crate::training::trainer::{TrainerBridgeTarget, TrainerEvaluationReport};
use crate::dendrites::text_dendrite::DendriteType;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TrainingLoopConfig {
    pub epochs: usize,
    pub target: TrainerBridgeTarget,
    pub dendrite_type: DendriteType,
    pub early_stopping_patience: Option<usize>,
    pub early_stopping_min_delta: f64,
    pub checkpoint_dir: Option<PathBuf>,
    pub checkpoint_every_n_epochs: Option<usize>,
    pub start_from_checkpoint_id: Option<String>,
    pub save_best_checkpoint: bool,
    pub save_last_checkpoint: bool,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            target: TrainerBridgeTarget::Cognitive,
            dendrite_type: DendriteType::Statement,
            early_stopping_patience: None,
            early_stopping_min_delta: 0.0,
            checkpoint_dir: None,
            checkpoint_every_n_epochs: None,
            start_from_checkpoint_id: None,
            save_best_checkpoint: true,
            save_last_checkpoint: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EpochTrainingSummary {
    pub epoch: usize,
    pub trained_examples: usize,
    pub skipped_examples: usize,
    pub validation_accuracy: f64,
    pub validation_macro_f1: f64,
    pub validation_micro_f1: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TrainingLoopReport {
    pub epochs_ran: usize,
    pub best_epoch: Option<usize>,
    pub best_validation_accuracy: f64,
    pub history: Vec<EpochTrainingSummary>,
    pub final_validation: Option<TrainerEvaluationReport>,
    pub epoch_checkpoint_paths: BTreeMap<usize, Vec<PathBuf>>,
    pub best_checkpoint_paths: Vec<PathBuf>,
    pub last_checkpoint_paths: Vec<PathBuf>,
    pub resumed_from_checkpoint: Option<String>,
}

impl MultiModalBrain {

    pub fn run_supervised_training_loop(
        &mut self,
        train_samples: &[SupervisedSample],
        validation_samples: &[SupervisedSample],
        metadata: &NodeMetadata,
        config: &TrainingLoopConfig,
    ) -> TrainingLoopReport {
        run_supervised_training_loop_for_model(self, train_samples, validation_samples, metadata, config)
    }
}

pub fn run_supervised_training_loop_for_model<M: TrainableModel>(
    model: &mut M,
    train_samples: &[SupervisedSample],
    validation_samples: &[SupervisedSample],
    metadata: &NodeMetadata,
    config: &TrainingLoopConfig,
) -> TrainingLoopReport {
    
    let mut report = TrainingLoopReport {
        best_validation_accuracy: -1.0,
        ..TrainingLoopReport::default()
    };

    if config.epochs == 0 || train_samples.is_empty() {
        return report;
    }

    if let Some(checkpoint_dir) = &config.checkpoint_dir {
        let _ = fs::create_dir_all(checkpoint_dir);

        if let Some(checkpoint_id) = &config.start_from_checkpoint_id
            && model
                .load_checkpoint(checkpoint_id, checkpoint_dir.as_path())
                .is_ok()
        {
            report.resumed_from_checkpoint = Some(checkpoint_id.clone());
        }
    }

    let mut stale_epochs = 0usize;

    for epoch in 1..=config.epochs {
        let train_report = model.train_epoch(
            train_samples,
            metadata,
            config.target,
            config.dendrite_type,
        );

        let validation_report = if validation_samples.is_empty() {
            TrainerEvaluationReport::default()
        } else {
            model.evaluate_dataset(validation_samples)
        };

        let summary = EpochTrainingSummary {
            epoch,
            trained_examples: train_report.trained_examples,
            skipped_examples: train_report.skipped_examples,
            validation_accuracy: validation_report.accuracy,
            validation_macro_f1: validation_report.macro_f1,
            validation_micro_f1: validation_report.micro_f1,
        };

        report.history.push(summary);
        report.epochs_ran = epoch;
        report.final_validation = Some(validation_report.clone());

        let improved = validation_report.accuracy
            > (report.best_validation_accuracy + config.early_stopping_min_delta);

        if improved {
            report.best_validation_accuracy = validation_report.accuracy;
            report.best_epoch = Some(epoch);
            stale_epochs = 0;

            if config.save_best_checkpoint
                && let Some(checkpoint_dir) = &config.checkpoint_dir
                && let Ok(paths) = model.save_checkpoint("best", checkpoint_dir.as_path())
            {
                report.best_checkpoint_paths = paths;
            }
        } else {
            stale_epochs += 1;
        }

        if let Some(every_n) = config.checkpoint_every_n_epochs
            && every_n > 0
            && epoch % every_n == 0
            && let Some(checkpoint_dir) = &config.checkpoint_dir
        {
            let checkpoint_id = format!("epoch_{:04}", epoch);
            if let Ok(paths) = model.save_checkpoint(&checkpoint_id, checkpoint_dir.as_path()) {
                report.epoch_checkpoint_paths.insert(epoch, paths);
            }
        }

        if let Some(patience) = config.early_stopping_patience
            && stale_epochs >= patience
        {
            break;
        }
    }

    if config.save_last_checkpoint
        && let Some(checkpoint_dir) = &config.checkpoint_dir
        && let Ok(paths) = model.save_checkpoint("last", checkpoint_dir.as_path())
    {
        report.last_checkpoint_paths = paths;
    }

    report
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::model::TrainableModel;
    use crate::core::brain::MultiModalNeuralNetwork;
    use crate::training::dense_baseline::DenseTokenBaseline;
    use crate::helpers::multimodal_controller::MultiModalInput;
    use std::fs;
    use std::path::PathBuf;

    fn checkpoint_test_dir(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("./target/training_loop_checkpoints");
        path.push(test_name);
        path
    }

    fn cleanup_checkpoint_test_dir(test_name: &str) {
        let _ = fs::remove_dir_all(checkpoint_test_dir(test_name));
    }

    #[test]
    fn training_loop_runs_and_reports_history() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
            SupervisedSample::new("animal_bird", MultiModalInput::Text("bird in sky".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 3,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert_eq!(report.epochs_ran, 3);
        assert_eq!(report.history.len(), 3);
        assert!(report.best_epoch.is_some());
        assert!(
            report
                .final_validation
                .as_ref()
                .map(|final_eval| final_eval.accuracy >= 0.60)
                .unwrap_or(false)
        );
    }

    #[test]
    fn training_loop_stops_early_when_patience_reached() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 10,
            early_stopping_patience: Some(2),
            early_stopping_min_delta: 0.0001,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert!(report.epochs_ran < 10);
        assert!(!report.history.is_empty());
    
    }

    #[test]
    fn generic_training_loop_runs_with_dense_baseline_model() {
        let mut model = DenseTokenBaseline::new();
        let metadata = NodeMetadata::with_lang("en");

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
            SupervisedSample::new("animal_bird", MultiModalInput::Text("bird in sky".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 4,
            early_stopping_patience: Some(2),
            early_stopping_min_delta: 0.0001,
            ..TrainingLoopConfig::default()
        };

        let report = run_supervised_training_loop_for_model(
            &mut model,
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert!(report.epochs_ran >= 1);
        assert!(!report.history.is_empty());
        assert!(report.final_validation.is_some());
    }

    #[test]
    fn training_loop_persists_epoch_best_and_last_checkpoints() {
        let test_name = "checkpoint_lifecycle";
        cleanup_checkpoint_test_dir(test_name);

        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
            SupervisedSample::new("animal_bird", MultiModalInput::Text("bird in sky".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 4,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            checkpoint_every_n_epochs: Some(2),
            save_best_checkpoint: true,
            save_last_checkpoint: true,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert!(report.epoch_checkpoint_paths.contains_key(&2));
        assert!(!report.best_checkpoint_paths.is_empty());
        assert!(!report.last_checkpoint_paths.is_empty());
        assert!(report
            .best_checkpoint_paths
            .iter()
            .all(|path| path.exists()));
        assert!(report
            .last_checkpoint_paths
            .iter()
            .all(|path| path.exists()));

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn training_loop_resume_from_best_checkpoint_restores_predictions() {
        let test_name = "resume_equivalence_dense";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);
        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];
        let probe = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
            SupervisedSample::new("animal_bird", MultiModalInput::Text("bird in sky".to_string())),
        ];

        let mut model = DenseTokenBaseline::new();
        let train_config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            save_best_checkpoint: true,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let initial_report = run_supervised_training_loop_for_model(
            &mut model,
            train.as_slice(),
            probe.as_slice(),
            &metadata,
            &train_config,
        );

        assert!(!initial_report.best_checkpoint_paths.is_empty());

        let expected_eval = model.evaluate_dataset(probe.as_slice());

        let mut resumed_model = DenseTokenBaseline::new();
        resumed_model
            .load_checkpoint("best", checkpoint_dir.as_path())
            .unwrap_or_else(|err| panic!("failed to load best checkpoint for resume test: {err}"));

        let resumed_eval = resumed_model.evaluate_dataset(probe.as_slice());

        assert!((resumed_eval.accuracy - expected_eval.accuracy).abs() < f64::EPSILON);
        assert!((resumed_eval.micro_f1 - expected_eval.micro_f1).abs() < f64::EPSILON);
        assert_eq!(resumed_eval.confusion_matrix, expected_eval.confusion_matrix);

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn training_loop_does_not_resume_when_checkpoint_id_is_missing() {
        let test_name = "missing_checkpoint_id_dense";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);
        let mut model = DenseTokenBaseline::new();

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            start_from_checkpoint_id: Some("does_not_exist".to_string()),
            checkpoint_every_n_epochs: None,
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let report = run_supervised_training_loop_for_model(
            &mut model,
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert_eq!(report.resumed_from_checkpoint, None);
        assert_eq!(report.epochs_ran, 1);

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn training_loop_does_not_resume_when_checkpoint_payload_is_corrupt() {
        let test_name = "corrupt_checkpoint_dense";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);
        fs::create_dir_all(&checkpoint_dir)
            .unwrap_or_else(|err| panic!("failed to create checkpoint dir for corrupt test: {err}"));

        let corrupt_checkpoint = checkpoint_dir.join("best.dense.json");
        fs::write(&corrupt_checkpoint, b"{ definitely-not-valid-json")
            .unwrap_or_else(|err| panic!("failed to write corrupt checkpoint: {err}"));

        let mut model = DenseTokenBaseline::new();
        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            start_from_checkpoint_id: Some("best".to_string()),
            checkpoint_every_n_epochs: None,
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let report = run_supervised_training_loop_for_model(
            &mut model,
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert_eq!(report.resumed_from_checkpoint, None);
        assert_eq!(report.epochs_ran, 1);

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn training_loop_metrics_fixture_remains_stable() {
        let metadata = NodeMetadata::with_lang("en");
        let mut model = DenseTokenBaseline::new();

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
            SupervisedSample::new("animal_bird", MultiModalInput::Text("bird in sky".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            ..TrainingLoopConfig::default()
        };

        let report = run_supervised_training_loop_for_model(
            &mut model,
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        let final_eval = report
            .final_validation
            .unwrap_or_else(|| panic!("training loop should produce final validation report"));

        assert!((final_eval.accuracy - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((final_eval.micro_precision - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((final_eval.micro_recall - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((final_eval.micro_f1 - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert_eq!(
            final_eval
                .confusion_matrix
                .get("animal_cat")
                .and_then(|row| row.get("animal_cat")),
            Some(&1usize)
        );
        assert_eq!(
            final_eval
                .confusion_matrix
                .get("animal_dog")
                .and_then(|row| row.get("animal_dog")),
            Some(&1usize)
        );
        assert_eq!(
            final_eval
                .confusion_matrix
                .get("animal_bird")
                .and_then(|row| row.get("animal_dog")),
            Some(&1usize)
        );
    }

    #[test]
    fn brain_training_loop_does_not_resume_when_checkpoint_id_is_missing() {
        let test_name = "missing_checkpoint_id_brain";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);
        let mut network = MultiModalNeuralNetwork::new_multimodal();

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            start_from_checkpoint_id: Some("does_not_exist".to_string()),
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert_eq!(report.resumed_from_checkpoint, None);
        assert_eq!(report.epochs_ran, 1);

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn brain_training_loop_does_not_resume_when_checkpoint_payload_is_corrupt() {
        let test_name = "corrupt_checkpoint_brain";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);
        fs::create_dir_all(&checkpoint_dir)
            .unwrap_or_else(|err| panic!("failed to create checkpoint dir for brain corrupt test: {err}"));

        let network_for_paths = MultiModalNeuralNetwork::new_multimodal();
        let bundle_path =
            network_for_paths.snapshot_bundle_path_for_instance_in_dir("best", checkpoint_dir.as_path());

        fs::write(bundle_path, b"invalid-brain-checkpoint")
        .unwrap_or_else(|err| panic!("failed to write corrupt snapshot bundle: {err}"));

        let mut network = MultiModalNeuralNetwork::new_multimodal();
        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            start_from_checkpoint_id: Some("best".to_string()),
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
        );

        assert_eq!(report.resumed_from_checkpoint, None);
        assert_eq!(report.epochs_ran, 1);

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn brain_training_loop_does_not_resume_when_classifier_file_is_missing() {
        let test_name = "missing_classifier_checkpoint_brain";
        cleanup_checkpoint_test_dir(test_name);

        let metadata = NodeMetadata::with_lang("en");
        let checkpoint_dir = checkpoint_test_dir(test_name);

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::Text("cat on mat".to_string())),
            SupervisedSample::new("animal_dog", MultiModalInput::Text("dog in park".to_string())),
        ];

        let mut checkpoint_source = MultiModalNeuralNetwork::new_multimodal();
        let save_config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            save_best_checkpoint: true,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let save_report = checkpoint_source.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &save_config,
        );
        assert!(!save_report.best_checkpoint_paths.is_empty());

        let network_for_paths = MultiModalNeuralNetwork::new_multimodal();
        let bundle_path =
            network_for_paths.snapshot_bundle_path_for_instance_in_dir("best", checkpoint_dir.as_path());
        fs::remove_file(&bundle_path)
            .unwrap_or_else(|err| panic!("failed to remove snapshot bundle file: {err}"));

        let mut resumed_network = MultiModalNeuralNetwork::new_multimodal();
        let resume_config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            start_from_checkpoint_id: Some("best".to_string()),
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let resume_report = resumed_network.run_supervised_training_loop(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &resume_config,
        );

        assert_eq!(resume_report.resumed_from_checkpoint, None);
        assert_eq!(resume_report.epochs_ran, 1);

        cleanup_checkpoint_test_dir(test_name);
    }

}