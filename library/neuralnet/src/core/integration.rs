use crate::core::brain::MultiModalBrain;
use crate::core::model::SupervisedSample;
use crate::core::nodenet::NodeMetadata;
use crate::helpers::multimodal_controller::MultiModalInput;
use crate::helpers::pattern_classifier::ClassificationResult;
use crate::training::trainer::{TrainerEvaluationReport, TrainerEvaluationSample};
use crate::training::training_loop::{TrainingLoopConfig, TrainingLoopReport};

#[derive(Debug, Clone)]
pub struct ProbePrediction {
    pub input: MultiModalInput,
    pub prediction: Option<ClassificationResult>,
}

#[derive(Debug, Clone)]
pub struct BrainSupervisedPipelineReport {
    pub training_loop_report: TrainingLoopReport,
    pub final_evaluation: TrainerEvaluationReport,
    pub probe_predictions: Vec<ProbePrediction>,
}

impl MultiModalBrain {

    pub fn run_supervised_pipeline(
        &mut self,
        train_samples: &[SupervisedSample],
        validation_samples: &[SupervisedSample],
        metadata: &NodeMetadata,
        config: &TrainingLoopConfig,
        probe_inputs: &[MultiModalInput],
    ) -> BrainSupervisedPipelineReport {

        let training_loop_report = self.run_supervised_training_loop(
            train_samples,
            validation_samples,
            metadata,
            config,
        );

        let final_evaluation = if validation_samples.is_empty() {
            TrainerEvaluationReport::default()
        } else {
            let eval_samples: Vec<TrainerEvaluationSample> = validation_samples
                .iter()
                .map(|sample| {
                    TrainerEvaluationSample::new(
                        sample.label.as_str(),
                        sample.content.clone(),
                    )
                })
                .collect();

            self.evaluate_labeled_samples(eval_samples.as_slice())
        };

        let probe_predictions = probe_inputs
            .iter()
            .map(|input| ProbePrediction {
                input: input.clone(),
                prediction: self.classify_pattern(input),
            })
            .collect();

        BrainSupervisedPipelineReport {
            training_loop_report,
            final_evaluation,
            probe_predictions,
        }

    }

    pub fn resume_then_run_supervised_pipeline(
        &mut self,
        train_samples: &[SupervisedSample],
        validation_samples: &[SupervisedSample],
        metadata: &NodeMetadata,
        config: &TrainingLoopConfig,
        checkpoint_id: &str,
        probe_inputs: &[MultiModalInput],
    ) -> BrainSupervisedPipelineReport {

        let mut resumed_config = config.clone();
        resumed_config.start_from_checkpoint_id = Some(checkpoint_id.to_string());

        self.run_supervised_pipeline(
            train_samples,
            validation_samples,
            metadata,
            &resumed_config,
            probe_inputs,
        )
        
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain::MultiModalNeuralNetwork;
    use std::fs;
    use std::path::PathBuf;

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

    fn checkpoint_test_dir(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("./target/integration_pipeline_checkpoints");
        path.push(test_name);
        path
    }

    fn cleanup_checkpoint_test_dir(test_name: &str) {
        let _ = fs::remove_dir_all(checkpoint_test_dir(test_name));
    }

    #[test]
    fn supervised_pipeline_reports_final_eval_and_probe_predictions() {
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

        let probes = vec![
            MultiModalInput::Text("cat on mat".to_string()),
            MultiModalInput::Text("bird in sky".to_string()),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_pipeline(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
            probes.as_slice(),
        );

        assert_eq!(report.training_loop_report.epochs_ran, 1);
        assert!((report.final_evaluation.accuracy - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert_eq!(report.probe_predictions.len(), 2);
        assert_eq!(
            report.probe_predictions[0]
                .prediction
                .as_ref()
                .map(|p| p.label.clone()),
            Some("animal_cat".to_string())
        );
    }

    #[test]
    fn resume_pipeline_marks_checkpoint_source_when_best_exists() {
        let test_name = "resume_pipeline";
        cleanup_checkpoint_test_dir(test_name);

        let mut initial = MultiModalNeuralNetwork::new_multimodal();
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

        let train_config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            save_best_checkpoint: true,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let _ = initial.run_supervised_pipeline(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &train_config,
            &[],
        );

        let mut resumed = MultiModalNeuralNetwork::new_multimodal();
        let resume_config = TrainingLoopConfig {
            epochs: 1,
            checkpoint_dir: Some(checkpoint_dir.clone()),
            save_best_checkpoint: false,
            save_last_checkpoint: false,
            ..TrainingLoopConfig::default()
        };

        let report = resumed.resume_then_run_supervised_pipeline(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &resume_config,
            "best",
            &[],
        );

        assert_eq!(
            report.training_loop_report.resumed_from_checkpoint,
            Some("best".to_string())
        );

        cleanup_checkpoint_test_dir(test_name);
    }

    #[test]
    fn supervised_pipeline_with_cnn_image_path_improves_image_classification() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        network.enable_default_cnn_image_path();
        network.classifier_mut().set_min_confidence(0.95);

        let metadata = NodeMetadata::with_lang("img");
        let cat = vertical_stripes_image_8x8();
        let dog = horizontal_stripes_image_8x8();
        let unknown = diagonal_gradient_image_8x8();

        let pre_training_prediction =
            network.classify_pattern(&MultiModalInput::ImageBytes(cat.clone()));
        assert!(pre_training_prediction.is_none());

        let train = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::ImageBytes(cat.clone())),
            SupervisedSample::new("animal_dog", MultiModalInput::ImageBytes(dog.clone())),
        ];

        let validation = vec![
            SupervisedSample::new("animal_cat", MultiModalInput::ImageBytes(cat.clone())),
            SupervisedSample::new("animal_dog", MultiModalInput::ImageBytes(dog.clone())),
            SupervisedSample::new("animal_bird", MultiModalInput::ImageBytes(unknown.clone())),
        ];

        let probes = vec![
            MultiModalInput::ImageBytes(cat.clone()),
            MultiModalInput::ImageBytes(unknown),
        ];

        let config = TrainingLoopConfig {
            epochs: 1,
            ..TrainingLoopConfig::default()
        };

        let report = network.run_supervised_pipeline(
            train.as_slice(),
            validation.as_slice(),
            &metadata,
            &config,
            probes.as_slice(),
        );

        assert!(report.final_evaluation.accuracy >= (2.0 / 3.0));
        assert!(report.final_evaluation.micro_f1 >= 0.80);
        assert_eq!(
            report
                .final_evaluation
                .confusion_matrix
                .get("animal_cat")
                .and_then(|row| row.get("animal_cat")),
            Some(&1usize)
        );
        assert_eq!(
            report
                .final_evaluation
                .confusion_matrix
                .get("animal_dog")
                .and_then(|row| row.get("animal_dog")),
            Some(&1usize)
        );
        assert_eq!(
            report.probe_predictions[0]
                .prediction
                .as_ref()
                .map(|prediction| prediction.label.clone()),
            Some("animal_cat".to_string())
        );
    }
}
