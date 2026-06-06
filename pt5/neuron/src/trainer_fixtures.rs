use neuralnet::core::model::SupervisedSample;
use neuralnet::core::nodenet::NodeMetadata;
use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::training::trainer::{TrainerBatch, TrainerEvaluationSample};
use neuralnet::training::training_loop::TrainingLoopConfig;
use std::path::PathBuf;

pub fn english_metadata() -> NodeMetadata {
    NodeMetadata::with_lang("en")
}

pub fn image_metadata() -> NodeMetadata {
    NodeMetadata::with_lang("img")
}

pub fn vertical_stripes_image_8x8() -> Vec<u8> {
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

pub fn horizontal_stripes_image_8x8() -> Vec<u8> {
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

pub fn diagonal_gradient_image_8x8() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for y in 0..8 {
        for x in 0..8 {
            bytes.push(((x + y) * 16) as u8);
        }
    }
    bytes
}

pub fn initial_trainer_batches() -> Vec<TrainerBatch> {
    vec![
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
    ]
}

pub fn eval_samples() -> Vec<TrainerEvaluationSample> {
    vec![
        TrainerEvaluationSample::new(
            "animal_cat",
            MultiModalInput::Text("cat on sofa".to_string()),
        ),
        TrainerEvaluationSample::new(
            "animal_dog",
            MultiModalInput::Text("dog in park".to_string()),
        ),
        TrainerEvaluationSample::new(
            "animal_bird",
            MultiModalInput::Text("bird in sky".to_string()),
        ),
    ]
}

pub fn train_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample::new(
            "animal_cat",
            MultiModalInput::Text("cat on mat".to_string()),
        ),
        SupervisedSample::new(
            "animal_cat",
            MultiModalInput::Text("cat on sofa".to_string()),
        ),
        SupervisedSample::new(
            "animal_dog",
            MultiModalInput::Text("dog in park".to_string()),
        ),
    ]
}

pub fn validation_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample::new(
            "animal_cat",
            MultiModalInput::Text("cat on sofa".to_string()),
        ),
        SupervisedSample::new(
            "animal_dog",
            MultiModalInput::Text("dog in park".to_string()),
        ),
        SupervisedSample::new(
            "animal_bird",
            MultiModalInput::Text("bird in sky".to_string()),
        ),
    ]
}

pub fn probe_inputs() -> Vec<MultiModalInput> {
    vec![
        MultiModalInput::Text("cat on mat".to_string()),
        MultiModalInput::Text("bird in sky".to_string()),
    ]
}

pub fn cnn_image_train_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample::new(
            "animal_cat",
            MultiModalInput::ImageBytes(vertical_stripes_image_8x8()),
        ),
        SupervisedSample::new(
            "animal_dog",
            MultiModalInput::ImageBytes(horizontal_stripes_image_8x8()),
        ),
    ]
}

pub fn cnn_image_validation_samples() -> Vec<SupervisedSample> {
    vec![
        SupervisedSample::new(
            "animal_cat",
            MultiModalInput::ImageBytes(vertical_stripes_image_8x8()),
        ),
        SupervisedSample::new(
            "animal_dog",
            MultiModalInput::ImageBytes(horizontal_stripes_image_8x8()),
        ),
        SupervisedSample::new(
            "animal_bird",
            MultiModalInput::ImageBytes(diagonal_gradient_image_8x8()),
        ),
    ]
}

pub fn cnn_image_probe_inputs() -> Vec<MultiModalInput> {
    vec![
        MultiModalInput::ImageBytes(vertical_stripes_image_8x8()),
        MultiModalInput::ImageBytes(diagonal_gradient_image_8x8()),
    ]
}

pub fn loop_config() -> TrainingLoopConfig {
    TrainingLoopConfig {
        epochs: 8,
        early_stopping_patience: Some(2),
        early_stopping_min_delta: 0.0001,
        checkpoint_dir: Some(PathBuf::from("./target/trainer_checkpoints")),
        checkpoint_every_n_epochs: Some(2),
        save_best_checkpoint: true,
        save_last_checkpoint: true,
        ..TrainingLoopConfig::default()
    }
}

pub fn resume_config(loop_config: &TrainingLoopConfig) -> TrainingLoopConfig {
    TrainingLoopConfig {
        epochs: 2,
        early_stopping_patience: Some(1),
        early_stopping_min_delta: 0.0001,
        checkpoint_dir: loop_config.checkpoint_dir.clone(),
        checkpoint_every_n_epochs: None,
        save_best_checkpoint: false,
        save_last_checkpoint: false,
        ..TrainingLoopConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnn_image_fixture_shapes_are_consistent() {
        assert_eq!(vertical_stripes_image_8x8().len(), 64);
        assert_eq!(horizontal_stripes_image_8x8().len(), 64);
        assert_eq!(diagonal_gradient_image_8x8().len(), 64);
    }

    #[test]
    fn cnn_image_sample_builders_include_expected_modal_inputs() {
        let train = cnn_image_train_samples();
        let validation = cnn_image_validation_samples();
        let probes = cnn_image_probe_inputs();

        assert_eq!(train.len(), 2);
        assert_eq!(validation.len(), 3);
        assert_eq!(probes.len(), 2);

        assert!(matches!(train[0].content, MultiModalInput::ImageBytes(_)));
        assert!(matches!(train[1].content, MultiModalInput::ImageBytes(_)));
        assert!(matches!(validation[0].content, MultiModalInput::ImageBytes(_)));
        assert!(matches!(validation[1].content, MultiModalInput::ImageBytes(_)));
        assert!(matches!(validation[2].content, MultiModalInput::ImageBytes(_)));
        assert!(matches!(probes[0], MultiModalInput::ImageBytes(_)));
        assert!(matches!(probes[1], MultiModalInput::ImageBytes(_)));
    }
}
