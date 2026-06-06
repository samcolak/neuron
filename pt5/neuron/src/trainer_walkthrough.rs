use neuralnet::core::brain::MultiModalNeuralNetwork;
use crate::trainer_presentation::{
    format_prediction,
    print_confusion_matrix,
    print_label_metrics,
    print_loop_summary,
};
use crate::trainer_fixtures::{
    cnn_image_probe_inputs,
    cnn_image_train_samples,
    cnn_image_validation_samples,
    english_metadata,
    eval_samples,
    image_metadata,
    initial_trainer_batches,
    loop_config,
    probe_inputs,
    resume_config,
    train_samples,
    validation_samples,
};
use neuralnet::training::trainer::{
    TrainerBridgeTarget,
};
use neuralnet::training::training_loop::TrainingLoopConfig;
use neuralnet::dendrites::text_dendrite::DendriteType;
use neuralnet::helpers::multimodal_controller::MultiModalInput;

pub fn run_trainer_walkthrough() {
    
    let mut network = MultiModalNeuralNetwork::new_multimodal();
    let metadata = english_metadata();

    println!("\nTrainer walkthrough");
    println!("  step 1: train a single labeled pattern");

    network.train_labeled_pattern(
        "animal_cat",
        &MultiModalInput::Text("cat on mat".to_string()),
        &metadata,
        DendriteType::Statement,
        TrainerBridgeTarget::Cognitive,
    );

    let single_prediction = network.classify_pattern(&MultiModalInput::Text("cat on mat".to_string()));
    println!(
        "    classify('cat on mat') -> {}",
        format_prediction(single_prediction.as_ref())
    );

    println!("  step 2: batch train multiple labels");
    let batches = initial_trainer_batches();

    let train_report = network.train_labeled_batches(
        &batches,
        &metadata,
        DendriteType::Statement,
        TrainerBridgeTarget::Cognitive,
    );

    println!(
        "    trained={} skipped={} labels={:?}",
        train_report.trained_examples,
        train_report.skipped_examples,
        train_report.per_label_counts
    );

    println!("  step 3: evaluate and inspect confusion matrix");
    let eval_samples = eval_samples();

    let eval_report = network.evaluate_labeled_samples(&eval_samples);
    println!(
        "    accuracy={:.3} correct={} unknown={}",
        eval_report.accuracy,
        eval_report.correct_predictions,
        eval_report.unknown_predictions
    );
    println!(
        "    macro: precision={:.3} recall={:.3} f1={:.3}",
        eval_report.macro_precision,
        eval_report.macro_recall,
        eval_report.macro_f1
    );
    println!(
        "    micro: precision={:.3} recall={:.3} f1={:.3}",
        eval_report.micro_precision,
        eval_report.micro_recall,
        eval_report.micro_f1
    );

    print_confusion_matrix(&eval_report);
    print_label_metrics(&eval_report);

    println!("  step 4: run epoch training loop with early stopping");
    let train_samples = train_samples();
    let validation_samples = validation_samples();
    let loop_config = loop_config();
    let probe_inputs = probe_inputs();

    let pipeline_report = network.run_supervised_pipeline(
        train_samples.as_slice(),
        validation_samples.as_slice(),
        &metadata,
        &loop_config,
        probe_inputs.as_slice(),
    );
    let loop_report = &pipeline_report.training_loop_report;

    print_loop_summary(loop_report);

    println!(
        "    pipeline final eval: accuracy={:.3} micro_f1={:.3}",
        pipeline_report.final_evaluation.accuracy,
        pipeline_report.final_evaluation.micro_f1
    );

    println!("  step 5: resume from 'best' checkpoint and run a short continuation");
    let mut resumed_network = MultiModalNeuralNetwork::new_multimodal();
    let probe = MultiModalInput::Text("cat on mat".to_string());
    let probe_before_load = resumed_network.classify_pattern(&probe);
    println!(
        "    probe before checkpoint load: {}",
        format_prediction(probe_before_load.as_ref())
    );

    if let Some(checkpoint_dir) = loop_config.checkpoint_dir.as_ref() {
        match resumed_network.load_snapshot_instance_in_dir("best", checkpoint_dir.as_path()) {
            Ok(status) => {
                let probe_after_load = resumed_network.classify_pattern(&probe);
                println!(
                    "    probe after checkpoint load: {} (cognitive_loaded={} memory_loaded={} classifier_loaded={})",
                    format_prediction(probe_after_load.as_ref()),
                    status.cognitive_loaded,
                    status.memory_loaded,
                    status.classifier_loaded,
                );
            }
            Err(err) => {
                println!("    probe load from checkpoint failed: {}", err);
            }
        }
    }

    let resume_config = resume_config(&loop_config);

    let resumed_pipeline_report = resumed_network.resume_then_run_supervised_pipeline(
        train_samples.as_slice(),
        validation_samples.as_slice(),
        &metadata,
        &resume_config,
        "best",
        probe_inputs.as_slice(),
    );
    let resumed_report = &resumed_pipeline_report.training_loop_report;

    println!(
        "    resumed: from={} epochs_ran={} best_epoch={:?} best_acc={:.3}",
        resumed_report
            .resumed_from_checkpoint
            .as_deref()
            .unwrap_or("<none>"),
        resumed_report.epochs_ran,
        resumed_report.best_epoch,
        resumed_report.best_validation_accuracy
    );
    println!(
        "    resumed final eval: accuracy={:.3} micro_f1={:.3}",
        resumed_pipeline_report.final_evaluation.accuracy,
        resumed_pipeline_report.final_evaluation.micro_f1
    );

    println!("  step 6: cnn image path demo on app pipeline");
    let mut image_network = MultiModalNeuralNetwork::new_multimodal();
    image_network.enable_default_cnn_image_path();
    image_network.classifier_mut().set_min_confidence(0.95);

    let image_metadata = image_metadata();
    let image_train = cnn_image_train_samples();
    let image_validation = cnn_image_validation_samples();
    let image_probes = cnn_image_probe_inputs();

    let image_pre = image_network.classify_pattern(&image_probes[0]);
    println!(
        "    pre-train image probe -> {}",
        format_prediction(image_pre.as_ref())
    );

    let image_config = TrainingLoopConfig {
        epochs: 1,
        ..TrainingLoopConfig::default()
    };

    let image_report = image_network.run_supervised_pipeline(
        image_train.as_slice(),
        image_validation.as_slice(),
        &image_metadata,
        &image_config,
        image_probes.as_slice(),
    );

    println!(
        "    cnn image final eval: accuracy={:.3} micro_f1={:.3}",
        image_report.final_evaluation.accuracy,
        image_report.final_evaluation.micro_f1
    );
    print_confusion_matrix(&image_report.final_evaluation);

    println!(
        "    post-train image probe -> {}",
        format_prediction(image_report.probe_predictions[0].prediction.as_ref())
    );

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnn_image_walkthrough_flow_improves_probe_prediction() {
        let mut network = MultiModalNeuralNetwork::new_multimodal();
        network.enable_default_cnn_image_path();
        network.classifier_mut().set_min_confidence(0.95);

        let metadata = image_metadata();
        let train = cnn_image_train_samples();
        let validation = cnn_image_validation_samples();
        let probes = cnn_image_probe_inputs();

        let pre = network.classify_pattern(&probes[0]);
        assert!(pre.is_none());

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
            report.probe_predictions[0]
                .prediction
                .as_ref()
                .map(|prediction| prediction.label.as_str()),
            Some("animal_cat")
        );
    }
}
