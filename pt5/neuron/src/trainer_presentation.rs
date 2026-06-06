use neuralnet::helpers::pattern_classifier::ClassificationResult;
use neuralnet::training::trainer::TrainerEvaluationReport;
use neuralnet::training::training_loop::TrainingLoopReport;

pub fn format_prediction(prediction: Option<&ClassificationResult>) -> String {
    prediction
        .map(|result| format!("{} ({:.3})", result.label, result.score))
        .unwrap_or_else(|| "<unknown>".to_string())
}

pub fn print_confusion_matrix(report: &TrainerEvaluationReport) {
    println!("    confusion matrix (expected -> predicted=count)");

    for (expected, predictions) in &report.confusion_matrix {
        let columns = predictions
            .iter()
            .map(|(predicted, count)| format!("{}={}", predicted, count))
            .collect::<Vec<String>>()
            .join(", ");

        println!("      {} -> {}", expected, columns);
    }
}

pub fn print_label_metrics(report: &TrainerEvaluationReport) {
    println!("    label metrics (precision / recall / f1 / support)");

    for (label, metrics) in &report.per_label_metrics {
        println!(
            "      {} -> {:.3} / {:.3} / {:.3} / {}",
            label, metrics.precision, metrics.recall, metrics.f1, metrics.support
        );
    }
}

pub fn print_loop_summary(loop_report: &TrainingLoopReport) {
    println!(
        "    loop epochs_ran={} best_epoch={:?} best_acc={:.3}",
        loop_report.epochs_ran,
        loop_report.best_epoch,
        loop_report.best_validation_accuracy
    );
    println!(
        "    checkpoints: epoch_saved={} best_files={} last_files={} resumed_from={}",
        loop_report.epoch_checkpoint_paths.len(),
        loop_report.best_checkpoint_paths.len(),
        loop_report.last_checkpoint_paths.len(),
        loop_report
            .resumed_from_checkpoint
            .as_deref()
            .unwrap_or("<none>")
    );

    for epoch in &loop_report.history {
        println!(
            "      epoch={} trained={} skipped={} val_acc={:.3} val_macro_f1={:.3} val_micro_f1={:.3}",
            epoch.epoch,
            epoch.trained_examples,
            epoch.skipped_examples,
            epoch.validation_accuracy,
            epoch.validation_macro_f1,
            epoch.validation_micro_f1,
        );
    }
}
