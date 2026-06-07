use crate::trainer_fixtures::{
    diagonal_gradient_image_8x8,
    horizontal_stripes_image_8x8,
    vertical_stripes_image_8x8,
};
use crate::trainer_presentation::{print_confusion_matrix, print_label_metrics};
use neuralnet::cnn::classifier::CnnImageClassifier;
use neuralnet::cnn::cnn_trainer::{CnnEvaluationSample, CnnTrainerBatch};

pub fn run_cnn_classifier_walkthrough() {
    println!("\nCNN classifier walkthrough");

    let mut classifier = CnnImageClassifier::new(
        vec!["animal_cat".to_string(), "animal_dog".to_string()],
        16,
        16,
        0.2,
    )
    .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
    classifier.set_min_confidence(1.0);

    let cat_image = vertical_stripes_image_8x8();
    let dog_image = horizontal_stripes_image_8x8();

    let pre = classifier
        .predict_with_confidence(cat_image.as_slice())
        .ok()
        .flatten();

    println!(
        "  pre-train prediction(cat) -> {}",
        pre.map(|value| format!("{} ({:.3})", value.0, value.1))
            .unwrap_or_else(|| "<unknown>".to_string())
    );

    let batches = vec![
        CnnTrainerBatch::new(
            "animal_cat",
            vec![cat_image.clone(), vertical_stripes_image_8x8()],
        ),
        CnnTrainerBatch::new(
            "animal_dog",
            vec![dog_image.clone(), horizontal_stripes_image_8x8()],
        ),
    ];

    for _ in 0..40 {
        let _ = classifier.train_labeled_image_batches(batches.as_slice());
    }

    classifier.set_min_confidence(0.0);

    let eval = vec![
        CnnEvaluationSample::new("animal_cat", cat_image.clone()),
        CnnEvaluationSample::new("animal_dog", dog_image.clone()),
        CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
    ];

    let report = classifier.evaluate_labeled_images(eval.as_slice());

    println!(
        "  final eval: accuracy={:.3} micro_f1={:.3}",
        report.accuracy,
        report.micro_f1
    );
    print_confusion_matrix(&report);
    print_label_metrics(&report);

    let post = classifier
        .predict_with_confidence(cat_image.as_slice())
        .ok()
        .flatten();

    println!(
        "  post-train prediction(cat) -> {}",
        post.map(|value| format!("{} ({:.3})", value.0, value.1))
            .unwrap_or_else(|| "<unknown>".to_string())
    );

    let _ = diagonal_gradient_image_8x8();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnn_classifier_walkthrough_improves_cat_prediction() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(1.0);

        let cat_image = vertical_stripes_image_8x8();
        let dog_image = horizontal_stripes_image_8x8();

        let pre = classifier
            .predict_with_confidence(cat_image.as_slice())
            .ok()
            .flatten();
        assert!(pre.is_none());

        let batches = vec![
            CnnTrainerBatch::new(
                "animal_cat",
                vec![cat_image.clone(), vertical_stripes_image_8x8()],
            ),
            CnnTrainerBatch::new(
                "animal_dog",
                vec![dog_image.clone(), horizontal_stripes_image_8x8()],
            ),
        ];

        for _ in 0..40 {
            let _ = classifier.train_labeled_image_batches(batches.as_slice());
        }

        classifier.set_min_confidence(0.0);
        let post = classifier
            .predict_with_confidence(cat_image.as_slice())
            .ok()
            .flatten();

        assert_eq!(post.map(|value| value.0), Some("animal_cat".to_string()));

        let eval = vec![
            CnnEvaluationSample::new("animal_cat", cat_image),
            CnnEvaluationSample::new("animal_dog", dog_image),
            CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
        ];

        let report = classifier.evaluate_labeled_images(eval.as_slice());
        assert!(report.correct_predictions >= 2);
        assert!(report.micro_f1 >= 0.8);
    }
}
