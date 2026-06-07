use std::collections::BTreeMap;

use crate::cnn::classifier::CnnImageClassifier;
use crate::training::metrics::{compute_quality_metrics, increment_confusion_count};
use crate::training::trainer::{
    TrainerBatchReport,
    TrainerEvaluationReport,
};

#[derive(Debug, Clone)]
pub struct CnnTrainerBatch {
    pub label: String,
    pub samples: Vec<Vec<u8>>,
}

impl CnnTrainerBatch {
    pub fn new(label: &str, samples: Vec<Vec<u8>>) -> Self {
        Self {
            label: label.to_string(),
            samples,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CnnEvaluationSample {
    pub expected_label: String,
    pub image: Vec<u8>,
}

impl CnnEvaluationSample {

    pub fn new(expected_label: &str, image: Vec<u8>) -> Self {
        Self {
            expected_label: expected_label.to_string(),
            image,
        }
    }

}

impl CnnImageClassifier {

    pub fn train_labeled_image_batches(&mut self, batches: &[CnnTrainerBatch]) -> TrainerBatchReport {

        let mut report = TrainerBatchReport::default();

        for batch in batches {
            report.total_examples += batch.samples.len();
            let normalized_label = batch.label.trim().to_ascii_lowercase();
            if normalized_label.is_empty() {
                continue;
            }

            for image in &batch.samples {
                if self.train_image(&normalized_label, image.as_slice()).is_ok() {
                    report.trained_examples += 1;
                    *report
                        .per_label_counts
                        .entry(normalized_label.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        report.skipped_examples = report.total_examples.saturating_sub(report.trained_examples);
        report
        
    }

    pub fn evaluate_labeled_images(
        &self,
        samples: &[CnnEvaluationSample],
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

            match self.predict_with_confidence(sample.image.as_slice()) {
                Ok(Some((predicted, _confidence))) if predicted == expected => {
                    report.correct_predictions += 1;
                    *report.per_label_correct.entry(expected.clone()).or_insert(0) += 1;
                    increment_confusion_count(&mut report.confusion_matrix, expected, predicted);
                }
                Ok(Some((predicted, _confidence))) => {
                    increment_confusion_count(&mut report.confusion_matrix, expected, predicted);
                }
                _ => {
                    report.unknown_predictions += 1;
                    increment_confusion_count(
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

    pub fn train_labeled_image_samples(
        &mut self,
        samples: &[(String, Vec<u8>)],
    ) -> TrainerBatchReport {

        let mut grouped: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();

        for (label, image) in samples {
            let normalized = label.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            grouped.entry(normalized).or_default().push(image.clone());
        }

        let batches: Vec<CnnTrainerBatch> = grouped
            .into_iter()
            .map(|(label, grouped_samples)| CnnTrainerBatch::new(&label, grouped_samples))
            .collect();

        self.train_labeled_image_batches(batches.as_slice())
    
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::classifier::CnnImageClassifier;

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
    fn cnn_trainer_batches_report_counts() {

        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("classifier should initialize"));

        let batches = vec![
            CnnTrainerBatch::new(
                "animal_cat",
                vec![vertical_stripes_image_8x8(), vertical_stripes_image_8x8()],
            ),
            CnnTrainerBatch::new("animal_dog", vec![horizontal_stripes_image_8x8()]),
        ];

        let report = classifier.train_labeled_image_batches(batches.as_slice());

        assert_eq!(report.total_examples, 3);
        assert_eq!(report.trained_examples, 3);
        assert_eq!(report.skipped_examples, 0);
        assert_eq!(report.per_label_counts.get("animal_cat"), Some(&2usize));
        assert_eq!(report.per_label_counts.get("animal_dog"), Some(&1usize));

    }

    #[test]
    fn cnn_trainer_evaluation_returns_confusion_and_micro_metrics() {

        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("classifier should initialize"));
        classifier.set_min_confidence(0.0);

        for _ in 0..80 {
            let _ = classifier.train_image("animal_cat", vertical_stripes_image_8x8().as_slice());
            let _ = classifier.train_image("animal_dog", horizontal_stripes_image_8x8().as_slice());
        }

        let eval = vec![
            CnnEvaluationSample::new("animal_cat", vertical_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_dog", horizontal_stripes_image_8x8()),
            // Non-square payload forces deterministic extractor failure -> <unknown> bucket.
            CnnEvaluationSample::new("animal_bird", vec![7u8; 1000]),
        ];

        let report = classifier.evaluate_labeled_images(eval.as_slice());

        assert_eq!(report.total_samples, 3);
        assert_eq!(report.evaluated_samples, 3);
        assert!(report.correct_predictions >= 1);
        assert!(report.micro_f1 >= 0.5);
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_bird")
                .and_then(|row| row.get("<unknown>")),
            Some(&1usize)
        );

    }

    #[test]
    fn cnn_trainer_confidence_threshold_can_force_unknown_predictions() {

        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("classifier should initialize"));

        for _ in 0..4 {
            let _ = classifier.train_image("animal_cat", vertical_stripes_image_8x8().as_slice());
            let _ = classifier.train_image("animal_dog", horizontal_stripes_image_8x8().as_slice());
        }

        classifier.set_min_confidence(1.0);
        let eval = vec![CnnEvaluationSample::new(
            "animal_cat",
            vertical_stripes_image_8x8(),
        )];

        let report = classifier.evaluate_labeled_images(eval.as_slice());

        assert_eq!(report.evaluated_samples, 1);
        assert_eq!(report.correct_predictions, 0);
        assert_eq!(report.unknown_predictions, 1);
        assert_eq!(
            report
                .confusion_matrix
                .get("animal_cat")
                .and_then(|row| row.get("<unknown>")),
            Some(&1usize)
        );

    }

    #[test]
    fn cnn_trainer_loss_trends_down_on_repeated_batches() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("classifier should initialize"));

        let cat_batch = vec![vertical_stripes_image_8x8(), vertical_stripes_image_8x8()];
        let dog_batch = vec![horizontal_stripes_image_8x8(), horizontal_stripes_image_8x8()];

        let initial = classifier
            .train_image_batch("animal_cat", cat_batch.as_slice())
            .unwrap_or_else(|_| panic!("initial cat batch train should succeed"))
            + classifier
                .train_image_batch("animal_dog", dog_batch.as_slice())
                .unwrap_or_else(|_| panic!("initial dog batch train should succeed"));

        let mut final_loss = initial;
        for _ in 0..30 {
            let cat_loss = classifier
                .train_image_batch("animal_cat", cat_batch.as_slice())
                .unwrap_or_else(|_| panic!("cat batch train should succeed"));
            let dog_loss = classifier
                .train_image_batch("animal_dog", dog_batch.as_slice())
                .unwrap_or_else(|_| panic!("dog batch train should succeed"));
            final_loss = cat_loss + dog_loss;
        }

        assert!(final_loss < initial);
    }
}
