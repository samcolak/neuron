use std::collections::BTreeMap;
use std::time::Instant;

use crate::cnn::classifier::CnnImageClassifier;
use crate::cnn::data_pipeline::{
    CnnDataLoader,
    ImageTransformPipeline,
    PipelineMode,
    TransformRng,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CnnScaleTrainConfig {
    pub micro_batch_size: usize,
    pub accumulation_steps: usize,
}

impl CnnScaleTrainConfig {
    pub fn effective_batch_size(&self) -> usize {
        self.micro_batch_size
            .max(1)
            .saturating_mul(self.accumulation_steps.max(1))
    }
}

impl Default for CnnScaleTrainConfig {
    fn default() -> Self {
        Self {
            micro_batch_size: 1,
            accumulation_steps: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CnnScaleTrainReport {
    pub train_report: TrainerBatchReport,
    pub micro_batch_size: usize,
    pub accumulation_steps: usize,
    pub effective_batch_size: usize,
    pub optimizer_steps: usize,
    pub elapsed_ms: u128,
    pub throughput_samples_per_sec: f64,
    pub max_inflight_samples: usize,
    pub max_inflight_bytes: usize,
    pub transform_elapsed_ms: u128,
    pub update_elapsed_ms: u128,
    pub flush_elapsed_ms: u128,
}

fn pop_prefix(samples: &mut Vec<Vec<u8>>, prefix_len: usize) -> Vec<Vec<u8>> {
    let take = prefix_len.min(samples.len());
    let remainder = samples.split_off(take);
    std::mem::replace(samples, remainder)
}

impl CnnImageClassifier {

    pub fn train_with_data_loader(
        &mut self,
        data_loader: &CnnDataLoader,
        pipeline: &ImageTransformPipeline,
        epoch: u64,
    ) -> TrainerBatchReport {

        self.train_with_data_loader_scaled(
            data_loader,
            pipeline,
            epoch,
            CnnScaleTrainConfig::default(),
        )
        .train_report
    }

    pub fn train_with_data_loader_scaled(
        &mut self,
        data_loader: &CnnDataLoader,
        pipeline: &ImageTransformPipeline,
        epoch: u64,
        config: CnnScaleTrainConfig,
    ) -> CnnScaleTrainReport {

        let start = Instant::now();
        let mut report = TrainerBatchReport::default();
        let mut rng = TransformRng::new(data_loader.options().seed.wrapping_add(epoch));
        let effective_batch_size = config.effective_batch_size();
        let mut optimizer_steps = 0usize;
        let mut max_inflight_samples = 0usize;
        let mut max_inflight_bytes = 0usize;
        let mut inflight_samples = 0usize;
        let mut inflight_bytes = 0usize;
        let mut transform_elapsed_ms = 0u128;
        let mut update_elapsed_ms = 0u128;
        let mut flush_elapsed_ms = 0u128;
        let mut pending_by_label: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();

        for batch in data_loader.epoch_batches(epoch) {
            report.total_examples += batch.records.len();

            for record in batch.records {
                let normalized_label = record.label.trim().to_ascii_lowercase();
                if normalized_label.is_empty() {
                    continue;
                }

                let transform_start = Instant::now();
                let transformed = pipeline.apply(
                    record.image.as_slice(),
                    &mut rng,
                    PipelineMode::Train,
                );
                transform_elapsed_ms += transform_start.elapsed().as_millis();
                let transformed_len = transformed.len();

                let should_flush = {
                    let pending = pending_by_label
                        .entry(normalized_label.clone())
                        .or_default();
                    pending.push(transformed);
                    pending.len() >= effective_batch_size
                };
                inflight_samples = inflight_samples.saturating_add(1);
                inflight_bytes = inflight_bytes.saturating_add(transformed_len);

                max_inflight_samples = max_inflight_samples.max(inflight_samples);
                max_inflight_bytes = max_inflight_bytes.max(inflight_bytes);

                if should_flush {
                    let train_chunk = {
                        let pending = pending_by_label
                            .entry(normalized_label.clone())
                            .or_default();
                        pop_prefix(pending, effective_batch_size)
                    };
                    let train_count = train_chunk.len();
                    let train_chunk_bytes = train_chunk.iter().map(Vec::len).sum::<usize>();
                    inflight_samples = inflight_samples.saturating_sub(train_count);
                    inflight_bytes = inflight_bytes.saturating_sub(train_chunk_bytes);

                    let update_start = Instant::now();
                    if self
                        .train_image_batch(&normalized_label, train_chunk.as_slice())
                        .is_ok()
                    {
                        report.trained_examples += train_count;
                        *report
                            .per_label_counts
                            .entry(normalized_label.clone())
                            .or_insert(0) += train_count;
                        optimizer_steps += 1;
                    }
                    update_elapsed_ms += update_start.elapsed().as_millis();
                }
            }

        }

        for (label, pending) in &mut pending_by_label {
            while !pending.is_empty() {
                let train_chunk = pop_prefix(pending, effective_batch_size);
                let train_count = train_chunk.len();
                let train_chunk_bytes = train_chunk.iter().map(Vec::len).sum::<usize>();
                inflight_samples = inflight_samples.saturating_sub(train_count);
                inflight_bytes = inflight_bytes.saturating_sub(train_chunk_bytes);

                let flush_start = Instant::now();
                if self.train_image_batch(label, train_chunk.as_slice()).is_ok() {
                    report.trained_examples += train_count;
                    *report.per_label_counts.entry(label.clone()).or_insert(0) += train_count;
                    optimizer_steps += 1;
                }
                flush_elapsed_ms += flush_start.elapsed().as_millis();
            }
        }

        report.skipped_examples = report.total_examples.saturating_sub(report.trained_examples);
        let elapsed_ms = start.elapsed().as_millis();
        let elapsed_secs = (elapsed_ms as f64 / 1000.0).max(1e-9);

        CnnScaleTrainReport {
            train_report: report,
            micro_batch_size: config.micro_batch_size.max(1),
            accumulation_steps: config.accumulation_steps.max(1),
            effective_batch_size,
            optimizer_steps,
            elapsed_ms,
            throughput_samples_per_sec: data_loader.len() as f64 / elapsed_secs,
            max_inflight_samples,
            max_inflight_bytes,
            transform_elapsed_ms,
            update_elapsed_ms,
            flush_elapsed_ms,
        }
        
    }

    pub fn evaluate_labeled_images_with_pipeline(
        &self,
        samples: &[CnnEvaluationSample],
        pipeline: &ImageTransformPipeline,
        seed: u64,
    ) -> TrainerEvaluationReport {
        let mut report = TrainerEvaluationReport {
            total_samples: samples.len(),
            ..TrainerEvaluationReport::default()
        };

        let mut rng = TransformRng::new(seed);

        for sample in samples {
            let expected = sample.expected_label.trim().to_ascii_lowercase();
            if expected.is_empty() {
                report.skipped_samples += 1;
                continue;
            }

            report.evaluated_samples += 1;
            *report.per_label_total.entry(expected.clone()).or_insert(0) += 1;

            let transformed = pipeline.apply(
                sample.image.as_slice(),
                &mut rng,
                PipelineMode::Eval,
            );

            match self.predict_with_confidence(transformed.as_slice()) {
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
