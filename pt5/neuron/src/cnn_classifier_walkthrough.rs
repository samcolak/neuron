use crate::trainer_fixtures::{
    diagonal_gradient_image_8x8,
    horizontal_stripes_image_8x8,
    vertical_stripes_image_8x8,
};
use crate::trainer_presentation::{print_confusion_matrix, print_label_metrics};
use neuralnet::cnn::classifier::CnnImageClassifier;
use neuralnet::cnn::cnn_trainer::{
    CnnEvaluationSample,
    CnnScaleTrainConfig,
    CnnTrainerBatch,
};
use neuralnet::cnn::data_pipeline::{
    CnnDataLoader,
    CnnDataLoaderOptions,
    ImageTransform,
    ImageTransformPipeline,
};
use neuralnet::tensor::backend::active_backend_label;
use neuralnet::training::linear_head::LinearOptimizer;
use std::time::Instant;

#[derive(Debug, Clone)]
struct OptimizerSummary {
    label: String,
    optimizer: LinearOptimizer,
    seeds: usize,
    mean_accuracy: f64,
    std_accuracy: f64,
    mean_micro_f1: f64,
    std_micro_f1: f64,
    best_micro_f1: f64,
    mean_epoch_to_threshold: f64,
}

#[derive(Debug, Clone)]
struct OptimizerSweepConfig {
    label: String,
    optimizer: LinearOptimizer,
    learning_rate: f32,
    weight_decay: f32,
    adam_beta1: Option<f32>,
    adam_beta2: Option<f32>,
    adam_epsilon: Option<f32>,
}

#[derive(Debug, Clone)]
struct ScaleOptionSummary {
    label: String,
    micro_batch_size: usize,
    accumulation_steps: usize,
    effective_batch_size: usize,
    mean_throughput_sps: f64,
    mean_epoch_to_threshold: f64,
    mean_ms_to_threshold: f64,
    peak_inflight_bytes: usize,
    final_mean_micro_f1: f64,
    mean_transform_ms_per_epoch: f64,
    mean_update_ms_per_epoch: f64,
    mean_flush_ms_per_epoch: f64,
}

#[derive(Debug, Clone)]
struct LargeBenchmarkSummary {
    image_side: usize,
    sample_count: usize,
    batch_size: usize,
    effective_batch_size: usize,
    feature_extract_samples_per_sec: f64,
    train_throughput_samples_per_sec: f64,
    epoch_elapsed_ms: f64,
    eval_micro_f1: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean_value: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean_value;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn collect_optimizer_summary(
    samples: &[(String, Vec<u8>)],
    eval: &[CnnEvaluationSample],
    train_seeds: &[u64],
    eval_seed: u64,
    max_epochs: usize,
    micro_f1_threshold: f64,
) -> Vec<OptimizerSummary> {
    let mut summaries = Vec::new();

    let configs = vec![
        OptimizerSweepConfig {
            label: "sgd_lr0.20_wd0.000".to_string(),
            optimizer: LinearOptimizer::Sgd,
            learning_rate: 0.20,
            weight_decay: 0.0,
            adam_beta1: None,
            adam_beta2: None,
            adam_epsilon: None,
        },
        OptimizerSweepConfig {
            label: "sgd_lr0.08_wd0.001".to_string(),
            optimizer: LinearOptimizer::Sgd,
            learning_rate: 0.08,
            weight_decay: 0.001,
            adam_beta1: None,
            adam_beta2: None,
            adam_epsilon: None,
        },
        OptimizerSweepConfig {
            label: "adam_lr0.06_b10.90_b20.999".to_string(),
            optimizer: LinearOptimizer::Adam,
            learning_rate: 0.06,
            weight_decay: 0.0,
            adam_beta1: Some(0.90),
            adam_beta2: Some(0.999),
            adam_epsilon: Some(1e-8),
        },
        OptimizerSweepConfig {
            label: "adam_lr0.03_b10.85_b20.995".to_string(),
            optimizer: LinearOptimizer::Adam,
            learning_rate: 0.03,
            weight_decay: 0.0,
            adam_beta1: Some(0.85),
            adam_beta2: Some(0.995),
            adam_epsilon: Some(1e-8),
        },
    ];

    for config in configs {
        let mut accuracy_values = Vec::new();
        let mut micro_f1_values = Vec::new();
        let mut epoch_to_threshold_values = Vec::new();
        let mut best_micro_f1 = f64::MIN;

        for seed in train_seeds {
            let mut candidate = CnnImageClassifier::new(
                vec!["animal_cat".to_string(), "animal_dog".to_string()],
                16,
                16,
                config.learning_rate,
            )
            .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
            candidate.set_min_confidence(0.0);
            candidate.set_head_optimizer(config.optimizer);
            candidate.set_head_weight_decay(config.weight_decay);
            if let (Some(beta1), Some(beta2), Some(epsilon)) =
                (config.adam_beta1, config.adam_beta2, config.adam_epsilon)
            {
                candidate.configure_head_adam(beta1, beta2, epsilon);
            }

            let loader = CnnDataLoader::from_samples(
                samples,
                CnnDataLoaderOptions {
                    batch_size: 2,
                    shuffle: true,
                    drop_last: false,
                    seed: *seed,
                    prefetch_hint: 2,
                },
            );

            let pipeline = ImageTransformPipeline::new(vec![
                ImageTransform::RandomHorizontalFlip { probability: 0.5 },
                ImageTransform::RandomCropResize { min_scale: 0.75 },
                ImageTransform::BrightnessContrastJitter {
                    max_brightness_delta: 0.15,
                    max_contrast_delta: 0.15,
                },
                ImageTransform::GaussianNoise {
                    probability: 0.3,
                    stddev: 0.03,
                },
                ImageTransform::NormalizeMinMax,
            ]);

            let mut reached_threshold_epoch = (max_epochs + 1) as f64;
            for epoch in 0..max_epochs {
                let _ = candidate.train_with_data_loader(&loader, &pipeline, epoch as u64);

                let epoch_report =
                    candidate.evaluate_labeled_images_with_pipeline(eval, &pipeline, eval_seed);
                if reached_threshold_epoch > max_epochs as f64
                    && epoch_report.micro_f1 >= micro_f1_threshold
                {
                    reached_threshold_epoch = (epoch + 1) as f64;
                }
            }

            let report =
                candidate.evaluate_labeled_images_with_pipeline(eval, &pipeline, eval_seed);

            accuracy_values.push(report.accuracy);
            micro_f1_values.push(report.micro_f1);
            epoch_to_threshold_values.push(reached_threshold_epoch);
            best_micro_f1 = best_micro_f1.max(report.micro_f1);
        }

        let seed_count = train_seeds.len().max(1);
        let mean_accuracy = mean(accuracy_values.as_slice());
        let mean_micro_f1 = mean(micro_f1_values.as_slice());
        let mean_epoch_to_threshold = mean(epoch_to_threshold_values.as_slice());

        summaries.push(OptimizerSummary {
            label: config.label,
            optimizer: config.optimizer,
            seeds: seed_count,
            mean_accuracy,
            std_accuracy: stddev(accuracy_values.as_slice(), mean_accuracy),
            mean_micro_f1,
            std_micro_f1: stddev(micro_f1_values.as_slice(), mean_micro_f1),
            best_micro_f1,
            mean_epoch_to_threshold,
        });
    }

    summaries
}

fn collect_scale_option_summary(
    samples: &[(String, Vec<u8>)],
    eval: &[CnnEvaluationSample],
    train_seeds: &[u64],
    eval_seed: u64,
    max_epochs: usize,
    micro_f1_threshold: f64,
) -> Vec<ScaleOptionSummary> {
    let options = vec![
        CnnScaleTrainConfig {
            micro_batch_size: 1,
            accumulation_steps: 1,
        },
        CnnScaleTrainConfig {
            micro_batch_size: 2,
            accumulation_steps: 2,
        },
        CnnScaleTrainConfig {
            micro_batch_size: 4,
            accumulation_steps: 2,
        },
    ];

    let pipeline = ImageTransformPipeline::new(vec![
        ImageTransform::RandomHorizontalFlip { probability: 0.5 },
        ImageTransform::RandomCropResize { min_scale: 0.75 },
        ImageTransform::BrightnessContrastJitter {
            max_brightness_delta: 0.15,
            max_contrast_delta: 0.15,
        },
        ImageTransform::GaussianNoise {
            probability: 0.3,
            stddev: 0.03,
        },
        ImageTransform::NormalizeMinMax,
    ]);

    let mut rows = Vec::new();

    for option in options {
        let mut throughput_values = Vec::new();
        let mut epoch_to_threshold_values = Vec::new();
        let mut ms_to_threshold_values = Vec::new();
        let mut final_micro_f1_values = Vec::new();
        let mut transform_ms_values = Vec::new();
        let mut update_ms_values = Vec::new();
        let mut flush_ms_values = Vec::new();
        let mut peak_inflight_bytes = 0usize;

        for seed in train_seeds {
            let loader = CnnDataLoader::from_samples(
                samples,
                CnnDataLoaderOptions {
                    batch_size: 8,
                    shuffle: true,
                    drop_last: false,
                    seed: *seed,
                    prefetch_hint: 2,
                },
            );

            let mut candidate = CnnImageClassifier::new(
                vec!["animal_cat".to_string(), "animal_dog".to_string()],
                16,
                16,
                0.06,
            )
            .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
            candidate.set_min_confidence(0.0);
            candidate.set_head_optimizer(LinearOptimizer::Adam);
            candidate.configure_head_adam(0.90, 0.999, 1e-8);

            let mut reached_epoch = (max_epochs + 1) as f64;
            let mut reached_ms = 0.0f64;
            let mut elapsed_ms_total = 0.0f64;
            let mut throughput_sum = 0.0f64;
            let mut transform_ms_sum = 0.0f64;
            let mut update_ms_sum = 0.0f64;
            let mut flush_ms_sum = 0.0f64;

            for epoch in 0..max_epochs {
                let scale_report = candidate.train_with_data_loader_scaled(
                    &loader,
                    &pipeline,
                    epoch as u64,
                    option,
                );

                elapsed_ms_total += scale_report.elapsed_ms as f64;
                throughput_sum += scale_report.throughput_samples_per_sec;
                transform_ms_sum += scale_report.transform_elapsed_ms as f64;
                update_ms_sum += scale_report.update_elapsed_ms as f64;
                flush_ms_sum += scale_report.flush_elapsed_ms as f64;
                peak_inflight_bytes = peak_inflight_bytes.max(scale_report.max_inflight_bytes);

                let epoch_report =
                    candidate.evaluate_labeled_images_with_pipeline(eval, &pipeline, eval_seed);
                if reached_epoch > max_epochs as f64 && epoch_report.micro_f1 >= micro_f1_threshold {
                    reached_epoch = (epoch + 1) as f64;
                    reached_ms = elapsed_ms_total;
                }
            }

            let final_report =
                candidate.evaluate_labeled_images_with_pipeline(eval, &pipeline, eval_seed);
            throughput_values.push(throughput_sum / max_epochs as f64);
            transform_ms_values.push(transform_ms_sum / max_epochs as f64);
            update_ms_values.push(update_ms_sum / max_epochs as f64);
            flush_ms_values.push(flush_ms_sum / max_epochs as f64);
            epoch_to_threshold_values.push(reached_epoch);
            ms_to_threshold_values.push(reached_ms);
            final_micro_f1_values.push(final_report.micro_f1);
        }

        rows.push(ScaleOptionSummary {
            label: format!(
                "mb{}xacc{}",
                option.micro_batch_size,
                option.accumulation_steps
            ),
            micro_batch_size: option.micro_batch_size,
            accumulation_steps: option.accumulation_steps,
            effective_batch_size: option.effective_batch_size(),
            mean_throughput_sps: mean(throughput_values.as_slice()),
            mean_epoch_to_threshold: mean(epoch_to_threshold_values.as_slice()),
            mean_ms_to_threshold: mean(ms_to_threshold_values.as_slice()),
            peak_inflight_bytes,
            final_mean_micro_f1: mean(final_micro_f1_values.as_slice()),
            mean_transform_ms_per_epoch: mean(transform_ms_values.as_slice()),
            mean_update_ms_per_epoch: mean(update_ms_values.as_slice()),
            mean_flush_ms_per_epoch: mean(flush_ms_values.as_slice()),
        });
    }

    rows
}

fn striped_image(side: usize, vertical: bool, phase: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let band = if vertical { (x + phase) / 8 } else { (y + phase) / 8 };
            let value = if band % 2 == 0 { 232 } else { 24 };
            bytes.push(value as u8);
        }
    }
    bytes
}

fn collect_large_benchmark_summary() -> LargeBenchmarkSummary {
    // Workload sized to stress the GPU path:
    // - 128x128 images  (large enough that GPU parallelism > transfer overhead)
    // - 16 feature channels  (increases conv arithmetic intensity)
    // - 300 samples total  (150 per class)
    // Run with `cargo run --release` for meaningful numbers.
    let image_side = 128usize;
    let feature_channels = 16usize;
    let samples_per_label = 150usize;
    let mut samples = Vec::with_capacity(samples_per_label * 2);
    for idx in 0..samples_per_label {
        samples.push((
            "animal_cat".to_string(),
            striped_image(image_side, true, idx % 16),
        ));
        samples.push((
            "animal_dog".to_string(),
            striped_image(image_side, false, idx % 16),
        ));
    }

    let eval = vec![
        CnnEvaluationSample::new("animal_cat", striped_image(image_side, true, 1)),
        CnnEvaluationSample::new("animal_dog", striped_image(image_side, false, 1)),
    ];

    let loader = CnnDataLoader::from_samples(
        samples.as_slice(),
        CnnDataLoaderOptions {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            seed: 91,
            prefetch_hint: 4,
        },
    );

    let pipeline = ImageTransformPipeline::new(vec![
        ImageTransform::RandomHorizontalFlip { probability: 0.5 },
        ImageTransform::RandomCropResize { min_scale: 0.85 },
        ImageTransform::BrightnessContrastJitter {
            max_brightness_delta: 0.1,
            max_contrast_delta: 0.1,
        },
        ImageTransform::NormalizeMinMax,
    ]);

    let mut classifier = CnnImageClassifier::new_with_feature_channels(
        vec!["animal_cat".to_string(), "animal_dog".to_string()],
        image_side,
        image_side,
        &[feature_channels],
        0.02,
    )
    .unwrap_or_else(|_| panic!("benchmark classifier should initialize"));
    classifier.set_min_confidence(0.0);
    classifier.set_head_optimizer(LinearOptimizer::Adam);
    classifier.configure_head_adam(0.90, 0.999, 1e-8);

    // Feature-extraction throughput: fused path per image.
    let bench_images: Vec<Vec<u8>> = samples.iter().map(|(_, image)| image.clone()).collect();
    let feature_start = Instant::now();
    for image in &bench_images {
        let _ = classifier
            .extract_features(image.as_slice())
            .unwrap_or_else(|_| panic!("feature extraction benchmark should succeed"));
    }
    let feature_elapsed = feature_start.elapsed().as_secs_f64();

    let scale_report = classifier.train_with_data_loader_scaled(
        &loader,
        &pipeline,
        0,
        CnnScaleTrainConfig {
            micro_batch_size: 16,
            accumulation_steps: 2,
        },
    );
    let eval_report = classifier.evaluate_labeled_images_with_pipeline(
        eval.as_slice(),
        &pipeline,
        99,
    );

    LargeBenchmarkSummary {
        image_side,
        sample_count: samples.len(),
        batch_size: 32,
        effective_batch_size: 32,
        feature_extract_samples_per_sec: bench_images.len() as f64 / feature_elapsed.max(1e-9),
        train_throughput_samples_per_sec: scale_report.throughput_samples_per_sec,
        epoch_elapsed_ms: scale_report.elapsed_ms as f64,
        eval_micro_f1: eval_report.micro_f1,
    }
}

pub fn run_cnn_classifier_walkthrough() {
    println!("\nCNN classifier walkthrough");
    println!("  tensor backend in use: {}", active_backend_label());

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

    println!("  augmented pipeline wireup demo");

    let mut augmented_classifier = CnnImageClassifier::new(
        vec!["animal_cat".to_string(), "animal_dog".to_string()],
        16,
        16,
        0.2,
    )
    .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
    augmented_classifier.set_min_confidence(0.0);

    let augmented_samples = vec![
        ("animal_cat".to_string(), vertical_stripes_image_8x8()),
        ("animal_cat".to_string(), cat_image.clone()),
        ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ("animal_dog".to_string(), dog_image.clone()),
    ];
    let loader = CnnDataLoader::from_samples(
        augmented_samples.as_slice(),
        CnnDataLoaderOptions {
            batch_size: 2,
            shuffle: true,
            drop_last: false,
            seed: 7,
            prefetch_hint: 2,
        },
    );

    let pipeline = ImageTransformPipeline::new(vec![
        ImageTransform::RandomHorizontalFlip { probability: 0.5 },
        ImageTransform::RandomCropResize { min_scale: 0.75 },
        ImageTransform::BrightnessContrastJitter {
            max_brightness_delta: 0.15,
            max_contrast_delta: 0.15,
        },
        ImageTransform::GaussianNoise {
            probability: 0.4,
            stddev: 0.03,
        },
        ImageTransform::NormalizeMinMax,
    ]);

    for epoch in 0..30 {
        let _ = augmented_classifier.train_with_data_loader(&loader, &pipeline, epoch);
    }

    let augmented_eval = augmented_classifier
        .evaluate_labeled_images_with_pipeline(eval.as_slice(), &pipeline, 123);

    println!(
        "  augmented eval: accuracy={:.3} micro_f1={:.3}",
        augmented_eval.accuracy,
        augmented_eval.micro_f1
    );
    print_confusion_matrix(&augmented_eval);
    print_label_metrics(&augmented_eval);

    println!("  optimizer comparison (sgd vs adam)");
    let optimizers = [LinearOptimizer::Sgd, LinearOptimizer::Adam];

    for optimizer in optimizers {
        let mut candidate = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        candidate.set_min_confidence(0.0);
        candidate.set_head_optimizer(optimizer);

        for epoch in 0..30 {
            let _ = candidate.train_with_data_loader(&loader, &pipeline, epoch);
        }

        let optimizer_report =
            candidate.evaluate_labeled_images_with_pipeline(eval.as_slice(), &pipeline, 303);
        println!(
            "    {:?}: accuracy={:.3} micro_f1={:.3} correct={} unknown={}",
            optimizer,
            optimizer_report.accuracy,
            optimizer_report.micro_f1,
            optimizer_report.correct_predictions,
            optimizer_report.unknown_predictions,
        );
    }

    let summary_samples = vec![
        ("animal_cat".to_string(), vertical_stripes_image_8x8()),
        ("animal_cat".to_string(), cat_image.clone()),
        ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ("animal_dog".to_string(), dog_image.clone()),
    ];
    let summary = collect_optimizer_summary(
        summary_samples.as_slice(),
        eval.as_slice(),
        &[5, 11, 19, 29, 41],
        202,
        30,
        0.80,
    );

    println!("  optimizer summary across seeds (mean +- std, plus convergence speed)");
    for row in summary {
        println!(
            "    {} ({:?}): runs={} mean_accuracy={:.3} +- {:.3} mean_micro_f1={:.3} +- {:.3} best_micro_f1={:.3} mean_epoch_to_f1>=0.80={:.2}",
            row.label,
            row.optimizer,
            row.seeds,
            row.mean_accuracy,
            row.std_accuracy,
            row.mean_micro_f1,
            row.std_micro_f1,
            row.best_micro_f1,
            row.mean_epoch_to_threshold,
        );
    }

    let scale_summary = collect_scale_option_summary(
        summary_samples.as_slice(),
        eval.as_slice(),
        &[5, 11, 19, 29, 41],
        707,
        24,
        0.80,
    );

    println!("  scale options (throughput, convergence speed, memory proxy)");
    for row in scale_summary {
        println!(
            "    {}: micro_batch={} accumulation={} effective_batch={} mean_throughput={:.1} samples/s mean_epoch_to_f1>=0.80={:.2} mean_ms_to_f1>=0.80={:.1} peak_inflight_bytes={} final_mean_micro_f1={:.3} timing_ms(epoch): transform={:.2} update={:.2} flush={:.2}",
            row.label,
            row.micro_batch_size,
            row.accumulation_steps,
            row.effective_batch_size,
            row.mean_throughput_sps,
            row.mean_epoch_to_threshold,
            row.mean_ms_to_threshold,
            row.peak_inflight_bytes,
            row.final_mean_micro_f1,
            row.mean_transform_ms_per_epoch,
            row.mean_update_ms_per_epoch,
            row.mean_flush_ms_per_epoch,
        );
    }

    let large_benchmark = collect_large_benchmark_summary();
    println!("  larger backend benchmark (128x128 / 16ch / 300 samples — run --release for meaningful numbers)");
    println!(
        "    images={} side={} ch=16 batch={} effective_batch={} feature_extract={:.1} samples/s train_epoch={:.1} samples/s epoch_ms={:.1} eval_micro_f1={:.3}",
        large_benchmark.sample_count,
        large_benchmark.image_side,
        large_benchmark.batch_size,
        large_benchmark.effective_batch_size,
        large_benchmark.feature_extract_samples_per_sec,
        large_benchmark.train_throughput_samples_per_sec,
        large_benchmark.epoch_elapsed_ms,
        large_benchmark.eval_micro_f1,
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

    #[test]
    fn cnn_classifier_walkthrough_pipeline_wireup_trains_with_augmentations() {
        let mut classifier = CnnImageClassifier::new(
            vec!["animal_cat".to_string(), "animal_dog".to_string()],
            16,
            16,
            0.2,
        )
        .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
        classifier.set_min_confidence(0.0);

        let samples = vec![
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ];
        let loader = CnnDataLoader::from_samples(
            samples.as_slice(),
            CnnDataLoaderOptions {
                batch_size: 2,
                shuffle: true,
                drop_last: false,
                seed: 11,
                prefetch_hint: 2,
            },
        );

        let pipeline = ImageTransformPipeline::new(vec![
            ImageTransform::RandomHorizontalFlip { probability: 0.5 },
            ImageTransform::RandomCropResize { min_scale: 0.75 },
            ImageTransform::BrightnessContrastJitter {
                max_brightness_delta: 0.15,
                max_contrast_delta: 0.15,
            },
            ImageTransform::GaussianNoise {
                probability: 0.3,
                stddev: 0.03,
            },
            ImageTransform::NormalizeMinMax,
        ]);

        for epoch in 0..25 {
            let _ = classifier.train_with_data_loader(&loader, &pipeline, epoch);
        }

        let eval = vec![
            CnnEvaluationSample::new("animal_cat", vertical_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_dog", horizontal_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
        ];

        let report = classifier.evaluate_labeled_images_with_pipeline(eval.as_slice(), &pipeline, 22);
        assert!(report.correct_predictions >= 2);
        assert!(report.micro_f1 >= 0.8);
    }

    #[test]
    fn cnn_classifier_walkthrough_optimizer_comparison_supports_sgd_and_adam() {
        let samples = vec![
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ];
        let loader = CnnDataLoader::from_samples(
            samples.as_slice(),
            CnnDataLoaderOptions {
                batch_size: 2,
                shuffle: true,
                drop_last: false,
                seed: 19,
                prefetch_hint: 2,
            },
        );

        let pipeline = ImageTransformPipeline::new(vec![
            ImageTransform::RandomHorizontalFlip { probability: 0.5 },
            ImageTransform::RandomCropResize { min_scale: 0.75 },
            ImageTransform::BrightnessContrastJitter {
                max_brightness_delta: 0.10,
                max_contrast_delta: 0.10,
            },
            ImageTransform::GaussianNoise {
                probability: 0.2,
                stddev: 0.02,
            },
            ImageTransform::NormalizeMinMax,
        ]);

        let eval = vec![
            CnnEvaluationSample::new("animal_cat", vertical_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_dog", horizontal_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
        ];

        for optimizer in [LinearOptimizer::Sgd, LinearOptimizer::Adam] {
            let mut classifier = CnnImageClassifier::new(
                vec!["animal_cat".to_string(), "animal_dog".to_string()],
                16,
                16,
                0.2,
            )
            .unwrap_or_else(|_| panic!("cnn classifier should initialize"));
            classifier.set_min_confidence(0.0);
            classifier.set_head_optimizer(optimizer);

            for epoch in 0..25 {
                let _ = classifier.train_with_data_loader(&loader, &pipeline, epoch);
            }

            let report = classifier.evaluate_labeled_images_with_pipeline(eval.as_slice(), &pipeline, 44);
            assert!(report.micro_f1.is_finite());
            assert!(report.correct_predictions >= 2);
        }
    }

    #[test]
    fn cnn_classifier_walkthrough_optimizer_summary_reports_both_optimizers() {
        let samples = vec![
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ];
        let eval = vec![
            CnnEvaluationSample::new("animal_cat", vertical_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_dog", horizontal_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
        ];

        let summary =
            collect_optimizer_summary(samples.as_slice(), eval.as_slice(), &[3, 7, 13], 88, 20, 0.80);

        assert_eq!(summary.len(), 4);
        assert_eq!(summary[0].seeds, 3);
        assert_eq!(summary[1].seeds, 3);
        assert_eq!(summary[2].seeds, 3);
        assert_eq!(summary[3].seeds, 3);
        assert!(summary.iter().all(|row| row.mean_accuracy.is_finite()));
        assert!(summary.iter().all(|row| row.std_accuracy.is_finite()));
        assert!(summary.iter().all(|row| row.mean_micro_f1.is_finite()));
        assert!(summary.iter().all(|row| row.std_micro_f1.is_finite()));
        assert!(summary.iter().all(|row| row.best_micro_f1.is_finite()));
        assert!(summary
            .iter()
            .all(|row| row.mean_epoch_to_threshold.is_finite()));
    }

    #[test]
    fn cnn_classifier_walkthrough_scale_summary_reports_throughput_and_threshold_metrics() {
        let samples = vec![
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_cat".to_string(), vertical_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
            ("animal_dog".to_string(), horizontal_stripes_image_8x8()),
        ];
        let eval = vec![
            CnnEvaluationSample::new("animal_cat", vertical_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_dog", horizontal_stripes_image_8x8()),
            CnnEvaluationSample::new("animal_bird", vec![9u8; 1000]),
        ];

        let summary =
            collect_scale_option_summary(samples.as_slice(), eval.as_slice(), &[3, 7], 99, 12, 0.70);

        assert_eq!(summary.len(), 3);
        assert!(summary.iter().all(|row| row.mean_throughput_sps.is_finite()));
        assert!(summary.iter().all(|row| row.mean_epoch_to_threshold.is_finite()));
        assert!(summary.iter().all(|row| row.mean_ms_to_threshold.is_finite()));
        assert!(summary.iter().all(|row| row.peak_inflight_bytes > 0));
        assert!(summary.iter().all(|row| row.final_mean_micro_f1.is_finite()));
        assert!(summary
            .iter()
            .all(|row| row.mean_transform_ms_per_epoch.is_finite()));
        assert!(summary
            .iter()
            .all(|row| row.mean_update_ms_per_epoch.is_finite()));
        assert!(summary
            .iter()
            .all(|row| row.mean_flush_ms_per_epoch.is_finite()));
    }
}
