use std::cmp::{max, min};

use super::cnn_trainer::CnnTrainerBatch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    Train,
    Eval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformRng {
    state: u64,
}

impl TransformRng {
    pub fn new(seed: u64) -> Self {
        let initialized = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state: initialized }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64();
        (value as f64 / u64::MAX as f64) as f32
    }

    fn next_inclusive(&mut self, min_value: usize, max_value: usize) -> usize {
        if min_value >= max_value {
            return min_value;
        }
        let span = max_value - min_value + 1;
        min_value + (self.next_u64() as usize % span)
    }

    fn next_signed_unit(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    pub fn maybe(&mut self, probability: f32) -> bool {
        self.next_f32() < probability.clamp(0.0, 1.0)
    }

    fn gaussian_sample(&mut self, stddev: f32) -> f32 {
        if stddev <= 0.0 {
            return 0.0;
        }

        let u1 = self.next_f32().max(1.0e-7);
        let u2 = self.next_f32().max(1.0e-7);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z0 * stddev
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageTransform {
    RandomHorizontalFlip { probability: f32 },
    RandomCropResize { min_scale: f32 },
    BrightnessContrastJitter {
        max_brightness_delta: f32,
        max_contrast_delta: f32,
    },
    GaussianNoise {
        probability: f32,
        stddev: f32,
    },
    NormalizeMinMax,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageTransformPipeline {
    transforms: Vec<ImageTransform>,
}

impl ImageTransformPipeline {
    pub fn new(transforms: Vec<ImageTransform>) -> Self {
        Self { transforms }
    }

    pub fn transforms(&self) -> &[ImageTransform] {
        self.transforms.as_slice()
    }

    pub fn apply(&self, image_bytes: &[u8], rng: &mut TransformRng, mode: PipelineMode) -> Vec<u8> {
        let mut sample = match ImageSample::from_square_image_bytes(image_bytes) {
            Some(sample) => sample,
            None => return image_bytes.to_vec(),
        };

        for transform in &self.transforms {
            let should_apply = match transform {
                ImageTransform::RandomHorizontalFlip { .. }
                | ImageTransform::RandomCropResize { .. }
                | ImageTransform::BrightnessContrastJitter { .. }
                | ImageTransform::GaussianNoise { .. } => mode == PipelineMode::Train,
                ImageTransform::NormalizeMinMax => true,
            };

            if !should_apply {
                continue;
            }

            apply_transform(&mut sample, transform, rng);
        }

        sample.bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabeledImageRecord {
    pub label: String,
    pub image: Vec<u8>,
}

impl LabeledImageRecord {
    pub fn new(label: &str, image: Vec<u8>) -> Self {
        Self {
            label: label.to_string(),
            image,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnDataBatch {
    pub records: Vec<LabeledImageRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnDataLoaderOptions {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub seed: u64,
    pub prefetch_hint: usize,
}

impl Default for CnnDataLoaderOptions {
    fn default() -> Self {
        Self {
            batch_size: 16,
            shuffle: true,
            drop_last: false,
            seed: 1,
            prefetch_hint: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnDataLoader {
    records: Vec<LabeledImageRecord>,
    options: CnnDataLoaderOptions,
}

impl CnnDataLoader {
    pub fn new(records: Vec<LabeledImageRecord>, options: CnnDataLoaderOptions) -> Self {
        Self { records, options }
    }

    pub fn from_samples(samples: &[(String, Vec<u8>)], options: CnnDataLoaderOptions) -> Self {
        let records = samples
            .iter()
            .map(|(label, image)| LabeledImageRecord {
                label: label.clone(),
                image: image.clone(),
            })
            .collect();
        Self { records, options }
    }

    pub fn options(&self) -> &CnnDataLoaderOptions {
        &self.options
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn epoch_batches(&self, epoch: u64) -> Vec<CnnDataBatch> {
        let mut indices: Vec<usize> = (0..self.records.len()).collect();
        if self.options.shuffle && indices.len() > 1 {
            fisher_yates_shuffle(&mut indices, self.options.seed.wrapping_add(epoch));
        }

        let mut batches = Vec::new();
        let batch_size = max(1, self.options.batch_size);
        let mut start = 0usize;

        while start < indices.len() {
            let end = min(start + batch_size, indices.len());
            if self.options.drop_last && (end - start) < batch_size {
                break;
            }

            let records = indices[start..end]
                .iter()
                .map(|index| self.records[*index].clone())
                .collect();

            batches.push(CnnDataBatch { records });
            start = end;
        }

        batches
    }

    pub fn epoch_as_label_batches(&self, epoch: u64) -> Vec<CnnTrainerBatch> {
        self.epoch_batches(epoch)
            .into_iter()
            .flat_map(|batch| {
                let mut grouped: std::collections::BTreeMap<String, Vec<Vec<u8>>> =
                    std::collections::BTreeMap::new();

                for record in batch.records {
                    let normalized = record.label.trim().to_ascii_lowercase();
                    if normalized.is_empty() {
                        continue;
                    }
                    grouped.entry(normalized).or_default().push(record.image);
                }

                grouped
                    .into_iter()
                    .map(|(label, samples)| CnnTrainerBatch::new(&label, samples))
                    .collect::<Vec<CnnTrainerBatch>>()
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImageSample {
    height: usize,
    width: usize,
    channels: usize,
    bytes: Vec<u8>,
}

impl ImageSample {
    fn from_square_image_bytes(image_bytes: &[u8]) -> Option<Self> {
        let (height, width, channels) = infer_square_dimensions_and_channels(image_bytes)?;
        Some(Self {
            height,
            width,
            channels,
            bytes: image_bytes.to_vec(),
        })
    }
}

fn fisher_yates_shuffle(values: &mut [usize], seed: u64) {
    if values.len() < 2 {
        return;
    }

    let mut rng = TransformRng::new(seed);
    for idx in (1..values.len()).rev() {
        let swap_with = rng.next_inclusive(0, idx);
        values.swap(idx, swap_with);
    }
}

fn apply_transform(sample: &mut ImageSample, transform: &ImageTransform, rng: &mut TransformRng) {
    match transform {
        ImageTransform::RandomHorizontalFlip { probability } => {
            if rng.maybe(*probability) {
                horizontal_flip(sample);
            }
        }
        ImageTransform::RandomCropResize { min_scale } => {
            random_crop_resize(sample, min_scale.clamp(0.1, 1.0), rng);
        }
        ImageTransform::BrightnessContrastJitter {
            max_brightness_delta,
            max_contrast_delta,
        } => {
            let brightness_delta = rng.next_signed_unit() * max_brightness_delta.clamp(0.0, 1.0) * 255.0;
            let contrast_delta = rng.next_signed_unit() * max_contrast_delta.clamp(0.0, 1.0);
            apply_brightness_contrast(sample, brightness_delta, contrast_delta);
        }
        ImageTransform::GaussianNoise { probability, stddev } => {
            if rng.maybe(*probability) {
                add_gaussian_noise(sample, *stddev, rng);
            }
        }
        ImageTransform::NormalizeMinMax => {
            normalize_min_max(sample);
        }
    }
}

fn horizontal_flip(sample: &mut ImageSample) {
    let channels = sample.channels;
    for y in 0..sample.height {
        for x in 0..(sample.width / 2) {
            for c in 0..channels {
                let left = index(sample.width, channels, x, y, c);
                let right = index(sample.width, channels, sample.width - 1 - x, y, c);
                sample.bytes.swap(left, right);
            }
        }
    }
}

fn random_crop_resize(sample: &mut ImageSample, min_scale: f32, rng: &mut TransformRng) {
    let min_side = min(sample.width, sample.height);
    if min_side <= 2 {
        return;
    }

    let min_crop = max(2, (min_side as f32 * min_scale).floor() as usize);
    let crop_size = rng.next_inclusive(min_crop, min_side);
    let max_x = sample.width - crop_size;
    let max_y = sample.height - crop_size;
    let start_x = rng.next_inclusive(0, max_x);
    let start_y = rng.next_inclusive(0, max_y);

    let mut cropped = vec![0u8; crop_size * crop_size * sample.channels];
    for y in 0..crop_size {
        for x in 0..crop_size {
            for c in 0..sample.channels {
                let src = index(sample.width, sample.channels, start_x + x, start_y + y, c);
                let dst = index(crop_size, sample.channels, x, y, c);
                cropped[dst] = sample.bytes[src];
            }
        }
    }

    sample.bytes = resize_nearest(
        cropped.as_slice(),
        crop_size,
        crop_size,
        sample.width,
        sample.height,
        sample.channels,
    );
}

fn resize_nearest(
    source: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h * channels];
    for y in 0..dst_h {
        let src_y = min(src_h - 1, (y * src_h) / dst_h);
        for x in 0..dst_w {
            let src_x = min(src_w - 1, (x * src_w) / dst_w);
            for c in 0..channels {
                let src = index(src_w, channels, src_x, src_y, c);
                let dst = index(dst_w, channels, x, y, c);
                out[dst] = source[src];
            }
        }
    }
    out
}

fn apply_brightness_contrast(sample: &mut ImageSample, brightness_delta: f32, contrast_delta: f32) {
    let contrast_factor = 1.0 + contrast_delta;
    for value in &mut sample.bytes {
        let centered = *value as f32 - 127.5;
        let adjusted = centered * contrast_factor + 127.5 + brightness_delta;
        *value = adjusted.clamp(0.0, 255.0) as u8;
    }
}

fn add_gaussian_noise(sample: &mut ImageSample, stddev: f32, rng: &mut TransformRng) {
    for value in &mut sample.bytes {
        let noise = rng.gaussian_sample(stddev.max(0.0));
        let adjusted = *value as f32 + noise * 255.0;
        *value = adjusted.clamp(0.0, 255.0) as u8;
    }
}

fn normalize_min_max(sample: &mut ImageSample) {
    let mut min_value = u8::MAX;
    let mut max_value = u8::MIN;

    for value in &sample.bytes {
        min_value = min(min_value, *value);
        max_value = max(max_value, *value);
    }

    if min_value == max_value {
        return;
    }

    let range = (max_value - min_value) as f32;
    for value in &mut sample.bytes {
        let shifted = (*value - min_value) as f32;
        *value = ((shifted / range) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

fn index(width: usize, channels: usize, x: usize, y: usize, channel: usize) -> usize {
    ((y * width + x) * channels) + channel
}

fn infer_square_dimensions_and_channels(image_bytes: &[u8]) -> Option<(usize, usize, usize)> {
    if image_bytes.is_empty() {
        return None;
    }

    let len = image_bytes.len();
    for channels in [1usize, 3usize, 4usize] {
        if !len.is_multiple_of(channels) {
            continue;
        }

        let pixels = len / channels;
        let side = (pixels as f64).sqrt() as usize;
        if side.saturating_mul(side) == pixels {
            return Some((side, side, channels));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn striped_square_8x8() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                if (x + y) % 2 == 0 {
                    bytes.push(230);
                } else {
                    bytes.push(10);
                }
            }
        }
        bytes
    }

    #[test]
    fn pipeline_is_deterministic_with_same_seed() {
        let pipeline = ImageTransformPipeline::new(vec![
            ImageTransform::RandomHorizontalFlip { probability: 0.5 },
            ImageTransform::RandomCropResize { min_scale: 0.7 },
            ImageTransform::BrightnessContrastJitter {
                max_brightness_delta: 0.2,
                max_contrast_delta: 0.2,
            },
            ImageTransform::GaussianNoise {
                probability: 1.0,
                stddev: 0.05,
            },
            ImageTransform::NormalizeMinMax,
        ]);
        let image = striped_square_8x8();

        let mut rng1 = TransformRng::new(42);
        let mut rng2 = TransformRng::new(42);
        let out1 = pipeline.apply(image.as_slice(), &mut rng1, PipelineMode::Train);
        let out2 = pipeline.apply(image.as_slice(), &mut rng2, PipelineMode::Train);

        assert_eq!(out1, out2);
    }

    #[test]
    fn train_and_eval_modes_apply_different_randomness() {
        let pipeline = ImageTransformPipeline::new(vec![
            ImageTransform::RandomHorizontalFlip { probability: 1.0 },
            ImageTransform::BrightnessContrastJitter {
                max_brightness_delta: 0.25,
                max_contrast_delta: 0.2,
            },
            ImageTransform::NormalizeMinMax,
        ]);

        let image = striped_square_8x8();
        let mut rng_train = TransformRng::new(7);
        let mut rng_eval = TransformRng::new(7);

        let train = pipeline.apply(image.as_slice(), &mut rng_train, PipelineMode::Train);
        let eval = pipeline.apply(image.as_slice(), &mut rng_eval, PipelineMode::Eval);

        assert_ne!(train, eval);
        assert_eq!(train.len(), image.len());
        assert_eq!(eval.len(), image.len());
    }

    #[test]
    fn dataloader_drop_last_and_shuffle_are_stable() {
        let records = vec![
            LabeledImageRecord::new("a", vec![1u8; 64]),
            LabeledImageRecord::new("b", vec![2u8; 64]),
            LabeledImageRecord::new("c", vec![3u8; 64]),
            LabeledImageRecord::new("d", vec![4u8; 64]),
            LabeledImageRecord::new("e", vec![5u8; 64]),
        ];
        let loader = CnnDataLoader::new(
            records,
            CnnDataLoaderOptions {
                batch_size: 2,
                shuffle: true,
                drop_last: true,
                seed: 123,
                prefetch_hint: 2,
            },
        );

        let epoch0 = loader.epoch_batches(0);
        let epoch0_again = loader.epoch_batches(0);

        assert_eq!(epoch0, epoch0_again);
        assert_eq!(epoch0.len(), 2);
        assert!(epoch0.iter().all(|batch| batch.records.len() == 2));
    }
}
