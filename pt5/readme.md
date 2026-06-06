
# Pt.5 Learning

This phase adds CNN-ready tensor tooling and wires it into both the `neuralnet` library and the `neuron` app walkthrough.

## What Changed

- Pattern classifier + trainer bridge
- Training loop with checkpoint lifecycle (epoch, best, last, resume)
- Tensor foundation (`Tensor4D`) with CNN primitives
- CNN feature extractor for image-to-feature token conversion
- Batch image training/evaluation with confusion matrix + micro metrics
- App walkthrough integration for the CNN image path

## Project Layout

- `neuralnet/`: core library (brain, training, tensors, integration tests)
- `neuron/`: runnable app walkthrough and demo harness

## Run And Test

From the repository root (`pt5`):

```bash
cd neuralnet && cargo test
```

```bash
cd neuron && cargo test
```

```bash
cd neuron && cargo run --quiet
```

## App Walkthrough (Trainer)

The trainer walkthrough is in `neuron/src/trainer_walkthrough.rs` and now includes six steps.

- Step 1: single labeled training sample
- Step 2: batch training
- Step 3: evaluation + confusion matrix + macro/micro metrics
- Step 4: supervised training loop with early stopping/checkpoints
- Step 5: resume from best checkpoint
- Step 6: CNN image path demo on app pipeline

## Step 6: CNN Image Path Demo

Step 6 enables the optional CNN image feature path on a fresh `MultiModalBrain`:

- Calls `enable_default_cnn_image_path()`
- Trains on synthetic 8x8 grayscale image patterns
- Evaluates with confusion matrix and micro-F1
- Shows pre-train vs post-train probe behavior

Expected behavior in output:

- `pre-train image probe -> <unknown>`
- `cnn image final eval: accuracy=... micro_f1=...`
- confusion matrix rows for `animal_cat`, `animal_dog`, and unknown image bucket
- `post-train image probe -> animal_cat (...)`

## Key Files

- `neuralnet/src/tensor/tensor4d.rs`: tensor structure + `conv2d_valid` + `max_pool2d`
- `neuralnet/src/core/cnn_feature_extractor.rs`: CNN feature extraction for images
- `neuralnet/src/core/brain.rs`: optional classifier preprocessing hook for image CNN path
- `neuralnet/src/training/trainer.rs`: training/evaluation + confusion matrix + macro/micro metrics
- `neuralnet/src/core/integration.rs`: supervised pipeline + integration tests
- `neuron/src/trainer_fixtures.rs`: app walkthrough fixtures, including CNN image samples
- `neuron/src/trainer_walkthrough.rs`: app-side walkthrough and CNN step

## Current Baseline

- `neuralnet`: all tests passing
- `neuron`: all tests passing