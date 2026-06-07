
# Neural Network (RUST) Library

Built without any dependancies on existing Neural Network foundation models or libraries via 3rd parties
Do not use this (YET) in operational code - alot of changes forthcoming (and optimizations)
This is not intented for production environments (YET)
Distributed under GPL-3.0 License - Please observe the license (in the root) - All derived works MUST remain open source for the community

NB. Whilst i am not a qualified / certified mathematician, alot of work on Tensors uses mathematical models etc - If you find an error, please inform me asap !! Many thanks

Written by Samuel Colak (sam@samcolak.com)

## What Changed

- Pattern classifier + trainer bridge
- Training loop with checkpoint lifecycle (epoch, best, last, resume)
- Tensor foundation (`Tensor4D`) with CNN primitives
- CNN feature extractor for image-to-feature token conversion
- Batch image training/evaluation with confusion matrix + micro metrics
- App walkthrough integration for the CNN image path
- Support for CUDA (Not implemented YET!), MLX (Implemented) and CPU (Implemented) offloading of tensor models

## Project Layout

- `/library/neuralnet/`: core library (brain, training, tensors, integration tests)
- `/pt5/neuron/`: runnable app walkthrough and demo harness

Parts 1-4 are evolutions in the library... This is intended as the primary stable instance

## Run And Test

From the repository root (`pt5`):

```bash
cd /library/neuralnet && cargo test
```

```bash
cd /pt5/neuron && cargo test
```

```bash
cd /pt5/neuron && cargo run --quiet
```

## App Walkthrough (Trainer)

The trainer walkthrough is in `/pt5/neuron/src/trainer_walkthrough.rs` and now includes six steps.

- Step 1: single labeled training sample
- Step 2: batch training
- Step 3: evaluation + confusion matrix + macro/micro metrics
- Step 4: supervised training loop with early stopping/checkpoints
- Step 5: resume from best checkpoint
- Step 6: CNN image path demo on app pipeline

## App Walkthrough (Standalone CNN Classifier)

The standalone trainable image-classifier walkthrough is in `/pt5/neuron/src/cnn_classifier_walkthrough.rs`.

It validates:

- pre-train probe behavior (strict confidence threshold -> unknown)
- repeated batch training on synthetic image patterns
- post-train probe behavior
- confusion matrix + label metrics using the CNN trainer adapter

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

## CNN Classifier Validation

Use these focused commands when validating the standalone trainable image classifier path:

```bash
cd neuralnet && cargo test cnn_classifier -- --nocapture
```

```bash
cd neuralnet && cargo test cnn_classifier_snapshot_round_trip_preserves_predictions -- --nocapture
```

```bash
cd neuralnet && cargo test cnn_trainer -- --nocapture
```

```bash
cd neuralnet && cargo test linear_head -- --nocapture
```

Use these broader commands for full regression confidence:

```bash
cd neuralnet && cargo test
```

```bash
cd neuron && cargo test cnn_classifier_walkthrough -- --nocapture
```

```bash
cd neuron && cargo run --quiet
```

Expected validation signals:

- Core classifier tests pass (simple pattern learning + invalid image rejection).
- CNN classifier snapshot round-trip test passes (save/load preserves predictions).
- Trainer adapter tests pass (batch counts, confusion matrix wiring, confidence-threshold behavior, loss-trend behavior).
- Linear head tests pass (probability normalization, input-gradient shape, loss decrease).
- App walkthrough prints pre-train unknown and post-train labeled prediction for the cat probe.
- App walkthrough prints confusion matrix and label metrics for the CNN classifier flow.

## Key Files

- `/library/neuralnet/src/tensor/tensor4d.rs`: tensor structure + `conv2d_valid` + `max_pool2d`
- `/library/neuralnet/src/cnn/feature_extractor.rs`: CNN feature extraction for images
- `/library/neuralnet/src/cnn/classifier.rs`: trainable CNN image classifier backed by `LinearHead`, including save/load snapshot lifecycle and optional two-layer conv backbone (`new_two_layer`)
- `/library/neuralnet/src/core/brain.rs`: optional classifier preprocessing hook for image CNN path
- `/library/neuralnet/src/cnn/cnn_trainer.rs`: image trainer adapter that emits standard training reports/metrics
- `/library/neuralnet/src/training/trainer.rs`: training/evaluation + confusion matrix + macro/micro metrics
- `/library/neuralnet/src/core/integration.rs`: supervised pipeline + integration tests
- `/library/neuron/src/trainer_fixtures.rs`: app walkthrough fixtures, including CNN image samples
- `/library/neuron/src/trainer_walkthrough.rs`: app-side walkthrough and CNN step
- `/library/neuron/src/cnn_classifier_walkthrough.rs`: dedicated standalone CNN classifier walkthrough

## Current Baseline

- `/library/neuralnet`: all tests passing
- `/pt5/neuron`: all tests passing