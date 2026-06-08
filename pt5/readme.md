
# Neuron App

Neuron is the runnable companion app for the `neuralnet` library. It is designed to showcase the library in practice, compare backend behavior, and demonstrate realistic training, inference, and multimodal workflows.

If you want to understand what the library does, how it performs, and how it behaves across CPU, CUDA, and MLX environments, this is the place to start.

## What You Get

- A command-line entry point for exercising the library end to end.
- Walkthroughs for multimodal demos, trainer behavior, CNN training, batch inference, and RAG flows.
- A practical way to compare CPU, CUDA, and MLX execution on the current machine.
- A repeatable harness for validating performance, quality, and backend policy changes.

## At a Glance

- CPU is the default and the stable baseline.
- GPU-style execution is enabled only when the relevant backend is available.
- The app reflects the same runtime backend policy used by the library.
- The walkthroughs are intentionally concrete so they can serve as reference examples.

## Included Walkthroughs

When you run the app, it executes the following demos in sequence:

- Multimodal brain demo
- Trainer walkthrough
- CNN classifier walkthrough
- RAG walkthrough
- RAG dataset walkthrough
- Multimodal tensor walkthrough
- Brain stress walkthrough

These are intentionally written to demonstrate current library behavior, not to provide a general-purpose UI.

## Recommended Workflow

1. Run the app on CPU first to see the default behavior.
2. Enable MLX or CUDA only when your environment is ready.
3. Compare the walkthrough output to confirm backend policy, throughput, and quality changes.
4. Use the app as a sanity check before promoting library changes into a larger system.

## Why It Matters

- CPU remains the stable baseline for all runs.
- Accelerators can be enabled when available without changing the app structure.
- The app reflects the runtime backend selection and fallback policy used by the library.
- It is useful both as a test surface and as a reference implementation for integrating the library into a larger service.

## Quick Start

Run the default CPU-backed app:

```bash
cargo run
```

Run with MLX when your Apple Silicon environment is ready:

```bash
cargo run --features offloading-mlx
```

Run with CUDA when the feature and hardware are available:

```bash
cargo run --features offloading-cuda
```

## Backend Support

### CPU

CPU is always available and is the default execution path.

### CUDA

CUDA support is feature-gated and remains less battle-tested than the CPU path.

### MLX for Apple Silicon

The app and library can run with MLX on Apple Silicon, but you must point the build at an external MLX prefix.

The external prefix must contain:

- `lib/libmlx.dylib`
- `lib/libjaccl.dylib`
- `lib/mlx.metallib`
- `share/cmake/MLX/MLXConfig.cmake`

Configure it with:

```bash
export APPLE_MLX_PREFIX="/absolute/path/to/mlx-prefix"
```

or, if you prefer a local link:

```bash
ln -s /absolute/path/to/mlx-prefix ../library/neuralnet/vendor/apple-mlx/.linked/mlx-prefix
```

If you use Homebrew MLX, point to the prefix explicitly:

```bash
export APPLE_MLX_PREFIX="/opt/homebrew/opt/mlx"
```

If the Metal tooling is not ready, confirm Xcode Command Line Tools are set correctly:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun -sdk macosx metal -v
```

## Notes

- The app is intentionally focused on demonstrating the library rather than providing a general-purpose UI.
- It is a good place to validate backend policy, performance changes, and new inference/training workflows.