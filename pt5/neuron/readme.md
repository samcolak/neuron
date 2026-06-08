
# Neuron App

Neuron is the runnable companion app for the `neuralnet` library. It exists to show the library in action, compare backend behavior, and provide realistic walkthroughs for training, inference, and multimodal flows.

## What this app gives you

- A polished entry point for exercising the library from the command line.
- Real walkthroughs for CNN training, batch inference, backend comparison, and multimodal demonstrations.
- A practical way to see how CPU, CUDA, and MLX execution modes behave on the current machine.
- A reproducible harness for validating performance, quality, and backend policy changes.

## Why it matters

- CPU remains the stable baseline for all runs.
- Accelerators can be enabled when available without changing the app structure.
- The app reflects the current runtime backend selection and fallback policy used by the library.
- It is useful both as a test surface and as a reference implementation for integrating the library into a larger service.

## Backend support

### CPU

CPU is always available and is the default execution path.

### CUDA

CUDA support is available through feature gating, but it remains less battle-tested than the CPU path.

### MLX for Apple Silicon

The app and library can run with MLX on Apple Silicon, but you must point the build at an external MLX prefix.

The external prefix must contain:

- `lib/libmlx.dylib`
- `lib/libjaccl.dylib`
- `lib/mlx.metallib`
- `share/cmake/MLX/MLXConfig.cmake`

You can configure it with:

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

## Running the app

Use the app as a comparison and validation harness while developing the library.

```bash
cargo run
```

To enable MLX when the environment is ready:

```bash
cargo run --features offloading-mlx
```

To enable CUDA:

```bash
cargo run --features offloading-cuda
```

## Notes

- The app is intentionally focused on demonstrating the library rather than providing a general-purpose UI.
- It is a good place to validate backend policy, performance changes, and new inference/training workflows.