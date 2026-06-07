# apple-mlx

Rust bindings for Apple MLX via the official `mlx-c` C API.

This crate provides:

- `apple_mlx::raw`: generated raw bindings for the full `mlx-c` surface
- a thin safe layer for `Device`, `Stream`, `Array`, and `Complex32`
- runnable examples, including complex matrix multiplication and graph export

## What This Repo Does

This repository vendors `mlx-c`, generates Rust bindings in `build.rs`, and links them against an installed MLX build.

The build model is explicit:

- `mlx-c` is vendored in this crate
- MLX itself is installed locally by the repo `Makefile`
- no hidden MLX fetch happens during `cargo build`

## Clone

```bash
git clone https://github.com/ms3c/apple-mlx.git
cd apple-mlx
```

All commands below assume you are at the repo root:

```bash
ls Cargo.toml Makefile build.rs src/lib.rs
```

## Requirements

- macOS on Apple silicon
- Rust toolchain
- Xcode command line tools
- CMake

Install the basics if needed:

```bash
xcode-select --install
brew install cmake
rustup toolchain install stable
```

## CPU Build From Scratch

This is the shortest fully reproducible path.

1. Build and install MLX into the repo-local prefix:

```bash
make install-mlx
```

2. Build the Rust crate:

```bash
make build
```

3. Run tests:

```bash
make test
```

4. Run the main demo:

```bash
make run
```

5. Run a specific example:

```bash
make run-example EXAMPLE=example_graph
```

The local MLX install prefix used by the repo is:

```bash
$(pwd)/.local/apple-mlx
```

The `Makefile` exports these automatically:

```bash
CMAKE_PREFIX_PATH="$(pwd)/.local/apple-mlx"
MLX_DIR="$(pwd)/.local/apple-mlx/share/cmake/MLX"
```

## GPU Build With Metal

If the Metal compiler is available, the repo will build MLX with Metal enabled and the GPU examples will run against the GPU backend.

Install the Metal toolchain:

```bash
./scripts/install-metal-toolchain.sh
```

Verify it:

```bash
./scripts/check-metal-toolchain.sh
xcrun -sdk macosx metal -v
```

Then rebuild with Metal enabled:

```bash
make clean-mlx
make install-mlx
make build
```

Run the Metal example:

```bash
make run-example EXAMPLE=example_metal_kernel
```

The `Makefile` records whether MLX was built with `MLX_BUILD_METAL=ON` or `OFF`. If that mode changes, `make build`, `make run`, and `make run-example` will rebuild the local MLX install automatically.

## Examples

Run the core examples:

```bash
make run-complex
make run-example EXAMPLE=example
make run-example EXAMPLE=example_graph
make run-example EXAMPLE=example_export
make run-example EXAMPLE=example_grad
make run-example EXAMPLE=example_closure
make run-example EXAMPLE=example_safe_tensors
make run-example EXAMPLE=example_gguf
```

Run the GPU/Metal example:

```bash
make run-example EXAMPLE=example_metal_kernel
```

Check all examples compile:

```bash
make examples-check
```

## How The FFI Build Works

`build.rs` does three jobs:

1. runs `bindgen` on `vendor/mlx-c/mlx/c/mlx.h`
2. builds vendored `mlx-c` with CMake
3. links it against the installed MLX package exposed through `CMAKE_PREFIX_PATH` and `MLX_DIR`

Key repo files:

- `build.rs`
- `src/lib.rs`
- `src/main.rs`
- `examples/`
- `vendor/mlx-c/`
- `Makefile`

## Integrating Into Another Rust Project

There are two parts:

1. add the Rust crate
2. make sure your project can find an installed MLX prefix at build time

### From crates.io

Add this to your `Cargo.toml`:

```toml
[dependencies]
apple-mlx = "0.1"
```

Use it from Rust:

```rust
use apple_mlx::raw;
use apple_mlx::{Array, Complex32, Device, Stream};
```

### From a local checkout

If you want to work against this repo directly:

```toml
[dependencies]
apple-mlx = { path = "../apple-mlx" }
```

### Build Environment In The Consumer Project

Your consuming project must expose the MLX install location when Cargo builds `apple-mlx`.

If you installed MLX with this repo’s `Makefile`, export:

```bash
export CMAKE_PREFIX_PATH="/path/to/apple-mlx/.local/apple-mlx"
export MLX_DIR="/path/to/apple-mlx/.local/apple-mlx/share/cmake/MLX"
```

Then build your own project normally:

```bash
cargo build
```

### Consumer Project Makefile Example

If you want the same workflow in another repo:

```make
APPLE_MLX_PREFIX ?= /absolute/path/to/apple-mlx/.local/apple-mlx

export CMAKE_PREFIX_PATH := $(APPLE_MLX_PREFIX)
export MLX_DIR := $(APPLE_MLX_PREFIX)/share/cmake/MLX

build:
	cargo build

run:
	cargo run
```

## Minimal Consumer Example

```rust
use apple_mlx::demo_complex_matmul;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo_complex_matmul()?;
    Ok(())
}
```

Or use the raw bindings directly:

```rust
use apple_mlx::raw;

fn main() {
    unsafe {
        let cpu = raw::mlx_device_new(raw::mlx_device_type__MLX_CPU, 0);
        let _ = raw::mlx_device_free(cpu);
    }
}
```

## Verified Runs

Verified CPU flow:

```bash
make install-mlx
make run-example EXAMPLE=example_graph
```

Verified GPU flow:

```bash
./scripts/install-metal-toolchain.sh
make run-example EXAMPLE=example_metal_kernel
```

## Packaging Notes

- crate name: `apple-mlx`
- docs: `https://docs.rs/apple-mlx`
- repo: `https://github.com/ms3c/apple-mlx`
- docs.rs uses the `docs-only` feature to avoid native compilation during documentation builds

## Current Limits

- the raw binding surface is broad, but the safe Rust wrapper is still thin
- MLX must be installed on the machine building the crate
- this is currently a macOS Apple-silicon-focused crate
