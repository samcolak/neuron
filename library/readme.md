# neuralnet

neuralnet is a Rust-native machine learning library for multimodal experimentation and production-minded model pipelines.

It is built for teams that want practical model performance, strict control over execution behavior, and predictable deployment ergonomics across CPU and accelerator-backed environments.

## Why this library is valuable

- Strong performance baseline on CPU with optional accelerator offloading paths.
- Unified tensor backend abstraction with centralized runtime backend selection and fallback policy.
- Practical CNN training and inference utilities that support batched and coalesced workflows.
- Multimodal model interfaces that combine image and text oriented paths in one crate.
- Snapshot and checkpoint support for repeatable training and recovery workflows.
- Rust ownership and type safety benefits for long-running training services.

## Core benefits in practice

### 1) Faster iteration from one codebase

The same model and training code can run on CPU-first setups and accelerator-capable setups using the backend abstraction layer. This reduces environment-specific branching in trainer logic and keeps experimentation loops tight.

### 2) Better production control

Backend selection is centralized and policy-driven, so availability and fallback are managed in one place rather than spread across training code.

### 3) Throughput-focused inference paths

The CNN pipeline includes batch-oriented prediction APIs and coalescing scheduler patterns designed to improve request throughput and reduce overhead from small request bursts.

### 4) Enterprise-friendly operational behavior

The library includes checkpointing and snapshot flows so model state can be persisted and restored cleanly during iterative training and deployment rollouts.

## Feature highlights

- Tensor backend abstraction across CPU, CUDA, and MLX-capable implementations.
- CNN classifier and trainer APIs for supervised image workflows.
- Multimodal core modules for broader experiments beyond pure image classification.
- Training utilities and metrics for evaluating model quality trends.
- Optional Adam optimizer support enabled by default.

## Backend and runtime strategy

- CPU is always available and acts as the reliability baseline.
- Optional accelerator backends are enabled through Cargo features.
- Runtime backend preference can be set through environment variables:
	- NEURALNET_TENSOR_BACKEND
	- NEURALNET_BACKEND
- Accepted backend values:
	- cpu
	- cuda
	- mlx
	- distributed (aliases: p2p, swarm)
	- auto
- Fallback order is centralized:
	- preferred backend -> cpu fallback on runtime error
	- auto mode probes in this order: cuda -> mlx -> distributed -> cpu

This design keeps trainer/core code simpler while making backend expansion safer as new kernels are introduced.

## Distributed tensor units (DTUs)

Distributed tensor units provide remote tensor operation execution over the built-in libp2p transport so heavy tensor work can be offloaded to remote peers while keeping the same high-level tensor API.

### What DTUs execute today

- Remote tensor operations:
	- Conv2dValid
	- MaxPool2d
	- GlobalAveragePool2d
	- Relu
	- ConvReluMaxPool2dValid
- Remote feature-stack forward execution:
	- FeatureStackForward over one or more convolution blocks

The distributed backend is exposed as tensor backend name distributed and runs through a transport-backed executor.

### Activation and feature flags

- Enable distributed backend support at build time:
	- backend-distributed
- Select distributed backend at runtime:
	- NEURALNET_TENSOR_BACKEND=distributed
	- NEURALNET_BACKEND=distributed

Note: auto backend selection checks cuda, then mlx, then distributed. If distributed is required, set it explicitly.

### Peer and transport configuration

Client-side distributed tensor execution uses these environment variables:

- NEURALNET_DISTRIBUTED_LOCAL_PEER
- NEURALNET_DISTRIBUTED_LOCAL_PLATFORM
- NEURALNET_DISTRIBUTED_LOCAL_ACCELERATOR
- NEURALNET_DISTRIBUTED_TARGET_PEER
- NEURALNET_DISTRIBUTED_TARGET_PLATFORM
- NEURALNET_DISTRIBUTED_TARGET_ACCELERATOR
- NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS
- NEURALNET_DISTRIBUTED_TIMEOUT_MS
- NEURALNET_DISTRIBUTED_DISCOVERY_WAIT_MS
- NEURALNET_DISTRIBUTED_SWARM_NAME (or NEURALNET_SWARM_NAME)
- NEURALNET_DISTRIBUTED_SWARM_VERSION (or NEURALNET_SWARM_VERSION)

Distributed libp2p transport/server controls also include:

- NEURALNET_DISTRIBUTED_SERVER_BACKEND
- NEURALNET_DISTRIBUTED_DEBUG

Bootstrap peer format for NEURALNET_DISTRIBUTED_BOOTSTRAP_PEERS is a comma-separated list of peer@multiaddr values, for example:

- 12D3KooWExamplePeerA@/ip4/10.0.0.12/udp/9000/quic-v1,12D3KooWExamplePeerB@/ip4/10.0.0.13/udp/9000/quic-v1

### Execution behavior and fallback

- The distributed executor initializes once and warms up peer discovery before first remote tensor operation.
- If exactly one remote peer is discovered, it is selected automatically.
- If multiple remote peers are discovered, set NEURALNET_DISTRIBUTED_TARGET_PEER explicitly.
- If no peer is discovered, distributed execution fails with guidance to set bootstrap peers or wait for discovery.
- During tensor op failover, backend routing can degrade to local CPU fallback (when CPU fallback is enabled).

### Server-side runtime

The distributed server runtime is available through:

- DistributedServerConfig
- DistributedServerRuntime

Server capability announcements and peer discovery are handled through the same libp2p transport layer used by distributed tensor execution.

## Versioning guidance

- Keep the source directory name stable as neuralnet.
- Bump the crate version in Cargo.toml when publishing releases.
- Use git tags to mark released versions.
- If multiple versions need to coexist locally, use separate checkouts or git worktrees.

## Current state

- Package name: neuralnet
- Crate version: 0.1.0

## MLX offloading notes

- offloading-mlx uses the repo-local vendor/apple-mlx override.
- The vendored override skips the stale mlx/c/fft.cpp wrapper for compatibility with current Homebrew MLX headers.
- The repo does not ship MLX runtime binaries.
- To use offloading-mlx, link an external MLX prefix via APPLE_MLX_PREFIX, MLX_DIR, CMAKE_PREFIX_PATH, or a symlink at vendor/apple-mlx/.linked/mlx-prefix.
