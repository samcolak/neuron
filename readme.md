# Neural Networks (in Rust)

A convolutional neural network with enterprise skillz built in Rust with no dependancies supporting hardware acceleration

Used in part of a 5 part article on LinkedIn (https://www.linkedin.com/in/samcolak) to understand how to develop Neural Networks using a stable and solid foundation

Written by Samuel Colak (sam@samcolak.com)

## Overview

- /library      : Multimodal Convolutional Neural Network Library for rust
- /pt1          : Part 1 of developing a neural network
- /pt2          : Part 2
- /pt3          : Part 3
- /pt4          : Part 4
- /pt5          : Active workload to test library functionality

## Distribution

The is distributed under the GPL-3 license

Any derived works released must be open-sourced - This is to benefit the community and foster development in the ML / NeuralNet space

## Coverage

Coverage checks are wired via `cargo-llvm-cov` for both active crates:

- `library/neuralnet`
- `pt5/neuron`

Run locally:

```bash
cargo install cargo-llvm-cov
bash scripts/coverage.sh
```

Optional local gate (line coverage %):

```bash
COVERAGE_MIN_LINES=35 bash scripts/coverage.sh
```

CI workflow:

- `.github/workflows/coverage.yml`
- uploads LCOV artifacts for both crates as `coverage/*.lcov`