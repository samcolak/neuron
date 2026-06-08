#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURALNET_DIR="$ROOT_DIR/library/neuralnet"
NEURON_DIR="$ROOT_DIR/pt5/neuron"
OUT_DIR="$ROOT_DIR/coverage"

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  echo "cargo-llvm-cov is not installed. Install with: cargo install cargo-llvm-cov"
  exit 1
fi

MIN_LINES="${COVERAGE_MIN_LINES:-0}"
echo "Using coverage line threshold: ${MIN_LINES}%"

mkdir -p "$OUT_DIR"

run_cov() {
  local crate_dir="$1"
  local crate_name="$2"

  echo
  echo "==> Running coverage for ${crate_name} (${crate_dir})"
  pushd "$crate_dir" >/dev/null

  cargo llvm-cov clean --workspace
  cargo llvm-cov \
    --workspace \
    --summary-only \
    --fail-under-lines "$MIN_LINES"

  cargo llvm-cov \
    --workspace \
    --lcov \
    --output-path "$OUT_DIR/${crate_name}.lcov"

  popd >/dev/null
}

run_cov "$NEURALNET_DIR" "neuralnet"
run_cov "$NEURON_DIR" "neuron"

echo
echo "Coverage reports generated:"
echo "  $OUT_DIR/neuralnet.lcov"
echo "  $OUT_DIR/neuron.lcov"
