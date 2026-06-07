#!/usr/bin/env bash
set -euo pipefail

echo "Checking Metal toolchain..."
xcrun -sdk macosx metal -v

echo "Checking Metal SDK path..."
xcrun -sdk macosx --show-sdk-path

echo "Metal toolchain check passed."
