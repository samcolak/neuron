#!/usr/bin/env bash
set -euo pipefail

echo "Checking Xcode command line tools..."
xcode-select -p >/dev/null

if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
  echo "Metal toolchain is already installed."
  xcrun -sdk macosx metal -v
  exit 0
fi

echo "Metal toolchain not found. Installing..."
xcodebuild -downloadComponent MetalToolchain

echo "Verifying Metal toolchain..."
xcrun -sdk macosx metal -v

echo "Metal toolchain is ready."
