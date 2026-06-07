# neuralnet

`neuralnet` is versioned through `Cargo.toml` and git tags, not by renaming the source directory.

## Versioning guidance

- Keep the source directory name stable as `neuralnet/`.
- Bump the crate version in `Cargo.toml` when publishing releases.
- Use git tags to mark released versions.
- If multiple versions need to coexist locally, use separate checkouts or git worktrees instead of a versioned directory plus symlink.

## Current state

- Package name: `neuralnet`
- Crate version: `0.1.0`

## MLX offloading

- `offloading-mlx` now uses the repo-local `vendor/apple-mlx` override instead of the crates.io copy directly.
- The vendored override skips the stale `mlx/c/fft.cpp` wrapper so builds stay compatible with the current Homebrew MLX headers used on this machine.
- The repo no longer ships MLX runtime binaries. To use `offloading-mlx`, link an external MLX prefix via `APPLE_MLX_PREFIX`, `MLX_DIR`, `CMAKE_PREFIX_PATH`, or a symlink at `vendor/apple-mlx/.linked/mlx-prefix`.
