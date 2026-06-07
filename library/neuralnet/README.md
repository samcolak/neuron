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
