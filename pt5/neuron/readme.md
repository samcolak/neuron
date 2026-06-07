
# Neuron Application Test Suite

Latest test suite and examples for the Library (in ../library) - Note that the code is test functionality and gives an overview of the way the library functions

The platform has been rigerously built and tested on M1/M2 and M5 Cores without issue - It also is performant on an Intel i9(Apple) Processor too

# Using external backends...

The functionality (Tensors) enable you to offload the performance to external hardware (GPU or vCores) based upon accessibility to the appropriate hardware

Again this is NOT feature complete - I welcome any additional help if someone wishes to assist

## Using CUDA (For NVidia GPUS)

Sorry at the moment this is not implemented (yet) - Ill sort this out shortly tho !

## Using MLX (For Apple Mx Device)

The code actively runs all tests through the CPU unless you modify the default feature flag to MLX - Be aware though that there are some hoops to jump through.

## Using MLX - A quick guide

Ok - Firstly make sure CMAKE is installed via brew or your package manager

The make sure you have installed the metal framework

```bash
xcrun -sdk macosx metal -v
```

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun -sdk macosx metal -v
```

Head to the root of the directory in which you have cloned the repo.

If you are using the crate-based MLX path and already have MLX installed locally, you can skip the source build steps below.

Only run these commands if you want to build MLX from local source in `.deps/mlx` and install it into `.local/apple-mlx`:

```bash
rm -rf .deps/mlx/build .local/apple-mlx
cmake -S .deps/mlx -B .deps/mlx/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_METAL=ON
cmake --build .deps/mlx/build -j
cmake --install .deps/mlx/build --prefix .local/apple-mlx
```

Once this is done, MLX should be then correctly bounded to the GPU cores (as opposed to the CPU)