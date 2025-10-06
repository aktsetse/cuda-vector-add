# CUDA Vector Addition

A beginner-friendly CUDA program that compares CPU and GPU performance for vector addition.

What It Does
- Adds two large arrays on both CPU and GPU
- Measures execution time
- Reports the GPU speedup
- Verifies correctness

Build Instructions
### Requirements
- NVIDIA GPU + CUDA Toolkit
- CMake ≥ 3.18
- C++17 compiler

Build and Run
```bash
mkdir build && cd build
cmake ..
cmake --build . -j
./vector_add
