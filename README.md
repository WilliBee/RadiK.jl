# RadiK.jl

A backend-agnostic GPU top-k selection library for Julia implementing radix-based filtering with efficient batch processing.

Adapted from original CUDA C++ [RadiK](https://github.com/leefige/radik/) implementation described in the paper ["RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection"](https://arxiv.org/abs/2501.14336).

## Features

- **Backend-agnostic GPU implementation** using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl)
- **3-pass radix filtering**: Efficiently reduces search space using sequential radix scan
- **Batch processing**: Find top-k from multiple independent datasets in single kernel launches
- **Vectorized memory loads**: Generates vload instructions for optimal throughput
- **GPU batched bitonic sort**: For K ≤ 4096, falls back to AcceleratedKernels for larger K

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/WilliBee/BitonicSort.jll")  # not yet registered
Pkg.add(url="https://github.com/WilliBee/RadiK.jl")         # not yet registered
```

### Backend Requirements

RadiK.jl supports all GPU backends available through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl):

```julia
# For Apple Silicon
using Pkg
Pkg.add("Metal")

# For NVIDIA GPUs
Pkg.add("CUDA")
```

## Quick Start

### Single Task (Basic)

```julia
using RadiK
using CUDA  # or Metal, AMDGPU, oneAPI

# Create GPU data
data = CUDA.randn(Float32, 1_000_000)

# Find top 100 largest elements (sorted descending)
values, indices = topk(data, 100; largest=true, rev=false)

# Result: values contains top-100 elements, indices track positions
```

### Workspace Reuse (High Performance)

```julia
using RadiK
using CUDA

k = 100
n = 10_000_000

# Pre-allocate workspace (can be reused!)
ws = RadiKWorkspace(CUDABackend(), n, 1)
result = CUDA.zeros(Float32, k)
idx_in = CUDA.zeros(Int32, n)
idx_out = CUDA.zeros(Int32, k)
data = CUDA.randn(Float32, n)

# Reuse workspace for multiple calls - zero allocation overhead
for i in 1:100
    rand!(data)
    topk_radix_select!(result, idx_out, ws, data, idx_in, Int32[n], k)
    # Workspace and buffers reused automatically
end
```

### Multi-Task Batch Processing

```julia
using RadiK
using CUDA

# 4 independent tasks with different sizes
task_lens = [1000, 2000, 1500, 500]
total_len = sum(task_lens)

# Concatenated input data
data = CUDA.randn(Float32, total_len)
indices = CUDA.collect(Int32(1):Int32(total_len))

# Pre-allocate workspace (max stride = 2000, 4 tasks)
ws = RadiKWorkspace(CUDABackend(), 2000, 4)

# Pre-allocate outputs (2D: k × num_tasks)
result = CUDA.zeros(Float32, 64, 4)
indices_out = CUDA.zeros(Int32, 64, 4)

# Find top-64 for all 4 tasks in one call
result, indices_out = topk_radix_select!(
    result, indices_out, ws,
    data, indices, Int32.(task_lens), Int32(64)
)

# Each task's results are in result[:, task_id]
```

## API Reference

### RadiK.topk

```
topk(data, k; largest=true, rev=false)
```

Convenience wrapper that allocates all necessary buffers.

**Arguments:**
- `data::AbstractArray{Float32}`: Input data array (any BackendArray)
- `k::Integer`: Number of top elements to return
- `largest::Bool=true`: Find largest (true) or smallest (false) elements
- `rev::Bool=false`: Sort output in descending (true) or ascending (false) order

**Returns:**
- `values::AbstractArray{Float32}`: Top-k values (length k)
- `indices::AbstractArray{Int32}`: Original positions of top-k values

**Example:**
```julia
values, indices = topk(data, 100; largest=true, rev=false)
```

### RadiK.topk_radix_select!

```
topk_radix_select!(val_in, val_out, k, ws, idx_in, idx_out, task_lens, [::Val{LARGEST}=Val(true)], [::Val{ASCEND}=Val(true)], [::Val{WITHSCALE}=Val(false)], [::Val{WITHIDXIN}=Val(false)], [::Val{WITHPACKING}=Val(true)])
```

Find top-k elements using radix-based selection (full control, pre-allocated).

**Positional Arguments:**
- `val_in::AbstractArray{Float32}`: Input data array (any BackendArray)
- `val_out::AbstractArray{Float32}`: Pre-allocated output array [k, num_tasks]
- `k::Int32`: Number of top elements to select
- `ws::RadiKWorkspace`: Pre-allocated workspace for memory reuse
- `idx_in::AbstractArray{Int32}`: Input indices array (for index tracking)
- `idx_out::AbstractArray{Int32}`: Pre-allocated output indices [k, num_tasks]
- `task_lens::AbstractVector{Int32}`: Length of each task

**Compile-time Parameters (via Val{}):**
- `Val{LARGEST}=Val(true)`: Find largest (true) or smallest (false)
- `Val{ASCEND}=Val(true)`: Sort output ascending (true) or descending (false)
- `Val{WITHSCALE}=Val(false)`: Enable value scaling for numerical stability
- `Val{WITHIDXIN}=Val(false)`: Whether input indices are provided
- `Val{WITHPACKING}=Val(true)`: Use vectorized loads (recommended)

**Returns:**
- `val_out::AbstractArray{Float32}`: Top-k values (modified in-place)
- `idx_out::AbstractArray{Int32}`: Top-k indices (modified in-place)

### RadiK.RadiKWorkspace

```
RadiKWorkspace(backend, n, num_tasks, [T=Int], [ValT=Float32])
```

Pre-allocated workspace for zero-allocation repeated calls.

**Arguments:**
- `backend`: KernelAbstractions backend (CUDABackend, MetalBackend, etc.)
- `n::Integer`: Maximum stride (longest task length this workspace can handle)
- `num_tasks::Integer`: Number of independent tasks
- `T::Type=Int`: Index type for buffers (default: Int)
- `ValT::Type=Float32`: Value type for buffers (default: Float32)

**Returns:**
- `ws::RadiKWorkspace`: Pre-allocated workspace for reuse

**Example:**
```julia
# Allocate workspace for max stride of 10000 with 4 tasks
ws = RadiKWorkspace(CUDABackend(), 10000, 4)

# Can reuse for any call where max(task_lens) ≤ 10000
topk_radix_select!(data1, result1, Int32(64), ws, idx1_in, idx1_out, Int32[5000, 6000, 4000, 5500])
topk_radix_select!(data2, result2, Int32(64), ws, idx2_in, idx2_out, Int32[2500, 3000, 2000, 2500])
```

## Performance

### Design Characteristics
- **3-pass radix filtering**: Reduces search space exponentially (1/4096 → 1/256 → 1/16)
- **Vectorized loads**: Uses 4-wide loads for optimal memory bandwidth
- **Batch efficient**: Multi-task processing amortizes kernel launch overhead
- **Consistent performance**: Independent of data distribution (radix-based)

### Benchmark Results (single-task)

**Running benchmarks :**
```bash
cd benchmarks
BACKEND=cuda julia --project=. radik_single_task_benchmark.jl
```



#### Original C++ benchmark

We run the benchmark compiled directly from the orginal [RadiK](https://github.com/leefige/radik/) repo in Google Colab using `benchmarks/radiK_paper_benchmark.ipynb` :
Running on Google Colab with T4 GPU :

```
Timings CUDA C++ radik (Burst in milliseconds):
         N\K               16       32       64      128      256      512     1024     2048     4096
-----------------------------------------------------------------------------------------------------
2^21 ( 2,097,152):       4.18     3.19     3.08     3.10     3.16     3.09     3.10     3.19     3.43
2^23 ( 8,388,608):       3.41     3.33     3.52     3.41     3.43     3.33     3.61     3.68     3.53
2^25 (33,554,432):       4.44     4.47     5.78     5.35     4.52     4.54     4.54     4.52     4.68
2^27 (134,217,728):      9.04     8.89     8.85    10.54     8.98     8.97     8.89     8.96     9.01
2^29 (536,870,912):     27.30    27.08    27.11    27.11    27.47    27.11    28.17    28.72    28.64
```

Timings are slightly slower than reported in the paper, likely due to overhead.


#### Julia RadiK.jl

Running on Google Colab with T4 GPU :

```
Timings RadiK.jl (Burst / Steady in milliseconds):
┌──────┬───────────────┬───────────────┬───────────────┬──────────────┬───────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
│      │          K=16 │          K=32 │          K=64 │        K=128 │         K=256 │         K=512 │        K=1024 │        K=2048 │        K=4096 │
├──────┼───────────────┼───────────────┼───────────────┼──────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ 2^21 │   0.54 / 0.66 │   0.56 / 0.64 │   0.55 / 0.61 │  0.55 / 0.61 │   0.55 / 0.63 │   0.56 / 0.67 │   0.57 / 0.66 │   0.58 / 0.64 │   0.64 / 0.71 │
│ 2^23 │    1.04 / 1.2 │   1.04 / 1.23 │   1.07 / 1.22 │   1.04 / 1.2 │    1.04 / 1.2 │   1.05 / 1.23 │   1.06 / 1.25 │    1.2 / 1.32 │   1.26 / 1.38 │
│ 2^25 │   2.77 / 3.69 │   2.94 / 3.69 │   2.78 / 3.69 │  2.93 / 3.68 │   2.85 / 3.69 │   2.87 / 3.69 │   2.88 / 3.71 │   3.13 / 3.87 │   3.16 / 3.93 │
│ 2^27 │  8.54 / 12.04 │  8.84 / 12.01 │  8.73 / 12.06 │ 8.58 / 12.04 │  8.71 / 12.03 │  8.82 / 12.05 │  8.66 / 12.03 │  9.39 / 12.57 │  9.39 / 12.62 │
│ 2^29 │  25.73 / 33.5 │ 28.79 / 39.92 │ 29.28 / 39.61 │ 29.0 / 40.06 │ 29.15 / 39.74 │ 29.39 / 39.45 │ 28.87 / 39.46 │ 30.39 / 40.46 │ 30.58 / 40.49 │
└──────┴───────────────┴───────────────┴───────────────┴──────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
```

### Other Benchmark Results (multi-task, other backends)

See `bencharmarks/README.md`

## TODO

- [ ] Test and support additional datatypes (Float16, Float64, Int16, Int64, etc.)
- [ ] Explore precompilation for faster first-call performance
- [ ] Add multi-backend testing to CI/CD pipeline

## Testing

Run the test suite:

```bash
# Metal (Apple Silicon)
BACKEND=metal julia --project=. -e 'using Pkg; Pkg.test()'

# CUDA (NVIDIA)
BACKEND=cuda julia --project=. -e 'using Pkg; Pkg.test()'
```

## References and Acknowledgments

**RadiK Paper:**
- Li, Y., Zhou, B., Zhang, J., Wei, X., Li, Y., & Chen, Y. (2024). "RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection." *Proceedings of the 38th ACM International Conference on Supercomputing* [arXiv:2501.14336](https://arxiv.org/abs/2501.14336)

**Original CUDA C++ Implementation:**
- [RadiK](https://github.com/leefige/radik/) by Li, Y., Zhou, B., Zhang, J., Wei, X., Li, Y., & Chen, Y.

**Foundational JuliaGPU Ecosystem:**
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) - The foundational backend-agnostic GPU kernel framework that makes portable GPU programming possible
- [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl) - The awesome library providing essential warp-level GPU intrinsics (shuffles, vload, etc.)
- [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) - High-performance GPU kernels used for fallback sorting
- [JuliaGPU](https://github.com/JuliaGPU) - The incredible community driving GPU computing innovation in Julia

**Backend Packages:**
- [Metal.jl](https://github.com/JuliaGPU/Metal.jl) - Apple Silicon GPU backend
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) - NVIDIA GPU backend
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) - AMD GPU backend
- [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) - Intel GPU backend

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
