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
Pkg.add("RadiK")
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
ws = RadiKWorkspace(CUDABackend(), k, 1, Int32[n])
result = CUDA.zeros(Float32, k)
idx_in = CUDA.zeros(Int32, k)
idx_out = CUDA.zeros(Int32, k)
data = CUDA.randn(Float32, n)

# Reuse workspace for multiple calls - zero allocation overhead
for i in 1:100
    rand!(data)
    topk_radix_select!(data, result, k, ws, idx_in, idx_out, Int32[n])
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

# Pre-allocate workspace
ws = RadiKWorkspace(CUDABackend(), 64, 4, Int32.(task_lens))

# Pre-allocate outputs (2D: k × num_tasks)
result = CUDA.zeros(Float32, 64, 4)
indices_out = CUDA.zeros(Int32, 64, 4)

# Find top-64 for all 4 tasks in one call
result, indices_out = topk_radix_select!(
    data, result, Int32(64), ws,
    indices, indices_out, Int32.(task_lens)
)

# Each task's results are in result[:, task_id]
```

## API Reference

### `topk(data, k; largest=true, rev=false)`

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

### `topk_radix_select!(val_in, val_out, k, ws, idx_in, idx_out, task_lens; ...)`

Find top-k elements using radix-based selection (full control, pre-allocated).

**Arguments:**
- `val_in::AbstractArray{Float32}`: Input data array (any BackendArray)
- `val_out::AbstractArray{Float32}`: Pre-allocated output array [k, num_tasks]
- `k::Int32`: Number of top elements to select
- `ws::RadiKWorkspace`: Pre-allocated workspace for memory reuse
- `idx_in::AbstractArray{Int32}`: Input indices array (for index tracking)
- `idx_out::AbstractArray{Int32}`: Pre-allocated output indices [k, num_tasks]
- `task_lens::AbstractVector{Int32}`: Length of each task
- `largest::Bool=true`: Find largest (true) or smallest (false)
- `ascend::Bool=true`: Sort output ascending (true) or descending (false)
- `with_scale::Bool=false`: Enable value scaling for numerical stability
- `with_idx_in::Bool=false`: Whether input indices are provided
- `with_packing::Bool=true`: Use vectorized loads (recommended)

**Returns:**
- `val_out::AbstractArray{Float32}`: Top-k values (modified in-place)
- `idx_out::AbstractArray{Int32}`: Top-k indices (modified in-place)

### `RadiKWorkspace(backend, k, num_tasks, task_lens, ValT=Float32)`

Pre-allocated workspace for zero-allocation repeated calls.

**Arguments:**
- `backend`: KernelAbstractions backend (CUDABackend, MetalBackend, etc.)
- `k::Integer`: Number of top elements
- `num_tasks::Integer`: Number of independent tasks
- `task_lens::AbstractVector{<:Integer}`: Length of each task
- `ValT::Type=Float32`: Value type (default: Float32)

**Returns:**
- `ws::RadiKWorkspace`: Pre-allocated workspace for reuse

## Performance

**Design Characteristics:**
- **3-pass radix filtering**: Reduces search space exponentially (1/4096 → 1/256 → 1/16)
- **Vectorized loads**: Uses 4-wide loads for optimal memory bandwidth
- **Batch efficient**: Multi-task processing amortizes kernel launch overhead
- **Consistent performance**: Independent of data distribution (radix-based)

**Benchmark Results**

RadiK vs CPU (`partialsort`) and Metal Performance Shaders (MPS) on Apple M4 10/10 24GB:

TO BE UPDATED


**Running benchmarks:**
```bash
cd benchmarks
julia --project=. -e 'using Metal; backend=MetalBackend(); include("radik_jl_benchmark.jl")'
```



## Testing

Run the test suite:

```bash
# Metal (Apple Silicon)
BACKEND=metal julia --project=. -e 'using Pkg; Pkg.test()'

# CUDA (NVIDIA)
BACKEND=cuda julia --project=. -e 'using Pkg; Pkg.test()'
```

## TODO

- [ ] Float16 and BFloat16 support ?
- [ ] Additional backend-specific optimizations ?

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
