# RadiK.jl

A backend-agnostic GPU top-k selection library for Julia implementing radix-based filtering with efficient batch processing.

Adapted from original CUDA C++ [RadiK](https://github.com/leefige/radik/) implementation described in the paper ["RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection"](https://arxiv.org/abs/2501.14336).

## Features

- **Backend-agnostic GPU implementation** using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl)
- **3-pass radix filtering**: Efficiently reduces search space using sequential radix scan
- **Batch processing**: Find top-k from multiple independent datasets in single kernel launches
- **Vectorized memory loads**: Generates vload instructions for optimal throughput
- **GPU batched bitonic sort**: For K ≤ 4096, falls back to AcceleratedKernels for larger K

> **⚠️ Important Type Constraint**
>
> RadiK.jl currently only supports `Float32` input data. The algorithm is specifically optimized for IEEE 754 32-bit floating-point operations. If you have data in other types (e.g., `Float64`, `Float16`), you must convert it to `Float32` before use:
>
> ```julia
> data = adapt(backend, Float32.(your_data))
> ```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/WilliBee/KernelIntrinsics.jl")  # fork awaiting PR
Pkg.add(url="https://github.com/WilliBee/BitonicSort.jl")       # not yet registered
Pkg.add(url="https://github.com/WilliBee/RadiK.jl")             # not yet registered
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

### Simple Usage (Automatic Allocation)

```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()

# Create GPU data
data = adapt(backend, randn(Float32, 1_000_000))

# Find top 100 largest elements (sorted descending)
values, indices = topk(data, 100; largest=true, rev=false)
```

### Find Smallest Elements

```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()
data = adapt(backend, randn(Float32, 1_000_000))

# Find bottom 100 smallest elements
values, indices = topk(data, 100; largest=false)
```

### Control Sort Order

```julia
# Largest values, sorted ascending (1st is smallest of top-k)
values, indices = topk(data, 100; largest=true, rev=false)

# Smallest values, sorted descending (1st is largest of bottom-k)
values, indices = topk(data, 100; largest=false, rev=true)
```

### Custom Indices

```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()

# Your custom indexing (e.g., global dataset indices)
data = adapt(backend, Float32[1000, 1001, 1002, 1003, 1004])
indices_array = adapt(backend, Int32[1000, 1001, 1002, 1003, 1004])
values, idxs = topk(data, 2; indices=indices_array)
```

### Batch Processing

```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()

# Concatenated data with 3 tasks
data = adapt(backend, randn(Float32, 10_000))
task_lens = [3000, 4000, 3000]

# Find top-100 for all tasks
values, indices = topk(data, task_lens, 100)
# Returns: values (100×3 Matrix), indices (100×3 Matrix)
```

### Mutating Pre-allocated Outputs

```julia
using RadiK, CUDA, KernelAbstractions as KA, Adapt

backend = CUDABackend()

# Pre-allocate outputs for 1000 iterations
val_out = KA.zeros(backend, Float32, 100, 1000)
idx_out = KA.zeros(backend, Int32, 100, 1000)
data = adapt(backend, randn(Float32, 1_000_000))
task_lens = repeat([1000], 1000)
topk!(val_out, idx_out, data, task_lens, 100)
```

### Maximum Performance (Workspace Reuse)

```julia
using RadiK, CUDA, KernelAbstractions as KA, Adapt

backend = CUDABackend()

# Pre-allocate workspace for repeated calls
ws = RadiKWorkspace(backend, 10_000, 1)

val_out = KA.zeros(backend, Float32, 100)
idx_out = KA.zeros(backend, Int32, 100)
data = adapt(backend, randn(Float32, 10_000))

# Reuse workspace
topk_radix_select!(val_out, idx_out, ws, data, KA.zeros(backend, Int32, 0), [10_000], 100)
```

## API Reference

### RadiK.topk / topk!

**Non-mutating API:**
```julia
topk(data, k; indices=nothing, largest=true, rev=false)
topk(data, task_lens, k; indices=nothing, largest=true, rev=false)
```

Allocates output arrays automatically.

**Mutating API:**
```julia
topk!(val_out, idx_out, data, k; indices=nothing, largest=true, rev=false)
topk!(val_out, idx_out, data, task_lens, k; indices=nothing, largest=true, rev=false)
```

Writes to pre-allocated output arrays.

**Arguments:**
- `data::AbstractArray{Float32}`: Input data array
- `k::Integer`: Number of top elements to select
- `task_lens::AbstractVector{<:Integer}`: For batch mode, elements per task (use `[n]` for single task)
- `val_out::AbstractArray`: Pre-allocated output array (mutated)
- `idx_out::AbstractArray`: Pre-allocated output array (mutated)
- `indices::Union{Nothing, AbstractArray}`: Custom input indices (default: nothing for automatic)
- `largest::Bool=true`: Find largest (true) or smallest (false)
- `rev::Bool=false`: Sort output descending (true) or ascending (false)

**Returns:**
- `topk`: Tuple `(values, indices)` - allocated arrays
- `topk!`: Nothing (mutates outputs in-place)

### RadiK.topk_radix_select!

Low-level API with maximum control over pre-allocated workspace and outputs.

```julia
topk_radix_select!(val_out, idx_out, ws, data, indices, task_lens, k;
    ::Val{LARGEST}=Val(true), ::Val{ASCEND}=Val(true),
    ::Val{WITHSCALE}=Val(false), ::Val{WITHPACKING}=Val(true),
    ::Val{VLOAD_SIZE_BYTES}=Val(32))
```

**Arguments:**
- `val_out::AbstractArray{Float32}`: Pre-allocated output array [k, num_tasks]
- `idx_out::AbstractArray{IdxT}`: Pre-allocated output indices [k, num_tasks]
- `ws::RadiKWorkspace`: Pre-allocated workspace
- `data::AbstractArray{Float32}`: Input data array
- `indices::AbstractArray{IdxT}`: Input indices array
- `task_lens::AbstractVector{I}`: Task lengths (for single task, use `[n]` where `n` is the task length)
- `k::Integer`: Number of top elements

**Compile-Time Parameters:**
- `Val{LARGEST}`: Find largest (true) or smallest (false) elements
- `Val{ASCEND}`: Sort output in ascending (true) or descending (false) order
- `Val{WITHSCALE}`: Enable value scaling for numerical stability
- `Val{WITHPACKING}`: Whether to use vectorized loads (default: true)
- `Val{VLOAD_SIZE_BYTES}`: Vectorized load width in bytes (default: 32)
  - Controls memory vectorization: `PACKSIZE = VLOAD_SIZE_BYTES ÷ sizeof(Float32)`
  - **Recommended**: 32 bytes (PACKSIZE=8) - optimal for both CUDA and Metal
  - **CUDA**: Can use 64 bytes (PACKSIZE=16) for potentially better performance
  - **Metal**: Maximum is 32 bytes - larger values will fail due to 32KB shared memory limit

### RadiK.RadiKWorkspace

Pre-allocated workspace for zero workspace re-allocation repeated calls.

```julia
RadiKWorkspace(backend, n, num_tasks, ::Type{I}=Int32)
```

**Arguments:**
- `backend`: KernelAbstractions backend
- `n::Integer`: Maximum stride (longest task length)
- `num_tasks::Integer`: Number of tasks
- `I::Type=Int32`: Integer type for buffers

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
- **Configurable vectorized loads**: Default 8-wide (32-byte) loads for optimal memory bandwidth across GPU backends
- **Batch efficient**: Multi-task processing amortizes kernel launch overhead
- **Consistent performance**: Independent of data distribution (radix-based)

### Benchmark Results (single-task)

Benchmarks are maintained in a separate repository: [https://github.com/WilliBee/RadiKBenchmarks](https://github.com/WilliBee/RadiKBenchmarks). To run benchmarks, see instructions in the repo's README.

#### Original C++ benchmark

We run the benchmark compiled directly from the orginal [RadiK](https://github.com/leefige/radik/) repo in Google Colab using [RadiKBenchmarks/radiK_paper_benchmark.ipynb](https://github.com/WilliBee/RadiKBenchmarks/blob/main/radiK_paper_benchmark.ipynb) :
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

Benchmark script: [RadiKBenchmarks/radik_single_task_benchmark.jl](https://github.com/WilliBee/RadiKBenchmarks/blob/main/radik_single_task_benchmark.jl)

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

See [RadiKBenchmarks/README.md](https://github.com/WilliBee/RadiKBenchmarks/blob/main/README.md)

## TODO

- [ ] Explore precompilation for faster first-call performance (https://github.com/JuliaLang/julia/pull/60747 might be helping ?)
- ?

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
