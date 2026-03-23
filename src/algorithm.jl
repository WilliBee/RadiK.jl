using Adapt
using AcceleratedKernels
using BitonicSort
using KernelAbstractions: get_backend, synchronize
import KernelAbstractions as KA
using KernelIntrinsics: get_warpsize, device

const MAX_GRID_SIZE = 65535

"""
    RadiKWorkspace(backend, n, num_tasks, [I=Int32])

Temporary workspace buffers for radix-based top-k selection.

Pre-allocates with a fixed stride `n` to allow memory reuse across multiple calls
with varying task lengths. Follows the C++ approach: allocate once with maximum
expected stride, then reuse for any call with `max(task_lens) ≤ n`.

# Arguments
- `backend`: KernelAbstractions backend (CUDABackend, MetalBackend, etc.)
- `n`: Maximum stride (longest task length this workspace can handle)
- `num_tasks`: Number of tasks (1 for single-task, >1 for multi-task batch)
- `I`: (Optional) Integer type for internal buffers (default: Int32, GPU-optimized)

# Type Constraint
**Currently only Float32 is supported**. The value buffers are hardcoded as Float32
due to the 3-pass radix design optimized for IEEE 754 32-bit floating-point representation.

# Fields
- `histogram`: Histogram buffer (4096 bins × num_tasks)
- `val_buffer_1`: First ping-pong value buffer (n × num_tasks, Float32)
- `val_buffer_2`: Second ping-pong value buffer (n × num_tasks, Float32)
- `global_count`: Per-task global counters
- `task_len_buffer_1`: First ping-pong task lengths buffer
- `task_len_buffer_2`: Second ping-pong task lengths buffer
- `k_values`: K values per task
- `bin_ids`: Selected bin IDs per task

# Example
```
# Allocate workspace for maximum expected task length of 10000
ws = RadiKWorkspace(CUDABackend(), 10000, 4)

# Can reuse for any calls with max(task_lens) ≤ 10000
topk_radix_select!(data1, out1, Int32(100), ws, idx1_in, idx1_out, Int32[5000, 6000, 4000, 5500])
topk_radix_select!(data2, out2, Int32(100), ws, idx2_in, idx2_out, Int32[2500, 3000, 2000, 2500])

# Works with single-task too
ws = RadiKWorkspace(CUDABackend(), 10000, 1)
topk_radix_select!(data, result, Int32(100), ws, idx_in, idx_out, Int32[10000])
```

# Notes
- Value buffers are sized for `n` elements per task, not the actual task lengths
- This allows reuse across calls with different task lengths (as long as max ≤ n)
- For multi-task processing, `num_tasks` must match the number of tasks in each call
"""
struct RadiKWorkspace{I}
    histogram::AbstractArray{I, 2}
    val_buffer_1::AbstractArray{Float32, 2}
    val_buffer_2::AbstractArray{Float32, 2}
    global_count::AbstractArray{I}
    task_len_buffer_1::AbstractArray{I}
    task_len_buffer_2::AbstractArray{I}
    k_values::AbstractArray{I}
    bin_ids::AbstractArray{I}
end

function RadiKWorkspace(backend, n, num_tasks, ::Type{I}=Int32) where {I}
    histogram = KA.zeros(backend, I, 4096, num_tasks)
    val_buffer_1 = KA.zeros(backend, Float32, n, num_tasks)
    val_buffer_2 = KA.zeros(backend, Float32, n, num_tasks)
    global_count = KA.zeros(backend, I, num_tasks)
    task_len_buffer_1 = KA.zeros(backend, I, num_tasks)
    task_len_buffer_2 = KA.zeros(backend, I, num_tasks)
    k_values = KA.zeros(backend, I, num_tasks)
    bin_ids = KA.zeros(backend, I, num_tasks)

    return RadiKWorkspace(
        histogram, val_buffer_1, val_buffer_2, global_count,
        task_len_buffer_1, task_len_buffer_2, k_values, bin_ids
    )
end

"""
    topk_radix_select!(val_out, idx_out, ws, val_in, idx_in, task_lens, k; ...)

Main entry point for radix-based top-k selection.

# Template Parameters (compile-time via Val{})
- `Val{LARGEST}`: Find largest (true) or smallest (false) elements
- `Val{ASCEND}`: Sort output in ascending (true) or descending (false) order
- `Val{WITHSCALE}`: Enable value scaling for numerical stability
- `Val{WITHIDXIN}`: Whether input indices are provided
- `Val{WITHPACKING}`: Whether to use vectorized loads (compile-time)

# Arguments
- `val_out`: Output array for top-k results (mutated)
- `idx_out`: Output array for indices (mutated)
- `ws`: Pre-allocated RadiKWorkspace for memory reuse (mutated)
- `val_in`: Input data array (AbstractArray{Float32})
- `idx_in`: Input indices array (for index tracking)
- `task_lens`: Task lengths for multi-task batch processing (Integer type)
- `k`: Number of top elements to select (Integer type)

# Type Parameters
- `I`: Workspace integer type (default: Int32, GPU-optimized)
- `IdxT`: Index array type (can be different from workspace type)

# Type Constraints
- Value arrays must be `Float32` (algorithm is optimized for IEEE 754 32-bit floating-point)

# Returns
- `val_out`: Array containing top-k elements (sorted)
- `idx_out`: Array containing indices

# Examples
```
# Single task (allocate all required buffers)
data = CUDA.randn(Float32, 10000)
result = CUDA.zeros(Float32, 100)
idx_in = CUDA.zeros(Int32, 10000)
idx_out = CUDA.zeros(Int32, 100)
ws = RadiKWorkspace(CUDABackend(), 10000, 1)  # Int32 by default
topk_radix_select!(result, idx_out, ws, data, idx_in, [10000], 100)  # k auto-converted to Int32

# Single task with workspace reuse (avoid allocation overhead)
ws = RadiKWorkspace(CUDABackend(), 10000, 1)
result = CUDA.zeros(Float32, 100)
idx_in = CUDA.zeros(Int32, 10000)
idx_out = CUDA.zeros(Int32, 100)

for i in 1:100
    data = CUDA.randn(Float32, 10000)
    topk_radix_select!(result, idx_out, ws, data, idx_in, [10000], 100)
end

# Multi-task processing with workspace reuse
ws = RadiKWorkspace(CUDABackend(), 10000, 4)  # max stride=10000, 4 tasks
data = CUDA.randn(Float32, 10000)
result = CUDA.zeros(Float32, 100, 4)
idx_in = CUDA.collect(Int32(1):Int32(10000))
idx_out = CUDA.zeros(Int32, 100, 4)
topk_radix_select!(result, idx_out, ws, data, idx_in, [5000, 6000, 4000, 5500], 100)
```

# Type Conversion
- `k` and `task_lens` are automatically converted to the workspace's integer type
- Default workspace uses Int32 for optimal GPU performance and memory efficiency
- For Int64 support (large datasets >2B elements), use: `RadiKWorkspace(backend, n, num_tasks, Int64)`

# Workspace Reuse
When reusing a workspace across calls:
- Workspace `num_tasks` must match the number of tasks in each call
- Workspace stride `n` must be ≥ `max(task_lens)` for each call
- Different `k` values are allowed across calls
"""
function topk_radix_select!(
    val_out::AbstractArray{Float32},
    idx_out::AbstractArray{IdxT},
    ws::RadiKWorkspace{I},
    val_in::AbstractArray{Float32},
    idx_in::AbstractArray{IdxT},
    task_lens::AbstractVector{<:Integer},
    k::Integer,
    ::Val{LARGEST}=Val(true),
    ::Val{ASCEND}=Val(true),
    ::Val{WITHSCALE}=Val(false),
    ::Val{WITHIDXIN}=Val(false),
    ::Val{WITHPACKING}=Val(true)
) where {IdxT, I, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, WITHPACKING}

    backend = get_backend(val_in)
    WARP_SIZE = get_warpsize(device(backend))

    if WITHPACKING
        pack_size = 16 ÷ sizeof(Float32)  # = 4 for Float32
    else
        pack_size = 1
    end

    # ========================================================================
    # Buffer reset
    # ========================================================================
    ws.k_values .= I(k)

    # ========================================================================
    # Workspace setup
    # ========================================================================
    histogram = ws.histogram
    val_buffer_1 = ws.val_buffer_1
    val_buffer_2 = ws.val_buffer_2
    global_count = ws.global_count
    task_len_buffer_1 = ws.task_len_buffer_1
    task_len_buffer_2 = ws.task_len_buffer_2
    k_values = ws.k_values
    bin_ids = ws.bin_ids

    num_tasks = length(task_lens)
    stride = I(max(task_lens..., num_tasks))
    task_offsets = adapt(backend, I.(vcat(0, accumulate(+, task_lens))))

    # Ping-pong flag: tracks which buffer was last written to (1 = val_buffer_2, 0 = val_buffer_1)
    flag = 1

    # ========================================================================
    # Pass 1: 12-bit radix [31:20]
    # ========================================================================
    histogram .= 0
    global_count .= 0

    min_task_len = minimum(task_lens)
    grid_size_x = min(MAX_GRID_SIZE, max(1, cld(min_task_len, 1024 * pack_size)))
    grid_size_y = min(MAX_GRID_SIZE ÷ grid_size_x, num_tasks)

    count_bin_ex_kernel!(backend, 1024)(
        histogram, val_in, task_offsets,
        Val(0), Val(20), Val(4096), Val(pack_size), Val(WITHSCALE), Val(LARGEST);
        ndrange=(grid_size_x * 1024, grid_size_y))

    select_bin_kernel!(backend, 512)(
        bin_ids, k_values, task_len_buffer_2, histogram,
        Val(LARGEST), Val(512), Val(4096), Val(8), Val(WARP_SIZE);
        ndrange=num_tasks * 512)

    grid_size_x = min(MAX_GRID_SIZE, max(1, cld(min_task_len, 256 * pack_size)))
    grid_size_y = min(MAX_GRID_SIZE ÷ grid_size_x, num_tasks)

    select_candidate_ex_kernel!(backend, 256)(
        val_buffer_2, global_count, val_in, bin_ids, task_offsets,
        Val(0), Val(20), Val(256), Val(pack_size), Val(WITHSCALE), Val(LARGEST);
        ndrange=(grid_size_x * 256, grid_size_y))

    # ========================================================================
    # Pass 2: 12-bit radix [19:8]
    # ========================================================================
    synchronize(backend)
    # Read back updated task_len_buffer_2 (modified by select_bin! in Pass 1)
    max_task_len = maximum(task_len_buffer_2)
    pass_2 = false

    if max_task_len > 1
        # clear hist and global_count
        histogram .= 0
        global_count .= 0

        grid_size = cld(max_task_len, 1024)
        count_bin_kernel!(backend, 1024)(
            histogram, val_buffer_2, task_len_buffer_2, stride,
            Val(12), Val(20), Val(4096);
            ndrange=(grid_size * 1024, num_tasks))

        select_bin_kernel!(backend, 512)(
            bin_ids, k_values, task_len_buffer_1, histogram,
            Val(LARGEST), Val(512), Val(4096), Val(8), Val(WARP_SIZE);
            ndrange=num_tasks * 512)

        grid_size = cld(max_task_len, 256)
        select_candidate_kernel!(backend, 256)(
            val_buffer_1, global_count, val_buffer_2, bin_ids, task_len_buffer_2, stride,
            Val(12), Val(20), Val(256),
            ndrange=(grid_size * 256, num_tasks))
        flag = 1 - flag  # Flip flag: now candidates are in val_buffer_1
        pass_2 = true
    end

    # ========================================================================
    # Pass 3: 8-bit radix [7:0]
    # ========================================================================
    synchronize(backend)
    # Read back updated task_len_buffer_1 (modified by select_bin! in Pass 2)
    max_task_len = maximum(task_len_buffer_1)

    if max_task_len > 1 && pass_2
        histogram .= 0
        global_count .= 0

        grid_size = cld(max_task_len, 256)
        count_bin_kernel!(backend, 256)(
            histogram, val_buffer_1, task_len_buffer_1, stride,
            Val(24), Val(24), Val(256);
            ndrange=(grid_size * 256, num_tasks))

        select_bin_kernel!(backend, 256)(
            bin_ids, k_values, task_len_buffer_2, histogram,
            Val(LARGEST), Val(256), Val(256), Val(1), Val(WARP_SIZE);
            ndrange=num_tasks * 256)

        select_candidate_kernel!(backend, 256)(
            val_buffer_2, global_count, val_buffer_1, bin_ids, task_len_buffer_1, stride,
            Val(24), Val(24), Val(256),
            ndrange=(grid_size * 256, num_tasks))
        flag = 1 - flag  # Flip flag: now candidates are in val_buffer_2
    end

    # ========================================================================
    # Final filter (C++ lines 232-260)
    # ========================================================================
    global_count .= 0

    # Extract k-th element from val_buffer[flag] (whichever buffer was last written to)
    # flag == 1 → val_buffer_2, flag == 0 → val_buffer_1
    kth_element = flag == 1 ? val_buffer_2[1, :] : val_buffer_1[1, :]

    min_task_len = minimum(task_lens)
    grid_size_x = min(MAX_GRID_SIZE, max(1, cld(min_task_len, 256 * pack_size)))
    grid_size_y = min(MAX_GRID_SIZE ÷ grid_size_x, num_tasks)

    if k <= 1024
        if k <= 128
            cache_size = 128
        elseif k <= 256
            cache_size = 256
        elseif k <= 512
            cache_size = 512
        else
            cache_size = 1024
        end
        filter_kernel!(backend, 256)(
            val_out, idx_out, global_count, k_values, val_in, idx_in,
            kth_element, task_offsets, k,
            Val(0), Val(20), Val(256), Val(pack_size), Val(cache_size), Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN),
            ndrange=(grid_size_x * 256, grid_size_y))
    else
        filter_general_kernel!(backend, 256)(
            val_out, idx_out, global_count, k_values, val_in, idx_in,
            kth_element, task_offsets, k,
            Val(0), Val(20), Val(256), Val(pack_size), Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN),
            ndrange=(grid_size_x * 256, grid_size_y))
    end

    # ========================================================================
    # Final sort (C++ lines 267-308)
    # ========================================================================
    bitonic_task_offsets = vcat([0], accumulate(+, fill(k, num_tasks)))

    if k <= 4096
        bitonic_sort!(val_out, idx_out, ascend=ASCEND, task_offsets=bitonic_task_offsets)
    else
        for task_id in 1:num_tasks
            AcceleratedKernels.merge_sort_by_key!(
                view(val_out, :, task_id),
                view(idx_out, :, task_id);
                rev=!ASCEND
            )
        end
    end
    synchronize(backend)

    # ========================================================================
    # Post-processing
    # ========================================================================
    # set idle positions to default values (index: -1, value: 0)
    # val_out and idx_out are 2D arrays with shape (k, num_tasks)
    for task_id in 1:num_tasks
        if task_lens[task_id] < k
            @inbounds val_out[task_lens[task_id]+1:k, task_id] .= 0
            @inbounds idx_out[task_lens[task_id]+1:k, task_id] .= -1
        end
    end

    return val_out, idx_out
end

"""
    topk(data, k; indices=nothing, largest=true, rev=false)
    topk(data, task_lens, k; indices=nothing, largest=true, rev=false)
    topk!(val_out, idx_out, data, k; indices=nothing, largest=true, rev=false)
    topk!(val_out, idx_out, data, task_lens, k; indices=nothing, largest=true, rev=false)

Top-k selection on GPU arrays.

Two variants:
- **`topk`**: Non-mutating, allocates output arrays automatically
- **`topk!`**: Mutating, writes to pre-allocated output arrays

Two modes:
- **Single-task**: Process one array
- **Batch**: Process multiple tasks (provide `task_lens` vector)

# Arguments
- `data`: Input data array (Float32, GPU-backed)
- `k`: Number of top elements to select
- `val_out`: (For `topk!`) Pre-allocated output array for values (mutated)
- `idx_out`: (For `topk!`) Pre-allocated output array for indices (mutated)
- `task_lens`: (Optional) For batch mode, number of elements per task [num_tasks] (use `[n]` for single task)
- `indices`: (Optional) Custom input indices array (default: automatic sequential indices)
- `largest`: Find largest (true) or smallest (false) elements (default: true)
- `rev`: Sort output in descending (true) or ascending (false) order (default: false)

# Returns
- **`topk`**: Tuple `(values, indices)` - allocated arrays
  - Single-task: shape `(k,)`
  - Batch: shape `(k, num_tasks)`
- **`topk!`**: Nothing (mutates `val_out` and `idx_out`)

# Examples

## Basic usage - find largest elements
```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()
data = adapt(backend, randn(Float32, 1_000_000))
values, indices = topk(data, 100; largest=true)
```

## Find smallest elements
```julia
values, indices = topk(data, 100; largest=false)
```

## Control sort order
```julia
# Largest values, sorted ascending (1st value is smallest of top-k)
values, indices = topk(data, 100; largest=true, rev=false)

# Smallest values, sorted descending (1st value is largest of bottom-k)
values, indices = topk(data, 100; largest=false, rev=true)
```

## Custom indices - track original positions
```julia
# Your custom indexing (e.g., global dataset indices)
data = adapt(backend, Float32[1000, 1001, 1002, 1003, 1004])
indices_array = adapt(backend, Int32[1000, 1001, 1002, 1003, 1004])
values, idxs = topk(data, 2; indices=indices_array)
```

## Single task using batch API explicitly
```julia
# Explicit single-task with batch function
values, indices = topk(data, [length(data)], 100)
```

## Batch processing - multiple datasets in one call
```julia
using RadiK, CUDA, Adapt

backend = CUDABackend()
data = adapt(backend, randn(Float32, 10_000))
task_lens = [3000, 4000, 3000]
values, indices = topk(data, task_lens, 100)
# Returns: (values::Matrix{Float32}, indices::Matrix{Int32}), both (100, 3)
```

## Mutating (pre-allocated outputs)
```julia
using RadiK, CUDA, KernelAbstractions as KA, Adapt

backend = CUDABackend()
val_out = KA.zeros(backend, Float32, 100, 1000)
idx_out = KA.zeros(backend, Int32, 100, 1000)
data = adapt(backend, randn(Float32, 1_000_000))
task_lens = repeat([1000], 1000)
topk!(val_out, idx_out, data, task_lens, 100)
```

# Type Constraints
- `data` must be `Float32` (algorithm is optimized for IEEE 754 32-bit floating-point)
- `data` must be on a GPU backend (CUDA, ROCm, Metal, etc.)

# Performance Notes
- **`topk`**: Allocates all GPU buffers on each call (simpler, slower for repeated calls)
- **`topk!`**: Reuses pre-allocated outputs
- **`topk_radix_select!`**: Maximum control, reuses workspace and outputs

# See Also
- `topk_radix_select!`: Lower-level API with workspace reuse
- `RadiKWorkspace`: Pre-allocated workspace for repeated operations
"""
function topk!(
    val_out::AbstractArray{Float32, 2},
    idx_out::AbstractArray{IdxT, 2},
    data::AbstractArray{Float32},
    task_lens::AbstractVector{<:Integer},
    k::Integer;
    indices::Union{Nothing, AbstractArray{IdxT}}=nothing,
    largest=true,
    rev=false
) where IdxT
    backend = get_backend(data)
    num_tasks = length(task_lens)
    max_task_len = maximum(task_lens)

    # Validate input data type
    eltype(data) == Float32 || error(
        "RadiK.jl only supports Float32 input data. " *
        "Got $(eltype(data)). Convert with: adapt(backend, Float32.(your_data))")

    # Validate output dimensions
    expected_size = (k, num_tasks)
    size(val_out) == expected_size || error(
        "val_out must have size $expected_size, got $(size(val_out))")
    size(idx_out) == expected_size || error(
        "idx_out must have size $expected_size, got $(size(idx_out))")

    ws = RadiKWorkspace(backend, Int(max_task_len), num_tasks)

    # Validate indices if provided
    if indices !== nothing && !isempty(indices)
        length(indices) == length(data) || error(
            "indices length must equal data length. " *
            "Got $(length(indices)) vs $(length(data))")
    end

    # Handle input indices
    if indices === nothing
        indices = KA.zeros(backend, Int32, 0)
    end

    # Call main algorithm
    topk_radix_select!(
        val_out, idx_out, ws, data, indices, task_lens, k,
        Val(largest), Val(!rev), Val(false), Val(indices !== nothing && !isempty(indices)), Val(true)
    )
end

function topk!(val_out, idx_out, data, k; indices=nothing, largest=true, rev=false)
    topk!(val_out, idx_out, data, [length(data)], k, indices=indices, largest=largest, rev=rev)
end

function topk(
    data::AbstractArray{Float32},
    task_lens::AbstractVector{<:Integer},
    k::Integer;
    indices::Union{Nothing, AbstractArray}=nothing,
    largest=true,
    rev=false
)

    # Validate input data type
    eltype(data) == Float32 || error(
        "RadiK.jl only supports Float32 input data. " *
        "Got $(eltype(data)). Convert with: adapt(backend, Float32.(your_data))")

    # Allocate output (2D: k x num_tasks)
    backend = get_backend(data)
    num_tasks = length(task_lens)
    val_out = KA.zeros(backend, Float32, k, num_tasks)
    
    # Determine index type from indices array, or default to Int32
    idx_type = indices === nothing ? Int32 : eltype(indices)
    idx_out = KA.zeros(backend, idx_type, k, num_tasks)

    topk!(val_out, idx_out, data, task_lens, k, indices=indices, largest=largest, rev=rev)

    return val_out, idx_out
end

function topk(data, k; indices=nothing, largest=true, rev=false)
    val_out, idx_out = topk(data, [length(data)], k; indices=indices, largest=largest, rev=rev)
    return val_out[:, 1], idx_out[:, 1]
end
