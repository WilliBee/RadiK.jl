using Adapt
using AcceleratedKernels
using BitonicSort
using KernelAbstractions: get_backend, synchronize
import KernelAbstractions as KA
using KernelIntrinsics: get_warpsize, device

const MAX_GRID_SIZE = 65535

"""
    RadiKWorkspace(backend, n, num_tasks, [T=Int], [ValT=Float32])

Temporary workspace buffers for radix-based top-k selection.

Pre-allocates with a fixed stride `n` to allow memory reuse across multiple calls
with varying task lengths. Follows the C++ approach: allocate once with maximum
expected stride, then reuse for any call with `max(task_lens) ≤ n`.

# Arguments
- `backend`: KernelAbstractions backend (CUDABackend, MetalBackend, etc.)
- `n`: Maximum stride (longest task length this workspace can handle)
- `num_tasks`: Number of tasks (1 for single-task, >1 for multi-task batch)
- `T`: (Optional) Index type for buffers (default: Int)
- `ValT`: (Optional) Value type for buffers (default: Float32)

# Fields
- `histogram`: Histogram buffer (4096 bins × num_tasks)
- `val_buffer_1`: First ping-pong value buffer (n × num_tasks)
- `val_buffer_2`: Second ping-pong value buffer (n × num_tasks)
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
struct RadiKWorkspace{I, V}
    histogram::AbstractArray{I, 2}
    val_buffer_1::AbstractArray{V, 2}
    val_buffer_2::AbstractArray{V, 2}
    global_count::AbstractArray{I}
    task_len_buffer_1::AbstractArray{I}
    task_len_buffer_2::AbstractArray{I}
    k_values::AbstractArray{I}
    bin_ids::AbstractArray{I}
end

function RadiKWorkspace(backend, n, num_tasks, ::Type{T}=Int, ::Type{ValT}=Float32) where {T, ValT}
    histogram = KA.zeros(backend, T, 4096, num_tasks)
    val_buffer_1 = KA.zeros(backend, ValT, n, num_tasks)
    val_buffer_2 = KA.zeros(backend, ValT, n, num_tasks)
    global_count = KA.zeros(backend, T, num_tasks)
    task_len_buffer_1 = KA.zeros(backend, T, num_tasks)
    task_len_buffer_2 = KA.zeros(backend, T, num_tasks)
    k_values = KA.zeros(backend, T, num_tasks)
    bin_ids = KA.zeros(backend, T, num_tasks)

    return RadiKWorkspace(
        histogram, val_buffer_1, val_buffer_2, global_count,
        task_len_buffer_1, task_len_buffer_2, k_values, bin_ids
    )
end

"""
    topk_radix_select!(val_in, val_out, k; ...)

Main entry point for radix-based top-k selection.

# Template Parameters (compile-time via Val{})
- `Val{LARGEST}`: Find largest (true) or smallest (false) elements
- `Val{ASCEND}`: Sort output in ascending (true) or descending (false) order
- `Val{WITHSCALE}`: Enable value scaling for numerical stability
- `Val{WITHIDXIN}`: Whether input indices are provided
- `Val{WITHPACKING}`: Whether to use vectorized loads (compile-time)

# Arguments
- `val_in`: Input data array (AbstractArray{Float32})
- `val_out`: Output array for top-k results
- `k`: Number of top elements to select (Int32)
- `ws`: Pre-allocated RadiKWorkspace for memory reuse
- `idx_in`: Input indices array (for index tracking)
- `idx_out`: Output array for indices
- `task_lens`: Task lengths for multi-task batch processing

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
ws = RadiKWorkspace(CUDABackend(), 10000, 1)
topk_radix_select!(data, result, Int32(100), ws, idx_in, idx_out, Int32[10000])

# Single task with workspace reuse (avoid allocation overhead)
ws = RadiKWorkspace(CUDABackend(), 10000, 1)
result = CUDA.zeros(Float32, 100)
idx_in = CUDA.zeros(Int32, 10000)
idx_out = CUDA.zeros(Int32, 100)

for i in 1:100
    data = CUDA.randn(Float32, 10000)
    topk_radix_select!(data, result, Int32(100), ws, idx_in, idx_out, Int32[10000])
end

# Multi-task processing with workspace reuse
ws = RadiKWorkspace(CUDABackend(), 10000, 4)  # max stride=10000, 4 tasks
data = CUDA.randn(Float32, 10000)
result = CUDA.zeros(Float32, 100, 4)
idx_in = CUDA.collect(Int32(1):Int32(10000))
idx_out = CUDA.zeros(Int32, 100, 4)
topk_radix_select!(data, result, Int32(100), ws, idx_in, idx_out, Int32.[5000, 6000, 4000, 5500])
```

# Workspace Reuse
When reusing a workspace across calls:
- Workspace `num_tasks` must match the number of tasks in each call
- Workspace stride `n` must be ≥ `max(task_lens)` for each call
- Different `k` values are allowed across calls
"""
function topk_radix_select!(
    val_in::AbstractArray{ValT},
    val_out::AbstractArray{ValT},
    k::Int32,
    ws::RadiKWorkspace,
    idx_in::AbstractArray{IdxT},
    idx_out::AbstractArray{IdxT},
    task_lens::AbstractVector{T},
    ::Val{LARGEST}=Val(true),
    ::Val{ASCEND}=Val(true),
    ::Val{WITHSCALE}=Val(false),
    ::Val{WITHIDXIN}=Val(false),
    ::Val{WITHPACKING}=Val(true)
) where {ValT, IdxT, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, WITHPACKING, T}

    backend = get_backend(val_in)
    WARP_SIZE = get_warpsize(device(backend))

    if WITHPACKING
        @assert sizeof(ValT) <= 16 "radix topk: requires sizeof(ValT) <= 16 for with_packing=true, got $(sizeof(ValT))"
        pack_size = 16 ÷ sizeof(ValT)
    else
        pack_size = 1
    end

    # ========================================================================
    # Buffer reset
    # ========================================================================
    ws.k_values .= k

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
    stride = T(max(task_lens..., length(task_lens)))
    task_offsets = adapt(backend, vcat(zero(T), accumulate(+, task_lens)))

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
        histogram, bin_ids, k_values, task_len_buffer_2,
        Val(LARGEST), Val(512), Val(4096), Val(8), Val(WARP_SIZE);
        ndrange=num_tasks * 512)

    grid_size_x = min(MAX_GRID_SIZE, max(1, cld(min_task_len, 256 * pack_size)))
    grid_size_y = min(MAX_GRID_SIZE ÷ grid_size_x, num_tasks)

    select_candidate_ex_kernel!(backend, 256)(
        val_in, val_buffer_2, global_count, bin_ids, task_offsets,
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
            histogram, bin_ids, k_values, task_len_buffer_1,
            Val(LARGEST), Val(512), Val(4096), Val(8), Val(WARP_SIZE);
            ndrange=num_tasks * 512)

        grid_size = cld(max_task_len, 256)
        select_candidate_kernel!(backend, 256)(
            val_buffer_2, val_buffer_1, global_count, bin_ids, task_len_buffer_2, stride,
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
            histogram, bin_ids, k_values, task_len_buffer_2,
            Val(LARGEST), Val(256), Val(256), Val(1), Val(WARP_SIZE);
            ndrange=num_tasks * 256)

        select_candidate_kernel!(backend, 256)(
            val_buffer_1, val_buffer_2, global_count, bin_ids, task_len_buffer_1, stride,
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
            val_in, idx_in, kth_element, val_out, idx_out,
            global_count, k_values, task_offsets, stride, k,
            Val(0), Val(20), Val(256), Val(pack_size), Val(cache_size), Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN),
            ndrange=(grid_size_x * 256, grid_size_y))
    else
        filter_general_kernel!(backend, 256)(
            val_in, idx_in, kth_element, val_out, idx_out,
            global_count, k_values, task_offsets, stride, k,
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
    topk(data::AbstractArray{ValT}, k::Int, [::Type{IxT}=Int32]; largest=true, rev=false)

Convenience wrapper that allocates output array, indices, and workspace.

Allocates the result array, indices array, and workspace, then calls `topk_radix_select!`.
Use this for simple single-task top-k operations where you don't need to reuse buffers.

# Arguments
- `data`: Input data array (AbstractArray{Float32})
- `k`: Number of top elements to select
- `IxT`: (Optional) Index type (default: Int32)
- `largest`: Find largest (true) or smallest (false) elements (default: true)
- `rev`: Sort output in descending (true) or ascending (false) order (default: false)

# Returns
- `result`: Array containing top-k elements (sorted)
- `indices`: Array containing indices of top-k elements

# Example
```
using RadiK, CUDA

data = CUDA.randn(Float32, 1_000_000)
values, indices = topk(data, 100; largest=true, rev=false)
```

# Notes
- For repeated calls or multi-task processing, use `topk_radix_select!` directly
  with a pre-allocated `RadiKWorkspace` for better performance
"""
function topk(data::AbstractArray{ValT}, k, ::Type{IxT}=Int32; largest=true, rev=false) where {ValT, IxT}
    backend = get_backend(data)
    n = length(data)

    # Allocate workspace (n is the data length, which is the stride for single-task)
    ws = RadiKWorkspace(backend, n, 1, IxT, ValT)

    # Allocate output
    result = KA.zeros(backend, ValT, k)
    indices = KA.zeros(backend, IxT, k)

    # Call main algorithm
    topk_radix_select!(data, result, IxT(k), ws, indices, indices, IxT[n], Val(largest), Val(!rev), Val(false), Val(false), Val(true))
    return result, indices
end
