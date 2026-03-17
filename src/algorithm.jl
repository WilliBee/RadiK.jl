using Adapt
using AcceleratedKernels
using BitonicSort
using KernelAbstractions: get_backend, synchronize
import KernelAbstractions as KA
using KernelIntrinsics: get_warpsize, device

const MAX_GRID_SIZE = 65535

"""
    RadiKWorkspace(backend, k, task_num, task_lens, ValT=Float32)

Temporary workspace buffers for radix-based top-k selection.

Allows memory reuse across multiple calls by allocating once and passing to
`topk_radix_select!` via the `workspace` kwarg.

# Arguments
- `backend`: KernelAbstractions backend (CUDABackend, MetalBackend, etc.)
- `k`: Number of top elements to select
- `task_num`: Number of tasks (1 for single-task, >1 for multi-task batch)
- `task_lens`: Task lengths for each task (Vector{<:Integer} of length task_num)
- `ValT`: (Optional) Value type for buffers (default: Float32)

# Fields
- `histogram`: Histogram buffer (4096 bins × num_tasks)
- `val_buffer_1`: First ping-pong value buffer (num_tasks × stride)
- `val_buffer_2`: Second ping-pong value buffer (num_tasks × stride)
- `global_count`: Per-task global counters
- `task_offsets`: Task boundary offsets for prefix sum
- `task_len_buffer_1`: First ping-pong task lengths buffer
- `task_len_buffer_2`: Second ping-pong task lengths buffer
- `k_values`: K values per task
- `bin_ids`: Selected bin IDs per task

# Example
```
# Single-task with Float32
ws = RadiKWorkspace(CUDABackend(), 100, 1, [10000])

# Single-task with Float16
ws = RadiKWorkspace(CUDABackend(), 100, 1, [10000], Float16)

# Multi-task (4 tasks)
task_lens = [2500, 3000, 2000, 2500]
ws = RadiKWorkspace(CUDABackend(), 100, 4, task_lens)

# Reuse for multiple calls
topk_radix_select!(data1, out1, 100, ws)
topk_radix_select!(data2, out2, 100, ws)
```
"""
mutable struct RadiKWorkspace{I, V}
    histogram::AbstractArray{I, 2}
    val_buffer_1::AbstractArray{V, 2}
    val_buffer_2::AbstractArray{V, 2}
    global_count::AbstractArray{I}
    task_offsets::AbstractArray{I}
    task_len_buffer_1::AbstractArray{I}
    task_len_buffer_2::AbstractArray{I}
    k_values::AbstractArray{I}
    bin_ids::AbstractArray{I}
    num_tasks::I
    stride::I
end

function RadiKWorkspace(backend, k::T, num_tasks::T, task_lens::AbstractVector{T}, ::Type{ValT}=Float32) where {T, ValT}
    stride = max(task_lens..., length(task_lens))

    histogram = KA.zeros(backend, T, 4096, Int(num_tasks))
    val_buffer_1 = KA.zeros(backend, ValT, stride, Int(num_tasks))
    val_buffer_2 = KA.zeros(backend, ValT, stride, Int(num_tasks))
    global_count = KA.zeros(backend, T, Int(num_tasks))

    task_offsets = adapt(backend, vcat(zero(T), accumulate(+, task_lens)))
    task_len_buffer_1 = adapt(backend, task_lens)
    task_len_buffer_2 = KA.zeros(backend, T, Int(num_tasks))
    k_values = adapt(backend, fill(k, Int(num_tasks)))
    bin_ids = KA.zeros(backend, T, Int(num_tasks))

    return RadiKWorkspace(
        histogram, val_buffer_1, val_buffer_2, global_count, task_offsets,
        task_len_buffer_1, task_len_buffer_2, k_values, bin_ids, num_tasks, T(stride)
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
- `k`: Number of top elements to select
- `idx_in`: (Optional) Input indices array
- `idx_out`: (Optional) Output array for indices
- `task_lens`: (Optional) Task lengths for multi-task batch processing
- `workspace`: (Optional) Pre-allocated RadiKWorkspace for memory reuse

# Returns
- `val_out`: Array containing top-k elements (sorted)
- `idx_out`: Array containing indices (if provided), otherwise nothing

# Examples
```
# Single call (workspace allocated internally)
data = CUDA.randn(Float32, 10000)
result = CUDA.zeros(Float32, 100)
topk_radix_select!(data, result, 100; largest=true)

# Multiple calls with workspace reuse (avoid allocation overhead)
ws = RadiKWorkspace(CUDABackend(), 10000, 100)
for i in 1:100
    data = CUDA.randn(Float32, 10000)
    result = CUDA.zeros(Float32, 100)
    topk_radix_select!(data, result, 100; workspace=ws)
end
```
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
    # Workspace setup
    # ========================================================================
    histogram = ws.histogram
    val_buffer_1 = ws.val_buffer_1
    val_buffer_2 = ws.val_buffer_2
    global_count = ws.global_count
    task_offsets = ws.task_offsets
    task_len_buffer_1 = ws.task_len_buffer_1
    task_len_buffer_2 = ws.task_len_buffer_2
    k_values = ws.k_values
    bin_ids = ws.bin_ids
    stride = ws.stride

    num_tasks = length(task_lens)

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
    topk(data::AbstractArray{ValT}, k::Int; ::Val{LARGEST}=Val(true))

Convenience wrapper that allocates output array, indices, and workspace.

Allocates the result array, indices array, and workspace, then calls `topk_radix_select!`.

# Returns
- `result`: Array containing top-k elements (sorted)
- `indices`: Array containing indices of top-k elements
"""
function topk(data::AbstractArray{ValT}, k::Int32; largest=true, rev=false) where {ValT}
    backend = get_backend(data)
    n = length(data)

    # Allocate workspace
    ws = RadiKWorkspace(backend, Int32(k), Int32(1), Int32[n], ValT)

    # Allocate output
    result = KA.zeros(backend, ValT, Int(k))
    indices = KA.zeros(backend, Int32, Int(k))

    # Call main algorithm
    topk_radix_select!(data, result, k, ws, indices, indices, Int32[n], Val(largest), Val(!rev), Val(false), Val(false), Val(true))
    return result, indices
end
