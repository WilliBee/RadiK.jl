# ==============================================================================
# select_candidate: Filter Elements by Bin ID
# ==============================================================================
# Based on radik/radik/RadixSelect/radixselect_l.cuh:128-168 (selectCandidate)
#                   and radik/radik/RadixSelect/radixselect_l.cuh:269-408 (selectCandidateEx)

using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @inbounds
import KernelAbstractions as KA
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics: vload
using Atomix

"""
    select_candidate_kernel!(data_in, data_out, global_counts, bin_ids, task_lens, stride, ...)

GPU kernel that filters elements belonging to selected histogram bins.

After `select_bin!` identifies which bin contains the k-th element, this kernel extracts
all elements from that bin to progressively narrow the search space.

# Algorithm
1. **Block-level caching**: Use shared memory to cache filtered elements
2. **Filtering**: Each thread reads elements, computes bin ID, checks for match
3. **Counting**: Matching elements atomically added to block-level counter
4. **Global offset**: Atomically reserve space in global output array
5. **Write back**: Collaboratively write cached elements to reserved space

# Arguments
- `data_in`: Input data array [task_id * stride + idx]
- `data_out`: Output data array for filtered elements
- `global_counts`: Global count array [num_tasks] tracking elements per task
- `bin_ids`: Selected bin ID for each task [num_tasks]
- `task_lens`: Number of elements per task [num_tasks]
- `stride`: Stride/padding between task data in arrays
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters for bin calculation
- `Val{BLOCK}`: Threads per block

# Example

Filter elements from bin 5 (using LEFT=0, RIGHT=4):

**Input:**
```
data_in = [100, 85, 92, 78, 95, 88, 90, 82]
bin_id  = 5
```

**Bin contents:**
```
Bin 0: [78, 82]       Bin 3: [92]
Bin 1: []             Bin 4: []
Bin 2: [85, 88]       Bin 5: [90, 95, 100]  ← TARGET
```

**Execution (BLOCK=4 threads):**
```
Thread 0: 100 → bin 5 → MATCH → cache[0]=100
Thread 1:  85 → bin 2 → skip
Thread 2:  92 → bin 3 → skip
Thread 3:  78 → bin 0 → skip
Thread 4:  95 → bin 5 → MATCH → cache[1]=95
Thread 5:  88 → bin 2 → skip
Thread 6:  90 → bin 5 → MATCH → cache[2]=90
Thread 7:  82 → bin 0 → skip
```

**Output:**
```
data_out     = [100, 95, 90, 0, ...]
global_count = [3]  # Found 3 elements in bin 5
```
"""
@kernel function select_candidate_kernel!(
    data_in::AbstractArray{T},
    data_out::AbstractArray{T},
    global_counts::AbstractArray{Int32},
    bin_ids::AbstractArray{Int32},
    task_lens::AbstractArray{Int32},
    stride::Int32,
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{BLOCK}
) where {T, LEFT, RIGHT, BLOCK}

    # ========================================================================
    # Shared memory: counter and element cache
    # ========================================================================
    block_count = @localmem Int32 (1,)      # Atomic counter for cache
    block_cache = @localmem T (BLOCK,)      # Cache for filtered elements

    # ========================================================================
    # Thread and task identification
    # ========================================================================
    block_x = @index(Group, Cartesian)[1]   # Block ID in x-dimension (1-indexed)
    task_id = @index(Group, Cartesian)[2]   # Task ID (1-indexed)
    thread_x = @index(Local, Linear)        # Thread ID in block (1-indexed)

    task_len = @inbounds task_lens[task_id]
    mask = @inbounds bin_ids[task_id]
    idx = (block_x - 1) * BLOCK + thread_x  # Global thread index (1-indexed)

    # ========================================================================
    # Step 1: Initialize block counter (first thread in block)
    # ========================================================================
    if idx <= task_len && thread_x == 1
        @inbounds block_count[1] = 0
    end
    @synchronize()

    # ========================================================================
    # Step 2: Filter and cache elements
    # ========================================================================
    if idx <= task_len
        val = @inbounds data_in[(task_id - 1) * stride + idx]
        if mask == get_bin_id(val, Val(LEFT), Val(RIGHT)) + Int32(1)
            pos = Atomix.@atomic block_count[1] += 1
            @inbounds block_cache[pos] = val
        end
    end
    @synchronize()

    # ========================================================================
    # Step 3: Reserve space in global output
    # ========================================================================
    count = @inbounds block_count[1]
    @synchronize()

    if idx <= task_len && thread_x == 1
        # Atomix returns NEW value (after addition), unlike CUDA's atomicAdd
        # We need the OLD value as the starting offset for writing
        block_count[1] = @inbounds Atomix.@atomic global_counts[task_id] += count
        @inbounds block_count[1] -= count
    end
    @synchronize()

    # ========================================================================
    # Step 4: Write cached elements to global memory
    # ========================================================================
    if idx <= task_len && thread_x <= count
        @inbounds data_out[(task_id - 1) * stride + block_count[1] + thread_x] =
            @inbounds block_cache[thread_x]
    end
end

# ========================================================================
# Helper functions for select_candidate_ex_kernel!
# ========================================================================
struct SelectCandidateExContext{GC, BCo, BCa, DO, LEFT, RIGHT, BLOCK}
    global_counts::GC
    block_count::BCo
    block_cache::BCa
    data_out::DO
    thread_x::Int
end

@inline function reset_block_count_sync!(ctx)
    if ctx.thread_x == 1
        @inbounds ctx.block_count[1] = 0
    end
    @synchronize()
end

# Check if value matches bin mask, stage to cache if yes
@inline function stage_if_hit!(
    ctx::SelectCandidateExContext{GC, BCo, BCa, DO, LEFT, RIGHT, BLOCK}, 
    val, mask
    ) where {GC, BCo, BCa, DO, LEFT, RIGHT, BLOCK}
    bin = get_bin_id(val, Val(LEFT), Val(RIGHT)) + Int32(1)
    if mask == bin
        @inbounds pos = Atomix.@atomic ctx.block_count[1] += 1
        @inbounds ctx.block_cache[pos] = val
    end
end

# Flush cached elements to global memory
# WARNING: after flushing, threads may be unsynchronized
@inline function flush_cache!(
    ctx::SelectCandidateExContext{GC, BCo, BCa, DO, LEFT, RIGHT, BLOCK}, 
    task_id, count
    ) where {GC, BCo, BCa, DO, LEFT, RIGHT, BLOCK}

    if ctx.thread_x == 1
        # Atomix returns NEW value (after addition), unlike CUDA's atomicAdd
        # We need the OLD value as the starting offset for writing
        @inbounds new_count = Atomix.@atomic ctx.global_counts[task_id] += count
        @inbounds ctx.block_count[1] = new_count - count
    end
    @synchronize()

    @inbounds offset = ctx.block_count[1]
    @synchronize()

    pos = ctx.thread_x
    while pos <= count
        @inbounds ctx.data_out[offset + pos, task_id] = ctx.block_cache[pos]
        pos += BLOCK
    end
end

"""
    select_candidate_ex_kernel!(data_in, data_out, global_counts, bin_ids, task_offsets, ...)

Enhanced GPU kernel with vectorization and scaling support for filtering by bin ID.

Same as `select_candidate_kernel!` but with three key optimizations:

**1. Vectorization**: Reads PACKSIZE elements (e.g., 4) per memory transaction instead of 1,
reducing memory bandwidth pressure.

**2. Scaling**: When data has clustered values (e.g., all between 1.001 and 1.002), histogram
bins may collide. Scaling subtracts a sample value to spread out the distribution.

**3. Cache Flushing**: For large datasets, periodically writes cached elements to global memory
to prevent shared memory overflow.

# Algorithm
1. Vectorized processing: load PACKSIZE elements per iteration
2. Cache management: periodically flush cache when capacity exceeded
3. Scaling support: apply adaptive scaling for adversarial distributions
4. Padding handling: handle misaligned starting addresses

# Arguments
- `data_in`: Input data array (packed vectorized reads)
- `data_out`: Output data array for filtered elements
- `global_counts`: Global count array [task_num] tracking elements per task
- `bin_ids`: Selected bin ID for each task [task_num]
- `task_offsets`: Prefix sum array defining task boundaries [task_num + 1]
- `stride`: Stride/padding between task data in arrays
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters for bin calculation
- `Val{BLOCK}`: Threads per block
- `Val{PACKSIZE}`: Elements per vectorized load
- `Val{WITHSCALE}`: Whether to apply adaptive scaling
- `Val{LARGEST}`: NaN handling (true → min, false → max)

# When to Use
- **Use `select_candidate!` (basic)** for learning/debugging, well-distributed data
- **Use `select_candidate_ex!` (enhanced)** for maximum performance on large datasets,
  clustered/continuous values, many concurrent tasks, or memory bandwidth bottlenecks
"""
@kernel function select_candidate_ex_kernel!(
    data_in::AbstractArray{T},
    data_out::AbstractArray{T},
    global_counts::AbstractArray{Int32},
    bin_ids::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32},
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{BLOCK},
    ::Val{PACKSIZE},
    ::Val{WITHSCALE},
    ::Val{LARGEST}
) where {T, LEFT, RIGHT, BLOCK, PACKSIZE, WITHSCALE, LARGEST}

    block_x = @index(Group, Cartesian)[1]   # Block ID in x-dimension (1-indexed)
    block_y = @index(Group, Cartesian)[2]   # Task group ID (1-indexed)
    thread_x = @index(Local, Linear)        # Thread ID in block (1-indexed)

    ndrange_x = @ndrange()[1]
    grid_dim_y = @ndrange()[2] ÷ @groupsize()[2]

    # ========================================================================
    # Shared memory: counter and element cache (doubled for safety)
    # ========================================================================
    block_count = @localmem Int32 (1,)                    # Atomic counter for cache
    block_cache = @localmem T (2 * BLOCK * PACKSIZE,)     # Cache for filtered elements

    # ========================================================================
    # Thread and task identification
    # ========================================================================
    tid = (block_x - 1) * BLOCK + thread_x  # Linear thread ID across x-grid
    task_num = length(task_offsets) - 1
    step_size = ndrange_x * PACKSIZE

    # ========================================================================
    # Thread context
    # ========================================================================
    ctx = SelectCandidateExContext{
        typeof(global_counts), typeof(block_count), typeof(block_cache), typeof(data_out), 
        LEFT, RIGHT, BLOCK
    }(
        global_counts, block_count, block_cache, data_out, thread_x
    )

    # ========================================================================
    # Process tasks in strided fashion (block_y acts as task group ID)
    # ========================================================================
    for task_id in block_y:grid_dim_y:task_num

        # Reset block counter for each task
        reset_block_count_sync!(ctx)

        # Get task boundaries
        offset = task_offsets[task_id]
        pad = offset % PACKSIZE
        offset -= pad

        # Get scaling factor and bin mask
        scaler = sample_scaler(data_in, offset + pad + 1, Val(WITHSCALE))
        mask = bin_ids[task_id]

        task_len = task_offsets[task_id + 1] - offset
        nb_steps = task_len ÷ step_size

        # ====================================================================
        # Vectorized section: Process PACKSIZE elements per iteration
        # ====================================================================
        if nb_steps > 0
            idx = tid + offset ÷ PACKSIZE

            # First iteration: handle misaligned start (padding)
            vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
            if tid == 1
                @unroll for (j, v) in enumerate(vals)
                    if j > pad
                        scaled_val = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                        stage_if_hit!(ctx, scaled_val, mask)
                    end
                end
            else
                @unroll for v in vals
                    scaled_val = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                    stage_if_hit!(ctx, scaled_val, mask)
                end
            end
            idx += ndrange_x

            # Main loop: full vectorized loads
            for _ in 2:nb_steps
                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                @unroll for v in vals
                    scaled_val = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                    stage_if_hit!(ctx, scaled_val, mask)
                end
                idx += ndrange_x

                # Check cache capacity, flush if necessary
                @synchronize()
                @inbounds count = block_count[1]
                @synchronize()

                if count > BLOCK * PACKSIZE
                    flush_cache!(ctx, task_id, count)
                    reset_block_count_sync!(ctx)
                end
            end

            # tail : skip elements already processed in vectorized loops
            i = offset + tid + nb_steps * step_size
            while i <= offset + task_len
                @inbounds v = data_in[i]
                scaled_val = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                stage_if_hit!(ctx, scaled_val, mask)
                i += ndrange_x
            end

            # Flush cache
            @synchronize()
            count = @inbounds block_count[1]
            @synchronize()
            flush_cache!(ctx, task_id, count)
            # no need to reset & sync, because this is done at the beginning of
            # each pass
        else
            # No vectorized iterations: skip padding in tail section
            i = offset + tid + pad
            while i <= offset + task_len
                @inbounds v = data_in[i]
                scaled_val = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                stage_if_hit!(ctx, scaled_val, mask)
                i += ndrange_x
            end

            # Flush cache
            @synchronize()
            @inbounds count = block_count[1]
            @synchronize()
            flush_cache!(ctx, task_id, count)
            # no need to reset & sync, because this is done at the beginning of
            # each pass
        end
    end
end


# ==============================================================================
# Convenience wrappers
# ==============================================================================

"""
    select_candidate!(data_in, data_out, global_counts, bin_ids, task_lens, stride;
                     LEFT=0, RIGHT=20, threads_per_block=256)

Filter elements belonging to selected histogram bins for each task.

After `select_bin!` identifies which bin contains the k-th element, this function
filters all input elements to extract only those in the selected bin, narrowing the
search space for the next radix select iteration.

# Arguments
- `data_in`: Input data array (layout: [task_id * stride + idx])
- `data_out`: Output data array for filtered elements
- `global_counts`: Count array [num_tasks], updated with filtered element counts
- `bin_ids`: Selected bin ID for each task [num_tasks] (0-indexed)
- `task_lens`: Number of elements per task [num_tasks]
- `stride`: Stride between task data in arrays
- `LEFT`: Left bit shift for bin calculation (default 0)
- `RIGHT`: Right bit shift for bin calculation (default 20)
- `threads_per_block`: GPU block size (default 256)

# Constraints
- Number of tasks ≤ available GPU blocks in y-dimension
- Global counts should be initialized to 0 before first call
- Output array must be large enough to hold filtered elements

# Example
```julia
using CUDA, RadiK

# Setup: find elements in bin 5
data_in = CuArray{Float32}([1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0])
data_out = CUDA.zeros(Float32, 5)
global_counts = CUDA.zeros(Int32, 1)
bin_ids = CuArray{Int32}([5])
task_lens = CuArray{Int32}([5])
stride = Int32(5)

select_candidate!(data_in, data_out, global_counts, bin_ids, task_lens, stride)
println("Filtered count: ", Array(global_counts)[1])
```

# See also
- [`count_bin!`](@ref): Build histogram from data
- [`select_bin!`](@ref): Find bin containing k-th element
"""
function select_candidate!(
    data_in::AbstractArray{T},
    data_out::AbstractArray{T},
    global_counts::AbstractArray{Int32},
    bin_ids::AbstractArray{Int32},
    task_lens::AbstractArray{Int32},
    stride::Int32;
    LEFT::Int = 0,
    RIGHT::Int = 20,
    threads_per_block = 256
) where T
    backend = KA.get_backend(data_in)
    num_tasks = length(task_lens)
    max_task_len = maximum(Array(task_lens))
    num_blocks = cld(max_task_len, threads_per_block)

    data_out .= 0
    global_counts .= 0

    select_candidate_kernel!(backend, threads_per_block)(
        data_in, data_out, global_counts, bin_ids, task_lens, stride,
        Val(LEFT), Val(RIGHT), Val(threads_per_block);
        ndrange=(num_blocks * threads_per_block, num_tasks)
    )
    KA.synchronize(backend)

    return data_out, global_counts
end


"""
    select_candidate_ex!(data_in, data_out, global_counts, bin_ids, task_offsets, stride;
                        LEFT=0, RIGHT=20, threads_per_block=256, blocks_x=16,
                        pack_size=4, with_scale=true, largest=true)

Enhanced version with vectorization and scaling support for filtering by bin ID.

# Arguments
- `data_in`: Input data array (any Float type)
- `data_out`: Output data array for filtered elements
- `global_counts`: Count array [num_tasks], updated with filtered element counts
- `bin_ids`: Selected bin ID for each task [num_tasks] (0-indexed)
- `task_offsets`: Prefix sum array defining task boundaries [num_tasks + 1]
- `stride`: Stride between task data in arrays
- `LEFT`: Left bit shift for bin calculation (default 0)
- `RIGHT`: Right bit shift for bin calculation (default 20)
- `threads_per_block`: GPU block size (default 256)
- `blocks_x`: Number of blocks in x-dimension (default 16)
- `pack_size`: Elements per vectorized load (default 4)
- `with_scale`: Enable adaptive scaling (default true)
- `largest`: Finding largest or smallest elements (default true)

# Constraints
- Number of tasks ≤ available GPU blocks in y-dimension
- Global counts should be initialized to 0 before first call
- `pack_size` must be compatible with data type

# Example
```julia
using CUDA, RadiK

# Task offsets: 3 tasks with varying sizes
task_offsets = CuArray{Int32}([0, 100, 250, 400])
data_in = rand(Float32, 400)
data_out = CUDA.zeros(Float32, 400)
global_counts = CUDA.zeros(Int32, 3)
bin_ids = CuArray{Int32}([42, 17, 99])

select_candidate_ex!(data_in, data_out, global_counts, bin_ids, task_offsets, Int32(400))

println("Task 1 filtered: ", Array(global_counts)[1])
println("Task 2 filtered: ", Array(global_counts)[2])
println("Task 3 filtered: ", Array(global_counts)[3])
```

# See also
- [`select_candidate!`](@ref): Basic version without vectorization
- [`count_bin_ex!`](@ref): Build histogram with vectorization
"""
function select_candidate_ex!(
    data_in::AbstractArray{T},
    data_out::AbstractArray{T},
    global_counts::AbstractArray{Int32},
    bin_ids::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32};
    LEFT::Int = 0,
    RIGHT::Int = 20,
    threads_per_block = 256,
    blocks_x = 16,
    pack_size = 4,
    with_scale::Bool = true,
    largest::Bool = true
) where {T}
    backend = get_backend(data_in)

    num_tasks = length(task_offsets) - 1

    data_out .= 0
    global_counts .= 0

    kernel! = select_candidate_ex_kernel!(backend, threads_per_block)

    kernel!(
        data_in, data_out, global_counts, bin_ids, task_offsets,
        Val(LEFT), Val(RIGHT), Val(threads_per_block), Val(pack_size),
        Val(with_scale), Val(largest);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )

    KA.synchronize(backend)
end
