using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @inbounds, @ndrange, @groupsize, get_backend
import KernelAbstractions as KA
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics: vload
using Atomix

# ==============================================================================
# Module-level helper functions
# ==============================================================================
struct KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT} end

struct ThreadFilterContext{GC, BouC, BCo, VBCa, IBCa, VO, IDXIn, IDXOut, T, U, V}
    global_counts::GC
    boundary_counts::BouC
    block_count::BCo
    val_block_cache::VBCa
    idx_block_cache::IBCa
    val_out::VO
    idx_in::IDXIn
    idx_out::IDXOut
    thread_x::T
    offset::T
    pad::T
    kth_val::U
    task_id::V
end

# Flush cached elements to global memory
# WARNING: after flushing, threads may be unsynchronized
@inline function flush_cache!(
    ctx::ThreadFilterContext, ::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    task_id
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}

    @synchronize()
    count = ctx.block_count[1]
    @synchronize()

    if ctx.thread_x == 1
        new_count = @inbounds Atomix.@atomic ctx.global_counts[task_id] += count
        @inbounds ctx.block_count[1] = new_count - count
    end
    @synchronize()

    offset = @inbounds ctx.block_count[1]
    @synchronize()

    pos = ctx.thread_x
    while pos <= count
        @inbounds ctx.val_out[offset + pos, task_id] = ctx.val_block_cache[pos]
        @inbounds ctx.idx_out[offset + pos, task_id] = ctx.idx_block_cache[pos]
        pos += BLOCK
    end
end

@inline function get_packed_index(
    ctx::ThreadFilterContext, ::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    idx, j
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}
    
    @inbounds WITHIDXIN ? ctx.idx_in[(idx - 1) * PACKSIZE + j] : 
                          IdxT((idx - 1) * PACKSIZE + j - ctx.pad - ctx.offset)
end

@inline function get_item_index(
    ctx::ThreadFilterContext, ::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    idx
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}
    
    @inbounds WITHIDXIN ? ctx.idx_in[idx] : 
                          IdxT(idx - ctx.pad - ctx.offset)
end

@inline function stage_pair!(
    ctx::ThreadFilterContext, ::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    value, index
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}

    pos = Atomix.@atomic ctx.block_count[1] += 1
    @inbounds ctx.val_block_cache[pos] = ValT(value)
    @inbounds ctx.idx_block_cache[pos] = index
end

# ==============================================================================
# Filter staging functions
# ==============================================================================

"""
Filter a single packed element: compare scaled value, stage original value if it passes.
"""
@inline function filter_stage_packed!(
    ctx::ThreadFilterContext, kctx::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    vals, vals_scaled, idx, j
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}

    if (LARGEST && (vals_scaled[j] > ctx.kth_val)) || (!LARGEST && (vals_scaled[j] < ctx.kth_val))
        stage_pair!(ctx, kctx, vals[j], get_packed_index(ctx, kctx, idx, j))
    end

    if vals_scaled[j] == ctx.kth_val
        if (Atomix.@atomic ctx.boundary_counts[ctx.task_id] -= 1) > -1
            stage_pair!(ctx, kctx, vals[j], get_packed_index(ctx, kctx, idx, j))
        end
    end
end

"""
Filter a single scalar element: compare scaled value, stage original value if it passes.
"""
@inline function filter_stage_item!(
    ctx::ThreadFilterContext, kctx::KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT},
    val, val_scaled, idx
    ) where {LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}

    if (LARGEST && (val_scaled > ctx.kth_val)) || (!LARGEST && (val_scaled < ctx.kth_val))
        stage_pair!(ctx, kctx, val, get_item_index(ctx, kctx, idx))
    end

    if val_scaled == ctx.kth_val
        if (@inbounds Atomix.@atomic ctx.boundary_counts[ctx.task_id] -= 1) > -1
            stage_pair!(ctx, kctx, val, get_item_index(ctx, kctx, idx))
        end
    end
end

"""
    filter_kernel!(data_in, idx_in, kth_element, val_out, idx_out, global_counts,
                   boundary_counts, task_offsets, stride, K, ...)

GPU kernel for final top-K selection using k-th element threshold.

Compares SCALED values against threshold but outputs ORIGINAL (unscaled) values.
Optimizes for small datasets (N ≤ K) by copying all elements.

**⚠️ CRITICAL**: `kth_element` must be computed by previous radix passes such that
≈K elements pass the filter. Output arrays sized for exactly K elements - buffer
overflow will occur if more than K elements qualify.

# Arguments
- `data_in`: Input data array (original values)
- `idx_in`: Input indices array
- `kth_element`: K-th element values (scaled, from previous iterations)
- `val_out`: Output values array [K × num_tasks]
- `idx_out`: Output indices array [K × num_tasks]
- `global_counts`: Global count array tracking elements per task
- `boundary_counts`: Boundary count for elements equal to k-th element
- `task_offsets`: Task offset array defining task boundaries
- `stride`: Stride between task data in arrays
- `K`: Number of top elements to select per task
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters
- `Val{BLOCK}`: Threads per block
- `Val{PACKSIZE}`: Elements per vectorized load
- `Val{CACHESIZE}`: Cache size for filtered elements (K ≤ 1024)
- `Val{WITHSCALE}`: Apply adaptive scaling
- `Val{LARGEST}`: Select largest (true) or smallest (false)
- `Val{WITHIDXIN}`: Whether input indices are provided
"""
@kernel function filter_kernel!(
    data_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    kth_vals::AbstractArray{ValT},
    val_out::AbstractArray{ValT},
    idx_out::AbstractArray{IdxT},
    global_counts::AbstractArray{Int32},
    boundary_counts::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32},
    stride::Int32,
    K::Int32,
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{BLOCK},
    ::Val{PACKSIZE},
    ::Val{CACHESIZE},
    ::Val{WITHSCALE},
    ::Val{LARGEST},
    ::Val{WITHIDXIN}
) where {ValT, IdxT, LEFT, RIGHT, BLOCK, PACKSIZE, CACHESIZE, WITHSCALE, LARGEST, WITHIDXIN}

    block_x = @index(Group, Cartesian)[1]   # Block ID in x-dimension (1-indexed)
    block_y = @index(Group, Cartesian)[2]   # Task group ID (1-indexed)
    thread_x = @index(Local, Linear)        # Thread ID in block (1-indexed)

    ndrange_x = @ndrange()[1]
    grid_dim_y = @ndrange()[2] ÷ @groupsize()[2]

    # ========================================================================
    # Shared memory: counter and element cache
    # ========================================================================
    block_count = @localmem Int32 (1,)                    # Atomic counter for cache
    val_block_cache = @localmem ValT (CACHESIZE,)                  # Cache for original values
    idx_block_cache = @localmem IdxT (CACHESIZE,)               # Cache for indices

    # ========================================================================
    # Kernel Context
    # ========================================================================
    kctx = KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}()

    # ========================================================================
    # Thread and task identification
    # ========================================================================
    tid = (block_x - 1) * BLOCK + thread_x  # Linear thread ID across x-grid
    task_num = length(task_offsets) - 1
    step_size = PACKSIZE * ndrange_x

    # ========================================================================
    # Process each task (using module-level helper functions)
    # ========================================================================
    for task_id in block_y:grid_dim_y:task_num
        # Reset block counter for each task
        if thread_x == 1
            @inbounds block_count[1] = 0
        end
        @synchronize()

        # Get task boundaries
        @inbounds offset = task_offsets[task_id]
        @inbounds original_len = task_offsets[task_id + 1] - offset

        # Handle padding for misaligned addresses
        pad = offset % PACKSIZE
        offset -= pad
        task_len = original_len + pad
        nb_steps = task_len ÷ step_size

        # ====================================================================
        # Capture current thread context
        # ====================================================================
        ctx = ThreadFilterContext{
            typeof(global_counts), 
            typeof(boundary_counts), 
            typeof(block_count), 
            typeof(val_block_cache), 
            typeof(idx_block_cache), 
            typeof(val_out), 
            typeof(idx_in), 
            typeof(idx_out), 
            typeof(thread_x),
            eltype(kth_vals), 
            typeof(task_id)
        }(
            global_counts, 
            boundary_counts, 
            block_count, 
            val_block_cache, 
            idx_block_cache, 
            val_out, 
            idx_in, 
            idx_out, 
            thread_x, 
            offset, 
            pad, 
            kth_vals[task_id], 
            task_id
        )

        # ====================================================================
        # Small dataset optimization: N ≤ K, just copy all
        # ====================================================================
        if original_len <= K
            if nb_steps > 0
                idx = tid + offset ÷ PACKSIZE

                # First iteration: handle misaligned start (padding)
                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                if tid == 1
                    @unroll for (j, v) in enumerate(vals)
                        if j > pad
                            stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                        end
                    end
                else
                    @unroll for (j, v) in enumerate(vals)
                        stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                    end
                end
                idx += ndrange_x

                # Main iterations
                for _ in 2:nb_steps
                    vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                    @unroll for (j, v) in enumerate(vals)
                        stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                    end
                    idx += ndrange_x
                end

                # tail : skip elements already processed in vectorized loops
                i = offset + tid + nb_steps * step_size

                while i <= offset + task_len
                    @inbounds v = data_in[i]
                    stage_pair!(ctx, kctx, v, get_item_index(ctx, kctx, i))
                    i += ndrange_x
                end
            else
                # No vectorized iterations: skip padding in tail section
                i = offset + tid + pad

                while i <= offset + task_len
                    @inbounds v = data_in[i]
                    stage_pair!(ctx, kctx, v, get_item_index(ctx, kctx, i))
                    i += ndrange_x
                end
            end

            # Flush cache once for small dataset case
            flush_cache!(ctx, kctx, task_id)
            continue
        end

        # ====================================================================
        # Large dataset: filter by k-th element threshold
        # ====================================================================

        # Sample scaling factor and get k-th element
        scaler = sample_scaler(data_in, offset + pad + 1, Val(WITHSCALE))

        if nb_steps > 0
            idx = tid + offset ÷ PACKSIZE

            # First iteration: handle misaligned start (padding)
            vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
            vals_scaled = apply_scaling(vals, scaler, Val(WITHSCALE), Val(LARGEST))
            if tid == 1
                @unroll for j in (pad + 1):PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
            else
                @unroll for j in 1:PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
            end
            idx += ndrange_x

            # Main iterations
            for _ in 2:nb_steps
                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                vals_scaled = apply_scaling(vals, scaler, Val(WITHSCALE), Val(LARGEST))
                @unroll for j in 1:PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
                idx += ndrange_x
            end

            # Tail : skip elements already processed in vectorized loops
            i = offset + tid + nb_steps * step_size
            while i <= offset + task_len
                @inbounds v = data_in[i]
                v_scaled = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                filter_stage_item!(ctx, kctx, v, v_scaled, i)
                i += ndrange_x
            end
        else
            # No vectorized iterations: skip padding in tail section
            i = offset + tid + pad
            while i <= offset + task_len
                @inbounds v = data_in[i]
                v_scaled = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                filter_stage_item!(ctx, kctx, v, v_scaled, i)
                i += ndrange_x
            end
        end

        # Flush cache once for small dataset case
        flush_cache!(ctx, kctx, task_id)
    end
end


"""
    filter_general_kernel!(data_in, idx_in, kth_element, val_out, idx_out, ...)

Enhanced GPU kernel for final top-K selection with aggressive cache flushing.

Same as `filter_kernel!` but optimized for K > 1024. Flushes cache after each
main loop iteration for safer overflow handling at the cost of higher overhead.

**⚠️ CRITICAL**: Same precondition as `filter_kernel!` - `kth_element` must ensure
≈K elements pass the filter to prevent buffer overflow.

**When to use:** K > 1024 (use `filter_kernel!` for K ≤ 1024 for better performance).
See `filter_kernel!` for parameter documentation.
"""
@kernel function filter_general_kernel!(
    data_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    kth_vals::AbstractArray{ValT},
    val_out::AbstractArray{ValT},
    idx_out::AbstractArray{IdxT},
    global_counts::AbstractArray{Int32},
    boundary_counts::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32},
    stride::Int32,
    K::Int32,
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{BLOCK},
    ::Val{PACKSIZE},
    ::Val{WITHSCALE},
    ::Val{LARGEST},
    ::Val{WITHIDXIN}
) where {ValT, IdxT, LEFT, RIGHT, BLOCK, PACKSIZE, WITHSCALE, LARGEST, WITHIDXIN}

    block_x = @index(Group, Cartesian)[1]
    block_y = @index(Group, Cartesian)[2]
    thread_x = @index(Local, Linear)

    ndrange_x = @ndrange()[1]
    grid_dim_y = @ndrange()[2] ÷ @groupsize()[2]

    # ========================================================================
    # Shared memory: counter and element cache
    # ========================================================================
    block_count = @localmem Int32 (1,)
    val_block_cache = @localmem ValT (BLOCK * PACKSIZE,)
    idx_block_cache = @localmem IdxT (BLOCK * PACKSIZE,)

    # ========================================================================
    # Kernel Context
    # ========================================================================
    kctx = KernelFilterContext{LARGEST, BLOCK, PACKSIZE, WITHIDXIN, ValT, IdxT}()

    # ========================================================================
    # Thread and task identification
    # ========================================================================
    tid = (block_x - 1) * BLOCK + thread_x
    task_num = length(task_offsets) - 1
    step_size = PACKSIZE * ndrange_x

    # ========================================================================
    # Helper: reset block counter and synchronize
    # ========================================================================
    @inline function reset_block_count_sync!(thread_x, block_count)
        if thread_x == 1
            @inbounds block_count[1] = 0
        end
        @synchronize()
    end

    # ========================================================================
    # Process each task
    # ========================================================================
    for task_id in block_y:grid_dim_y:task_num
        reset_block_count_sync!(thread_x, block_count)

        # Get task boundaries
        @inbounds offset = task_offsets[task_id]
        original_len = task_offsets[task_id + 1] - offset

        # Handle padding for misaligned addresses
        pad = offset % PACKSIZE
        offset -= pad
        task_len = original_len + pad
        nb_steps = task_len ÷ step_size

        # ====================================================================
        # Create thread context
        # ====================================================================
        ctx = ThreadFilterContext{
            typeof(global_counts),
            typeof(boundary_counts),
            typeof(block_count),
            typeof(val_block_cache),
            typeof(idx_block_cache),
            typeof(val_out),
            typeof(idx_in),
            typeof(idx_out),
            typeof(thread_x),
            eltype(kth_vals),
            typeof(task_id)
        }(
            global_counts,
            boundary_counts,
            block_count,
            val_block_cache,
            idx_block_cache,
            val_out,
            idx_in,
            idx_out,
            thread_x,
            offset,
            pad,
            kth_vals[task_id],
            task_id
        )

        # ====================================================================
        # Small dataset (N ≤ K): copy all elements
        # ====================================================================
        if original_len <= K
            if nb_steps > 0
                idx = tid + offset ÷ PACKSIZE

                # First iteration: handle misaligned start (padding)
                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                if tid == 1
                    @unroll for (j, v) in enumerate(vals)
                        if j > pad
                            stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                        end
                    end
                else
                    @unroll for (j, v) in enumerate(vals)
                        stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                    end
                end
                flush_cache!(ctx, kctx, task_id)
                idx += ndrange_x
                
                # Main iterations
                for _ in 2:nb_steps
                    vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                    @unroll for (j, v) in enumerate(vals)
                        stage_pair!(ctx, kctx, v, get_packed_index(ctx, kctx, idx, j))
                    end
                    flush_cache!(ctx, kctx, task_id)
                    idx += ndrange_x
                end

                reset_block_count_sync!(thread_x, block_count)

                # Tail : skip elements already processed in vectorized loops 
                i = offset + tid + nb_steps * step_size

                while i <= offset + task_len
                    @inbounds v = data_in[i]
                    stage_pair!(ctx, kctx, v, get_item_index(ctx, kctx, i))
                    i += ndrange_x
                end
            else
                # No vectorized iterations: skip padding in tail section
                i = offset + tid + pad

                while i <= offset + task_len
                    @inbounds v = data_in[i]
                    stage_pair!(ctx, kctx, v, get_item_index(ctx, kctx, i))
                    i += ndrange_x
                end
            end

            flush_cache!(ctx, kctx, task_id)
            # no need to reset & sync, because this is done at the beginning of
            # each pass
            
            continue
        end

        # ====================================================================
        # Large dataset: filter by k-th element
        # ====================================================================
        scaler = sample_scaler(data_in, offset + pad + 1, Val(WITHSCALE))

        if nb_steps > 0
            idx = tid + offset ÷ PACKSIZE

            # First iteration: handle misaligned start (padding)
            vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
            vals_scaled = apply_scaling(vals, scaler, Val(WITHSCALE), Val(LARGEST))
            if tid == 1
                @unroll for j in (pad + 1):PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
            else
                @unroll for j in 1:PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
            end
            flush_cache!(ctx, kctx, task_id)
            idx += ndrange_x

            # Main iterations
            for _ in 2:nb_steps
                reset_block_count_sync!(thread_x, block_count)

                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                vals_scaled = apply_scaling(vals, scaler, Val(WITHSCALE), Val(LARGEST))
                @unroll for j in 1:PACKSIZE
                    filter_stage_packed!(ctx, kctx, vals, vals_scaled, idx, j)
                end
                flush_cache!(ctx, kctx, task_id)
                idx += ndrange_x
            end

            reset_block_count_sync!(thread_x, block_count)

            # Update tail start: skip elements already processed in vectorized loops
            i = offset + tid + nb_steps * step_size
            while i <= offset + task_len
                @inbounds v = data_in[i]
                v_scaled = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                filter_stage_item!(ctx, kctx, v, v_scaled, i)
                i += ndrange_x
            end
        else
            # No vectorized iterations: skip padding in tail section
            i = offset + tid + pad
            while i <= offset + task_len
                @inbounds v = data_in[i]
                v_scaled = apply_scaling(v, scaler, Val(WITHSCALE), Val(LARGEST))
                filter_stage_item!(ctx, kctx, v, v_scaled, i)
                i += ndrange_x
            end
        end

        flush_cache!(ctx, kctx, task_id)
        # no need to reset & sync, because this is done at the beginning of
        # each pass
    end
end

# ==============================================================================
# Convenience wrappers
# ==============================================================================

"""
    filter!(data_in, idx_in, kth_element, val_out, idx_out, global_counts,
            boundary_counts, task_offsets, stride, K; ...)

Convenience wrapper for `filter_kernel!`. Filters elements by k-th element threshold.

# Arguments
- `data_in`: Input data array (original values)
- `idx_in`: Input indices array
- `kth_element`: K-th element values (scaled, one per task)
- `val_out`: Output values array [K × num_tasks]
- `idx_out`: Output indices array [K × num_tasks]
- `global_counts`: Global count array [num_tasks] (initialize to 0)
- `boundary_counts`: Boundary count array [num_tasks] (elements equal to k-th)
- `task_offsets`: Task offset array [num_tasks + 1]
- `stride`: Stride between task data in arrays
- `K`: Number of top elements to select per task

# Keyword Arguments
- `LEFT=0`, `RIGHT=20`: Bit shift parameters
- `threads_per_block=256`: GPU block size
- `blocks_x=16`: Number of blocks in x-dimension
- `pack_size=4`: Elements per vectorized load
- `with_scale=true`: Apply adaptive scaling
- `largest=true`: Select largest (true) or smallest (false)
- `with_idx_in=false`: Whether input indices are provided

# Constraints
K ≤ 1024 (use `filter_general!` for larger K).
"""
function filter_!(
    data_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    kth_element::AbstractArray{ValT},
    val_out::AbstractArray{ValT},
    idx_out::AbstractArray{IdxT},
    global_counts::AbstractArray{Int32},
    boundary_counts::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32},
    stride::Int32,
    K::Int32;
    LEFT::Int=0,
    RIGHT::Int=20,
    threads_per_block=256,
    blocks_x=16,
    pack_size::Int=4,
    with_scale::Bool=true,
    largest::Bool=true,
    with_idx_in::Bool=false
) where {ValT, IdxT}
    backend = get_backend(data_in)
    num_tasks = length(task_offsets) - 1

    # Select cache size based on K
    cachesize = if K <= 128
        128
    elseif K <= 256
        256
    elseif K <= 512
        512
    elseif K <= 1024
        1024
    else
        error("K > 1024 not supported by filter!. Use filter_general! instead.")
    end

    kernel! = filter_kernel!(backend, threads_per_block)

    kernel!(
        data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
        task_offsets, stride, K,
        Val(LEFT), Val(RIGHT), Val(threads_per_block), Val(pack_size), Val(cachesize),
        Val(with_scale), Val(largest), Val(with_idx_in);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )

    KA.synchronize(backend)
end


"""
    filter_general!(data_in, idx_in, kth_element, val_out, idx_out, global_counts,
                    boundary_counts, task_offsets, stride, K; ...)

Convenience wrapper for `filter_general_kernel!`. For K > 1024 with aggressive
cache flushing to prevent overflow.

Same parameters as `filter!`. Use when K > 1024.
"""
function filter_general!(
    data_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    kth_element::AbstractArray{ValT},
    val_out::AbstractArray{ValT},
    idx_out::AbstractArray{IdxT},
    global_counts::AbstractArray{Int32},
    boundary_counts::AbstractArray{Int32},
    task_offsets::AbstractArray{Int32},
    stride::Int32,
    K::Int32;
    LEFT::Int=0,
    RIGHT::Int=20,
    threads_per_block=256,
    blocks_x=16,
    pack_size::Int=4,
    with_scale::Bool=true,
    largest::Bool=true,
    with_idx_in::Bool=false
) where {ValT, IdxT}
    backend = get_backend(data_in)
    num_tasks = length(task_offsets) - 1

    kernel! = filter_general_kernel!(backend, threads_per_block)

    kernel!(
        data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
        task_offsets, stride, K,
        Val(LEFT), Val(RIGHT), Val(threads_per_block), Val(pack_size),
        Val(with_scale), Val(largest), Val(with_idx_in);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )

    KA.synchronize(backend)
end
