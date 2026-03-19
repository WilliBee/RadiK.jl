using KernelAbstractions
import KernelAbstractions as KA
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics: vload
using Atomix

"""
    count_bin_kernel!(histogram, data_in, task_lens, stride, ...)

GPU kernel for histogram bin counting. Uses per-block shared memory with atomic merging.

# Arguments
- `histogram`: 2D output array [hist_len, num_tasks]
- `data_in`: Input data array
- `task_lens`: Length of each task
- `stride`: Maximum stride for data access
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters for bin calculation
- `Val{HIST_LEN}`: Number of histogram bins
"""
@kernel function count_bin_kernel!(
    histogram,
    data_in,
    task_lens,
    stride,
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{HIST_LEN}
) where {LEFT, RIGHT, HIST_LEN}

    # 2D grid/block indices
    block_x = @index(Group, Cartesian)[1]
    block_y = @index(Group, Cartesian)[2]
    thread_x = @index(Local, Linear)
    BLOCK = @groupsize()[1]

    # Allocate and zero shared memory histogram
    block_hist = @localmem Int32 (HIST_LEN,)
    for i in thread_x:BLOCK:HIST_LEN
        @inbounds block_hist[i] = 0
    end
    @synchronize()

    task_id = block_y
    task_len = task_lens[task_id]
    tid = (block_x - 1) * BLOCK + thread_x
    
    if tid ≤ task_len
        val = @inbounds data_in[(task_id - 1) * stride + tid]
        bin_id = get_bin_id(val, Val(LEFT), Val(RIGHT))
        Atomix.@atomic block_hist[bin_id + 1] += 1
    end
    @synchronize()

    # Atomic merge into single global histogram
    for bin in thread_x:BLOCK:HIST_LEN
        count = block_hist[bin]
        count > 0 && Atomix.@atomic histogram[bin, block_y] += count
    end
end


"""
    count_bin_ex_kernel!(histogram, data_in, task_offsets, ...)

GPU kernel for histogram bin counting with vectorized loads across multiple tasks.
Uses shared memory per-block with atomic merging to global histogram.

# Arguments
- `histogram`: 2D output array [hist_len, num_tasks]
- `data_in`: Input data array
- `task_offsets`: Prefix sum array defining task boundaries
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters for bin calculation
- `Val{HIST_LEN}`: Number of histogram bins
- `Val{PACKSIZE}`: Elements per vectorized load
- `Val{WITHSCALE}`: Apply adaptive scaling for adversarial distributions
- `Val{LARGEST}`: NaN handling (true → min, false → max)
"""
@kernel function count_bin_ex_kernel!(
    histogram::AbstractArray{Int32, 2},
    data_in::AbstractArray{T},
    task_offsets::AbstractArray{Int32},
    ::Val{LEFT},
    ::Val{RIGHT},
    ::Val{HIST_LEN},
    ::Val{PACKSIZE},
    ::Val{WITHSCALE},
    ::Val{LARGEST},
) where {T, LEFT, RIGHT, HIST_LEN, PACKSIZE, WITHSCALE, LARGEST}

    # 2D grid/block indices
    block_x = @index(Group, Cartesian)[1]
    block_y = @index(Group, Cartesian)[2]
    thread_x = @index(Local, Linear)
    grid_dim_x = @ndrange()[1] ÷ @groupsize()[1]
    grid_dim_y = @ndrange()[2] ÷ @groupsize()[2]
    BLOCK = @groupsize()[1]

    # Number of tasks
    task_num = length(task_offsets) - 1

    # Allocate shared memory histogram (within block)
    block_hist = @localmem Int32 (HIST_LEN, )

    # Linear thread ID across all blocks in x-dimension
    tid = (block_x - 1) * BLOCK + thread_x

    # Elements processed by one thread iteration (PACKSIZE elements per load)
    step_size = PACKSIZE * BLOCK * grid_dim_x

    @inline function update_hist!(hist, val, scaler)
        if !isnan(val)
            scaled_val = apply_scaling(val, scaler, Val(WITHSCALE), Val(LARGEST))
            bin_id = get_bin_id(scaled_val, Val(LEFT), Val(RIGHT))
            @inbounds Atomix.@atomic hist[bin_id + 1] += 1
        end
    end
    
    # Process tasks in strided fashion (block_y acts as task group ID)
    for task_id in block_y:grid_dim_y:task_num

        # Initialize shared memory histogram to zeros
        i = thread_x
        while i <= HIST_LEN
            @inbounds block_hist[i] = 0
            i += BLOCK
        end
        @synchronize()
        
        # Get task boundaries from prefix sum
        @inbounds offset = task_offsets[task_id]
        pad = offset % PACKSIZE
        offset -= pad

        # Adaptive scaling: subtract element to handle adversarial distributions
        # Reduces bin collisions by adjusting exponents, making elements more distinguishable
        scaler = sample_scaler(data_in, offset + pad + 1, Val(WITHSCALE))

        @inbounds task_len = task_offsets[task_id + 1] - offset   # Number of elements for current task
        nb_steps = task_len ÷ step_size                 # Number of vectorized iterations

        # ===========================
        # Vectorized section: Process PACKSIZE elements per iteration
        # ===========================
        if nb_steps > 0
            idx = tid + offset ÷ PACKSIZE

            # First iteration: handle misaligned start (padding)
            vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
            if tid == 1
                # Thread 1 skips 'pad' elements to align to PACKSIZE boundary
                @unroll for (j, v) in enumerate(vals)
                    if j > pad
                        update_hist!(block_hist, v, scaler)
                    end
                end
            else
                @unroll for v in vals
                    update_hist!(block_hist, v, scaler)
                end
            end
            idx += BLOCK * grid_dim_x

            # Main loop: full vectorized loads, each thread processes PACKSIZE values
            for _ in 2:nb_steps
                vals = vload(data_in, idx, Val(PACKSIZE), Val(true), Val(1))
                @unroll for v in vals
                    update_hist!(block_hist, v, scaler)
                end
                idx += BLOCK * grid_dim_x
            end

            # Update tail start: skip elements already processed in vectorized loops 
            i = offset + tid + nb_steps * step_size
            while i <= offset + task_len
                @inbounds v = data_in[i]
                update_hist!(block_hist, v, scaler)
                i += BLOCK * grid_dim_x
            end
        else
            # No vectorized iterations: skip padding in tail section
            i = offset + tid + pad
            while i <= offset + task_len
                @inbounds v = data_in[i]
                update_hist!(block_hist, v, scaler)
                i += BLOCK * grid_dim_x
            end
        end

        @synchronize()

        # Merge block-level histogram into global histogram (2D: [bin, task_id])
        bin = thread_x
        while bin <= HIST_LEN
            @inbounds count = block_hist[bin]
            if count > 0
                @inbounds Atomix.@atomic histogram[bin, task_id] += count
            end
            bin += BLOCK
        end

        @synchronize()
    end
end

# ========================================
# Convenience wrappers
# ========================================

"""
    count_bin!(histogram, data_in, task_lens, stride; ...)

Convenience wrapper for `count_bin_kernel!`. Computes histogram bins for multi-task data.

# Arguments
- `histogram`: 2D output array [hist_len, num_tasks]
- `data_in`: Input data array (AbstractArray{Float32})
- `task_lens`: Length of each task
- `stride`: Maximum stride for data access
- `LEFT`: Bit shift parameter (default: 0)
- `RIGHT`: Bit shift parameter (default: 28)
- `threads_per_block`: Thread block size (default: 256)
"""
function count_bin!(
    histogram::AbstractArray{Int32}, 
    data_in::AbstractArray{Float32},
    task_lens::AbstractArray{Int32},
    stride::Int32;
    LEFT::Int = 0,
    RIGHT::Int = 28,
    threads_per_block = 256
)
    backend = get_backend(data_in)
    n = length(data_in)
    
    num_blocks = cld(n, threads_per_block)
    num_tasks = length(task_lens)
    histogram .= 0

    kernel! = count_bin_kernel(backend, threads_per_block)
    kernel!(
        histogram, data_in, 
        task_lens, stride,
        Val(LEFT), Val(RIGHT), Val(hist_len),
        ndrange=(num_blocks * threads_per_block, num_tasks)
    )
end


"""
    count_bin_ex!(histogram, data_in, task_offsets; ...)

Convenience wrapper for `count_bin_ex_kernel!`. Computes histogram bins with vectorized loads.

# Arguments
- `histogram`: 2D output array [hist_len, num_tasks]
- `data_in`: Input data array
- `task_offsets`: Prefix sum array defining task boundaries
- `LEFT`: Bit shift parameter (default: 0)
- `RIGHT`: Bit shift parameter (default: 28)
- `threads_per_block`: Thread block size (default: 1024)
- `blocks_x`: Number of blocks in x-dimension (default: 16)
- `pack_size`: Elements per vectorized load (default: 4)
- `with_scale`: Apply adaptive scaling (default: true)
- `largest`: NaN handling (default: true)
"""
function count_bin_ex!(
    histogram::AbstractArray{Int32, 2},
    data_in::AbstractArray{T},
    task_offsets::AbstractArray{Int32};
    LEFT::Int = 0,
    RIGHT::Int = 28,
    threads_per_block = 1024,
    blocks_x = 16,
    pack_size = 4,
    with_scale::Bool = true,
    largest::Bool = true
) where {T}

    backend = get_backend(data_in)

    histogram .= 0

    hist_len = 1 << (8 * sizeof(T) - RIGHT)
    kernel! = count_bin_ex_kernel!(backend, threads_per_block)

    # 2D grid: blocks_x × num_tasks
    num_tasks = length(task_offsets) - 1

    kernel!(
        histogram, data_in, task_offsets,
        Val(LEFT), Val(RIGHT), Val(hist_len), Val(pack_size), Val(with_scale), Val(largest);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )

    KA.synchronize(backend)
end