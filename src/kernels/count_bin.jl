using KernelAbstractions
# using KernelAbstractions: @index, @localmem, @kernel, @groupsize, get_backend, synchronize
using KernelIntrinsics: vload
using Atomix


@kernel function count_bin_kernel!(
    histogram, 
    data_in, 
    ::Val{LEFT}, 
    ::Val{RIGHT}
) where {LEFT, RIGHT}
    
    tile_id = @index(Group, Linear)
    local_idx = @index(Local, Linear)
    local_size = @groupsize()[1]

    # Allocate and zero shared memory histogram
    local_hist = @localmem Int32 (256,)
    for i in local_idx:local_size:256
        @inbounds local_hist[i] = 0
    end
    @synchronize()

    # Count elements: write to LOCAL histogram first
    local_size_i = Int32(local_size)
    tid = (tile_id - 1) * local_size_i + local_idx
    if tid <= length(data_in)
        val = @inbounds data_in[tid]
        bin_id = get_bin_id(val, Val(LEFT), Val(RIGHT))
        Atomix.@atomic local_hist[bin_id + 1] += 1  # ← FIXED: local_hist not global
    end
    @synchronize()

    # Atomic merge into single global histogram
    for bin in local_idx:local_size:256  # ← FIXED: was hist_size
        if local_hist[bin] > 0
            Atomix.@atomic histogram[bin] += local_hist[bin]
        end
    end
end


"""
    count_bin_ka!(histogram, data_in, task_offsets, task_num, ...)

GPU kernel for counting histogram bins across multiple tasks with vectorized loads.

# Algorithm
- Each task processes a contiguous range of `data_in`
- Threads within a block collaborate to build a shared (local) histogram
- Vectorized loads (`vload`) read PACKSIZE elements at once for efficiency
- Three processing phases per task:
  1. **First loop**: Handle misaligned start (padding) if offset not PACKSIZE-aligned
  2. **Main loop**: Full vectorized loads, each thread processes PACKSIZE values
  3. **Tail**: Scalar loads for remaining elements (< PACKSIZE)

# Parameters
- `histogram`: 2D output array [hist_len, num_tasks]
- `data_in`: Input data array
- `task_offsets`: Prefix sum array defining task boundaries
- `task_num`: Number of tasks to process
- `Val{LEFT}`, `Val{RIGHT}`: Bit shift parameters for bin calculation
- `Val{HIST_LEN}`: Number of histogram bins
- `Val{PACKSIZE}`: Elements per vectorized load
- `Val{WITHSCALE}`: Whether to apply adaptive scaling
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
    block_hist = @localmem Int32 (HIST_LEN,)

    # Linear thread ID across all blocks in x-dimension
    tid = (block_x - 1) * BLOCK + thread_x

    # Elements processed by one thread iteration (PACKSIZE elements per load)
    step_size = PACKSIZE * BLOCK * grid_dim_x

    @inline function update_hist(hist, val, scaler)
        scaled_val = apply_scaling(val, scaler, Val(WITHSCALE), Val(LARGEST))
        bin_id = get_bin_id(scaled_val, Val(LEFT), Val(RIGHT))
        @inbounds Atomix.@atomic hist[bin_id + 1] += 1
    end
    
    # Process tasks in strided fashion (block_y acts as task group ID)
    for task_id in block_y:grid_dim_y:task_num

        # Initialize shared memory histogram to zeros
        for i in thread_x:BLOCK:HIST_LEN
            @inbounds block_hist[i] = 0
        end
        @synchronize()
        
        # Get task boundaries from prefix sum
        offset = task_offsets[task_id]
        pad = offset % PACKSIZE
        offset -= pad

        # Adaptive scaling: subtract element to handle adversarial distributions
        # Reduces bin collisions by adjusting exponents, making elements more distinguishable
        scaler = sample_scaler(data_in, offset + pad + 1, Val(WITHSCALE))

        task_len = task_offsets[task_id + 1] - offset   # Number of elements for current task
        nb_steps = task_len ÷ step_size                 # Number of vectorized iterations
        start_tail = offset + tid                       # Starting index for tail (scalar) processing

        # ===========================
        # Vectorized section: Process PACKSIZE elements per iteration
        # ===========================
        if nb_steps > 0
            idx = tid + offset ÷ PACKSIZE
            start_main = 1

            # First iteration: handle misaligned start (padding)
            # Thread 1 skips 'pad' elements to align to PACKSIZE boundary
            if tid == 1
                vals = vload(data_in, idx, Val(PACKSIZE))
                for j in (pad+1):PACKSIZE
                    update_hist(block_hist, vals[j], scaler)
                end
                idx += BLOCK * grid_dim_x
                start_main += 1
            end

            # Main loop: full vectorized loads, each thread processes PACKSIZE values
            for _ in start_main:nb_steps
                vals = vload(data_in, idx, Val(PACKSIZE))
                for v in vals
                    update_hist(block_hist, v, scaler)
                end
                idx += BLOCK * grid_dim_x
            end

            # Update tail start: skip elements already processed in vectorized loops 
            start_tail += nb_steps * step_size
        else
            # No vectorized iterations: skip padding in tail section
            start_tail += pad
        end

        # ===========================
        # Tail section: Scalar loads for remaining elements (< PACKSIZE)
        # ===========================
        for i in start_tail:(BLOCK * grid_dim_x):(offset + task_len)
            v = data_in[i]
            update_hist(block_hist, v, scaler)
        end
        @synchronize()

        # Merge block-level histogram into global histogram (2D: [bin, task_id])
        for bin in thread_x:BLOCK:HIST_LEN
            count = @inbounds block_hist[bin]
            count > 0 && Atomix.@atomic histogram[bin, task_id] += count
        end
        @synchronize()
    end
end

# ========================================
# Convenience wrapper
# ========================================

function count_bin!(
    data_in::AbstractArray{Float32}, 
    histogram::AbstractArray{Int32}, 
    LEFT::Int = 0,
    RIGHT::Int = 28,
)
    backend = get_backend(data_in)
    n = length(data_in)
    threads_per_block = 256
    num_blocks = cld(n, threads_per_block)

    histogram .= 0

    kernel! = _count_bin_kernel(backend, threads_per_block)
    kernel!(
        data_in, histogram, Val(LEFT), Val(RIGHT), 
        ndrange=num_blocks * threads_per_block
    )
end


function count_bin_ex!(
    data_in::AbstractArray{T},
    histogram::AbstractArray{Int32, 2},
    task_offsets::AbstractArray{Int32};
    LEFT::Int = 0,
    RIGHT::Int = 28,
    with_scale::Bool = true,
    largest::Bool = true
) where {T}

    backend = get_backend(data_in)

    ########### KERNEL PARAMS ######### 
    threads_per_block = 1024
    blocks_x = 16
    pack_size = 4
    ################################## 

    histogram .= 0

    hist_len = 1 << (8 * sizeof(T) - RIGHT)
    kernel! = count_bin_kernel!(backend, threads_per_block)

    # 2D grid: blocks_x × num_tasks
    num_tasks = length(task_offsets) - 1

    kernel!(
        histogram, data_in, task_offsets,
        Val(LEFT), Val(RIGHT), Val(hist_len), Val(pack_size), Val(with_scale), Val(largest);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )
end