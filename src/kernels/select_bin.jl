# ==============================================================================
# select_bin: Find Bin Containing K-th Element
# ==============================================================================
# Based on radik/radik/RadixSelect/radixselect_l.cuh:45-126

using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, @inbounds
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics: @shfl

"""
    cross_warp_reduction!(prefix_sum, warp_id, thread_x, sum, ...)

Tree reduction across warps to compute global prefix sum. Doubles distance each iteration
(i=1,2,4,8...). LARGEST scans forward (adds from higher warps), smallest scans backward.

**Parameters:**
- `prefix_sum`: Shared memory array [BLOCK] storing per-thread prefix sums
- `warp_id`: 0-indexed warp ID (0 to NUM_WARPS-1)
- `thread_x`: 1-indexed thread ID in block (1 to BLOCK)
- `sum`: This thread's prefix sum from warp-level scan (modified in-place)
- `Val{LARGEST}`: Scan direction (true=forward, false=backward)
- `Val{NUM_WARPS}`: Total warps in block
- `Val{WARP_SIZE}`: Threads per warp (typically 32)

**Returns:** Updated global prefix sum for this thread
"""
@inline function cross_warp_reduction!(
    prefix_sum,
    warp_id,
    thread_x,
    sum,
    ::Val{LARGEST},
    ::Val{NUM_WARPS},
    ::Val{WARP_SIZE}
) where {LARGEST, NUM_WARPS, WARP_SIZE}

    i = 1
    while i < NUM_WARPS
        if LARGEST
            # Forward scan: add from higher warps
            # Each iteration doubles the warp distance (tree reduction)
            if (warp_id + i) < NUM_WARPS
                # Read first thread of next warp
                sum += @inbounds prefix_sum[(warp_id + i) * WARP_SIZE + 1]
            end
        else
            # Backward scan: add from lower warps
            # Each iteration doubles the warp distance (tree reduction)
            if (warp_id - i) >= 0
                # Read last thread of previous warp
                sum += @inbounds prefix_sum[(warp_id - i + 1) * WARP_SIZE]
            end
        end

        @synchronize()
        @inbounds prefix_sum[thread_x] = sum
        @synchronize()

        i *= 2  # Double distance: 1, 2, 4, 8, ...
    end

    return sum
end

"""
    compute_neighbor(prefix_sum, thread_x, ::Val{LARGEST}, ::Val{BLOCK}, T)

Compute the neighbor thread's prefix sum to determine this thread's element range.

# Algorithm
- For `LARGEST`: neighbor = next thread's prefix sum (exclusive scan from right)
- For `!LARGEST`: neighbor = previous thread's prefix sum (exclusive scan from left)
- Boundary threads return 0 (no neighbor)

# Arguments
- `prefix_sum`: Shared memory array storing per-thread prefix sums
- `thread_x`: 1-indexed thread ID in block (1 to BLOCK)
- `Val{LARGEST}`: Direction (true=from right/next, false=from left/previous)
- `Val{BLOCK}`: Total threads in block
- `T`: Element type (for zero initialization)

# Returns
Neighbor's prefix sum, or 0 if no neighbor.

# Examples
```julia
# BLOCK=4, prefix_sum=[2,5,9,15], LARGEST=true
# Thread 1: neighbor = prefix_sum[2] = 5
# Thread 2: neighbor = prefix_sum[3] = 9
# Thread 3: neighbor = prefix_sum[4] = 15
# Thread 4: neighbor = 0 (no next thread)
```
"""
@inline function compute_neighbor(
    prefix_sum, thread_x, ::Val{LARGEST}, ::Val{BLOCK}, T
) where {LARGEST, BLOCK}
    if LARGEST
        # For largest: neighbor is next thread's prefix sum (exclusive from right)
        return thread_x < BLOCK ? @inbounds(prefix_sum[thread_x + 1]) : T(0)
    else
        # For smallest: neighbor is previous thread's prefix sum (exclusive from left)
        return thread_x > 1 ? @inbounds(prefix_sum[thread_x - 1]) : T(0)
    end
end

"""
    find_bin_and_write!(counts, old_k, neighbor, sum, thread_x, UNROLL,
                       task_id, bin_ids, k_values, task_lens, ::Val{LARGEST})

Find the histogram bin containing the k-th element and write results.

# Algorithm
1. Check if k-th element falls within this thread's range: neighbor < k ≤ sum
2. If yes, search within this thread's bins to find exact bin
3. Write bin ID (0-indexed), updated k value, and bin count to output arrays

# Search Direction
- `LARGEST=true`: search backward from highest bin to lowest
- `LARGEST=false`: search forward from lowest bin to highest

# Arguments
- `counts`: Private memory array of bin counts assigned to this thread
- `old_k`: k value to find
- `neighbor`: Previous thread's cumulative count (exclusive lower bound)
- `sum`: This thread's cumulative count (inclusive upper bound)
- `thread_x`: 1-indexed thread ID in block
- `UNROLL`: Number of bins per thread
- `task_id`: Task ID (1-indexed)
- `bin_ids`: Output array for selected bin IDs (0-indexed)
- `k_values`: Output array for updated k values (1-indexed within bin)
- `task_lens`: Output array for counts in selected bin
- `Val{LARGEST}`: Search direction
"""
@inline function find_bin_and_write!(
    counts, old_k, neighbor, sum, thread_x, UNROLL,
    task_id, bin_ids, k_values, task_lens, ::Val{LARGEST}
) where LARGEST
    # Check if k-th element is in this thread's range
    if neighbor < old_k <= sum
        old_k -= neighbor  # Adjust k to be relative to this thread's first element

        # Search within this thread's bins
        # For largest: search backward (from highest bin to lowest)
        # For smallest: search forward (from lowest bin to highest)
        for i in (LARGEST ? (UNROLL:-1:1) : (1:UNROLL))
            if @inbounds counts[i] >= old_k
                # Found bin, compute global bin index
                bin_idx = (thread_x - 1) * UNROLL + i

                # Write results
                @inbounds bin_ids[task_id] = bin_idx
                @inbounds k_values[task_id] = old_k
                @inbounds task_lens[task_id] = @inbounds counts[i]
                break
            end
            old_k -= @inbounds counts[i]
        end
    end
end

"""
    select_bin_kernel!(histogram, bin_ids, k_values, task_lens, ...)

GPU kernel that finds the histogram bin containing the k-th element for each task.

# Purpose
Given a histogram and a rank k, determines which histogram bin contains the k-th element.
This is used in radix select to progressively narrow down the search space.

# Algorithm Overview
1. **Histogram loading**: Each thread loads its assigned bins into local storage
2. **Warp-level scan**: Compute inclusive prefix sum within each warp using shuffle
3. **Cross-warp reduction**: Tree reduction combines warp results into global prefix sum
4. **Binary search**: Each thread checks if k falls within its range, searches its bins

# Arguments
- `histogram`: 2D array [HISTLEN, num_tasks] of bin counts
- `bin_ids`: Output array [num_tasks] for selected bin IDs (0-indexed)
- `k_values`: Input/output array [num_tasks] for k values (updated to position within bin)
- `task_lens`: Output array [num_tasks] for counts in selected bin
- `Val{LARGEST}`: Find k-th largest (true) or k-th smallest (false)
- `Val{BLOCK}`: Threads per block (must be multiple of 32)
- `Val{HISTLEN}`: Histogram length (must be multiple of BLOCK)
- `Val{UNROLL}`: Bins per thread (HISTLEN ÷ BLOCK)
- `Val{WARP_SIZE}`: Threads per warp (typically 32)

# Example

Setup:

    WARP_SIZE = 4, BLOCK = 8, HISTLEN = 24
    UNROLL = 16/8 = 2 (each thread handles 2 bins)
    NUM_WARPS = 2 (warp 0: threads 0-3, warp 1: threads 4-7)

Step 1 : Initial Data (let's say we're finding LARGEST):

    Thread:   0   1   2   3   4   5   6   7   8  10  11  12
    count[0]: 3   2   5   1   4   2   3   2   1   1   3   2 (bins 0,2,4,6,8,10,12,14,16,18,20,22)
    count[1]: 1   4   2   3   2   5   1   4   0   2   4   0 (bins 1,3,5,7,9,11,13,15,17,19,21,23)
    sum:      4   6   7   4   6   7   4   6   1   3   7   2 (per-thread sum)

Step 2 : Warp prefix sum

    For warp 0 (threads 0-3), lane = threadIdx % 4:

        i=1: __shfl_down_sync gets data from lane+1
            lane 0: gets 6 from lane 1, sum = 4+6 = 10
            lane 1: gets 7 from lane 2, sum = 6+7 = 13
            lane 2: gets 4 from lane 3, sum = 7+4 = 11
            lane 3: condition lane < 3 false, stays 4
        i=2: __shfl_down_sync gets data from lane+2
            lane 0: gets 11 from lane 2, sum = 10+11 = 21
            lane 1: gets 4 from lane 3, sum = 13+4 = 17
            lanes 2,3: condition false

    Warp 0 prefix sums: [21, 17, 11, 4] (cumulative from left, descending)
    Similarly for   warp 1: sums become [23, 17, 10, 6]
                    warp 2: sums become [13, 12,  9, 2]

    Storing to shared memory :

        prefixSum = [21, 17, 11, 4, 23, 17, 10, 6, 13, 12,  9, 2]


Step 3: Block-level prefix sum (LARGEST path)

    Loop i=1:
        warp 0 (threads 0-3): warpId + 1 = 1 < 3, so add prefixSum[1*4] = prefixSum[4] = 23
            sum becomes: 21+23=44, 17+23=40, 11+23=34, 4+23=27
        warp 1 (threads 4-7): warpId + 1 = 2 < 3, so add prefixSum[2*4] = prefixSum[8] = 13
            sum becomes: 36, 30, 23, 19

    After __syncthreads and store:

        prefixSum = [44, 40, 34, 27, 36, 30, 23, 19, 13, 12,  9, 2]

    Loop i=2: condition 1*2 < 3 true, loop continues

        prefixSum = [57, 53, 47, 40, 36, 30, 23, 19, 13, 12,  9, 2]


    Loop i=3: condition 1*4 < 3 false, loop stops
    
    Finally
        prefixSum = [57, 53, 47, 40, 36, 30, 23, 19, 13, 12, 9, 2]

    This represents cumulative sums from the right (largest bin indices):

        Thread 11 (bins 22,23): sum = 2
        Thread 10 (bins 20,21): sum = 7+2 = 9
        Thread  9 (bins 18,19): sum = 3+9 = 12
        ...

Step 4: Select bin

    For each thread, neighbor = prefixSum[threadIdx.x + 1] (next thread's cumulative):

        Thread 0: neighbor = 53, sum = 57
        Thread 1: neighbor = 47, sum = 53
        etc.

    If oldK (say k=50) satisfies 50 > 47 && 50 <= 53, thread 1 is selected.
    Then within thread 1's count[1]=4, count[0]=2 (bins 3,2), scan from largest:

        i=1 (bin 3): count=4 >= 50-47=3? Yes! Select bin 3, k=3, taskLen=4.

"""
@kernel function select_bin_kernel!(
    histogram::AbstractArray{T, 2},
    bin_ids::AbstractArray{T},
    k_values::AbstractArray{T},
    task_lens::AbstractArray{T},
    ::Val{LARGEST},
    ::Val{BLOCK},
    ::Val{HISTLEN},
    ::Val{UNROLL},
    ::Val{WARP_SIZE}
) where {T, LARGEST, BLOCK, HISTLEN, UNROLL, WARP_SIZE}

    # ========================================================================
    # Initialization
    # ========================================================================
    @assert HISTLEN % BLOCK == 0 "select_bin_kernel! requires HISTLEN % BLOCK == 0"
    @assert HISTLEN ÷ BLOCK == UNROLL "select_bin_kernel! requires HISTLEN ÷ BLOCK == UNROLL"

    # Shared memory for prefix sums across all threads in block
    prefix_sum = @localmem T (BLOCK,)

    # Thread and task identification
    thread_x = @index(Local, Linear)           # 1-indexed: 1 to BLOCK
    task_id = @index(Group, Linear)            # 1-indexed: 1 to num_tasks
    warp_id = (thread_x - 1) ÷ WARP_SIZE      # 0-indexed: 0 to NUM_WARPS-1
    lane = (thread_x - 1) % WARP_SIZE + 1     # 1-indexed: 1 to 32

    # ========================================================================
    # Step 1: Load histogram bins assigned to this thread
    # ========================================================================
    # Each thread loads UNROLL consecutive bins
    # Thread 1: bins [1, UNROLL], Thread 2: bins [UNROLL+1, 2*UNROLL], etc.
    old_k = @inbounds k_values[task_id]
    counts = @private T (UNROLL,)
    sum = T(0)

    @unroll for i in 1:UNROLL
        bin_idx = (thread_x - 1) * UNROLL + i
        @inbounds counts[i] = @inbounds histogram[bin_idx, task_id]
        sum += counts[i]
    end

    # ========================================================================
    # Step 2: Warp-level inclusive scan (prefix sum)
    # ========================================================================
    # Compute inclusive scan within warp using shuffle operations
    # After this: each thread's sum = prefix sum of its warp
    if LARGEST
        @warpreduce(sum, lane, +, Down, WARP_SIZE)
    else
        @warpreduce(sum, lane, +, Up, WARP_SIZE)
    end
    @inbounds prefix_sum[thread_x] = sum
    @synchronize()

    @assert BLOCK % WARP_SIZE == 0 "BLOCK must be divisible by WARP_SIZE"
    NUM_WARPS = BLOCK ÷ WARP_SIZE

    # ========================================================================
    # Step 3: Cross-warp tree reduction
    # ========================================================================
    # Combine warp results using tree reduction pattern
    # Iteration i=1: adjacent warps, i=2: distance 2, i=4: distance 4, etc.
    sum = cross_warp_reduction!(
        prefix_sum, warp_id, thread_x, sum,
        Val(LARGEST), Val(NUM_WARPS), Val(WARP_SIZE)
    )

    # ========================================================================
    # Step 4: Find bin containing k-th element
    # ========================================================================
    # Each thread checks if k falls within its range and searches its bins
    neighbor = compute_neighbor(prefix_sum, thread_x, Val(LARGEST), Val(BLOCK), T)

    find_bin_and_write!(
        counts, old_k, neighbor, sum, thread_x, UNROLL,
        task_id, bin_ids, k_values, task_lens, Val(LARGEST)
    )

end


# ==============================================================================
# Convenience wrapper
# ==============================================================================

"""
    select_bin!(histogram, bin_ids, k_values, task_lens;
               hist_len=256, largest=true, threads_per_block=1024, warp_size=32)

Find the histogram bin containing the k-th element for each task.

For each task, determines which histogram bin contains the k-th smallest or largest element,
and updates the k value to be the position within that bin. Used iteratively in radix select
to progressively narrow down the search space.

# Arguments
- `histogram`: 2D histogram array [hist_len, num_tasks] with bin counts
- `bin_ids`: Output array [num_tasks] for selected bin IDs (0-indexed)
- `k_values`: Input/output array [num_tasks] for k values (updated to 1-indexed position within bin)
- `task_lens`: Output array [num_tasks] for counts in selected bin
- `hist_len`: Number of histogram bins (must be power of 2, default 256)
- `largest`: Find k-th largest (true) or k-th smallest (false) elements (default true)
- `threads_per_block`: GPU block size (must be multiple of 32, default 1024)
- `warp_size`: Warp size for GPU (default 32)

# Constraints
- `hist_len` must be divisible by `threads_per_block`
- `threads_per_block` must be divisible by `warp_size`
- Number of tasks ≤ available GPU blocks

# Algorithm
1. One task per GPU block
2. Each thread loads `hist_len / threads_per_block` bins
3. Warp-level prefix sum using shuffle operations
4. Tree reduction across warps
5. Binary search to find bin containing k-th element

# Outputs
- `bin_ids`: Histogram bin index containing k-th element (0-indexed)
- `k_values`: Position of k-th element within that bin (1-indexed)
- `task_lens`: Number of elements in that bin

# Example
```julia
using CUDA, RadiK

# Simple histogram: 10 bins with 1 element each
histogram = CUDA.zeros(Int32, 256, 1)
histogram[1:10, 1] .= 1
bin_ids = CUDA.zeros(Int32, 1)
k_values = CuArray{Int32}([5])   # Find 5th element
task_lens = CUDA.zeros(Int32, 1)

select_bin!(histogram, bin_ids, k_values, task_lens)

println("Target bin: ", Array(bin_ids)[1])    # Output: 4 (0-indexed, 5th element is in bin 4)
println("New k: ", Array(k_values)[1])        # Output: 1 (1st element within that bin)
println("Bin count: ", Array(task_lens)[1])   # Output: 1 (bin has 1 element)
```

# See also
- [`count_bin!`](@ref): Build histogram from data
- [`select_bin_kernel!`](@ref): GPU kernel implementation
"""
function select_bin!(
    histogram::AbstractArray{T, 2},
    bin_ids::AbstractArray{T},
    k_values::AbstractArray{T},
    task_lens::AbstractArray{T};
    largest::Bool = true,
    threads_per_block = 32,
    warp_size = 32
) where T
    backend = get_backend(histogram)

    # Get dimensions
    hist_len, num_tasks = size(histogram)

    # Validate inputs
    @assert hist_len % threads_per_block == 0 "hist_len must be divisible by threads_per_block"
    @assert threads_per_block % warp_size == 0 "threads_per_block must be divisible by warp_size"

    # Compute bins per thread (must be compile-time constant for @private)
    unroll = hist_len ÷ threads_per_block

    # Compile and launch kernel
    kernel! = select_bin_kernel!(backend, threads_per_block)

    kernel!(
        histogram, bin_ids, k_values, task_lens,
        Val(largest), Val(threads_per_block), Val(hist_len), Val(unroll), Val(warp_size);
        ndrange=num_tasks * threads_per_block
    )

    KA.synchronize(backend)

    return bin_ids, k_values, task_lens
end
