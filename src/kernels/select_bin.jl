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

# Arguments
- `counts`: Thread's local bin counts
- `old_k`: k value to find
- `neighbor`: Previous thread's cumulative count (exclusive)
- `sum`: This thread's cumulative count (inclusive)
- `thread_x`: Thread ID in block (1-indexed)
- `UNROLL`: Bins per thread
- `task_id`: Task ID (1-indexed)
- `bin_ids`: Output array for selected bin IDs
- `k_values`: Output array for k values
- `task_lens`: Output array for counts
- `Val{LARGEST}`: Search direction (true=backward, false=forward)
"""
@inline function find_bin_and_write!(
    counts, 
    old_k, 
    neighbor, 
    sum, 
    thread_x,
    task_id, 
    bin_ids, 
    k_values, 
    task_lens, 
    ::Val{UNROLL},
    ::Val{LARGEST}
) where {LARGEST, UNROLL}
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
    bin_ids,
    k_values,
    task_lens::AbstractArray{T},
    histogram::AbstractArray{T, 2},
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
        counts, old_k, neighbor, sum, thread_x,
        task_id, bin_ids, k_values, task_lens, 
        Val(UNROLL), Val(LARGEST)
    )

end
