using RadiK:
    cross_warp_reduction!,
    compute_neighbor,
    find_bin_and_write!,
    select_bin!

@testset "Test select_bin.jl" begin

    # ==============================================================================
    # Unit Tests for Helper Functions
    # ==============================================================================

    @testset "cross_warp_reduction!" begin
        @kernel function test_cross_warp_reduction_kernel!(
            output::AbstractArray{T},
            input::AbstractArray{T},
            ::Val{LARGEST},
            ::Val{BLOCK},
            ::Val{WARP_SIZE}
        ) where {T, LARGEST, BLOCK, WARP_SIZE}

            thread_x = @index(Local, Linear)           # 1-indexed: 1 to BLOCK
            warp_id = (thread_x - 1) ÷ WARP_SIZE       # 0-indexed: 0 to NUM_WARPS-1
            NUM_WARPS = BLOCK ÷ WARP_SIZE
            sum = cross_warp_reduction!(
                input, warp_id, thread_x, input[thread_x],
                Val(LARGEST), Val(NUM_WARPS), Val(WARP_SIZE)
            )
            @inbounds output[thread_x] = sum
        end

        # LARGEST = true
        BLOCK = 12
        input = adapt(backend, Int32.(BLOCK:-1:1))
        output = KA.zeros(backend, Int32, BLOCK)
        expected = Int32.(BLOCK:-1:1) .+ Int32.(vcat(fill(8+4, 4), fill(4, 4), fill(0, 4)))
        
        test_cross_warp_reduction_kernel!(backend)(
            output, input, Val(true), Val(BLOCK), Val(4), ndrange=BLOCK)
        KA.synchronize(backend)

        @test Array(output) == expected

        # LARGEST = false
        BLOCK = 12
        input = adapt(backend, Int32.(1:BLOCK))
        output = KA.zeros(backend, Int32, BLOCK)
        expected = Int32.(1:BLOCK) .+ Int32.(vcat(fill(0, 4), fill(4, 4), fill(8+4, 4)))

        test_cross_warp_reduction_kernel!(backend)(
            output, input, Val(false), Val(BLOCK), Val(4), ndrange=BLOCK)
        KA.synchronize(backend)

        @test Array(output) == expected
    end

    # ========================================
    # Test compute_neighbor (OK)
    # ========================================
    @testset "compute_neighbor" begin
        BLOCK = 3
        prefix_sum_largest = [9, 5, 2]
        expected_largest = [5, 2, 0]
        T = eltype(prefix_sum_largest)
        @test compute_neighbor.(Ref(prefix_sum_largest), 1:3, Val(true), Val(BLOCK), T) == expected_largest

        prefix_sum_smallest = [2, 5, 9]
        expected_smallest = [0, 2, 5]
        @test compute_neighbor.(Ref(prefix_sum_smallest), 1:3, Val(false), Val(BLOCK), T) == expected_smallest

        # Single thread
        @test compute_neighbor([42], 1, Val(true), Val(1), T) == 0
        @test compute_neighbor([42], 1, Val(false), Val(1), T) == 0
    end

    # ========================================
    # Test find_bin_and_write!
    # ========================================
    @testset "find_bin_and_write! - k in bins" begin
        UNROLL = 4
        thread_x = 2
        task_id = 1

        # Test case: counts = [3, 5, 2, 8], k=10
        
        counts = [3, 5, 2, 8]
        old_k = 10
        neighbor = 5
        sum = 23

        # ======== CASE LARGEST ========
        # Backward search: check bin 4 (8), then bin 3 (2), then bin 2 (5)
        # 8 >= (10 - 5), found in bin 4
        bin_ids = zeros(Int, 1)
        k_values = zeros(Int, 1)
        task_lens = zeros(Int, 1)

        find_bin_and_write!(counts, old_k, neighbor, sum, thread_x, UNROLL,
                        task_id, bin_ids, k_values, task_lens, Val(true))

        @test bin_ids[1] == (thread_x - 1) * UNROLL + 4
        @test k_values[1] == 5
        @test task_lens[1] == 8

        # ======== CASE SMALLEST ========
        # Forward search: check bin 1 (3), then bin 2 (5), etc
        #  5 >= (10 - 5 - 3)=2, found in bin 2
        bin_ids = zeros(Int, 1)
        k_values = zeros(Int, 1)
        task_lens = zeros(Int, 1)

        find_bin_and_write!(counts, old_k, neighbor, sum, thread_x, UNROLL,
                        task_id, bin_ids, k_values, task_lens, Val(false))

        @test bin_ids[1] == (thread_x - 1) * UNROLL + 2
        @test k_values[1] == 2
        @test task_lens[1] == 5
    end

    @testset "find_bin_and_write! - Not in range" begin
        UNROLL = 4
        thread_x = 2
        task_id = 1

        counts = [3, 5, 2, 8]
        old_k = 24  # Outside range
        neighbor = 10
        sum = 23

        bin_ids = zeros(Int, 2)
        k_values = zeros(Int, 2)
        task_lens = zeros(Int, 2)

        find_bin_and_write!(counts, old_k, neighbor, sum, thread_x, UNROLL,
                        task_id, bin_ids, k_values, task_lens, Val(true))

        # Should not write anything (k is out of range)
        @test bin_ids[1] == 0
        @test k_values[1] == 0
        @test task_lens[1] == 0
    end

    @testset "find_bin_and_write! - Boundary cases" begin
        UNROLL = 3
        thread_x = 1
        task_id = 1

        # k=1 (first element)
        counts = Int32[5, 3, 2]
        old_k = 1
        neighbor = 0
        sum = 10
        LARGEST = false

        bin_ids = zeros(Int, 1)
        k_values = zeros(Int, 1)
        task_lens = zeros(Int, 1)

        find_bin_and_write!(counts, old_k, neighbor, sum, 1, 3,
                        1, bin_ids, k_values, task_lens, Val(LARGEST))

        @test bin_ids[1] == 1
        @test k_values[1] == 1
        @test task_lens[1] == 5

        # k equals total sum (last element)
        counts = Int32[5, 3, 2]
        old_k = 10
        neighbor = 0
        sum = 10
        LARGEST = false

        bin_ids = zeros(Int, 1)
        k_values = zeros(Int, 1)
        task_lens = zeros(Int, 1)

        find_bin_and_write!(counts, old_k, neighbor, sum, 1, 3,
                        1, bin_ids, k_values, task_lens, Val(LARGEST))

        @test bin_ids[1] == 3
        @test k_values[1] == 2
        @test task_lens[1] == 2
    end

    # ==============================================================================
    # Integration Tests
    # ==============================================================================

    @testset "select_bin! - Largest elements" begin
        # Each of 10 bins has exactly 1 element
        histogram = zeros(Int32, 256, 1)
        histogram[1:4, 1] .= [2, 8, 14, 3]
        histogram_gpu = adapt(backend, histogram)

        bin_ids_gpu = KA.zeros(backend, Int32, 1)
        k_values_gpu = adapt(backend, Int32[5])     # Find 5th largest
        task_lens_gpu = KA.zeros(backend, Int32, 1)

        select_bin!(histogram_gpu, bin_ids_gpu, k_values_gpu, task_lens_gpu;
            largest=true, threads_per_block=256, warp_size=WARP_SIZE)

        bin_id = Array(bin_ids_gpu)[1]
        k_value = Array(k_values_gpu)[1]
        task_len = Array(task_lens_gpu)[1]

        @test bin_id == 3
        @test k_value == 2
        @test task_len == 14
    end

    @testset "select_bin! - Smallest elements" begin
        # Each of 10 bins has exactly 1 element
        histogram = zeros(Int32, 256, 1)
        histogram[1:4, 1] .= [2, 8, 14, 3]
        histogram_gpu = adapt(backend, histogram)
        
        bin_ids_gpu = KA.zeros(backend, Int32, 1)
        k_values_gpu = adapt(backend, Int32[5])     # Find 5th smallest
        task_lens_gpu = KA.zeros(backend, Int32, 1)

        select_bin!(histogram_gpu, bin_ids_gpu, k_values_gpu, task_lens_gpu;
            largest=false, threads_per_block=256, warp_size=WARP_SIZE)

        bin_id = Array(bin_ids_gpu)[1]
        k_value = Array(k_values_gpu)[1]
        task_len = Array(task_lens_gpu)[1]

        @test bin_id == 2
        @test k_value == 3
        @test task_len == 8
    end

    @testset "select_bin! - Various block sizes" begin
        for threads_per_block in [32, 64, 128, 256]
            histogram = zeros(Int32, 256, 1)
            histogram[1:10, 1] .= 1

            histogram_gpu = adapt(backend, histogram)
            bin_ids_gpu = KA.zeros(backend, Int32, 1)
            k_values_gpu = adapt(backend, Int32[5])
            task_lens_gpu = KA.zeros(backend, Int32, 1)

            select_bin!(histogram_gpu, bin_ids_gpu, k_values_gpu, task_lens_gpu;
                largest=true, threads_per_block=threads_per_block, warp_size=WARP_SIZE)

            bin_id = Array(bin_ids_gpu)[1]
            k_value = Array(k_values_gpu)[1]
            task_len = Array(task_lens_gpu)[1]

            @test bin_id == 6
            @test k_value == 1
            @test task_len == 1
        end
    end

    @testset "select_bin! - Multiple tasks" begin
        num_tasks = 3

        # Task 1: first 10 bins have 1 element each
        # Task 2: bins 100-109 have 2 elements each
        # Task 3: bins 200-205 have 3 elements each
        histogram = zeros(Int32, 256, num_tasks)
        histogram[1:10, 1] .= 1
        histogram[101:110, 2] .= 2
        histogram[201:206, 3] .= 3

        histogram_gpu = adapt(backend, histogram)
        bin_ids_gpu = KA.zeros(backend, Int32, num_tasks)
        k_values_gpu = adapt(backend, Int32[5, 7, 8])
        task_lens_gpu = KA.zeros(backend, Int32, num_tasks)

        select_bin!(histogram_gpu, bin_ids_gpu, k_values_gpu, task_lens_gpu;
            largest=true, threads_per_block=256, warp_size=WARP_SIZE)

        bin_ids = Array(bin_ids_gpu)
        k_values = Array(k_values_gpu)
        task_lens = Array(task_lens_gpu)

        @test bin_ids[1] == 6
        @test k_values[1] == 1
        @test task_lens[1] == 1

        @test bin_ids[2] == 107
        @test k_values[2] == 1
        @test task_lens[2] == 2

        @test bin_ids[3] == 204
        @test k_values[3] == 2
        @test task_lens[3] == 3
    end

    @testset "select_bin! - Edge cases" begin
        # Test k=1 (first element)
        histogram = zeros(Int32, 256, 1)
        histogram[1, 1] = 5  # All 5 elements in bin 0

        histogram_gpu = adapt(backend, histogram)
        bin_ids_gpu = KA.zeros(backend, Int32, 1)
        k_values_gpu = adapt(backend, Int32[3])
        task_lens_gpu = KA.zeros(backend, Int32, 1)

        select_bin!(histogram_gpu, bin_ids_gpu, k_values_gpu, task_lens_gpu;
            largest=true, threads_per_block=256, warp_size=WARP_SIZE)

        @test Array(bin_ids_gpu)[1] == 1
        @test Array(k_values_gpu)[1] == 3
        @test Array(task_lens_gpu)[1] == 5
    end

end # @testset "Test select_bin.jl"