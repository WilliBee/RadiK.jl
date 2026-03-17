using RadiK:
    filter_kernel!,
    filter_general_kernel!,
    filter_!,
    filter_general!

@testset "Test filter.jl" begin

    @testset "Tiny dataset" begin
        data = Float32.(1:1000)
        K = Int32(1000)  # K > N, should copy all

        # Setup GPU arrays
        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[0])  # Not used when N ≤ K
        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        # Call the kernel
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true
        )

        # Verify results
        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]
        filtered_idxs = Array(idx_out)[1:count]

        @test count == length(data)
        @test Set(filtered_vals) == Set(data)
        @test Set(filtered_idxs) == Set(Int32.(1:1000))
    end

    @testset "filter! - Single task - Largest" begin
        data = Float32.(1:1000)
        K = Int32(5)

        # With scaling, kth_element is already scaled
        scaler = data[1]
        scaled_kth = data[996] - scaler

        # Setup GPU arrays
        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[scaled_kth])

        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        # Scaling - boundary_counts = 0
        boundary_counts = adapt(backend, Int32[0])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 4
        @test Set(filtered_vals) == Set(Float32[1000, 999, 998, 997])

        # Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 5
        @test Set(filtered_vals) == Set(Float32[1000, 999, 998, 997, 996])

        # No Scaling - boundary_counts = 0
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 5
        @test Set(filtered_vals) == Set(Float32[1000, 999, 998, 997, 996])
    end

    @testset "filter! - Single task - Smallest" begin
        data = Float32.(1:1000)
        K = Int32(5)

        # With scaling, kth_element is already scaled
        scaler = data[1]
        scaled_kth = data[5] - scaler

        # Setup GPU arrays
        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[scaled_kth])

        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        # Scaling - boundary_counts = 0
        boundary_counts = adapt(backend, Int32[0])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 4
        @test Set(filtered_vals) == Set(Float32[1, 2, 3, 4])

        # Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 5
        @test Set(filtered_vals) == Set(Float32[1, 2, 3, 4, 5])

        # No Scaling - boundary_counts = 0
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 3
        @test Set(filtered_vals) == Set(Float32[1, 2, 3])

        # No Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == 4
        @test Set(filtered_vals) == Set(Float32[1, 2, 3, 4])
    end

    # ==============================================================================
    # Test: filter! - Multiple tasks
    # ==============================================================================

    @testset "filter! - Multiple tasks" begin
        task1_data = Float32[1, 2, 3, 4, 5]
        task2_data = Float32[10, 20, 30, 40, 50]
        task3_data = Float32[100, 200, 300, 400, 500]

        all_data = vcat(task1_data, task2_data, task3_data)
        task_offsets = Int32[0, length(task1_data), length(task1_data) + length(task2_data), length(all_data)]

        K = Int32(2)

        # K-th elements (1-indexed: 2nd element in each task)
        kth_elements = Float32[
            task1_data[4],
            task2_data[4],
            task3_data[4]
        ]

        # Setup GPU arrays
        data_in = adapt(backend, all_data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, kth_elements)
        val_out = KA.zeros(backend, Float32, Int(K), 3)
        idx_out = KA.zeros(backend, Int32, Int(K), 3)
        global_counts = KA.zeros(backend, Int32, 3)
        boundary_counts = adapt(backend, Int32[K, K, K])
        task_offsets_gpu = adapt(backend, task_offsets)
        stride = Int32(maximum(length.([task1_data, task2_data, task3_data])))

        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets_gpu, stride, K;
            with_scale=false, largest=true
        )

        counts = Array(global_counts)
        filtered_data = Array(val_out)

        @test counts[1] == 2
        @test counts[2] == 2
        @test counts[3] == 2

        # Extract per-task outputs
        task1_output = Array(filtered_data[1:counts[1], 1])
        task2_output = Array(filtered_data[1:counts[2], 2])
        task3_output = Array(filtered_data[1:counts[3], 3])

        # Verify each task has correct elements
        @test Set(task1_output) == Set(Float32[5, 4])
        @test Set(task2_output) == Set(Float32[50, 40])
        @test Set(task3_output) == Set(Float32[500, 400])
    end

    # ==============================================================================
    # Test: filter_general! - Large K - Copy all elements
    # ==============================================================================

    @testset "filter_general! - Large K (K > 1024)" begin
        data = Float32.(1:2000)
        K = Int32(2001)

        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[500])
        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[10])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == length(data)
        @test Set(filtered_vals) == Set(data)
    end

    @testset "filter_general! - Large K (K > 1024)" begin
        data = Float32.(1:2000)
        K = Int32(1500)

        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[500])
        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == K
        @test Set(filtered_vals) == Set(data[501:2000])
    end

    @testset "filter_general! - Single task - Largest" begin
        data = Float32.(1:10000)
        K = Int32(2000)

        # With scaling, kth_element is already scaled
        scaler = data[1]
        scaled_kth = data[8001] - scaler

        # Setup GPU arrays
        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[scaled_kth])

        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        # Scaling - boundary_counts = 0
        boundary_counts = adapt(backend, Int32[0])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == K - 1 # boundary_count = 0
        @test Set(filtered_vals) == Set(data[8002:10000])

        # Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == K
        @test Set(filtered_vals) == Set(data[8001:10000])

        # No Scaling - boundary_counts = 0
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == K
        @test Set(filtered_vals) == Set(data[8001:10000])
    end

    @testset "filter_general! - Single task - Smallest" begin
        data = Float32.(1:10000)
        K = Int32(2000)

        # With scaling, kth_element is already scaled
        scaler = data[1]
        scaled_kth = data[2000] - scaler

        # Setup GPU arrays
        data_in = adapt(backend, data)
        idx_in = adapt(backend, Int32[])
        kth_element = adapt(backend, Float32[scaled_kth])

        val_out = KA.zeros(backend, Float32, 1, Int(K))
        idx_out = KA.zeros(backend, Int32, 1, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        # Scaling - boundary_counts = 0
        boundary_counts = adapt(backend, Int32[0])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1, 1:count]

        @test count == K - 1
        @test Set(filtered_vals) == Set(data[1:(K - 1)])

        # Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=true, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1, 1:count]

        @test count == K
        @test Set(filtered_vals) == Set(data[1:K])

        # No Scaling - boundary_counts = 0
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[0])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1, 1:count]

        @test count == K - 2
        @test Set(filtered_vals) == Set(data[1:(K-2)])

        # No Scaling - boundary_counts = 1
        val_out .= 0
        idx_out .= 0
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[1])
        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=false
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]

        @test count == K - 1
        @test Set(filtered_vals) == Set(data[1:(K-1)])
    end

    # ==============================================================================
    # Test: Various block sizes
    # ==============================================================================
    @testset "Various block sizes - Single task" begin
        data = randn(Float32, 10_000_000) # Float32.(1:100_000_000)
        K = Int32(1000)
        num_tasks = 1
        PACKSIZE = 4
        LARGEST = false
        WITHSCALE = true
        WITHIDXIN = false

        for threads_per_block in (32, 64, 128, 256), block_x in (16, 32, 64),
            LARGEST in (true, false), WITHSCALE in (true, false), WITHIDXIN in (true, false), GENERAL in (true, false)

            kth_element_cpu = partialsort(data, 1:K, rev=LARGEST)[K:K] .- (WITHSCALE ? data[1] : zero(Float32))

            data_in = adapt(backend, data)
            idx_in = adapt(backend, Int32.(1:length(data)))
            kth_element = adapt(backend, kth_element_cpu)
            val_out = KA.zeros(backend, Float32, Int(K))
            idx_out = KA.zeros(backend, Int32, 1, Int(K))
            global_counts = adapt(backend, Int32[0])
            boundary_counts = adapt(backend, Int32[1])
            task_offsets = adapt(backend, Int32[0, length(data)])
            stride = Int32(length(data))

            if GENERAL
                filter_general_kernel!(backend, threads_per_block)(
                    data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
                    task_offsets, stride, K,
                    Val(0), Val(20), Val(threads_per_block), Val(PACKSIZE),
                    Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN);
                    ndrange=(threads_per_block * block_x, num_tasks)
                )
            else
                filter_kernel!(backend, threads_per_block)(
                    data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
                    task_offsets, stride, K,
                    Val(0), Val(20), Val(threads_per_block), Val(PACKSIZE), Val(1024),
                    Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN);
                    ndrange=(threads_per_block * block_x, num_tasks)
                )
            end
            KA.synchronize(backend)

            count = Array(global_counts)[1]
            @test count == K
            @test Set(Array(val_out)) == Set(partialsort(data, 1:K, rev=LARGEST))
            @test Set(Array(idx_out)) == Set(partialsortperm(data, 1:K, rev=LARGEST))
        end
    end

    @testset "Various block sizes" begin
        data = randn(Float32, 10_000_000) # Float32.(1:100_000_000)
        K = 3
        num_tasks = 3
        PACKSIZE = 4

        task1_data = randn(Float32, 9_000_000)
        task2_data = randn(Float32, 10_000_100)
        task3_data = randn(Float32, 8_888_888)

        data_array = [task1_data, task2_data, task3_data]
        data = vcat(data_array...)
        idx = Int32.(vcat(1:9_000_000, 1:10_000_100, 1:8_888_888))
        task_offsets_cpu = Int32[0, cumsum(length.(data_array))...]

        for threads_per_block in (32, 64, 128, 256), block_x in (16, 32, 64),
            LARGEST in (true, false), WITHSCALE in (true, false), WITHIDXIN in (true, false), GENERAL in (true, false)

            kth_element_cpu = Float32[
                partialsort(task1_data, 1:K, rev=LARGEST)[K] - (WITHSCALE ? task1_data[1] : zero(Float32)),
                partialsort(task2_data, 1:K, rev=LARGEST)[K] - (WITHSCALE ? task2_data[1] : zero(Float32)),
                partialsort(task3_data, 1:K, rev=LARGEST)[K] - (WITHSCALE ? task3_data[1] : zero(Float32))
            ]

            data_in = adapt(backend, data)
            idx_in = adapt(backend, idx)
            kth_element = adapt(backend, kth_element_cpu)
            val_out = KA.zeros(backend, Float32, K, num_tasks)
            idx_out = KA.zeros(backend, Int32, K, num_tasks)
            global_counts = KA.zeros(backend, Int32, 3)
            boundary_counts = adapt(backend, Int32[1, 1, 1])
            task_offsets = adapt(backend, task_offsets_cpu)
            stride = Int32(maximum(length.([task1_data, task2_data, task3_data])))

            if GENERAL
                filter_general_kernel!(backend, threads_per_block)(
                    data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
                    task_offsets, stride, Int32(K),
                    Val(0), Val(20), Val(threads_per_block), Val(PACKSIZE),
                    Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN);
                    ndrange=(threads_per_block * block_x, num_tasks)
                )
            else
                filter_kernel!(backend, threads_per_block)(
                    data_in, idx_in, kth_element, val_out, idx_out, global_counts, boundary_counts,
                    task_offsets, stride, Int32(K),
                    Val(0), Val(20), Val(threads_per_block), Val(PACKSIZE), Val(1024),
                    Val(WITHSCALE), Val(LARGEST), Val(WITHIDXIN);
                    ndrange=(threads_per_block * block_x, num_tasks)
                )
            end
            KA.synchronize(backend)

            @test all(Array(global_counts) .== K)
            
            for (i, task_data) in enumerate(data_array)
                @test Set(Array(val_out)[:, i]) == Set(partialsort(task_data, 1:K, rev=LARGEST))
                @test Set(Array(idx_out)[:, i]) == Set(partialsortperm(task_data, 1:K, rev=LARGEST))
            end
        end
    end


    # ==============================================================================
    # Test: With input indices (WITHIDXIN)
    # ==============================================================================

    @testset "filter! - WITHIDXIN=true" begin
        data = Float32[100, 85, 92, 78, 95, 88, 90, 82]
        indices = Int32[10, 11, 12, 13, 14, 15, 16, 17]
        K = Int32(3)

        data_in = adapt(backend, data)
        idx_in = adapt(backend, indices)
        kth_element = adapt(backend, Float32[92])
        val_out = KA.zeros(backend, Float32, Int(K))
        idx_out = KA.zeros(backend, Int32, Int(K))
        global_counts = adapt(backend, Int32[0])
        boundary_counts = adapt(backend, Int32[K])
        task_offsets = adapt(backend, Int32[0, length(data)])
        stride = Int32(length(data))

        filter_!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true, with_idx_in=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]
        filtered_idxs = Array(idx_out)[1:count]

        @test count == 3
        @test Set(filtered_vals) == Set(Float32[100, 95, 92])

        # Indices should match the input indices
        expected_indices = Int32[10, 14, 12]  # Indices of 100, 95, 92
        @test Set(filtered_idxs) == Set(expected_indices)

        # ----------------
        # filter_general!
        # ----------------
        val_out .= 0
        idx_out .= 0
        global_counts .= 0
        boundary_counts = adapt(backend, Int32[K])

        filter_general!(
            data_in, idx_in, kth_element, val_out, idx_out,
            global_counts, boundary_counts, task_offsets, stride, K;
            with_scale=false, largest=true, with_idx_in=true
        )

        count = Array(global_counts)[1]
        filtered_vals = Array(val_out)[1:count]
        filtered_idxs = Array(idx_out)[1:count]

        @test count == 3
        @test Set(filtered_vals) == Set(Float32[100, 95, 92])

        # Indices should match the input indices
        expected_indices = Int32[10, 14, 12]  # Indices of 100, 95, 92
        @test Set(filtered_idxs) == Set(expected_indices)

    end

    # ==============================================================================
    # Test: Edge cases
    # ==============================================================================

    @testset "filter! - Edge cases" begin
        # Empty result
        @testset "Empty result" begin
            data = Float32[1, 2, 3]
            K = Int32(2)

            data_in = adapt(backend, data)
            idx_in = adapt(backend, Int32[])
            kth_element = adapt(backend, Float32[100])  # No elements > 100
            val_out = KA.zeros(backend, Float32, Int(K))
            idx_out = KA.zeros(backend, Int32, Int(K))
            global_counts = adapt(backend, Int32[0])
            boundary_counts = adapt(backend, Int32[0])
            task_offsets = adapt(backend, Int32[0, length(data)])
            stride = Int32(length(data))

            filter_!(
                data_in, idx_in, kth_element, val_out, idx_out,
                global_counts, boundary_counts, task_offsets, stride, K;
                with_scale=false, largest=true
            )

            count = Array(global_counts)[1]
            @test count == 0

            filter_general!(
                data_in, idx_in, kth_element, val_out, idx_out,
                global_counts, boundary_counts, task_offsets, stride, K;
                with_scale=false, largest=true
            )

            count = Array(global_counts)[1]
            @test count == 0
        end

        # All elements equal
        @testset "All elements equal" begin
            data = Float32[50, 50, 50, 50, 50]
            K = Int32(3)

            data_in = adapt(backend, data)
            idx_in = adapt(backend, Int32[])
            kth_element = adapt(backend, Float32[50])
            val_out = KA.zeros(backend, Float32, Int(K))
            idx_out = KA.zeros(backend, Int32, Int(K))
            global_counts = adapt(backend, Int32[0])
            boundary_counts = adapt(backend, Int32[3])  # Take 3 boundary elements
            task_offsets = adapt(backend, Int32[0, length(data)])
            stride = Int32(length(data))

            filter_!(
                data_in, idx_in, kth_element, val_out, idx_out,
                global_counts, boundary_counts, task_offsets, stride, K;
                with_scale=false, largest=true
            )

            count = Array(global_counts)[1]
            filtered_vals = Array(val_out)[1:count]

            @test count == 3
            @test all(v == 50 for v in filtered_vals)

            # filter_general!
            val_out .= 0
            idx_out .= 0
            global_counts .= 0
            boundary_counts = adapt(backend, Int32[3])

            filter_general!(
                data_in, idx_in, kth_element, val_out, idx_out,
                global_counts, boundary_counts, task_offsets, stride, K;
                with_scale=false, largest=true
            )

            count = Array(global_counts)[1]
            filtered_vals = Array(val_out)[1:count]

            @test count == 3
            @test all(v == 50 for v in filtered_vals)
        end
    end

end # @testset "Test filter.jl"