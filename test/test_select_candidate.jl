using RadiK:
    select_candidate_kernel!,
    select_candidate_ex_kernel!,
    get_bin_id

@testset "Test select_candidate.jl" begin

    # ==============================================================================
    # select_candidate!
    # ==============================================================================

    @testset "select_candidate_kernel! - Single task simple filter" begin
        data = Float32.(1:10)
        data_in = adapt(backend, data)
        data_out = KA.zeros(backend, Float32, length(data))
        global_counts = adapt(backend, Int32[0])
        task_lens = adapt(backend, Int32[length(data)])
        stride = Int32(length(data))
        LEFT = 0
        RIGHT = 23
        threads_per_block = 4

        # Get the bin ID for value 3.0 (the value we want to filter)
        target_bin = get_bin_id(Float32(3.0), Val(LEFT), Val(RIGHT)) + 1    # 1-based
        bin_ids = adapt(backend, Int32[target_bin])

        num_tasks = length(task_lens)
        max_task_len = maximum(Array(task_lens))
        num_blocks = cld(max_task_len, threads_per_block)

        data_out .= 0
        global_counts .= 0

        kernel! = select_candidate_kernel!(backend, threads_per_block)
        kernel!(
            data_out, global_counts, data_in, bin_ids, task_lens, stride,
            Val(LEFT), Val(RIGHT), Val(threads_per_block);
            ndrange=(num_blocks * threads_per_block, num_tasks)
        )
        KA.synchronize(backend)

        count = Array(global_counts)[1]
        filtered_data = Array(data_out)


        @test count == 2
        @test Set(filtered_data[1:count]) == Set([2.0, 3.0])
    end

    @testset "select_candidate_kernel! - Multiple tasks" begin
        # Task 1: values 1-10
        # Task 2: values 11-20
        # Task 3: values 21-30
        num_tasks = 3
        task1_data = Float32.(1:10)
        task2_data = Float32.(11:20)
        task3_data = Float32.(21:30)

        # Create padded data layout
        stride = 10
        data_padded = zeros(Float32, num_tasks * stride)
        data_padded[1:10] .= task1_data
        data_padded[11:20] .= task2_data
        data_padded[21:30] .= task3_data

        data_in = adapt(backend, data_padded)
        data_out = KA.zeros(backend, Float32, length(data_padded))
        global_counts = KA.zeros(backend, Int32, num_tasks)
        task_lens = adapt(backend, Int32[10, 10, 10])

        # Each task wants to filter a different bin
        # Task 1: filter bin for value 3
        # Task 2: filter bin for value 15
        # Task 3: filter bin for value 25
        bin1_id = get_bin_id(Float32(3.0), Val(0), Val(20)) + 1
        bin2_id = get_bin_id(Float32(15.0), Val(0), Val(20)) + 1
        bin3_id = get_bin_id(Float32(25.0), Val(0), Val(20)) + 1
        bin_ids = adapt(backend, Int32[bin1_id, bin2_id, bin3_id])

        LEFT = 0
        RIGHT = 20
        threads_per_block = 8

        max_task_len = maximum(Array(task_lens))
        num_blocks = cld(max_task_len, threads_per_block)

        data_out .= 0
        global_counts .= 0

        kernel! = select_candidate_kernel!(backend, threads_per_block)
        kernel!(
            data_out, global_counts, data_in, bin_ids, task_lens, Int32(stride),
            Val(LEFT), Val(RIGHT), Val(threads_per_block);
            ndrange=(num_blocks * threads_per_block, num_tasks)
        )
        KA.synchronize(backend)

        counts = Array(global_counts)
        bin_ids_cpu = Array(bin_ids)
        filtered_data = Array(data_out)

        # Each task should have found the same count as CPU
        @test counts[1] == count(==(bin_ids_cpu[1]), Int32.(get_bin_id.(task1_data, Val(0), Val(20)) .+ 1))
        @test counts[2] == count(==(bin_ids_cpu[2]), Int32.(get_bin_id.(task2_data, Val(0), Val(20)) .+ 1))
        @test counts[3] == count(==(bin_ids_cpu[3]), Int32.(get_bin_id.(task3_data, Val(0), Val(20)) .+ 1))

        # Check target values are in respective task outputs
        task1_output = filtered_data[1:counts[1]]
        task2_output = filtered_data[stride .+ (1:counts[2])]
        task3_output = filtered_data[2*stride .+ (1:counts[3])]

        @test 3.0f0 in task1_output
        @test 15.0f0 in task2_output
        @test 25.0f0 in task3_output
    end

    @testset "select_candidate_kernel! - Various block sizes" begin
        data = Float32.(1:20)
        data_in = adapt(backend, data)
        data_out = KA.zeros(backend, Float32, length(data))
        global_counts = adapt(backend, Int32[0])
        task_lens = adapt(backend, Int32[length(data)])
        stride = Int32(length(data))

        # Target value 10
        target_bin = get_bin_id(Float32(10.0), Val(0), Val(20)) + 1
        bin_ids = adapt(backend, Int32[target_bin])

        for threads_per_block in [1, 2, 4, 8, 16, 32]
            num_tasks = length(task_lens)
            max_task_len = maximum(Array(task_lens))
            num_blocks = cld(max_task_len, threads_per_block)

            data_out .= 0
            global_counts .= 0

            kernel! = select_candidate_kernel!(backend, threads_per_block)
            kernel!(
                data_out, global_counts, data_in, bin_ids, task_lens, stride,
                Val(0), Val(20), Val(threads_per_block);
                ndrange=(num_blocks * threads_per_block, num_tasks)
            )
            KA.synchronize(backend)

            count = Array(global_counts)[1]
            filtered_data = Array(data_out)

            @test count >= 1
            @test 10.0f0 in filtered_data[1:count]
        end
    end

    @testset "select_candidate_kernel! - Edge cases" begin
        # Empty task (no elements)
        data_in = adapt(backend, Float32[])
        data_out = KA.zeros(backend, Float32, 0)
        global_counts = adapt(backend, Int32[0])
        task_lens = adapt(backend, Int32[0])
        stride = Int32(0)

        target_bin = get_bin_id(Float32(1.0), Val(0), Val(20)) + 1
        bin_ids = adapt(backend, Int32[target_bin])

        threads_per_block = 4
        num_tasks = length(task_lens)
        max_task_len = maximum(Array(task_lens))
        num_blocks = cld(max_task_len, threads_per_block)

        data_out .= 0
        global_counts .= 0

        # Should not crash
        kernel! = select_candidate_kernel!(backend, threads_per_block)
        kernel!(
            data_out, global_counts, data_in, bin_ids, task_lens, stride,
            Val(0), Val(20), Val(threads_per_block);
            ndrange=(num_blocks * threads_per_block, num_tasks)
        )
        KA.synchronize(backend)

        @test Array(global_counts)[1] == 0
    end

    @testset "select_candidate_kernel! - Correctness vs CPU" begin
        # Create test data with known distribution
        data = Float32.(1:30)
        stride = 30

        # Pick a target value and compute its bin
        target_value = Float32(15.0)
        target_bin = get_bin_id(target_value, Val(0), Val(20)) + 1

        # CPU reference implementation
        cpu_filtered = Float32[]
        for val in data
            if get_bin_id(val, Val(0), Val(20)) + 1 == target_bin
                push!(cpu_filtered, val)
            end
        end

        # GPU implementation
        data_in = adapt(backend, data)
        data_out = KA.zeros(backend, Float32, length(data))
        global_counts = adapt(backend, Int32[0])
        task_lens = adapt(backend, Int32[length(data)])
        bin_ids = adapt(backend, Int32[target_bin])

        threads_per_block = 8
        num_tasks = length(task_lens)
        max_task_len = maximum(Array(task_lens))
        num_blocks = cld(max_task_len, threads_per_block)

        data_out .= 0
        global_counts .= 0

        kernel! = select_candidate_kernel!(backend, threads_per_block)
        kernel!(
            data_out, global_counts, data_in, bin_ids, task_lens, Int32(stride),
            Val(0), Val(20), Val(threads_per_block);
            ndrange=(num_blocks * threads_per_block, num_tasks)
        )
        KA.synchronize(backend)

        gpu_count = Array(global_counts)[1]
        gpu_filtered = Array(data_out)[1:gpu_count]

        # Compare counts
        @test gpu_count == length(cpu_filtered)

        # Compare values (order may differ)
        @test Set(gpu_filtered) == Set(cpu_filtered)
    end

    # ==============================================================================
    # Tests for select_candidate_ex_kernel! (with vectorization and scaling)
    # ==============================================================================

    @testset "select_candidate_ex_kernel! - Multiple tasks, with/without scaling" begin
        # Create 3 tasks with different sizes
        task1_data = Float32.(1:2800)
        task2_data = Float32.(2801:5500)
        task3_data = Float32.(5501:8210)

        data_array = [task1_data, task2_data, task3_data]
        data_in = adapt(backend, vcat(data_array...))

        task_offsets = Int32.(vcat([0], cumsum(length.(data_array))))
        task_offsets_gpu = adapt(backend, task_offsets)

        data_out = KA.zeros(backend, Float32, maximum(length.(data_array)), 3)
        global_counts = KA.zeros(backend, Int32, 3)

        threads_per_block = 64
        blocks_x = 16
        pack_size = 4
        num_tasks = 3

        for SCALE in (true, false)
            offset_1 = SCALE ? first(task1_data) : 0
            offset_2 = SCALE ? first(task2_data) : 0
            offset_3 = SCALE ? first(task3_data) : 0
            bin_ids = Int32[
                get_bin_id(task1_data[10] - offset_1, Val(0), Val(20)) + 1,
                get_bin_id(task2_data[10] - offset_2, Val(0), Val(20)) + 1,
                get_bin_id(task3_data[10] - offset_3, Val(0), Val(20)) + 1,
            ]
            bin_ids_gpu = adapt(backend, bin_ids)

            data_out .= 0
            global_counts .= 0

            kernel! = select_candidate_ex_kernel!(backend, threads_per_block)
            kernel!(
                data_out, global_counts, data_in, bin_ids_gpu, task_offsets_gpu,
                Val(0), Val(20), Val(threads_per_block), Val(pack_size),
                Val(SCALE), Val(true);
                ndrange=(threads_per_block * blocks_x, num_tasks)
            )
            KA.synchronize(backend)

            counts = Array(global_counts)

            bins_1 = get_bin_id.(task1_data .- offset_1, Val(0), Val(20)) .+ 1 .|> Int32
            bins_2 = get_bin_id.(task2_data .- offset_2, Val(0), Val(20)) .+ 1 .|> Int32
            bins_3 = get_bin_id.(task3_data .- offset_3, Val(0), Val(20)) .+ 1 .|> Int32

            @test counts[1] == count(==(bin_ids[1]), bins_1)
            @test counts[2] == count(==(bin_ids[2]), bins_2)
            @test counts[3] == count(==(bin_ids[3]), bins_3)

            # Check target values are in respective task outputs
            filtered_data = Array(data_out)
            task1_output = filtered_data[1:counts[1], 1]
            task2_output = filtered_data[1:counts[2], 2]
            task3_output = filtered_data[1:counts[3], 3]

            @test Set(task1_output) == Set(task1_data[bins_1 .== bin_ids[1]] .- offset_1)
            @test Set(task2_output) == Set(task2_data[bins_2 .== bin_ids[2]] .- offset_2)
            @test Set(task3_output) == Set(task3_data[bins_3 .== bin_ids[3]] .- offset_3)
        end
    end

    @testset "select_candidate_ex_kernel! - Various pack sizes" begin
        data = Float32.(1:64)
        task_offsets = Int32[0, length(data)]

        data_in = adapt(backend, data)
        data_out = KA.zeros(backend, Float32, length(data))
        global_counts = KA.zeros(backend, Int32, 1)
        bin_ids = adapt(backend, Int32[get_bin_id(data[1], Val(0), Val(20)) + 1])
        task_offsets_gpu = adapt(backend, task_offsets)

        threads_per_block = 32
        blocks_x = 16
        num_tasks = 1

        for pack_size in [2, 4, 8, 16]
            data_out .= 0
            global_counts .= 0

            kernel! = select_candidate_ex_kernel!(backend, threads_per_block)
            kernel!(
                data_out, global_counts, data_in, bin_ids, task_offsets_gpu,
                Val(0), Val(20), Val(threads_per_block), Val(pack_size),
                Val(false), Val(true);
                ndrange=(threads_per_block * blocks_x, num_tasks)
            )
            KA.synchronize(backend)

            count = Array(global_counts)[1]
            @test count > 0
        end
    end

end # @testset "Test select_candidate.jl"