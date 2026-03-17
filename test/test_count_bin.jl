using RadiK: 
    _compute_hist_len,
    sample_scaler,
    apply_scaling,
    get_bin_id,
    count_bin_ex_kernel!,
    count_bin_kernel!


@testset "Test count_bin.jl" begin

    # Test data: 3 tasks with varying sizes
    scale = 10
    task1_data = Float32.(1:(20_000 * scale))
    task2_data = Float32.((50 * scale):(80 * scale))
    task3_data = Float32.((100 * scale):(200 * scale))
    data_array = [task1_data, task2_data, task3_data]
    all_data = vcat(data_array...)
    task_offsets = vcat([0], cumsum(length.(data_array))) .|> Int32
    num_tasks = length(data_array)

    # Parameters
    LEFT = 0
    RIGHT = 30
    hist_len = _compute_hist_len(Float32, RIGHT)

    # ========================================
    # Test count_bin_ex_kernel!
    # ========================================

    @testset "count_bin_ex_kernel!" begin
        # Ground truth from CPU calculation
        histogram_scaled_cpu = []
        for data in data_array
            hist = zeros(Int32, hist_len)
            scaler = sample_scaler(data, 1, Val(true))
            for val in data
                scaled_val = apply_scaling(val, scaler, Val(true), Val(true))
                bin_id = get_bin_id(scaled_val, Val(LEFT), Val(RIGHT))
                hist[bin_id + 1] += 1
            end
            push!(histogram_scaled_cpu, hist)
        end

        data_gpu = adapt(backend, all_data)
        histogram_gpu = KA.zeros(backend, Int32, hist_len, num_tasks)
        task_offsets_gpu = adapt(backend, task_offsets)

        for threads_per_block in (4, 8, 64, 256, 1024), blocks_x in (2, 16, 32, 128)
        # for threads_per_block in (1024), blocks_x in (16)
            
            histogram_gpu .= 0

            count_bin_ex_kernel!(backend, threads_per_block)(
                histogram_gpu, data_gpu, task_offsets_gpu,
                Val(LEFT), Val(RIGHT), Val(hist_len), Val(4), Val(true), Val(true);
                ndrange=(threads_per_block * blocks_x, num_tasks)
            )

            for i in 1:num_tasks
                @test histogram_scaled_cpu[i] == Array(histogram_gpu)[:, i]
            end
        end
    end

    # ========================================
    # Test count_bin_kernel!
    # ========================================
    @testset "count_bin_kernel!" begin
        # Ground truth from CPU calculation
        histogram_cpu = []
        for data in data_array
            hist = zeros(Int32, hist_len)
            for v in data
                bin_id = get_bin_id(v, Val(LEFT), Val(RIGHT))
                hist[bin_id + 1] += 1
            end
            push!(histogram_cpu, hist)
        end

        histogram_gpu = KA.zeros(backend, Int32, hist_len, num_tasks)

        task_lens = adapt(backend, length.(data_array) .|> Int32)
        stride = maximum(task_lens)

        data_padded_cpu = zeros(Float32, length(task_lens) * stride)

        for (i, a) in enumerate(data_array)
            offset = (i-1) * stride
            data_padded_cpu[offset + 1: offset + length(a) ] .= a
        end

        data_padded_gpu = adapt(backend, data_padded_cpu)

        for threads_per_block in (1, 8, 64, 256, 1024)
        
            histogram_gpu .= 0

            n = length(data_padded_gpu)
            num_blocks = cld(n, threads_per_block)

            count_bin_kernel!(backend, threads_per_block)(
                histogram_gpu, data_padded_gpu, task_lens, stride,
                Val(LEFT), Val(RIGHT), Val(hist_len),;
                ndrange=(threads_per_block * num_blocks, num_tasks)
            )

            for i in 1:num_tasks
                @test histogram_cpu[i] == Array(histogram_gpu)[:, i]
            end
        end
    end

end # @testset "Test count_bin.jl"