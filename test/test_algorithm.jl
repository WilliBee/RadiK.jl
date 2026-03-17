using Random

Random.seed!(42)

@testset "Test algorithm.jl" begin
    @testset "Basic functionality with topk wrapper" begin
        N_powers = (14, )
        K_values = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

        for (i, pow_2) in enumerate(N_powers)
            n = 2^pow_2

            for (j, k) in enumerate(K_values)
                for LARGEST in (true, false), REV in (true, false)
                    data = rand!(allocate(backend, Float32, n))
                    results, indices = topk(data, Int32(k); largest=LARGEST, rev=REV)

                    @test Array(results) == sort(partialsort(Array(data), 1:k, rev=LARGEST), rev=REV)

                    # Verify indices point to correct values
                    # (When values are duplicated, index order may differ from sortperm)
                    idx = Array(indices) .|> Int
                    data_array = Array(data)

                    if REV == LARGEST
                        expected_order = sortperm(data_array, rev=LARGEST)[1:k] .|> Int
                    else
                        expected_order = sortperm(data_array, rev=LARGEST)[k:-1:1] .|> Int
                    end

                    # Check that values at indices match (allows different index order for duplicates)
                    @test data_array[idx] == data_array[expected_order]

                    # Verify indices are valid: within bounds and unique
                    @test all(1 .<= idx .<= length(data))
                    @test length(unique(idx)) == length(idx)  # No duplicate indices
                end
            end
        end
    end

    @testset "Multitask" begin
        task_lens = [1000, 2000, 1500, 500, 5000]
        num_tasks = length(task_lens)
        k = Int32(64)

        total_len = sum(task_lens)
        data = rand!(allocate(backend, Float32, total_len))

        indices_in = adapt(backend, collect(Int32(1):Int32(total_len)))

        # Create output arrays (2D: k x num_tasks)
        result = KA.zeros(backend, Float32, Int(k), Int(num_tasks))
        indices_out = KA.zeros(backend, Int32, Int(k), Int(num_tasks))

        ws = RadiKWorkspace(backend, k, Int32(num_tasks), Int32.(task_lens))

        LARGEST = true
        REV = false

        result, indices_out = topk_radix_select!(
            data, result, k, ws,
            indices_in, indices_out, Int32.(task_lens),
            Val(LARGEST), Val(!REV), Val(false), Val(true), Val(true)
        )

        offset = 0
        for (task_id, task_len) in enumerate(task_lens)
            data_array = Array(data[offset+1:offset+task_len])

            # Extract task result (k elements for this task)
            task_result = Array(result[:, task_id])
            idx = Array(indices_out[:, task_id]) .- offset

            # Get the k largest values and their indices
            expected_vals = sort(partialsort(data_array, 1:k, rev=LARGEST), rev=REV)

            @test Array(task_result) == expected_vals

            if REV == LARGEST
                expected_order = sortperm(data_array, rev=LARGEST)[1:k] .|> Int
            else
                expected_order = sortperm(data_array, rev=LARGEST)[k:-1:1] .|> Int
            end

            # Check that values at indices match (allows different index order for duplicates)
            @test data_array[idx] == data_array[expected_order]

            # Verify indices are valid: within bounds and unique
            @test all(1 .<= idx .<= length(data))
            @test length(unique(idx)) == length(idx)  # No duplicate indices

            offset += task_len
        end
    end
end