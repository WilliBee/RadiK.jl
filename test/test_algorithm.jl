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
                    results, indices = topk(data, k; largest=LARGEST, rev=REV)

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
        num_tasks = 5
        k = 64

        # Create output arrays (2D: k x num_tasks)
        result = KA.zeros(backend, Float32, k, Int(num_tasks))
        indices_out = KA.zeros(backend, Int32, k, Int(num_tasks))

        ws = RadiKWorkspace(backend, 5000, num_tasks, Int32)

        LARGEST = true
        REV = false

        for task_lens in ([1000, 2000, 1500, 500, 5000], ) #[100, 100, 100, 100, 100]) 

            total_len = sum(task_lens)
            data = rand!(allocate(backend, Float32, total_len))
            indices_in = adapt(backend, collect(Int32(1):Int32(total_len)))

            result, indices_out = topk_radix_select!(
                result, indices_out, ws,
                data, indices_in, Int32.(task_lens), Int32(k),
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

    @testset "Test topk convenience overloads" begin
        data = rand!(allocate(backend, Float32, 1000))
        k = 100

        values1, indices1 = topk(data, k; largest=true, rev=false)

        indices = adapt(backend, collect(Int32(1):Int32(1000)))
        values2, indices2 = topk(data, k; indices=indices, largest=true, rev=false)

        @test Array(values1) == Array(values2)
        @test Array(indices1) == Array(indices2)

        expected = sort(partialsort(Array(data), 1:k, rev=true), rev=false)
        @test Array(values1) == expected
    end

    @testset "Test topk batch processing" begin
        num_tasks = 3
        task_lens = [1000, 1500, 800]
        k = 50

        # Create concatenated data
        total_len = sum(task_lens)
        data = rand!(allocate(backend, Float32, total_len))

        # Test batch without custom indices
        values1, indices1 = topk(data, task_lens, k; largest=true, rev=false)

        # Test batch with custom indices (per-task sequential indices)
        indices_parts = map(task_lens) do len
            collect(Int32(1):Int32(len))
        end
        indices = adapt(backend, vcat(indices_parts...))
        values2, indices2 = topk(data, task_lens, k; indices=indices, largest=true, rev=false)

        # Both should produce same values
        @test Array(values1) == Array(values2)

        # Both should produce same indices (when using sequential indices)
        @test Array(indices1) == Array(indices2)

        # Verify each task's results are correct
        offset = 0
        for task_id in 1:num_tasks
            task_data = Array(data[offset+1:offset+task_lens[task_id]])
            task_values = Array(values1[:, task_id])

            expected = sort(partialsort(task_data, 1:k, rev=true), rev=false)
            @test task_values == expected

            offset += task_lens[task_id]
        end
    end

    @testset "Error messages and validation" begin
        # Test dimension validation in topk!
        data = adapt(backend, randn(Float32, 1000))
        val_out_wrong = KA.zeros(backend, Float32, 50, 1)  # Wrong size
        idx_out_wrong = KA.zeros(backend, Int32, 50, 1)
        @test_throws ErrorException topk!(val_out_wrong, idx_out_wrong, data, [1000], 100)

        # Test indices length validation
        indices_wrong = adapt(backend, collect(Int32(1):Int32(500)))  # Wrong length
        @test_throws ErrorException topk(data, 100; indices=indices_wrong)
    end
end