# Recreating the single task benchmark from 
# Li, Y., Zhou, B., Zhang, J., Wei, X., Li, Y., & Chen, Y. (2024). 
# "RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection." ,
# Proceedings of the 38th ACM International Conference on Supercomputing

const BACKEND = get(ENV, "BACKEND") do
    error("Usage: BACKEND=[cuda|metal] julia --project")
end

using BenchmarkTools
using KernelAbstractions
using PrettyTables
using Printf
using RadiK
using Random
using Test

import KernelAbstractions as KA
using RadiK: RadiKWorkspace

if BACKEND == "cuda"
    using CUDA
    backend = CUDABackend()
elseif BACKEND == "metal"
    using Metal
    backend = MetalBackend()
end

LARGEST = true
REV = true

# Configuration
N_powers = (21, 23, 25, 27, 29)
K_values = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Store results
timings_burst = Matrix{Float64}(undef, length(N_powers), length(K_values))
timings_steady = Matrix{Float64}(undef, length(N_powers), length(K_values))

timings_cpu = Matrix{Float64}(undef, length(N_powers), length(K_values))
timings_mps = Matrix{Float64}(undef, length(N_powers), length(K_values))

for (i, pow_2) in enumerate(N_powers)
    println("n = 2^$pow_2")
    n = 2^pow_2
    data = rand!(allocate(backend, Float32, n))

    # Allocate workspace
    ws = RadiKWorkspace(backend, n, 1, Int32, Float32)

    for (j, k) in enumerate(K_values)
        print("$k ")

        # Allocate output
        result = KA.zeros(backend, Float32, k)
        idx_in = KA.zeros(backend, Int32, k);
        idx_out = similar(idx_in);

        # Warm-up
        topk_radix_select!(result, idx_out, ws, data, idx_in, Int32[n], Int32(k), Val(LARGEST), Val(!REV), Val(false), Val(false), Val(true))

        # Correctness test
        @test Array(result) == partialsort(Array(data), 1:k, rev=true)

        # This shows a burst mode followed by a steady mode, we will report both
        """
        btimes = [
            begin
                GC.gc(true)
                rand!(data)

                CUDA.@time topk_radix_select!(
                        result, idx_out, ws, data, idx_in, Int32[n], Int32(k),
                        Val(LARGEST), Val(!REV), Val(false), Val(false), Val(true))
            end
            for i in 1:20
        ]
        """

        burst_t = []
        steady_t = []

        for _ in 1:10
            rand!(data)

            b = @benchmark begin
                topk_radix_select!(
                    $result, $idx_out, $ws, $data, $idx_in, Int32[$n], Int32($k),
                    Val($LARGEST), Val(!$REV), Val(false), Val(false), Val(true)
                )
            end

            push!(burst_t, minimum(b).time / 1e6)
            push!(steady_t, median(b).time / 1e6)

            GC.gc(true)
        end

        timings_burst[i, j] = mean(burst_t)
        timings_steady[i, j] = mean(steady_t)

        if BACKEND == "metal"
            data_cpu = Array(data)
            b_cpu = @benchmark partialsort($data_cpu, 1:$k, rev=true)
            timings_cpu[i, j] = median(b_cpu).time / 1e6
        end

        if k <= 16 && BACKEND == "metal"
            data_mat = reshape(data, :, 1)
            I = KA.zeros(backend, UInt32, k, 1)
            V = KA.zeros(backend, Float32, k, 1)
            b_mps = @benchmark Metal.MPS.topk!($data_mat, $I, $V, $k)
            timings_mps[i, j] = median(b_mps).time / 1e6
            # Free MPS arrays
            I = V = data_mat = nothing
        end

        GC.gc(true)
    end
    print("\n")
    println("# ------------------------------------- ")

    # Clean up after each N power iteration
    data = nothing
    GC.gc(true)
end

# Combine burst and steady timings into single table
combined_timings = ["$(round(timings_burst[i,j], digits=2)) / $(round(timings_steady[i,j], digits=2))"
                    for i in 1:length(N_powers), j in 1:length(K_values)]

println("\nTimings RadiK (Burst / Steady in milliseconds):")
pretty_table(
    combined_timings;
    row_labels = ["2^$(p)" for p in N_powers],
    row_label_column_alignment = :l,
    column_labels = ["K=$k" for k in K_values],
    display_size = (typemax(Int), typemax(Int))
)

println("\nTimings CPU (milliseconds):")
pretty_table(
    round.(timings_cpu, digits=2);
    row_labels = ["2^$(p)" for p in N_powers],
    row_label_column_alignment = :l,
    column_labels = ["K=$k" for k in K_values],
    display_size = (typemax(Int), typemax(Int))
)

println("\nTimings MPS (milliseconds):")
pretty_table(
    round.(timings_mps, digits=2);
    row_labels = ["2^$(p)" for p in N_powers],
    row_label_column_alignment = :l,
    column_labels = ["K=$k" for k in K_values],
    display_size = (typemax(Int), typemax(Int))
)
