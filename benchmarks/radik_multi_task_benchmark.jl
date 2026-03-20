# Recreating the multitask benchmark from
# Li, Y., Zhou, B., Zhang, J., Wei, X., Li, Y., & Chen, Y. (2024).
# "RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection." ,
# Proceedings of the 38th ACM International Conference on Supercomputing

const BACKEND = get(ENV, "BACKEND") do
    error("Usage: BACKEND=[cuda|metal] julia --project")
end

using Adapt
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

# =============================================================================
# Benchmark A: Input Length Variation (Figure 9a)
# Vary N from 2^16 to 2^22, keep BATCH=16, K=512
# =============================================================================

println("\n" * "="^80)
println("Benchmark A: Input Length Variation")
println("="^80)

BATCH_a = 16
K_a = 512
N_powers_a = (16, 17, 18, 19, 20, 21, 22)

timings_a_burst = Vector{Float64}(undef, length(N_powers_a))
timings_a_steady = Vector{Float64}(undef, length(N_powers_a))

for (i, pow_2) in enumerate(N_powers_a)
    println("BATCH=$BATCH_a, N=2^$pow_2, K=$K_a")
    n = 2^pow_2

    task_lens = fill(Int32(n), BATCH_a)
    total_len = sum(task_lens)

    Random.seed!(42)
    data = rand!(allocate(backend, Float32, total_len))

    ws = RadiKWorkspace(backend, n, BATCH_a, Int32, Float32)

    result = KA.zeros(backend, Float32, Int(K_a), Int(BATCH_a))
    indices_in = adapt(backend, collect(Int32(1):Int32(total_len)))
    indices_out = KA.zeros(backend, Int32, Int(K_a), Int(BATCH_a))

    # Warm-up
    topk_radix_select!(result, indices_out, ws, data, indices_in, task_lens, Int32(K_a), Val(LARGEST), Val(!REV), Val(false), Val(true), Val(true))

    # Collect burst and steady timings
    burst_t = []
    steady_t = []

    for _ in 1:10
        rand!(data)

        b = @benchmark topk_radix_select!($result, $indices_out, $ws, $data, $indices_in, $task_lens, Int32(K_a), Val($LARGEST), Val(!$REV), Val(false), Val(true), Val(true))

        push!(burst_t, minimum(b).time / 1e6)
        push!(steady_t, median(b).time / 1e6)
    end

    timings_a_burst[i] = mean(burst_t)
    timings_a_steady[i] = mean(steady_t)
end

# Combine burst and steady timings
combined_a = ["$(round(timings_a_burst[i], digits=2)) / $(round(timings_a_steady[i], digits=2))" for i in 1:length(N_powers_a)]

println("\nTimings (Burst / Steady in milliseconds) - BATCH=16, K=512:")
pretty_table(
    combined_a;
    row_labels = ["2^$(p)" for p in N_powers_a],
    row_label_column_alignment = :l,
    column_labels = ["Latency (ms)"],
    display_size = (typemax(Int), typemax(Int))
)

# =============================================================================
# Benchmark B: k Variation with Batches (Figure 9b)
# Vary K from 2^4 to 2^10, keep BATCH=16, N=2^22
# =============================================================================

println("\n" * "="^80)
println("Benchmark B: k Variation with Batches")
println("="^80)

BATCH_b = 16
N_b = 1 << 22
K_powers_b = (4, 5, 6, 7, 8, 9, 10)

timings_b_burst = Vector{Float64}(undef, length(K_powers_b))
timings_b_steady = Vector{Float64}(undef, length(K_powers_b))

for (i, pow_2) in enumerate(K_powers_b)
    k = 2^pow_2
    println("BATCH=$BATCH_b, N=2^22, K=2^$pow_2")

    task_lens = fill(Int32(N_b), BATCH_b)
    total_len = sum(task_lens)

    Random.seed!(42)
    data = rand!(allocate(backend, Float32, total_len))

    ws = RadiKWorkspace(backend, N_b, BATCH_b, Int32, Float32)

    result = KA.zeros(backend, Float32, Int(k), Int(BATCH_b))
    indices_in = adapt(backend, collect(Int32(1):Int32(total_len)))
    indices_out = KA.zeros(backend, Int32, Int(k), Int(BATCH_b))

    # Warm-up
    topk_radix_select!(result, indices_out, ws, data, indices_in, task_lens, Int32(k), Val(LARGEST), Val(!REV), Val(false), Val(true), Val(true))

    # Collect burst and steady timings
    burst_t = []
    steady_t = []

    for _ in 1:10
        rand!(data)

        b = @benchmark topk_radix_select!($result, $indices_out, $ws, $data, $indices_in, $task_lens, Int32($k), Val($LARGEST), Val(!$REV), Val(false), Val(true), Val(true))

        push!(burst_t, minimum(b).time / 1e6)
        push!(steady_t, median(b).time / 1e6)
    end

    timings_b_burst[i] = mean(burst_t)
    timings_b_steady[i] = mean(steady_t)
end

# Combine burst and steady timings
combined_b = ["$(round(timings_b_burst[i], digits=2)) / $(round(timings_b_steady[i], digits=2))" for i in 1:length(K_powers_b)]

println("\nTimings (Burst / Steady in milliseconds) - BATCH=16, N=2^22:")
pretty_table(
    combined_b;
    row_labels = ["$(k)" for k in [16, 32, 64, 128, 256, 512, 1024]],
    row_label_column_alignment = :l,
    column_labels = ["Latency (ms)"],
    display_size = (typemax(Int), typemax(Int))
)

# =============================================================================
# Benchmark C: Batch Size Variation (Figure 9c)
# Vary BATCH from 2^0 to 2^6, keep N=2^22, K=512
# =============================================================================

println("\n" * "="^80)
println("Benchmark C: Batch Size Variation")
println("="^80)

N_c = 1 << 22
K_c = 512
BATCH_powers_c = (0, 1, 2, 3, 4, 5, 6)

timings_c_burst = Vector{Float64}(undef, length(BATCH_powers_c))
timings_c_steady = Vector{Float64}(undef, length(BATCH_powers_c))

for (i, pow_2) in enumerate(BATCH_powers_c)
    batch_size = 2^pow_2
    println("BATCH=2^$pow_2, N=2^22, K=$K_c")

    task_lens = fill(Int32(N_c), batch_size)
    total_len = sum(task_lens)

    Random.seed!(42)
    data = rand!(allocate(backend, Float32, total_len))

    ws = RadiKWorkspace(backend, N_c, batch_size, Int32, Float32)

    result = KA.zeros(backend, Float32, Int(K_c), Int(batch_size))
    indices_in = adapt(backend, collect(Int32(1):Int32(total_len)))
    indices_out = KA.zeros(backend, Int32, Int(K_c), Int(batch_size))

    # Warm-up
    topk_radix_select!(result, indices_out, ws, data, indices_in, task_lens, Int32(K_c), Val(LARGEST), Val(!REV), Val(false), Val(true), Val(true))

    # Collect burst and steady timings
    burst_t = []
    steady_t = []

    for _ in 1:10
        rand!(data)

        b = @benchmark topk_radix_select!($result, $indices_out, $ws, $data, $indices_in, $task_lens, Int32(K_c), Val($LARGEST), Val(!$REV), Val(false), Val(true), Val(true))

        push!(burst_t, minimum(b).time / 1e6)
        push!(steady_t, median(b).time / 1e6)
    end

    timings_c_burst[i] = mean(burst_t)
    timings_c_steady[i] = mean(steady_t)
end

# Combine burst and steady timings
combined_c = ["$(round(timings_c_burst[i], digits=2)) / $(round(timings_c_steady[i], digits=2))" for i in 1:length(BATCH_powers_c)]

println("\nTimings (Burst / Steady in milliseconds) - N=2^22, K=512:")
pretty_table(
    combined_c;
    row_labels = ["$(bs)" for bs in [1, 2, 4, 8, 16, 32, 64]],
    row_label_column_alignment = :l,
    column_labels = ["Latency (ms)"],
    display_size = (typemax(Int), typemax(Int))
)
