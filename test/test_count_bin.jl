
# include("../src/kernels/count_bin.jl")
# include("../src/utils.jl")

using CUDA, BenchmarkTools
using Test

# Test data: 3 tasks with varying sizes
task1_data = Float32.(1:20_000_000)
task2_data = Float32.(50_000_000:80_400_000)
task3_data = Float32.(100_000_000:200_140_000)
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

data_gpu = CuArray(all_data)
histogram_gpu = CUDA.zeros(Int32, hist_len, num_tasks)
task_offsets_gpu = CuArray(task_offsets)
backend = get_backend(data_gpu)

for threads_per_block in (4, 8, 64, 256, 1024), blocks_x in (2, 16, 32, 128)
# for threads_per_block in (1024), blocks_x in (16)
    
    histogram_gpu .= 0

    count_bin_ex_kernel!(backend, threads_per_block)(
        histogram_gpu, data_gpu, task_offsets_gpu,
        Val(LEFT), Val(RIGHT), Val(hist_len), Val(4), Val(true), Val(true);
        ndrange=(threads_per_block * blocks_x, num_tasks)
    )

    @show threads_per_block
    @show blocks_x

    for i in 1:num_tasks
        @test histogram_scaled_cpu[i] == histogram_gpu[:, i]
    end

end

# ========================================
# Test count_bin_kernel!
# ========================================

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

histogram_gpu_1d = CUDA.zeros(Int32, hist_len)

for i in 1:num_tasks 
    # Convert to GPU
    data_gpu = CuArray(data_array[i])
    
    n = length(data_gpu)

    for threads_per_block in (1, 8, 64, 256, 1024)
        histogram_gpu .= 0

        num_blocks = cld(n, threads_per_block)
        count_bin_kernel!(backend, threads_per_block)(
            histogram_gpu_1d, data_gpu,
            Val(LEFT), Val(RIGHT);
            ndrange = num_blocks * threads_per_block
        )

        @test histogram_cpu[i] == histogram_gpu_1d
    end
end


# ========================================
# Benchmark
# ========================================

# BENCHMARK TIME 
@benchmark count_bin_ex!(
    $data_gpu, $histogram_gpu, $task_offsets_gpu;
    LEFT=$LEFT, RIGHT=$RIGHT,
    with_scale=true, largest=true
)
