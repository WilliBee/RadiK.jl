using Adapt
using KernelAbstractions
import KernelAbstractions as KA
using KernelIntrinsics
import KernelIntrinsics as KI
using RadiK
using Test

const BACKEND = get(ENV, "BACKEND") do
    error("Usage: BACKEND=[cuda|metal] julia --project")
end

if BACKEND == "cuda"
    using CUDA
    const backend = CUDABackend()

    @testset "CUDA" begin
        include("test_utils.jl")
        include("test_count_bin.jl")
        include("test_select_bin.jl")
        include("test_select_candidate.jl")
        include("test_filter.jl")
        include("test_algorithm.jl")
    end
elseif BACKEND == "metal"
    using Metal
    const backend = MetalBackend()
    const WARP_SIZE = KI.get_warpsize(KI.device(backend))

    @testset "Metal" begin
        include("test_utils.jl")
        include("test_count_bin.jl")
        include("test_select_bin.jl")
        include("test_select_candidate.jl")
        include("test_filter.jl")
        include("test_algorithm.jl")
    end
else
    error("Usage: BACKEND=[cuda|metal] julia --project")
end
