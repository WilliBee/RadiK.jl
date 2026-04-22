using Adapt
using KernelAbstractions
import KernelAbstractions as KA
using KernelIntrinsics
import KernelIntrinsics as KI
using Pkg
using RadiK
using Test

include("meta_helpers.jl")

TEST_BACKEND = get(ENV, "TEST_BACKEND") do
    backend_str = has_cuda() ? "cuda" : has_roc() ? "roc" : has_metal() ? "metal" : "unknown"
    @info "TEST_BACKEND not set, defaulting to $backend_str"
    backend_str
end

Pkg.activate("test/envs/$TEST_BACKEND")
Pkg.activate("envs/$TEST_BACKEND") # when running tests
Pkg.instantiate()

if TEST_BACKEND == "cuda"
    using CUDA
    if !CUDA.functional()
        @warn "No CUDA device found — skipping tests"
        exit(0)
    end
    backend = CUDABackend()
    include("general_routine.jl")
elseif TEST_BACKEND == "roc"
    using AMDGPU
    if !AMDGPU.functional()
        @warn "No AMDGPU device found — skipping tests"
        exit(0)
    end
    backend = ROCBackend()
    include("general_routine.jl")
elseif TEST_BACKEND == "metal"
    using Metal
    if !Metal.functional()
        @warn "No Metal device found — skipping tests"
        exit(0)
    end
    backend = MetalBackend()
    include("general_routine.jl")
else
    error("Unknown backend: $TEST_BACKEND")
end