module RadiK

using KernelAbstractions
using Adapt
using AcceleratedKernels
using KernelIntrinsics

using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @inbounds, @groupsize
using KernelIntrinsics: vload_multi
using Atomix: @atomic

include("utils.jl")
include("kernels/count_bin.jl")
include("kernels/select_bin.jl")
include("kernels/select_candidate.jl")
include("kernels/filter.jl")
include("algorithm.jl")

# Export main API
export topk_radix_select!
export topk
export RadiKWorkspace

end # module RadiK
