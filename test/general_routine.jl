const WARP_SIZE = KI.get_warpsize(KI.device(backend))

include("test_utils.jl")
include("test_count_bin.jl")
include("test_select_bin.jl")
include("test_select_candidate.jl")
include("test_filter.jl")
include("test_algorithm.jl")
include("Aqua.jl")