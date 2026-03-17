# ==============================================================================
# Utility Functions for RadiK
# ==============================================================================
# Based on radik/radik/RadixSelect/utils.cuh

import KernelAbstractions.Extras: @unroll


@inline _compute_hist_len(::Type{T}, RIGHT) where {T} = 1 << (8 * sizeof(T) - RIGHT)

"""
    get_bin_id(val::Float32, ::Val{LEFT}, ::Val{RIGHT}) -> Int32

Extract a radix sort bin index from a `Float32` by selecting bits `[31-LEFT : RIGHT-LEFT]`
from its IEEE 754 representation.

Transforms floats into sortable integers via XOR mask (handles sign/magnitude ordering),
then extracts the specified bit range using `<< LEFT` followed by `>>> RIGHT`.

# Examples
```
# 3-pass radix top-k (12+12+8 bits)
get_bin_id(x, Val(0),  Val(20))  # bits [31:20], 4096 bins, coarse
get_bin_id(x, Val(12), Val(20))  # bits [19:8],  4096 bins, medium  
get_bin_id(x, Val(24), Val(24))  # bits [7:0],    256 bins, fine

# 3-bit histogram bins
get_bin_id(x, Val(0), Val(29))   # bits [31:29], 8 bins
get_bin_id(x, Val(8), Val(29))   # bits [23:21], 8 bins
```
"""
@inline function get_bin_id(val::Float32, ::Val{LEFT}, ::Val{RIGHT}) where {LEFT, RIGHT}
    bits = reinterpret(UInt32, val)
    mask = ((~(bits >> 31)) + 0x00000001) | 0x80000000
    return ((bits ⊻ mask) << LEFT) >> RIGHT
end

"""
    is_valid_value(val::Float32) -> Bool

Check if a value is valid (not NaN or Inf).
"""
@inline function is_valid_value(val::Float32)
    return !isnan(val) && !isinf(val)
end

"""
    safe_value(val::Float32, largest::Bool=true) -> Float32

Replace NaN/Inf with min/max finite values.

Used when NaN filtering is enabled.
"""
@inline function safe_value(val::Float32, largest::Bool=true)
    if isnan(val) || isinf(val)
        return largest ? floatmin(Float32) : floatmax(Float32)
    end
    return val
end

# ==============================================================================
# Scaling Utilities (optional, for numerical stability)
# ==============================================================================

@inline function sample_scaler(data::AbstractArray{T}, idx::Int, ::Val{WITHSCALE}) where {T, WITHSCALE}
    WITHSCALE || return zero(T)
    scaler = data[idx]
    return (isnan(scaler) || isinf(scaler)) ? zero(T) : scaler
end

@inline function apply_scaling(val::T, scaler::T, ::Val{WITHSCALE}, ::Val{LARGEST}) where {T<:AbstractFloat, WITHSCALE, LARGEST}
    processed_val = isnan(val) ? (LARGEST ? floatmin(T) : floatmax(T)) : val
    return WITHSCALE ? (processed_val - scaler) : processed_val
end

# Vectorized version (Julia handles this automatically via broadcasting!)
@inline function apply_scaling(vals::NTuple{N, T}, scaler::T, ::Val{WITHSCALE}, ::Val{LARGEST}) where {N, T<:AbstractFloat, WITHSCALE, LARGEST}
    return ntuple(i -> apply_scaling(vals[i], scaler, Val(WITHSCALE), Val(LARGEST)), Val(N))
end

# ==============================================================================
# Prefix sum
# ==============================================================================

import KernelIntrinsics: Up, Down

macro warpreduce(val, lane, op=:+, dir=:Up, ws=32, mask=0xffffffff)
    condition = if dir == :Up
        :($(esc(lane)) > offset)
    elseif dir == :Down
        :($(esc(lane)) + offset ≤ $(esc(ws)))
    end

    quote
        local offset = 1
        while offset < $(esc(ws))
            shuffled = @shfl($(esc(dir)), $(esc(val)), offset, $(esc(mask)))
            if $condition
                $(esc(val)) = $(esc(op))(shuffled, $(esc(val)))
            end
            offset <<= 1
        end
    end
end