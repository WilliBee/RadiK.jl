using RadiK: 
    get_bin_id,
    is_valid_value,
    safe_value,
    @warpreduce

@testset "Test utils.jl" begin

    @testset "get_bin_id" begin
        # Sort order preservation
        vals = Float32[Inf, 100.0, 2.0, 1.0, 0.0, -1.0, -2.0, -100.0, -Inf]

        # Extract 8 first bits
        bins = get_bin_id.(vals, Val(0), Val(24))

        # All bins should be in valid range [0, 255] for 8-bit
        @test all(0 <= b < 256 for b in bins)
        @test first(bins) == 255
        @test last(bins) == 0

        # Bins should be ordered consistently with float values
        @test all( <(0), diff(Int.(bins)))
    end

    @testset "is_valid_value" begin
        # Valid values
        @test is_valid_value(1.0f0)
        @test is_valid_value(-1.0f0)
        @test is_valid_value(0.0f0)
        @test is_valid_value(3.14f0)

        # Invalid values
        @test !is_valid_value(NaN32)
        @test !is_valid_value(Inf32)
        @test !is_valid_value(-Inf32)
    end

    @testset "safe_value" begin
        # Normal values pass through
        @test safe_value(1.0f0, true) == 1.0f0
        @test safe_value(-1.0f0, true) == -1.0f0

        # NaN/Inf are replaced
        @test safe_value(NaN32, true) == floatmin(Float32)
        @test safe_value(Inf32, true) == floatmin(Float32)
        @test safe_value(-Inf32, true) == floatmin(Float32)

        @test safe_value(NaN32, false) == floatmax(Float32)
        @test safe_value(Inf32, false) == floatmax(Float32)

    end

    @testset "warp_prefix_sum" begin
        # Test kernel for ascending scan (LARGEST=false)
        @kernel function test_ascending(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            lane = (I - 1) % 32 + 1
            
            # val = warp_prefix_sum(lane, val, Val(false))
            @warpreduce(val, lane, +, Up)
            
            dst[I] = val
        end

        # Test kernel for descending scan (LARGEST=true)  
        @kernel function test_descending(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            lane = (I - 1) % 32 + 1
            
            # val = warp_prefix_sum(lane, val, Val(true))
            @warpreduce(val, lane, +, Down)
            
            dst[I] = val
        end
        
        # Test 1: Ascending scan with 1:32
        src = adapt(backend, Int32.(1:WARP_SIZE))
        dst = adapt(backend, zeros(Int32, WARP_SIZE))
        test_ascending(backend)(dst, src; ndrange=WARP_SIZE)
        
        expected_ascending = cumsum(1:WARP_SIZE)
        @test Array(dst) == expected_ascending

        # Test 2: Descending scan with 1:32
        src = adapt(backend, Int32.(1:WARP_SIZE))
        dst = adapt(backend, zeros(Int32, WARP_SIZE))
        test_descending(backend)(dst, src; ndrange=WARP_SIZE)
        
        # Descending: lane i gets sum of lanes i to 32
        # i.e., reverse cumsum: [528, 527, 525, 522, ...]
        expected_descending = reverse(cumsum(reverse(1:WARP_SIZE)))
        @test Array(dst) == expected_descending
        
        # Test 3: Multiple warps (64 elements, 2 warps)
        src = adapt(backend, Int32.(1:64))
        dst = adapt(backend, zeros(Int32, 64))
        test_ascending(backend)(dst, src; ndrange=64)
        
        result = Array(dst)
        # First warp: 1, 3, 6, ..., 528
        @test result[1:32] == cumsum(1:32)
        # Second warp: 33, 33+34=67, 67+35=102, ...
        @test result[33:64] == cumsum(33:64)
    end

end # @testset "Test utils.jl"