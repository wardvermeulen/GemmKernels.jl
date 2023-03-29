using BenchmarkTools
using GemmKernels
using GemmKernels.Tiling
using GemmKernels.Layout
using CUDA
using Test
using Statistics
using Printf

# using TensorOperations

# TCCG benchmark ?: D_abcde = A_efbad * B_cf

# sizes for each dimension
SA = 16
SB = 16
SC = 128
SD = 16
SE = 16
SF = 128

A_tmp = rand(Float16, (SE, SF, SB, SA, SD))
B_tmp = rand(Float16, (SC, SF))

A = CuArray(A_tmp)
B = CuArray(B_tmp)

# layout for the A tensor
abstract type LayoutA{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    K = tile.base.K + tile.offset.K

    f = K

    # e = M % Base.size(workspace, 1)
    # b = (M ÷ Base.size(workspace, 1)) % Base.size(workspace, 3)
    # a = ((M ÷ Base.size(workspace, 1)) ÷ Base.size(workspace, 3)) % Base.size(workspace, 4)
    # d = ((M ÷ Base.size(workspace, 1)) ÷ Base.size(workspace, 3)) ÷ Base.size(workspace, 4)

    e = M % Base.size(workspace, 1)
    b = (M ÷ Base.size(workspace, 1)) % Base.size(workspace, 3)
    a = (M ÷ (Base.size(workspace, 1) * Base.size(workspace, 3))) % Base.size(workspace, 4)
    d = M ÷ (Base.size(workspace, 1) * Base.size(workspace, 3) * Base.size(workspace, 4))

    offset  = 1
    offset += e 
    offset += f * Base.size(workspace, 1) 
    offset += b * Base.size(workspace, 1) * Base.size(workspace, 2)
    offset += a * Base.size(workspace, 1) * Base.size(workspace, 2) * Base.size(workspace, 3)
    offset += d * Base.size(workspace, 1) * Base.size(workspace, 2) * Base.size(workspace, 3) * Base.size(workspace, 4)

    # if (threadIdx().x == 3 || threadIdx().x == 2)
    #     @cushow (M, d, b, a, offset)
    # end

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the B tensor
abstract type LayoutB{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    K = tile.base.K + tile.offset.K
    N = tile.base.N + tile.offset.N

    f = K
    c = N

    offset  = 1 
    offset += c 
    offset += f * Base.size(workspace, 1)

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the C tensor
abstract type LayoutC{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    ntuple(Val(N)) do i
        VecElement{T}(zero(T))
    end
end

# layout for the D tensor
abstract type LayoutD{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.store!(::Type{LayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    for i = 1 : NUMEL
        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        c = N

        e = (M + i - 1) % Base.size(workspace, 5)
        b = ((M + i - 1) ÷ Base.size(workspace, 5)) % Base.size(workspace, 2)
        a = ((M + i - 1) ÷ (Base.size(workspace, 5) * Base.size(workspace, 2))) % Base.size(workspace, 1)
        d = (M + i - 1) ÷ (Base.size(workspace, 5) * Base.size(workspace, 2) * Base.size(workspace, 1))

        @inbounds workspace[a + 1, b + 1, c + 1, d + 1, e + 1] = value[i].value
    end
end

# implementation using GemmKernels.jl
function gemmkernels_impl(;benchmark = false)
    D = CuArray(zeros(Float16, (SA, SB, SC, SD, SE)))

    M = SE * SB * SA * SD
    N = SC
    K = SF

    conf = GemmKernels.get_config(
                                  gemm_shape = (M = M, N = N, K = K),
                                  operator = Operator.WMMAOp{16, 16, 16, Float16},

                                  global_a_layout = LayoutA{Float16},
                                  global_b_layout = LayoutB{Float16},
                                  global_c_layout = LayoutC{Float16},
                                  global_d_layout = LayoutD{Float16},

                                  shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
                                  shared_b_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
                                  shared_c_layout = Layout.AlignedColMajor{Float16},
                                  shared_d_layout = Layout.AlignedColMajor{Float16},

                                  is_a_col_major = true,
                                  is_b_col_major = true,
                                 )

    if !benchmark
        GemmKernels.matmul(A, B, D, D, conf;
                        kernel = Kernel.matmul_pipelined
                        )
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed GemmKernels.matmul(A, B, D, D, conf;
                            kernel = Kernel.matmul_pipelined
                        )
            push!(times, time)
        end

        times
    end
end

# cuTENSOR implementation
function cutensor_impl(;algo = CUDA.CUTENSOR.CUTENSOR_ALGO_DEFAULT, benchmark = false)
    D = CuArray(zeros(Float16, (SA, SB, SC, SD, SE)))

    plan = CUDA.CUTENSOR.plan_contraction(A, [ 'e', 'f', 'b', 'a', 'd' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          B, [ 'c', 'f' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          D, [ 'a', 'b', 'c', 'd', 'e' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          algo = algo,
                                          compute_type = Float16)



    if !benchmark
        CUDA.CUTENSOR.contraction!(1,
                                A, [ 'e', 'f', 'b', 'a', 'd' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                B, [ 'c', 'f' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                0,
                                D, [ 'a', 'b', 'c', 'd', 'e' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                compute_type = Float16,
                                plan = plan)
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed CUDA.CUTENSOR.contraction!(1,
                                    A, [ 'e', 'f', 'b', 'a', 'd' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    B, [ 'c', 'f' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    0,
                                    D, [ 'a', 'b', 'c', 'd', 'e' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    compute_type = Float16,
                                    plan = plan)
            push!(times, time)
        end

        times
    end
end

# compare
function test()
    D_reference = cutensor_impl(algo = CUDA.CUTENSOR.CUTENSOR_ALGO_GETT)
    D_gemmkernels = gemmkernels_impl()
    # D_cpu = zeros(Float16, (SA, SB, SC, SD, SE))
    
    # @tensor begin
    #     D_cpu[a, b, c, d, e] = A[e, f, b, a, d] * B[c, f]
    # end

    # @show D_cpu[1:10, 1:10, 1, 1, 1]

    @show D_reference[1:10, 1:10, 1, 1, 1]
    @show D_gemmkernels[1:10, 1:10, 1, 1, 1]

    display(@test all(isapprox.(Array(D_reference), Array(D_gemmkernels); rtol = sqrt(eps(Float16)))))
end

# Taken from BenchmarkTools.jl: src/trials.jl
function rmskew(t::Vector)
    st = sort(t)
    return st[1:(BenchmarkTools.skewcutoff(st) - 1)]
end

# Benchmark function
function bench()
    bench_cutensor = cutensor_impl(benchmark = true)

    bench_cutensor_tgett = cutensor_impl(algo = CUDA.CUTENSOR.CUTENSOR_ALGO_TGETT, benchmark = true)

    bench_cutensor_gett = cutensor_impl(algo = CUDA.CUTENSOR.CUTENSOR_ALGO_GETT, benchmark = true)

    bench_cutensor_ttgt = cutensor_impl(algo = CUDA.CUTENSOR.CUTENSOR_ALGO_TTGT, benchmark = true)

    bench_gemmkernels = gemmkernels_impl(benchmark = true)

    for (title, bench) in [("cuTENSOR (let heuristic choose algorithm):", bench_cutensor),
                           ("cuTENSOR (force GETT):", bench_cutensor_gett),
                           ("cuTENSOR (force TGETT):", bench_cutensor_tgett),
                           ("cuTENSOR (force TTGT):", bench_cutensor_ttgt),
                           ("GemmKernels:", bench_gemmkernels),
                           ("rmskew: cuTENSOR (let heuristic choose algorithm):", rmskew(bench_cutensor)),
                           ("rmskew: cuTENSOR (force GETT):", rmskew(bench_cutensor_gett)),
                           ("rmskew: cuTENSOR (force TGETT):", rmskew(bench_cutensor_tgett)),
                           ("rmskew: cuTENSOR (force TTGT):", rmskew(bench_cutensor_ttgt)),
                           ("rmskew: GemmKernels:", rmskew(bench_gemmkernels))
                          ]
        println(title)
        bench .= bench .* 1e6 # convert seconds to us
        @printf("Samples:            %u\n", length(bench))
        @printf("Range (min .. max): %.2f us .. %.2f us\n", minimum(bench), maximum(bench))
        @printf("Median:             %.2f us\n", median(bench))
        @printf("Mean ± std:         %.2f us ± %.2f us\n", mean(bench), std(bench))
        println()
    end

    nothing
end

# Main entry point
function main()
    display(test())
    println()

    bench()
end

!isinteractive() && test()