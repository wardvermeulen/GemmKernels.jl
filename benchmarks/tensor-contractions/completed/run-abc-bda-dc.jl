using BenchmarkTools
using GemmKernels
using GemmKernels.Tiling
using GemmKernels.Layout
using GemmKernels.GETT
using GemmKernels.TensorPlan
using CUDA
using NVTX
using Test
using Statistics
using Printf

# TCCG benchmark 1: D_abc = A_bda * B_dc

test_or_bench::Bool = false
if (size(ARGS, 1) == 1)
    test_or_bench = parse(Bool, ARGS[1])
end

# using CUDA; SA = 64; SB = 32; SC = 2048; SD = 2048; A = CuArray(rand(Float16, (SB, SD, SA))); B = CuArray(rand(Float16, (SD, SC))); D = CuArray(zeros(Float16, (SA, SB, SC)));

# sizes for each dimension
SA = 64
SB = 32
SC = 2048
SD = 2048

A = CuArray(rand(Float16, (SB, SD, SA)))
B = CuArray(rand(Float16, (SD, SC)))
C = rand(Float16, (SA, SB, SC)) * Float16(20.0)

# layout for the A tensor
abstract type LayoutA{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    K = tile.base.K + tile.offset.K

    d = K

    a = M ÷ Base.size(workspace, 1)
    b = M % Base.size(workspace, 1)

    offset = 1 + b + d * Base.size(workspace, 1) + a * Base.size(workspace, 1) * Base.size(workspace, 2)

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the B tensor
abstract type LayoutB{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    K = tile.base.K + tile.offset.K
    N = tile.base.N + tile.offset.N

    d = K
    c = N

    offset = 1 + d + c * Base.size(workspace, 1)

    Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
end

# layout for the C tensor
abstract type LayoutC{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.load(::Type{LayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    N = tile.base.N + tile.offset.N

    a = M ÷ Base.size(workspace, 2)
    b = M % Base.size(workspace, 2)

    c = N

    offset = 1 + a + b * Base.size(workspace, 1) + c * Base.size(workspace, 1) * Base.size(workspace, 2)
    tmp_offset = offset

    strided_over_size = Base.size(workspace, 1)
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[tmp_offset + (i - 1) * strided_over_size])
    end
end

# layout for the D tensor
abstract type LayoutD{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.store!(::Type{LayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    M = tile.base.M + tile.offset.M
    N = tile.base.N + tile.offset.N
    
    a = M ÷ Base.size(workspace, 2)
    b = M % Base.size(workspace, 2)

    c = N

    offset = 1 + a + b * Base.size(workspace, 1) + c * Base.size(workspace, 1) * Base.size(workspace, 2)
    tmp_offset = offset

    strided_over_size = Base.size(workspace, 1)
    for i = 1 : NUMEL
        @inbounds workspace[tmp_offset + (i - 1) * strided_over_size] = value[i].value
    end
end

# implementation using GemmKernels.jl
function gemmkernels_impl(C ;benchmark = false)
    C = CuArray(C)
    D = CuArray(zeros(Float16, (SA, SB, SC)))

    M = SA * SB
    N = SC
    K = SD

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

    # @show conf

    if !benchmark
        GemmKernels.matmul(A, B, C, D, conf;
                        kernel = Kernel.matmul_pipelined
                        )
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed GemmKernels.matmul(A, B, C, D, conf;
                            kernel = Kernel.matmul_pipelined
                        )
            push!(times, time)
        end

        times
    end
end

function gettcontractions_impl(C ;benchmark = false)
    C = CuArray(C)
    D = CuArray(zeros(Float16, (SA, SB, SC)))
    
    # D[A, B, C] = 1 * A[B, D, A] * B[D, C] + 0 * D[A, B, C]

    plan = TensorPlan.ContractionPlan(
        A, ['b', 'd', 'a'], 
        B, ['d', 'c'], 
        D, ['a', 'b', 'c'], 
        D, ['a', 'b', 'c']
    )

    # TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(0.0), D, D)

    if !benchmark
        TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(1.0), C, D)
        D
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(0.0), D, D) 
            push!(times, time)
        end

        times
    end
end

# cuTENSOR implementation
function cutensor_impl(C ;algo = CUDA.CUTENSOR.CUTENSOR_ALGO_DEFAULT, benchmark = false)
    C = CuArray(C)

    plan = CUDA.CUTENSOR.plan_contraction(A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          C, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                          algo = algo,
                                          compute_type = Float16)



    if !benchmark
        CUDA.CUTENSOR.contraction!(1,
                                A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                1,
                                C, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                compute_type = Float16,
                                plan = plan)
        C
    else
        times = []

        for i = 1 : 10000
            synchronize(context())
            time = CUDA.@elapsed CUDA.CUTENSOR.contraction!(1,
                                    A, [ 'b', 'd', 'a' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    B, [ 'd', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    1,
                                    C, [ 'a', 'b', 'c' ], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                                    compute_type = Float16,
                                    plan = plan)
            push!(times, time)
        end

        times
    end
end

# compare
function test(C)
    D_reference = cutensor_impl(C)
    D_gemmkernels = gemmkernels_impl(C)
    D_gettcontractions = gettcontractions_impl(C)

    display(D_reference[1:10, 1:10, 1])
    display(D_gemmkernels[1:10, 1:10, 1])
    display(D_gettcontractions[1:10, 1:10, 1])

    display(@test all(isapprox.(Array(D_reference), Array(D_gemmkernels); rtol = sqrt(eps(Float16)))))
    display(@test all(isapprox.(Array(D_gemmkernels), Array(D_gettcontractions); rtol = sqrt(eps(Float16)))))
    display(@test all(isapprox.(Array(D_reference), Array(D_gettcontractions); rtol = sqrt(eps(Float16)))))
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

    bench_gettcontractions = gettcontractions_impl(benchmark = true)

    for (title, bench) in [("cuTENSOR (let heuristic choose algorithm):", bench_cutensor),
                           ("cuTENSOR (force GETT):", bench_cutensor_gett),
                           ("cuTENSOR (force TGETT):", bench_cutensor_tgett),
                           ("cuTENSOR (force TTGT):", bench_cutensor_ttgt),
                           ("GemmKernels:", bench_gemmkernels),
                           ("GETTContraction:", bench_gettcontractions),
                           ("rmskew: cuTENSOR (let heuristic choose algorithm):", rmskew(bench_cutensor)),
                           ("rmskew: cuTENSOR (force GETT):", rmskew(bench_cutensor_gett)),
                           ("rmskew: cuTENSOR (force TGETT):", rmskew(bench_cutensor_tgett)),
                           ("rmskew: cuTENSOR (force TTGT):", rmskew(bench_cutensor_ttgt)),
                           ("rmskew: GemmKernels:", rmskew(bench_gemmkernels)),
                           ("rmskew: GETTContraction:", rmskew(bench_gettcontractions)),
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

!isinteractive() && (if test_or_bench == false display(test()) else main() end)