using CUDA

include("src/GettContractions.jl")
using .GETT

function main()
    SA, SB, SC, SD = 32, 16, 512, 512;

    plan = PLAN(
        algo = GETT.ALGO_GETT,

        M = SA * SB,
        N = SC,
        K = SD,

        a_MK_strides = ([1, 3], [2]),
        is_a_load_strided = false,
        a_strided_over = [],

        b_KN_strides = ([1], [2]),
        is_b_load_strided = false,
        b_strided_over = [],

        d_MN_strides = ([2, 1], [3]),
        is_d_store_strided = true,
    )

    A = CuArray(rand(Float16, (SB, SD, SA)))
    B = CuArray(rand(Float16, (SD, SC)))
    D = CuArray(zeros(Float16, (SA, SB, SC)))

    GETTCreateLayoutTypes(plan)

    @time GETTContraction(plan, Float16(1.0), A, B, Float16(0.0), D, D)
    @time GETTContraction(plan, Float16(1.0), A, B, Float16(0.0), D, D)
    println()
    time = CUDA.@elapsed GETTContraction(plan, Float16(1.0), A, B, Float16(0.0), D, D)
    @show Float64(time * 1e6)

    println()
    D[1:10, 1:10, 1]
end

isinteractive() || main()