using CUDA

include("src/GettContractions.jl")
using .GETT

function main()
    SA, SB, SC, SD = 64, 32, 2048, 2048;

    plan = PLAN(
        # algo
        GETT.ALGO_GETT,

        # M
        SA * SB,
        # N
        SC,
        # K
        SD,

        2,

        # A_MK_strides
        ([1, 3], [2]),
        # is_a_load_strided
        false,

        # B_KN_strides
        ([1], [2]),
        # is_b_load_strided
        false,

        # d_MN_strides
        ([1, 3], [2]),
        # is_d_store_strided
        true,
    )


    A = CuArray(rand(Float16, (SB, SD, SA)))
    B = CuArray(rand(Float16, (SD, SC)))
    D = CuArray(zeros(Float16, (SA, SB, SC)))

    GETTContraction(plan, Float16(1.0), A, B, Float16(0.0), D, D)[1:10, 1:10, 1]
end

isinteractive() || main()