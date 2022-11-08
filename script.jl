using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Test


function main()
    transpose_a = true
    transpose_b = true
    A_type = Float16
    B_type = Float16
    CD_type = Float32
    min_dimension = 128

    @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1]] )
        M = min_dimension
        N = min_dimension
        K = min_dimension
        alpha = convert(A_type, 2)
        beta  = convert(CD_type, 3)

        a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
        b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
        c_h = rand(CD_type, (M, N))

        # Transpose input if necessary
        a_h = transpose_a ? transpose(a_h) : a_h
        b_h = transpose_b ? transpose(b_h) : b_h

        a   = CuArray(a_h)
        b   = CuArray(b_h)
        c   = CuArray(c_h)
        d   = similar(c)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 128, N = 64, K = 64),
                                        operator = Operator.FPUOp{4, 8, 1, CD_type},
                                        compute_warp = (M = 32, N = 32, K = 1),
                                        global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                        global_c_layout = Layout.AlignedColMajor{CD_type},
                                        global_d_layout = Layout.AlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        GemmKernels.matmul(a, b, c, d, conf;
                            transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                            transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                            kernel = Kernel.matmul_testing
                            )

        # Transpose outputs, if necessary
        new_a_h = transpose_a ? transpose(a_h) : a_h
        new_b_h = transpose_b ? transpose(b_h) : b_h

        @test all(isapprox.(beta * c_h, Array(d); rtol = sqrt(eps(Float32))))
        # @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(A_type))))
    end
end

isinteractive() || main()