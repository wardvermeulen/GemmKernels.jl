using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Test

function main()
    A_type = Float32
    B_type = Float32
    CD_type = Float32

    transpose_a = false
    transpose_b = false

    i = 11
    (M, N, K) = (2^i, 2^i, 2^i)

    (OP_M, OP_N, OP_K) = (8, 8, 1)

    alpha = convert(A_type, 1)
    beta  = convert(CD_type, 1)

    a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
    # a_h = rand(A_type, (M, K))
    b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
    # b_h = rand(B_type, (K, N))
    c_h = rand(CD_type, (M, N))
    # d_h = similar(c_h)

    # Transpose input if necessary
    a_h = transpose_a ? transpose(a_h) : a_h
    b_h = transpose_b ? transpose(b_h) : b_h

    a   = CuArray(a_h)
    b   = CuArray(b_h)
    c   = CuArray(c_h)
    d   = similar(c)


    conf = GemmKernels.get_config(
                                    gemm_shape = (M = M, N = N, K = K),
                                    # TODO: Does not work with N = 64, investigate.
                                    block_shape = (M = 128, N = 128, K = 32),
                                    operator = Operator.FPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
                                    # operator = Operator.WMMAOp{16, 16, 16, CD_type},
                                    global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                    global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                    global_c_layout = Layout.AlignedColMajor{CD_type},
                                    global_d_layout = Layout.AlignedColMajor{CD_type},

                                    is_a_col_major = !transpose_a,
                                    is_b_col_major = !transpose_b,
                                    )
    @show conf.block_shape
    @show conf.shared_a_layout

    # CUDA.CUBLAS.gemmEx!(
    #     'N', 'N',
    #     alpha,
    #     a,
    #     b,
    #     beta,
    #     c
    # )

    GemmKernels.matmul(a, b, c, d, conf;
                        transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                        transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                        kernel = Kernel.matmul_pipelined
                        )

    # Transpose outputs, if necessary
    new_a_h = transpose_a ? transpose(a_h) : a_h
    new_b_h = transpose_b ? transpose(b_h) : b_h

    # tropical mma
    # for i in 1 : M
    #     for j in 1 : N
    #         for k in 1 : K
    #             c_h[i, j] = max(a_h[i, k] + b_h[k, j], c_h[i, j]) 
    #         end
    #     end
    # end

    compare = alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h
    # compare = c_h
    host = Array(d)

    display(host[1:10, 1:10])
    display(compare[1:10, 1:10])

    @test all(isapprox.(compare, host; rtol = sqrt(eps(A_type))))

    # @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(c); rtol = sqrt(eps(A_type))))
end

function test()
    @testset "FPU GEMM $(A_type)*$(B_type)+$(CD_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))" for 
        (A_type, B_type, CD_type, min_dimension) in [
            (Float16, Float16, Float32, 128), (Float32, Float32, Float32, 128), (Float32, Float32, Float64, 128), (Float64, Float64, Float64, 128),
            (Int16, Int16, Int16, 128), (Int32, Int32, Int32, 128), (Int64, Int64, Int64, 128), 
        ], 
        transpose_a = [false, true], 
        transpose_b = [false, true], 
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2, 2, 1], [1, 1, 2], [2, 2, 2]], [[2048, 2048, 2048]])
            alpha = convert(A_type, 2)
            beta  = convert(CD_type, 3)

            if A_type <: Integer
                a_h = rand(A_type, (M, K))
                b_h = rand(B_type, (K, N))
            else
                a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
                b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            end
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
                                            block_shape = (M = 64, N = 64, K = 32),
                                            operator = Operator.FPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
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
                                kernel = Kernel.matmul_pipelined
                                )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h
            
            if A_type <: Integer
                @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d)))
            else
                @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(A_type))))
            end
        end
    end

end

function test2()
    @testset "FPU GEMM OPERATOR SHAPE ($(OP_M), $(OP_N), $(OP_K)) (NN, NT, TN, TT)" for (OP_M, OP_N, OP_K) in [
            (4, 8, 1), (8, 8, 1), (4, 16, 1), (4, 8, 2), (8, 16, 2) 
        ]
        @testset "$( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )" for (transpose_a, transpose_b) in [(false, false), (false, true), (true, false), (true, true)]
            (M, N, K) = (128, 128, 128)
            (A_type, B_type, CD_type) = (Float32, Float32, Float32)

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
                                            # TODO: Does not work with N = 64, investigate.
                                            block_shape = (M = 128, N = 64, K = 32),
                                            operator = Operator.FPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
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
                                kernel = Kernel.matmul_pipelined
                                )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(A_type))))
        end
    end
end

function test3()
    @testset "WMMA GEMM (A = diagonal, B = $( !transpose_b ? 'N' : 'T' ))" for transpose_b = [false, true]
        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256), (4096, 4096, 4096)]
            @assert M == K "Diagonal only supports square A matrix (M == K)"

            transpose_a = false

            a_h = rand(Float16, M);
            b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
            c_h = rand(Float32, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMAOp{16, 16, 16, Float32},
                                          global_a_layout = Layout.Diagonal{Float16},
                                          global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                          global_c_layout = Layout.AlignedColMajor{Float32},
                                          global_d_layout = Layout.AlignedColMajor{Float32},

                                          shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(a, b, c, d, conf)

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(Float32.(Diagonal(new_a_h)) * Float32.(new_b_h) + c_h, Array(d); rtol = sqrt(eps(Float16))))
        end
    end
end

function test4()
    @testset "TROPICAL GEMM $(A_type)*$(B_type)+$(CD_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))" for 
        (A_type, B_type, CD_type, min_dimension) in [
            (Float32, Float32, Float32, 128)
        ], 
        transpose_a = [false, true], 
        transpose_b = [false, true], 
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2, 2, 1], [1, 1, 2], [2, 2, 2]])
            a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
            b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            c_h = rand(CD_type, (M, N))
            d_h = similar(c_h)

            for i in 1 : M
                for j in 1 : N
                    d_h[i, j] = c_h[i, j]
                    for k in 1 : K
                        d_h[i, j] = max(a_h[i, k] + b_h[k, j], d_h[i, j]) 
                    end
                end
            end

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            conf = GemmKernels.get_config(
                                            gemm_shape = (M = M, N = N, K = K),
                                            block_shape = (M = 64, N = 64, K = 32),
                                            operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
                                            global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                            global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                            global_c_layout = Layout.AlignedColMajor{CD_type},
                                            global_d_layout = Layout.AlignedColMajor{CD_type},

                                            is_a_col_major = !transpose_a,
                                            is_b_col_major = !transpose_b,
                                            )

            GemmKernels.matmul(a, b, c, d, conf; kernel = Kernel.matmul_pipelined)

            # # Transpose outputs, if necessary
            # new_a_h = transpose_a ? transpose(a_h) : a_h
            # new_b_h = transpose_b ? transpose(b_h) : b_h
            
            @test all(isapprox.(d_h, Array(d); rtol = sqrt(eps(A_type))))
        end
    end
end

isinteractive() || main()