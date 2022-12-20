using NVTX
using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra

function benchmark_fpu()
    transpose_a = false
    transpose_b = false
    A_type = Float16
    B_type = Float16
    CD_type = Float16
    min_dimension = 2048
    OP_M = 4
    OP_N = 8
    (M, N, K) = min_dimension .*[1, 1, 1]

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
                                    # TODO: Does not work with N = 128, investigate.
                                    block_shape = (M = 128, N = 64, K = 64),
                                    operator = Operator.FPUOp{OP_M, OP_N, 1, CD_type},
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

    return

end

function benchmark_wmma()
    transpose_a = false
    transpose_b = false
    A_type = Float16
    B_type = Float16
    CD_type = Float16
    min_dimension = 2048
    OP_M = 4
    OP_N = 8
    (M, N, K) = min_dimension .*[1, 1, 1]

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
                                    # TODO: Does not work with N = 128, investigate.
                                    block_shape = (M = 128, N = 64, K = 64),
                                    operator = Operator.WMMAOp{16, 16, 16, CD_type},
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

    return

end
 
function profiler_main()
    benchmark_wmma()
    benchmark_fpu()

    CUDA.@profile begin
        NVTX.@mark "FPU 1" benchmark_fpu()
        NVTX.@mark "FPU 2" benchmark_fpu()

        NVTX.@mark "WMMA 1" benchmark_wmma()
        NVTX.@mark "WMMA 2" benchmark_wmma()
    end
end