using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using NVTX
using Test

CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)

transpose_a = false
transpose_b = false

A_type = Float32
B_type = Float32
CD_type = Float32

(M, N, K) = 256 .* [1, 1, 1]

function benchmark_fpu()
    operator = Operator.FPUOp{8, 8, 1, CD_type}

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
                                    block_shape = (M = 32, N = 32, K = 32),
                                    operator = operator,
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

    return d
end

function benchmark_cublas()
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

    CUDA.CUBLAS.gemmEx!(
        !transpose_a ? 'N' : 'T',
        !transpose_b ? 'N' : 'T',
        alpha,
        a,
        b,
        beta,
        c
    )

    return c
end
 
function profiler_main()
    r1 = benchmark_fpu()
    r2 = benchmark_cublas()

    CUDA.@profile begin
        NVTX.@mark "FPU 1" 
        benchmark_fpu()
        NVTX.@mark "FPU 2" 
        benchmark_fpu()

        NVTX.@mark "BENCHMARK 1" 
        benchmark_cublas()
        NVTX.@mark "BENCHMARK 2" 
        benchmark_cublas()
    end
end

isinteractive() || profiler_main()