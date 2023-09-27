using CUDA, GemmKernels
using Octavian

function main()
    M = K = N = 128

    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)

    C_h = zeros(Float32, M, N)
    Octavian.matmul!(C_h, Array(A), Array(B))

    # pow2-sized, 128-bit aligned inputs, so we can use aligned layouts.
    # we don't have transposed inputs, so everything is column major.
    @assert stride(A, 2) % 16 == 0
    global_a_layout = Layout.UnsafeAlignedColMajor{eltype(A)}
    @assert stride(B, 2) % 16 == 0
    global_b_layout = Layout.UnsafeAlignedColMajor{eltype(B)}
    # we want to do a simple C = A * B, so no need to load C first.
    global_c_layout = Layout.Zero{eltype(C)}
    @assert stride(C, 2) % 16 == 0
    global_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # shared layouts are similar.
    # the frequently-accessed a/b shmems are padded to avoid bank conflicts.
    shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(A)},8}
    shared_b_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(B)},8}
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # we use the single-stage kernel, for simplicity
    kernel = Kernel.matmul_singlestage


    # ! Set these variables
    OPERATOR_M = 4
    OPERATOR_N = 8
    OPERATOR_K = 1

    # or 8, 16, 4
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    op_shape = (M=OPERATOR_M, N=OPERATOR_N, K=OPERATOR_K)
    block_shape = (M=BLOCK_M, N=BLOCK_N, K=BLOCK_K)

    compute_type = promote_type(eltype(A), eltype(B))
    operator = Operator.FPUOp{8, 8, 4, 8, 4, 1, compute_type, eltype(C)}

    println("Operator: ", operator)

    conf = GemmKernels.get_config(;
        gemm_shape=(; M, N, K), block_shape, operator, global_a_layout, global_b_layout, global_c_layout, global_d_layout,
        shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout, is_a_col_major=true,
        is_b_col_major=true
    )

    C .= 0
    GemmKernels.matmul(conf, A, B, C, C; kernel)

    if !(Array(C) ≈ C_h)
        println("Invalid result")
        @error "Invalid result"
    end

    display(Array(C)[1:10,1:10])
    display(C_h[1:10,1:10])
end

isinteractive() || main()