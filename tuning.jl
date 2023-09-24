using CUDA, GemmKernels
using Hyperopt
using Octavian

# we don't need super-accurate timings
const samples = 250

function main()
    M = K = N = 2048

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
    shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(A)}, 8}
    shared_b_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(B)}, 8}
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # we use the single-stage kernel, for simplicity
    kernel = Kernel.matmul_singlestage

    # TODO: compute_warp is partially hardcoded in config.jl, requiring M>=4 and N >=2
    # TODO: tune warps_per_block (which may affect correctness)

    total = 0
    attempts = 0
    benchmarks = 0

    operator_larger_than_block = 0
    operator_base_shape = 0
    localarray_size_limits = 0

    ho = @hyperopt for i = 1000,
                       OPERATOR_M = 2 .^ (1:4),
                       OPERATOR_N = 2 .^ (1:4),
                       OPERATOR_K = 2 .^ (1:4),
                       BLOCK_M = 2 .^ (1:8),
                       BLOCK_N = 2 .^ (1:8),
                       BLOCK_K = 2 .^ (1:8)
        op_shape = (M = OPERATOR_M, N = OPERATOR_N, K = OPERATOR_K)
        block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K)
        total += 1

        # validate the operator shape
        ## may not be larger than the block shape
        if op_shape.M > block_shape.M ||
           op_shape.N > block_shape.N ||
           op_shape.K > block_shape.K
           operator_larger_than_block += 1
            return Inf
        end
        ## the FPU operator's base shape is 4x8x1. can we relax this?
        if op_shape.M < 4 || op_shape.M % 4 != 0 ||
           op_shape.N < 8 || op_shape.N % 8 != 0
           operator_base_shape += 1
            return Inf
        end
        ## LocalArray size limits (these are the ways FPUOp instantiates them)
        if op_shape.M÷4 * op_shape.K >= 32 ||
           op_shape.K * op_shape.N÷8 >= 32 ||
           op_shape.M÷4 * op_shape.N÷8 >= 32
           localarray_size_limits += 1
            # in isolation, i.e. https://github.com/JuliaGPU/GemmKernels.jl/issues/99,
            # a LocalArray of 32 elements is fine, but in the context of the kernel,
            # it's too large. I don't know why.
            return Inf
        end


        # TODO: fix warps_per_block and no_warps
        # if block_shape.M ÷ op_shape.M < 4 ||
        #    block_shape.N ÷ op_shape.N < 2 ||
        #    block_shape.K < 8
        #     return Inf
        # end


        # validate the block shape
        ## needs to exactly covers the inputs, so that we can use the unsafe layouts.
        if M % block_shape.M != 0 || N % block_shape.N != 0 || K % block_shape.K != 0
            return Inf
        end
        ## need to be 128-bit aligned so that we can perform vectorized loads
        # XXX: is this correct?
        if block_shape.M * sizeof(eltype(A)) % 16 != 0 ||
           block_shape.N * sizeof(eltype(B)) % 16 != 0 ||
           block_shape.K * sizeof(eltype(C)) % 16 != 0
            return Inf
        end

        compute_type = promote_type(eltype(A), eltype(B))
        operator = Operator.FPUOp{OPERATOR_M, OPERATOR_N, OPERATOR_K, compute_type, eltype(C)}

        conf = GemmKernels.get_config(;
            gemm_shape = (; M, N, K), block_shape, operator,

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = true,
            is_b_col_major = true
        )

        ## another LocalArray size limit, these are in the kernel
        num_fragments_m = conf.compute_warp.M ÷ conf.compute_op_shape.M
        num_fragments_n = conf.compute_warp.N ÷ conf.compute_op_shape.N
        if num_fragments_m * num_fragments_n >= 32
            localarray_size_limits += 1
            return Inf
        end

        try
            # warm-up & correctness check
            attempts += 1
            C .= 0
            GemmKernels.matmul(conf, A, B, C, C; kernel)
            if !(Array(C) ≈ C_h)
                @warn "Configuration produced invalid result: $conf ($BLOCK_M, $BLOCK_N, $BLOCK_K)"
                return Inf
            end

            # benchmark
            benchmarks += 1
            device_synchronize()
            GC.gc(true)
            timings = zeros(samples)
            for i in 1:samples
                synchronize(stream())
                timings[i] = CUDA.@elapsed GemmKernels.matmul(conf, A, B, C, C; kernel)
            end

            minimum(timings)
        catch err
            if isa(err, CuError)
                @error "Configuration failed: $conf"
                rethrow()
            end
            @info "Skipping configuration: $conf\n" * sprint(Base.showerror, err)
            # TODO: introduce GemmKernels.ConfigError, to differentiate from e.g.
                #   compilation errors, which we want to report verbosely.
            Inf
        end
    end

    skips = total - attempts
    errors = attempts - benchmarks
    println("Out of $total configurations, $skips ($(round(100*skips/total; digits=1))%) were skipped, $errors ($(round(100*errors/total; digits=1))%) errored, and $benchmarks ($(round(100*benchmarks/total; digits=1))%) were actually tested.")

    println(operator_larger_than_block) # 484
    println(operator_base_shape) # 356
    println(localarray_size_limits) # 60
    # tot 900

    ho
end

isinteractive() || println(main())
