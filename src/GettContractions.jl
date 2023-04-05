module GETT

# GETT: GEMM-like Tensor Tensor contraction
using CUDA
using GemmKernels
using GemmKernels.Layout
using GemmKernels.Tiling

# TODO LIST
# - [ ] Test if vectorised store works properly
# - [ ] Refactor code (layouts to another file)
# - [ ] Check if the dynamic function invocation error disappears when using a function for
#       the strided load
# - [ ] Also provide a function for the strided store
# - [ ] Experiment with a non-zero C matrix in the run files, and then generalise to LocalLayoutC

export ALGO

@enum ALGO::Int32 begin
    ALGO_DEFAULT_PATIENT = -6
    ALGO_GETT = -4
    ALGO_TGETT = -3
    ALGO_TTGT = -2
    ALGO_DEFAULT = -1
end

export PLAN

struct PLAN
    algo::ALGO

    M::Int32
    N::Int32
    K::Int32

    # A tensor plan variables
    a_MK_strides::Tuple{Vector{Int32}, Vector{Int32}}
    is_a_load_strided::Bool
    a_strided_over::Vector{Int32}

    # B tensor plan variables
    b_KN_strides::Tuple{Vector{Int32}, Vector{Int32}}
    is_b_load_strided::Bool
    b_strided_over::Vector{Int32}

    # D tensor plan variables
    d_MN_strides::Tuple{Vector{Int32}, Vector{Int32}}
    is_d_store_strided::Bool
end

export GETTContraction

function GETTContraction(
    plan::PLAN,
    α, A::CuArray, B::CuArray,
    β, C::CuArray,
    D::CuArray,
    )

    # TODO: put all this in another file, something like tensor-layout.jl

    # A tensor layout
    @eval abstract type LocalLayoutA{T} <: Layout.AlignedColMajor{T} end

    # ? HACK: Convert vectors to tuples
    local_a_MK_strides = (
        Tuple(x for x in plan.a_MK_strides[1]),
        Tuple(x for x in plan.a_MK_strides[2])
    )

    local_a_strided_over = Tuple(x for x in plan.a_strided_over)

    (M_GEMM_strides, K_GEMM_strides, is_load_strided, strided_over) = (
        local_a_MK_strides[1], 
        local_a_MK_strides[2], 
        Int(plan.is_a_load_strided), 
        local_a_strided_over
    )

    @eval @inline function Layout.load(::Type{LocalLayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        K = tile.base.K + tile.offset.K

        tmp_offset = 1

        # ? UGLY: The same thing twice 
        divisor = 1

        for GEMM_stride in $M_GEMM_strides
            stride_offset = (M ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            tmp_offset += stride_offset * multiplicator
        end

        divisor = 1

        for GEMM_stride in $K_GEMM_strides
            stride_offset = (K ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            tmp_offset += stride_offset * multiplicator
        end

        # This is a hack to prevent the 'unsupported dynamic function invocation' error.
        offset = tmp_offset

        if ($is_load_strided == false)
            # Vectorised load

            # For some reason, the hack is not needed were this vectorised load used.
            # Probably because the offset is calculated before it is dispatched to the function.
            # Solution: make a separate function for the strided load?
            # TODO: Experiment with this.
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        else
            # Strided load

            # Use a temporary variable to avoid dynamic invocation
            tmp_strided_over_size = 1

            for stride_idx in $strided_over
                if (stride_idx == 0)
                    tmp_strided_over_size *= 1
                else
                    tmp_strided_over_size *= Base.size(workspace, stride_idx)
                end
            end

            # The same hack used for the offset is also applied here.
            strided_over_size = tmp_strided_over_size

            x = ntuple(Val(NUMEL)) do i
                @inbounds VecElement{T}(workspace[offset + (i - 1) * strided_over_size])
            end

            return x
        end
    end

    # B tensor layout
    @eval abstract type LocalLayoutB{T} <: Layout.AlignedColMajor{T} end

    local_b_KN_strides = (
        Tuple(x for x in plan.b_KN_strides[1]),
        Tuple(x for x in plan.b_KN_strides[2])
    )
    
    local_b_strided_over = Tuple(x for x in plan.b_strided_over)

    (K_GEMM_strides, N_GEMM_strides, is_load_strided, strided_over) = (
        local_b_KN_strides[1],
        local_b_KN_strides[2],
        Int(plan.is_b_load_strided),
        local_b_strided_over
    )

    @eval @inline function Layout.load(::Type{LocalLayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        K = tile.base.K + tile.offset.K
        N = tile.base.N + tile.offset.N

        tmp_offset = 1

        # ? UGLY: The same thing twice 
        divisor = 1

        for GEMM_stride in $K_GEMM_strides
            stride_offset = (K ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            tmp_offset += stride_offset * multiplicator
        end

        divisor = 1

        for GEMM_stride in $N_GEMM_strides
            stride_offset = (N ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            tmp_offset += stride_offset * multiplicator
        end

        offset = tmp_offset

        if ($is_load_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        else
            tmp_strided_over_size = 1

            for stride_idx in $strided_over
                if (stride_idx == 0)
                    tmp_strided_over_size *= 1
                else
                    tmp_strided_over_size *= Base.size(workspace, stride_idx)
                end
            end

            strided_over_size = tmp_strided_over_size

            x = ntuple(Val(NUMEL)) do i
                @inbounds VecElement{T}(workspace[offset + (i - 1) * strided_over_size])
            end

            return x
        end
    end

    # C tensor layout
    # TODO: Add non-zero variant
    @eval abstract type LocalLayoutC{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
        N = 16 ÷ sizeof(T)

        ntuple(Val(N)) do i
            VecElement{T}(zero(T))
        end
    end

    # D tensor layout
    @eval abstract type LocalLayoutD{T} <: Layout.AlignedColMajor{T} end

    # ? HACK: Convert vectors to tuples
    local_d_MN_strides = (
        Tuple(x for x in plan.d_MN_strides[1]),
        Tuple(x for x in plan.d_MN_strides[2])
    )

    (M_GEMM_strides, N_GEMM_strides, is_store_strided) = (local_d_MN_strides[1], local_d_MN_strides[2], Int(plan.is_d_store_strided))

    @eval @inline function Layout.store!(::Type{LocalLayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        # TODO: Add vectorised store

        if ($is_store_strided == false)
            return Layout.vstorea(value, Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        end

        for i = 1 : NUMEL
            offset = 1

            # ? UGLY: The same thing twice 
            divisor = 1

            for GEMM_stride in $M_GEMM_strides
                stride_offset = ((M + i - 1) ÷ divisor) % Base.size(workspace, GEMM_stride)
                divisor *= Base.size(workspace, GEMM_stride)

                multiplicator = 1
                for i = 1 : (GEMM_stride - 1)
                    multiplicator *= Base.size(workspace, i)
                end

                offset += stride_offset * multiplicator
            end

            divisor = 1

            for GEMM_stride in $N_GEMM_strides
                stride_offset = (N ÷ divisor) % Base.size(workspace, GEMM_stride)
                divisor *= Base.size(workspace, GEMM_stride)

                multiplicator = 1
                for i = 1 : (GEMM_stride - 1)
                    multiplicator *= Base.size(workspace, i)
                end

                offset += stride_offset * multiplicator
            end

            @inbounds workspace[offset] = value[i].value
        end
    end

    conf = GemmKernels.get_config(
        gemm_shape = (M = plan.M, N = plan.N, K = plan.K),
        operator = Operator.WMMAOp{16, 16, 16, Float16},

        global_a_layout = LocalLayoutA{Float16},
        global_b_layout = LocalLayoutB{Float16},
        global_c_layout = LocalLayoutC{Float16},
        global_d_layout = LocalLayoutD{Float16},

        shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
        shared_b_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
        shared_c_layout = Layout.AlignedColMajor{Float16},
        shared_d_layout = Layout.AlignedColMajor{Float16},

        is_a_col_major = true,
        is_b_col_major = true,
    )

    GemmKernels.matmul(A, B, C, D, conf;
        kernel = Kernel.matmul_singlestage)

    return D
end

end