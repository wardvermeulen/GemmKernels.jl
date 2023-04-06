export GETT
module GETT

# GETT: GEMM-like Tensor Tensor contraction
using CUDA
using GemmKernels
using GemmKernels.Layout
using GemmKernels.Tiling
using GemmKernels.TensorLayout
using GemmKernels.TensorPlan

using KernelAbstractions.Extras: @unroll

# TODO LIST
# - [ ] Test if vectorised store works properly
# - [ ] Refactor code (layouts to another file)
# - [ ] Check if the dynamic function invocation error disappears when using a function for
#       the strided load
# - [ ] Also provide a function for the strided store
# - [ ] Experiment with a non-zero C matrix in the run files, and then generalise to LocalLayoutC


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
    N = 16 ÷ sizeof(T)

    ntuple(Val(N)) do i
        VecElement{T}(zero(T))
    end
end

# layout for the D tensor
abstract type LayoutD{T} <: Layout.AlignedColMajor{T} end

@inline function Layout.store!(::Type{LayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
    NUMEL = 16 ÷ sizeof(T)

    for i = 1 : NUMEL
        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        c = N

        a = (M + i - 1) ÷ Base.size(workspace, 2)
        b = (M + i - 1) % Base.size(workspace, 2)

        @inbounds workspace[a + 1, b + 1, c + 1] = value[i].value
    end
end

export GETTCreateLayoutTypes

function GETTCreateLayoutTypes(plan::PLAN)
    # TODO: put all this in another file, something like tensor-layout.jl

    # A tensor layout
    @eval abstract type LocalLayoutA{T} <: Layout.AlignedColMajor{T} end

    (
        TM_strides, TK_strides,
        TM_div, TK_div,
        T_mod,
        GM_mul, GK_mul,
        is_load_strided, strided_over_size
    ) = TensorLayout.precomputeGETTLayoutConstants([16, 512, 32], plan.a_MK_strides, plan.is_a_load_strided, plan.a_strided_over)

    @eval @inline function Layout.load(::Type{LocalLayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        K = tile.base.K + tile.offset.K

        offset = 1

        i = 1
        @unroll for TM_stride in $TM_strides
            stride_offset = (M ÷ ($TM_div)[i]) % ($T_mod)[TM_stride]

            offset += stride_offset * ($GM_mul)[i]
            i += 1
        end

        i = 1
        @unroll for TK_stride in $TK_strides
            stride_offset = (K ÷ ($TK_div)[i]) % ($T_mod)[TK_stride]

            offset += stride_offset * ($GK_mul)[i]
            i += 1
        end

        if ($is_load_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        else
            return TensorLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, $strided_over_size)
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

    plan.gemm_conf = GemmKernels.get_config(
        gemm_shape = (M = plan.M, N = plan.N, K = plan.K),
        operator = Operator.WMMAOp{16, 16, 16, Float16},

        global_a_layout = LocalLayoutA{Float16},
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

    plan.are_types_created = true
end

export GETTContraction

function GETTContraction(
    plan::PLAN,
    α, A::CuArray, B::CuArray,
    β, C::CuArray,
    D::CuArray,
    )

    GemmKernels.matmul(A, B, C, D, plan.gemm_conf;
        kernel = Kernel.matmul_pipelined)
    # time = CUDA.@elapsed GemmKernels.matmul(A, B, C, D, plan.gemm_conf;
    #     kernel = Kernel.matmul_pipelined)
    # @show Float64(time * 1e6)
end

    # if (plan.are_types_created == false || isnothing(plan.gemm_conf) == true)
    #     @show "creating types"
    #     GETTCreateLayoutTypes(plan)
    # end

end