module GETT

# GETT: GEMM-like Tensor Tensor contraction
using CUDA
using GemmKernels
using GemmKernels.Layout
using GemmKernels.Tiling

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

    TEST::Int32

    # A tensor plan variables
    a_MK_strides::Tuple{Vector{Int32}, Vector{Int32}}
    is_a_load_strided::Bool

    # B tensor plan variables
    b_KN_strides::Tuple{Vector{Int32}, Vector{Int32}}
    is_b_load_strided::Bool

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

    # ? HACK: Convert vectors to tuples
    local_a_MK_strides = (
        Tuple(x for x in plan.a_MK_strides[1]),
        Tuple(x for x in plan.a_MK_strides[2])
    )

    @eval abstract type LocalLayoutA{T} <: Layout.AlignedColMajor{T} end

    (M_GEMM_strides, K_GEMM_strides, is_load_strided) = (local_a_MK_strides[1], local_a_MK_strides[2], Int(plan.is_a_load_strided))

    @eval @inline function Layout.load(::Type{LocalLayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        K = tile.base.K + tile.offset.K

        offset = 1

        # ? UGLY: The same thing twice 
        divisor = 1

        for GEMM_stride in $M_GEMM_strides
            stride_offset = (M ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            offset += stride_offset * multiplicator
        end

        divisor = 1

        for GEMM_stride in $K_GEMM_strides
            stride_offset = (K ÷ divisor) % Base.size(workspace, GEMM_stride)
            divisor *= Base.size(workspace, GEMM_stride)

            multiplicator = 1
            for i = 1 : (GEMM_stride - 1)
                multiplicator *= Base.size(workspace, i)
            end

            offset += stride_offset * multiplicator
        end

        # TODO: Add strided load
        if ($is_load_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        end
    end

    @eval abstract type LocalLayoutB{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        K = tile.base.K + tile.offset.K
        N = tile.base.N + tile.offset.N

        d = K
        c = N

        offset = 1 + d + c * Base.size(workspace, 1)

        Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    end

    # @eval @inline function Layout.load(::Type{LocalLayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
    #     NUMEL = 16 ÷ sizeof(T)

    #     K = tile.base.K + tile.offset.K
    #     N = tile.base.N + tile.offset.N

    #     offset = 1

    #     for (stride, strides_plan, offsetdim) in [(K, plan.B_K_strides, plan.B_K_offsetdim), (N, plan.B_N_strides, plan.B_N_offsetdim)]
    #         divisor = 1

    #         for (stride_idx, offset_idx) in zip((strides_plan, offsetdim))
    #             stride_offset = (stride ÷ divisor) % Base.size(workspace, stride_idx)
    #             divisor *= Base.size(workspace, stride_idx)

    #             offset += stride_offset * Base.size(workspace, offset_idx)
    #         end
    #     end

    #     if (!plan.is_B_access_strided)
    #         return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
    #     end
    # end

    @eval abstract type LocalLayoutC{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
        N = 16 ÷ sizeof(T)

        ntuple(Val(N)) do i
            VecElement{T}(zero(T))
        end
    end

    # @eval @inline function Layout.load(::Type{LocalLayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
    #     NUMEL = 16 ÷ sizeof(T)

    #     ntuple(Val(N)) do i
    #         @inbounds VecElement{T}(zero(T))
    #     end
    # end

    @eval abstract type LocalLayoutD{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.store!(::Type{LocalLayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
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

    # @eval @inline function Layout.store!(::Type{LocalLayoutD{T}}, workspace, tile::Tile{size}, value) where {T, size}
    #     NUMEL = 16 ÷ sizeof(T)

    #     M = tile.base.M + tile.offset.M
    #     N = tile.base.N + tile.offset.N

    #     for i = 1:NUMEL

    #         for (stride_nr, (stride, strides_plan, offsetdim)) in enumerate([(M, plan.D_M_strides, plan.D_M_offsetdim), (N, plan.D_N_strides, plan.D_N_offsetdim)])

    #             divisor = 1

    #             if stride_nr == 1
    #                 stride += (i - 1)
    #             end

    #             for (stride_idx, offset_idx) in zip((strides_plan, offsetdim))
    #                 stride_offset = (stride ÷ divisor) % Base.size(workspace, stride_idx)
    #                 divisor *= Base.size(workspace, stride_idx)

    #                 offset += stride_offset * Base.size(workspace, offset_idx)
    #             end

    #             if (plan.is_D_access_strided == true)
    #                 @inbounds workspace[offset] = value[i].value
    #             end

    #         end

    #     end
    # end

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