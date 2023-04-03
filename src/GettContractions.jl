# GETT: GEMM-like Tensor Tensor contraction
using CUDA
using GemmKernels.Layout

@enum ALGO::Int32 begin
    DEFAULT_PATIENT = -6
    GETT = -4
    TGETT = -3
    TTGT = -2
    DEFAULT = -1
end

struct PLAN
    algo::ALGO

    # A tensor plan variables
    A_M_strides::Vector{Int32}
    A_M_offsetdim::Vector{Int32}

    A_K_strides::Vector{Int32}
    A_K_offsetdim::Vector{Int32}

    is_A_access_strided::Bool

    # B tensor plan variables
    B_K_strides::Vector{Int32}
    B_K_offsetdim::Vector{Int32}

    B_N_strides::Vector{Int32}
    B_N_offsetdim::Vector{Int32}

    is_B_access_strided::Bool

    # D tensor plan variables
    D_M_strides::Vector{Int32}
    D_M_offsetdim::Vector{Int32}

    D_N_strides::Vector{Int32}
    D_N_offsetdim::Vector{Int32}

    is_D_access_strided::Bool
end

basic_plan = PLAN(
    DEFAULT,
    [1, 2], [1, 2, 3],
    [1], [1, 2, 3],
    false,
    [1], [1, 2, 3],
    [1], [1, 2, 3],
    false,
    [2, 1], [1, 2, 3],
    [1], [1, 2, 3],
    true
)


function GETTContraction(
    plan::PLAN,
    α, A::CuArray, B::CuArray,
    β, C::CuArray,
    D::CuArray,
    workspace, workspace_size::UInt64)

    # Start with most simple version:
    # (Einstein notation): D[a,b,c] = α * A[b,d,a] * B[d,c] + β * C[a,b,c] (with β = 0)

    # TODO: put all this in another file, something like tensor-layout.jl
    @eval abstract type LocalLayoutA{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        K = tile.base.K + tile.offset.K

        offset = 1

        # TODO: maybe unroll?
        # TODO: pick better variable names
        for (stride, strides_plan, offsetdim) in [(M, plan.A_M_strides, plan.A_M_offsetdim), (K, plan.A_K_strides, plan.A_K_offsetdim)]
            divisor = 1

            for (stride_idx, offset_idx) in zip((strides_plan, offsetdim))
                stride_offset = (stride ÷ divisor) % Base.size(workspace, stride_idx)
                divisor *= Base.size(workspace, stride_idx)

                offset += stride_offset * Base.size(workspace, offset_idx)
            end
        end

        if (is_A_access_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        end
    end

    @eval abstract type LocalLayoutB{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        K = tile.base.K + tile.offset.K
        N = tile.base.N + tile.offset.N

        offset = 1

        for (stride, strides_plan, offsetdim) in [(K, plan.B_K_strides, plan.B_K_offsetdim), (N, plan.B_N_strides, plan.B_N_offsetdim)]
            divisor = 1

            for (stride_idx, offset_idx) in zip((strides_plan, offsetdim))
                stride_offset = (stride ÷ divisor) % Base.size(workspace, stride_idx)
                divisor *= Base.size(workspace, stride_idx)

                offset += stride_offset * Base.size(workspace, offset_idx)
            end
        end

        if (is_B_access_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        end
    end

    @eval abstract type LocalLayoutC{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{LocalLayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        ntuple(Val(N)) do i
            @inbounds VecElement{T}(zero(T))
        end
    end

    @eval abstract type LocalLayoutD{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.store!(::Type{LocalLayoutD{T}}, workspace, tile::Tile{size}, value) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        for i = 1:NUMEL

            for (stride_nr, (stride, strides_plan, offsetdim)) in enumerate([(M, plan.D_M_strides, plan.D_M_offsetdim), (N, plan.D_N_strides, plan.D_N_offsetdim)])

                divisor = 1

                if stride_nr == 1
                    stride += (i - 1)
                end

                for (stride_idx, offset_idx) in zip((strides_plan, offsetdim))
                    stride_offset = (stride ÷ divisor) % Base.size(workspace, stride_idx)
                    divisor *= Base.size(workspace, stride_idx)

                    offset += stride_offset * Base.size(workspace, offset_idx)
                end

                if (plan.is_D_access_strided == true)
                    @inbounds workspace[offset] = value[i].value
                end

            end

        end
    end

end
