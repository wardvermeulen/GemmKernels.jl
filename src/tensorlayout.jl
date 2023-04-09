export TensorLayout
module TensorLayout

using CUDA
using GemmKernels.Layout
using GemmKernels.Tiling
using KernelAbstractions.Extras: @unroll

@inline function sloada(::Type{Layout.Vec{NUMEL, T}}, workspace, offset::Int, strided_over_size::Int) where {NUMEL, T}
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[offset + (i - 1) * strided_over_size])
    end
end

# TODO: Write test. One example given below.
# precomputeGETTLayoutConstants([16, 512, 32], ([1, 3], [2]))
# result: ((1, 3), (2,), (1, 16), (1,), (16, 512, 32), (1, 8192), (16,))

# TODO: Write docstring.

export precomputeGETTLayoutConstants

function precomputeGETTLayoutConstants(
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_or_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    # 1. Convert the tensor strides from Tuple{Vector{Int}, Vector{int}}  to two separate 
    # Tuple{Int, Int, ...}.
    # ? @eval cannot work with Vector, but can work with Tuple, because the size of Tuple is 
    # ? known at compile time. 

    # → For the A matrix this will contain the tensor strides corresponding to the M stride.
    # e.g. for D[A, B, C] = A[B, D, A] * B[D, C] this will be (1, 3), since B and A belong to the
    # M stride.
    T1_strides = Tuple(x for x in T_strides[1])
    # → analogous, T2_stride will be equal to (2,) for the above example, since D belongs to the
    # K stride.
    T2_strides = Tuple(x for x in T_strides[2])


    # 2. Precompute the divisors used to calculate the tensor stride offsets.
    # → T1_stride_offset = (M ÷ T1_div[i]) % T1_mod[i] 
    T1_div = Vector{Int}(undef, length(T1_strides))
    div = 1
    for (idx, stride_idx) in enumerate(T1_strides)
        T1_div[idx] = div
        div *= T_strides_sizes[stride_idx]
    end
    T1_div = Tuple(x for x in T1_div)

    # 2b. Do the same for T2_stride.
    T2_div = Vector{Int}(undef, length(T2_strides))
    div = 1
    for (idx, stride_idx) in enumerate(T2_strides)
        T2_div[idx] = div
        div *= T_strides_sizes[stride_idx]
    end
    T2_div = Tuple(x for x in T2_div)


    # 3. Precompute the moduli used to calculate the tensor stride offsets.
    # These are simply the sizes of the tensor strides. Again, converted to Tuple{Int, Int, ...}.
    T_mod = Tuple(x for x in T_strides_sizes)


    # 4. Precompute the multiplicative terms used to calculate the GEMM stride offsets.
    # → offset += T1_stride_offset * G1_mul[i]
    G1_mul = Vector{Int}(undef, length(T1_strides))
    for (idx, stride_idx) in enumerate(T1_strides)
        G1_mul[idx] = 1
        for j = 1 : (stride_idx - 1) 
            G1_mul[idx] *= T_strides_sizes[j]
        end
    end
    G1_mul = Tuple(x for x in G1_mul)

    # 4b. Do the same for G2_mul.
    G2_mul = Vector{Int}(undef, length(T2_strides))
    for (idx, stride_idx) in enumerate(T2_strides)
        G2_mul[idx] = 1
        for j = 1 : (stride_idx - 1) 
            G2_mul[idx] *= T_strides_sizes[j]
        end
    end
    G2_mul = Tuple(x for x in G2_mul)

    # 5. Convert the Bool to an Int.
    is_load_or_store_strided = Int(is_load_or_store_strided)

    # 5.b If the load or store is strided, then precompute the size of the dimensions to stride 
    # over.
    strided_over_size = 1
    if (isnothing(load_or_store_strided_over) == false && is_load_or_store_strided == true)
        for stride_idx in load_or_store_strided_over
            strided_over_size *= T_strides_sizes[stride_idx]
        end
    end

    return (
        T1_strides, T2_strides,
        T1_div, T2_div,
        T_mod,
        G1_mul, G2_mul,
        is_load_or_store_strided, strided_over_size,
    )
end

function createALayout(
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_or_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TM_strides, TK_strides,
        TM_div, TK_div,
        T_mod,
        GM_mul, GK_mul,
        is_load_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_or_store_strided, load_or_store_strided_over)

    @eval abstract type TensorLayoutA{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{TensorLayoutA{T}}, workspace, tile::Tile{size}) where {T, size}
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

    return TensorLayoutA
end

function createBLayout(
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_or_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TK_strides, TN_strides,
        TK_div, TN_div,
        T_mod,
        GK_mul, GN_mul,
        is_load_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_or_store_strided, load_or_store_strided_over)

    @eval abstract type TensorLayoutB{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{TensorLayoutB{T}}, workspace, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        K = tile.base.K + tile.offset.K
        N = tile.base.N + tile.offset.N

        offset = 1

        i = 1
        @unroll for TK_stride in $TK_strides
            stride_offset = (K ÷ ($TK_div)[i]) % ($T_mod)[TK_stride]

            offset += stride_offset * ($GK_mul)[i]
            i += 1
        end

        i = 1
        @unroll for TN_stride in $TN_strides
            stride_offset = (N ÷ ($TN_div)[i]) % ($T_mod)[TN_stride]

            offset += stride_offset * ($GN_mul)[i]
            i += 1
        end

        if ($is_load_strided == false)
            return Layout.vloada(Layout.Vec{NUMEL, T}, pointer(workspace), offset)
        else
            return TensorLayout.sloada(Layout.Vec{NUMEL, T}, workspace, offset, $strided_over_size)
        end
    end

    return TensorLayoutB
end

# TODO: Add non-zero variant
function createCLayout()
    @eval abstract type TensorLayoutC{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.load(::Type{TensorLayoutC{T}}, workspace, tile::Tile{size}) where {T, size}
        N = 16 ÷ sizeof(T)

        ntuple(Val(N)) do i
            VecElement{T}(zero(T))
        end
    end

    return TensorLayoutC
end

# TODO: Make this also use the strided_over contant. It will probably be more efficient.
function createDLayout(
    T_strides_sizes::Vector{Int},
    T_strides::Tuple{Vector{Int}, Vector{Int}},
    is_load_or_store_strided::Bool,
    load_or_store_strided_over::Union{Vector{Int}, Nothing} = nothing,
)
    (
        TM_strides, TN_strides,
        TM_div, TN_div,
        T_mod,
        GM_mul, GN_mul,
        is_store_strided, strided_over_size
    ) = precomputeGETTLayoutConstants(T_strides_sizes, T_strides, is_load_or_store_strided, load_or_store_strided_over)
    
    @eval abstract type TensorLayoutD{T} <: Layout.AlignedColMajor{T} end

    @eval @inline function Layout.store!(::Type{TensorLayoutD{T}}, workspace, value, tile::Tile{size}) where {T, size}
        NUMEL = 16 ÷ sizeof(T)

        M = tile.base.M + tile.offset.M
        N = tile.base.N + tile.offset.N

        if ($is_store_strided == false)
            offset = 1

            j = 1
            @unroll for TM_stride in $TM_strides
                stride_offset = (M ÷ ($TM_div)[j]) % ($T_mod)[TM_stride]

                offset += stride_offset * ($GM_mul)[j]
                j += 1
            end

            j = 1
            @unroll for TN_stride in $TN_strides
                stride_offset = (N ÷ ($TN_div)[j]) % ($T_mod)[TN_stride]

                offset += stride_offset * ($GN_mul)[j]
                j += 1
            end

            return Layout.vstorea!(Layout.Vec{NUMEL, T}, pointer(workspace), value, offset)
        end
    
        for i = 1 : NUMEL
            offset = 1

            j = 1
            @unroll for TM_stride in $TM_strides
                stride_offset = ((M + i - 1) ÷ ($TM_div)[j]) % ($T_mod)[TM_stride]

                offset += stride_offset * ($GM_mul)[j]
                j += 1
            end

            j = 1
            @unroll for TN_stride in $TN_strides
                stride_offset = (N ÷ ($TN_div)[j]) % ($T_mod)[TN_stride]

                offset += stride_offset * ($GN_mul)[j]
                j += 1
            end

            @inbounds workspace[offset] = value[i].value
        end
    end

    return TensorLayoutD
end

end