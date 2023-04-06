export TensorLayout
module TensorLayout

using CUDA
using GemmKernels.Layout
using GemmKernels.TensorPlan

# TODO: Write test. One example given below.
# precomputeGETTLayoutConstants([16, 512, 32], ([1, 3], [2]))
# result: ((1, 3), (2,), (1, 16), (1,), (16, 512, 32), (1, 8192), (16,))

# TODO: Write docstring.

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
    if (is_load_or_store_strided == true)
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

@inline function sloada(::Type{Layout.Vec{NUMEL, T}}, workspace, offset::Int, strided_over_size::Int) where {NUMEL, T}
    return ntuple(Val(NUMEL)) do i
        @inbounds VecElement{T}(workspace[offset + (i - 1) * strided_over_size])
    end
end

function testCreateALayout(plan::PLAN)
    plan
end

end