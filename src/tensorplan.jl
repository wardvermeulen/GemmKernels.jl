module TensorPlan

using GemmKernels.TensorLayout

ModeType = AbstractVector{<:Union{Char,Integer}}

export ALGO

@enum ALGO::Int begin
    ALGO_DEFAULT_PATIENT = -6
    ALGO_GETT = -4
    ALGO_TGETT = -3
    ALGO_TTGT = -2
    ALGO_DEFAULT = -1
end

@enum UNARY_OPERATOR::Int begin
    UOP_ELEMENTWISE = 0
end

export PLAN

Base.@kwdef mutable struct PLAN
    algo::ALGO

    M::Int
    N::Int
    K::Int

    # A tensor plan variables
    a_MK_strides_sizes::Vector{Int}
    a_MK_strides::Tuple{Vector{Int}, Vector{Int}}
    is_a_load_strided::Bool
    a_strided_over::Union{Vector{Int}, Nothing} = nothing

    # B tensor plan variables
    b_KN_strides_sizes::Vector{Int}
    b_KN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_b_load_strided::Bool
    b_strided_over::Union{Vector{Int}, Nothing} = nothing

    # D tensor plan variables
    d_MN_strides_sizes::Vector{Int}
    d_MN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_d_store_strided::Bool

    TensorLayoutA = nothing
    TensorLayoutB = nothing
    TensorLayoutC = nothing
    TensorLayoutD = nothing
    gemm_conf = nothing
end



export TensorDescriptor

Base.@kwdef mutable struct TensorDescriptor
    numModes::Int
    extent::Vector{Int}
    stride::Vector{Int}
    dataType::DataType
    unaryOp::UNARY_OPERATOR

    function TensorDescriptor(a; numModes=length(size(a)), extent=size(a), stride=strides(a), dataType=eltype(a), unaryOp=UOP_ELEMENTWISE)
        return new(
            numModes,
            collect(Int, extent), collect(Int, stride),
            dataType, unaryOp
        )
    end
end

export ContractionDescriptor

mutable struct ContractionDescriptor
    descA::TensorDescriptor
    modeA::ModeType

    descB::TensorDescriptor
    modeB::ModeType

    descC::TensorDescriptor
    modeC::ModeType

    descD::TensorDescriptor
    modeD::ModeType

    computeType::DataType

    function ContractionDescriptor(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        computeType=eltype(a)
    )
        return new(
            TensorDescriptor(a), modeA,
            TensorDescriptor(b), modeB,
            TensorDescriptor(c), modeC,
            TensorDescriptor(d), modeD,
            computeType
        )
    end
end

export ContractionPlan

mutable struct ContractionPlan
    desc::ContractionDescriptor
    algo::ALGO

    GEMM_size::NamedTuple{(:M, :N, :K),Tuple{Int,Int,Int}}

    TensorLayoutA,
    TensorLayoutB,
    TensorLayoutC,
    TensorLayoutD,

    # For now, default algo is GETT
    function ContractionPlan(desc::ContractionDescriptor, algo::ALGO=ALGO_GETT)
        (
            TensorLayoutA,
            TensorLayoutB,
            TensorLayoutC,
            TensorLayoutD,
        ) = createGETTContractionPlan(desc)

    end

    function ContractionPlan(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        algo::ALGO=ALGO_GETT,
        computeType=eltype(a)
    )
        desc = ContractionDescriptor(
            a, modeA,
            b, modeB,
            c, modeC,
            d, modeD,
            computeType=computeType
        )
        return ContractionPlan(desc, algo)
    end


end

function createGETTContractionPlan(desc::ContractionDescriptor)
    modeA, modeB, modeC, modeD = desc.modeA, desc.modeB, desc.modeC, desc.modeD

    if modeC != modeD
        throw(ArgumentError("modeC and modeD must be the same"))
    end

    stridesToContract = intersect(modeB, modeA)

    AMStrides = setdiff(modeA, stridesToContract)
    BNStrides = setdiff(modeB, stridesToContract)

    AMStridesIndices = Vector{Int}(undef, 0)
    for stride in AMStrides
        append!(AMStridesIndices, findall(x -> x == stride, modeA))
    end

    BNStridesIndices = Vector{Int}(undef, 0)
    for stride in BNStrides
        append!(BNStridesIndices, findall(x -> x == stride, modeB))
    end

    AKStridesIndices = Vector{Int}(undef, 0)
    BKStridesIndices = Vector{Int}(undef, 0)
    for stride in stridesToContract
        append!(AKStridesIndices, findall(x -> x == stride, modeA))
        append!(BKStridesIndices, findall(x -> x == stride, modeB))
    end

    isALoadStrided = false
    AStridedOver = Vector{Int}(undef, 0)
    if !(1 in AMStridesIndices)
        isALoadStrided = true
        append!(AStridedOver, 1 : AMStridesIndices[1] - 1)
    end

    isBLoadStrided = false
    BStridedOver = Vector{Int}(undef, 0)
    if !(1 in BKStridesIndices)
        isBLoadStrided = true
        append!(BStridedOver, 1 : BKStridesIndices[1] - 1)
    end

    DMStridesIndices = Vector{Int}(undef, 0)
    for strideIndex in AMStridesIndices
        append!(DMStridesIndices, findall(x -> x == modeA[strideIndex], modeD))
    end

    DNStridesIndices = Vector{Int}(undef, 0)
    for strideIndex in BNStridesIndices
        append!(DNStridesIndices, findall(x -> x == modeB[strideIndex], modeD))
    end

    isDStoreStrided = (vcat(DMStridesIndices, DNStridesIndices) != 1:length(modeD))

    TensorLayoutA = TensorLayout.createALayout(desc.descA.extent, (AMStridesIndices, AKStridesIndices), isALoadStrided, AStridedOver)
    TensorLayoutB = TensorLayout.createBLayout(desc.descB.extent, (BKStridesIndices, BNStridesIndices), isBLoadStrided, BStridedOver)
    TensorLayoutC = TensorLayout.createCLayout()
    TensorLayoutD = TensorLayout.createDLayout(desc.descD.extent, (DMStridesIndices, DNStridesIndices), isDStoreStrided)

    return (
        TensorLayoutA,
        TensorLayoutB,
        TensorLayoutC,
        TensorLayoutD,
    )

end


# # TODO: Add workspaceSize
# function initContractionPlan(desc::contractionDescriptor, algo::ALGO=ALGO_GETT)

#     return 1
# end

# # TODO: Add unaryOp
# function initTensorDescriptor(numModes::Int, extent::Vector{Int}, stride::Vector{Int}, ::T) where T

#     return 1
# end


end