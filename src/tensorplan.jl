module TensorPlan

export ALGO

@enum ALGO::Int begin
    ALGO_DEFAULT_PATIENT = -6
    ALGO_GETT = -4
    ALGO_TGETT = -3
    ALGO_TTGT = -2
    ALGO_DEFAULT = -1
end

export PLAN

Base.@kwdef mutable struct PLAN
    algo::ALGO

    M::Int
    N::Int
    K::Int

    # A tensor plan variables
    a_MK_strides::Tuple{Vector{Int}, Vector{Int}}
    is_a_load_strided::Bool
    a_strided_over::Union{Vector{Int}, Nothing} = nothing

    # B tensor plan variables
    b_KN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_b_load_strided::Bool
    b_strided_over::Union{Vector{Int}, Nothing} = nothing

    # D tensor plan variables
    d_MN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_d_store_strided::Bool

    are_types_created::Bool = false
    gemm_conf = nothing
end

end