@enum ALGO::Int32 begin
    DEFAULT_PATIENT = -6
    GETT = -4
    TGETT = -3
    TTGT = -2
    DEFAULT = -1
end

struct PLAN
    algo::ALGO,



end

function GETTContraction(
    plan::Plan,
    α, A::CuArray, B::Cuarray,
    β, C::CuArray,
    D::CuArray,
    workspace, workspace_size::UInt64)


end
