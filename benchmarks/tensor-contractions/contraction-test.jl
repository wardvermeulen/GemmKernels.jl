using CUDA
using GemmKernels.TensorPlan
using Test

modes = eval(Meta.parse(ARGS[1]))
extents = eval(Meta.parse(ARGS[2]))

A = CuArray(rand(Float16, extents[modes[2]]) / sqrt(Float16(2048)))
B = CuArray(rand(Float16, extents[modes[3]])) / sqrt(Float16(2048))
D = CuArray(zeros(Float16, extents[modes[1]]))

plan = TensorPlan.ContractionPlan(
    A, modes[2],
    B, modes[3],
    D, modes[1],
    D, modes[1],
)

TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(0.0), D, D)

D1 = Array(D)

algo = CUDA.CUTENSOR.CUTENSOR_ALGO_GETT

plan = CUDA.CUTENSOR.plan_contraction(
    A, modes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    D, modes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    algo = algo,
    compute_type = Float16
)

CUDA.CUTENSOR.contraction!(
    1,
    A, modes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    0,
    D, modes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    compute_type = Float16,
    plan = plan
)

D2 = Array(D)

display(@test all(isapprox.(D1, D2; rtol = sqrt(eps(Float16)))))