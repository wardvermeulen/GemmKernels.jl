using CUDA

using GemmKernels
using GemmKernels.Operator
using GemmKernels: LocalArray

using Base: setindex

function kernel()
    c_frags = LocalArray{Tuple{4, 8}, Operator.fragtype_accum(Operator.FPUOp{4, 8, 1, Float32}, GemmKernels.Layout.AlignedColMajor{Float32})}(undef)
    @cushow c_frags[1]
    c_frags = setindex(c_frags, Float32(0.0), 1)
    return
end

    