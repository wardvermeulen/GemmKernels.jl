using CUDA
using GemmKernels.TensorPlan
using Test
using JSON

function main()
    fp = open("benchmarks/tensor-contractions/benchmark-suite.json", "r")

    jsonData = JSON.parse(read(fp, String))

    for el in jsonData
        parseableName = el["parseableName"]

        tensorModes = Vector{Vector{Int}}(undef, 0)
        for tensor in split(parseableName, "-")
            tensorMode = Vector{Int}(undef, 0)

            for mode in split(tensor, ".")
                push!(tensorMode, parse(Int, mode))
            end

            push!(tensorModes, tensorMode)
        end

        extents = Tuple(x for x in el["extents"])

        println(el["name"])
        
        test(extents, tensorModes)
    end

    nothing
end

function test(extents, tensorModes)
    A = CuArray(rand(Float16, extents[tensorModes[2]]) / sqrt(Float16(2048)))
    B = CuArray(rand(Float16, extents[tensorModes[3]])) / sqrt(Float16(2048))
    D = CuArray(zeros(Float16, extents[tensorModes[1]]))

    plan = TensorPlan.ContractionPlan(
        A, tensorModes[2],
        B, tensorModes[3],
        D, tensorModes[1],
        D, tensorModes[1],
    )
    @show plan.TensorLayoutA

    TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(0.0), D, D)
    D1 = Array(D)

    # CUTENSOR
    algo = CUDA.CUTENSOR.CUTENSOR_ALGO_DEFAULT

    plan = CUDA.CUTENSOR.plan_contraction(
        A, tensorModes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        B, tensorModes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        D, tensorModes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        algo = algo,
        compute_type = Float16
    )

    CUDA.CUTENSOR.contraction!(
        1,
        A, tensorModes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        B, tensorModes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        0,
        D, tensorModes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type = Float16,
        plan = plan
    )
    D2 = Array(D)

    display(@test all(isapprox.(Array(D1), Array(D2); rtol = sqrt(eps(Float16)))))
end