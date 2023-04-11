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

        # return (tensorModes, extents)
        tensorModes = repr(tensorModes)
        extents = repr(extents)

        println(el["name"])
        
        cmd = `/home/wjvermeu/julia-versions/julia-1.8.3/bin/julia --project /home/wjvermeu/GemmKernels.jl/benchmarks/tensor-contractions/contraction-test.jl $tensorModes $extents`

        run(cmd)
    end

    nothing
end

isinteractive() || main()