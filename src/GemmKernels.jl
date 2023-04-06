module GemmKernels

include("tiling.jl")

include("config.jl")
include("epilogue.jl")
include("array.jl")
include("kernel.jl")
include("layout.jl")
include("operator.jl")
include("transform.jl")

include("launch.jl")

include("blas.jl")

include("tensorplan.jl")
include("tensorlayout.jl")
include("GettContractions.jl")

end
