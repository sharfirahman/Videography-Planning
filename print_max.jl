using Pkg
Pkg.activate(".")
include("src/CompositionAwareFilming/OriginalPPAvssmooth.jl")
println("MAX PPA IS: ", max_ppa)
