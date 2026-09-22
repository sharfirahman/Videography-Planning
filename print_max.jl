using Pkg
Pkg.activate(".")
include("src/mdma_greedy/OriginalPPAvssmooth.jl")
println("MAX PPA IS: ", max_ppa)
