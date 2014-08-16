# Bayesian Non-Negative Matrix Factorization (BNMF)

module BNMF

export GaPNMF, fit!
export goodk, bound, xbar, updateE!, updateW!, updateH!, updateT!, clearbadk!

# GaP-NMF
include("gapnmf.jl")
include("gig.jl")
# include("deprecated.jl")

end # module BNMF
