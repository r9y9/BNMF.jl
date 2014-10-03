# Julia translation of MATLAB code for GaP-NMF as in
# "Bayesian Nonparametric Matrix Factorization for Recorded Music"
# by Matthew D. Hoffman et al. in ICML 2010.
# 
# The original matlab code: 
# http://www.cs.princeton.edu/~mdhoffma/code/gapnmfmatlab.tar

# For gamma rand
import Distributions

# Gamma Processg Non-Negative Matrix Factorization (GaP-NMF)
type GaPNMF
    X::Matrix{Float64}

    K::Int64 # number of components

    # Hyper parameters
    a::Float64
    b::Float64
    α::Float64
    
    # q(W)
    ρʷ::Matrix{Float64}
    τʷ::Matrix{Float64}

    # q(H)
    ρʰ::Matrix{Float64}
    τʰ::Matrix{Float64}

    # q(theta)
    ρᵗ::Vector{Float64}
    τᵗ::Vector{Float64}

    # Expectations
    Ew::Matrix{Float64}
    Ewinv::Matrix{Float64}
    Ewinvinv::Matrix{Float64}
    Eh::Matrix{Float64}
    Ehinv::Matrix{Float64}
    Ehinvinv::Matrix{Float64}
    Et::Vector{Float64}
    Etinv::Vector{Float64}
    Etinvinv::Vector{Float64}
end

function GaPNMF(X; K=100, a=0.1, b=0.1, α=1.0, smoothness=100)
    Y = X / mean(X)
    M, N = size(X)

    gamma = Distributions.Gamma(smoothness, 1.0/smoothness)

    GaPNMF(Y, K, a, b, α, 
           10000*rand(gamma, (M, K)),
           10000*rand(gamma, (M, K)),
           10000*rand(gamma, (K, N)),
           10000*rand(gamma, (K, N)),
           K*10000*rand(gamma, K),
           1.0/K*10000*rand(gamma, K),
           zeros(M, K),
           zeros(M, K),
           zeros(M, K),
           zeros(K, N),
           zeros(K, N),
           zeros(K, N),
           zeros(K),
           zeros(K),
           zeros(K))
end

# updateEw! updates expectations of W.
function updateEw!(gap::GaPNMF, ks=1:gap.K)
    gap.Ew[:,ks], gap.Ewinv[:,ks] = gigexpect(gap.a, 
                                              gap.ρʷ[:,ks], 
                                              gap.τʷ[:,ks])
    gap.Ewinvinv[:,ks] = gap.Ewinv[:,ks].^-1
end

# updateEh! updates expectations of H.
function updateEh!(gap::GaPNMF, ks=1:gap.K)
    gap.Eh[ks,:], gap.Ehinv[ks,:] = gigexpect(gap.b,
                                              gap.ρʰ[ks,:],
                                              gap.τʰ[ks,:])
    gap.Ehinvinv[ks,:] = gap.Ehinv[ks,:].^-1
end

# updateEt! updates expectations of theta.
function updateEt!(gap::GaPNMF, ks=1:gap.K)
    gap.Et[ks], gap.Etinv[ks] = gigexpect(gap.α/gap.K,
                                          gap.ρᵗ[ks],
                                          gap.τᵗ[ks])
    gap.Etinvinv[ks] = gap.Etinv[ks].^-1
end

function updateE!(gap::GaPNMF)
    updateEw!(gap)
    updateEh!(gap)
    updateEt!(gap)
end

function updateW!(gap::GaPNMF)
    good = goodk(gap)
    xxtwidinvsq = gap.X .* xtwid(gap, good).^-2
    xbarinv = xbar(gap, good).^-1
    dEt = diagm(gap.Et[good])
    dEtinvinv = diagm(gap.Etinvinv[good])

    gap.ρʷ[:,good] = gap.a + xbarinv * gap.Eh[good,:]' * dEt
    gap.τʷ[:,good] = gap.Ewinvinv[:,good].^2 .* 
                       (xxtwidinvsq * gap.Ehinvinv[good,:]' * dEtinvinv)
    gap.τʷ[gap.τʷ .< 1.0e-100] = 0
    updateEw!(gap, good)    
end

function updateH!(gap::GaPNMF)
    good = goodk(gap)
    xxtwidinvsq = gap.X .* xtwid(gap, good).^-2
    xbarinv = xbar(gap, good).^-1
    dEt = diagm(gap.Et[good])
    dEtinvinv = diagm(gap.Etinvinv[good])

    gap.ρʰ[good,:] = gap.b + dEt * (gap.Ew[:,good]' * xbarinv)
    gap.τʰ[good,:] = gap.Ehinvinv[good,:].^2 .* 
                       (dEtinvinv * (gap.Ewinvinv[:,good]' * xxtwidinvsq))
    gap.τʰ[gap.τʰ .< 1.0e-100] = 0
    updateEh!(gap, good)        
end

function updateT!(gap::GaPNMF)
    good = goodk(gap)
    xxtwidinvsq = gap.X .* xtwid(gap, good).^-2
    xbarinv = xbar(gap, good).^-1

    gap.ρᵗ[good] = gap.α + 
                     sum(gap.Ew[:,good]' * xbarinv .* gap.Eh[good,:], 2)
    gap.τᵗ[good] = gap.Etinvinv[good].^2 .* sum(gap.Ewinvinv[:,good]' *
                                                  xxtwidinvsq .* 
                                                  gap.Ehinvinv[good,:], 2)
    gap.τᵗ[gap.τᵗ .< 1.0e-100] = 0
    updateEt!(gap, good)
end

function goodk(gap::GaPNMF; cutoff=1.0e-10)
    cutoff *= maximum(gap.X)

    powers = (gap.Et .* maximum(gap.Ew, 1)' .* maximum(gap.Eh, 2))[:]
    perm = sortperm(powers, rev=true)
    sorted = powers[perm]
    
    indices = perm[find(sorted/maximum(sorted) .> cutoff)]
end

function clearbadk!(gap::GaPNMF)
    good = goodk(gap)
    bad = setdiff(1:gap.K, good)
    gap.ρʷ[:,bad] = gap.a
    gap.τʷ[:,bad] = 0.0
    gap.ρʰ[bad,:] = gap.b
    gap.τʰ[bad,:] = 0.0

    updateE!(gap)
end

function bound(gap::GaPNMF)
    score::Float64 = 0.0
    good = goodk(gap)
    
    xb = xbar(gap, good)
    xtw = xtwid(gap, good)

    score -= sum(gap.X ./ xtw + log(xb))
    score += giggammaterm(gap.Ew, gap.Ewinv, gap.ρʷ, gap.τʷ,
                          gap.a, gap.a)
    score += giggammaterm(gap.Eh, gap.Ehinv, gap.ρʰ, gap.τʰ, 
                          gap.b, gap.b)
    score += giggammaterm(gap.Et, gap.Etinv, gap.ρᵗ, gap.τᵗ, 
                          gap.α/gap.K, gap.α)
    return score
end

function fit!(gap::GaPNMF; 
              epochs::Int=20, criterion=None, verbose::Bool=false)
    score::Float64 = -Inf

    # update all expectations
    updateE!(gap)

    for epoch=1:epochs
        # Update variational parameters
        updateW!(gap)
        updateH!(gap)
        updateT!(gap)
        
        # truncate
        clearbadk!(gap)

        # score
        lastscore = score
        score = bound(gap)
        improvement = (score - lastscore) / abs(lastscore)

        if verbose
            println("#$(epoch) bound: $(score),
                    improvement: $(improvement),
                    activek: $(length(goodk(gap)))")
        end

        if criterion != None && improvement < criterion
            if verbose
                println("converged")
            end
            break
        end
    end

    if verbose
        println("iteration finished")
    end
end

function xbar(gap::GaPNMF, ks=1:gap.K)
    gap.Ew[:,ks] * diagm(gap.Et[ks]) * gap.Eh[ks,:]
end

function xtwid(gap::GaPNMF, ks=1:gap.K)
    gap.Ewinvinv[:,ks] * diagm(gap.Etinvinv[ks]) * gap.Ehinvinv[ks,:]
end
