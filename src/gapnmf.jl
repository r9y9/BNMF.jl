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
    alpha::Float64
    
    # q(W)
    rhow::Matrix{Float64}
    tauw::Matrix{Float64}

    # q(H)
    rhoh::Matrix{Float64}
    tauh::Matrix{Float64}

    # q(theta)
    rhot::Vector{Float64}
    taut::Vector{Float64}

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

function GaPNMF(X; K=100, a=0.1, b=0.1, alpha=1.0, smoothness=100)
    Y = X / mean(X)
    M, N = size(X)

    gamma = Distributions.Gamma(smoothness, 1.0/smoothness)

    GaPNMF(Y, K, a, b, alpha, 
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
                                              gap.rhow[:,ks], 
                                              gap.tauw[:,ks])
    gap.Ewinvinv[:,ks] = gap.Ewinv[:,ks].^-1
end

# updateEh! updates expectations of H.
function updateEh!(gap::GaPNMF, ks=1:gap.K)
    gap.Eh[ks,:], gap.Ehinv[ks,:] = gigexpect(gap.b,
                                              gap.rhoh[ks,:],
                                              gap.tauh[ks,:])
    gap.Ehinvinv[ks,:] = gap.Ehinv[ks,:].^-1
end

# updateEt! updates expectations of theta.
function updateEt!(gap::GaPNMF, ks=1:gap.K)
    gap.Et[ks], gap.Etinv[ks] = gigexpect(gap.alpha/gap.K,
                                          gap.rhot[ks],
                                          gap.taut[ks])
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

    gap.rhow[:,good] = gap.a + xbarinv * gap.Eh[good,:]' * dEt
    gap.tauw[:,good] = gap.Ewinvinv[:,good].^2 .* 
                       (xxtwidinvsq * gap.Ehinvinv[good,:]' * dEtinvinv)
    gap.tauw[gap.tauw .< 1.0e-100] = 0
    updateEw!(gap, good)    
end

function updateH!(gap::GaPNMF)
    good = goodk(gap)
    xxtwidinvsq = gap.X .* xtwid(gap, good).^-2
    xbarinv = xbar(gap, good).^-1
    dEt = diagm(gap.Et[good])
    dEtinvinv = diagm(gap.Etinvinv[good])

    gap.rhoh[good,:] = gap.b + dEt * (gap.Ew[:,good]' * xbarinv)
    gap.tauh[good,:] = gap.Ehinvinv[good,:].^2 .* 
                       (dEtinvinv * (gap.Ewinvinv[:,good]' * xxtwidinvsq))
    gap.tauh[gap.tauh .< 1.0e-100] = 0
    updateEh!(gap, good)        
end

function updateT!(gap::GaPNMF)
    good = goodk(gap)
    xxtwidinvsq = gap.X .* xtwid(gap, good).^-2
    xbarinv = xbar(gap, good).^-1

    gap.rhot[good] = gap.alpha + 
                     sum(gap.Ew[:,good]' * xbarinv .* gap.Eh[good,:], 2)
    gap.taut[good] = gap.Etinvinv[good].^2 .* sum(gap.Ewinvinv[:,good]' *
                                                  xxtwidinvsq .* 
                                                  gap.Ehinvinv[good,:], 2)
    gap.taut[gap.taut .< 1.0e-100] = 0
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
    gap.rhow[:,bad] = gap.a
    gap.tauw[:,bad] = 0
    gap.rhoh[bad,:] = gap.b
    gap.tauh[bad,:] = 0

    updateE!(gap)
end

function bound(gap::GaPNMF)
    score = 0
    good = goodk(gap)
    
    xb = xbar(gap, good)
    xtw = xtwid(gap, good)

    score -= sum(gap.X ./ xtw + log(xb))
    score += giggammaterm(gap.Ew, gap.Ewinv, gap.rhow, gap.tauw,
                          gap.a, gap.a)
    score += giggammaterm(gap.Eh, gap.Ehinv, gap.rhoh, gap.tauh, 
                          gap.b, gap.b)
    score += giggammaterm(gap.Et, gap.Etinv, gap.rhot, gap.taut, 
                          gap.alpha/gap.K, gap.alpha)
    return score
end

function fit!(gap::GaPNMF; epochs=20, criterion=None, verbose=false)
    score = -Inf

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
