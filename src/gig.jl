# gigexpect returns E(x) and E(1/x) of Generalized Inverse Gaussian (GIG)
# distribution that is parametrized by gamma, rho and tau.
function gigexpect(gamma, rho, tau)
    if length(gamma) == 1
        gamma = gamma * ones(size(rho))
    end

    Ex, Exinv = zeros(size(rho)), zeros(size(rho))

    # For very small values of tau and positive values of gamma, the GIG
    # distribution becomes a gamma distribution, and its expectations are both
    # cheaper and more stable to compute that way.
    giginds = find(tau[:] .> 1e-200)
    gaminds = find(tau[:] .<= 1e-200)

    if sum(gamma[gaminds] .< 0) > 0
        error("problem with arguments.")
    end

    # GIG
    sqrtRho = sqrt(rho[giginds])
    sqrtTau = sqrt(tau[giginds])
    sqrtRatio = sqrtTau ./ sqrtRho

    # Note that we're using the *scaled* version here, since we're just
    # computing ratios and it's more stable.
    besselPlus = besselkx(gamma[giginds]+1, 2*sqrtRho .* sqrtTau)
    bessel = besselkx(gamma[giginds], 2*sqrtRho .* sqrtTau)
    besselMinus = besselkx(gamma[giginds]-1, 2*sqrtRho .* sqrtTau)
    
    Ex[giginds] = besselPlus .* sqrtRatio ./ bessel
    Exinv[giginds] = besselMinus ./ (sqrtRatio .* bessel)

    # Compute expectations for gamma distribution where we can get away with
    # it.
    Ex[gaminds] = gamma[gaminds] ./ rho[gaminds]
    Exinv[gaminds] = rho[gaminds] ./ (gamma[gaminds] - 1)
    Exinv[Exinv .< 0] = Inf

    return Ex, Exinv
end

# giggammaterm computes 
function giggammaterm(Ex, Exinv, rho, tau, a, b; cutoff=1e-200)
    score = 0.0
    zerotau = find(tau[:] .<= cutoff)
    nonzerotau = find(tau[:] .> cutoff)

    score += length(Ex) * (a * log(b) - lgamma(a))
    score -= sum((b - rho[:]) .* Ex[:])

    score -= length(nonzerotau) * log(0.5)
    score += sum(tau[nonzerotau] .* Exinv[nonzerotau])
    score -= 0.5 * a * sum(log(rho[nonzerotau]) - log(tau[nonzerotau]))

    # It's numerically safer to use scaled version of besselk    
    innerlog = besselkx(a, 2*sqrt(rho[nonzerotau] .* tau[nonzerotau]))
    score += sum(log(innerlog) - 2*sqrt(rho[nonzerotau] .* tau[nonzerotau]))

    score += sum(-a*log(rho[zerotau]) + lgamma(a))

    return score
end
