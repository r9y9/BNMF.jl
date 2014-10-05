# gigexpect returns E(x) and E(1/x) of Generalized Inverse Gaussian (GIG)
# distribution that is parametrized by γ, ρ and τ.
function gigexpect(γ, ρ, τ)
    τ
    if length(γ) == 1
        γ = γ * ones(size(ρ))
    end

    Ex, Exinv = zeros(size(ρ)), zeros(size(ρ))

    # For very small values of τ and positive values of γ, the GIG
    # distribution becomes a γ distribution, and its expectations are both
    # cheaper and more stable to compute that way.
    giginds = find(vec(τ) .> 1e-200)
    gaminds = find(vec(τ) .<= 1e-200)

    if sum(γ[gaminds] .< 0) > 0
        error("problem with arguments.")
    end

    # GIG
    sqrt_ρ = sqrt(ρ[giginds])
    sqrt_τ = sqrt(τ[giginds])
    sqrt_ratio = sqrt_τ ./ sqrt_ρ

    # Note that we're using the *scaled* version here, since we're just
    # computing ratios and it's more stable.
    bessel_plus = besselkx(γ[giginds]+1, 2*sqrt_ρ .* sqrt_τ)
    bessel = besselkx(γ[giginds], 2*sqrt_ρ .* sqrt_τ)
    bessel_minus = besselkx(γ[giginds]-1, 2*sqrt_ρ .* sqrt_τ)
    
    Ex[giginds] = bessel_plus .* sqrt_ratio ./ bessel
    Exinv[giginds] = bessel_minus ./ (sqrt_ratio .* bessel)

    # Compute expectations for γ distribution where we can get away with
    # it.
    Ex[gaminds] = γ[gaminds] ./ ρ[gaminds]
    Exinv[gaminds] = ρ[gaminds] ./ (γ[gaminds] - 1)
    Exinv[Exinv .< 0] = Inf

    return Ex, Exinv
end

# giggammaterm computes 
function giggammaterm(Ex, Exinv, ρ, τ, a, b; cutoff::Float64=1e-200)
    score = 0.0
    zerotau = find(vec(τ) .<= cutoff)
    nonzerotau = find(vec(τ) .> cutoff)

    score += length(Ex) * (a * log(b) - lgamma(a))
    score -= sum((b - vec(ρ)) .* vec(Ex))

    score -= length(nonzerotau) * log(0.5)
    score += sum(τ[nonzerotau] .* Exinv[nonzerotau])
    score -= 0.5 * a * sum(log(ρ[nonzerotau]) - log(τ[nonzerotau]))

    # It's numerically safer to use scaled version of besselk    
    innerlog = besselkx(a, 2*sqrt(ρ[nonzerotau] .* τ[nonzerotau]))
    score += sum(log(innerlog) - 2*sqrt(ρ[nonzerotau] .* τ[nonzerotau]))

    score += sum(-a*log(ρ[zerotau]) + lgamma(a))

    return score
end
