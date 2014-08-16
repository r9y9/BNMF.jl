### Deprecated methods ###

function fitnaive!(gap::GaPNMF; epochs=20)
    updateE!(gap)

    for n=1:epochs
        # Update parameters
        updateWnaive!(gap)
        updateHnaive!(gap)
        updateTnaive!(gap)
        
        # truncate
        clearbadk!(gap)

        # show bound
        println("iter: #$(n) bound: $(bound(gap))")
        println("remained goodk: $(length(goodk(gap)))")
    end
end

function updateAux!(gap::GaPNMF, good)
    M, N = size(gap.X)
    K = gap.K

    phi = zeros(M, N, K)
    for k in good
        phi[:,:,k] = gap.Ewinvinv[:,k] * gap.Etinvinv[k] * gap.Ehinvinv[k,:]
    end

    phinorm = 1.0 ./ sum(phi, 3)
    for k in good
        phi[:,:,k] = phi[:,:,k] .* phinorm
    end

    omega = gap.Ew[:,good] * diagm(gap.Et[good]) * gap.Eh[good,:]

    return phi, omega
end

function updateWnaive!(gap::GaPNMF)
    good = goodk(gap)

    phi, omega = updateAux!(gap, good)

    gap.rhow[:,good] = gap.a + (1./omega) * gap.Eh[good,:]' * 
                       diagm(gap.Et[good])
    for k in good
        gap.tauw[:,k] = phi[:,:,k].^2 .* gap.X * gap.Ehinv[k,:]' * gap.Etinv[k]
    end
    gap.tauw[gap.tauw .< 1.0e-100] = 0

    updateEw!(gap, good)
end

function updateHnaive!(gap::GaPNMF)
    good = goodk(gap)
    
    phi, omega = updateAux!(gap, good)
    
    gap.rhoh[good,:] = gap.b + diagm(gap.Et[good]) * gap.Ew[:,good]' *
                       (1./omega)
    for k in good
        gap.tauh[k,:] = gap.Ewinv[:,k]' * gap.Etinv[k] * 
                        (phi[:,:,k].^2 .* gap.X)
    end
    gap.tauh[gap.tauh .< 1.0e-100] = 0

    updateEh!(gap, good)
end

function updateTnaive!(gap::GaPNMF)
    good = goodk(gap)

    phi, omega = updateAux!(gap, good)
    
    gap.rhot[good] = gap.alpha + 
                     sum(gap.Ew[:,good]'*(1./omega).*gap.Eh[good], 2)[:]
    for k in good
        gap.taut[k] = sum(gap.Ewinv[:,k]' * gap.Etinv[k] * 
                          (phi[:,:,k].^2 .* gap.X) .* gap.Ehinv[k,:], 2)[1,1]
    end
    gap.taut[gap.taut .< 1.0e-100] = 0
    
    updateEt!(gap, good)
end
