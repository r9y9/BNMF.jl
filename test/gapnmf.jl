using BNMF
using Base.Test

function monotonic_increase(X)
    gap = GaPNMF(X)
    
    # update all expectations
    updateE!(gap)
    
    score = -Inf
    epochs = 100
    for epoch=1:epochs
        # W
        lastscore = score
        updateW!(gap)
        score = bound(gap)
        @test score >= lastscore

        # H
        lastscore = score
        updateH!(gap)
        score = bound(gap)
        @test score >= lastscore

        # Theta
        lastscore = score
        updateT!(gap)
        score = bound(gap)
        @test score >= lastscore
        
        # truncate
        clearbadk!(gap)
    end
end

# brief usage
function fit(X)
    gap = GaPNMF(X)
    fit!(gap, verbose=false)
    
    @test !any(isnan(gap.rhow))
    @test !any(isnan(gap.tauw))
    @test !any(isnan(gap.rhoh))
    @test !any(isnan(gap.tauh))
    @test !any(isnan(gap.rhot))
    @test !any(isnan(gap.taut))
    @test !any(isnan(gap.Ew))
    @test !any(isnan(gap.Ewinv))
    @test !any(isnan(gap.Ewinvinv))
    @test !any(isnan(gap.Eh))
    @test !any(isnan(gap.Ehinv))
    @test !any(isnan(gap.Ehinvinv))
    @test !any(isnan(gap.Et))
    @test !any(isnan(gap.Etinv))
    @test !any(isnan(gap.Etinvinv))
end

srand(98765)
A = rand(10, 10)
B = rand(100, 100)

monotonic_increase(A)
monotonic_increase(B)

fit(A)
fit(B)
