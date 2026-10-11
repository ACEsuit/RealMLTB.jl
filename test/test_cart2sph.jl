using RealMLTB
using Test
using LinearAlgebra

# ---------------------------------------------------------------------------
# The Cartesian -> real-spherical transform C is only a valid transform when
# the Cartesian shell carries ONE common normalisation factor (the libcint /
# PySCF / GaussianBasis convention), because different (l,m) must come out
# orthogonal:  C' S_cart C has to be diagonal.
#
# For l <= 2 every real-spherical component happens to be built from
# Cartesians that share the same (2lx-1)!!(2ly-1)!!(2lz-1)!! factor, so the
# transform survives even a per-Cartesian normalisation.  From l = 3 on it
# does not, which is what this test pins down.
# ---------------------------------------------------------------------------

cartlist(l) = [(lx, ly, l - lx - ly) for lx in l:-1:0 for ly in (l - lx):-1:0]

function _mom(p::Int, b::Float64)
    isodd(p) && return 0.0
    df = 1.0
    for t in 1:2:(p - 1)
        df *= t
    end
    return df / (2b)^(p ÷ 2) * sqrt(pi / b)
end

function _S_shell(l::Int; b = 2.0)
    cs = cartlist(l)
    n = length(cs)
    S = zeros(n, n)
    for i in 1:n, j in 1:n
        a, c = cs[i], cs[j]
        S[i, j] = _mom(a[1] + c[1], b) * _mom(a[2] + c[2], b) * _mom(a[3] + c[3], b)
    end
    return S
end

_reloff(M) = maximum(abs, M - Diagonal(diag(M))) / maximum(abs, diag(M))

@testset "generic and hardcoded agree for l <= 2" begin
    for l in 1:2
        G = Matrix(RealMLTB.cart2sph_generic(l))
        H = RealMLTB.cart2sph_hardcoded(l)
        s = [(G[:, m]' * H[:, m]) / (H[:, m]' * H[:, m]) for m in 1:(2l + 1)]
        @test maximum(abs, G .- H * Diagonal(s)) < 1e-12
        @test all(>(0), s)        # same sign convention, not just same span
    end
end

@testset "C' S C is diagonal in the shell-normalised convention" begin
    for l in 0:4
        C = Matrix(RealMLTB.cart2sph(l))
        @test _reloff(C' * _S_shell(l) * C) < 1e-12
    end
end

@testset "per-Cartesian normalisation breaks the transform from l = 3" begin
    for l in 0:4
        S = _S_shell(l)
        d = 1.0 ./ sqrt.(diag(S))
        C = Matrix(RealMLTB.cart2sph(l))
        off = _reloff(C' * (S .* (d * d')) * C)
        if l <= 2
            @test off < 1e-12          # accidentally safe
        else
            @test off > 1e-3           # genuinely broken -- do not pre-normalise
        end
    end
end
