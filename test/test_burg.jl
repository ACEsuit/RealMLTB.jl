using RealMLTB
using Test
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Burg barrier on S_out:  (w/2) * mean_k (tr G - logdet G - M),  >= 0, zero only
# at G = I, divergent as any eigenvalue -> 0.  See src/train/burg.jl.
# ---------------------------------------------------------------------------
@testset "burg_penalty" begin
    M = 6
    # G = I is the unique zero
    @test burg_penalty([Matrix{Float64}(I, M, M)]) ≈ 0.0 atol = 1e-14
    @test burg_penalty([Matrix{Float64}(I, M, M) for _ in 1:4]) ≈ 0.0 atol = 1e-14

    # anything else is strictly positive
    Random.seed!(31)
    A = randn(M, M)
    S = A' * A + 0.5I
    @test burg_penalty([S]) > 0

    # scaling the whole matrix away from I must increase it, in both directions
    @test burg_penalty([2.0 * Matrix{Float64}(I, M, M)]) > 0
    @test burg_penalty([0.5 * Matrix{Float64}(I, M, M)]) > 0

    # divergence: one eigenvalue going to zero
    vals = [1e-2, 1e-3, 1e-4, 1e-5]
    pens = [burg_penalty([Diagonal([g; ones(M - 1)]) |> Matrix]) for g in vals]
    @test issorted(pens)                       # monotone as g shrinks
    @test pens[end] > pens[1] * 1.5

    # the weight is linear and the normalisation is per k and per M
    @test burg_penalty([S]; weight = 3.0) ≈ 3.0 * burg_penalty([S])
    @test burg_penalty([S, S]) ≈ burg_penalty([S])      # mean over k, not sum

    # analytic check: a single eigenvalue g, rest 1  =>  (g - log g - 1)/(2M)
    g = 0.3
    D = Matrix(Diagonal([g; ones(M - 1)]))
    @test burg_penalty([D]) ≈ (g - log(g) - 1) / (2M)

    # already indefinite: the barrier cannot rescue it, and says so
    @test_throws Union{PosDefException,LinearAlgebra.LAPACKException} burg_penalty(
        [Matrix(Diagonal([-1.0; ones(M - 1)]))])
end

@testset "burg_w in the band loss" begin
    # a mock whose Sout is tunably far from I
    struct_g = (; base = [[0.0, 1.0], [0.5, 1.5]])
    nk, nb = 2, 2
    λ_ref = [[0.0, 1.0], [0.5, 1.5]]

    # p[1] scales Sout away from the identity; the band part is exactly zero
    m = let
        f = (G, p, st) -> ((; λ = [[0.0, 1.0], [0.5, 1.5]],
                              Sout = [Matrix{Float64}(I, 2, 2) .* p[1]
                                      for _ in 1:nk]), st)
        f
    end
    G = struct_g
    p1 = [1.0]           # Sout = I  => barrier 0
    p2 = [1.6]           # Sout = 1.6 I => barrier > 0

    @test RMSE_bands(G, λ_ref, m, p1, nothing; gauge = :absolute) ≈ 0.0 atol = 1e-12
    @test RMSE_bands(G, λ_ref, m, p1, nothing; gauge = :absolute, burg_w = 1.0) ≈ 0.0 atol = 1e-12
    # with burg_w = 0 the loss cannot see Sout at all
    @test RMSE_bands(G, λ_ref, m, p2, nothing; gauge = :absolute) ≈ 0.0 atol = 1e-12
    # with burg_w > 0 it can
    v = RMSE_bands(G, λ_ref, m, p2, nothing; gauge = :absolute, burg_w = 1.0)
    @test v > 1e-3
    # and it is monotone in the weight
    @test RMSE_bands(G, λ_ref, m, p2, nothing; gauge = :absolute, burg_w = 4.0) > v
end
