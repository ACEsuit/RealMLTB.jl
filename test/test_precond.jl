using RealMLTB
using Test
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Block preconditioning and the restart protocol.
#
# The mock model from test_band_loss stands in for the chain:
#     λ[k][n] = p1 * base[k][n] + p2 + p3 * k
# so the Gauss-Newton Gram of the band map is a 3x3 matrix we can inspect
# directly.  p2 is a rigid shift, which the :global gauge is flat along, so the
# Gram built from the λ map is well conditioned while the OBJECTIVE has an
# exactly flat direction -- a useful asymmetry to keep in mind when reading
# GATE_P3.
# ---------------------------------------------------------------------------

struct _PG; base::Vector{Vector{Float64}}; end
struct _PM; end
function (m::_PM)(G, p, st)
    λ = [p[1] .* G.base[k] .+ p[2] .+ p[3] * k for k in eachindex(G.base)]
    return ((; λ = λ), st)
end

function _fixture()
    Random.seed!(21)
    nk, nb = 6, 4
    base  = [sort(randn(nb)) for _ in 1:nk]
    λ_ref = [sort(randn(nb)) for _ in 1:nk]
    return _PG(base), λ_ref, _PM(), [1.07, 0.21, -0.03]
end

@testset "band_gram" begin
    G, λ_ref, m, p0 = _fixture()
    Gm, J, w = band_gram([G], [λ_ref], m, p0, nothing; train_part = :all)

    @test size(Gm) == (3, 3)
    @test size(J) == (6 * 4, 3)
    @test length(w) == 6 * 4
    @test issymmetric(Gm)
    @test minimum(eigvals(Symmetric(Matrix(Gm)))) > -1e-10      # PSD

    # J is the Jacobian of λ: dλ/dp2 == 1 and dλ/dp1 == base, exactly
    @test all(≈(1.0), J[:, 2])
    @test J[1, 1] ≈ G.base[1][1]
    @test J[end, 3] ≈ 6.0                                        # dλ/dp3 = k

    # the weights must be the loss's weights
    _, _, w2 = band_gram([G], [λ_ref], m, p0, nothing; train_part = :all,
                         kweights_list = [fill(2.0, 6)], band_weights = fill(3.0, 4))
    @test all(≈(6.0), w2)
end

@testset "build_preconditioner whitens" begin
    G, λ_ref, m, p0 = _fixture()
    Gm, _, _ = band_gram([G], [λ_ref], m, p0, nothing; train_part = :all)
    lam = eigvals(Symmetric(Matrix(Gm)))
    cond_before = maximum(lam) / minimum(lam)

    pc = build_preconditioner(Gm; tau = 1e-6, mode = :full)
    @test pc.mode === :full
    @test norm(pc.P * pc.Pinv - I) < 1e-10
    @test issymmetric(Matrix(pc.P))

    # the whole point: P' G P has a condition number near 1
    W = Symmetric(Matrix(pc.P' * Gm * pc.P))
    lw = eigvals(W)
    @test maximum(lw) / minimum(lw) < 1.01
    @test maximum(lw) / minimum(lw) < cond_before

    pcj = build_preconditioner(Gm; tau = 1e-6, mode = :jacobi)
    @test pcj.mode === :jacobi
    @test pcj.P isa Diagonal
    @test norm(pcj.P * pcj.Pinv - I) < 1e-12

    @test_throws ErrorException build_preconditioner(Gm; mode = :nonsense)
end

@testset "preconditioner gates" begin
    G, λ_ref, m, p0 = _fixture()
    Gm, _, _ = band_gram([G], [λ_ref], m, p0, nothing; train_part = :all)
    pc = build_preconditioner(Gm; tau = 1e-6, mode = :full)

    # an objective with no exactly flat direction, so GATE_P3 is meaningful
    loss(p) = RMSE_bands(G, λ_ref, m, p, nothing; gauge = :absolute)
    θ0, f, _ = RealMLTB._make_objective_over_part(loss, p0, :all)

    rep = precond_report(pc, θ0, f)
    @test rep.P1 < 1e-10          # exact change of variables
    @test rep.P2 < 1e-10          # invertible
    @test rep.P3 < 1e-4           # the chain rule holds
    @test rep.cond_after <= rep.cond_before + 1e-12
end

@testset "restart protocol" begin
    G, λ_ref, m, p0 = _fixture()
    loss(p) = RMSE_bands(G, λ_ref, m, p, nothing; gauge = :absolute)

    # the default is one round and no restart
    ps1, h1, r1, invH1 = RealMLTB._bfgs_refine(loss, p0; iterations = 30,
                                               show_trace = false)
    @test invH1 isa Matrix && size(invH1) == (3, 3)
    @test loss(ps1) < loss(p0)

    # a short restart interval still runs, warm and cold
    for warm in (false, true)
        ps, h, r, invH = RealMLTB._bfgs_refine(loss, p0; iterations = 30,
                                               restart_steps = 5,
                                               warm_restart = warm,
                                               show_trace = false)
        @test size(invH) == (3, 3)
        @test loss(ps) < loss(p0)
    end

    # a wrong-sized invH is dropped with a note, not an error
    ps, h, r, invH = RealMLTB._bfgs_refine(loss, p0; iterations = 10,
                                           initial_invH = zeros(7, 7),
                                           show_trace = false)
    @test size(invH) == (3, 3)

    # handing the curvature on is accepted and the fit does not get worse
    ps2, _, _, _ = RealMLTB._bfgs_refine(loss, ps1; iterations = 30,
                                         initial_invH = invH1, show_trace = false)
    @test loss(ps2) <= loss(ps1) + 1e-12

    # preconditioned optimisation runs through _bfgs_refine
    Gm, _, _ = band_gram([G], [λ_ref], m, p0, nothing; train_part = :all)
    pc = build_preconditioner(Gm; tau = 1e-6, mode = :full)
    ps3, _, _, _ = RealMLTB._bfgs_refine(loss, p0; iterations = 30, precond = pc,
                                         show_trace = false)
    @test loss(ps3) < loss(p0)

    # a preconditioner of the wrong size is an error, not a silent mismatch
    bad = build_preconditioner(Matrix{Float64}(I, 5, 5))
    @test_throws ErrorException RealMLTB._bfgs_refine(loss, p0; iterations = 2,
                                                      precond = bad, show_trace = false)
end

# ---------------------------------------------------------------------------
# Curvature hand-off between phases.
#
# Within one objective this is settled: warm restarting is indistinguishable
# from never restarting, and cold restarting at the same budget was 173x worse
# on the real model.  ACROSS a change of objective the inverse Hessian
# approximates the curvature of the OLD function, so it was measured before
# being made the default -- 2-atom Si / def2-SVP, phase A on bands 1:4 then
# phase B on bands 1:8, 40 + 40 iterations, five random starts:
#
#   seed   cold       carried    ratio
#    11    0.062095   0.046235   1.343
#    23    1.2444     1.1874     1.048
#    37    0.85473    0.080762   10.58
#    51    0.042955   0.024103   1.782
#    73    0.12563    0.11723    1.072
#
# 5 of 5 to the carry.  This testset asserts the PLUMBING, not that ratio: a
# three-parameter toy is not where that effect lives.
# ---------------------------------------------------------------------------
@testset "curvature hand-off" begin
    G, λ_ref, m, p0 = _fixture()
    lossA(p) = RMSE_bands(G, λ_ref, m, p, nothing; bands = 1:2, gauge = :absolute)
    lossB(p) = RMSE_bands(G, λ_ref, m, p, nothing; bands = 1:4, gauge = :absolute)

    psA, _, _, invH = RealMLTB._bfgs_refine(lossA, p0; iterations = 8, show_trace = false)
    @test size(invH) == (3, 3)

    for ih in (nothing, invH)
        ps, _, _, out = RealMLTB._bfgs_refine(lossB, psA; iterations = 8,
                                              initial_invH = ih, show_trace = false)
        @test size(out) == (3, 3)
        @test isfinite(lossB(ps))
    end

    # train() now returns (ps, history, invH); phase 3 is skipped without ψ_ref
    ps, hist, invH2 = RealMLTB.train(G, m, p0, nothing, 3, λ_ref;
                                     iterations = 8, adam_iterations = 3,
                                     adam_patience = 3)
    @test size(invH2) == (3, 3)
    @test !isempty(hist)

    # and it takes one in
    ps, hist, invH3 = RealMLTB.train(G, m, p0, nothing, 3, λ_ref;
                                     iterations = 8, adam_iterations = 3,
                                     adam_patience = 3, initial_invH = invH2)
    @test size(invH3) == (3, 3)

    # carry_invH = false must still work and still hand a matrix back
    ps, hist, invH4 = RealMLTB.train(G, m, p0, nothing, 3, λ_ref;
                                     iterations = 8, adam_iterations = 3,
                                     adam_patience = 3, carry_invH = false)
    @test size(invH4) == (3, 3)
end
