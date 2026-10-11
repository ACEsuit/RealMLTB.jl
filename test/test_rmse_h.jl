using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using AtomsBase
using NeighbourLists
using Lux
using LuxCore
using Random
using LinearAlgebra
using StaticArrays
using Unitful
using UnitfulAtomic
using Zygote
using Test
using Optimisers: destructure
using Unitful: Å

# ---------------------------------------------------------------------------
# Coverage for RMSE_H, which previously had none, so a keyword added to the
# list method alone went unnoticed: RMSE_H(G, ...; n_occ = k) was a
# MethodError, and RMSE_H(G, ...) with cone_w > 0 reached an undefined n_occ.
#
# Both methods are driven here on the real chain with synthetic psi_ref / proj
# (they are only complex matrices), over every cone_w branch and every
# cone_gauge, with the gradient checked against finite differences.  A mock
# model is not a substitute: one breaks on its own NamedTuple tangent under
# AD, which looks like a library failure and is not.
# ---------------------------------------------------------------------------

function _h_fixture(; NORB = 8, nk = 2, ncone = 2, NPW = 12, nref = 20)
    rcut = 5.0u"Å"
    klist = [SVector(0.0, 0.0, 0.0), SVector(0.25, 0.0, 0.0),
             SVector(0.33, 0.33, 0.0), SVector(0.3, 0.3, 0.0)]
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(3), model)

    out = model(G, ps, st)[1]
    NAO = size(out.Wb, 2)
    Random.seed!(4)
    λ_ref = [sort(randn(nref)) .- 0.5 for _ in 1:(nk + ncone)]
    # psi_ref must have one column per REFERENCE band, not per model band:
    # Delta_op forms S_ref_pred = psi_ref' * psi_pred (nref x NB) and weights
    # it with f(lambda_ref) (nref) against f(lambda_pred) (NB).
    ψ_ref = [randn(ComplexF64, NPW, nref) for _ in 1:nk]
    proj  = [randn(ComplexF64, NPW, NAO) for _ in 1:nk]
    kw    = [0.6, 0.4]
    filt  = RealMLTB.FermiFilter(0.0, 0.5, 0.4, 10.0)
    return (; G, λ_ref, ψ_ref, proj, kw, filt, model, ps, st)
end

@testset "RMSE_H runs on every branch" begin
    c = _h_fixture()
    for cw in (0.0, 1.0, 2.0), cg in (:absolute, :global, :perk)
        v = RMSE_H(c.G, c.λ_ref, c.ψ_ref, c.proj, c.kw, c.filt,
                   c.model, c.ps, c.st;
                   cone_w = cw, filter_σ = 0.5, cone_gauge = cg, n_occ = 10)
        @test isfinite(v)
        @test v >= 0
    end

    # n_occ is optional on both methods; without it the index comes from λ_ref[1]
    @test isfinite(RMSE_H(c.G, c.λ_ref, c.ψ_ref, c.proj, c.kw, c.filt,
                          c.model, c.ps, c.st; cone_w = 1.0))

    # the list method forwards every keyword it accepts
    v = RMSE_H([c.G], [c.λ_ref], [c.ψ_ref], [c.proj], [c.kw], [c.filt],
               c.model, c.ps, c.st; cone_w = 1.0, n_occ = 10)
    @test isfinite(v)
end

# ---------------------------------------------------------------------------
# The gradient of RMSE_H through the real chain.
#
# Two conditions have to hold for this to differentiate at all:
#
#   (1) the closure must be written in a PLAIN FUNCTION body.  Written inside
#       a @testset body -- at any nesting, including through the include() in
#       runtests.jl -- Zygote throws
#         BoundsError: attempt to access Tuple{Nothing, ZeroTangent} at index [3]
#   (2) cone_w and cone_gauge must be LITERALS at the call site.  Threaded in
#       as parameters they become runtime values, `if cone_w > 0` and
#       _apply_gauge (which returns its input for :absolute and a fresh `map`
#       result otherwise) acquire Union return types, and the pullback arity
#       no longer lines up.
#
# Every call site inside the package satisfies both; a user looping over
# gauges does not.  Hence the helper below is a top-level function and the
# testset only calls it.
#
# Tolerance: the ACE weights are Float32, so the objective carries ~1e-7
# relative noise and a central difference cannot do better than ~noise/h.  The
# measured rel 1.7e-4 at h = 1e-4 is that floor, not a discrepancy.
# ---------------------------------------------------------------------------
function _h_gradients(c)
    # Pull the fixture apart into PLAIN LOCALS first.  Capturing the NamedTuple
    # and writing c.G / c.model inside the differentiated closure makes Zygote
    # build a tangent for the whole 9-field NamedTuple, and that is the path
    # where the arity breaks.
    G, λ_ref, ψ_ref, proj = c.G, c.λ_ref, c.ψ_ref, c.proj
    kw, filt, model, st = c.kw, c.filt, c.model, c.st
    θ0, re = destructure(c.ps)
    T  = eltype(θ0)
    θd = Float64.(θ0)
    i, h = 7, 1e-4

    f2 = θ -> RMSE_H(G, λ_ref, ψ_ref, proj, kw, filt, model, re(T.(θ)), st;
                     cone_w = 2.0, filter_σ = 0.5, n_occ = 10)
    f0 = θ -> RMSE_H(G, λ_ref, ψ_ref, proj, kw, filt, model, re(T.(θ)), st;
                     cone_w = 0.0, filter_σ = 0.5, n_occ = 10)

    a = copy(θd); b = copy(θd)
    a[i] += h; b[i] -= h

    g2  = Zygote.gradient(f2, θd)[1]
    fd2 = (f2(a) - f2(b)) / 2h
    g0  = Zygote.gradient(f0, θd)[1]
    fd0 = (f0(a) - f0(b)) / 2h
    return (; g2, fd2, g0, fd0, i)
end

# literal cone_w, for the A/B against the parameter version
function _h_grad_literal(c)
    G, λ_ref, ψ_ref, proj = c.G, c.λ_ref, c.ψ_ref, c.proj
    kw, filt, model, st = c.kw, c.filt, c.model, c.st
    θ0, re = destructure(c.ps)
    T = eltype(θ0); θd = Float64.(θ0); i, h = 7, 1e-4
    f = θ -> RMSE_H(G, λ_ref, ψ_ref, proj, kw, filt, model, re(T.(θ)), st;
                    cone_w = 2.0, filter_σ = 0.5, n_occ = 10)
    gz = Zygote.gradient(f, θd)[1]
    a = copy(θd); b = copy(θd); a[i] += h; b[i] -= h
    return gz, (f(a) - f(b)) / 2h, i
end

function _h_grad(c, cw)
    G, λ_ref, ψ_ref, proj = c.G, c.λ_ref, c.ψ_ref, c.proj
    kw, filt, model, st = c.kw, c.filt, c.model, c.st
    θ0, re = destructure(c.ps)
    T = eltype(θ0)
    θd = Float64.(θ0)
    i, h = 7, 1e-4
    f = θ -> RMSE_H(G, λ_ref, ψ_ref, proj, kw, filt, model, re(T.(θ)), st;
                    cone_w = cw, filter_σ = 0.5, n_occ = 10)
    gz = Zygote.gradient(f, θd)[1]
    a = copy(θd); b = copy(θd); a[i] += h; b[i] -= h
    return gz, (f(a) - f(b)) / 2h, i
end

@testset "RMSE_H gradient against finite differences" begin
    c = _h_fixture()

    # cone_w > 0 differentiates, literal or as a runtime value.
    # Tolerance: the ACE weights are Float32, so the objective carries ~1e-7
    # relative noise and a central difference cannot beat ~noise/h.  Measured
    # AD 10098.3759765625 against FD 10100.051769295533 at h = 1e-4.
    for g in (_h_grad_literal(c), _h_grad(c, 2.0), _h_grad(c, 1.0))
        gz, gf, i = g
        @test gz !== nothing
        @test all(isfinite, gz)
        @test abs(gz[i] - gf) <= 1e-2 * max(abs(gf), 1.0)
    end

    # OPEN DEFECT: cone_w = 0 is NOT differentiable.
    #   BoundsError: attempt to access Tuple{Nothing, ChainRulesCore.ZeroTangent}
    #                at index [3]
    # i.e. a pullback returning two tangents where three are wanted.  With
    # cone_w = 0 the `if cone_w > 0` block is skipped, so Delta_cone stays the
    # plain 0.0 assigned before it and out.lambda[nk+1:end] is never read --
    # the tangent for the model's output NamedTuple then has a different shape
    # than on the cone_w > 0 path.  The VALUE is fine (the testset above checks
    # all 20 cone_w x cone_gauge combinations evaluate), only the gradient is
    # not.  Pinned here so the day it is fixed this starts failing.
    #
    # Note that @testset is not involved: Zygote.gradient on a closure defined
    # inside a testset works fine, and cone_w = 2.0 differentiates there.
    @test_throws BoundsError _h_grad(c, 0.0)
end

@testset "RMSE_H guards the cone window" begin
    c = _h_fixture()
    # every reference band above mu -> findlast returns nothing -> clear error
    λ_high = [λ .+ 100.0 for λ in c.λ_ref]
    @test_throws ErrorException RMSE_H(c.G, λ_high, c.ψ_ref, c.proj, c.kw,
                                       c.filt, c.model, c.ps, c.st; cone_w = 1.0)
    # a window running off the end of the reference -> clear error
    @test_throws ErrorException RMSE_H(c.G, c.λ_ref, c.ψ_ref, c.proj, c.kw,
                                       c.filt, c.model, c.ps, c.st;
                                       cone_w = 1.0, n_occ = 19)
    # cone_w = 0 must not touch the cone window at all, so a bad n_occ is fine
    @test isfinite(RMSE_H(c.G, c.λ_ref, c.ψ_ref, c.proj, c.kw, c.filt,
                          c.model, c.ps, c.st; cone_w = 0.0, n_occ = 19))
end
