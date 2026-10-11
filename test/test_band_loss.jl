using RealMLTB
using Test
using Zygote
using Random
using LinearAlgebra
using GaussianBasis
using Molecules
using AtomsBuilder
using AtomsBase
using NeighbourLists
using Lux
using LuxCore
using StaticArrays
using Unitful
using UnitfulAtomic
using Unitful: Å

# ---------------------------------------------------------------------------
# Band-only loss: RMSE_bands / band_report / the one-global-shift gauge.
#
# A mock model stands in for the real chain so the gauge algebra and the AD
# path are tested on their own:
#     λ[k][n] = p1 * base[k][n] + p2 + p3 * k
# p2 is exactly a rigid shift of the whole band structure, so :global must be
# flat along it.
# ---------------------------------------------------------------------------

struct _BG; base::Vector{Vector{Float64}}; end
struct _BM; end
function (m::_BM)(G, p, st)
    λ = [p[1] .* G.base[k] .+ p[2] .+ p[3] * k for k in eachindex(G.base)]
    return ((; λ = λ), st)
end

_fd(f, p; h = 1e-6) = [
    (f([i == j ? p[j] + h : p[j] for j in eachindex(p)]) -
     f([i == j ? p[j] - h : p[j] for j in eachindex(p)])) / 2h
    for i in eachindex(p)]

@testset "RMSE_bands differentiates" begin
    Random.seed!(3)
    nk, nb = 7, 5
    base  = [sort(randn(nb)) for _ in 1:nk]
    λ_ref = [sort(randn(nb)) for _ in 1:nk]
    G, m  = _BG(base), _BM()
    p0 = [1.07, 0.21, -0.03]
    w  = rand(nk) .+ 0.3
    bw = rand(nb) .+ 0.5

    for gauge in (:absolute, :global, :perk), kw in (nothing, w), bb in (nothing, bw)
        f(p) = RMSE_bands(G, λ_ref, m, p, nothing;
                          kweights = kw, band_weights = bb, gauge = gauge)
        gz = Zygote.gradient(f, p0)[1]
        gf = _fd(f, p0)
        @test maximum(abs, gz .- gf) / maximum(abs, gf) < 1e-5
    end

    Glist, Rlist = [G, _BG([b .+ 1.3 for b in base])], [λ_ref, λ_ref]
    for gauge in (:absolute, :global, :perk), sc in (:structure, :dataset)
        f(p) = RMSE_bands(Glist, Rlist, m, p, nothing; gauge = gauge, shift_scope = sc)
        gz = Zygote.gradient(f, p0)[1]
        gf = _fd(f, p0)
        @test maximum(abs, gz .- gf) / maximum(abs, gf) < 1e-5
    end
end

@testset "the global shift is exactly profiled out" begin
    Random.seed!(4)
    nk, nb = 6, 4
    base  = [sort(randn(nb)) for _ in 1:nk]
    λ_ref = [sort(randn(nb)) for _ in 1:nk]
    G, m  = _BG(base), _BM()
    p0 = [1.0, 0.3, 0.0]
    p1 = [1.0, 0.3 + 4.2, 0.0]          # same bands, shifted rigidly

    fg(p) = RMSE_bands(G, λ_ref, m, p, nothing; gauge = :global)
    fa(p) = RMSE_bands(G, λ_ref, m, p, nothing; gauge = :absolute)

    @test fg(p0) ≈ fg(p1) atol = 1e-12          # invariant
    @test fa(p0) < fa(p1)                        # :absolute is not
    @test abs(Zygote.gradient(fg, p0)[1][2]) < 1e-12   # flat direction

    # ordering: :perk <= :global <= :absolute
    fp(p) = RMSE_bands(G, λ_ref, m, p, nothing; gauge = :perk)
    @test fp(p0) <= fg(p0) + 1e-12
    @test fg(p0) <= fa(p0) + 1e-12
end

@testset "shift scope" begin
    Random.seed!(5)
    nk, nb = 5, 4
    b1 = [sort(randn(nb)) for _ in 1:nk]
    b2 = [sort(randn(nb)) for _ in 1:nk]
    λ1 = [sort(randn(nb)) for _ in 1:nk]
    λ2 = [b + fill(0.9, nb) for b in b2]    # structure 2 needs a +0.9 shift
    m = _BM()
    Glist, Rlist = [_BG(b1), _BG(b2)], [λ1, λ2]
    p0 = [1.0, 0.0, 0.0]

    per  = RMSE_bands(Glist, Rlist, m, p0, nothing; gauge = :global, shift_scope = :structure)
    data = RMSE_bands(Glist, Rlist, m, p0, nothing; gauge = :global, shift_scope = :dataset)
    # one shift per structure has strictly more freedom than one for all
    @test per <= data + 1e-12
    @test per < data        # structure 2 wants its own shift here
end

@testset "band windows and band weights" begin
    Random.seed!(6)
    nk, nbmodel, nbref = 4, 3, 8
    base  = [sort(randn(nbmodel)) for _ in 1:nk]
    λ_ref = [sort(randn(nbref)) for _ in 1:nk]
    G, m  = _BG(base), _BM()
    p0 = [1.0, 0.0, 0.0]

    # a 3-band model fitted to reference bands 4:6
    v = RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:3, ref_bands = 4:6)
    @test isfinite(v) && v > 0
    # ref_bands defaults to bands, and 1:3 of the reference is a different target
    @test RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:3) != v

    # a band weight of 0 must equal dropping that band
    w3 = [1.0, 1.0, 0.0]
    a = RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:3, ref_bands = 4:6, band_weights = w3)
    b = RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:2, ref_bands = 4:5)
    @test a ≈ b

    # uniform band weights are a no-op
    @test RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:3, ref_bands = 4:6,
                     band_weights = [2.0, 2.0, 2.0]) ≈ v
end

@testset "RMSE_bands agrees with RMSE where they overlap" begin
    Random.seed!(7)
    nk, nb = 6, 5
    base  = [sort(randn(nb)) for _ in 1:nk]
    λ_ref = [sort(randn(nb)) for _ in 1:nk]
    G, m  = _BG(base), _BM()
    p0 = [1.1, 0.2, -0.05]
    for gauge in (:absolute, :global, :perk)
        @test RMSE_bands(G, λ_ref, m, p0, nothing; bands = 1:4, gauge = gauge) ≈
              RMSE(G, λ_ref, m, p0, nothing; keep_bands = 4, gauge = gauge)
    end
end

@testset "band_report identities" begin
    Random.seed!(8)
    nk, nb = 9, 6
    λ_ref  = [sort(randn(nb)) for _ in 1:nk]
    offs   = 0.13 .* randn(nk)
    λ_pred = [λ_ref[k] .+ 0.41 .+ offs[k] .+ 0.02 .* randn(nb) for k in 1:nk]

    for kw in (nothing, rand(nk) .+ 0.2), bw in (nothing, rand(nb) .+ 0.4)
        r = band_report(λ_pred, λ_ref; kweights = kw, band_weights = bw)
        @test isapprox(r.L_abs^2, r.L_global^2 + r.c_global^2; rtol = 1e-9)
        @test isapprox(r.L_global^2, r.L_perk^2 + r.L_disp; rtol = 1e-9)
        @test r.L_perk <= r.L_global + 1e-12
        @test r.L_global <= r.L_abs + 1e-12
        @test length(r.per_band) == nb
        @test length(r.per_k) == nk
        @test r.max_abs_error >= r.L_global
    end

    # a pure rigid shift is invisible to the global gauge
    r = band_report([λ + fill(0.5, nb) for λ in λ_ref], λ_ref)
    @test r.L_abs ≈ 0.5
    @test r.L_global < 1e-12
    @test r.c_global ≈ -0.5
    @test r.L_disp < 1e-24
end

# ---------------------------------------------------------------------------
# The real model, rather than the stub used above.
#
# The gradient tests above all run against a stub model with st = nothing,
# which cannot see the two ways this path breaks under AD:
#
#  1. a tuple returned from the differentiated map in the LIST method of
#     RMSE_bands.  Zygote builds a Tangent{NTuple{4,Float64}} for it and
#     cannot add it back, so that method's gradient fails while the
#     single-structure one succeeds -- and train_bands takes a Glist.
#  2. tuple destructuring inside assemble_Wb_multiplet, whose reverse pass
#     then receives a `nothing`, making every model built with wb_n_out
#     non-differentiable.
#
# Neither shows up in a forward-only test.  This testset drives the real chain
# for both Wb parameterisations and checks against finite differences.
# ---------------------------------------------------------------------------
@testset "the real model differentiates, both Wb parameterisations" begin
    rcut = 5.0u"Å"
    klist = [SVector(0.0, 0.0, 0.0), SVector(0.137, 0.221, 0.079)]
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)

    # bump one entry of Wb, which is where both bugs lived
    function bump(ps, δ, idx...)
        l1 = ps.layer_1
        Wb2 = if l1.Wb isa Tuple
            b = l1.Wb[idx[1]]
            c = copy(b); c[idx[2], idx[3]] += δ
            Tuple(k == idx[1] ? c : l1.Wb[k] for k in eachindex(l1.Wb))
        else
            c = copy(l1.Wb); c[idx[1], idx[2]] += δ; c
        end
        merge(ps, (layer_1 = merge(l1, (Wb = Wb2,)),))
    end

    for (name, kw, bidx) in (("dense Wb", (;), (1, 1)),
                             ("wb_n_out = (1,1,1)", (; wb_n_out = (1, 1, 1)),
                              (2, 1, 1)))
        NORB = haskey(kw, :wb_n_out) ?
               RealMLTB.wb_multiplet_norb(bs, kw.wb_n_out) : 9
        model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3,
                                       ORD = 2, rng = MersenneTwister(0), kw...)
        ps, st = LuxCore.setup(MersenneTwister(1), model)

        # a reference that is NOT a constant shift of the prediction: with the
        # global gauge that would give residual exactly 0, where d(sqrt)/dx is
        # infinite and the gradient is meaningless
        Random.seed!(3)
        λ_ref = [sort(real.(model(G, ps, st)[1].λ[i]))[1:4] .+ 0.05 .* randn(4)
                 for i in eachindex(klist)]

        one_g(p)  = RMSE_bands(G, λ_ref, model, p, st; bands = 1:4)
        list_g(p) = RMSE_bands([G], [λ_ref], model, p, st; bands = 1:4)

        @test one_g(ps) > 1e-3                       # not at the kink
        @test list_g(ps) ≈ one_g(ps)

        # both methods differentiate at all, which is bug 1 and bug 2
        g1 = Zygote.gradient(one_g, ps)[1]
        g2 = Zygote.gradient(list_g, ps)[1]
        @test g1 !== nothing
        @test g2 !== nothing

        # one structure: the list method must agree with the single one
        n1 = g1.layer_1.Wb isa Tuple ? norm(norm.(g1.layer_1.Wb)) :
                                       norm(g1.layer_1.Wb)
        n2 = g2.layer_1.Wb isa Tuple ? norm(norm.(g2.layer_1.Wb)) :
                                       norm(g2.layer_1.Wb)
        @test n1 > 1e-6                              # and is not trivially 0
        @test isapprox(n1, n2; rtol = 1e-10)

        # and against finite differences, through the whole chain
        h = 1e-6
        fd = (list_g(bump(ps, h, bidx...)) - list_g(bump(ps, -h, bidx...))) / (2h)
        ad = g2.layer_1.Wb isa Tuple ?
             g2.layer_1.Wb[bidx[1]][bidx[2], bidx[3]] :
             g2.layer_1.Wb[bidx[1], bidx[2]]
        @test abs(fd) > 1e-6                         # the probe has signal
        @test isapprox(ad, fd; rtol = 1e-4)
    end
end
