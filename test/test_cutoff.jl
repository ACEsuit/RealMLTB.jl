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
using Test
using Unitful: Å

# ---------------------------------------------------------------------------
# The cutoff-convergence criterion, and the definiteness of the overlap it
# decides.
#
# cal_err(V_list, dist_list, ρcut) is the largest integral element among the
# edges a cutoff at ρcut would drop, and actual_rcut is the first ρcut whose
# dropped tail falls below `cutoff`.  That test is evaluated on a sample the
# pair-list radius has already truncated, so beyond the outermost shell
# present there is nothing left to look at; returning 0.0 there would read as
# "converged" and put actual_rcut just past that outermost shell, leaving the
# smooth cutoff to taper it whatever rcut was.  On 2-atom Si (shells at 2.351,
# 3.840, 4.502, 5.430, 5.917, 6.650 Å):
#
#   pair rcut   actual_rcut   tapered shell   max|V2| there   min eig S(k)
#   5.0            4.6          4.502            2.4e-1        -2.43e-2
#   6.0            6.0          5.917            3.0e-2        +9.46e-2
#   7.0            6.8          6.650            7.6e-3        +6.08e-2
#
# i.e. the damage scales with how large the overlap still is out there, and at
# 5 Å it is enough to make the assembled overlap indefinite.
#
# The exact Bloch-summed overlap is a Gram matrix, hence positive definite at
# every k; the windowed sum Σ_R w(|R|) S_R exp(ikR) is PSD only when w is a
# positive-definite radial function on R³ (Bochner, plus the Schur product
# theorem), and fun_cut's smoothstep is not one.  min eig S(k) at rcut = 5 Å
# under different windows:
#
#   smoothstep           +2.28e-2  -2.43e-2  +4.50e-2  +9.00e-4  +3.54e-2  no
#   hard cut at 5.0 Å    +1.22e-1  +2.09e-2  +4.39e-2  +6.72e-2  +2.72e-2  yes
#   Wendland  rc = 5.0   +8.38e-1  +8.05e-1  +7.03e-1  +7.60e-1  +7.88e-1  yes
#   Askey     rc = 5.0   +7.72e-1  +7.54e-1  +6.81e-1  +7.19e-1  +7.47e-1  yes
#
# A positive-definite window would guarantee it, but a compactly supported one
# must decay from r = 0: Wendland at rc = 5 Å damps the nearest-neighbour
# overlap to 0.227, and at rc = 8 Å to 0.541.  So the window stays, and
# cal_err returns Inf for "no evidence" instead, which falls through to
# ρcuts[end] and keeps every computed edge at full weight.
# ---------------------------------------------------------------------------
@testset "cal_err does not read absence of evidence as convergence" begin
    V = [fill(0.5, 2, 2), fill(0.25, 2, 2), fill(1e-9, 2, 2)]
    d = [1.0, 2.0, 3.0]

    # with evidence: the largest dropped element
    @test RealMLTB.cal_err(V, d, 0.5) == 0.5
    @test RealMLTB.cal_err(V, d, 1.5) == 0.25
    @test RealMLTB.cal_err(V, d, 2.5) == 1e-9

    # beyond the sample there is no evidence -- and that is not convergence
    @test RealMLTB.cal_err(V, d, 3.5) == Inf
    @test RealMLTB.cal_err(V, d, 1e6) == Inf
    @test RealMLTB.cal_err(Matrix{Float64}[], Float64[], 1.0) == Inf

    # the criterion actual_rcut is built from must therefore NOT fire past the
    # end of the sample
    ρcuts = 1.0:0.5:5.0
    errs = [RealMLTB.cal_err(V, d, ρ) for ρ in ρcuts]
    @test findfirst(x -> x < 1e-6, errs) == findfirst(≈(2.5), ρcuts)
end

@testset "the assembled overlap is positive definite at every k" begin
    rcut = 5.0u"Å"
    klist = [SVector(0.0, 0.0, 0.0), SVector(0.137, 0.221, 0.079),
             SVector(0.5, 0.0, 0.0), SVector(0.25, 0.25, 0.25),
             SVector(0.4, -0.3, 0.15)]

    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    n_out = (1, 1, 1)
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 2,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)

    # the outermost shell present (4.502 Å) is no longer tapered
    Soff = G.graph_data.cut_integrals.S_off
    dist = [norm(ed.Rvec) for ed in Soff]
    @test maximum(dist) > 4.4                      # that shell IS included
    @test all(ed.w == 1.0 for ed in Soff)          # and at full weight
    # and it is not a negligible shell, which is why tapering it mattered
    @test maximum(norm(ed.V2) for ed in Soff if norm(ed.Rvec) > 4.4) > 0.1

    # min eig S(k) > 0 at every k: measured +1.22e-1, +2.09e-2, +4.39e-2,
    # +6.72e-2, +2.72e-2, so the 1e-3 threshold is well clear of the smallest
    _, Slist = RealMLTB.assembled_HS(model, G, ps, st)
    lmin = [minimum(real.(eigvals(Hermitian(Matrix(S))))) for S in Slist]
    @test length(lmin) == length(klist)
    @test all(lmin .> 1e-3)

    # hence the eigensolver returns rather than throwing
    out = model(G, ps, st)[1]
    @test length(out.λ) == length(klist)
    @test all(length(out.λ[i]) == NORB * 2 for i in eachindex(klist))

    # and overlap_report agrees with the direct measurement
    rows = RealMLTB.overlap_report(model, [G], ps, st; verbose = false)
    @test length(rows) == 1
    @test rows[1].ok
    @test rows[1].lmin ≈ minimum(lmin)
    @test rows[1].k_worst == argmin(lmin)
end

# ---------------------------------------------------------------------------
# The pre-flight check at the training entry points.
#
# An indefinite S(k) does not degrade a fit, it prevents one: every evaluation
# throws PosDefException, _make_objective_over_part turns that into Inf, and
# the optimiser spends its whole budget on a constant objective.  Hence the
# check runs before the first step.
# ---------------------------------------------------------------------------
@testset "overlap pre-flight at the training entry points" begin
    _row(i, l) = (structure = i, lmin = l, lmax = 1.0,
                  k_worst = 2, ok = l > 0)

    good = [_row(1, 0.12), _row(2, 0.03)]
    bad  = [_row(1, 0.12), _row(2, -2.4e-2)]

    # the verdict, on synthetic rows
    @test RealMLTB._overlap_verdict(good, :error; verbose = false) === good
    @test RealMLTB._overlap_verdict(good, :warn;  verbose = false) === good
    @test_throws ErrorException RealMLTB._overlap_verdict(bad, :error; verbose = false)
    @test_logs (:warn,) RealMLTB._overlap_verdict(bad, :warn; verbose = false)

    # the message has to be actionable: it must name the symptom and the cure
    msg = try
        RealMLTB._overlap_verdict(bad, :error; verbose = false); ""
    catch e
        e.msg
    end
    @test occursin("NOT positive definite", msg)
    @test occursin("PosDefException", msg)
    @test occursin("cannot start", msg)
    @test occursin("Burg barrier cannot reach it", msg)
    @test occursin("Widen the pair-list cutoff", msg)
    @test occursin("check_overlap = :warn", msg)
    @test occursin("-0.024", msg)                 # the measured value is in it

    # modes
    rcut = 5.0u"Å"
    klist = [SVector(0.0, 0.0, 0.0), SVector(0.137, 0.221, 0.079)]
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    n_out = (1, 1, 1)
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 2,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)

    @test RealMLTB.check_overlap_pd(model, G, ps, st; mode = :off) === nothing
    @test_throws ArgumentError RealMLTB.check_overlap_pd(model, G, ps, st;
                                                         mode = :nonsense)
    # a single graph and a list of graphs are both accepted
    r1 = RealMLTB.check_overlap_pd(model, G, ps, st; verbose = false)
    r2 = RealMLTB.check_overlap_pd(model, [G, G], ps, st; verbose = false)
    @test length(r1) == 1 && r1[1].ok
    @test length(r2) == 2 && all(r.ok for r in r2)

    # and the knob is actually wired into every entry point
    kwargs_of(f) = union(Base.kwarg_decl.(collect(methods(f)))...)
    for f in (RealMLTB.train, RealMLTB.train_bands,
              RealMLTB.multilevel_train, RealMLTB.multilevel_train_H,
              RealMLTB._train_level, RealMLTB._train_level_H)
        @test :check_overlap in kwargs_of(f)
    end
end
