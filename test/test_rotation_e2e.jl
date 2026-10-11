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
using Rotations
using Zygote
using Test
using Unitful: Å

# ---------------------------------------------------------------------------
# End-to-end rotational invariance of the predicted band energies.
#
# test_rotation.jl checks the components separately -- raw ACE output, the V
# matrix, Wpsi Case-A/B -- and skips the two tests that would check the
# composition.  This file drives the whole pipeline,
#
#   nlist2graph_cut -> ACE -> assemble_Wv -> construct_integral
#                   -> assembly -> generalized eigvals
#
# rotating both the lattice and the positions.
#
# H_out = Wb H_AO Wb', and a rotation sends H_AO -> D' H_AO D with D block
# diagonal in the AO (l,m) multiplets, so the output eigenvalues are invariant
# only when the row space of Wb is D-invariant, i.e. when Wb mixes complete
# (l,m) multiplets with one coefficient per (l, shell).  The dense
# glorot_normal(NORB, Nb) that ACEv3Center initialises by default does not
# qualify; the `wb_n_out` option builds ⊕_l (C_l ⊗ I_{2l+1}) instead.  Both
# are covered here.  max|Δλ| at Gamma on 2-atom Si / def2-SVP, in meV:
#
#   Wb                                  random 1   random 2   90 deg about z
#   glorot_normal (default)              56290      41470       1.5e-9
#   multiplet selector (one s, one p)     6.2e-9     1.1e-8      9.3e-9
#   multiplet mix (s,s / p,p)             4.6e-9     7.0e-9      3.9e-9
#
# The third column is a trap: a rotation from the cubic point group is exact
# even for a dense Wb, so rotating by 90 degrees about a lattice axis tests
# nothing.
#
# Several k, not only Gamma.  A rotation maps H(k) to H(Rk), so the statement
# is λ_n(Qk; rotated) = λ_n(k; original), and at Gamma Qk = k with every Bloch
# phase equal to 1.  klist is in reduced coordinates and
# physical_k(sys, k) = 2π inv(lattice)' k, so rotating the cell gives
# G_basis' = Q G_basis and physical_k(rotated, k) = Q physical_k(orig, k)
# (checked below to 1e-12).  Passing the same reduced list to both structures
# therefore evaluates the rotated one at Qk, with no explicit rotation of k.
# The five k below have teeth: max|Im H(k)| = 0, 0.373, 4e-17, 0.360, 0.188,
# so the phases are genuinely complex at three of them and real at Gamma and
# at the time-reversal-invariant (0.5,0,0); the bands differ from Gamma by
# 3.18 to 25.6 Ha, and comparing against the wrong k fails by 18.7 Ha.
# ---------------------------------------------------------------------------
const _KRED = [SVector(0.0, 0.0, 0.0),        # Gamma
               SVector(0.137, 0.221, 0.079),  # generic, complex phases
               SVector(0.5, 0.0, 0.0),        # TRIM, real
               SVector(0.25, 0.25, 0.25),     # complex
               SVector(0.4, -0.3, 0.15)]      # complex

_rotate(sys, Q) = AtomsBase.FlexibleSystem(
    map(p -> AtomsBase.Atom(AtomsBase.species(p),
             SVector{3}(Q * ustrip.(Å, AtomsBase.position(p))) * u"Å";
             mass = AtomsBase.mass(p)), sys.particles);
    cell_vectors = [SVector{3}(Q * ustrip.(Å, c)) * u"Å"
                    for c in AtomsBase.cell_vectors(sys)],
    periodicity = sys.cell.periodicity)


# ---------------------------------------------------------------------------
# What is invariant, and why the tests below compare H(k) and S(k) separately
# rather than only the bands.
#
# For a multiplet-complete reduction Wb = ⊕_l (C_l ⊗ I_{2l+1}),
#     Wb D_AO' = Ũ Wb      with    Ũ = ⊕_atoms ⊕_l D^(l),
# since (C_l ⊗ I_{2l+1})(I_nshell ⊗ D^(l)) = C_l ⊗ D^(l)
#                                          = (1 ⊗ D^(l))(C_l ⊗ I_{2l+1}).
# Ũ is ORTHOGONAL, so a rotation acts on the assembled matrices as a unitary
# congruence,  H'(k) = Ũ H(k) Ũ'  and  S'(k) = Ũ S(k) Ũ'.  The Hermitian
# spectra of H(k) and of S(k) are therefore each invariant -- strictly more
# than invariance of the generalized bands, and, unlike the bands, defined at
# every k.
#
# Being defined at every k matters, because the assembled S(k) is not
# guaranteed positive definite: the windowed Bloch sum is PSD only for a
# positive-definite radial window (Bochner plus Schur) and fun_cut's
# smoothstep is not one.  So _specs compares the spectra of H(k) and S(k) at
# every k and the generalized bands only where S(k) ≻ 0.  That also covers Wb
# losing row rank, which makes S_out singular however good S is.  See
# cal_err in etgraph_cut_builder.jl and overlap_report.
# ---------------------------------------------------------------------------
_hspec(A) = sort(real.(eigvals(Hermitian(Matrix(A)))))

function _specs(Hl, Sl)
    map(eachindex(Hl)) do i
        H = Matrix(Hl[i]); S = Matrix(Sl[i])
        (h = _hspec(H), s = _hspec(S),
         g = isposdef(Hermitian(S)) ?
             sort(real.(eigvals(Hermitian(H), Hermitian(S)))) : Float64[])
    end
end

# Compare two spec lists; `tol` is in Hartree.  The generalized bands get a
# looser `gtol` because their error carries a factor 1/λmin(S) that the
# Hermitian spectra do not: on the (1,1,1) model on ideal Si, λmin(S) ranges
# over 9.0e-4 to 4.5e-2 across the five k, and max|Δ bands| under rotation
# degrades from 2.5e-13 to 3.7e-10 with it while max|Δ spec H| stays at
# ~2e-14 throughout.  Hence gtol = 100 * tol, still well below the observed
# 3.7e-10 at tol = 1e-10, with a violation of the symmetry itself showing up
# in the h/s comparisons at full strength.
function _cmp_specs(a, b, tol; gtol = 100 * tol)
    @test length(a) == length(b)
    for i in eachindex(a)
        @test maximum(abs, a[i].h - b[i].h) < tol
        @test maximum(abs, a[i].s - b[i].s) < tol
        @test length(a[i].g) == length(b[i].g)
        if !isempty(a[i].g)
            @test maximum(abs, a[i].g - b[i].g) < gtol
        end
    end
end


function _e2e_HS(sys, bs_name, rcut, ace, psa, sta, basisset, Wb_atom, natom)
    G = RealMLTB.nlist2graph_cut(bs_name, PairList(sys, rcut), sys,
                                 _KRED)
    val, _ = RealMLTB._ace_Wv_forward(ace, G, psa, sta)
    Wv = Float64.(RealMLTB.assemble_Wv(val))
    Wb = RealMLTB.kron_I_atoms(natom, Wb_atom)
    vH, vS = RealMLTB._3center_triplet(G, basisset, Wv, Wb)
    # _assemble_dense wants PHYSICAL k: it forms exp(i k·R_cell) with R_cell
    # Cartesian, and nlist2graph_cut has already stored
    # klist_phys = physical_k.(sys, _KRED) in the graph.  Passing the reduced
    # list here instead is exactly right at Gamma and wrong at every other k,
    # and is not rotation invariant either, since R_cell rotates with the cell
    # while a reduced k does not.
    H, S = RealMLTB._assemble_dense(G, vH, vS, G.graph_data.klist)
    return H, S
end

@testset "Gamma eigenvalues are invariant under a multiplet-complete reduction" begin
    rcut = 5.0u"Å"
    sys0 = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys0, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)

    sh, _ = RealMLTB.shells_for_atom(bs, 1)
    ls = [s.l for s in sh]
    off = cumsum([0; [2l + 1 for l in ls][1:end-1]])
    Nb = sum(2l + 1 for l in ls)

    model = RealMLTB.acev_3c_model(bs, 4, rcut; Dtot = 4, maxl = 3, ORD = 2,
                                   rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    ace = model.layers.layer_1.ace_model

    # one complete s multiplet + one complete p multiplet -> NORB = 4
    is = findfirst(==(0), ls)
    ip = findfirst(==(1), ls)
    Wb = zeros(4, Nb)
    Wb[1, off[is] + 1] = 1.0
    for m in 1:3
        Wb[1 + m, off[ip] + m] = 1.0
    end

    args = (("def2-SVP", rcut, ace, ps.layer_1.ace, st.layer_1.ace,
             st.layer_1.basisset, Wb, 2))
    sp0 = _specs(_e2e_HS(sys0, args...)...)
    @test length(sp0) == length(_KRED)
    # this selector picks orthonormal AOs, so S(k) ≈ I and is definite at
    # every k -- the generalized bands exist here
    @test all(!isempty(x.g) for x in sp0)

    Random.seed!(9)
    for Q in (Matrix(RotZ(pi/2)),
              Matrix(AngleAxis(2pi/3, 1, 1, 1)),
              Matrix(rand(RotMatrix3)),
              Matrix(rand(RotMatrix3)))
        _cmp_specs(_specs(_e2e_HS(_rotate(sys0, Q), args...)...), sp0, 1e-10)
    end
end


# ---------------------------------------------------------------------------
# The same property through the model rather than a hand-built Wb: the chain
# from acev_3c_model(..., wb_n_out = (1,1,1)) must be rotation invariant.
# This is the configuration a fit actually trains.
# ---------------------------------------------------------------------------
@testset "the model with wb_n_out is rotation invariant" begin
    rcut = 5.0u"Å"
    klist = _KRED
    sys0 = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys0, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)

    n_out = (1, 1, 1)                       # 1 s + 1 p + 1 d multiplet
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    @test NORB == 9                         # 1 + 3 + 5, the production M/atom

    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 2,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)

    # the parameters are the per-l coefficient blocks, not a dense matrix
    @test ps.layer_1.Wb isa Tuple
    @test size(ps.layer_1.Wb[1]) == (1, 4)      # 4 s shells in def2-SVP Si
    @test size(ps.layer_1.Wb[2]) == (1, 3)      # 3 p shells
    @test size(ps.layer_1.Wb[3]) == (1, 1)      # 1 d shell

    # assembled_HS: the first two layers only, so an indefinite S(k) cannot
    # abort the comparison from inside the eigensolver
    function specs(sys)
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)
        H, S = RealMLTB.assembled_HS(model, G, ps, st)
        return _specs(H, S)
    end

    sp0 = specs(sys0)
    @test length(sp0) == length(_KRED)
    @test length(sp0[1].h) == NORB * 2
    # S(k) is definite at all five k here, so the generalized bands exist
    @test all(!isempty(x.g) for x in sp0)
    @test all(minimum(x.s) > 1e-3 for x in sp0)

    Random.seed!(9)
    for Q in (Matrix(RotZ(pi/2)),
              Matrix(AngleAxis(2pi/3, 1, 1, 1)),
              Matrix(rand(RotMatrix3)),
              Matrix(rand(RotMatrix3)))
        _cmp_specs(specs(_rotate(sys0, Q)), sp0, 1e-10)
    end

    # the dense default on the same system is not invariant, which is why
    # wb_n_out exists.  A cubic-axis rotation would not show it.
    dense = RealMLTB.acev_3c_model(bs, 4, rcut; Dtot = 4, maxl = 3, ORD = 2,
                                   rng = MersenneTwister(0))
    psd, std = LuxCore.setup(MersenneTwister(1), dense)
    function specs_dense(sys)
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)
        H, S = RealMLTB.assembled_HS(dense, G, psd, std)
        return _specs(H, S)
    end
    Random.seed!(9)
    Qr = Matrix(rand(RotMatrix3))
    sd0 = specs_dense(sys0)
    sd1 = specs_dense(_rotate(sys0, Qr))
    # spec H(k) exists unconditionally; the dense Wb breaks it, at Gamma too
    @test maximum(maximum(abs, sd1[i].h - sd0[i].h) for i in eachindex(_KRED)) > 1e-3
    @test maximum(abs, sd1[1].h - sd0[1].h) > 1e-3
end

# ---------------------------------------------------------------------------
# The rotation test MUST be run on a low-symmetry structure.
#
# What the band invariance actually requires of Wb: generalized eigenvalues are
# invariant under CONGRUENCE, and the Galerkin eigenvalues of
# (Wb H Wb', Wb S Wb') depend only on row(Wb).  Under a rotation
# H -> D' H D, so the condition is exactly
#
#     row(Wb)  is a D-invariant subspace of the AO representation,
#
# and the D-invariant subspaces of ⊕_atoms ⊕_shells D^(l) are exactly
# ⊕_l (C_l ⊗ R^{2l+1}).  Multiplet completeness is therefore necessary and
# sufficient, not one option among many.
#
# A violating subspace can nevertheless look invariant on a high-symmetry
# crystal, because site symmetry can make the Bloch-summed block isotropic.
# Silicon's site group is T_d, where p is the 3-dimensional T_2 irrep, so the
# summed p-p block is a multiple of I and a truncated p subspace is
# accidentally safe; d splits as e + t_2 and a truncated d subspace is not.
# 1-dimensional subspaces per atom, max|Δλ| at Gamma in meV:
#
#   subspace          ideal Si      rattled Si
#   s alone            0.0           1.5e-5      <- l = 0 really is invariant
#   p m=1 alone        1.45e-10      647         <- accident, then exposed
#   d m=1 alone        2077          6343
#   d m=3 alone        13960         3939
#   s + complete d     3.1e-9         -          <- invariant, as required
#   s + 2 of 5 d       10890          -
#   multiplet s + p     -            1.9e-5      <- invariant
#   dense glorot       59450        136200
#
# The ideal crystal hides a genuine violation by twelve orders of magnitude,
# so this testset rattles the cell first.
# ---------------------------------------------------------------------------
@testset "rotation invariance on a low-symmetry structure" begin
    rcut = 5.0u"Å"
    klist = _KRED

    Random.seed!(2)
    sys0 = bulk(:Si)
    ats = map(sys0.particles) do p
        AtomsBase.Atom(AtomsBase.species(p),
            (ustrip.(Å, AtomsBase.position(p)) .+ 0.3 .* randn(3)) * u"Å";
            mass = AtomsBase.mass(p))
    end
    sys = AtomsBase.FlexibleSystem(ats;
        cell_vectors = AtomsBase.cell_vectors(sys0),
        periodicity = sys0.cell.periodicity)

    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    n_out = (1, 1, 1)
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)

    function specs(s)
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(s, rcut), s, klist)
        H, S = RealMLTB.assembled_HS(model, G, ps, st)
        return _specs(H, S)
    end

    sp0 = specs(sys)
    Random.seed!(9)
    for Q in (Matrix(rand(RotMatrix3)), Matrix(rand(RotMatrix3)))
        # 1e-5 meV = 4e-10 Ha; the floor is higher on a rattled cell than on
        # the ideal one because the neighbour list is less degenerate
        _cmp_specs(specs(_rotate(sys, Q)), sp0, 1e-8)
    end
end

# ---------------------------------------------------------------------------
# The k-dependence of the statement, which every test above had taken on trust.
#
#   H_{QR} ' = D' H_R D  and the lattice vectors rotate, so
#     H'(k') = Σ_R D' H_R D exp(i k'·QR) = D' H(Q'k') D
#   i.e.  H'(Qk) = D' H(k) D   and   λ_n(Qk ; rotated) = λ_n(k ; original).
#
# This testset establishes the three things that make the multi-k tests above
# meaningful rather than five copies of the Gamma test:
#   (a) the same reduced k on the rotated cell really is Qk;
#   (b) the Bloch phases at those k are genuinely complex, so the phase
#       machinery is exercised;
#   (c) the comparison has teeth -- against the wrong k it fails by Ha.
# ---------------------------------------------------------------------------
@testset "the k-dependence: spec at Qk on the rotated cell = spec at k" begin
    rcut = 5.0u"Å"

    Random.seed!(2)
    s0 = bulk(:Si)
    ats = map(s0.particles) do p
        AtomsBase.Atom(AtomsBase.species(p),
            (ustrip.(Å, AtomsBase.position(p)) .+ 0.3 .* randn(3)) * u"Å";
            mass = AtomsBase.mass(p))
    end
    sys = AtomsBase.FlexibleSystem(ats; cell_vectors = AtomsBase.cell_vectors(s0),
                                   periodicity = s0.cell.periodicity)
    Random.seed!(9)
    Q = Matrix(rand(RotMatrix3))
    sr = _rotate(sys, Q)

    # (a) THE CONVENTION.  The same reduced k denotes Qk on the rotated cell,
    #     so handing the same reduced list to both structures is exactly the
    #     comparison λ(Qk; rotated) = λ(k; original).
    for kr in _KRED
        @test norm(RealMLTB.physical_k(sr, kr) -
                   Q * RealMLTB.physical_k(sys, kr)) < 1e-12
    end
    # and the five k are genuinely different points
    @test length(unique(round.(norm.(RealMLTB.physical_k.(Ref(sys), _KRED)),
                               digits = 6))) >= 4

    Y = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, Y[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    n_out = (1, 1, 1)
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)

    HS(sy) = RealMLTB.assembled_HS(model,
                RealMLTB.nlist2graph_cut("def2-SVP", PairList(sy, rcut), sy,
                                         _KRED), ps, st)
    H0, S0 = HS(sys)
    H1, S1 = HS(sr)
    sp0 = _specs(H0, S0)
    sp1 = _specs(H1, S1)

    # (b) the phases are exercised: complex where they should be, real where
    #     they must be
    imH = [maximum(abs, imag.(Matrix(H0[i]))) for i in eachindex(_KRED)]
    @test imH[1] < 1e-14               # Gamma
    @test imH[3] < 1e-14               # (1/2,0,0): time-reversal invariant
    @test imH[2] > 1e-2                # generic k: genuinely complex
    @test imH[4] > 1e-2
    @test imH[5] > 1e-2

    # (c) everything depends on k.  The rattled cell has S(k) ≻ 0 at all five
    #     k (λmin = 0.361 to 0.437), so the bands exist throughout.  Spread
    #     from Gamma over k2..k5: bands 3.18 to 25.60 Ha, spec H 0.19 to 0.86.
    @test all(!isempty(x.g) for x in sp0)
    for i in 2:length(_KRED)
        @test maximum(abs, sp0[i].g - sp0[1].g) > 1.0
        @test maximum(abs, sp0[i].h - sp0[1].h) > 0.1
    end

    # the statement itself, at every k.  Invariance floors over the five k are
    # 1.8e-9 (spec H), 2.7e-15 (spec S) and 9.0e-10 (bands), against the
    # 0.1-25 Ha k-dependence above.
    _cmp_specs(sp1, sp0, 1e-8)

    # (d) TEETH.  Against the WRONG k it fails by Hartrees -- so the loop over
    #     k above really is comparing k to k and not collapsing to one point.
    #     bands: k2 vs k3 = 18.68 Ha; k2 vs Gamma = 3.18; and
    #     ‖λ4-λ5‖ ≥ ‖λ5-λ1‖ - ‖λ4-λ1‖ = 25.60 - 6.51 = 19.1 Ha.
    @test maximum(abs, sp1[2].g - sp0[3].g) > 1.0
    @test maximum(abs, sp1[2].g - sp0[1].g) > 1.0
    @test maximum(abs, sp1[4].g - sp0[5].g) > 1.0
    @test maximum(abs, sp1[2].h - sp0[3].h) > 0.1
    @test maximum(abs, sp1[4].h - sp0[5].h) > 0.1

    # (e) the reduced-vs-physical k regression: feeding _assemble_dense the
    #     reduced list instead of the physical one from the graph is right at
    #     Gamma and wrong elsewhere, and is not rotation invariant.  Built
    #     deliberately here so the mistake cannot return unnoticed.
    function bad(sy)
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sy, rcut), sy, _KRED)
        val, _ = RealMLTB._ace_Wv_forward(model.layers.layer_1.ace_model, G,
                                          ps.layer_1.ace, st.layer_1.ace)
        Wv = Float64.(RealMLTB.assemble_Wv(val))
        Wb = RealMLTB.kron_I_atoms(2,
                 RealMLTB.assemble_Wb_multiplet(ps.layer_1.Wb,
                                                st.layer_1.basisset, n_out))
        vH, vS = RealMLTB._3center_triplet(G, st.layer_1.basisset, Wv, Wb)
        RealMLTB._assemble_dense(G, vH, vS, _KRED)          # REDUCED: wrong
    end
    bH0, bS0 = bad(sys); bH1, bS1 = bad(sr)
    b0 = _specs(bH0, bS0); b1 = _specs(bH1, bS1)
    # 1e-8, not 1e-10: the rattled cell's invariance floor is 1.8e-9
    @test maximum(abs, b0[1].h - sp0[1].h) < 1e-8           # agrees at Gamma
    @test maximum(abs, b1[1].h - b0[1].h) < 1e-8            # invariant at Gamma
    @test maximum(maximum(abs, b1[i].h - b0[i].h)
                  for i in 2:length(_KRED)) > 1e-6          # and not elsewhere
end

# ---------------------------------------------------------------------------
# assemble_Wb_multiplet: the structure it is defined to have, and a usable
# gradient.
#
# Multiplet completeness is necessary and sufficient for rotation covariance
# (see the header), so this is the one Wb that has to be trainable, and
# forward tests cannot tell whether it is.  The things that break the gradient
# are tuple destructuring inside the differentiated region and mutation, so
# both are checked here against finite differences rather than assumed.
# ---------------------------------------------------------------------------
@testset "assemble_Wb_multiplet: structure, and a usable gradient" begin
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    n_out = (1, 1, 1)
    _, offsets, Nb, sbl = RealMLTB.wb_multiplet_layout(bs)
    lmax = length(sbl) - 1

    Random.seed!(7)
    C = Tuple(randn(n_out[l+1], length(sbl[l+1])) for l in 0:lmax)
    W = RealMLTB.assemble_Wb_multiplet(C, bs, n_out)

    # against the structure it is defined to have, built independently:
    # rows in (l, a, m) order, row[offsets[s] + m] = C[l+1][a, si]
    ref = zeros(RealMLTB.wb_multiplet_norb(bs, n_out), Nb)
    r = 0
    for l in 0:lmax, a in 1:n_out[l+1], m in 1:(2l + 1)
        r += 1
        for (si, s) in enumerate(sbl[l+1])
            ref[r, offsets[s] + m] = C[l+1][a, si]
        end
    end
    @test size(W) == size(ref)
    @test W == ref                              # bitwise, not just approx

    # the block structure: a row built from l never touches another l's AOs
    for l in 0:lmax
        cols = vcat([(offsets[s] + 1):(offsets[s] + 2l + 1) for s in sbl[l+1]]...)
        rws = findall(rr -> any(!=(0), W[rr, cols]), 1:size(W, 1))
        @test !isempty(rws)
        @test all(W[rr, c] == 0 for rr in rws for c in setdiff(1:Nb, cols))
    end

    # and the gradient, against finite differences
    f(Cv) = sum(abs2, RealMLTB.assemble_Wb_multiplet(Cv, bs, n_out) *
                      collect(1.0:Nb))
    gz = Zygote.gradient(f, C)[1]
    @test gz !== nothing
    h = 1e-7
    for (lb, i, j) in ((1, 1, 1), (2, 1, 2), (3, 1, 1))
        Cp = Tuple(k == lb ? (m = copy(C[k]); m[i, j] += h; m) : C[k] for k in 1:3)
        Cm = Tuple(k == lb ? (m = copy(C[k]); m[i, j] -= h; m) : C[k] for k in 1:3)
        fd = (f(Cp) - f(Cm)) / (2h)
        @test abs(fd) > 1e-6
        @test isapprox(gz[lb][i, j], fd; rtol = 1e-6)
    end
end
