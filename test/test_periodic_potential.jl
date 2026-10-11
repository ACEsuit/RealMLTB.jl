using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using AtomsBase
using NeighbourLists
using Test
using LinearAlgebra
using StaticArrays
using Unitful
using UnitfulAtomic
using Random
using Unitful: Å

# ---------------------------------------------------------------------------
# The model is  H = T + V  with a PERIODIC potential
#
#   V(r) = Σ_p Σ_{R'} Σ_b Wv[p,b] φ_{p,b}(r - R_p - R')
#
# so every Hamiltonian block needs the full sum
#
#   A_ij(R)[a,c] = T_ij(R)[a,c]
#                + Σ_{p,R'} Σ_b Wv[p,b] ∫ φ_i[a](r) φ_{p,b}(r-R_p-R') φ_j[c](r-R_j-R)
#
# Rebuild A_ij(R) from the blocks nlist2graph_cut + _3center_triplet_cut
# produce and compare against that sum done by brute force over every atom and
# every image.  Three terms used to be absent:
#
#   * p = i, R' = 0 in every off-diagonal and self-image block
#     (the potential sitting on the bra atom -- the largest 3c term there is)
#   * every p != i in the R = 0 on-site block (the crystal field)
#   * everything beyond sqrt(d_ij^2+d_ik^2+d_jk^2) <= pair cutoff, a screen
#     applied to a three-distance norm but compared against a pair cutoff
#
# Before the fix this test reported 42-45% of the three-centre integral mass
# missing from the off-diagonal and self-image blocks, and 0.95 eV missing
# from the on-site block.
# ---------------------------------------------------------------------------

const _EV  = 27.211386245988
const _A2B = 1 / 0.529177210903

function _setup(; basis = "sto-3g", rcut_A = 6.0, nim = 3)
    sys   = bulk(:Si)
    nlist = PairList(sys, rcut_A * u"Å")
    G     = RealMLTB.nlist2graph_cut(basis, nlist, sys, [SVector(0.0, 0.0, 0.0)])
    nat   = length(G.first) - 1

    Zs  = [Int64(AtomsBase.atomic_number(species(p))) for p in sys.particles]
    bs0 = Dict(z => BasisSet(basis, [Molecules.Atom(z, 28.0, [0.0, 0.0, 0.0])];
                             spherical = false, lib = :acsint) for z in unique(Zs))
    Cc  = Dict(z => RealMLTB.cart2sph_coeff(bs0[z]) for z in keys(bs0))
    Sc  = Dict(z => overlap(bs0[z]) for z in keys(bs0))

    X    = [ustrip.(Å, r) for r in position(sys, :)]
    cell = [ustrip.(Å, c) for c in AtomsBase.cell_vectors(sys)]
    images = vec([n1 * cell[1] + n2 * cell[2] + n3 * cell[3]
                  for n1 in -nim:nim, n2 in -nim:nim, n3 in -nim:nim])

    return (; sys, nlist, G, nat, Zs, bs0, Cc, Sc, X, images)
end

function _blocks_and_reference(c; potcut = 8.0)
    G, nat, Zs, bs0, Cc, Sc, X, images =
        c.G, c.nat, c.Zs, c.bs0, c.Cc, c.Sc, c.X, c.images
    cut = G.graph_data.cut_integrals
    NAO = size(Cc[Zs[1]], 2)

    Random.seed!(11)
    Wv = randn(nat * NAO)
    Wb = Matrix{Float64}(I, nat * NAO, nat * NAO)

    vH, _ = RealMLTB._3center_triplet_cut(G, bs0[Zs[1]], Wv, Wb)

    key(i, j, R) = (i, j, round.(R; digits = 6) .+ 0.0)   # +0.0 kills -0.0
    Acode = Dict{Any, Matrix{Float64}}()
    add!(k, M) = (Acode[k] = haskey(Acode, k) ? Acode[k] + M : copy(M))

    for i in 1:nat
        add!(key(i, i, SVector(0.0, 0.0, 0.0)), Matrix(vH[i]))
    end
    off = nat
    for (e, ed) in enumerate(cut.T_off)
        i, j = ed.integral_idx
        add!(key(i, j, ed.R_cell), Matrix(vH[off + e]))
    end
    off += length(cut.T_off) + length(cut.S_off)   # S_off adds nothing to H
    for (e, ed) in enumerate(cut.V_off)
        i, _, j = ed.integral_idx
        add!(key(i, j, ed.R_cell), Matrix(vH[off + e]))
    end

    one2c(M, a, b) = RealMLTB.one2c_spherical_AO_integrals(M, Cc[a], Cc[b], Sc[a], Sc[b])
    v3c(M, a, b, d) = RealMLTB.v3c_spherical_AO_integrals(M, Cc[a], Cc[b], Cc[d],
                                                          Sc[a], Sc[b], Sc[d])

    function Aref(i, j, Rb)
        dvec = (X[j] + Vector(SVector{3}(Rb)) ./ _A2B) - X[i]
        bsi = bs0[Zs[i]]
        bsj = RealMLTB.shift_bs(bs0[Zs[j]], SVector{3}(dvec))
        A = one2c(kinetic(bsi, bsj), Zs[i], Zs[j])
        for pc in 1:nat, Rp in images
            dp = (X[pc] + Rp) - X[i]
            (norm(dp) > potcut || norm(dp - dvec) > potcut) && continue
            bsp = RealMLTB.shift_bs(bs0[Zs[pc]], SVector{3}(dp))
            V3 = v3c(RealMLTB.EPI_1e3c(bsi, bsp, bsj), Zs[i], Zs[pc], Zs[j])
            A .+= RealMLTB._contract_v3_dim2(V3, Wv[(pc - 1) * NAO .+ (1:NAO)])
        end
        return A
    end

    return Acode, Aref
end

@testset "periodic three-centre potential sum is complete" begin
    c = _setup()
    Acode, Aref = _blocks_and_reference(c)

    worst = Dict("onsite" => 0.0, "self-image" => 0.0, "off-diag" => 0.0)

    for (k, Ac) in Acode
        i, j, R = k
        Ar = Aref(i, j, SVector{3}(R))
        d  = maximum(abs, Ac - Ar) * _EV
        cls = (i == j && norm(R) < 1e-10) ? "onsite" :
              (i == j ? "self-image" : "off-diag")
        worst[cls] = max(worst[cls], d)
    end

    # What is left is the deliberate truncation: the smooth cutoff weight `w`
    # damps the outermost shell, so this is a meV bound, not machine
    # precision.  It converges with the pair cutoff (sto-3g Si, random Wv):
    #     rcut = 4.5 Å   14.2 / 11.0 /  9.4 meV
    #     rcut = 6.0 Å    0.05 / 0.09 /  0.59 meV
    # Before the fix, at rcut = 6.0 Å, these were 955 / 36 / 831 meV.
    @test worst["onsite"]     < 5e-3
    @test worst["self-image"] < 5e-3
    @test worst["off-diag"]   < 5e-3
end

@testset "the potential reaches at least as far as T and S" begin
    c = _setup()
    cut = c.G.graph_data.cut_integrals

    dT = maximum(norm(ed.Rvec) for ed in cut.T_off)
    dV = maximum(norm(ed.Rpair[2]) for ed in cut.V_off)   # bra-ket distance

    # A block with kinetic energy but no potential at all is unphysical.
    @test dV >= dT - 1e-8

    # The bra-atom potential term must be present in off-diagonal blocks.
    nbra = count(ed -> norm(ed.Rpair[1]) < 1e-10 && ed.integral_idx[1] != ed.integral_idx[3],
                 cut.V_off)
    @test nbra > 0

    # The R = 0 on-site block must receive potential centres other than itself.
    nonsite = count(ed -> ed.integral_idx[1] == ed.integral_idx[3] &&
                          norm(ed.R_cell) < 1e-10, cut.V_off)
    @test nonsite > 0
end

# ---------------------------------------------------------------------------
# The potential centre must not be tied to the bra's neighbourhood.  A centre
# inside the ket's sphere but outside the bra's used to be produced only by the
# ket-anchored triplet, land in the transposed block, and then be HALVED by
# assembly.jl's  H .= (H+H')/2  -- which is exactly the kind of error that
# symmetrisation hides.  With the potential centre also running over the ket's
# neighbours, the p-set of block (i,j,R) and of block (j,i,-R) are the same
# set and the RAW accumulation is already Hermitian.
#
# Measured on Si / sto-3g, max|A_ij(R) - A_ji(-R)'| in eV:
#     rcut (Å)   bra-side only      both sides
#       3.5        0.0994            7.6e-16
#       4.0        0.0166            1.1e-15
#       4.5        0.0167            1.0e-15
#       5.0        8.9e-4            9.9e-16
#       6.0        9.9e-16           9.9e-16   (the two p-sets coincide here)
# ---------------------------------------------------------------------------
@testset "raw accumulation is Hermitian without symmetrisation" begin
    for rcut_A in (4.0, 6.0)
        c = _setup(; rcut_A = rcut_A)
        Acode, _ = _blocks_and_reference(c)

        # +0.0 normalises -0.0, which isequal (and therefore Dict) rejects
        tkey(i, j, R) = (i, j, round.(R; digits = 6) .+ 0.0)

        worst = 0.0
        npair = 0
        for (k, M) in Acode
            i, j, R = k
            kt = tkey(j, i, -SVector{3}(R))
            haskey(Acode, kt) || continue
            npair += 1
            worst = max(worst, maximum(abs, M - Acode[kt]') * _EV)
        end

        @test npair == length(Acode)        # every block has its transpose
        @test worst < 1e-12                 # 0.0166 eV at rcut 4.0 before the fix
    end
end

# ---------------------------------------------------------------------------
# The two-pass cutoff estimate must not change the graph.
#
# `two_pass = true` estimates actual_rcut from a stratified subsample and drops
# the candidates that estimate would multiply by w = 0, instead of computing
# them and then zeroing them.  Measured on 2-atom Si at rcut = 6 Å:
#   sto-3g    4260 integrals -> 2796 + 516 probes = 3312, 22% fewer, and the
#             graph is bit identical (V_off 1308 both ways, 0 keys in one and
#             not the other, max|Δw| = 0.0, max|ΔV3| = 0.0)
#   def2-SVP  nothing to drop (its actual_rcut lands at the grid top 10.2,
#             sto-3g's at 7.8), so identical and no saving -- the waste is a
#             property of the basis, not of the code
# Wall clock, warm: 0.37-0.44 s -> 0.35-0.39 s, about 7%.  The three-centre
# integrals are not the bottleneck at this size; the 22% integral saving only
# turns into 22% wall clock when they are.
# ---------------------------------------------------------------------------
@testset "two_pass reproduces the one-pass graph exactly" begin
    sys = bulk(:Si)
    nlist = PairList(sys, 6.0 * u"Å")
    klist = [SVector(0.0, 0.0, 0.0)]

    _k(ed) = (ed.integral_idx, round.(ed.R_cell; digits = 6) .+ 0.0,
              round.(ed.Rpair[1]; digits = 5), round.(ed.Rpair[2]; digits = 5))

    for basis in ("sto-3g",)
        G1 = RealMLTB.nlist2graph_cut(basis, nlist, sys, klist; two_pass = false)
        G2 = RealMLTB.nlist2graph_cut(basis, nlist, sys, klist; two_pass = true)
        a = G1.graph_data.cut_integrals.V_off
        b = G2.graph_data.cut_integrals.V_off
        da = Dict(_k(ed) => (ed.w, ed.V3) for ed in a)
        db = Dict(_k(ed) => (ed.w, ed.V3) for ed in b)

        @test length(a) == length(b)
        @test isempty(setdiff(keys(da), keys(db)))
        @test isempty(setdiff(keys(db), keys(da)))
        for k in keys(da)
            @test da[k][1] == db[k][1]
            @test da[k][2] == db[k][2]
        end
    end
end

# The Cauchy-Schwarz pre-screen is rigorous, |∫φaφbφc| <= sqrt(∫φa²φb²), and
# off-switchable.  At cutoff = 1e-6 on these systems it removes NOTHING -- the
# squared overlap of two AOs inside the pair cutoff stays far above 1e-6 even at
# 6 Å -- so it is in for larger or more compact bases and tighter cutoffs, not
# because it pays here.  Measured: ntasks 4260 with the screen on and off.
@testset "cs_screen does not change the graph" begin
    sys = bulk(:Si)
    nlist = PairList(sys, 5.0 * u"Å")
    klist = [SVector(0.0, 0.0, 0.0)]
    n_on  = length(RealMLTB.nlist2graph_cut("sto-3g", nlist, sys, klist;
                                            cs_screen = true).graph_data.cut_integrals.V_off)
    n_off = length(RealMLTB.nlist2graph_cut("sto-3g", nlist, sys, klist;
                                            cs_screen = false).graph_data.cut_integrals.V_off)
    @test n_on == n_off
end
