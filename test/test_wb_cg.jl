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
# The general CG-coupled reduction,
#     W[(α L M),(s l m)] = Σ_{λμ} G^{L l λ}_{M m μ} v^(λ)_{α s μ}(env)
# of which the two existing layers are orthogonal slices: λ = 0 forces L = l
# (Case A, the Kronecker form) and L = 0 forces λ = l (Case B).  See
# src/model/wb_cg.jl.
# ---------------------------------------------------------------------------

_rot(sys, Q) = AtomsBase.FlexibleSystem(
    map(p -> AtomsBase.Atom(AtomsBase.species(p),
        SVector{3}(Q * ustrip.(Å, AtomsBase.position(p))) * u"Å";
        mass = AtomsBase.mass(p)), sys.particles);
    cell_vectors = [SVector{3}(Q * ustrip.(Å, c)) * u"Å"
                    for c in AtomsBase.cell_vectors(sys)],
    periodicity = sys.cell.periodicity)

@testset "ao_sh is orthonormal in the AO convention" begin
    q = RealMLTB._sphere_quad(12)
    for l in 0:4
        M = zeros(2l + 1, 2l + 1)
        for (w, r) in q
            y = ao_sh(l, r)
            M .+= w .* (y * y')
        end
        # per-m normalisation matters: cart2sph_hardcoded(2) has unequal column
        # norms (ratios [1.09,1.09,0.55,1.09,0.55]), so one factor for the whole
        # shell left l = 2 at ||<Y,Y> - I|| = 1.49
        @test norm(M - I) < 1e-12
    end
end

@testset "the Gaunt tensor is an intertwiner" begin
    Random.seed!(4)
    for (L, l, λ) in ((0,1,1), (1,0,1), (1,1,0), (2,1,1), (1,1,2),
                      (2,2,2), (3,2,1), (1,2,3))
        G = gaunt_real(L, l, λ)
        @test size(G) == (2L + 1, 2l + 1, 2λ + 1)

        r1 = normalize(randn(3)); r2 = normalize(randn(3)); r3 = normalize(randn(3))
        s(a, b, c) = sum(G[M, m, μ] * ao_sh(L, c)[M] * ao_sh(l, a)[m] * ao_sh(λ, b)[μ]
                         for M in 1:(2L+1), m in 1:(2l+1), μ in 1:(2λ+1))
        s0 = s(r1, r2, r3)
        @test abs(s0) > 1e-6                      # not a trivial zero
        for _ in 1:3
            Q = Matrix(rand(RotMatrix3))
            @test abs(s(Q*r1, Q*r2, Q*r3) - s0) < 1e-12 * max(abs(s0), 1.0)
        end
    end
end

@testset "Gaunt selection rules" begin
    for (L, l, λ) in ((1,1,1), (0,1,2), (3,1,1), (0,1,1), (2,1,1), (0,2,2))
        allowed = abs(l - λ) <= L <= l + λ && iseven(L + l + λ)
        m = maximum(abs, gaunt_real(L, l, λ))
        allowed ? (@test m > 1e-3) : (@test m == 0.0)
    end
end

# ---------------------------------------------------------------------------
# Regression guard.  Zygote.Buffer does not retain the contents of the array it
# is constructed from; entries never written come back as uninitialised memory.
# assemble_Wb_cg writes only the positions a coupling key contributes to, so
# without an explicit fill it returned subnormal garbage -- repeated calls
# differed by 1.96e-308 and the band drift wandered between 9e-10 and 90 Ha.
# ---------------------------------------------------------------------------
@testset "assemble_Wb_cg is deterministic and fully initialised" begin
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, [0.0, 0.0, 0.0])];
                  spherical = true, lib = :acsint)
    _, _, Nb, sbl = RealMLTB.wb_multiplet_layout(bs)
    n_out = (1, 1, 1)
    ks = [(L, L, 0) for L in 0:2]
    Random.seed!(13)
    W0 = [randn(n_out[L+1], length(sbl[l+1])) for (L, l, λ) in ks]
    V = [reshape(W0[ki], size(W0[ki])..., 1) for ki in eachindex(ks)]

    Ws = [assemble_Wb_cg(ks, V, bs, n_out) for _ in 1:8]
    for i in 2:8
        @test Ws[1] == Ws[i]            # bitwise, not approximately
    end
    # no entry may be a subnormal left over from uninitialised memory
    @test all(x -> x == 0 || abs(x) > 1e-300, Ws[1])
    # and the zero pattern has to be the structural one: with λ = 0 only, an
    # output of order L touches exactly the AO shells of l = L
    @test count(!iszero, Ws[1]) == sum(n_out[L+1] * (2L + 1) * length(sbl[L+1])
                                       for L in 0:2)
end

@testset "λ = 0 reproduces the multiplet form exactly" begin
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, [0.0, 0.0, 0.0])];
                  spherical = true, lib = :acsint)
    _, _, Nb, sbl = RealMLTB.wb_multiplet_layout(bs)
    n_out = (1, 1, 1)
    Random.seed!(7)
    C = Tuple(randn(n_out[li], length(sbl[li])) for li in 1:3)

    Wm = RealMLTB.assemble_Wb_multiplet(C, bs, n_out)
    V = Dict((L, L, 0) =>
             reshape(C[L+1] ./ gaunt_real(L, L, 0)[1, 1, 1],
                     n_out[L+1], length(sbl[L+1]), 1) for L in 0:2)
    Wc = assemble_Wb_cg(V, bs, n_out)

    @test size(Wc) == size(Wm)
    @test maximum(abs, Wc - Wm) < 1e-12
end

# ---------------------------------------------------------------------------
# Band-pipeline end to end.  These were out of the suite while the uninitialised
# Buffer made them wander; with that fixed the identical computation gives
# 9.34165744759241e-10 Ha at top level, inside a @testset, and on repeat calls
# in either place.
# ---------------------------------------------------------------------------
@testset "λ >= 1 is invariant, is new, and needs the feature to co-rotate" begin
    rcut = 5.0u"Å"
    # several k, not just Gamma: at Gamma every Bloch phase is 1, so a
    # Gamma-only test does not exercise the phase machinery.  The same reduced
    # k on the rotated cell is Qk (see test_rotation_e2e.jl), which is the
    # comparison the statement λ(Qk; rotated) = λ(k; original) asks for.
    klist = [SVector(0.0, 0.0, 0.0), SVector(0.137, 0.221, 0.079),
             SVector(0.25, 0.25, 0.25)]

    Random.seed!(2)
    s0 = bulk(:Si)
    ats = map(s0.particles) do p
        AtomsBase.Atom(AtomsBase.species(p),
            (ustrip.(Å, AtomsBase.position(p)) .+ 0.3 .* randn(3)) * u"Å";
            mass = AtomsBase.mass(p))
    end
    sys = AtomsBase.FlexibleSystem(ats; cell_vectors = AtomsBase.cell_vectors(s0),
                                   periodicity = s0.cell.periodicity)
    bs_of(sy) = (Y = [ustrip.(Å, r) for r in position(sy, :)];
                 BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, Y[i]) for i in 1:2];
                          spherical = true, lib = :acsint))
    bs = bs_of(sys)
    _, _, Nb, sbl = RealMLTB.wb_multiplet_layout(bs)
    n_out = (1, 1, 1)
    NORB = wb_cg_norb(n_out)
    @test NORB == 9

    ks0 = [(L, L, 0) for L in 0:2]
    ks1 = vcat(ks0, [(0,1,1), (1,0,1), (2,1,1), (1,2,1), (0,2,2)])
    Random.seed!(13)
    W0 = [randn(n_out[L+1], length(sbl[l+1])) for (L, l, λ) in ks0]
    W1 = [randn(n_out[L+1], length(sbl[l+1])) for (L, l, λ) in ks1]

    model = RealMLTB.acev_3c_model(bs, 4, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    ace = model.layers.layer_1.ace_model

    function bands(sy, ks, Wsc; freeze = nothing)
        bsy = bs_of(sy)
        f = freeze === nothing ? bond_features(sy, rcut, 2) : freeze
        Wb = zeros(2NORB, 2Nb)
        for A in 1:2
            V = [begin
                    L, l, λ = ks[ki]
                    ns = length(sbl[l+1])
                    a = zeros(n_out[L+1], ns, 2λ + 1)
                    for α in 1:n_out[L+1], s in 1:ns, μ in 1:(2λ + 1)
                        a[α, s, μ] = Wsc[ki][α, s] * f[A][λ+1][μ]
                    end
                    a
                 end for ki in eachindex(ks)]
            Wb[(A-1)*NORB .+ (1:NORB), (A-1)*Nb .+ (1:Nb)] =
                assemble_Wb_cg(ks, V, bsy, n_out)
        end
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sy, rcut), sy, klist)
        val, _ = RealMLTB._ace_Wv_forward(ace, G, ps.layer_1.ace, st.layer_1.ace)
        Wv = Float64.(RealMLTB.assemble_Wv(val))
        vH, vS = RealMLTB._3center_triplet(G, bsy, Wv, Wb)
        # physical k, from the graph -- see the note in test_rotation_e2e.jl
        H, S = RealMLTB._assemble_dense(G, vH, vS, G.graph_data.klist)
        λ, _ = RealMLTB.generate_λ_and_vecs(H, S)
        return [sort(real.(λ[i])) for i in eachindex(klist)]
    end

    f0 = bond_features(sys, rcut, 2)
    λ_cg = bands(sys, ks1, W1)
    λ_a  = bands(sys, ks0, W0)

    Random.seed!(9)
    for _ in 1:3
        Q = Matrix(rand(RotMatrix3))
        sr = _rot(sys, Q)
        b1 = bands(sr, ks1, W1)
        ba = bands(sr, ks0, W0)
        for i in eachindex(klist)
            @test maximum(abs, b1[i] - λ_cg[i]) < 1e-8
            @test maximum(abs, ba[i] - λ_a[i]) < 1e-8
        end
        broke = try
            bf = bands(sr, ks1, W1; freeze = f0)
            maximum(maximum(abs, bf[i] - λ_cg[i]) for i in eachindex(klist)) > 1e-3
        catch e
            e isa PosDefException || e isa LinearAlgebra.LAPACKException || rethrow()
            true
        end
        @test broke
    end
    @test maximum(maximum(abs, λ_cg[i] - λ_a[i]) for i in eachindex(klist)) > 1e-2
end

# ---------------------------------------------------------------------------
# Wv's form is FORCED, not a choice: for V(r) = Σ Wv[p,b] φ_b(r - R_p) to be an
# invariant function the coefficient on each (s,l,m) channel must transform as
# order l, and by Schur that is the only possibility.  What can be tested is
# that the implemented form's equivariance is LOAD-BEARING: freeze Wv in the old
# frame and the bands must move.
# ---------------------------------------------------------------------------
@testset "the bands depend on Wv's equivariance" begin
    rcut = 5.0u"Å"
    klist = [SVector(0.0, 0.0, 0.0)]
    Random.seed!(2)
    s0 = bulk(:Si)
    ats = map(s0.particles) do p
        AtomsBase.Atom(AtomsBase.species(p),
            (ustrip.(Å, AtomsBase.position(p)) .+ 0.3 .* randn(3)) * u"Å";
            mass = AtomsBase.mass(p))
    end
    sys = AtomsBase.FlexibleSystem(ats; cell_vectors = AtomsBase.cell_vectors(s0),
                                   periodicity = s0.cell.periodicity)
    bs_of(sy) = (Y = [ustrip.(Å, r) for r in position(sy, :)];
                 BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, Y[i]) for i in 1:2];
                          spherical = true, lib = :acsint))
    bs = bs_of(sys)
    n_out = (1, 1, 1)
    NORB = RealMLTB.wb_multiplet_norb(bs, n_out)
    model = RealMLTB.acev_3c_model(bs, NORB, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   wb_n_out = n_out, rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    ace = model.layers.layer_1.ace_model
    Wb_atom = RealMLTB.assemble_Wb_multiplet(ps.layer_1.Wb, bs, n_out)

    function bands(sy; freeze_Wv = nothing)
        bsy = bs_of(sy)
        G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sy, rcut), sy, klist)
        Wv = if freeze_Wv === nothing
            val, _ = RealMLTB._ace_Wv_forward(ace, G, ps.layer_1.ace, st.layer_1.ace)
            Float64.(RealMLTB.assemble_Wv(val))
        else
            freeze_Wv
        end
        Wb = RealMLTB.kron_I_atoms(2, Wb_atom)
        vH, vS = RealMLTB._3center_triplet(G, bsy, Wv, Wb)
        # physical k, from the graph -- see the note in test_rotation_e2e.jl
        H, S = RealMLTB._assemble_dense(G, vH, vS, G.graph_data.klist)
        λ, _ = RealMLTB.generate_λ_and_vecs(H, S)
        return [sort(real.(λ[i])) for i in eachindex(klist)]
    end

    G0 = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys, klist)
    v0, _ = RealMLTB._ace_Wv_forward(ace, G0, ps.layer_1.ace, st.layer_1.ace)
    Wv0 = Float64.(RealMLTB.assemble_Wv(v0))

    λ0 = bands(sys)
    Random.seed!(9)
    Q = Matrix(rand(RotMatrix3))
    sr = _rot(sys, Q)
    br = bands(sr)                       # equivariant Wv: invariant at every k
    for i in eachindex(klist)
        @test maximum(abs, br[i] - λ0[i]) < 1e-8
    end

    # Freezing Wv in the old frame must BREAK it -- that is the whole point.
    # Measured here: 0.0275 Ha = 748 meV.
    #
    # The probe only bites because the L >= 1 blocks of Wv are nonzero on a
    # RATTLED cell (norms: L=0 8.79, L=1 0.4896, L=2 0.2721).  On the ideal
    # cell they are exactly 0, and not by accident of the seed: the site group
    # T_d admits no l = 1 or l = 2 invariant, so an atom-centred invariant
    # potential has no such components there.  Hence the cell is rattled
    # before this test, or the freeze would look like a no-op.
    bf = bands(sr; freeze_Wv = Wv0)
    @test maximum(maximum(abs, bf[i] - λ0[i]) for i in eachindex(klist)) > 1e-3
end

# ---------------------------------------------------------------------------
# Feeding the CG coupling from the ACE model instead of a hand-made descriptor.
#
#     v^(λ)_{α s μ} = Σ_c A^{(L,l,λ)}[α,s,c] · Ŷ^(λ)_{c μ}
#
# A is the learnable invariant mixing; Ŷ is the ACE equivariant output reordered
# into the AO m convention.  That reordering is the only place a convention can
# break, so the first testset cross-validates it against assemble_Wv's OWN
# output rather than merely writing the same-looking code.
#
# Three Zygote obstacles were hit getting the gradient through this, all now
# handled in src/model/wb_cg.jl and worth knowing about:
#   * Float64.(v) on the Vector coming out of an SVector sent Zygote down its
#     StaticArrays broadcast path into the ForwardDiff fallback, which died on
#     bin_wrap_or_trunc(::Dual{Nothing,Int64,3}, ...) -- the INDEX arithmetic
#     was being handed Dual numbers.  Replaced by a comprehension.
#   * a loop over a Dict is not differentiable (MethodError: iterate(::Nothing)
#     while building the Dict tangent), so the primary API takes parallel
#     vectors and the Dict form is a non-AD wrapper.
#   * the memoising `get!` on the Gaunt cache, and the layout query, are
#     constants but were still traced and broke the pullback the same way;
#     both are now inside ChainRulesCore.ignore_derivatives.
# ---------------------------------------------------------------------------
@testset "the ACE m-ordering bridge is identical to assemble_Wv's" begin
    rcut = 5.0u"Å"
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    model = RealMLTB.acev_3c_model(bs, 4, rcut; Dtot = 4, maxl = 3, ORD = 1,
                                   rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, rcut), sys,
                                 [SVector(0.0, 0.0, 0.0)])
    val, _ = RealMLTB._ace_Wv_forward(model.layers.layer_1.ace_model, G,
                                      ps.layer_1.ace, st.layer_1.ace)
    Wv = Float64.(RealMLTB.assemble_Wv(val))

    cursor = 1
    worst = 0.0
    for i in 1:2, (Lidx, blk) in enumerate(val)
        L = Lidx - 1
        for j in 1:size(blk, 2)
            mine = Float64.(ace_feature_ao_order(val, L, i, j))
            worst = max(worst, maximum(abs, mine .- Wv[cursor:(cursor + 2L)]))
            cursor += 2L + 1
        end
    end
    @test cursor - 1 == length(Wv)      # the whole vector was covered
    @test worst == 0.0                  # bit-identical, not merely close
end

@testset "the mixing is differentiable" begin
    sys = bulk(:Si)
    X = [ustrip.(Å, r) for r in position(sys, :)]
    bs = BasisSet("def2-SVP", [Molecules.Atom(14, 28.0, X[i]) for i in 1:2];
                  spherical = true, lib = :acsint)
    _, _, Nb, sbl = RealMLTB.wb_multiplet_layout(bs)
    lmax, nch = RealMLTB.lmax_and_nchannels_per_atom(bs)
    n_out = (1, 1, 1)
    model = RealMLTB.acev_3c_model(bs, 4, 5.0u"Å"; Dtot = 4, maxl = 3, ORD = 1,
                                   rng = MersenneTwister(0))
    ps, st = LuxCore.setup(MersenneTwister(1), model)
    G = RealMLTB.nlist2graph_cut("def2-SVP", PairList(sys, 5.0u"Å"), sys,
                                 [SVector(0.0, 0.0, 0.0)])
    val, _ = RealMLTB._ace_Wv_forward(model.layers.layer_1.ace_model, G,
                                      ps.layer_1.ace, st.layer_1.ace)
    Yf = ace_feature_matrices(val, 1, lmax)

    ks = vcat([(L, L, 0) for L in 0:2], [(0,1,1), (1,0,1), (2,1,1), (1,2,1), (0,2,2)])
    A = init_cg_mixing(MersenneTwister(21), ks, n_out, sbl, nch)
    sk = sort(collect(keys(A)))
    sizes = [size(A[kk]) for kk in sk]
    offs = cumsum([0; [prod(sz) for sz in sizes]])
    x0 = vcat((vec(A[kk]) for kk in sk)...)
    unflat(x) = [reshape(x[(offs[i]+1):offs[i+1]], sizes[i]) for i in eachindex(sizes)]
    f(x) = sum(abs2, assemble_Wb_cg(sk, ace_cg_features(Yf, sk, unflat(x), n_out, sbl),
                                    bs, n_out))

    g = Zygote.gradient(f, x0)[1]
    @test g !== nothing && length(g) == length(x0)
    for i in (1, 5, 17, length(x0))
        h = 1e-6
        a = copy(x0); b = copy(x0); a[i] += h; b[i] -= h
        @test abs(g[i] - (f(a) - f(b)) / 2h) <= 1e-5 * max(abs((f(a) - f(b)) / 2h), 1.0)
    end
end
