# ---------------------------------------------------------------------------
# General CG-coupled reduction matrix.
#
# Every equivariant Wpsi is a Clebsch-Gordan contraction of an environment
# feature with the AO angular index,
#
#     W[(α L M), (s l m)] = Σ_{λ μ} C^{LM}_{l m; λ μ} · v^(λ)_{α s μ}(env)
#
# of which the two previously implemented layers are orthogonal slices:
#     L = 0, any λ (forced λ = l)  -> Case B,      H invariant
#     any L, λ = 0 (forces L = l)  -> Case A,      H equivariant, Kronecker form
# leaving the interior, any L with any λ, unavailable.  The headroom is real: a
# λ = 1 channel built by hand changes the bands by 139 eV at identical
# rotation invariance (1.9e-5 meV), and freezing the feature breaks that
# invariance (1787 meV), so it is the new channel's equivariance doing the
# work rather than extra parameters.
#
# The intertwiner.  For real spherical harmonics the coupling
# D^(l) ⊗ D^(λ) -> D^(L) is the real Gaunt coefficient
#
#     G^{L l λ}_{M m μ} = ∫ Y_LM Y_lm Y_λμ dΩ
#
# which is invariant under rotating all three indices together -- exactly the
# equivariance needed.  The overall normalisation is irrelevant because the
# features are learned.
#
# Convention, which is where the risk sits.  G must be in the same real-SH
# convention as the AO (l,m) ordering, or the coupling silently breaks
# equivariance.  So the angular functions are defined from the AO side via
# cart2sph, and G is integrated numerically from those same functions; nothing
# is imported from an external CG table, so there is no convention to
# reconcile.
# ---------------------------------------------------------------------------

export ao_sh, gaunt_real, assemble_Wb_cg, bond_features, wb_cg_norb,
       ace_cg_features, ace_feature_ao_order, ace_feature_matrices,
       init_cg_mixing

const _CARTLIST = Dict{Int,Vector{NTuple{3,Int}}}()
_cartlist(l) = get!(_CARTLIST, l) do
    [(lx, ly, l - lx - ly) for lx in l:-1:0 for ly in (l - lx):-1:0]
end

"""
    ao_sh(l, r̂) -> Vector of length 2l+1

The AO basis's own real spherical harmonics at a unit vector, built from
`cart2sph(l)` so the (l,m) ordering and signs match the AO index exactly, and
normalised to ∫|Y|² dΩ = 1.
"""
function ao_sh(l::Int, r::AbstractVector)
    C = cart2sph(l)
    mono = [r[1]^c[1] * r[2]^c[2] * r[3]^c[3] for c in _cartlist(l)]
    return vec(mono' * C) ./ _ao_sh_norm(l)
end

const _AO_SH_NORM = Dict{Int,Vector{Float64}}()
"""
Per-m normalisation, NOT one factor for the whole shell.

`cart2sph_hardcoded(2)` does not have equal column norms: the scale ratios
against the generic formula are [1.0925, 1.0925, 0.5463, 1.0925, 0.5463], a
factor of two between {dxy,dyz,dxz} and {dz², dx²-y²}.  The AO pipeline hides
this, because `d_sph` renormalises diag(S_sph) = 1 afterwards, i.e. per
function.  One RMS factor for the whole shell leaves l = 2 non-orthonormal
(||<Y,Y> - I|| = 1.49, against 1e-15 for l = 0,1,3,4), which stops the Gaunt
tensor from being an intertwiner for exactly the triples with an index of 2.
Normalising per m also matches the AO convention.
"""
function _ao_sh_norm(l::Int)
    get!(_AO_SH_NORM, l) do
        q = _sphere_quad(2l + 4)
        C = cart2sph(l)
        acc = zeros(2l + 1)
        for (w, r) in q
            mono = [r[1]^c[1] * r[2]^c[2] * r[3]^c[3] for c in _cartlist(l)]
            acc .+= w .* abs2.(vec(mono' * C))
        end
        sqrt.(acc)
    end
end

"""
    _sphere_quad(deg) -> Vector{(weight, point)}

Gauss-Legendre in cos θ times the trapezoid in φ, which is exact for spherical
harmonics up to the requested degree.  Weights sum to 4π.
"""
function _sphere_quad(deg::Int)
    nθ = deg + 2
    nφ = 2 * deg + 4
    x, wx = _gauss_legendre(nθ)
    out = Tuple{Float64,SVector{3,Float64}}[]
    for (ct, w) in zip(x, wx), j in 0:(nφ - 1)
        st = sqrt(max(0.0, 1 - ct^2))
        φ = 2π * j / nφ
        push!(out, (w * 2π / nφ, SVector(st * cos(φ), st * sin(φ), ct)))
    end
    out
end

function _gauss_legendre(n::Int)
    # Newton on the Legendre polynomial; n is small here
    x = zeros(n); w = zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, 0.0
            for k in 1:n
                p1, p0 = p0, ((2k - 1) * z * p0 - (k - 1) * p1) / k
            end
            dp = n * (z * p0 - p1) / (z^2 - 1)
            dz = p0 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0, p1 = 1.0, 0.0
        for k in 1:n
            p1, p0 = p0, ((2k - 1) * z * p0 - (k - 1) * p1) / k
        end
        dp = n * (z * p0 - p1) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    x, w
end

"""
    gaunt_real(L, l, λ) -> Array{Float64,3}    size (2L+1, 2l+1, 2λ+1)

`G[M,m,μ] = ∫ Y_LM Y_lm Y_λμ dΩ` in the AO convention, by quadrature.  Zero
(to quadrature accuracy) unless |l-λ| <= L <= l+λ and L+l+λ is even.
"""
const _GAUNT = Dict{NTuple{3,Int},Array{Float64,3}}()
function gaunt_real(L::Int, l::Int, λ::Int)
    get!(_GAUNT, (L, l, λ)) do
        q = _sphere_quad(L + l + λ + 2)
        G = zeros(2L + 1, 2l + 1, 2λ + 1)
        for (w, r) in q
            YL = ao_sh(L, r); Yl = ao_sh(l, r); Yλ = ao_sh(λ, r)
            for M in 1:(2L + 1), m in 1:(2l + 1), μ in 1:(2λ + 1)
                G[M, m, μ] += w * YL[M] * Yl[m] * Yλ[μ]
            end
        end
        map!(x -> abs(x) < 1e-12 ? 0.0 : x, G, G)
        G
    end
end

"""
    wb_cg_norb(n_out) -> Int

`Σ_L n_out[L+1] (2L+1)`: the output index carries (L,M) exactly as in the
λ = 0 case, so the group-theory argument for the degeneracies at high-symmetry
points is kept.
"""
wb_cg_norb(n_out) = sum(n_out[L+1] * (2L + 1)
                        for L in 0:(length(n_out)-1) if n_out[L+1] > 0)

"""
    assemble_Wb_cg(V, bs, n_out; lmax_feature) -> Matrix

`V[(L, l, λ)][α, s, μ]` is the order-λ feature for output channel α of order L
against the AO shells of angular momentum l; `s` indexes those shells in order.
Missing keys are treated as zero.  Returns the atom's NORB x Nb block,
NORB = wb_cg_norb(n_out).

λ = 0 reproduces `assemble_Wb_multiplet` exactly (G^{L l 0} ∝ δ_{LM,lm}), and
L = 0 reproduces Case B's form, so this subsumes both slices.

Built by accumulation over keys with no mutation of the differentiated path.
"""
# A Dict is convenient to write but Zygote cannot differentiate a loop over one
# -- it fails with `MethodError: iterate(::Nothing)` building the Dict tangent.
# So the primary method takes PARALLEL VECTORS, keys (not differentiated) and
# values, and the Dict form is a non-AD convenience wrapper.
function assemble_Wb_cg(V::AbstractDict, bs, n_out; kwargs...)
    ks = sort(collect(keys(V)))
    return assemble_Wb_cg(ks, [V[k] for k in ks], bs, n_out; kwargs...)
end

function assemble_Wb_cg(ks::AbstractVector, Vs::AbstractVector, bs, n_out;
                        lmax_feature::Int = 2)
    # The layout and the Gaunt tensors are CONSTANTS with respect to the
    # parameters, but Zygote traces them anyway and the memoising `get!` on the
    # Gaunt cache then fails the whole pullback with
    #   MethodError: no method matching iterate(::Nothing)
    # so they are explicitly fenced off.
    _, offsets, Nb, shells_by_l =
        ChainRulesCore.ignore_derivatives(() -> wb_multiplet_layout(bs))
    lmax_ao = length(shells_by_l) - 1
    Lmax = length(n_out) - 1
    NORB = wb_cg_norb(n_out)

    T = isempty(Vs) ? Float64 : promote_type(map(eltype, Vs)...)

    # row offset of the (L, α) block; the 2L+1 M rows follow contiguously, which
    # is what keeps whole multiplets together
    rowblk = Dict{Tuple{Int,Int},Int}()
    r = 0
    for L in 0:Lmax, α in 1:n_out[L+1]
        rowblk[(L, α)] = r
        r += 2L + 1
    end
    @assert r == NORB

    # One BLAS call per coupling key.  The first version evaluated
    #   [sum(vals[k] for k if rows[k]==i && cols[k]==j) for i in 1:NORB, j in 1:Nb]
    # which is O(NORB · Nb · nnz) and was the dominant cost of the whole
    # assembly.  Here the (M,m) block for every (α,s) at once is
    #   reshape(G, (2L+1)(2l+1), 2λ+1) * reshape(v, n_out·ns, 2λ+1)'
    # and the result is scattered into a Zygote.Buffer, which is mutable and
    # still differentiates.
    # Zygote.Buffer does NOT retain the contents of the array it is given -- it
    # tracks writes, and any entry never written comes back as whatever was in
    # memory.  Every position where no coupling key contributes is such an
    # entry, so without this fill the function returns uninitialised doubles.
    #
    # That bug cost a long detour.  It looked like a "@testset interaction":
    # the identical deterministic function returned 9.34e-10 Ha at top level
    # and 41.4 Ha inside a @testset, and two calls inside the same testset gave
    # 14.25 and 90.53.  Narrowing it by component showed bond_features,
    # Zygote.Buffer does NOT retain the contents of the array it is handed, so
    # this zero-fill is required and is not redundant.  Without it the result
    # varies between identical calls by ~1e-308, a subnormal: the signature of
    # reading uninitialised memory, and easily hidden by freshly zeroed pages.
    buf = Zygote.Buffer(zeros(T, NORB, Nb))
    for i in 1:NORB, j in 1:Nb
        buf[i, j] = zero(T)
    end

    for (ki, (L, l, λ)) in enumerate(ks)
        v = Vs[ki]
        0 <= L <= Lmax || error("key ($L,$l,$λ): L out of range for n_out")
        0 <= l <= lmax_ao || error("key ($L,$l,$λ): the basis has no l = $l shell")
        λ <= lmax_feature || error("key ($L,$l,$λ): λ exceeds lmax_feature")
        n_out[L+1] > 0 || error("key ($L,$l,$λ): n_out[$(L+1)] = 0")
        shells = shells_by_l[l+1]
        nα, ns, nμ = n_out[L+1], length(shells), 2λ + 1
        size(v) == (nα, ns, nμ) || error(
            "V[($L,$l,$λ)] is $(size(v)), expected $((nα, ns, nμ))")

        G = ChainRulesCore.ignore_derivatives(() -> gaunt_real(L, l, λ))
        all(iszero, G) && continue

        # ((2L+1)(2l+1)) × (nα·ns)
        B = reshape(G, (2L + 1) * (2l + 1), nμ) * reshape(v, nα * ns, nμ)'

        for α in 1:nα, (si, s) in enumerate(shells)
            c = (si - 1) * nα + α          # column of B for this (α, s)
            r0 = rowblk[(L, α)]
            o = offsets[s]
            for m in 1:(2l + 1), M in 1:(2L + 1)
                g = B[(m - 1) * (2L + 1) + M, c]
                g == 0 && continue
                buf[r0 + M, o + m] += g
            end
        end
    end

    return copy(buf)
end

"""
    bond_features(sys, rcut, lmax; f = d -> exp(-d/2)) -> Vector of tuples

Per-atom two-body equivariant features in the AO convention,

    v^(λ)_{A μ} = Σ_{B ∈ nbr(A)} Y^AO_{λ μ}(R̂_AB) f(|R_AB|)

so `out[A][λ+1]` is a vector of length 2λ+1 transforming as D^(λ).  This is a
legitimate if minimal descriptor -- two-body only -- and exists so the CG
machinery can be exercised and trained end to end without first solving the
separate problem of mapping the ACE output tuple onto the (L,l,λ,α,s) indexing.
Swapping in the ACE features changes only where v^(λ) comes from.
"""
function bond_features(sys, rcut, lmax::Int; f = d -> exp(-d / 2))
    nl = PairList(sys, rcut)
    nat = length(nl.first) - 1
    out = [Tuple(zeros(2λ + 1) for λ in 0:lmax) for _ in 1:nat]
    for i in 1:nat, e in nl.first[i]:(nl.first[i+1] - 1)
        Rv = NeighbourLists._getR(nl, e)
        d = norm(Rv)
        d < 1e-10 && continue
        r̂ = SVector{3}(Rv ./ d)
        wd = f(d)
        for λ in 0:lmax
            out[i][λ+1] .+= wd .* ao_sh(λ, r̂)
        end
    end
    out
end


# ---------------------------------------------------------------------------
# ACE features -> the (L, l, λ, α, s) indexing the CG coupling wants.
#
# The ACE model returns val_tuple[λ+1][atom, c] :: SVector{2λ+1}, one
# equivariant vector per channel c.  What assemble_Wb_cg needs is
# v^(λ)[α, s, μ] for each coupling key, so the adapter is a learnable INVARIANT
# mixing of the ACE channels onto the (output α, AO shell s) pairs:
#
#     v^(λ)_{α s μ} = Σ_c A^{(L,l,λ)}[α, s, c] · Ŷ^(λ)_{c μ}
#
# with Ŷ the ACE vector REORDERED into the AO m convention.  That reordering is
# the only place a convention can go wrong, so it is done with the same
# _perm_for_L that assemble_Wv uses, and `ace_feature_ao_order` is
# cross-validated against assemble_Wv's own output in the tests -- not merely
# written to look the same.
# ---------------------------------------------------------------------------

"""
    ace_feature_ao_order(val_tuple, λ, atom, c) -> Vector of length 2λ+1

One ACE equivariant vector, reordered from the ACE real-SH m ordering into the
AO ordering, exactly as assemble_Wv does it:  v[perm[k]] -> position k.
"""
function ace_feature_ao_order(val_tuple, λ::Int, atom::Int, c::Int)
    v = val_tuple[λ+1][atom, c]
    λ == 0 && return [only_scalar(v)]
    perm = _perm_for_L(λ)
    mlen = 2λ + 1
    return perm === nothing ? [v[k] for k in 1:mlen] : [v[perm[k]] for k in 1:mlen]
end

only_scalar(v) = v isa Number ? v : v[1]

"""
    init_cg_mixing(rng, keys, n_out, shells_by_l, n_channels) -> Dict

One mixing tensor per coupling key, `A[key] :: (n_out[L+1], n_shell_l,
n_channels[λ+1])`.  These are the learnable parameters; everything angular is
fixed by the Gaunt tensor.
"""
function init_cg_mixing(rng, keys, n_out, shells_by_l, n_channels)
    D = Dict{NTuple{3,Int},Array{Float64,3}}()
    for (L, l, λ) in keys
        nc = n_channels[λ+1]
        nc > 0 || error("key ($L,$l,$λ): the ACE model has no λ = $λ channel")
        D[(L, l, λ)] = randn(rng, n_out[L+1], length(shells_by_l[l+1]), nc) ./
                       sqrt(nc)
    end
    D
end

"""
    ace_feature_matrices(val_tuple, atom, lmax) -> Vector{Matrix}

`Y[λ+1]` of size (2λ+1, n_channels_λ): every ACE equivariant vector for one
atom, reordered into the AO m convention, as a plain dense matrix.

Separated from the contraction below deliberately.  Doing the extraction inside
the contraction made Zygote trace `reduce(hcat, <generator>)` over StaticArrays
once per coupling key; it fell back to ForwardDiff and died with
   MethodError: bin_wrap_or_trunc(::ForwardDiff.Dual{Nothing, Int64, 3}, ...)
i.e. the index arithmetic was being handed Dual numbers.  Extracting once per
atom into dense matrices leaves the differentiated path as pure matmul and
reshape -- and is also n_keys times less work.
"""
function ace_feature_matrices(val_tuple, atom::Int, lmax::Int)
    out = Vector{Matrix{Float64}}(undef, lmax + 1)
    for λ in 0:lmax
        nc = size(val_tuple[λ+1], 2)
        # a comprehension, not Float64.(...): broadcasting over the Vector that
        # comes out of an SVector sent Zygote down its StaticArrays broadcast
        # path and into the ForwardDiff fallback
        cols = [[Float64(z) for z in ace_feature_ao_order(val_tuple, λ, atom, c)]
                for c in 1:nc]
        out[λ+1] = reduce(hcat, cols)
    end
    out
end

"""
    ace_cg_features(Y, A, n_out, shells_by_l) -> Dict

`V[(L,l,λ)][α,s,μ] = Σ_c A[(L,l,λ)][α,s,c] · Y[λ+1][μ,c]`, ready for
assemble_Wb_cg.  `Y` comes from `ace_feature_matrices`.  Pure matmul, so it
differentiates with respect to A.
"""
function ace_cg_features(Y, A::AbstractDict, n_out, shells_by_l)
    ks = sort(collect(keys(A)))
    return ks, ace_cg_features(Y, ks, [A[k] for k in ks], n_out, shells_by_l)
end

function ace_cg_features(Y, ks::AbstractVector, As::AbstractVector,
                         n_out, shells_by_l)
    return [begin
                L, l, λ = ks[ki]
                a = As[ki]
                nα, ns, nc = size(a)
                size(Y[λ+1], 2) == nc || error(
                    "key ($L,$l,$λ): mixing has $nc channels, " *
                    "features have $(size(Y[λ+1], 2))")
                reshape(reshape(a, nα * ns, nc) * Y[λ+1]', nα, ns, 2λ + 1)
            end for ki in eachindex(ks)]
end
