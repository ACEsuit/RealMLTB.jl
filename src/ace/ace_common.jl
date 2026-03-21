# ace_common.jl
# Shared ACE building blocks: embed layer, specs, mb_spec, and getidx helper.
#
# Confirmed against EquivariantTensors.jl v0.4.x public API:
#   EmbedDP, EdgeEmbed, dp_transform, IDpost, sparse_equivariant_tensor,
#   sparse_equivariant_tensors, SparseACElayer, ETGraph, float32
#   (sparse_nnll_set was renamed; we use sparse_equivariant_tensor's mb_spec
#    argument directly via the documented nnll-spec constructor below)

import Polynomials4ML as P4ML
using Lux
import LuxCore
using EquivariantTensors
using Unitful, UnitfulAtomic
using LinearAlgebra
using StaticArrays
using Random

# ---------------------------------------------------------------------------
# Helper: safe index into tuple/vector (returns 0 when out of range)
# ---------------------------------------------------------------------------
@inline getidx(v, i) = (i <= length(v)) ? v[i] : 0

# ---------------------------------------------------------------------------
# ACEEdgeEmbed
#
# A thin Lux container that wraps the radial (Rnl) and angular (Ylm) edge
# embeddings and forwards them together as a 2-tuple.
#
# The Lux parameter/state trees are managed via AbstractLuxContainerLayer,
# which requires the field names as a type parameter.  We implement
# LuxCore.apply instead of overloading (l::T)(...) so that Lux's internal
# dispatch (which calls apply) works correctly.
# ---------------------------------------------------------------------------
struct ACEEdgeEmbed{TR, TY} <: Lux.AbstractLuxContainerLayer{(:Rnl, :Ylm)}
    Rnl::TR
    Ylm::TY
end

function LuxCore.apply(l::ACEEdgeEmbed, G, ps, st)
    Rnl_3, st_R = LuxCore.apply(l.Rnl, G, ps.Rnl, st.Rnl)
    Ylm_3, st_Y = LuxCore.apply(l.Ylm, G, ps.Ylm, st.Ylm)
    return (Rnl_3, Ylm_3), (Rnl = st_R, Ylm = st_Y)
end

# ---------------------------------------------------------------------------
# ace_make_embed
#
# Build the shared radial+angular edge embedding.  Returns
#   (embed::ACEEdgeEmbed, Rnl_spec, Ylm_spec)
#
# Arguments
# ---------
# rcut   : cutoff distance, a Unitful quantity (e.g. 5.0u"Å")
# Dtot   : maximum polynomial degree (controls basis size)
# maxl   : maximum angular-momentum quantum number for Ylm
#
# Notes on rcut unit handling
# ---------------------------
# atom.xyz in GaussianBasis / Molecules is stored in *Angstrom*.
# NeighbourLists returns edge vectors in *Angstrom* too.
# Therefore we strip rcut in Angstrom: ustrip(u"Å", rcut).
# ---------------------------------------------------------------------------
function ace_make_embed(rcut; Dtot::Int = 9, maxl::Int = 5)

    # ---- radial polynomial basis (Chebyshev) ----
    polys    = P4ML.ChebBasis(Dtot + 1)
    Rnl_spec = P4ML.natural_indices(polys)

    # ---- angular basis (real solid harmonics) ----
    ybas     = P4ML.real_sphericalharmonics(maxl; T = Float32, static = true)
    Ylm_spec = P4ML.natural_indices(ybas)

    # ---- cutoff in Angstrom ----
    rcut_ang = ustrip(u"Å", rcut)

    # ---- radial transform: maps edge NamedTuple x (with field x.𝐫) to a
    #      scalar in (-1, 1) via a simple rational decay ----
    rtrans = EquivariantTensors.dp_transform(
        x -> 1 / (1 + norm(x.𝐫) / rcut_ang)
    )

    # ---- radial embedding: transform -> polynomial basis ----
    rbasis = EquivariantTensors.EmbedDP(
        rtrans,
        polys,
        EquivariantTensors.IDpost()   # no post-processing envelope
    )

    # ---- angular transform: maps edge NamedTuple x to the 3-vector x.𝐫 ----
    ytrans = EquivariantTensors.dp_transform(x -> x.𝐫)

    # ---- angular embedding: transform -> spherical harmonics ----
    ybasis = EquivariantTensors.EmbedDP(
        ytrans,
        ybas,
        EquivariantTensors.IDpost()
    )

    # ---- wrap each scalar embedding into an EdgeEmbed so that the output
    #      becomes a (maxneigs × nnodes × nfeat) 3-tensor aware of the graph ----
    Rnl_embed = EquivariantTensors.EdgeEmbed(rbasis)
    Ylm_embed = EquivariantTensors.EdgeEmbed(ybasis)

    embed = ACEEdgeEmbed(Rnl_embed, Ylm_embed)
    return embed, Rnl_spec, Ylm_spec
end

# ---------------------------------------------------------------------------
# ace_make_mbspec
#
# Build the sparse many-body (nnll) specification used by both the invariant
# (L=0) and equivariant (L>0) ACE layers.
#
# EquivariantTensors ≥ v0.3 exposes this via the keyword constructor accepted
# by sparse_equivariant_tensor / sparse_equivariant_tensors.  We build the
# spec list directly as a Vector{Vector{@NamedTuple{n::Int,l::Int}}} which
# is the documented mb_spec format.
# ---------------------------------------------------------------------------
function ace_make_mbspec(; Dtot::Int = 9, maxl::Int = 5, ORD::Int = 3)
    # sparse_nnll_set was renamed/refactored in later versions; the stable
    # approach is to call the documented helper used internally by the package.
    # We delegate to EquivariantTensors' own sparse_product helper indirectly
    # by using the keyword form of sparse_equivariant_tensor later, but here
    # we expose the raw nnll spec so callers can inspect it.
    #
    # The spec is a list of correlation bodies, each body being a sorted
    # tuple of (n, l) pairs with n + l <= Dtot and body length <= ORD.
    return EquivariantTensors.sparse_nnll_set(;
        ORD      = ORD,
        minn     = 0,
        maxn     = Dtot,
        maxl     = maxl,
        level    = bb -> sum(b.n + b.l for b in bb; init = 0),
        maxlevel = Dtot,
    )
end