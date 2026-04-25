# ace_common.jl
# Shared ACE building blocks: embed layer, specs, mb_spec, and getidx helper.

import Polynomials4ML as P4ML
using Lux
import LuxCore
using EquivariantTensors
using Unitful, UnitfulAtomic
using LinearAlgebra
using StaticArrays
using Random

import EquivariantTensors as ET

# ---------------------------------------------------------------------------
# Helper: safe index into tuple/vector (returns 0 when out of range)
# ---------------------------------------------------------------------------
@inline getidx(v, i) = (i <= length(v)) ? v[i] : 0

# ---------------------------------------------------------------------------
# ACEEdgeEmbed
#
# Container for the radial (Rnl) and angular (Ylm) edge embeddings.
# Returns a 2-tuple: (Rnl_3, Ylm_3)
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
# Build the shared radial+angular edge embedding. Returns
#   (embed::ACEEdgeEmbed, Rnl_spec, Ylm_spec)
#
# Radial branch keeps the original envelope logic:
#   y(r) = 1 / (1 + |r| / rcut_ang)
#   R_n(r) = env(y(r)) * polys(y(r))
# ---------------------------------------------------------------------------
function ace_make_embed(rcut; Dtot::Int = 9, maxl::Int = 5)

    polys    = P4ML.ChebBasis(Dtot + 1)
    Rnl_spec = P4ML.natural_indices(polys)

    ybas     = P4ML.real_sphericalharmonics(maxl; T = Float32, static = true)
    Ylm_spec = P4ML.natural_indices(ybas)

    rcut_ang = ustrip(u"Å", rcut)

    env(y) = (y - 1)^2 * (y + 1)^2

    # generate the radial transform 
    trans_r = ET.dp_transform( (x, st) -> 1 / (1 + norm(x.𝐫) / st.rcut), 
                            (;rcut = rcut_ang) )
    
    rbasis = Lux.Chain(
        trans = trans_r, 
        post = Lux.SkipConnection(
            polys,
            Lux.WrappedFunction(PY -> env.(PY[2]) .* PY[1])
        )
    )    
    Rnl_embed = EquivariantTensors.EdgeEmbed(rbasis)

    ybasis = Lux.Chain(
        trans = ET.dp_transform((x, st) -> x.𝐫, NamedTuple()), 
        basis = ybas
    )
    Ylm_embed = EquivariantTensors.EdgeEmbed(ybasis)

    embed = ACEEdgeEmbed(Rnl_embed, Ylm_embed)
    return embed, Rnl_spec, Ylm_spec
end
# ---------------------------------------------------------------------------
# ace_make_mbspec
#
# Build the sparse many-body (nnll) specification shared by invariant and
# equivariant ACE layers.
# ---------------------------------------------------------------------------
function ace_make_mbspec(; Dtot::Int = 9, maxl::Int = 5, ORD::Int = 3)
    return EquivariantTensors.sparse_nnll_set(;
        ORD      = ORD,
        minn     = 0,
        maxn     = Dtot,
        maxl     = maxl,
        level    = bb -> sum(b.n + b.l for b in bb; init = 0),
        maxlevel = Dtot,
    )
end