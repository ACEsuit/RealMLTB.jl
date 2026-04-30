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

# this is not needed anymore, we can just use a `Parallel` layer
# 
# function simple_rembed(Dtot)  
#    rbasis = P4ML.legendre_basis(Dtot+1)
#    Rn_spec = P4ML.natural_indices(rbasis) 
#    Rembed = ET.EdgeEmbed( ET.EmbedDP( 
#                      ET.dp_transform( x -> 1 / (1 + norm(x.𝐫)) ), 
#                      rbasis ) )
#    return Rembed, Rn_spec
# end



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

    # the polynomial basis just stays the same 
    # but needs to be wrapped due to the envelope being applied 
    # wrapped_basis makes the Lux layer behave like a P4ML basis 
    env_polys = P4ML.wrapped_basis( BranchLayer(
            polys,   # y -> P
            WrappedFunction( y -> env.(y) ),  # y -> fₑₙᵥ
            fusion = WrappedFunction( Pe -> Pe[2] .* Pe[1] )
         ) )

    Rnl_embed = EquivariantTensors.EdgeEmbed( ET.EmbedDP( 
            trans_r, env_polys ) ) 

    Ylm_embed = ET.EdgeEmbed( ET.EmbedDP( 
            ET.dp_transform( x -> x.𝐫 ), ybas ) )

    return Parallel(nothing; Rnl = Rnl_embed, Ylm = Ylm_embed), Rnl_spec, Ylm_spec    
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