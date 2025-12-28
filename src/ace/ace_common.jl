# Shared ACE building blocks: embed, specs, mb_spec

import Polynomials4ML as P4ML
using Lux
using EquivariantTensors
using UnitfulAtomic

"""
    ace_make_embed(rcut; Dtot=9, maxl=5)

Create the common EdgeEmbed and return:
(embed, Rnl_spec, Ylm_spec)

- rcut: Unitful length
- Dtot: polynomial degree control
- maxl: maximum angular momentum used for spherical harmonics
"""
function ace_make_embed(rcut; Dtot::Int=9, maxl::Int=5)
    polys    = P4ML.ChebBasis(Dtot + 1)
    Rnl_spec = P4ML.natural_indices(polys)

    ybas     = P4ML.real_sphericalharmonics(maxl; T=Float32, static=true)
    Ylm_spec = P4ML.natural_indices(ybas)

    ycut = 1/2
    env  = y -> (y - ycut)^2 * (y + ycut)^2
    rcut_u = UnitfulAtomic.ustrip(rcut)

    rbasis = Lux.Chain(
        y = EquivariantTensors.NTtransformST((x, st) -> 1 / (1 + norm(x.𝐫 / rcut_u))),
        P = Lux.SkipConnection(polys, Lux.WrappedFunction(PY -> env.(PY[2]) .* PY[1]))
    )

    ybasis = Lux.Chain(
        trans = EquivariantTensors.NTtransformST((x, st) -> x.𝐫),
        basis = ybas
    )

    embed = EquivariantTensors.EdgeEmbed(
        BranchLayer(; Rnl = rbasis, Ylm = ybasis)
    )

    return embed, Rnl_spec, Ylm_spec
end

"""
    ace_make_mbspec(; Dtot=9, maxl=5, ORD=3)

Create the common sparse nnll spec.
"""
function ace_make_mbspec(; Dtot::Int=9, maxl::Int=5, ORD::Int=3)
    return EquivariantTensors.sparse_nnll_set(;
        ORD = ORD,
        minn = 0,
        maxn = Dtot,
        maxl = maxl,
        level = bb -> sum((b.n + b.l) for b in bb; init=0),
        maxlevel = Dtot
    )
end

# safe indexing for tuples/vectors (Case A uses it)
getidx(v, i) = (i <= length(v)) ? v[i] : 0
