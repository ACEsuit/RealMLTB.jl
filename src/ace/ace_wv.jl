# ace_wv.jl
# Equivariant ACE model for the real-space effective potential expansion
# coefficients W^V.
#
# Theory (from thesis Chapter 3):
#   V_θ[{R}](r) = Σ_{I,l,m}  W^V_{I,lm}(N_I) φ_{lm}(r - R_I)
#
# For the potential to be a scalar field, W^V_{I,lm} must transform as a
# rank-l SO(3) tensor (equivariant of order L = l).  We therefore output one
# equivariant channel per angular-momentum block l = 0, 1, ..., lmax, each
# with n_channels[l+1] independent "shells".
#
# The model:
#   G  --embed-->  (Rnl_3, Ylm_3)  --SparseACElayer(LL=0:lmax)--> val_tuple
#
# val_tuple is a Tuple of length (lmax+1):
#   val_tuple[L+1] is a Matrix of size (NATOM, n_channels[L+1])
#   where each entry val_tuple[L+1][i, j] is an SVector{2L+1} (for L > 0)
#   or a scalar Float32 (for L = 0).
#
# assemble_Wv flattens this into a single Float32 vector Wv of length
#   Σ_L  NATOM * n_channels[L+1] * (2L+1)
# following the atom-major, L-major, shell-major, m-minor layout that is
# consistent with the three-centre integral contraction:
#   V_{Ia,Jb} = Σ_{K,e}  Wv[K,e] * <φ_{Ia} φ_{Ke} φ_{Jb}>

using Lux, Random
import LuxCore
using StaticArrays
using EquivariantTensors
using ChainRulesCore
using LinearAlgebra

# ---------------------------------------------------------------------------
# build_ace_equivariant_model
# ---------------------------------------------------------------------------
"""
    build_ace_equivariant_model(n_channels, rcut; Dtot, maxl, ORD, rng)

Build an equivariant ACE model whose output is a tuple of SO(3) tensors at
angular momenta L = 0, 1, ..., lmax, with `n_channels[L+1]` channels each.

`n_channels` must be a tuple/vector of length (lmax + 1).

Returns `(model, ps, st)`.
"""
function build_ace_equivariant_model(n_channels, rcut;
    Dtot::Int          = 9,
    maxl::Int          = 5,
    ORD::Int           = 3,
    lmax::Int          = length(n_channels) - 1,
    rng::AbstractRNG   = MersenneTwister(1234),
)
    embed, Rnl_spec, Ylm_spec = ace_make_embed(rcut; Dtot = Dtot, maxl = maxl)
    mb_spec = ace_make_mbspec(; Dtot = Dtot, maxl = maxl, ORD = ORD)

    # Request equivariant tensors at all L = 0 … lmax simultaneously.
    target_LL = Tuple(0:lmax)
    𝔹basis = EquivariantTensors.sparse_equivariant_tensors(;
        LL       = target_LL,
        mb_spec  = mb_spec,
        Rnl_spec = Rnl_spec,
        Ylm_spec = Ylm_spec,
        basis    = real,
    )

    ace   = EquivariantTensors.SparseACElayer(𝔹basis, n_channels)
    model = Lux.Chain(; embed = embed, ace = ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

# ---------------------------------------------------------------------------
# _perm_for_L  –  m-index permutation to match the pyscf/GaussianBasis
#                 real-spherical-harmonic ordering convention.
#
# Convention used by GaussianBasis / acsint:
#   L=0  → [1]                       (trivial)
#   L=1  → [z, x, y]  i.e. perm [3,1,2] relative to [-1,0,+1] ordering
#   L≥2  → identity (no permutation needed)
#
# This must be consistent with construct_basis_D_matrix_pyscf in rotation.jl.
# ---------------------------------------------------------------------------
@inline function _perm_for_L(L::Int)
    if L == 0
        return nothing
    elseif L == 1
        return SVector{3,Int}(3, 1, 2)
    else
        return nothing  # identity
    end
end

# ---------------------------------------------------------------------------
# assemble_Wv
#
# Flatten the equivariant model output into a single coefficient vector that
# can be contracted with the three-centre integrals.
#
# Layout: for each atom i, for each L-block, for each shell j, the (2L+1)
# m-components are written consecutively (with the L=1 pyscf permutation).
# ---------------------------------------------------------------------------
function assemble_Wv(val_tuple)
    n_atoms = size(val_tuple[1], 1)
    T       = eltype(first(val_tuple[1]))   # scalar element type (Float32)

    total_len = 0
    for (Lidx, blk) in enumerate(val_tuple)
        L         = Lidx - 1
        n_shells  = size(blk, 2)
        total_len += n_atoms * n_shells * (2L + 1)
    end

    Wv     = zeros(T, total_len)
    cursor = 1

    @inbounds for i in 1:n_atoms
        for (Lidx, blk) in enumerate(val_tuple)
            L        = Lidx - 1
            n_shells = size(blk, 2)
            perm     = _perm_for_L(L)

            for j in 1:n_shells
                v = blk[i, j]   # SVector{2L+1} or scalar

                if L == 0
                    Wv[cursor] = v   # scalar
                    cursor += 1
                else
                    mlen = 2L + 1
                    if perm === nothing
                        for k in 1:mlen
                            Wv[cursor + k - 1] = v[k]
                        end
                    else
                        for k in 1:mlen
                            Wv[cursor + k - 1] = v[perm[k]]
                        end
                    end
                    cursor += mlen
                end
            end
        end
    end

    return Wv
end

# ---------------------------------------------------------------------------
# ChainRules pullback for assemble_Wv
# ---------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(assemble_Wv), val_tuple)
    y = assemble_Wv(val_tuple)

    function pullback_Wv(Δy)
        ΔWv     = ChainRulesCore.unthunk(Δy)
        n_atoms = size(val_tuple[1], 1)

        Δval_tuple = ntuple(length(val_tuple)) do Lidx
            blk = similar(val_tuple[Lidx])
            fill!(blk, zero(eltype(blk)))
            blk
        end

        cursor = 1
        @inbounds for i in 1:n_atoms
            for Lidx in 1:length(val_tuple)
                L        = Lidx - 1
                n_shells = size(val_tuple[Lidx], 2)
                perm     = _perm_for_L(L)

                for j in 1:n_shells
                    if L == 0
                        Δval_tuple[Lidx][i, j] = ΔWv[cursor]
                        cursor += 1
                    else
                        mlen = 2L + 1
                        Ts   = eltype(val_tuple[Lidx][i, j])
                        # build gradient SVector with inverse permutation
                        if perm === nothing
                            tmp = ntuple(k -> Ts(ΔWv[cursor + k - 1]), Val(mlen))
                        else
                            # inverse of perm: place ΔWv[cursor+k-1] back at
                            # position invperm[k] in the SVector
                            invp = _invperm_for_L(L, perm)
                            tmp  = ntuple(mlen) do k
                                Ts(ΔWv[cursor + invp[k] - 1])
                            end
                        end
                        Δval_tuple[Lidx][i, j] = SVector{mlen, Ts}(tmp)
                        cursor += mlen
                    end
                end
            end
        end

        return ChainRulesCore.NoTangent(), Δval_tuple
    end

    return y, pullback_Wv
end

# helper: build inverse permutation (as SVector) for the pullback
@inline function _invperm_for_L(L::Int, perm)
    mlen = 2L + 1
    ip   = zeros(Int, mlen)
    for k in 1:mlen
        ip[perm[k]] = k
    end
    return SVector{mlen, Int}(ip...)
end

function _ace_Wv_forward(ace_model, G, ps, st)
    return LuxCore.apply(ace_model, 
                                   G, 
                                   ps, st)
    # embed: stop gradient w.r.t. G (G is not a parameter)
    #Rnl_Ylm = Zygote.ignore() do
    #    LuxCore.apply(ace_model.layers.embed, G, ps.embed, st.embed)[1]
    #end
    #st_embed = LuxCore.apply(ace_model.layers.embed, G, ps.embed, st.embed)[2]
    #val_tuple, st_ace = LuxCore.apply(ace_model.layers.ace, Rnl_Ylm, ps.ace, st.ace)
    #return val_tuple, (embed=st_embed, ace=st_ace)
end
