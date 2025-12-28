# Wv: equivariant ACE model, assembly, and rrule

using Lux, Random, LinearAlgebra
import LuxCore
using StaticArrays
using EquivariantTensors
using ChainRulesCore

"""
    build_ace_equivariant_model(n_channels, rcut; Dtot=9, maxl=5, ORD=3, lmax=length(n_channels)-1, rng=...)

Builds the equivariant ACE model for Wv (tensor outputs by L blocks).
Returns (model, ps, st).
"""
function build_ace_equivariant_model(n_channels, rcut;
    Dtot::Int = 9,
    maxl::Int = 5,
    ORD::Int  = 3,
    lmax::Int = length(n_channels) - 1,
    rng::AbstractRNG = MersenneTwister(1234),
)
    embed, Rnl_spec, Ylm_spec = ace_make_embed(rcut; Dtot=Dtot, maxl=maxl)
    mb_spec = ace_make_mbspec(; Dtot=Dtot, maxl=maxl, ORD=ORD)

    target_LL = Tuple(0:lmax)
    𝔹basis = EquivariantTensors.sparse_equivariant_tensors(;
        LL=target_LL,
        mb_spec=mb_spec,
        Rnl_spec=Rnl_spec,
        Ylm_spec=Ylm_spec,
        basis=real
    )

    ace = EquivariantTensors.SparseACElayer(𝔹basis, n_channels)
    model = Lux.Chain(; embed=embed, ace=ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

"""
    assemble_Wv(val_tuple)

Flatten equivariant outputs (by L blocks) into a single vector Wv.
This is a deterministic "layout" function, NOT physics.
"""
function assemble_Wv(val_tuple)
    T = eltype(val_tuple[1])
    n_atoms = size(val_tuple[1], 1)

    total_len = 0
    for (L_idx, block) in enumerate(val_tuple)
        L = L_idx - 1
        n_funcs  = 2L + 1
        n_shells = size(block, 2)
        total_len += n_atoms * n_shells * n_funcs
    end

    Wv = zeros(T, total_len)
    cursor = 1

    # PySCF-ish ordering fix (keep consistent with your tests)
    perm_L1 = SVector{3, Int}(3, 1, 2)
    perm_L2 = SVector{5, Int}(1, 2, 3, 4, 5)  # adjust if you ever permute D

    @inbounds for i in 1:n_atoms
        for (L_idx, block) in enumerate(val_tuple)
            L = L_idx - 1
            n_shells = size(block, 2)

            for j in 1:n_shells
                v = block[i, j]

                if L == 0
                    Wv[cursor] = v
                    cursor += 1

                elseif L == 1
                    Wv[cursor]     = v[perm_L1[1]]
                    Wv[cursor + 1] = v[perm_L1[2]]
                    Wv[cursor + 2] = v[perm_L1[3]]
                    cursor += 3

                elseif L == 2
                    @inbounds for k in 1:5
                        Wv[cursor + k - 1] = v[perm_L2[k]]
                    end
                    cursor += 5

                else
                    @inbounds for k in eachindex(v)
                        Wv[cursor + k - 1] = v[k]
                    end
                    cursor += length(v)
                end
            end
        end
    end

    return Wv
end

"""
rrule for assemble_Wv
"""
function ChainRulesCore.rrule(::typeof(assemble_Wv), val_tuple)
    y = assemble_Wv(val_tuple)

    function pullback(Δy)
        ΔWv = unthunk(Δy)
        n_atoms = size(val_tuple[1], 1)

        Δval_tuple = ntuple(length(val_tuple)) do L_idx
            blk = similar(val_tuple[L_idx])
            fill!(blk, zero(eltype(blk)))
            blk
        end

        cursor = 1
        perm_L1 = SVector{3, Int}(3, 1, 2)
        invperm_L1 = SVector{3, Int}(2, 3, 1)  # inverse of (3,1,2)

        perm_L2 = SVector{5, Int}(1, 2, 3, 4, 5)
        invperm_L2 = perm_L2  # identity now; if you permute, invert it here

        @inbounds for i in 1:n_atoms
            for L_idx in 1:length(Δval_tuple)
                L = L_idx - 1
                n_shells = size(Δval_tuple[L_idx], 2)

                for j in 1:n_shells
                    if L == 0
                        Δval_tuple[L_idx][i, j] = ΔWv[cursor]
                        cursor += 1

                    elseif L == 1
                        d1 = ΔWv[cursor]
                        d2 = ΔWv[cursor + 1]
                        d3 = ΔWv[cursor + 2]
                        # forward: out[k] = v[perm[k]]
                        # backward: v[t] accumulates out[invperm[t]]
                        Δval_tuple[L_idx][i, j] = SVector(d1, d2, d3)[invperm_L1]
                        cursor += 3

                    elseif L == 2
                        len = 5
                        Ts = eltype(val_tuple[L_idx][i,j])
                        tmp = ntuple(k -> Ts(ΔWv[cursor + k - 1]), Val(len))
                        # inverse permute back
                        Δval_tuple[L_idx][i, j] = SVector{len, Ts}(tmp)[invperm_L2]
                        cursor += len

                    else
                        len = 2L + 1
                        Ts = eltype(val_tuple[L_idx][i,j])
                        tmp = ntuple(k -> Ts(ΔWv[cursor + k - 1]), Val(len))
                        Δval_tuple[L_idx][i, j] = SVector{len, Ts}(tmp)
                        cursor += len
                    end
                end
            end
        end

        return NoTangent(), Δval_tuple
    end

    return y, pullback
end
