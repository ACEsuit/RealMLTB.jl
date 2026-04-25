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

function assemble_Wv(val_tuple::Tuple{Vararg{<:CUDA.CuArray}})
    return _assemble_Wv_gpu(val_tuple)
end

using CUDA

@inline function _wv_layout_meta(val_tuple)
    n_atoms = size(val_tuple[1], 1)
    nL = length(val_tuple)

    L_offsets = Vector{Int}(undef, nL)

    offset = 0
    for Lidx in 1:nL
        L = Lidx - 1
        n_shells = size(val_tuple[Lidx], 2)
        mlen = 2L + 1

        L_offsets[Lidx] = offset
        offset += n_shells * mlen
    end

    atom_stride = offset
    total_len = n_atoms * atom_stride

    return n_atoms, L_offsets, atom_stride, total_len
end

@inline _wv_perm_src(::Val{1}, m::Int) = ifelse(m == 1, 3, ifelse(m == 2, 1, 2))
@inline _wv_perm_src(::Val{L}, m::Int) where {L} = m

function _assemble_wv_L0_kernel!(Wv, blk, n_shells::Int, atom_stride::Int, Loff::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_atoms = size(blk, 1)
    N = n_atoms * n_shells

    if q <= N
        t = q - 1
        j = (t % n_shells) + 1
        i = (t ÷ n_shells) + 1

        dst = (i - 1) * atom_stride + Loff + j
        Wv[dst] = blk[i, j]
    end

    return nothing
end

function _assemble_wv_L_kernel!(Wv, blk, n_shells::Int, atom_stride::Int, Loff::Int, ::Val{L}) where {L}
    mlen = 2L + 1

    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_atoms = size(blk, 1)
    N = n_atoms * n_shells * mlen

    if q <= N
        t = q - 1

        m = (t % mlen) + 1
        t2 = t ÷ mlen

        j = (t2 % n_shells) + 1
        i = (t2 ÷ n_shells) + 1

        src_m = _wv_perm_src(Val(L), m)

        dst = (i - 1) * atom_stride + Loff + (j - 1) * mlen + m
        Wv[dst] = blk[i, j][src_m]
    end

    return nothing
end

function _assemble_Wv_gpu(val_tuple)
    n_atoms, L_offsets, atom_stride, total_len = _wv_layout_meta(val_tuple)

    # L = 0 block has scalar element type Float32/Float64.
    T = eltype(val_tuple[1])

    Wv = CUDA.zeros(T, total_len)

    threads = 256

    for Lidx in 1:length(val_tuple)
        L = Lidx - 1
        blk = val_tuple[Lidx]
        n_shells = size(blk, 2)
        Loff = L_offsets[Lidx]

        if n_shells == 0
            continue
        end

        if L == 0
            N = n_atoms * n_shells
            blocks = cld(N, threads)
            @cuda threads=threads blocks=blocks _assemble_wv_L0_kernel!(
                Wv, blk, n_shells, atom_stride, Loff
            )
        else
            mlen = 2L + 1
            N = n_atoms * n_shells * mlen
            blocks = cld(N, threads)
            @cuda threads=threads blocks=blocks _assemble_wv_L_kernel!(
                Wv, blk, n_shells, atom_stride, Loff, Val(L)
            )
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




using CUDA
using StaticArrays
using ChainRulesCore
using ChainRulesCore: NoTangent, unthunk

const CuValTuple = Tuple{Vararg{<:CUDA.CuArray}}

# Same layout as assemble_Wv:
# atom-major -> L-major -> shell-major -> m-minor
@inline function _wv_layout_meta_gpu(val_tuple)
    n_atoms = size(val_tuple[1], 1)
    nL = length(val_tuple)

    L_offsets = Vector{Int}(undef, nL)

    offset = 0
    for Lidx in 1:nL
        L = Lidx - 1
        n_shells = size(val_tuple[Lidx], 2)
        mlen = 2L + 1

        L_offsets[Lidx] = offset
        offset += n_shells * mlen
    end

    atom_stride = offset
    total_len = n_atoms * atom_stride

    return n_atoms, L_offsets, atom_stride, total_len
end

function _assemble_Wv_gpu_pb_L0_kernel!(
    Δblk,
    ΔWv,
    n_shells::Int,
    atom_stride::Int,
    Loff::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_atoms = size(Δblk, 1)
    N = n_atoms * n_shells

    if q <= N
        t = q - 1

        j = (t % n_shells) + 1
        i = (t ÷ n_shells) + 1

        src = (i - 1) * atom_stride + Loff + j

        Δblk[i, j] = ΔWv[src]
    end

    return nothing
end

function _assemble_Wv_gpu_pb_L1_kernel!(
    Δblk,
    ΔWv,
    n_shells::Int,
    atom_stride::Int,
    Loff::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_atoms = size(Δblk, 1)
    N = n_atoms * n_shells

    if q <= N
        t = q - 1

        j = (t % n_shells) + 1
        i = (t ÷ n_shells) + 1

        base = (i - 1) * atom_stride + Loff + (j - 1) * 3 + 1

        T = eltype(eltype(Δblk))

        # forward L=1 convention:
        #   Wv[m=1] = v[3]
        #   Wv[m=2] = v[1]
        #   Wv[m=3] = v[2]
        #
        # backward:
        #   dv[1] = dWv[m=2]
        #   dv[2] = dWv[m=3]
        #   dv[3] = dWv[m=1]
        Δblk[i, j] = SVector{3,T}(
            T(ΔWv[base + 1]),
            T(ΔWv[base + 2]),
            T(ΔWv[base]),
        )
    end

    return nothing
end

function _assemble_Wv_gpu_pb_L_kernel!(
    Δblk,
    ΔWv,
    n_shells::Int,
    atom_stride::Int,
    Loff::Int,
    ::Val{L},
) where {L}
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_atoms = size(Δblk, 1)
    N = n_atoms * n_shells

    if q <= N
        t = q - 1

        j = (t % n_shells) + 1
        i = (t ÷ n_shells) + 1

        Ncomp = 2L + 1
        base = (i - 1) * atom_stride + Loff + (j - 1) * Ncomp + 1

        T = eltype(eltype(Δblk))

        Δblk[i, j] = SVector{Ncomp,T}(
            ntuple(k -> T(ΔWv[base + k - 1]), Val(Ncomp))
        )
    end

    return nothing
end

function ChainRulesCore.rrule(
    ::typeof(assemble_Wv),
    val_tuple::CuValTuple,
)
    y = assemble_Wv(val_tuple)

    n_atoms, L_offsets, atom_stride, total_len = _wv_layout_meta_gpu(val_tuple)

    function pullback_assemble_Wv_gpu(Δy)
        ΔWv = unthunk(Δy)

        Δval_tuple = ntuple(length(val_tuple)) do Lidx
            similar(val_tuple[Lidx])
        end

        threads = 256

        for Lidx in 1:length(val_tuple)
            L = Lidx - 1
            n_shells = size(val_tuple[Lidx], 2)
            Loff = L_offsets[Lidx]

            n_shells == 0 && continue

            N = n_atoms * n_shells
            blocks = cld(N, threads)

            if L == 0
                @cuda threads=threads blocks=blocks _assemble_Wv_gpu_pb_L0_kernel!(
                    Δval_tuple[Lidx],
                    ΔWv,
                    n_shells,
                    atom_stride,
                    Loff,
                )
            elseif L == 1
                @cuda threads=threads blocks=blocks _assemble_Wv_gpu_pb_L1_kernel!(
                    Δval_tuple[Lidx],
                    ΔWv,
                    n_shells,
                    atom_stride,
                    Loff,
                )
            else
                @cuda threads=threads blocks=blocks _assemble_Wv_gpu_pb_L_kernel!(
                    Δval_tuple[Lidx],
                    ΔWv,
                    n_shells,
                    atom_stride,
                    Loff,
                    Val(L),
                )
            end
        end

        return NoTangent(), Δval_tuple
    end

    return y, pullback_assemble_Wv_gpu
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
end
