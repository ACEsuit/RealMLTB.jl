# ace_wpsi.jl
# ACE models for the environment-dependent orbital contraction coefficients W^ψ.
#
# Two cases are implemented, following the thesis (Chapter 3, Section 2):
#
# ══════════════════════════════════════════════════════════════════════════════
# CASE A  –  co-rotating compressed basis  (D^ψ(Q) = D(Q))
# ══════════════════════════════════════════════════════════════════════════════
# The contracted basis functions rotate with the system, so within each
# angular-momentum block l the contraction matrix is block-diagonal:
#
#   W^ψ,(l)_I  =  I_{2l+1} ⊗ U^(l)_I        (kronecker structure)
#
# where U^(l)_I  is a  nψ[l+1] × nφ[l+1]  matrix of *scalar* (L=0)
# environment-dependent coefficients.  Only rotationally invariant (L=0)
# ACE features are needed.
#
# Equivariance check:
#   W^ψ_I(QN_I) D(Q) = D^ψ(Q)^T W^ψ_I(N_I)
#   LHS: (I ⊗ U(QN_I)) D(Q) = D(Q)^T (I ⊗ U(N_I)) D(Q)   [U invariant]
#   RHS: D(Q)^T (I ⊗ U(N_I))  ✓ (same, because D(Q)^T (I⊗U) D(Q) = I⊗U
#                                  when U is scalar and D is block diagonal)
#
# ══════════════════════════════════════════════════════════════════════════════
# CASE B  –  isotropic (scalar) target orbitals  (D^ψ(Q) = I)
# ══════════════════════════════════════════════════════════════════════════════
# The contracted basis functions are rotationally invariant scalars.  To make
# this work the contraction coefficients themselves must transform as rank-l
# tensors to absorb the angular dependence of φ_{lm}:
#
#   W^ψ_{I,a(lm)}(QN_I) = Σ_{m'} D^(l)_{mm'}(Q) W^ψ_{I,a(lm')}(N_I)
#
# so we need equivariant (L=l) ACE features for each l-block, and the model
# outputs a tuple of equivariant tensors (same structure as the Wv model).
#
# After assembly the NORB-row matrix Wpsi satisfies
#   W1 = W0 * D_φ   (right action of the AO rotation matrix)

using Lux, Random
import LuxCore
using StaticArrays
using EquivariantTensors
using ChainRulesCore
using LinearAlgebra

# ============================================================================
# ── CASE A ──────────────────────────────────────────────────────────────────
# ============================================================================

# ---------------------------------------------------------------------------
# build_ace_caseA_Wpsi_model
# ---------------------------------------------------------------------------
"""
    build_ace_caseA_Wpsi_model(n_channels_φ, nψ, rcut; Dtot, maxl, ORD, rng)

Case A: co-rotating compressed basis.

Outputs a (NATOM × nparam) scalar matrix where
    nparam = Σ_l  nψ[l+1] * nφ[l+1].

Arguments
---------
- n_channels_φ : tuple/vector of length (lmax+1), AO basis shells per l-block
- nψ            : tuple/vector, number of target compressed orbitals per l-block
- rcut          : cutoff (Unitful)
"""
function build_ace_caseA_Wpsi_model(n_channels_φ, nψ, rcut;
    Dtot::Int        = 9,
    maxl::Int        = 5,
    ORD::Int         = 3,
    rng::AbstractRNG = MersenneTwister(1234),
)
    lmax = max(length(n_channels_φ), length(nψ)) - 1
    @assert lmax ≥ 0

    nparam = sum(getidx(nψ, l+1) * getidx(n_channels_φ, l+1) for l in 0:lmax)
    @assert nparam > 0 "Case A: nparam = 0; check n_channels_φ and nψ."

    embed, Rnl_spec, Ylm_spec = ace_make_embed(rcut; Dtot = Dtot, maxl = maxl)
    mb_spec = ace_make_mbspec(; Dtot = Dtot, maxl = maxl, ORD = ORD)

    # Only invariant (L=0) features needed.
    𝔹basis = EquivariantTensors.sparse_equivariant_tensor(;
        L        = 0,
        mb_spec  = mb_spec,
        Rnl_spec = Rnl_spec,
        Ylm_spec = Ylm_spec,
        basis    = real,
    )

    ace   = EquivariantTensors.SparseACElayer(𝔹basis, (nparam,))
    model = Lux.Chain(; embed = embed, ace = ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

# ---------------------------------------------------------------------------
# _unwrap_caseA_scalar  –  accept Matrix or 1-tuple-of-Matrix
# ---------------------------------------------------------------------------
@inline function _unwrap_caseA_scalar(val)
    if val isa AbstractMatrix
        return val
    elseif val isa Tuple && length(val) == 1 && first(val) isa AbstractMatrix
        return first(val)
    else
        throw(ArgumentError(
            "assemble_Wpsi_caseA: expected NATOM×nparam matrix or a " *
            "1-element Tuple containing one; got $(typeof(val))"
        ))
    end
end

# ---------------------------------------------------------------------------
# assemble_Wpsi_caseA
#
# Constructs the full (NATOM*NORB) × (NATOM*NAO_atom) contraction matrix
# with the Kronecker structure  W^(l) = I_{2l+1} ⊗ U^(l).
#
# Row layout (within one atom block, row index):
#   l=0: nψ[1] rows of width 1
#   l=1: nψ[2] × 3 rows  (each U-row is tripled)
#   l=2: nψ[3] × 5 rows  ...
#
# Column layout mirrors the AO basis ordering:
#   l=0: nφ[1] × 1 cols
#   l=1: nφ[2] × 3 cols
#   ...
# ---------------------------------------------------------------------------
function assemble_Wpsi_caseA(val_scalar, n_channels_φ, nψ)
    val_mat = _unwrap_caseA_scalar(val_scalar)
    T       = eltype(val_mat)
    NATOM   = size(val_mat, 1)
    lmax    = max(length(n_channels_φ), length(nψ)) - 1

    # sizes
    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)
    NORB     = sum(getidx(nψ,           L+1) * (2L+1) for L in 0:lmax)

    Wpsi = zeros(T, NATOM * NORB, NATOM * NAO_atom)

    # column offset of each l-block within one atom's AO columns
    col_off = Vector{Int}(undef, lmax + 1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    @inbounds for I in 1:NATOM
        row0 = (I - 1) * NORB
        col0 = (I - 1) * NAO_atom

        row_cursor = 1   # local row within this atom's NORB rows
        pcur       = 1   # column cursor into val_mat[I, :]

        for L in 0:lmax
            nφ  = getidx(n_channels_φ, L+1)
            nψL = getidx(nψ,           L+1)
            (nφ == 0 || nψL == 0) && continue

            nblock = nψL * nφ
            @assert pcur + nblock - 1 <= size(val_mat, 2) "Case A: val_scalar has too few columns for L=$L."

            Umat = reshape(@view(val_mat[I, pcur:(pcur + nblock - 1)]), nψL, nφ)
            pcur += nblock

            mlen = 2L + 1

            # fill  I_{mlen} ⊗ Umat  block into Wpsi
            for μ in 1:nψL
                for s in 1:nφ
                    coeff = Umat[μ, s]
                    for mi in 1:mlen
                        r = row0 + row_cursor + (μ - 1) * mlen + (mi - 1)
                        c = col0 + col_off[L+1] + (s - 1) * mlen + (mi - 1)
                        Wpsi[r, c] = coeff
                    end
                end
            end

            row_cursor += nψL * mlen
        end
    end

    return Wpsi
end

# ---------------------------------------------------------------------------
# ChainRules pullback for assemble_Wpsi_caseA
# ---------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(assemble_Wpsi_caseA),
                              val_scalar, n_channels_φ, nψ)
    val_mat = _unwrap_caseA_scalar(val_scalar)
    Y       = assemble_Wpsi_caseA(val_mat, n_channels_φ, nψ)

    NATOM   = size(val_mat, 1)
    lmax    = max(length(n_channels_φ), length(nψ)) - 1
    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)
    NORB     = sum(getidx(nψ,           L+1) * (2L+1) for L in 0:lmax)

    col_off = Vector{Int}(undef, lmax + 1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    function pullback_caseA(Ȳ)
        Ȳ    = ChainRulesCore.unthunk(Ȳ)
        Δval = zeros(eltype(val_mat), size(val_mat))

        @inbounds for I in 1:NATOM
            row0 = (I - 1) * NORB
            col0 = (I - 1) * NAO_atom
            row_cursor = 1
            pcur       = 1

            for L in 0:lmax
                nφ  = getidx(n_channels_φ, L+1)
                nψL = getidx(nψ,           L+1)
                (nφ == 0 || nψL == 0) && continue

                mlen = 2L + 1

                for μ in 1:nψL
                    for s in 1:nφ
                        acc = zero(eltype(val_mat))
                        for mi in 1:mlen
                            r    = row0 + row_cursor + (μ - 1) * mlen + (mi - 1)
                            c    = col0 + col_off[L+1] + (s - 1) * mlen + (mi - 1)
                            acc += Ȳ[r, c]
                        end
                        Δval[I, pcur + (μ - 1) * nφ + (s - 1)] = acc
                    end
                end

                pcur       += nψL * nφ
                row_cursor += nψL * mlen
            end
        end

        # propagate gradient back through the possible Tuple-wrapping
        Δval_out = if val_scalar isa AbstractMatrix
            Δval
        elseif val_scalar isa Tuple
            (Δval,)
        else
            ChainRulesCore.NoTangent()
        end

        return (ChainRulesCore.NoTangent(), Δval_out,
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end

    return Y, pullback_caseA
end

# ============================================================================
# ── CASE B ──────────────────────────────────────────────────────────────────
# ============================================================================

# ---------------------------------------------------------------------------
# build_ace_caseB_Wpsi_model
# ---------------------------------------------------------------------------
"""
    build_ace_caseB_Wpsi_model(n_channels_φ, NORB, rcut; Dtot, maxl, ORD, rng)

Case B: isotropic (rotationally invariant) compressed orbitals.

The contraction coefficients W^ψ_{I,a(lm)} must be rank-l equivariant to
cancel the angular transformation of φ_{lm}.  We therefore output equivariant
features at each L = 0, 1, ..., lmax, with NORB * nφ[L+1] channels.

Assembled matrix Wpsi satisfies:
    Wpsi(QN_I) = Wpsi(N_I) * D_φ(Q)     (right action)

Arguments
---------
- n_channels_φ : tuple/vector of length (lmax+1), AO shells per l-block
- NORB          : number of target scalar orbitals per atom
- rcut          : cutoff (Unitful)
"""
function build_ace_caseB_Wpsi_model(n_channels_φ, NORB::Int, rcut;
    Dtot::Int        = 9,
    maxl::Int        = 5,
    ORD::Int         = 3,
    rng::AbstractRNG = MersenneTwister(1234),
)
    lmax = length(n_channels_φ) - 1
    @assert lmax ≥ 0

    # Each L-block needs NORB * nφ[L+1] independent equivariant channels.
    # The model outputs one (2L+1)-dimensional SVector per (atom, channel).
    n_channels_out = Tuple(NORB * getidx(n_channels_φ, L+1) for L in 0:lmax)

    embed, Rnl_spec, Ylm_spec = ace_make_embed(rcut; Dtot = Dtot, maxl = maxl)
    mb_spec = ace_make_mbspec(; Dtot = Dtot, maxl = maxl, ORD = ORD)

    target_LL = Tuple(0:lmax)
    𝔹basis = EquivariantTensors.sparse_equivariant_tensors(;
        LL       = target_LL,
        mb_spec  = mb_spec,
        Rnl_spec = Rnl_spec,
        Ylm_spec = Ylm_spec,
        basis    = real,
    )

    ace   = EquivariantTensors.SparseACElayer(𝔹basis, n_channels_out)
    model = Lux.Chain(; embed = embed, ace = ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

# ---------------------------------------------------------------------------
# assemble_Wpsi_caseB
#
# Converts the equivariant model output (same format as Wv's val_tuple) into
# the (NATOM*NORB) × (NATOM*NAO_atom) contraction matrix.
#
# For each atom I, target orbital a, l-block l, and magnetic quantum number m:
#   Wpsi[I*a, I*(l,m)] = W^ψ_{I,a(lm)}
#
# The row index runs over (I, a) and the column over (I, l, m) with the same
# l-ordering and pyscf permutation as in assemble_Wv.
#
# The equivariance W^ψ_{I,a(lm)}(QN_I) = Σ_{m'} D^(l)_{mm'} W^ψ_{I,a(lm')}
# ensures  Wpsi(QN_I) = Wpsi(N_I) * D_φ(Q).
# ---------------------------------------------------------------------------
function assemble_Wpsi_caseB(val_tuple, n_channels_φ, NORB::Int)
    n_atoms = size(val_tuple[1], 1)
    lmax    = length(val_tuple) - 1

    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)
    T        = eltype(first(val_tuple[1]))

    Wpsi = zeros(T, n_atoms * NORB, n_atoms * NAO_atom)

    # column offset of each l-block within one atom's AO columns
    col_off = Vector{Int}(undef, lmax + 1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    perm_L1 = SVector{3,Int}(3, 1, 2)   # pyscf L=1 permutation

    @inbounds for I in 1:n_atoms
        row0 = (I - 1) * NORB
        col0 = (I - 1) * NAO_atom

        for (Lidx, blk) in enumerate(val_tuple)
            L       = Lidx - 1
            mlen    = 2L + 1
            nφ      = getidx(n_channels_φ, L+1)
            nφ == 0 && continue

            # The model outputs NORB * nφ channels for this L.
            # Channel index encodes (a, s) with a = target orbital, s = AO shell.
            n_chan = size(blk, 2)   # should equal NORB * nφ
            @assert n_chan == NORB * nφ "Case B: channel count mismatch at L=$L: got $n_chan, expected $(NORB*nφ)"

            perm = (L == 1) ? perm_L1 : nothing

            for chan in 1:n_chan
                # decode (a, s) from linear channel index (a-major)
                a = div(chan - 1, nφ) + 1   # target orbital index (1-based)
                s = mod(chan - 1, nφ) + 1   # AO shell index within this l-block

                v = blk[I, chan]   # SVector{mlen} or scalar

                for m in 1:mlen
                    # m-component with pyscf permutation applied
                    m_src = (perm === nothing) ? m : perm[m]
                    val   = (L == 0) ? v : v[m_src]

                    r = row0 + (a - 1) + 1           # NORB ordering: row per target orbital
                    c = col0 + col_off[L+1] + (s - 1) * mlen + (m - 1)
                    Wpsi[r, c] = val
                end
            end
        end
    end

    return Wpsi
end

# ---------------------------------------------------------------------------
# ChainRules pullback for assemble_Wpsi_caseB
# ---------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(assemble_Wpsi_caseB),
                              val_tuple, n_channels_φ, NORB::Int)
    Y = assemble_Wpsi_caseB(val_tuple, n_channels_φ, NORB)

    n_atoms  = size(val_tuple[1], 1)
    lmax     = length(val_tuple) - 1
    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)

    col_off = Vector{Int}(undef, lmax + 1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    perm_L1    = SVector{3,Int}(3, 1, 2)
    invperm_L1 = SVector{3,Int}(2, 3, 1)

    function pullback_caseB(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)

        Δval_tuple = ntuple(length(val_tuple)) do Lidx
            blk = similar(val_tuple[Lidx])
            fill!(blk, zero(eltype(blk)))
            blk
        end

        @inbounds for I in 1:n_atoms
            row0 = (I - 1) * NORB
            col0 = (I - 1) * NAO_atom

            for Lidx in 1:length(val_tuple)
                L    = Lidx - 1
                mlen = 2L + 1
                nφ   = getidx(n_channels_φ, L+1)
                nφ == 0 && continue

                n_chan  = size(val_tuple[Lidx], 2)
                invperm = (L == 1) ? invperm_L1 : nothing
                Ts      = eltype(val_tuple[Lidx][I, 1])

                for chan in 1:n_chan
                    a = div(chan - 1, nφ) + 1
                    s = mod(chan - 1, nφ) + 1

                    r = row0 + (a - 1) + 1

                    if L == 0
                        c = col0 + col_off[L+1] + (s - 1) * 1
                        Δval_tuple[Lidx][I, chan] += Ȳ[r, c]
                    else
                        # accumulate gradient with inverse permutation
                        tmp_g = zeros(Ts, mlen)
                        for m in 1:mlen
                            c = col0 + col_off[L+1] + (s - 1) * mlen + (m - 1)
                            m_src = (invperm === nothing) ? m : invperm[m]
                            tmp_g[m_src] += Ȳ[r, c]
                        end
                        Δval_tuple[Lidx][I, chan] =
                            Δval_tuple[Lidx][I, chan] + SVector{mlen,Ts}(tmp_g...)
                    end
                end
            end
        end

        return (ChainRulesCore.NoTangent(), Δval_tuple,
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end

    return Y, pullback_caseB
end