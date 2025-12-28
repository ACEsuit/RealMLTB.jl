using Lux, Random, LinearAlgebra
import LuxCore
using EquivariantTensors
using StaticArrays
using ChainRulesCore

# ------------------------------------------------------------
# Case A: co-rotating ψ
# Output is invariant scalars (L=0) packed as U^(l) blocks.
# ------------------------------------------------------------

"""
    build_ace_caseA_Wpsi_model(n_channels_φ, nψ, rcut; ...)

Case A: Wpsi is constructed from invariant scalars U^(l)(R_I).
Model outputs NATOM×nparam scalars, where nparam = Σ_l nψ[l]*nφ[l].
Returns (model, ps, st).
"""
function build_ace_caseA_Wpsi_model(n_channels_φ, nψ, rcut;
    Dtot::Int = 9,
    maxl::Int = 5,
    ORD::Int  = 3,
    rng::AbstractRNG = MersenneTwister(1234),
)
    lmax = max(length(n_channels_φ), length(nψ)) - 1
    @assert lmax ≥ 0

    nparam = 0
    for l in 0:lmax
        nparam += getidx(nψ, l+1) * getidx(n_channels_φ, l+1)
    end
    @assert nparam > 0 "Case A: nparam=0; check n_channels_φ and nψ."

    embed, Rnl_spec, Ylm_spec = ace_make_embed(rcut; Dtot=Dtot, maxl=maxl)
    mb_spec = ace_make_mbspec(; Dtot=Dtot, maxl=maxl, ORD=ORD)

    𝔹basis = EquivariantTensors.sparse_equivariant_tensor(;
        L = 0,
        mb_spec = mb_spec,
        Rnl_spec = Rnl_spec,
        Ylm_spec = Ylm_spec,
        basis = real
    )

    ace = EquivariantTensors.SparseACElayer(𝔹basis, (nparam,))
    model = Lux.Chain(; embed=embed, ace=ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

"""
    assemble_Wpsi_caseA(val_scalar, n_channels_φ, nψ)

Construct Wpsi as block-diagonal-per-atom, and within each l block:
Wpsi^(l) = kron(I_{2l+1}, U^(l)).

- val_scalar: NATOM×nparam scalar matrix (model output)
- n_channels_φ: AO channels per l
- nψ: target orbitals per l (co-rotating)
"""
function assemble_Wpsi_caseA(val_scalar, n_channels_φ, nψ)
    T = eltype(val_scalar)

    NATOM = size(val_scalar, 1)
    lmax  = max(length(n_channels_φ), length(nψ)) - 1

    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)
    NORB     = sum(getidx(nψ,          L+1) * (2L+1) for L in 0:lmax)

    Wpsi = zeros(T, NATOM*NORB, NATOM*NAO_atom)

    # AO col offsets inside one atom
    col_off = Vector{Int}(undef, lmax+1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    @inbounds for I in 1:NATOM
        row0 = (I-1)*NORB
        col0 = (I-1)*NAO_atom

        row_cursor = 1
        pcur = 1

        for L in 0:lmax
            nφ  = getidx(n_channels_φ, L+1)
            nψL = getidx(nψ,          L+1)
            if nφ == 0 || nψL == 0
                continue
            end

            Umat = reshape(@view(val_scalar[I, pcur:(pcur + nψL*nφ - 1)]), nψL, nφ)
            pcur += nψL*nφ

            mlen = 2L + 1

            # kron(I_mlen, Umat) placement
            for μ in 1:nψL
                for s in 1:nφ
                    coeff = Umat[μ, s]
                    for mi in 1:mlen
                        r = row0 + row_cursor + (μ-1)*mlen + (mi-1)
                        c = col0 + col_off[L+1] + (s-1)*mlen + (mi-1)
                        Wpsi[r, c] = coeff
                    end
                end
            end

            row_cursor += nψL*mlen
        end
    end

    return Wpsi
end

function ChainRulesCore.rrule(::typeof(assemble_Wpsi_caseA),
                              val_scalar, n_channels_φ, nψ)
    Y = assemble_Wpsi_caseA(val_scalar, n_channels_φ, nψ)

    NATOM = size(val_scalar, 1)
    lmax  = max(length(n_channels_φ), length(nψ)) - 1

    NAO_atom = sum(getidx(n_channels_φ, L+1) * (2L+1) for L in 0:lmax)
    NORB     = sum(getidx(nψ,          L+1) * (2L+1) for L in 0:lmax)

    # AO col offsets inside one atom
    col_off = Vector{Int}(undef, lmax+1)
    tmp = 1
    for L in 0:lmax
        col_off[L+1] = tmp
        tmp += getidx(n_channels_φ, L+1) * (2L+1)
    end

    function pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        Δval = zeros(eltype(val_scalar), size(val_scalar))

        @inbounds for I in 1:NATOM
            row0 = (I-1)*NORB
            col0 = (I-1)*NAO_atom

            row_cursor = 1
            pcur = 1

            for L in 0:lmax
                nφ  = getidx(n_channels_φ, L+1)
                nψL = getidx(nψ,          L+1)
                if nφ == 0 || nψL == 0
                    continue
                end

                mlen = 2L + 1

                for μ in 1:nψL
                    for s in 1:nφ
                        acc = zero(eltype(val_scalar))
                        for mi in 1:mlen
                            r = row0 + row_cursor + (μ-1)*mlen + (mi-1)
                            c = col0 + col_off[L+1] + (s-1)*mlen + (mi-1)
                            acc += Ȳ[r, c]
                        end
                        Δval[I, pcur + (μ-1)*nφ + (s-1)] = acc
                    end
                end

                pcur       += nψL*nφ
                row_cursor += nψL*mlen
            end
        end

        return NoTangent(), Δval, NoTangent(), NoTangent()
    end

    return Y, pullback
end

# ------------------------------------------------------------
# Case B: Wpsi equivariant: W(QR) = W(R) * Dφ(Q)
# ------------------------------------------------------------

"""
    build_ace_caseB_Wpsi_model(n_channels_φ, NORB, rcut; ...)

Case B: outputs equivariant blocks for each L, but with channels multiplied by NORB
(i.e. for each target orbital μ, predict coefficients over AO channels).
Returns (model, ps, st).
"""
function build_ace_caseB_Wpsi_model(n_channels_φ, NORB, rcut;
    Dtot::Int = 9,
    maxl::Int = 5,
    ORD::Int  = 3,
    lmax::Int = length(n_channels_φ) - 1,
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

    n_channels_psi = map(c -> c * NORB, n_channels_φ)
    ace = EquivariantTensors.SparseACElayer(𝔹basis, n_channels_psi)
    model = Lux.Chain(; embed=embed, ace=ace)

    ps, st = LuxCore.setup(rng, model)
    ps = EquivariantTensors.float32(ps)
    st = EquivariantTensors.float32(st)
    return model, ps, st
end

"""
    assemble_Wpsi_caseB(val_tuple, n_channels_φ, NORB)

Assemble block-diagonal-per-atom Wpsi with shape:
(NATOM*NORB) × (NATOM*NAO_atom).

Layout convention matches your earlier working test_Hpsi_caseB.
"""
function assemble_Wpsi_caseB(val_tuple, n_channels_φ, NORB)
    T = eltype(val_tuple[1])

    NATOM = size(val_tuple[1], 1)
    lmax  = length(val_tuple) - 1

    NAO_atom = sum(n_channels_φ[L+1] * (2L+1) for L in 0:lmax)
    Wpsi = zeros(T, NATOM*NORB, NATOM*NAO_atom)

    @inbounds for I in 1:NATOM
        row0 = (I-1)*NORB
        col0 = (I-1)*NAO_atom

        for μ in 1:NORB
            cursor = col0 + 1

            for L_idx in 1:length(val_tuple)
                L  = L_idx - 1
                ns = n_channels_φ[L_idx]

                for s in 1:ns
                    ch = (μ-1)*ns + s
                    v  = val_tuple[L_idx][I, ch]   # scalar or SVector{2L+1}

                    if L == 0
                        Wpsi[row0 + μ, cursor] = v
                        cursor += 1
                    else
                        len = 2L + 1
                        @inbounds for k in 1:len
                            Wpsi[row0 + μ, cursor + k - 1] = v[k]
                        end
                        cursor += len
                    end
                end
            end
        end
    end

    return Wpsi
end

function ChainRulesCore.rrule(::typeof(assemble_Wpsi_caseB),
                              val_tuple, n_channels_φ, NORB)
    Y = assemble_Wpsi_caseB(val_tuple, n_channels_φ, NORB)

    NATOM = size(val_tuple[1], 1)
    lmax  = length(val_tuple) - 1
    NAO_atom = sum(n_channels_φ[L+1] * (2L+1) for L in 0:lmax)

    function pullback(Ȳ)
        Ȳ = unthunk(Ȳ)

        Δval_tuple = ntuple(length(val_tuple)) do L_idx
            blk = similar(val_tuple[L_idx])
            fill!(blk, zero(eltype(blk)))
            blk
        end

        @inbounds for I in 1:NATOM
            row0 = (I-1) * NORB
            col0 = (I-1) * NAO_atom

            for μ in 1:NORB
                cursor = col0 + 1

                for L_idx in 1:length(val_tuple)
                    L  = L_idx - 1
                    ns = n_channels_φ[L_idx]

                    for s in 1:ns
                        ch = (μ-1)*ns + s

                        if L == 0
                            Δval_tuple[L_idx][I, ch] = Ȳ[row0 + μ, cursor]
                            cursor += 1
                        else
                            len = 2L + 1
                            v0  = val_tuple[L_idx][I, ch]
                            Ts  = eltype(v0)

                            tmp = ntuple(k -> Ts(Ȳ[row0 + μ, cursor + k - 1]), Val(len))
                            Δval_tuple[L_idx][I, ch] = SVector{len, Ts}(tmp)
                            cursor += len
                        end
                    end
                end
            end
        end

        return NoTangent(), Δval_tuple, NoTangent(), NoTangent()
    end

    return Y, pullback
end
