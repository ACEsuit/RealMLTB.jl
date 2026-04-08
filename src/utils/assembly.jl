# assembly.jl  (AsmTrip2Dense layer + _assemble_dense + pullback)
#
# Converts (G, vH, vS) produced by _3center_triplet into k-dependent
# dense Bloch Hamiltonian and overlap matrices H(k), S(k).
#
# Rcell field name in edge_data: R_cell  (from nlist2graph)

using Base.Threads: @threads
using ChainRulesCore: unthunk, NoTangent
using StaticArrays
using LinearAlgebra: dot

struct AsmTrip2Dense <: Lux.AbstractLuxLayer end

LuxCore.initialparameters(::AbstractRNG, ::AsmTrip2Dense) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::AsmTrip2Dense)     = NamedTuple()

function (l::AsmTrip2Dense)(x, ps, st)
    G, vH, vS, Wb = x
    klist = G.graph_data.klist
    Hout, Sout = _assemble_dense(G, vH, vS, klist)
    return (Hout, Sout, Wb), st
end

# ---------------------------------------------------------------------------
# _assemble_dense
#
# Assembles H(k) = Σ_R  H_R * exp(i k·R)  for each k in klist.
#
# Layout of vH/vS (from _3center_triplet):
#   [1:nat]     = onsite blocks (i,i), no phase
#   [nat+1:end] = offsite blocks in edge order, phase = exp(i k·R_cell)
# ---------------------------------------------------------------------------
function _assemble_dense(G,
                         vH::AbstractVector{<:SMatrix{NORB,NORB,TH}},
                         vS::AbstractVector{<:SMatrix{NORB,NORB,TH}},
                         klist) where {NORB, TH}
    ii    = _get_ii_jj(G)[1]
    jj    = _get_ii_jj(G)[2]
    nat   = length(G.first) - 1
    nk    = length(klist)
    TC    = Complex{TH}

    # R_cell stored per edge in nlist2graph
    Rcell = [ed.R_cell for ed in G.edge_data]

    dim = NORB * nat

    Hout = [zeros(TC, dim, dim) for _ in 1:nk]
    Sout = [zeros(TC, dim, dim) for _ in 1:nk]

    @threads for ik in 1:nk
        k = klist[ik]
        H = Hout[ik]
        S = Sout[ik]

        # onsite (real, no phase)
        for i in 1:nat
            Ii = _indexblock(i, NORB)
            @inbounds for a in 1:NORB, b in 1:NORB
                H[Ii[a], Ii[b]] += vH[i][a, b]
                S[Ii[a], Ii[b]] += vS[i][a, b]
            end
        end

        # offsite
        for e in eachindex(ii)
            i     = ii[e]
            j     = jj[e]
            Ii    = _indexblock(i, NORB)
            Ij    = _indexblock(j, NORB)
            phase = exp(im * dot(k, Rcell[e]))

            @inbounds for a in 1:NORB, b in 1:NORB
                H[Ii[a], Ij[b]] += vH[nat + e][a, b] * phase
                S[Ii[a], Ij[b]] += vS[nat + e][a, b] * phase
            end
        end

        # Enforce Hermiticity (accumulation should already give Hermitian,
        # but floating-point rounding makes this necessary)
        H .= (H .+ H') ./ 2
        S .= (S .+ S') ./ 2
    end

    return Hout, Sout
end

@inline _indexblock(n::Integer, NORB) = (n-1)*NORB .+ (1:NORB)

# ---------------------------------------------------------------------------
# ChainRules pullback for _assemble_dense
# ---------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(_assemble_dense), G,
                              vH::AbstractVector{<:SMatrix{NORB,NORB,TH}},
                              vS::AbstractVector{<:SMatrix{NORB,NORB,TH}},
                              klist) where {NORB, TH}
    Hout, Sout = _assemble_dense(G, vH, vS, klist)

    function pullback(ΔΩ)
        ΔHout = unthunk(ΔΩ[1])
        ΔSout = unthunk(ΔΩ[2])

        ii    = _get_ii_jj(G)[1]
        jj    = _get_ii_jj(G)[2]
        nat   = length(G.first) - 1
        nk    = length(klist)
        Rcell = [ed.R_cell for ed in G.edge_data]
        L     = length(vH)
        TC    = Complex{TH}

        # Symmetrise incoming gradients (∂loss/∂H is Hermitian because H is)
        ΔHsym = [(ΔHout[ik] .+ ΔHout[ik]') ./ 2 for ik in 1:nk]
        ΔSsym = [(ΔSout[ik] .+ ΔSout[ik]') ./ 2 for ik in 1:nk]

        vH̄_mat = [zeros(TC, NORB, NORB) for _ in 1:L]
        vS̄_mat = [zeros(TC, NORB, NORB) for _ in 1:L]

        # onsite: accumulate over all k (no phase)
        @threads for i in 1:nat
            Ii = _indexblock(i, NORB)
            for ik in 1:nk
                @inbounds for a in 1:NORB, b in 1:NORB
                    vH̄_mat[i][a,b] += ΔHsym[ik][Ii[a], Ii[b]]
                    vS̄_mat[i][a,b] += ΔSsym[ik][Ii[a], Ii[b]]
                end
            end
        end

        # offsite: conj(phase) factor
        @threads for e in eachindex(ii)
            i   = ii[e]
            j   = jj[e]
            Ii  = _indexblock(i, NORB)
            Ij  = _indexblock(j, NORB)
            idx = nat + e

            for ik in 1:nk
                k      = klist[ik]
                cphase = conj(exp(im * dot(k, Rcell[e])))
                @inbounds for a in 1:NORB, b in 1:NORB
                    vH̄_mat[idx][a,b] += cphase * ΔHsym[ik][Ii[a], Ij[b]]
                    vS̄_mat[idx][a,b] += cphase * ΔSsym[ik][Ii[a], Ij[b]]
                end
            end
        end

        vH̄ = [SMatrix{NORB,NORB,TC}(vH̄_mat[i]) for i in 1:L]
        vS̄ = [SMatrix{NORB,NORB,TC}(vS̄_mat[i]) for i in 1:L]

        return NoTangent(), NoTangent(), vH̄, vS̄, NoTangent()
    end

    return (Hout, Sout), pullback
end