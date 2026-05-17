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


function _assembly_block_metadata(G)
    if @isdefined(_is_cut_graph) && _is_cut_graph(G)
        return _cut_block_metadata(G)
    end
    ii, jj = _get_ii_jj(G)
    return [(kind = :edge, i = ii[e], j = jj[e], k = 0, R_cell = G.edge_data[e].R_cell)
            for e in eachindex(ii)]
end

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
    block_meta = _assembly_block_metadata(G)
    nat   = length(G.first) - 1
    nk    = length(klist)
    TC    = Complex{TH}

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

        # offsite/additive cut blocks
        for e in eachindex(block_meta)
            meta  = block_meta[e]
            i     = meta.i
            j     = meta.j
            Ii    = _indexblock(i, NORB)
            Ij    = _indexblock(j, NORB)
            phase = exp(im * dot(k, meta.R_cell))

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

        block_meta = _assembly_block_metadata(G)
        nat   = length(G.first) - 1
        nk    = length(klist)
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

        # offsite/additive cut blocks: conj(phase) factor
        @threads for e in eachindex(block_meta)
            meta = block_meta[e]
            i   = meta.i
            j   = meta.j
            Ii  = _indexblock(i, NORB)
            Ij  = _indexblock(j, NORB)
            idx = nat + e

            for ik in 1:nk
                k      = klist[ik]
                cphase = conj(exp(im * dot(k, meta.R_cell)))
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


using CUDA
using LinearAlgebra

# ---------------------------------------------------------------------------
# GPU assembly kernels
#
# Input:
#   vH, vS :: CuArray{T,3}, size = (NORB, NORB, nat + nedges)
#
# Layout:
#   vH[:,:,1:nat]       = onsite blocks
#   vH[:,:,nat+1:end]   = offsite edge blocks
#
# Output:
#   Hout, Sout :: CuArray{Complex{T},3}, size = (dim, dim, nk)
# ---------------------------------------------------------------------------

function _assemble_onsite_kernel!(
    Hout,
    Sout,
    vH,
    vS,
    nat::Int,
    NORB::Int,
    nk::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nat * NORB * NORB * nk

    if q <= N
        t = q - 1

        ik = (t % nk) + 1
        t ÷= nk

        b = (t % NORB) + 1
        t ÷= NORB

        a = (t % NORB) + 1
        i = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (i - 1) * NORB + b

        Hout[row, col, ik] += complex(vH[a, b, i])
        Sout[row, col, ik] += complex(vS[a, b, i])
    end

    return nothing
end

function _assemble_offsite_edge_kernel!(
    Hout,
    Sout,
    vH,
    vS,
    klist,
    Rcell,
    e::Int,
    i::Int,
    j::Int,
    nat::Int,
    NORB::Int,
    nk::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB * nk

    if q <= N
        t = q - 1

        ik = (t % nk) + 1
        t ÷= nk

        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (j - 1) * NORB + b

        idx = nat + e

        θ =
            klist[1, ik] * Rcell[1, e] +
            klist[2, ik] * Rcell[2, e] +
            klist[3, ik] * Rcell[3, e]

        phase = complex(cos(θ), sin(θ))

        Hout[row, col, ik] += complex(vH[a, b, idx]) * phase
        Sout[row, col, ik] += complex(vS[a, b, idx]) * phase
    end

    return nothing
end

function _hermitize_dense_kernel!(Hout, Sout, dim::Int, nk::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = dim * dim * nk

    if q <= N
        t = q - 1

        ik = (t % nk) + 1
        t ÷= nk

        col = (t % dim) + 1
        row = (t ÷ dim) + 1

        if row <= col
            h = (Hout[row, col, ik] + conj(Hout[col, row, ik])) / 2
            s = (Sout[row, col, ik] + conj(Sout[col, row, ik])) / 2

            Hout[row, col, ik] = h
            Sout[row, col, ik] = s

            if row != col
                Hout[col, row, ik] = conj(h)
                Sout[col, row, ik] = conj(s)
            end
        end
    end

    return nothing
end

function _assemble_dense(
    G,
    vH::CUDA.CuArray{TH,3},
    vS::CUDA.CuArray{TH,3},
    klist_unused,
) where {TH <: Real}

    cache = G.graph_data.threecenter_cache

    ii = cache.ii
    jj = cache.jj

    Rcell = cache.Rcell      # 3 × nedges, CuArray
    klist = cache.klist      # 3 × nk, CuArray

    nat = cache.nat
    NORB = cache.NORB
    nedges = length(ii)
    nk = size(klist, 2)

    dim = nat * NORB
    TC = Complex{TH}

    Hout = CUDA.zeros(TC, dim, dim, nk)
    Sout = CUDA.zeros(TC, dim, dim, nk)

    threads = 256

    # onsite blocks
    N_on = nat * NORB * NORB * nk
    blocks_on = cld(N_on, threads)

    @cuda threads=threads blocks=blocks_on _assemble_onsite_kernel!(
        Hout,
        Sout,
        vH,
        vS,
        nat,
        NORB,
        nk,
    )

    # offsite edge blocks
    N_edge = NORB * NORB * nk
    blocks_edge = cld(N_edge, threads)

    for e in 1:nedges
        i = ii[e]
        j = jj[e]

        @cuda threads=threads blocks=blocks_edge _assemble_offsite_edge_kernel!(
            Hout,
            Sout,
            vH,
            vS,
            klist,
            Rcell,
            e,
            i,
            j,
            nat,
            NORB,
            nk,
        )
    end

    # Hermitize
    N_herm = dim * dim * nk
    blocks_herm = cld(N_herm, threads)

    @cuda threads=threads blocks=blocks_herm _hermitize_dense_kernel!(
        Hout,
        Sout,
        dim,
        nk,
    )

    return Hout, Sout
end

using CUDA
using ChainRulesCore
using ChainRulesCore: NoTangent, unthunk, ZeroTangent, NoTangent

@inline function _zero_if_not_array(x, like)
    x isa ZeroTangent && return CUDA.zeros(eltype(like), size(like))
    x isa NoTangent && return CUDA.zeros(eltype(like), size(like))
    return x
end

function _assemble_dense_pullback_onsite_kernel!(
    vHbar,
    vSbar,
    ΔHsym,
    ΔSsym,
    nat::Int,
    NORB::Int,
    nk::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nat * NORB * NORB

    if q <= N
        t = q - 1

        b = (t % NORB) + 1
        t ÷= NORB

        a = (t % NORB) + 1
        i = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (i - 1) * NORB + b

        accH = zero(eltype(vHbar))
        accS = zero(eltype(vSbar))

        for ik in 1:nk
            accH += ΔHsym[row, col, ik]
            accS += ΔSsym[row, col, ik]
        end

        vHbar[a, b, i] = accH
        vSbar[a, b, i] = accS
    end

    return nothing
end

function _assemble_dense_pullback_offsite_kernel!(
    vHbar,
    vSbar,
    ΔHsym,
    ΔSsym,
    klist,
    Rcell,
    e::Int,
    i::Int,
    j::Int,
    nat::Int,
    NORB::Int,
    nk::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB

    if q <= N
        t = q - 1

        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (j - 1) * NORB + b

        idx = nat + e

        accH = zero(eltype(vHbar))
        accS = zero(eltype(vSbar))

        for ik in 1:nk
            θ =
                klist[1, ik] * Rcell[1, e] +
                klist[2, ik] * Rcell[2, e] +
                klist[3, ik] * Rcell[3, e]

            cphase = complex(cos(θ), -sin(θ))

            accH += cphase * ΔHsym[row, col, ik]
            accS += cphase * ΔSsym[row, col, ik]
        end

        vHbar[a, b, idx] = accH
        vSbar[a, b, idx] = accS
    end

    return nothing
end

function ChainRulesCore.rrule(
    ::typeof(_assemble_dense),
    G,
    vH::CUDA.CuArray{TH,3},
    vS::CUDA.CuArray{TH,3},
    klist_unused,
) where {TH <: Real}

    Hout, Sout = _assemble_dense(G, vH, vS, klist_unused)

    cache = G.graph_data.threecenter_cache

    ii = cache.ii
    jj = cache.jj
    Rcell = cache.Rcell
    klist = cache.klist

    nat = cache.nat
    NORB = cache.NORB
    nedges = length(ii)
    nk = size(klist, 2)

    function pullback_assemble_dense_gpu(ΔΩ)
        ΔHout_raw = unthunk(ΔΩ[1])
        ΔSout_raw = unthunk(ΔΩ[2])

        ΔHout = _zero_if_not_array(ΔHout_raw, Hout)
        ΔSout = _zero_if_not_array(ΔSout_raw, Sout)

        # adjoint of hermitize: X -> (X + X')/2
        ΔHsym = (ΔHout .+ permutedims(conj.(ΔHout), (2, 1, 3))) ./ 2
        ΔSsym = (ΔSout .+ permutedims(conj.(ΔSout), (2, 1, 3))) ./ 2

        TC = eltype(Hout)

        vHbar = CUDA.zeros(TC, size(vH))
        vSbar = CUDA.zeros(TC, size(vS))

        threads = 256

        # onsite
        N_on = nat * NORB * NORB
        blocks_on = cld(N_on, threads)

        @cuda threads=threads blocks=blocks_on _assemble_dense_pullback_onsite_kernel!(
            vHbar,
            vSbar,
            ΔHsym,
            ΔSsym,
            nat,
            NORB,
            nk,
        )

        # offsite
        N_edge = NORB * NORB
        blocks_edge = cld(N_edge, threads)

        for e in 1:nedges
            i = ii[e]
            j = jj[e]

            @cuda threads=threads blocks=blocks_edge _assemble_dense_pullback_offsite_kernel!(
                vHbar,
                vSbar,
                ΔHsym,
                ΔSsym,
                klist,
                Rcell,
                e,
                i,
                j,
                nat,
                NORB,
                nk,
            )
        end

        return (
            NoTangent(),  # typeof(_assemble_dense)
            NoTangent(),  # G
            vHbar,
            vSbar,
            NoTangent(),  # klist
        )
    end

    return (Hout, Sout), pullback_assemble_dense_gpu
end