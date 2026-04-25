using Base.Threads: @threads, nthreads, threadid
using ChainRulesCore: unthunk, NoTangent
using StaticArrays

# ---------------------------------------------------------------------------
# Safe accessors for ETGraph internals
# ---------------------------------------------------------------------------
@inline function _get_ii_jj(G)
    fn = fieldnames(typeof(G))
    if :ii ∈ fn && :jj ∈ fn
        return G.ii, G.jj
    elseif :i ∈ fn && :j ∈ fn
        return G.i, G.j
    else
        first = G.first
        nat   = length(first) - 1
        ii = Int[]
        jj_arr = Int[]
        for n in 1:nat
            for e in first[n]:(first[n+1]-1)
                push!(ii, n)
            end
        end
        error("Cannot determine ii/jj from ETGraph fieldnames: $(fn)")
    end
end

@inline _edge_T(ed, integral_table) = integral_table[ed.integral_idx].T_pair
@inline _edge_V(ed, integral_table) = integral_table[ed.integral_idx].V_pair
@inline _edge_S(ed, integral_table) = integral_table[ed.integral_idx].S_pair

# ---------------------------------------------------------------------------
# _3center_triplet
#
# For each (onsite + offsite) pair (i,j) compute
#   H_{ij} = Wb_i * (T_{ij} + V(Wv)_{ij}) * Wb_j'
#   S_{ij} = Wb_i * S_{ij}^{AO} * Wb_j'
# and return them as SMatrix vectors vH, vS.
#
# Layout: vH[1:nat]     = onsite blocks (i,i)
#         vH[nat+1:end] = offsite blocks in edge order
# ---------------------------------------------------------------------------
function _3center_triplet(G, basis::BasisSet,
                          Wv::AbstractVector,
                          Wb::AbstractMatrix)
    ii, jj = _get_ii_jj(G)
    edge_data = G.edge_data
    integral_table = G.graph_data.integral_table

    T_on = G.graph_data.T_on
    V_on = G.graph_data.V_on
    S_on = G.graph_data.S_on

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV = eltype(Wv)

    nnz = length(ii) + nat
    vH  = Vector{SMatrix{NORB, NORB, TV, NORB*NORB}}(undef, nnz)
    vS  = Vector{SMatrix{NORB, NORB, TV, NORB*NORB}}(undef, nnz)

    # ---- onsite (serial: nat is small, usually 1-10 atoms) ----
    H_AO_on, S_AO_on = construct_integral(Wv, Wb, T_on, V_on, S_on)
    for i in 1:nat
        Hij, Sij = siteblock(H_AO_on, S_AO_on, ao_ranges, i, i)
        vH[i] = SMatrix{NORB, NORB, TV}(Hij)
        vS[i] = SMatrix{NORB, NORB, TV}(Sij)
    end

    # ---- offsite (threaded over edges) ----
    nedges = length(ii)
    @threads for e in 1:nedges
        i  = ii[e]
        j  = jj[e]
        ed = edge_data[e]

        H_AO_off, S_AO_off = construct_integral(
            Wv, Wb,
            _edge_T(ed, integral_table),
            _edge_V(ed, integral_table),
            _edge_S(ed, integral_table))

        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
        idx = nat + e
        vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
        vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
    end

    return vH, vS
end

# ---------------------------------------------------------------------------
# ChainRules pullback for _3center_triplet
# ---------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(_3center_triplet),
                              G,
                              basis::BasisSet,
                              Wv::AbstractVector,
                              Wb::AbstractMatrix)
    ii, jj    = _get_ii_jj(G)
    edge_data = G.edge_data
    integral_table = G.graph_data.integral_table

    T_on = G.graph_data.T_on
    V_on = G.graph_data.V_on
    S_on = G.graph_data.S_on

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV     = eltype(Wv)
    nedges = length(ii)
    nnz    = nedges + nat

    vH = Vector{SMatrix{NORB, NORB, TV, NORB*NORB}}(undef, nnz)
    vS = Vector{SMatrix{NORB, NORB, TV, NORB*NORB}}(undef, nnz)

    # Forward + store pullback closures
    (H_AO_on, S_AO_on), pb_on =
        ChainRulesCore.rrule(construct_integral, Wv, Wb, T_on, V_on, S_on)

    for i in 1:nat
        Hij, Sij = siteblock(H_AO_on, S_AO_on, ao_ranges, i, i)
        vH[i] = SMatrix{NORB, NORB, TV}(Hij)
        vS[i] = SMatrix{NORB, NORB, TV}(Sij)
    end

    pb_offs = Vector{Any}(undef, nedges)
    @threads for e in 1:nedges
        i  = ii[e]
        j  = jj[e]
        ed = edge_data[e]

        (H_AO_off, S_AO_off), pb_off =
            ChainRulesCore.rrule(construct_integral,
                                 Wv, Wb,
                                 _edge_T(ed, integral_table),
                                 _edge_V(ed, integral_table),
                                 _edge_S(ed, integral_table))
        pb_offs[e] = pb_off

        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
        idx = nat + e
        vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
        vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
    end

    function pullback(ΔΩ)
        ΔvH = unthunk(ΔΩ[1])
        ΔvS = unthunk(ΔΩ[2])
        TΔ  = eltype(eltype(ΔvH))
        N_AO = size(H_AO_on, 1)

        # ---- onsite adjoint scatter ----
        ΔH_on = zeros(TΔ, N_AO, N_AO)
        ΔS_on = zeros(TΔ, N_AO, N_AO)
        for i in 1:nat
            rows = ao_ranges[i]
            @inbounds for a in 1:NORB, b in 1:NORB
                ΔH_on[rows[a], rows[b]] += ΔvH[i][a, b]
                ΔS_on[rows[a], rows[b]] += ΔvS[i][a, b]
            end
        end
        _, Wv̄_on, Wb̄_on, _, _, _ = pb_on((ΔH_on, ΔS_on))

        # ---- thread-local accumulators ----
        nth = Threads.maxthreadid()
        Wv̄_tls = [zeros(TΔ, length(Wv̄_on)) for _ in 1:nth]
        Wb̄_tls = [zeros(TΔ, size(Wb̄_on))   for _ in 1:nth]

        @threads for e in 1:nedges
            tid  = threadid()
            i    = ii[e]
            j    = jj[e]
            idx  = nat + e
            rows = ao_ranges[i]
            cols = ao_ranges[j]

            ΔH_e = zeros(TΔ, N_AO, N_AO)
            ΔS_e = zeros(TΔ, N_AO, N_AO)
            @inbounds for a in 1:NORB, b in 1:NORB
                ΔH_e[rows[a], cols[b]] += ΔvH[idx][a, b]
                ΔS_e[rows[a], cols[b]] += ΔvS[idx][a, b]
            end

            _, Wv̄_e, Wb̄_e, _, _, _ = pb_offs[e]((ΔH_e, ΔS_e))
            Wv̄_tls[tid] .+= Wv̄_e
            Wb̄_tls[tid] .+= Wb̄_e
        end

        Wv̄ = copy(Wv̄_on)
        Wb̄ = copy(Wb̄_on)
        for t in 1:nth
            Wv̄ .+= Wv̄_tls[t]
            Wb̄ .+= Wb̄_tls[t]
        end

        return NoTangent(), NoTangent(), NoTangent(), Wv̄, Wb̄
    end

    return (vH, vS), pullback
end

using CUDA
using GaussianBasis

function _copy_siteblock_kernel!(
    vH, vS,
    Hfull, Sfull,
    i::Int, j::Int, idx::Int,
    NORB::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB

    if q <= N
        t = q - 1

        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (j - 1) * NORB + b

        vH[a, b, idx] = Hfull[row, col]
        vS[a, b, idx] = Sfull[row, col]
    end

    return nothing
end

function _3center_triplet(
    G,
    basis::BasisSet,
    Wv::CUDA.CuArray{<:Real,1},
    Wb::CUDA.CuArray{<:Real,2},
)
    cache = G.graph_data.threecenter_cache

    ii = cache.ii
    jj = cache.jj

    nat  = cache.nat
    NORB = cache.NORB

    TV = eltype(Wv)
    nedges = length(ii)
    nblocks = nat + nedges

    vH = CUDA.zeros(TV, NORB, NORB, nblocks)
    vS = CUDA.zeros(TV, NORB, NORB, nblocks)

    threads = 256
    blocks = cld(NORB * NORB, threads)

    # onsite: use cache, not G.graph_data
    H_AO_on, S_AO_on = construct_integral(
        Wv,
        Wb,
        cache.T_on,
        cache.V_on,
        cache.S_on,
    )

    for i in 1:nat
        @cuda threads=threads blocks=blocks _copy_siteblock_kernel!(
            vH, vS,
            H_AO_on, S_AO_on,
            i, i, i,
            NORB,
        )
    end

    # offsite: use cache.T_pair/V_pair/S_pair, not edge_data/integral_table
    for e in 1:nedges
        i = ii[e]
        j = jj[e]

        H_AO_off, S_AO_off = construct_integral(
            Wv,
            Wb,
            cache.T_pair[e],
            cache.V_pair[e],
            cache.S_pair[e],
        )

        idx = nat + e

        @cuda threads=threads blocks=blocks _copy_siteblock_kernel!(
            vH, vS,
            H_AO_off, S_AO_off,
            i, j, idx,
            NORB,
        )
    end

    return vH, vS
end


using CUDA
using ChainRulesCore
using ChainRulesCore: NoTangent, unthunk

function _scatter_siteblock_adjoint_kernel!(
    ΔHfull,
    ΔSfull,
    ΔvH,
    ΔvS,
    i::Int,
    j::Int,
    idx::Int,
    NORB::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB

    if q <= N
        t = q - 1

        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1

        row = (i - 1) * NORB + a
        col = (j - 1) * NORB + b

        ΔHfull[row, col] += ΔvH[a, b, idx]
        ΔSfull[row, col] += ΔvS[a, b, idx]
    end

    return nothing
end

function ChainRulesCore.rrule(
    ::typeof(_3center_triplet),
    G,
    basis::BasisSet,
    Wv::CUDA.CuArray{<:Real,1},
    Wb::CUDA.CuArray{<:Real,2},
)
    cache = G.graph_data.threecenter_cache

    ii = cache.ii
    jj = cache.jj

    nat = cache.nat
    NORB = cache.NORB
    nedges = length(ii)
    nblocks = nat + nedges
    dim = nat * NORB

    TV = eltype(Wv)

    vH = CUDA.zeros(TV, NORB, NORB, nblocks)
    vS = CUDA.zeros(TV, NORB, NORB, nblocks)

    threads = 256
    blocks = cld(NORB * NORB, threads)

    # -------------------------
    # onsite forward + pullback
    # -------------------------
    (H_AO_on, S_AO_on), pb_on = ChainRulesCore.rrule(
        construct_integral,
        Wv,
        Wb,
        cache.T_on,
        cache.V_on,
        cache.S_on,
    )

    for i in 1:nat
        @cuda threads=threads blocks=blocks _copy_siteblock_kernel!(
            vH,
            vS,
            H_AO_on,
            S_AO_on,
            i,
            i,
            i,
            NORB,
        )
    end

    # -------------------------
    # offsite forward + pullbacks
    # -------------------------
    pb_offs = Vector{Any}(undef, nedges)

    for e in 1:nedges
        (H_AO_off, S_AO_off), pb_e = ChainRulesCore.rrule(
            construct_integral,
            Wv,
            Wb,
            cache.T_pair[e],
            cache.V_pair[e],
            cache.S_pair[e],
        )

        pb_offs[e] = pb_e

        i = ii[e]
        j = jj[e]
        idx = nat + e

        @cuda threads=threads blocks=blocks _copy_siteblock_kernel!(
            vH,
            vS,
            H_AO_off,
            S_AO_off,
            i,
            j,
            idx,
            NORB,
        )
    end

    function pullback_3center_gpu(ΔΩ)
        ΔvH = unthunk(ΔΩ[1])
        ΔvS = unthunk(ΔΩ[2])

        TΔ = eltype(ΔvH)

        # -------------------------
        # onsite adjoint
        # -------------------------
        ΔH_on = CUDA.zeros(TΔ, dim, dim)
        ΔS_on = CUDA.zeros(TΔ, dim, dim)

        for i in 1:nat
            @cuda threads=threads blocks=blocks _scatter_siteblock_adjoint_kernel!(
                ΔH_on,
                ΔS_on,
                ΔvH,
                ΔvS,
                i,
                i,
                i,
                NORB,
            )
        end

        _, Wvbar_on, Wbbar_on, _, _, _ = pb_on((ΔH_on, ΔS_on))

        Wvbar = copy(Wvbar_on)
        Wbbar = copy(Wbbar_on)

        # -------------------------
        # offsite adjoints
        # -------------------------
        for e in 1:nedges
            ΔH_e = CUDA.zeros(TΔ, dim, dim)
            ΔS_e = CUDA.zeros(TΔ, dim, dim)

            i = ii[e]
            j = jj[e]
            idx = nat + e

            @cuda threads=threads blocks=blocks _scatter_siteblock_adjoint_kernel!(
                ΔH_e,
                ΔS_e,
                ΔvH,
                ΔvS,
                i,
                j,
                idx,
                NORB,
            )

            _, Wvbar_e, Wbbar_e, _, _, _ = pb_offs[e]((ΔH_e, ΔS_e))

            Wvbar .+= Wvbar_e
            Wbbar .+= Wbbar_e
        end

        return (
            NoTangent(),  # typeof(_3center_triplet)
            NoTangent(),  # G
            NoTangent(),  # basis
            Wvbar,
            Wbbar,
        )
    end

    return (vH, vS), pullback_3center_gpu
end


function generate_λ_and_vecs(H::Matrix{T}, S::Matrix{TS}) where {T, TS}
    L  = cholesky(S).L
    A  = L \ (H / L')
    AA = (A + A') / 2
    F  = GenericLinearAlgebra.eigen(AA)
    λ  = F.values
    V  = L' \ F.vectors
    return λ, V
end

function generate_λ_and_vecs(Hlist::AbstractVector{TH}, Slist::AbstractVector{TS}) where {TH, TS}
    results = [generate_λ_and_vecs(Hlist[i], Slist[i]) for i in 1:length(Hlist)]
    λlist = [r[1] for r in results]
    Vlist = [r[2] for r in results]
    return λlist, Vlist
end

using CUDA
using LinearAlgebra

function generate_λ_and_vecs(
    H::CUDA.CuArray{<:Complex,2},
    S::CUDA.CuArray{<:Complex,2},
)
    # S = U' * U
    Fchol = cholesky(Hermitian(S, :U))
    U = Fchol.U

    # Transform generalized eigenproblem:
    # H v = λ S v, S = U' U
    # A y = λ y, A = U' \ H / U, v = U \ y
    A = U' \ (H / U)
    A = (A .+ A') ./ 2

    F = eigen(Hermitian(A, :U))

    λ = F.values
    Y = F.vectors

    V = U \ Y

    return λ, V
end

function generate_λ_and_vecs(
    Hout::CUDA.CuArray{T,3},
    Sout::CUDA.CuArray{T,3},
) where {T <: Complex}

    nk = size(Hout, 3)

    λ1, V1 = generate_λ_and_vecs(
        copy(@view Hout[:, :, 1]),
        copy(@view Sout[:, :, 1]),
    )

    λlist = Vector{typeof(λ1)}(undef, nk)
    Vlist = Vector{typeof(V1)}(undef, nk)

    λlist[1] = λ1
    Vlist[1] = V1

    for ik in 2:nk
        Hk = copy(@view Hout[:, :, ik])
        Sk = copy(@view Sout[:, :, ik])

        λ, V = generate_λ_and_vecs(Hk, Sk)

        λlist[ik] = λ
        Vlist[ik] = V
    end

    return λlist, Vlist
end

using CUDA
using ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, unthunk

_to_cu(x) = x isa CUDA.CuArray ? x : CUDA.cu(x)

function _zero_like_complex_array(A)
    return CUDA.zeros(eltype(A), size(A))
end

function _is_zero_tangent(x)
    return x === nothing || x isa NoTangent || x isa ZeroTangent
end

function _eigenvalue_pullback_one_k(λk, Vk, Δλk, Hk)
    λg = _to_cu(real.(λk))
    Vg = _to_cu(Vk)
    gg = _to_cu(real.(Δλk))

    # V * diag(g) * V'
    VH = (Vg .* reshape(complex.(gg), 1, :)) * Vg'

    # -V * diag(g .* λ) * V'
    VS = -((Vg .* reshape(complex.(gg .* λg), 1, :)) * Vg')

    # Match H/S element type
    return eltype(Hk).(VH), eltype(Hk).(VS)
end

function ChainRulesCore.rrule(
    ::typeof(generate_λ_and_vecs),
    Hout::CUDA.CuArray{T,3},
    Sout::CUDA.CuArray{T,3},
) where {T <: Complex}

    λlist, Vlist = generate_λ_and_vecs(Hout, Sout)

    function pullback_generate_λ_and_vecs(ΔΩ)
        Δλ_raw = unthunk(ΔΩ[1])
        ΔV_raw = unthunk(ΔΩ[2])

        Hbar = _zero_like_complex_array(Hout)
        Sbar = _zero_like_complex_array(Sout)

        if _is_zero_tangent(Δλ_raw)
            return NoTangent(), Hbar, Sbar
        end

        if !_is_zero_tangent(ΔV_raw)
            @warn "generate_λ_and_vecs rrule currently ignores gradients through Vlist; only eigenvalue gradients are handled."
        end

        nk = size(Hout, 3)

        for ik in 1:nk
            Δλk = Δλ_raw[ik]

            if _is_zero_tangent(Δλk)
                continue
            end

            Hk = @view Hout[:, :, ik]

            Hbar_k, Sbar_k = _eigenvalue_pullback_one_k(
                λlist[ik],
                Vlist[ik],
                Δλk,
                Hk,
            )

            @views begin
                Hbar[:, :, ik] .= Hbar_k
                Sbar[:, :, ik] .= Sbar_k
            end
        end

        return NoTangent(), Hbar, Sbar
    end

    return (λlist, Vlist), pullback_generate_λ_and_vecs
end