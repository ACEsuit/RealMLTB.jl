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

@inline _edge_T(ed) = ed.T_pair
@inline _edge_V(ed) = ed.V_pair
@inline _edge_S(ed) = ed.S_pair

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
            Wv, Wb, _edge_T(ed), _edge_V(ed), _edge_S(ed))

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
                                 Wv, Wb, _edge_T(ed), _edge_V(ed), _edge_S(ed))
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
        nth = nthreads()
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

# ---------------------------------------------------------------------------
# generate_λ: solve generalised eigenvalue problem H C = S C Λ
# via Cholesky factorisation of S (avoids full generalised solver).
# ---------------------------------------------------------------------------
using ChainRulesCore: @thunk

function generate_λ(H::AbstractMatrix, S::AbstractMatrix)
    H_herm = Hermitian((H + H') / 2)
    S_herm = Hermitian((S + S') / 2)
    F = eigen(S_herm)
    tol  = maximum(abs.(F.values)) * size(S, 1) * eps(real(eltype(S)))
    keep = F.values .> tol
    V    = F.vectors[:, keep]
    d    = F.values[keep]
    L    = V * Diagonal(sqrt.(d))
    H_red = Hermitian(L' * H_herm * L)
    return real.(eigvals(H_red))
end

function generate_λ(Hlist::AbstractVector, Slist::AbstractVector)
    return [generate_λ(Hlist[i], Slist[i]) for i in eachindex(Hlist)]
end

function ChainRulesCore.rrule(::typeof(generate_λ), H::AbstractMatrix, S::AbstractMatrix)
    H_herm = Hermitian((H + H') / 2)
    S_herm = Hermitian((S + S') / 2)

    # 保护：NaN/Inf 时返回零梯度
    if !all(isfinite, H_herm) || !all(isfinite, S_herm)
        λ_dummy = zeros(real(eltype(H)), size(H, 1))
        function pullback_nan(Δλ)
            return NoTangent(),
                   zeros(eltype(H), size(H)),
                   zeros(eltype(S), size(S))
        end
        return λ_dummy, pullback_nan
    end

    FS   = eigen(S_herm)
    tol  = maximum(abs.(FS.values)) * size(S, 1) * eps(real(eltype(S)))
    keep = FS.values .> tol

    if !any(keep)
        λ_dummy = zeros(real(eltype(H)), size(H, 1))
        function pullback_empty(Δλ)
            return NoTangent(),
                   zeros(eltype(H), size(H)),
                   zeros(eltype(S), size(S))
        end
        return λ_dummy, pullback_empty
    end

    V   = FS.vectors[:, keep]
    d   = FS.values[keep]
    sqd = sqrt.(d)
    L   = V * Diagonal(sqd)

    H_red = Hermitian(L' * H_herm * L)

    if !all(isfinite, H_red)
        λ_dummy = zeros(real(eltype(H)), size(H, 1))
        function pullback_nan2(Δλ)
            return NoTangent(),
                   zeros(eltype(H), size(H)),
                   zeros(eltype(S), size(S))
        end
        return λ_dummy, pullback_nan2
    end

    FH    = eigen(H_red)
    λ     = real.(FH.values)
    C_red = FH.vectors
    C     = L * C_red

    function pullback(Δλ)
        Δλ_real = real.(unthunk(Δλ))

        H̄ = @thunk begin
            G_mat = C * Diagonal(Δλ_real) * C'
            real.((G_mat + G_mat') / 2)
        end

        S̄ = @thunk begin
            function _lambda_from_S(Svec)
                S2  = reshape(Svec, size(S))
                S2h = Hermitian((S2 + S2') / 2)
                if !all(isfinite, S2h)
                    return zeros(real(eltype(S)), length(λ))
                end
                FS2   = eigen(S2h)
                keep2 = FS2.values .> tol
                if !any(keep2)
                    return zeros(real(eltype(S)), length(λ))
                end
                V2    = FS2.vectors[:, keep2]
                d2    = FS2.values[keep2]
                L2    = V2 * Diagonal(sqrt.(d2))
                H_red2 = Hermitian(L2' * H_herm * L2)
                if !all(isfinite, H_red2)
                    return zeros(real(eltype(S)), length(λ))
                end
                return real.(eigvals(H_red2))
            end
            gs = Zygote.gradient(s -> dot(Δλ_real, _lambda_from_S(s)), vec(S))[1]
            if gs === nothing || !all(isfinite, gs)
                zeros(eltype(S), size(S))
            else
                reshape(real.(gs), size(S))
            end
        end

        return NoTangent(), H̄, S̄
    end

    return λ, pullback
end

function ChainRulesCore.rrule(::typeof(generate_λ),
                               Hlist::AbstractVector,
                               Slist::AbstractVector)
    results_and_pbs = [ChainRulesCore.rrule(generate_λ, Hlist[i], Slist[i])
                       for i in eachindex(Hlist)]
    λlist = [r[1] for r in results_and_pbs]
    pbs   = [r[2] for r in results_and_pbs]

    function pullback(Δλlist_thunk)
        Δλlist = unthunk(Δλlist_thunk)
        H̄list  = Vector{typeof(Hlist[1])}(undef, length(Hlist))
        S̄list  = Vector{typeof(Slist[1])}(undef, length(Slist))
        for i in eachindex(Hlist)
            Δλi = unthunk(Δλlist[i])
            if Δλi isa ChainRulesCore.ZeroTangent || Δλi isa ChainRulesCore.NoTangent
                H̄list[i] = zero(Hlist[i])
                S̄list[i] = zero(Slist[i])
            else
                _, H̄i, S̄i = pbs[i](Δλi)
                H̄list[i]   = unthunk(H̄i)
                S̄list[i]   = unthunk(S̄i)
            end
        end
        return NoTangent(), H̄list, S̄list
    end

    return λlist, pullback
end