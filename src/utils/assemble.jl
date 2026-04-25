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
