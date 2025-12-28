using Base.Threads: @threads, nthreads, threadid
using ChainRulesCore: unthunk, NoTangent

function _3center_triplet(G, basis::BasisSet, Wv::AbstractVector, Wb::AbstractMatrix)
    ii, jj = G.ii, G.jj
    Tlist = getfield.(G.edge_data, :t)
    Vlist = getfield.(G.edge_data, :v)
    Slist = getfield.(G.edge_data, :s)
    T = getfield(G.graph_data, :t)
    V = getfield(G.graph_data, :v)
    S = getfield(G.graph_data, :s)

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV = eltype(Wv)

    _init_mat() = SMatrix{NORB, NORB, TV}(zeros(TV, NORB, NORB))

    nnz = G.first[end] - 1 + nat
    vH  = [_init_mat() for _ = 1:nnz]
    vS  = [_init_mat() for _ = 1:nnz]

    # ---- onsite (serial, nat small) ----
    H_AO_onsite, S_AO_onsite = construct_integral(Wv, Wb, T, V, S)
    for i in 1:nat
        Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
        vH[i] = SMatrix{NORB, NORB, TV}(Hij)
        vS[i] = SMatrix{NORB, NORB, TV}(Sij)
    end

    # ---- offsite (threaded) ----
    nedges = length(ii)
    @threads for e in 1:nedges
        i = ii[e]
        j = jj[e]
        T_pair    = Tlist[e]
        V_pairraw = Vlist[e]
        S_pair    = Slist[e]

        H_AO_off, S_AO_off = construct_integral(Wv, Wb, T_pair, V_pairraw, S_pair)
        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)

        idx = nat + e
        vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
        vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
    end

    return vH, vS
end

function ChainRulesCore.rrule(::typeof(_3center_triplet),
                              G,
                              basis::BasisSet,
                              Wv::AbstractVector,
                              Wb::AbstractMatrix)

    ii, jj   = G.ii, G.jj
    Tlist    = getfield.(G.edge_data,  :t)
    Vlist    = getfield.(G.edge_data,  :v)
    Slist    = getfield.(G.edge_data,  :s)
    T        = getfield(G.graph_data,  :t)
    V        = getfield(G.graph_data,  :v)
    S        = getfield(G.graph_data,  :s)

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV = eltype(Wv)

    _init_mat() = SMatrix{NORB, NORB, TV}(zeros(TV, NORB, NORB))

    nnz  = G.first[end] - 1 + nat
    vH   = [_init_mat() for _ in 1:nnz]
    vS   = [_init_mat() for _ in 1:nnz]

    # ---------- onsite (match primal) ----------
    (H_AO_onsite, S_AO_onsite), pb_onsite =
        ChainRulesCore.rrule(construct_integral, Wv, Wb, T, V, S)

    for i in 1:nat
        Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
        vH[i] = SMatrix{NORB, NORB, TV}(Hij)
        vS[i] = SMatrix{NORB, NORB, TV}(Sij)
    end

    # ---------- offsite forward (threaded) ----------
    nedges  = length(ii)
    pb_offs = Vector{Any}(undef, nedges)   # closure container (type-erased but safe)

    @threads for e in 1:nedges
        i = ii[e]
        j = jj[e]
        T_pair    = Tlist[e]
        V_pairraw = Vlist[e]
        S_pair    = Slist[e]

        (H_AO_off, S_AO_off), pb_off =
            ChainRulesCore.rrule(construct_integral, Wv, Wb, T_pair, V_pairraw, S_pair)

        pb_offs[e] = pb_off

        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
        idx = nat + e
        vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
        vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
    end

    function pullback(ΔΩ)
        ΔvH_th, ΔvS_th = ΔΩ
        ΔvH = unthunk(ΔvH_th)
        ΔvS = unthunk(ΔvS_th)

        # tangent element type
        TΔ   = eltype(ΔvH[1])
        N_AO = size(H_AO_onsite, 1)

        # ---- scatter onsite adjoints back to AO space (serial) ----
        ΔH_AO_onsite = zeros(TΔ, N_AO, N_AO)
        ΔS_AO_onsite = zeros(TΔ, N_AO, N_AO)

        for i in 1:nat
            rows = ao_ranges[i]
            cols = ao_ranges[i]
            ΔHij = ΔvH[i]
            ΔSij = ΔvS[i]
            @inbounds for a in 1:NORB, b in 1:NORB
                ΔH_AO_onsite[rows[a], cols[b]] += ΔHij[a, b]
                ΔS_AO_onsite[rows[a], cols[b]] += ΔSij[a, b]
            end
        end

        # pb returns: (Δf, ΔWv, ΔWb, ΔT, ΔV, ΔS)
        _, Wv̄_ons, Wb̄_ons, _, _, _ =
            pb_onsite((ΔH_AO_onsite, ΔS_AO_onsite))

        # ---- thread-local accumulators for offsite ----
        Wv̄_tls = Vector{typeof(Wv̄_ons)}(undef, nthreads())
        Wb̄_tls = Vector{typeof(Wb̄_ons)}(undef, nthreads())
        for t in 1:nthreads()
            Wv̄_tls[t] = similar(Wv̄_ons); fill!(Wv̄_tls[t], zero(eltype(Wv̄_ons)))
            Wb̄_tls[t] = similar(Wb̄_ons); fill!(Wb̄_tls[t], zero(eltype(Wb̄_ons)))
        end

        # ---- offsite pullbacks (threaded) ----
        @threads for e in 1:nedges
            tid = threadid()

            i   = ii[e]
            j   = jj[e]
            idx = nat + e

            rows = ao_ranges[i]
            cols = ao_ranges[j]
            ΔHij = ΔvH[idx]
            ΔSij = ΔvS[idx]

            # still heavy: big AO matrices per edge (but thread-safe)
            ΔH_AO_e = zeros(TΔ, N_AO, N_AO)
            ΔS_AO_e = zeros(TΔ, N_AO, N_AO)
            @inbounds for a in 1:NORB, b in 1:NORB
                ΔH_AO_e[rows[a], cols[b]] += ΔHij[a, b]
                ΔS_AO_e[rows[a], cols[b]] += ΔSij[a, b]
            end

            _, Wv̄_e, Wb̄_e, _, _, _ = pb_offs[e]((ΔH_AO_e, ΔS_AO_e))

            Wv̄_tls[tid] .+= Wv̄_e
            Wb̄_tls[tid] .+= Wb̄_e
        end

        # ---- reduction ----
        Wv̄_total = copy(Wv̄_ons)
        Wb̄_total = copy(Wb̄_ons)
        for t in 1:nthreads()
            Wv̄_total .+= Wv̄_tls[t]
            Wb̄_total .+= Wb̄_tls[t]
        end

        return NoTangent(), NoTangent(), NoTangent(), Wv̄_total, Wb̄_total
    end

    return (vH, vS), pullback
end

function generate_λ(H::Matrix{T}, S::Matrix{TS}) where {T, TS}
    L = cholesky(S).L
    A = L \ (H / L')
    AA = (A + A') / 2
    λ = GenericLinearAlgebra.eigen(AA).values
    return λ
end

function generate_λ(Hlist::AbstractVector{TH}, Slist::AbstractVector{TS}) where {TH, TS}
   λlist = [generate_λ(Hlist[i], Slist[i]) for i in 1:length(Hlist)]
   return λlist
end