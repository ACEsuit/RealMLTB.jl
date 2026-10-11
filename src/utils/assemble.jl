using Base.Threads: @threads, nthreads, threadid
using ChainRulesCore: unthunk, NoTangent
using StaticArrays

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

@inline _is_cut_graph(G) = hasproperty(G.graph_data, :cut_integrals)

function _zero_smatrix(::Type{TV}, ::Val{NORB}) where {TV,NORB}
    return SMatrix{NORB,NORB,TV,NORB*NORB}(zeros(TV, NORB, NORB))
end

function _cut_atom_species(G, basis::BasisSet)
    if hasproperty(G.graph_data, :atom_species)
        return G.graph_data.atom_species
    end

    return [ChemicalSpecies(atomic_number = at.Z) for at in basis.atoms]
end

@inline function _cut_orb_range(i::Integer, NORB::Integer)
    return (i - 1) * NORB .+ (1:NORB)
end

@inline function _block_range(i::Integer, n::Integer)
    return (i - 1) * n .+ (1:n)
end

function _cut_ranges(G, Wv, Wb)
    nat = length(G.first) - 1

    NORB = size(Wb, 1) ÷ nat
    NAO  = size(Wb, 2) ÷ nat
    NWV  = length(Wv) ÷ nat

    @assert size(Wb, 1) == nat * NORB
    @assert size(Wb, 2) == nat * NAO
    @assert length(Wv)  == nat * NWV
    @assert NAO == NWV "Mismatch: Wb has NAO=$NAO per atom, but Wv has NWV=$NWV per atom."

    orb_ranges = [_block_range(i, NORB) for i in 1:nat]
    ao_ranges  = [_block_range(i, NAO)  for i in 1:nat]

    return NORB, nat, orb_ranges, ao_ranges
end

# Contract along the third index:
#   V3[a,b,c] * Wv[c] -> matrix[a,b]
# This is the old convention and is kept for non-reinterpreted uses.
function _contract_v3(V3::AbstractArray{<:Real,3}, Wvk::AbstractVector)
    ni, nj, nk = size(V3)
    @assert length(Wvk) == nk
    return reshape(reshape(V3, ni * nj, nk) * Wvk, ni, nj)
end

# Contract along the second index:
#   V3[a,b,c] * Wv[b] -> matrix[a,c]
#
# This is the convention needed for the cut graph:
#   integral_idx = (i, p, j)
#   V3 = ∫ φ_i φ_p φ_j
#   p is the potential/Wv center
#   j is the Hamiltonian column center and carries the Bloch phase.
function _contract_v3_dim2(V3::AbstractArray{<:Real,3}, Wvp::AbstractVector)
    ni, np, nj = size(V3)
    @assert length(Wvp) == np

    T = promote_type(eltype(V3), eltype(Wvp))
    out = zeros(T, ni, nj)

    @inbounds for j in 1:nj
        for p in 1:np
            wp = Wvp[p]
            for i in 1:ni
                out[i, j] += V3[i, p, j] * wp
            end
        end
    end

    return out
end

function _cut_block_metadata(G)
    cut = G.graph_data.cut_integrals

    T_off = cut.T_off
    S_off = cut.S_off
    V_off = cut.V_off

    meta = Vector{NamedTuple{(:kind, :i, :j, :k, :R_cell)}}()
    sizehint!(meta, length(T_off) + length(S_off) + length(V_off))

    for ed in T_off
        i, j = ed.integral_idx
        push!(meta, (kind = :T, i = i, j = j, k = 0, R_cell = ed.R_cell))
    end

    for ed in S_off
        i, j = ed.integral_idx
        push!(meta, (kind = :S, i = i, j = j, k = 0, R_cell = ed.R_cell))
    end

    for ed in V_off
        i, p, j = ed.integral_idx
        push!(meta, (kind = :V, i = i, j = j, k = p, R_cell = ed.R_cell))
    end

    return meta
end

function _3center_triplet_cut(G, basis::BasisSet,
                              Wv::AbstractVector,
                              Wb::AbstractMatrix)
    cut = G.graph_data.cut_integrals

    T_off = cut.T_off
    S_off = cut.S_off
    V_off = cut.V_off

    atom_species = _cut_atom_species(G, basis)

    NORB, nat, orb_ranges, ao_ranges = _cut_ranges(G, Wv, Wb)
    TV = eltype(Wv)

    nblocks = nat + length(T_off) + length(S_off) + length(V_off)

    vH = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nblocks)
    vS = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nblocks)

    zero_block = _zero_smatrix(TV, Val(NORB))

    for i in 1:nat
        rows_orb = orb_ranges[i]
        rows_ao  = ao_ranges[i]

        Zi = atom_species[i]

        Wbi = Wb[rows_orb, rows_ao]
        Wvi = Wv[rows_ao]

        Hii, Sii = construct_integral(
            Wvi,
            Wbi,
            G.graph_data.T_on[Zi],
            G.graph_data.V_on[Zi],
            G.graph_data.S_on[Zi],
        )

        vH[i] = SMatrix{NORB,NORB,TV}(Hii)
        vS[i] = SMatrix{NORB,NORB,TV}(Sii)
    end

    offset = nat

    @threads for e in eachindex(T_off)
        ed = T_off[e]

        i, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, aj = ao_ranges[i], ao_ranges[j]

        Hij = Wb[ri, ai] * (ed.w .* ed.V2) * Wb[rj, aj]'

        vH[offset + e] = SMatrix{NORB,NORB,TV}(Hij)
        vS[offset + e] = zero_block
    end

    offset += length(T_off)

    @threads for e in eachindex(S_off)
        ed = S_off[e]

        i, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, aj = ao_ranges[i], ao_ranges[j]

        Sij = Wb[ri, ai] * (ed.w .* ed.V2) * Wb[rj, aj]'

        vH[offset + e] = zero_block
        vS[offset + e] = SMatrix{NORB,NORB,TV}(Sij)
    end

    # -----------------------------------------------------------------------
    # Three-center potential contribution.
    #
    # Convention:
    #   ed.integral_idx = (i, p, j)
    #   ed.V3[a,b,c]    = ∫ φ_i[a] φ_p[b] φ_j[c]
    #
    # Contract with Wv on the potential center p:
    #   Vij[a,c] = Σ_b V3[a,b,c] Wv_p[b]
    #
    # Then place into Hamiltonian block H_{i,j}.
    # Bloch phase is handled later by assembly.jl using ed.R_cell.
    # -----------------------------------------------------------------------
    offset += length(S_off)

    @threads for e in eachindex(V_off)
        ed = V_off[e]

        i, p, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, ap, aj = ao_ranges[i], ao_ranges[p], ao_ranges[j]

        Vij = _contract_v3_dim2(ed.V3, Wv[ap])

        Hij = Wb[ri, ai] * (ed.w .* Vij) * Wb[rj, aj]'

        vH[offset + e] = SMatrix{NORB,NORB,TV}(Hij)
        vS[offset + e] = zero_block
    end

    return vH, vS
end

# ---------------------------------------------------------------------------
# _3center_triplet
#
# For each (onsite + offsite) pair (i,j) compute
#   H_{ij} = Wb_i * (T_{ij} + V(Wv)_{ij}) * Wb_j'
#   S_{ij} = Wb_i * S_{ij}^{AO} * Wb_j'
# and return them as SMatrix vectors vH, vS.
#
# Layout:
#   vH[1:nat]     = onsite blocks (i,i)
#   vH[nat+1:end] = offsite/additive blocks in metadata order
# ---------------------------------------------------------------------------
function _3center_triplet(G, basis::BasisSet,
                          Wv::AbstractVector,
                          Wb::AbstractMatrix)
    if _is_cut_graph(G)
        return _3center_triplet_cut(G, basis, Wv, Wb)
    end

    ii, jj = _get_ii_jj(G)

    edge_data = G.edge_data
    integral_table = G.graph_data.integral_table

    T_on = G.graph_data.T_on
    V_on = G.graph_data.V_on
    S_on = G.graph_data.S_on

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV = eltype(Wv)

    nnz = length(ii) + nat

    vH = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nnz)
    vS = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nnz)

    # ---- onsite ----
    H_AO_on, S_AO_on = construct_integral(Wv, Wb, T_on, V_on, S_on)

    for i in 1:nat
        Hij, Sij = siteblock(H_AO_on, S_AO_on, ao_ranges, i, i)
        vH[i] = SMatrix{NORB,NORB,TV}(Hij)
        vS[i] = SMatrix{NORB,NORB,TV}(Sij)
    end

    # ---- offsite ----
    nedges = length(ii)

    @threads for e in 1:nedges
        i  = ii[e]
        j  = jj[e]
        ed = edge_data[e]

        H_AO_off, S_AO_off = construct_integral(
            Wv,
            Wb,
            _edge_T(ed, integral_table),
            _edge_V(ed, integral_table),
            _edge_S(ed, integral_table),
        )

        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)

        idx = nat + e

        vH[idx] = SMatrix{NORB,NORB,TV}(Hij)
        vS[idx] = SMatrix{NORB,NORB,TV}(Sij)
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
    if _is_cut_graph(G)
        return _3center_triplet_cut_rrule(G, basis, Wv, Wb)
    end

    ii, jj = _get_ii_jj(G)

    edge_data = G.edge_data
    integral_table = G.graph_data.integral_table

    T_on = G.graph_data.T_on
    V_on = G.graph_data.V_on
    S_on = G.graph_data.S_on

    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
    TV     = eltype(Wv)
    nedges = length(ii)
    nnz    = nedges + nat

    vH = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nnz)
    vS = Vector{SMatrix{NORB,NORB,TV,NORB*NORB}}(undef, nnz)

    # Forward + store pullback closures
    (H_AO_on, S_AO_on), pb_on =
        ChainRulesCore.rrule(construct_integral, Wv, Wb, T_on, V_on, S_on)

    for i in 1:nat
        Hij, Sij = siteblock(H_AO_on, S_AO_on, ao_ranges, i, i)
        vH[i] = SMatrix{NORB,NORB,TV}(Hij)
        vS[i] = SMatrix{NORB,NORB,TV}(Sij)
    end

    pb_offs = Vector{Any}(undef, nedges)

    @threads for e in 1:nedges
        i  = ii[e]
        j  = jj[e]
        ed = edge_data[e]

        (H_AO_off, S_AO_off), pb_off =
            ChainRulesCore.rrule(
                construct_integral,
                Wv,
                Wb,
                _edge_T(ed, integral_table),
                _edge_V(ed, integral_table),
                _edge_S(ed, integral_table),
            )

        pb_offs[e] = pb_off

        Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)

        idx = nat + e

        vH[idx] = SMatrix{NORB,NORB,TV}(Hij)
        vS[idx] = SMatrix{NORB,NORB,TV}(Sij)
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
        Wb̄_tls = [zeros(TΔ, size(Wb̄_on)) for _ in 1:nth]

        @threads for e in 1:nedges
            tid = threadid()

            i = ii[e]
            j = jj[e]

            idx = nat + e

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
# ChainRules pullback for cut graph version
# ---------------------------------------------------------------------------
function _3center_triplet_cut_rrule(G, basis::BasisSet,
                                    Wv::AbstractVector,
                                    Wb::AbstractMatrix)
    vH, vS = _3center_triplet_cut(G, basis, Wv, Wb)

    cut = G.graph_data.cut_integrals

    T_off = cut.T_off
    S_off = cut.S_off
    V_off = cut.V_off

    atom_species = _cut_atom_species(G, basis)

    NORB, nat, orb_ranges, ao_ranges = _cut_ranges(G, Wv, Wb)

    function pullback_cut(ΔΩ)
        ΔvH = unthunk(ΔΩ[1])
        ΔvS = unthunk(ΔΩ[2])

        TΔ = eltype(eltype(ΔvH))

        Wv̄ = zeros(TΔ, size(Wv))
        Wb̄ = zeros(TΔ, size(Wb))

        # -------------------------------------------------------------------
        # Onsite adjoints
        # -------------------------------------------------------------------
        for i in 1:nat
            ro = orb_ranges[i]
            ao = ao_ranges[i]

            Zi = atom_species[i]

            Wbi = Wb[ro, ao]
            Wvi = Wv[ao]

            Tm = G.graph_data.T_on[Zi]
            Vm = G.graph_data.V_on[Zi]
            Sm = G.graph_data.S_on[Zi]

            Vpair = _contract_v3(Vm, Wvi)
            A = Tm + Vpair

            ΔH = Matrix(ΔvH[i])
            ΔS = Matrix(ΔvS[i])

            Wb̄[ro, ao] .+= real.(
                ΔH * Wbi * A' +
                ΔH' * Wbi * A +
                ΔS * Wbi * Sm' +
                ΔS' * Wbi * Sm
            )

            ΔA = real.(Wbi' * ΔH * Wbi)

            for c in axes(Vm, 3)
                Wv̄[ao[c]] += sum(@view(Vm[:, :, c]) .* ΔA)
            end
        end

        # -------------------------------------------------------------------
        # T_off adjoints
        # -------------------------------------------------------------------
        offset = nat

        for e in eachindex(T_off)
            ed = T_off[e]

            i, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, aj = ao_ranges[i], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            A = ed.w .* ed.V2

            ΔH = Matrix(ΔvH[offset + e])

            Wb̄[ri, ai] .+= real.(ΔH * Wbj * A')
            Wb̄[rj, aj] .+= real.(ΔH' * Wbi * A)
        end

        # -------------------------------------------------------------------
        # S_off adjoints
        # -------------------------------------------------------------------
        offset += length(T_off)

        for e in eachindex(S_off)
            ed = S_off[e]

            i, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, aj = ao_ranges[i], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            A = ed.w .* ed.V2

            ΔS = Matrix(ΔvS[offset + e])

            Wb̄[ri, ai] .+= real.(ΔS * Wbj * A')
            Wb̄[rj, aj] .+= real.(ΔS' * Wbi * A)
        end

        # -------------------------------------------------------------------
        # V_off adjoints
        #
        # Convention:
        #   ed.integral_idx = (i, p, j)
        #   V3[a,b,c] = ∫ φ_i[a] φ_p[b] φ_j[c]
        #   A[a,c]    = Σ_b V3[a,b,c] Wv_p[b]
        # -------------------------------------------------------------------
        offset += length(S_off)

        for e in eachindex(V_off)
            ed = V_off[e]

            i, p, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, ap, aj = ao_ranges[i], ao_ranges[p], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            V3 = ed.V3

            A = ed.w .* _contract_v3_dim2(V3, Wv[ap])

            ΔH = Matrix(ΔvH[offset + e])

            Wb̄[ri, ai] .+= real.(ΔH * Wbj * A')
            Wb̄[rj, aj] .+= real.(ΔH' * Wbi * A)

            ΔA = real.(Wbi' * ΔH * Wbj) .* ed.w

            @inbounds for b in axes(V3, 2)
                Wv̄[ap[b]] += sum(@view(V3[:, b, :]) .* ΔA)
            end
        end

        return NoTangent(), NoTangent(), NoTangent(), Wv̄, Wb̄
    end

    return (vH, vS), pullback_cut
end

# ---------------------------------------------------------------------------
# GPU path for original non-cut graph
# ---------------------------------------------------------------------------

using CUDA
using GaussianBasis


# ---------------------------------------------------------------------------
# GPU path for cut_integrals graph
# ---------------------------------------------------------------------------

function _copy_H_block_kernel!(vH, Hblk, idx::Int, NORB::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB

    if q <= N
        t = q - 1
        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1
        vH[a, b, idx] = Hblk[a, b]
    end

    return nothing
end

function _copy_S_block_kernel!(vS, Sblk, idx::Int, NORB::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * NORB

    if q <= N
        t = q - 1
        b = (t % NORB) + 1
        a = (t ÷ NORB) + 1
        vS[a, b, idx] = Sblk[a, b]
    end

    return nothing
end

function _contract_v3(V3::CUDA.CuArray{<:Real,3}, Wvk::CUDA.CuArray{<:Real,1})
    ni, nj, nk = size(V3)
    @assert length(Wvk) == nk
    return reshape(reshape(V3, ni * nj, nk) * Wvk, ni, nj)
end

function _contract_v3_dim2(V3::CUDA.CuArray{<:Real,3}, Wvp::CUDA.CuArray{<:Real,1})
    ni, np, nj = size(V3)
    @assert length(Wvp) == np

    # V3[a,b,c] * Wvp[b] -> out[a,c]
    # reshape(permutedims(V3, (1, 3, 2)), ni*nj, np) * Wvp
    Vperm = permutedims(V3, (1, 3, 2))
    return reshape(reshape(Vperm, ni * nj, np) * Wvp, ni, nj)
end

function _contract_v3_dim2_adjoint_wv(V3::CUDA.CuArray{<:Real,3}, ΔA::CUDA.CuArray{<:Real,2})
    ni, np, nj = size(V3)
    Vperm = permutedims(V3, (1, 3, 2))
    return reshape(Vperm, ni * nj, np)' * vec(ΔA)
end

function _contract_v3_adjoint_wv(V3::CUDA.CuArray{<:Real,3}, ΔA::CUDA.CuArray{<:Real,2})
    ni, nj, nk = size(V3)
    return reshape(V3, ni * nj, nk)' * vec(ΔA)
end

function _3center_triplet_cut_gpu(G, basis::BasisSet,
                                  Wv::CUDA.CuArray{<:Real,1},
                                  Wb::CUDA.CuArray{<:Real,2})
    cut = G.graph_data.cut_integrals

    T_off = cut.T_off
    S_off = cut.S_off
    V_off = cut.V_off

    atom_species = _cut_atom_species(G, basis)

    NORB, nat, orb_ranges, ao_ranges = _cut_ranges(G, Wv, Wb)
    TV = eltype(Wv)

    nblocks = nat + length(T_off) + length(S_off) + length(V_off)

    vH = CUDA.zeros(TV, NORB, NORB, nblocks)
    vS = CUDA.zeros(TV, NORB, NORB, nblocks)

    threads = 256
    blocks = cld(NORB * NORB, threads)

    # onsite
    for i in 1:nat
        rows_orb = orb_ranges[i]
        rows_ao  = ao_ranges[i]

        Zi = atom_species[i]

        Wbi = Wb[rows_orb, rows_ao]
        Wvi = Wv[rows_ao]

        Hii, Sii = construct_integral(
            Wvi,
            Wbi,
            G.graph_data.T_on[Zi],
            G.graph_data.V_on[Zi],
            G.graph_data.S_on[Zi],
        )

        @cuda threads=threads blocks=blocks _copy_H_block_kernel!(vH, Hii, i, NORB)
        @cuda threads=threads blocks=blocks _copy_S_block_kernel!(vS, Sii, i, NORB)
    end

    offset = nat

    # T offsite
    for e in eachindex(T_off)
        ed = T_off[e]

        i, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, aj = ao_ranges[i], ao_ranges[j]

        Hij = Wb[ri, ai] * (ed.w .* ed.V2) * Wb[rj, aj]'

        @cuda threads=threads blocks=blocks _copy_H_block_kernel!(vH, Hij, offset + e, NORB)
    end

    offset += length(T_off)

    # S offsite
    for e in eachindex(S_off)
        ed = S_off[e]

        i, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, aj = ao_ranges[i], ao_ranges[j]

        Sij = Wb[ri, ai] * (ed.w .* ed.V2) * Wb[rj, aj]'

        @cuda threads=threads blocks=blocks _copy_S_block_kernel!(vS, Sij, offset + e, NORB)
    end

    offset += length(S_off)

    # V offsite
    for e in eachindex(V_off)
        ed = V_off[e]

        i, p, j = ed.integral_idx

        ri, rj = orb_ranges[i], orb_ranges[j]
        ai, ap, aj = ao_ranges[i], ao_ranges[p], ao_ranges[j]

        Vij = _contract_v3_dim2(ed.V3, Wv[ap])
        Hij = Wb[ri, ai] * (ed.w .* Vij) * Wb[rj, aj]'

        @cuda threads=threads blocks=blocks _copy_H_block_kernel!(vH, Hij, offset + e, NORB)
    end

    return vH, vS
end

function _3center_triplet_cut_gpu_rrule(G, basis::BasisSet,
                                        Wv::CUDA.CuArray{<:Real,1},
                                        Wb::CUDA.CuArray{<:Real,2})
    vH, vS = _3center_triplet_cut_gpu(G, basis, Wv, Wb)

    cut = G.graph_data.cut_integrals

    T_off = cut.T_off
    S_off = cut.S_off
    V_off = cut.V_off

    atom_species = _cut_atom_species(G, basis)

    NORB, nat, orb_ranges, ao_ranges = _cut_ranges(G, Wv, Wb)

    function pullback_cut_gpu(ΔΩ)
        ΔvH = unthunk(ΔΩ[1])
        ΔvS = unthunk(ΔΩ[2])

        TΔ = eltype(ΔvH)

        Wv̄ = CUDA.zeros(TΔ, size(Wv))
        Wb̄ = CUDA.zeros(TΔ, size(Wb))

        # onsite adjoints
        for i in 1:nat
            ro = orb_ranges[i]
            ao = ao_ranges[i]

            Zi = atom_species[i]

            Wbi = Wb[ro, ao]
            Wvi = Wv[ao]

            Tm = G.graph_data.T_on[Zi]
            Vm = G.graph_data.V_on[Zi]
            Sm = G.graph_data.S_on[Zi]

            Vpair = _contract_v3(Vm, Wvi)
            A = Tm + Vpair

            ΔH = copy(@view ΔvH[:, :, i])
            ΔS = copy(@view ΔvS[:, :, i])

            @views Wb̄[ro, ao] .+= real.(
                ΔH * Wbi * A' +
                ΔH' * Wbi * A +
                ΔS * Wbi * Sm' +
                ΔS' * Wbi * Sm
            )

            ΔA = real.(Wbi' * ΔH * Wbi)
            @views Wv̄[ao] .+= real.(_contract_v3_adjoint_wv(Vm, ΔA))
        end

        # T_off adjoints
        offset = nat

        for e in eachindex(T_off)
            ed = T_off[e]

            i, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, aj = ao_ranges[i], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            A = ed.w .* ed.V2
            ΔH = copy(@view ΔvH[:, :, offset + e])

            @views Wb̄[ri, ai] .+= real.(ΔH * Wbj * A')
            @views Wb̄[rj, aj] .+= real.(ΔH' * Wbi * A)
        end

        # S_off adjoints
        offset += length(T_off)

        for e in eachindex(S_off)
            ed = S_off[e]

            i, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, aj = ao_ranges[i], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            A = ed.w .* ed.V2
            ΔS = copy(@view ΔvS[:, :, offset + e])

            @views Wb̄[ri, ai] .+= real.(ΔS * Wbj * A')
            @views Wb̄[rj, aj] .+= real.(ΔS' * Wbi * A)
        end

        # V_off adjoints
        offset += length(S_off)

        for e in eachindex(V_off)
            ed = V_off[e]

            i, p, j = ed.integral_idx

            ri, rj = orb_ranges[i], orb_ranges[j]
            ai, ap, aj = ao_ranges[i], ao_ranges[p], ao_ranges[j]

            Wbi, Wbj = Wb[ri, ai], Wb[rj, aj]

            V3 = ed.V3

            A = ed.w .* _contract_v3_dim2(V3, Wv[ap])
            ΔH = copy(@view ΔvH[:, :, offset + e])

            @views Wb̄[ri, ai] .+= real.(ΔH * Wbj * A')
            @views Wb̄[rj, aj] .+= real.(ΔH' * Wbi * A)

            ΔA = real.(Wbi' * ΔH * Wbj) .* ed.w
            @views Wv̄[ap] .+= real.(_contract_v3_dim2_adjoint_wv(V3, ΔA))
        end

        return NoTangent(), NoTangent(), NoTangent(), Wv̄, Wb̄
    end

    return (vH, vS), pullback_cut_gpu
end

function _copy_siteblock_kernel!(
    vH,
    vS,
    Hfull,
    Sfull,
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
    if _is_cut_graph(G)
        return _3center_triplet_cut_gpu(G, basis, Wv, Wb)
    end

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

    # offsite: use cache.T_pair/V_pair/S_pair
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
    if _is_cut_graph(G)
        return _3center_triplet_cut_gpu_rrule(G, basis, Wv, Wb)
    end

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

# ---------------------------------------------------------------------------
# Generalized eigenvalue helpers
# ---------------------------------------------------------------------------

function generate_λ_and_vecs(H::Matrix{T}, S::Matrix{TS}) where {T, TS}
    L  = cholesky(S).L
    A  = L \ (H / L')
    AA = (A + A') / 2
    F  = GenericLinearAlgebra.eigen(AA)
    λ  = F.values
    V  = L' \ F.vectors
    return λ, V
end

function generate_λ_and_vecs(Hlist::AbstractVector{TH},
                             Slist::AbstractVector{TS}) where {TH, TS}
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

# NOTE: an older CUDA rrule for generate_λ_and_vecs(::CuArray{T,3}, ::CuArray{T,3})
# used to sit here.  It handled only the eigenvalue gradient and warned that it
# ignored dVlist.  The block further down ("Full CUDA generalized eigenproblem
# rrule override") was written to replace it but was appended instead, so the
# same signature was defined twice: Julia refuses to precompile the package and
# the later definition silently won.  The old one is removed here; the override
# below, which handles both dlambda and dV, is the live definition.


# ============================================================================
# Full CUDA generalized eigenproblem rrule override
# ============================================================================

# eigen_full_rrule_patch.jl
#
# Paste this block near the end of src/utils/assemble.jl, replacing the old
# CUDA rrule for generate_λ_and_vecs(Hout::CuArray, Sout::CuArray).
#
# Purpose:
#   Handle gradients through both λlist and Vlist.
#
# Assumptions:
#   - Hermitian generalized eigenproblem H v = λ S v.
#   - Eigenvectors are S-normalized: V' * S * V = I.
#   - Eigenvalues are locally non-degenerate.  Near degeneracies, eigenvector
#     gradients are ill-conditioned by construction.

using CUDA
using LinearAlgebra
using ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, unthunk

_eig_to_cu(x) = x isa CUDA.CuArray ? x : CUDA.cu(x)

_eig_is_zero_tangent(x) =
    x === nothing || x isa NoTangent || x isa ZeroTangent

function _eig_zero_like(A)
    return CUDA.zeros(eltype(A), size(A))
end

function _eig_eye(::Type{T}, n::Integer) where {T}
    return CUDA.CuArray(Matrix{T}(I, n, n))
end

function _eig_λbar(Δλk, λk)
    if _eig_is_zero_tangent(Δλk)
        return CUDA.zeros(eltype(real.(λk)), length(λk))
    else
        return _eig_to_cu(real.(Δλk))
    end
end

function _eig_Vbar(ΔVk, Vk)
    if _eig_is_zero_tangent(ΔVk)
        return CUDA.zeros(eltype(Vk), size(Vk))
    else
        return _eig_to_cu(ΔVk)
    end
end

"""
    _generalized_eigen_pullback_one_k_full(λ, V, Δλ, ΔV, H, S)

Adjoint of the Hermitian generalized eigensolver

    H V = S V Diagonal(λ),     V' S V = I

with respect to H and S.

This includes both eigenvalue and eigenvector upstream sensitivities.
"""
function _generalized_eigen_pullback_one_k_full(λk, Vk, Δλk, ΔVk, Hk, Sk)
    λ = _eig_to_cu(real.(λk))
    V = _eig_to_cu(Vk)

    n = length(λ)

    λbar = _eig_λbar(Δλk, λ)
    Vbar = _eig_Vbar(ΔVk, V)

    Tλ = eltype(λ)
    TC = eltype(Hk)

    eye = _eig_eye(Tλ, n)
    offmask = one(Tλ) .- eye

    # M_ij = v_i' * ∂L/∂v_j in the Euclidean pairing.
    M = V' * Vbar

    # Denominator for eigenvector derivative:
    #   K_ij = (E_ij - λ_j F_ij) / (λ_j - λ_i), i != j
    λrow = reshape(λ, 1, n)   # λ_j
    λcol = reshape(λ, n, 1)   # λ_i

    denom = λrow .- λcol

    # Avoid NaN on diagonal; diagonal is removed by offmask.
    denom_safe = denom .+ eye

    C = (M ./ denom_safe) .* offmask

    # Contributions from eigenvalues:
    #   dλ_i = v_i' (dH - λ_i dS) v_i
    Ebar = complex.(eye .* reshape(λbar, n, 1))
    Fbar = -complex.(eye .* reshape(λ .* λbar, n, 1))

    # Contributions from eigenvectors:
    #   off-diagonal: K_ij = (E_ij - λ_j F_ij)/(λ_j - λ_i)
    Ebar .+= C
    Fbar .-= complex.(λrow) .* C

    # Diagonal normalization term:
    #   K_ii = -1/2 F_ii
    # This fixes the S-normalization part of the eigenvector derivative.
    Fbar .-= complex(Tλ(0.5)) .* (eye .* M)

    # Since H and S are Hermitian, return Hermitian adjoints.
    Ebar = (Ebar .+ Ebar') ./ 2
    Fbar = (Fbar .+ Fbar') ./ 2

    Hbar = V * Ebar * V'
    Sbar = V * Fbar * V'

    return TC.(Hbar), TC.(Sbar)
end

function ChainRulesCore.rrule(
    ::typeof(generate_λ_and_vecs),
    Hout::CUDA.CuArray{T,3},
    Sout::CUDA.CuArray{T,3},
) where {T <: Complex}

    λlist, Vlist = generate_λ_and_vecs(Hout, Sout)

    function pullback_generate_λ_and_vecs_full(ΔΩ)
        Δλ_raw = unthunk(ΔΩ[1])
        ΔV_raw = unthunk(ΔΩ[2])

        Hbar = _eig_zero_like(Hout)
        Sbar = _eig_zero_like(Sout)

        nk = size(Hout, 3)

        for ik in 1:nk
            Δλk =
                _eig_is_zero_tangent(Δλ_raw) ? NoTangent() : Δλ_raw[ik]

            ΔVk =
                _eig_is_zero_tangent(ΔV_raw) ? NoTangent() : ΔV_raw[ik]

            if _eig_is_zero_tangent(Δλk) && _eig_is_zero_tangent(ΔVk)
                continue
            end

            Hk = copy(@view Hout[:, :, ik])
            Sk = copy(@view Sout[:, :, ik])

            Hbar_k, Sbar_k = _generalized_eigen_pullback_one_k_full(
                λlist[ik],
                Vlist[ik],
                Δλk,
                ΔVk,
                Hk,
                Sk,
            )

            @views begin
                Hbar[:, :, ik] .= Hbar_k
                Sbar[:, :, ik] .= Sbar_k
            end
        end

        return NoTangent(), Hbar, Sbar
    end

    return (λlist, Vlist), pullback_generate_λ_and_vecs_full
end

function ChainRulesCore.rrule(
    ::typeof(generate_λ_and_vecs),
    H::CUDA.CuArray{T,2},
    S::CUDA.CuArray{T,2},
) where {T <: Complex}

    λ, V = generate_λ_and_vecs(H, S)

    function pullback_generate_λ_and_vecs_one_full(ΔΩ)
        Δλ = unthunk(ΔΩ[1])
        ΔV = unthunk(ΔΩ[2])

        if _eig_is_zero_tangent(Δλ) && _eig_is_zero_tangent(ΔV)
            return NoTangent(), _eig_zero_like(H), _eig_zero_like(S)
        end

        Hbar, Sbar = _generalized_eigen_pullback_one_k_full(
            λ,
            V,
            Δλ,
            ΔV,
            H,
            S,
        )

        return NoTangent(), Hbar, Sbar
    end

    return (λ, V), pullback_generate_λ_and_vecs_one_full
end
