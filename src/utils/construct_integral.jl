using LinearAlgebra
using ChainRulesCore
using GaussianBasis

function _sparse_contract(V_pair_raw::Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}, Wv, nbas)
    idxs, vals = V_pair_raw
    V_pair = zeros(eltype(Wv), nbas, nbas)
    for (n, (a, b, c)) in enumerate(idxs)
        V_pair[a, b] += vals[n] * Wv[c]
    end
    return V_pair
end

function _sparse_contract_pullback(V_pair_raw::Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}, Δy, nbas)
    idxs, vals = V_pair_raw
    Wv̄ = zeros(nbas)
    for (n, (a, b, c)) in enumerate(idxs)
        Wv̄[c] += vals[n] * Δy[(b-1)*nbas + a]
    end
    return real.(Wv̄)
end

function _construct_integral_core(Wv, Wb, T_pair, V_pair_raw::AbstractArray{<:Real,3}, S_pair)
    nbas   = size(V_pair_raw, 1)
    M      = reshape(V_pair_raw, nbas^2, nbas)
    V_pair = reshape(M * Wv, nbas, nbas)
    A = T_pair + V_pair
    H = Wb * A      * Wb'
    S = Wb * S_pair * Wb'
    return H, S
end

function _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw::AbstractArray{<:Real,3}, S_pair)
    nbas   = size(V_pair_raw, 1)
    M      = reshape(V_pair_raw, nbas^2, nbas)
    V_pair = reshape(M * Wv, nbas, nbas)
    A = T_pair + V_pair
    H = Wb * A      * Wb'
    S = Wb * S_pair * Wb'
    return H, S, A, M
end

function _construct_integral_rrule_common(Wv, Wb, T_pair, V_pair_raw::AbstractArray{<:Real,3}, S_pair)
    H, S, A, M = _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw, S_pair)
    function pullback_common(ΔΩ)
        ΔH, ΔS = ΔΩ
        ΔWb_H = ΔH * Wb * A'      + ΔH' * Wb * A
        ΔWb_S = ΔS * Wb * S_pair' + ΔS' * Wb * S_pair
        Wb̄  = real.(ΔWb_H + ΔWb_S)
        ΔA  = real.(Wb' * ΔH * Wb)
        Δy  = vec(ΔA)
        Wv̄  = real.(M' * Δy)
        return NoTangent(), Wv̄, Wb̄, NoTangent(), NoTangent(), NoTangent()
    end
    return (H, S), pullback_common
end

function _construct_integral_core(Wv, Wb, T_pair, V_pair_raw::Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}, S_pair)
    nbas   = size(T_pair, 1)
    V_pair = _sparse_contract(V_pair_raw, Wv, nbas)
    A = T_pair + V_pair
    H = Wb * A      * Wb'
    S = Wb * S_pair * Wb'
    return H, S
end

function _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw::Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}, S_pair)
    nbas   = size(T_pair, 1)
    V_pair = _sparse_contract(V_pair_raw, Wv, nbas)
    A = T_pair + V_pair
    H = Wb * A      * Wb'
    S = Wb * S_pair * Wb'
    return H, S, A, V_pair_raw
end

function _construct_integral_rrule_common(Wv, Wb, T_pair, V_pair_raw::Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}, S_pair)
    H, S, A, V_pair_raw = _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw, S_pair)
    function pullback_common(ΔΩ)
        ΔH, ΔS = ΔΩ
        ΔWb_H = ΔH * Wb * A'      + ΔH' * Wb * A
        ΔWb_S = ΔS * Wb * S_pair' + ΔS' * Wb * S_pair
        Wb̄  = real.(ΔWb_H + ΔWb_S)
        ΔA  = real.(Wb' * ΔH * Wb)
        Δy  = vec(ΔA)
        Wv̄  = _sparse_contract_pullback(V_pair_raw, Δy, size(T_pair, 1))
        return NoTangent(), Wv̄, Wb̄, NoTangent(), NoTangent(), NoTangent()
    end
    return (H, S), pullback_common
end

function construct_integral(Wv, Wb, T_pair, V_pair_raw, S_pair)
    return _construct_integral_core(Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function ChainRulesCore.rrule(::typeof(construct_integral),
                              Wv::AbstractVector,
                              Wb::AbstractMatrix,
                              T_pair,
                              V_pair_raw,
                              S_pair)
    return _construct_integral_rrule_common(Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function siteblock(H_AO, S_AO,
                   atom_ranges::Vector{UnitRange{Int}},
                   i::Integer, j::Integer)
    Iia = atom_ranges[i]
    Ija = atom_ranges[j]
    return H_AO[Iia, Ija], S_AO[Iia, Ija]
end

function atom_ao_ranges(bs::BasisSet, Wb)
    NORBs = size(Wb, 1)
    nat   = bs.natoms
    NORB  = Int(NORBs / nat)
    nao_per_atom = NORB * ones(Int, nat)
    ao_ranges = Vector{UnitRange{Int}}(undef, nat)
    offset = 0
    for i in 1:nat
        n = nao_per_atom[i]
        ao_ranges[i] = (offset + 1):(offset + n)
        offset += n
    end
    return NORB, nat, ao_ranges
end

_xyz_key(xyz; digits=10) = (round(xyz[1]; digits), round(xyz[2]; digits), round(xyz[3]; digits))

function shells_per_center(bs)
    shells = bs.basis
    groups = Dict{NTuple{3,Float64}, Vector{eltype(shells)}}()
    order  = NTuple{3,Float64}[]
    for sh in shells
        key = _xyz_key(sh.atom.xyz)
        if !haskey(groups, key)
            groups[key] = eltype(shells)[]
            push!(order, key)
        end
        push!(groups[key], sh)
    end
    return order, groups
end

function shells_for_atom(bs, atom_index::Int)
    order, groups = shells_per_center(bs)
    key = order[atom_index]
    return groups[key], key
end

function n_channels_from_shells(shells; lmax::Int=2)
    counts = zeros(Int, lmax + 1)
    for sh in shells
        L = sh.l
        if 0 <= L <= lmax
            counts[L + 1] += 1
        end
    end
    return Tuple(counts)
end

function lmax_and_nchannels_per_atom(bs, atom_index::Int=1)
    lmax = maximum(sh.l for sh in bs.basis)
    shells, center = shells_for_atom(bs, atom_index)
    n_channels = n_channels_from_shells(shells; lmax=lmax)
    return lmax, n_channels
end

using CUDA
using ChainRulesCore
using ChainRulesCore: NoTangent, unthunk

# ---------------------------------------------------------------------------
# Sparse3CGPU contraction
# V_pair[a,b] += vals[q] * Wv[c]
# ---------------------------------------------------------------------------

function _sparse_contract_gpu_kernel!(
    V_pair,
    ia,
    ib,
    ic,
    vals,
    Wv,
    nnz::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if q <= nnz
        a = ia[q]
        b = ib[q]
        c = ic[q]

        CUDA.@atomic V_pair[a, b] += vals[q] * Wv[c]
    end

    return nothing
end

function _sparse_contract_gpu(
    Vraw::Sparse3CGPU,
    Wv::CUDA.CuArray{<:Real,1},
    nbas::Int,
)
    V_pair = CUDA.zeros(eltype(Wv), nbas, nbas)

    nnz = length(Vraw.vals)
    threads = 256
    blocks = cld(nnz, threads)

    @cuda threads=threads blocks=blocks _sparse_contract_gpu_kernel!(
        V_pair,
        Vraw.ia,
        Vraw.ib,
        Vraw.ic,
        Vraw.vals,
        Wv,
        nnz,
    )

    return V_pair
end

function _construct_integral_core(
    Wv::CUDA.CuArray{<:Real,1},
    Wb::CUDA.CuArray{<:Real,2},
    T_pair::CUDA.CuArray{<:Real,2},
    V_pair_raw::Sparse3CGPU,
    S_pair::CUDA.CuArray{<:Real,2},
)
    nbas = size(T_pair, 1)

    V_pair = _sparse_contract_gpu(V_pair_raw, Wv, nbas)

    A = T_pair .+ V_pair

    H = Wb * A * Wb'
    S = Wb * S_pair * Wb'

    return H, S
end


function _sparse_contract_pullback_gpu_kernel!(
    Wvbar,
    ia,
    ib,
    ic,
    vals,
    ΔA,
    nnz::Int,
)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if q <= nnz
        a = ia[q]
        b = ib[q]
        c = ic[q]

        CUDA.@atomic Wvbar[c] += vals[q] * ΔA[a, b]
    end

    return nothing
end

function _sparse_contract_pullback_gpu(
    Vraw::Sparse3CGPU,
    ΔA::CUDA.CuArray{<:Real,2},
    nwv::Int,
)
    Wvbar = CUDA.zeros(eltype(ΔA), nwv)

    nnz = length(Vraw.vals)
    threads = 256
    blocks = cld(nnz, threads)

    @cuda threads=threads blocks=blocks _sparse_contract_pullback_gpu_kernel!(
        Wvbar,
        Vraw.ia,
        Vraw.ib,
        Vraw.ic,
        Vraw.vals,
        ΔA,
        nnz,
    )

    return Wvbar
end


function ChainRulesCore.rrule(
    ::typeof(construct_integral),
    Wv::CUDA.CuArray{<:Real,1},
    Wb::CUDA.CuArray{<:Real,2},
    T_pair::CUDA.CuArray{<:Real,2},
    V_pair_raw::Sparse3CGPU,
    S_pair::CUDA.CuArray{<:Real,2},
)
    nbas = size(T_pair, 1)

    V_pair = _sparse_contract_gpu(V_pair_raw, Wv, nbas)

    A = T_pair .+ V_pair

    H = Wb * A * Wb'
    S = Wb * S_pair * Wb'

    function pullback_construct_integral_sparsegpu(ΔΩ)
        ΔH, ΔS = unthunk(ΔΩ[1]), unthunk(ΔΩ[2])

        ΔWb_H = ΔH * Wb * A'      + ΔH' * Wb * A
        ΔWb_S = ΔS * Wb * S_pair' + ΔS' * Wb * S_pair

        Wbbar = real.(ΔWb_H .+ ΔWb_S)

        ΔA = real.(Wb' * ΔH * Wb)

        Wvbar = _sparse_contract_pullback_gpu(
            V_pair_raw,
            ΔA,
            length(Wv),
        )

        return (
            NoTangent(),
            Wvbar,
            Wbbar,
            NoTangent(),
            NoTangent(),
            NoTangent(),
        )
    end

    return (H, S), pullback_construct_integral_sparsegpu
end