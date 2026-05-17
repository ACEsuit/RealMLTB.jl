using UnitfulAtomic
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using NeighbourLists
using EquivariantTensors
using AtomsBase
using Base.Threads

function compute_pair_spherical_AO_integrals(bs_in::BasisSet, R_bohr)
    bs, bs_R, C, d_sph, S_pair_sph, T_pair_sph = one_body_pair_spherical_AO_integrals(bs_in, R_bohr)

    N_cart = bs_R.nbas
    V_pair_cart = EPI_1e3c(bs, bs_R)

    N_sph = size(C, 2)

    V_tmp1 = reshape(C' * reshape(V_pair_cart, N_cart, :), N_sph, N_cart, N_cart)
    V_p1   = permutedims(V_tmp1, (2, 1, 3))
    V_tmp2 = reshape(C' * reshape(V_p1, N_cart, :), N_sph, N_sph, N_cart)
    V_p2   = permutedims(V_tmp2, (3, 2, 1))
    V_tmp3 = reshape(C' * reshape(V_p2, N_cart, :), N_sph, N_sph, N_sph)

    V_pair_sph_raw = permutedims(V_tmp3, (2, 3, 1))

    V_pair_sph = zeros(N_sph, N_sph, N_sph)
    @inbounds for k in 1:N_sph, j in 1:N_sph, i in 1:N_sph
        V_pair_sph[i, j, k] = V_pair_sph_raw[i, j, k] * d_sph[i] * d_sph[j] * d_sph[k]
    end

    return T_pair_sph, V_pair_sph, S_pair_sph
end

function compute_pair_spherical_AO_integrals_sparse(bs_in::BasisSet, R_bohr;
                                                     cutoff=1e-12, ntasks=1)
    bs, bs_R, C, d_sph, S_pair_sph, T_pair_sph =
        one_body_pair_spherical_AO_integrals(bs_in, R_bohr)

    cart_cutoff = cutoff * 1e-3
    indexes, vals = sparseEPI_1e3c(bs, bs_R, cart_cutoff; ntasks=ntasks)

    N_sph  = size(C, 2)
    N_cart = bs_R.nbas
    V_pair_cart = zeros(Float64, N_cart, N_cart, N_cart)
    @inbounds for n in eachindex(indexes)
        i, j, k = indexes[n]
        V_pair_cart[i, j, k] = vals[n]
    end

    V_tmp1 = reshape(C' * reshape(V_pair_cart, N_cart, :), N_sph, N_cart, N_cart)
    V_p1   = permutedims(V_tmp1, (2, 1, 3))
    V_tmp2 = reshape(C' * reshape(V_p1, N_cart, :), N_sph, N_sph, N_cart)
    V_p2   = permutedims(V_tmp2, (3, 2, 1))
    V_tmp3 = reshape(C' * reshape(V_p2, N_cart, :), N_sph, N_sph, N_sph)
    
    V_pair_sph_raw = permutedims(V_tmp3, (2, 3, 1))

    out_indices = NTuple{3,Int}[]
    out_vals    = Float64[]
    @inbounds for c in 1:N_sph, b in 1:N_sph, a in 1:N_sph
        v = V_pair_sph_raw[a, b, c] * d_sph[a] * d_sph[b] * d_sph[c]
        if abs(v) > cutoff
            push!(out_indices, (a, b, c))
            push!(out_vals, v)
        end
    end

    return T_pair_sph, (out_indices, out_vals), S_pair_sph
end

function one_body_pair_spherical_AO_integrals(bs_in::BasisSet, R_bohr)
    bs =
        if (bs_in.basis[1] isa SphericalShell) || (bs_in.lib != :acsint)
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    mol_R = similar(bs.atoms)
    for (i, atoms) in enumerate(bs.atoms)
        mol_R[i] = Molecules.Atom(atoms.Z, atoms.mass, atoms.xyz + R_bohr ./ RealMLTB.ang2bohr)
    end
    bs_R = BasisSet(bs.name, mol_R; spherical=false, lib=:acsint)

    C = RealMLTB.cart2sph_coeff(bs)
    S = overlap(bs)

    d_sph = 1.0 ./ sqrt.(diag(C' * S * C))
    replace!(d_sph, Inf => 0.0, NaN => 0.0)

    S_pair_cart = overlap(bs, bs_R)
    T_pair_cart = kinetic(bs, bs_R)

    S_pair_sph_raw = C' * S_pair_cart * C
    T_pair_sph_raw = C' * T_pair_cart * C

    S_pair_sph = S_pair_sph_raw .* (d_sph * d_sph')
    T_pair_sph = T_pair_sph_raw .* (d_sph * d_sph')

    return (; bs, bs_R, C, d_sph, S_pair_sph, T_pair_sph)
end

function _rcell_key(R_cell::SVector{3,Float64}; digits=8)
    return (round(R_cell[1]; digits), round(R_cell[2]; digits), round(R_cell[3]; digits))
end

function nlist2graph(bs::BasisSet, nlist, sys, klist)
    bs_use = if bs.basis[1] isa SphericalShell
        bs
    else
        BasisSet(bs.name, bs.atoms; spherical=true, lib=:acsint)
    end
 
    X     = position(sys, :)
    ii    = copy(nlist.i)
    jj    = copy(nlist.j)
    first = copy(nlist.first)
 
    npairs      = length(ii)
    R_cart_list = [NeighbourLists._getR(nlist, e) for e in 1:npairs]
    z0_list     = [NeighbourLists.AtomsBase.species(sys, i) for i in ii]
    z1_list     = [NeighbourLists.AtomsBase.species(sys, j) for j in jj]
 
    Rcell_list = Vector{SVector{3,Float64}}(undef, npairs)
    for e in 1:npairs
        R_vec  = R_cart_list[e]
        R_cell = (R_vec - UnitfulAtomic.ustrip.(X[jj[e]] - X[ii[e]])) .* RealMLTB.ang2bohr
        Rcell_list[e] = SVector{3,Float64}(R_cell)
    end
 
    rcell_to_idx = Dict{NTuple{3,Float64}, Int}()
    unique_Rcells = SVector{3,Float64}[]
 
    integral_idx_per_edge = Vector{Int}(undef, npairs)
    for e in 1:npairs
        key = _rcell_key(Rcell_list[e])
        if !haskey(rcell_to_idx, key)
            push!(unique_Rcells, Rcell_list[e])
            rcell_to_idx[key] = length(unique_Rcells)
        end
        integral_idx_per_edge[e] = rcell_to_idx[key]
    end
 
    n_unique = length(unique_Rcells)
    println("$npairs edges → $n_unique unique R_cell vectors ($(round(100*(1 - n_unique/npairs); digits=1))% dedup)")
    flush(stdout)
 
    Tlist_u = Vector{Matrix{Float64}}(undef, n_unique)
    Vlist_u = Vector{Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}}(undef, n_unique)
    Slist_u = Vector{Matrix{Float64}}(undef, n_unique)
 
    println("Computing $n_unique unique pair integrals ...")
    flush(stdout)
 
    Threads.@threads for u in 1:n_unique
        R_cell = unique_Rcells[u]
        if norm(R_cell) < 1e-10
            T_pair, V_pair, S_pair = compute_spherical_AO_integrals_sparse(bs_use; ntasks=1)
        else
            T_pair, V_pair, S_pair = compute_pair_spherical_AO_integrals_sparse(bs_use, R_cell; ntasks=1)
        end
        Tlist_u[u] = T_pair
        Vlist_u[u] = V_pair
        Slist_u[u] = S_pair
    end
 
    integral_table = [
        (R_cell = unique_Rcells[u], T_pair = Tlist_u[u], V_pair = Vlist_u[u], S_pair = Slist_u[u])
        for u in 1:n_unique
    ]
 
    edge_data = [
        (
            𝐫            = R_cart_list[e],
            z0           = z0_list[e],
            z1           = z1_list[e],
            R_cell       = Rcell_list[e],
            integral_idx = integral_idx_per_edge[e],
        )
        for e in 1:npairs
    ]
 
    T_on, V_on, S_on = compute_spherical_AO_integrals_sparse(bs_use)
    klist_phys = physical_k.(Ref(sys), klist)
 
    G = EquivariantTensors.ETGraph(
        ii, jj;
        edge_data  = edge_data,
        graph_data = (
            klist           = klist_phys,
            T_on            = T_on,
            V_on            = V_on,
            S_on            = S_on,
            integral_table  = integral_table,
        ),
    )
 
    @assert G.first == first
    return G
end


function physical_k(sys, k_reduced::SVector{3,<:Real})
    lattice = get_lattice(sys)
    G_basis = reciprocal_lattice(lattice)
    return G_basis * k_reduced
end

function get_lattice(sys)
    return UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
end

function fractional_coordinates(sys)
    lattice = UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
    positions = map(sys) do atom
        coordinate = zeros(Float64, 3)
        coordinate[1:3] = lattice[1:3, 1:3] \ Float64.(austrip.(position(atom)))
        SVector{3,Float64}(coordinate)
    end
    return positions
end

function reciprocal_lattice(lattice)
    return 2π * SMatrix{3,3}(inv(lattice)')
end


using CUDA
using ChainRulesCore
using ChainRulesCore: NoTangent, unthunk

# ---------------------------------------------------------------------------
# Sparse3CGPU
#
# Stores a sparse 3-center tensor V[a,b,c] as separated GPU arrays:
#   ia[n], ib[n], ic[n], vals[n]
# so that
#   V_pair[a,b] += vals[n] * Wv[c]
# can be evaluated by a CUDA kernel.
# ---------------------------------------------------------------------------
struct Sparse3CGPU{TI, TV}
    ia::TI
    ib::TI
    ic::TI
    vals::TV
end

function _split_sparse3c_indices_kernel!(ia, ib, ic, idxs, nnz::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if q <= nnz
        t = idxs[q]
        ia[q] = Int32(t[1])
        ib[q] = Int32(t[2])
        ic[q] = Int32(t[3])
    end

    return nothing
end

"""
    sparse3c_to_gpu(V_pair_raw; T=Float32)

Convert a sparse 3-center tensor `(indices, vals)` to `Sparse3CGPU`.

Works for both:
  - CPU tuple: `(Vector{NTuple{3,Int}}, Vector{Float64})`
  - GPU tuple: `(CuArray{NTuple{3,Int},1}, CuArray{Float32,1})`

Use this function instead of calling `Sparse3CGPU(V_pair_raw)` directly.
"""
function sparse3c_to_gpu(V_pair_raw::Tuple; T = Float32)
    idxs, vals = V_pair_raw

    if idxs isa CUDA.CuArray
        return sparse3c_to_gpu_from_gpu_tuple(V_pair_raw; T = T)
    else
        return sparse3c_to_gpu_from_cpu_tuple(V_pair_raw; T = T)
    end
end

function sparse3c_to_gpu_from_cpu_tuple(V_pair_raw::Tuple; T = Float32)
    idxs, vals = V_pair_raw
    nnz = length(idxs)

    ia = Vector{Int32}(undef, nnz)
    ib = Vector{Int32}(undef, nnz)
    ic = Vector{Int32}(undef, nnz)

    @inbounds for n in 1:nnz
        a, b, c = idxs[n]
        ia[n] = Int32(a)
        ib[n] = Int32(b)
        ic[n] = Int32(c)
    end

    return Sparse3CGPU(
        CUDA.cu(ia),
        CUDA.cu(ib),
        CUDA.cu(ic),
        CUDA.cu(T.(vals)),
    )
end

function sparse3c_to_gpu_from_gpu_tuple(V_pair_raw::Tuple; T = Float32)
    idxs, vals = V_pair_raw

    @assert idxs isa CUDA.CuArray "idxs must be a CuArray"
    @assert vals isa CUDA.CuArray "vals must be a CuArray"
    @assert ndims(idxs) == 1 "idxs must be a GPU vector"
    @assert ndims(vals) == 1 "vals must be a GPU vector"

    nnz = length(idxs)

    ia = CUDA.zeros(Int32, nnz)
    ib = CUDA.zeros(Int32, nnz)
    ic = CUDA.zeros(Int32, nnz)

    threads = 256
    blocks = cld(nnz, threads)

    @cuda threads=threads blocks=blocks _split_sparse3c_indices_kernel!(
        ia, ib, ic, idxs, nnz
    )

    vals_gpu = eltype(vals) === T ? vals : T.(vals)

    return Sparse3CGPU(ia, ib, ic, vals_gpu)
end

# ---------------------------------------------------------------------------
# ThreeCenterGPUCache
#
# Build this from CPU G once, and keep using it in the GPU three-center layer.
# G_gpu = dev(ET.float32(G)) should still be used for ACE only.
# ---------------------------------------------------------------------------
struct ThreeCenterGPUCache{TI, TR, TM, TS}
    ii::TI                  # CPU Vector{Int}; small edge metadata
    jj::TI                  # CPU Vector{Int}; small edge metadata
    Rcell::TR               # CuArray{T,2}, 3 × nedges
    klist::TR               # CuArray{T,2}, 3 × nk
    nat::Int
    NORB::Int
    T_on::TM                # CuMatrix
    V_on::TS                # Sparse3CGPU
    S_on::TM                # CuMatrix
    T_pair::Vector{TM}      # per-edge CuMatrix
    V_pair::Vector{TS}      # per-edge Sparse3CGPU
    S_pair::Vector{TM}      # per-edge CuMatrix
end

# Local safe accessor, avoids depending on _get_ii_jj from another file.
@inline function _gpu_get_ii_jj(G)
    fn = fieldnames(typeof(G))
    if :ii ∈ fn && :jj ∈ fn
        return G.ii, G.jj
    elseif :i ∈ fn && :j ∈ fn
        return G.i, G.j
    else
        error("Cannot determine ii/jj from ETGraph fieldnames: $(fn)")
    end
end

function _gpu_hcat_svectors(xs)
    isempty(xs) && return zeros(Float64, 3, 0)
    return reduce(hcat, xs)
end

"""
    make_3center_gpu_cache(G, NORB; T=Float32)

Build a GPU cache for the three-center part from the original CPU graph `G`.

Do NOT pass `G_gpu = dev(ET.float32(G))` here.  Pass the original CPU `G`.
"""
function make_3center_gpu_cache(G, NORB::Int; T = Float32)
    ii, jj = _gpu_get_ii_jj(G)

    nat = length(G.first) - 1
    edge_data = G.edge_data
    integral_table = G.graph_data.integral_table

    # Keep these on CPU for simple dispatch/loop control. They are metadata.
    ii_cpu = Int.(collect(ii))
    jj_cpu = Int.(collect(jj))

    Rcell_cpu = _gpu_hcat_svectors([ed.R_cell for ed in edge_data])
    klist_cpu = _gpu_hcat_svectors(G.graph_data.klist)

    Rcell_gpu = CUDA.cu(T.(Rcell_cpu))
    klist_gpu = CUDA.cu(T.(klist_cpu))

    T_on_gpu = CUDA.cu(T.(G.graph_data.T_on))
    S_on_gpu = CUDA.cu(T.(G.graph_data.S_on))
    V_on_gpu = sparse3c_to_gpu(G.graph_data.V_on; T = T)

    T_pair_gpu = Vector{typeof(T_on_gpu)}(undef, length(edge_data))
    S_pair_gpu = Vector{typeof(S_on_gpu)}(undef, length(edge_data))
    V_pair_gpu = Vector{typeof(V_on_gpu)}(undef, length(edge_data))

    for e in eachindex(edge_data)
        ed = edge_data[e]
        tbl = integral_table[ed.integral_idx]

        T_pair_gpu[e] = CUDA.cu(T.(tbl.T_pair))
        S_pair_gpu[e] = CUDA.cu(T.(tbl.S_pair))
        V_pair_gpu[e] = sparse3c_to_gpu(tbl.V_pair; T = T)
    end

    return ThreeCenterGPUCache(
        ii_cpu,
        jj_cpu,
        Rcell_gpu,
        klist_gpu,
        nat,
        NORB,
        T_on_gpu,
        V_on_gpu,
        S_on_gpu,
        T_pair_gpu,
        V_pair_gpu,
        S_pair_gpu,
    )
end

# ---------------------------------------------------------------------------
# GPU sparse contraction: V_pair[a,b] += vals[n] * Wv[c]
# ---------------------------------------------------------------------------
function _sparse_contract_gpu_kernel!(V_pair, ia, ib, ic, vals, Wv, nnz::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if q <= nnz
        a = Int(ia[q])
        b = Int(ib[q])
        c = Int(ic[q])

        CUDA.@atomic V_pair[a, b] += vals[q] * Wv[c]
    end

    return nothing
end

function _sparse_contract_gpu(Vraw::Sparse3CGPU, Wv::CUDA.CuArray{<:Real,1}, nbas::Int)
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

function _sparse_contract_pullback_gpu_kernel!(Wvbar, ia, ib, ic, vals, ΔA, nnz::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if q <= nnz
        a = Int(ia[q])
        b = Int(ib[q])
        c = Int(ic[q])

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

@inline function _named_field(field::Symbol, value)
    return NamedTuple{(field,)}((value,))
end

function _named_field(name::Symbol, value)
    return NamedTuple{(name,)}((value,))
end

function _replace_graph_data_preserve_storage(G_gpu, new_graph_data)
    fns = fieldnames(typeof(G_gpu))

    @assert :graph_data in fns
    @assert :first in fns

    vals = ntuple(length(fns)) do k
        f = fns[k]
        if f === :graph_data
            new_graph_data
        else
            getfield(G_gpu, f)
        end
    end

    # Bypass ETGraph(ii,jj; ...) constructor.
    # This preserves first/ii/jj/edge_data exactly as in G_gpu.
    return (Base.typename(typeof(G_gpu)).wrapper)(vals...)
end


function _replace_edge_data_preserve_storage(G, new_edge_data)
    fns = fieldnames(typeof(G))

    @assert :edge_data in fns

    vals = ntuple(length(fns)) do k
        f = fns[k]
        if f === :edge_data
            new_edge_data
        else
            getfield(G, f)
        end
    end

    return (Base.typename(typeof(G)).wrapper)(vals...)
end

function strip_3center_edge_data(G)
    edge_data = [
        EquivariantTensors.PState(
            R_cell = e.R_cell,
            𝐫 = e.𝐫,
            z0 = e.z0,
            z1 = e.z1,
        )
        for e in G.edge_data
    ]

    return _replace_edge_data_preserve_storage(G, edge_data)
end

function strip_3center_graph_data(G)
    return _replace_graph_data_preserve_storage(G, (; klist = G.graph_data.klist))
end

function strip_3center_graph_for_gpu(G)
    return strip_3center_graph_data(strip_3center_edge_data(G))
end

"""
    gpu_3center_graph(dev, G, NORB; T=Float32)

Create the GPU graph used by the three-center Lux models.  The returned graph
keeps only the ACE edge payload and k-point graph data on the ETGraph itself,
then attaches a `ThreeCenterGPUCache` containing all three-center integral data
that the GPU kernels need.  Build the cache from the original CPU graph once;
subsequent forwards and pullbacks stay on GPU arrays.
"""
function gpu_3center_graph(dev, G, NORB::Int; T = Float32)
    cache = make_3center_gpu_cache(G, NORB; T = T)
    G_ace = strip_3center_graph_for_gpu(G)
    G_gpu = dev(EquivariantTensors.float32(G_ace))
    return attach_3center_gpu_cache(G_gpu, cache)
end

function attach_3center_gpu_cache(G_gpu, cache; field::Symbol = :threecenter_cache)
    new_graph_data = (
    klist = G_gpu.graph_data.klist,
    threecenter_cache = cache,)

    G1 = _replace_graph_data_preserve_storage(G_gpu, new_graph_data)

    @assert getfield(G1, :first) === getfield(G_gpu, :first)
    @assert typeof(getfield(G1, :first)) == typeof(getfield(G_gpu, :first))

    return G1
end

function with_3center_gpu_cache(G_cpu, G_gpu, NORB; T = Float32, field::Symbol = :threecenter_cache)
    cache = make_3center_gpu_cache(G_cpu, NORB; T = T)
    return attach_3center_gpu_cache(G_gpu, cache; field = field)
end

@inline function get_3center_gpu_cache(G; field::Symbol = :threecenter_cache)
    return getproperty(G.graph_data, field)
end