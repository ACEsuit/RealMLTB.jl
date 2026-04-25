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
