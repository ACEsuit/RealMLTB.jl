using UnitfulAtomic
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using NeighbourLists
using EquivariantTensors
using AtomsBase
using Base.Threads

function compute_pair_spherical_AO_integrals(bs::BasisSet, R_bohr)
    mol_pair = Molecules.Atom[]
    append!(mol_pair, bs.atoms)

    for atom in bs.atoms
        atom_R = Molecules.Atom(
            atom.Z,
            atom.mass,
            atom.xyz + R_bohr ./ RealMLTB.ang2bohr
        )
        push!(mol_pair, atom_R)
    end

    bs_pair_cart = BasisSet(bs.name, mol_pair; spherical=false, lib=:acsint)
    T_sph, V_sph, S_sph = compute_spherical_AO_integrals(bs_pair_cart)

    nbas   = div(size(T_sph, 1), 2)
    T_pair = T_sph[1:nbas, nbas+1:2*nbas]
    S_pair = S_sph[1:nbas, nbas+1:2*nbas]
    V_pair = V_sph[1:nbas, 1:nbas, nbas+1:2*nbas]

    return T_pair, V_pair, S_pair
end

function compute_pair_spherical_AO_integrals_sparse(bs::BasisSet, R_bohr)
    mol_pair = Molecules.Atom[]
    append!(mol_pair, bs.atoms)

    for atom in bs.atoms
        atom_R = Molecules.Atom(
            atom.Z,
            atom.mass,
            atom.xyz + R_bohr ./ RealMLTB.ang2bohr
        )
        push!(mol_pair, atom_R)
    end

    bs_pair_cart = BasisSet(bs.name, mol_pair; spherical=false, lib=:acsint)
    T_sph, V_sph, S_sph = compute_spherical_AO_integrals_sparse(bs_pair_cart)

    nbas   = div(size(T_sph, 1), 2)
    T_pair = T_sph[1:nbas, nbas+1:2*nbas]
    S_pair = S_sph[1:nbas, nbas+1:2*nbas]

    V_indices, V_vals = V_sph
    pair_indices = NTuple{3,Int}[]
    pair_vals    = Float64[]
    for (idx, v) in zip(V_indices, V_vals)
        a, b, c = idx
        if a <= nbas && b <= nbas && nbas < c <= 2 * nbas
            push!(pair_indices, (a, b, c - nbas))
            push!(pair_vals, v)
        end
    end
    V_pair = (pair_indices, pair_vals)

    return T_pair, V_pair, S_pair
end

function nlist2graph(bs::BasisSet, nlist, sys, klist)
    bs_use =
        if bs.basis[1] isa SphericalShell
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

    Tlist      = Vector{Matrix{Float64}}(undef, npairs)
    Vlist      = Vector{Tuple{Vector{NTuple{3,Int}}, Vector{Float64}}}(undef, npairs)
    Slist      = Vector{Matrix{Float64}}(undef, npairs)
    Rcell_list = Vector{SVector{3,Float64}}(undef, npairs)

    nthreads = Threads.nthreads()
    println("Computing $npairs pair integrals on $nthreads threads ...")
    flush(stdout)

    bs_per_thread = [BasisSet(bs_use.name, bs_use.atoms; spherical=true, lib=:acsint)
                     for _ in 1:nthreads]

    t0 = time()

    Threads.@threads for e in 1:npairs
        tid    = Threads.threadid()
        bs_loc = bs_per_thread[tid]

        i = ii[e]
        j = jj[e]

        R_vec  = R_cart_list[e]
        R_cell = (R_vec - UnitfulAtomic.ustrip.(X[j] - X[i])) .* RealMLTB.ang2bohr
        R_cell = SVector{3,Float64}(R_cell)

        if norm(R_cell) < 1e-10
            T_pair, V_pair, S_pair = compute_spherical_AO_integrals_sparse(bs_loc)
        else
            T_pair, V_pair, S_pair = compute_pair_spherical_AO_integrals_sparse(bs_loc, R_cell)
        end

        Rcell_list[e] = R_cell
        Tlist[e]      = T_pair
        Vlist[e]      = V_pair
        Slist[e]      = S_pair
    end

    elapsed = round(time() - t0; digits=1)
    println("  Done in $(elapsed)s. Building ETGraph ...")
    flush(stdout)

    edge_data = [
        (
            𝐫      = R_cart_list[e],
            z0     = z0_list[e],
            z1     = z1_list[e],
            R_cell = Rcell_list[e],
            T_pair = Tlist[e],
            V_pair = Vlist[e],
            S_pair = Slist[e],
        )
        for e in 1:npairs
    ]

    T_on, V_on, S_on = compute_spherical_AO_integrals_sparse(bs_use)
    klist_phys = physical_k.(Ref(sys), klist)

    G = EquivariantTensors.ETGraph(
        ii, jj;
        edge_data  = edge_data,
        graph_data = (
            klist = klist_phys,
            T_on  = T_on,
            V_on  = V_on,
            S_on  = S_on,
        )
    )

    @assert G.first == first
    return G
end

function physical_k(sys, k_reduced::AbstractVector{<:Real})
    @assert length(k_reduced) == 3
    lattice = get_lattice(sys)
    G_basis = reciprocal_lattice(lattice)
    return G_basis * SVector{3,Float64}(k_reduced)
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
