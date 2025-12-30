using UnitfulAtomic
using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using RealMLTB
using SparseArrays
using StaticArrays
using NeighbourLists
using EquivariantTensors

function compute_pair_spherical_AO_integrals(bs, R_bohr)
    molR_list = [bs.atoms...]
    for atoms in bs.atoms
        atoms_R = Molecules.Atom(atoms.Z, atoms.mass, atoms.xyz + R_bohr ./ RealMLTB.ang2bohr)
        push!(molR_list, atoms_R)
    end

    # Explicitly inherit parameters to ensure consistency
    bs_cart = BasisSet(bs.name, molR_list; spherical=false, lib=:acsint)

    T_sph, V_sph, S_sph = compute_spherical_AO_integrals(bs_cart)

    nbas = div(size(T_sph, 1), 2)
    S_pair_block = S_sph[1:nbas, nbas+1:2*nbas]
    T_pair_block = T_sph[1:nbas, nbas+1:2*nbas]
    V_pair_block = V_sph[1:nbas, 1:nbas, nbas+1:2*nbas]

    return T_pair_block, V_pair_block, S_pair_block
end

function nlist2graph(bs, nlist, sys, klist)
    # Always use a Cartesian AO basis for the integral routines here
    bs_cart = (bs.basis[1] isa SphericalShell) ? BasisSet(bs.name, bs.atoms; spherical=false, lib=:acsint) : bs

    X = position(sys, :)
    ii = copy(nlist.i)
    jj = copy(nlist.j)
    first = copy(nlist.first)

    R_ij = [NeighbourLists._getR(nlist, n) for n = 1:length(ii)]
    S_i  = [NeighbourLists.AtomsBase.species(sys, i) for i in ii]
    S_j  = [NeighbourLists.AtomsBase.species(sys, j) for j in jj]

    # This length should match the number of directed neighbor entries
    npairs = first[end] - 1

    Rcell = zeros(SVector{3, Float64}, npairs)
    Tlist = Vector{Any}(undef, npairs)
    Vlist = Vector{Any}(undef, npairs)
    Slist = Vector{Any}(undef, npairs)

    # Progress counter based on neighbor enumeration (should match npairs)
    total_pairs = 0
    for n = 1:length(sys)
        neigs, _ = NeighbourLists.neigs(nlist, n)
        total_pairs += length(neigs)
    end

    # If these mismatch, progress printing still works, but indexing must follow npairs.
    # Typically they should be identical.
    # @assert total_pairs == npairs

    idx = 0
    last_pct = -1
    t0 = time()

    for n = 1:length(sys)
        neigs, Rs = NeighbourLists.neigs(nlist, n)
        for m = 1:length(neigs)
            idx += 1
            @inbounds begin
                # Progress: print when percentage changes
                pct = Int(floor(100 * idx / max(total_pairs, 1)))
                if pct != last_pct
                    elapsed = time() - t0
                    println("progress: $(pct)%  ($idx / $total_pairs)  elapsed=$(round(elapsed; digits=1))s")
                    flush(stdout)
                    last_pct = pct
                end

                Rcell[idx] = (Rs[m] - UnitfulAtomic.ustrip.(X[neigs[m]] - X[n])) .* RealMLTB.ang2bohr

                if norm(Rcell[idx]) < 1e-10
                    T_pair, V_pair, S_pair = compute_spherical_AO_integrals(bs_cart)
                else
                    T_pair, V_pair, S_pair = compute_pair_spherical_AO_integrals(bs_cart, Rcell[idx])
                end

                Tlist[idx] = T_pair
                Vlist[idx] = V_pair
                Slist[idx] = S_pair
            end
        end
    end

    X_ij = [
        (𝐫 = 𝐫, s0 = si, s1 = sj, rij = rij, t = t, v = v, s = s)
        for (𝐫, si, sj, rij, t, v, s) in zip(R_ij, S_i, S_j, Rcell, Tlist, Vlist, Slist)
    ]

    # On-site graph-level integrals for the (spherical) basis `bs`
    T, V, S = compute_spherical_AO_integrals(bs)

    klist = RealMLTB.physical_k.(Ref(sys), klist)
    G = EquivariantTensors.ETGraph(ii, jj; edge_data = X_ij,
                                   graph_data = (klist = klist, t = T, v = V, s = S))

    @assert G.first == first
    return G
end
