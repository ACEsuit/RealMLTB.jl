using UnitfulAtomic
using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using RealMLTB
using SparseArrays 

function compute_pair_spherical_AO_integrals(bs, R_bohr)
   molR_list = [bs.atoms...] # Copy original atoms
   for atoms in bs.atoms
      # Shift atoms by R
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
   X = position(sys, :)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first) 
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ] 
   S_i = [ NeighbourLists.AtomsBase.species(sys, i) for i in ii ] 
   S_j = [ NeighbourLists.AtomsBase.species(sys, j) for j in jj ]

   Rcell = zeros(SVector{3, Float64}, first[end] - 1)
   Tlist = []
   Vlist = []
   Slist = []
   idx = 0

   # Iterate over atoms in the system
   for n = 1:length(sys)
      neigs, Rs = NeighbourLists.neigs(nlist, n)
      for m = 1:length(neigs)
         idx += 1
         # Calculate displacement vector R in Bohr
         Rcell[idx] = (Rs[m] - UnitfulAtomic.ustrip.(X[neigs[m]] - X[n])) .* RealMLTB.ang2bohr
         if norm(Rcell[idx]) == 0
            T_pair, V_pair, S_pair = compute_spherical_AO_integrals(bs)
         else
            T_pair, V_pair, S_pair = compute_pair_spherical_AO_integrals(bs, Rcell[idx])
         end
         
         push!(Tlist, T_pair)
         push!(Vlist, V_pair)
         push!(Slist, S_pair)
      end
   end
   
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj, rij = rij, t = t, v = v, s = s) for (𝐫, si, sj, rij, t, v, s) in zip(R_ij, S_i, S_j, Rcell, Tlist, Vlist, Slist) ]
   
   klist = RealMLTB.physical_k.(Ref(sys), klist)
   
   T, V, S = compute_spherical_AO_integrals(bs)
   klist = RealMLTB.physical_k.(Ref(sys), klist)
   G = EquivariantTensors.ETGraph(ii, jj; edge_data = X_ij, graph_data = (klist = klist, t = T, v = V, s = S))
   @assert G.first == first
   return G 
end







   