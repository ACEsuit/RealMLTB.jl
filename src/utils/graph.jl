# Graph Construction Helpers 
# Utilities for handling lattice geometry (coordinate/k-space transformations) 
# and building the `ETGraph` structure populated with neighbor lists and precomputed quantum integrals.

using NeighbourLists
using EquivariantTensors
using UnitfulAtomic
using AtomsBase

function get_lattice(sys)
   return UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
end

function fractional_coordinates(sys)
   lattice = UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
   positions = map(sys) do atom
      coordinate = zeros(Float64, 3)
      coordinate[1:3] = lattice[1:3, 1:3] \ Float64.(austrip.(position(atom)))
      SVector{3, Float64}(coordinate)
   end
   return positions
end

function reciprocal_lattice(lattice)
   return 2π * SMatrix{3,3}(inv(lattice)')
end

function physical_k(sys, k_reduced::SVector{3, <:Real})
   lattice = get_lattice(sys)
   G_basis = reciprocal_lattice(lattice)
   return G_basis * k_reduced
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
   for n = 1:length(sys)
      neigs, Rs = NeighbourLists.neigs(nlist, n)
      for m = 1:length(neigs)
         idx += 1
         Rcell[idx] = (Rs[m] - UnitfulAtomic.ustrip.(X[neigs[m]] - X[n])) .* ang2bohr
         if norm(Rcell[idx]) == 0
            T_pair, V_pair, S_pair = compute_AO_integrals(bs)
         else
            T_pair, V_pair, S_pair = compute_pair_AO_integrals(bs, Rcell[idx])
         end
         push!(Tlist, T_pair)
         push!(Vlist, V_pair)
         push!(Slist, S_pair)
      end
   end
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj, rij = rij, t = t, v = v, s = s) for (𝐫, si, sj, rij, t, v, s) in zip(R_ij, S_i, S_j, Rcell, Tlist, Vlist, Slist) ]
   
   T, V, S = compute_AO_integrals(bs)

   klist = physical_k.(Ref(sys), klist)
   G = EquivariantTensors.ETGraph(ii, jj; edge_data = X_ij, graph_data = (klist = klist, t = T, v = V, s = S))
   @assert G.first == first

   return G 
end 