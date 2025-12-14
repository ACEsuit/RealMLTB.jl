function _construct_integral_core(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
   nbas = bs.nbas
   M      = reshape(V_pair_raw, nbas^2, nbas)
   V_pair = reshape(M * Wv, nbas, nbas)

   A = T_pair + V_pair
   H = Wb * A      * Wb'
   S = Wb * S_pair * Wb'
   return H, S
end

function construct_integral(bs, Wv, Wb)
   T_pair, V_pair_raw, S_pair = compute_AO_integrals(bs)
   return _construct_integral_core(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function construct_integral(bs, Wv, Wb, Rij)
   T_pair, V_pair_raw, S_pair = compute_pair_AO_integrals(bs, Rij)
   return _construct_integral_core(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function construct_integral(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
   return _construct_integral_core(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function _construct_integral_primal(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
   nbas   = bs.nbas
   M      = reshape(V_pair_raw, nbas^2, nbas)
   V_pair = reshape(M * Wv, nbas, nbas)

   A = T_pair + V_pair
   H = Wb * A      * Wb'
   S = Wb * S_pair * Wb'
   return H, S, A, M
end

function _construct_integral_rrule_common(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
   H, S, A, M = _construct_integral_primal(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)

   function pullback_common(ΔΩ)
      ΔH, ΔS = ΔΩ

      # H = Wb * A * Wb'
      ΔWb_H = ΔH * Wb * A'      + ΔH' * Wb * A

      # S = Wb * S_pair * Wb'
      ΔWb_S = ΔS * Wb * S_pair' + ΔS' * Wb * S_pair

      Wb̄ = ΔWb_H + ΔWb_S

      # dL/dA = Wb' * ΔH * Wb
      ΔA = Wb' * ΔH * Wb

      # A = -0.5*T_pair + V_pair
      ΔV = ΔA

      # V_pair = reshape(M * Wv, nbas, nbas)
      # y = M * Wv, V = reshape(y, nbas, nbas)
      # ⇒ ∂L/∂Wv = M' * vec(ΔV)
      Δy  = vec(ΔV)
      Wv̄ = M' * Δy

      return NoTangent(),  # 
               NoTangent(),  # bs
               Wv̄,           # Wv
               Wb̄,           # Wb
               NoTangent(),  # T_pair
               NoTangent(),  # V_pair_raw
               NoTangent()   # S_pair
   end

   return (H, S), pullback_common
end

function ChainRulesCore.rrule(::typeof(construct_integral),
               bs,
               Wv::AbstractVector,
               Wb::AbstractMatrix)

   T_pair, V_pair_raw, S_pair = compute_AO_integrals(bs)

   (H_S, pullback_common) =
        _construct_integral_rrule_common(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)

   function pullback(ΔΩ)
      _f̄, bs̄, Wv̄, Wb̄, T̄, V̄, S̄ = pullback_common(ΔΩ)
      return NoTangent(), bs̄, Wv̄, Wb̄
   end

   return H_S, pullback
end

function ChainRulesCore.rrule(::typeof(construct_integral),
               bs,
               Wv::AbstractVector,
               Wb::AbstractMatrix,
               T_pair,
               V_pair_raw,
               S_pair)

    return _construct_integral_rrule_common(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function ChainRulesCore.rrule(::typeof(construct_integral),
               bs,
               Wv::AbstractVector,
               Wb::AbstractMatrix,
               Rij)

   T_pair, V_pair_raw, S_pair = compute_pair_AO_integrals(bs, Rij)

   (H_S, pullback_common) =
      _construct_integral_rrule_common(bs, Wv, Wb, T_pair, V_pair_raw, S_pair)

   function pullback(ΔΩ)
      _f̄, bs̄, Wv̄, Wb̄, T̄, V̄, S̄ = pullback_common(ΔΩ)
      return NoTangent(), bs̄, Wv̄, Wb̄, NoTangent()
   end

   return H_S, pullback
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
   NORB  = Int(NORBs/nat)
   nao_per_atom = NORB * ones(Int, nat)

   ao_ranges = Vector{UnitRange{Int}}(undef, nat)
   offset = 0
   for i in 1:nat
      n = nao_per_atom[i]
      ao_ranges[i] = (offset+1):(offset+n)
      offset += n
   end
   return NORB, nat, ao_ranges
end

function atom_ao_ranges(bs::BasisSet)
   nao_per_atom = bs.basis_per_atom
   nat   = bs.natoms

   ao_ranges = Vector{UnitRange{Int}}(undef, nat)
   offset = 0
   for i in 1:nat
      n = nao_per_atom[i]
      ao_ranges[i] = (offset+1):(offset+n)
      offset += n
   end
   return nao_per_atom[1], nat, ao_ranges
end

function generate_λ(H::Matrix{T}, S::Matrix{TS}) where {T, TS}
   L = cholesky(S).L           # S = LLᵀ
   A = L \ (H / L')            # i.e., A = L⁻¹ H L⁻ᵀ
   AA = (A + A')/2
   λ = GenericLinearAlgebra.eigen(AA).values
   return λ
end

function generate_λ(Hlist::AbstractVector{TH}, Slist::AbstractVector{TS}) where {TH, TS}
   λlist = [generate_λ(Hlist[i], Slist[i]) for i in 1:length(Hlist)]
   return λlist
end


# helpers for graph

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
         T_pair, V_pair, S_pair = compute_pair_AO_integrals(bs, Rcell[idx])
         push!(Tlist, T_pair)
         push!(Vlist, V_pair)
         push!(Slist, S_pair)
      end
   end
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj, rij = rij, t = t, v = v, s = s) for (𝐫, si, sj, rij, t, v, s) in zip(R_ij, S_i, S_j, Rcell, Tlist, Vlist, Slist) ]
   
   klist = physical_k.(Ref(sys), klist)
   G = EquivariantTensors.ETGraph(ii, jj; edge_data = X_ij, graph_data = klist)
   @assert G.first == first

   return G 
end 