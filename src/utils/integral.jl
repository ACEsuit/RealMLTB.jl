# Implements differentiable construction of Hamiltonian and Overlap matrices from raw integrals, 
# along with basis set indexing and channel helpers.

function _construct_integral_core(Wv, Wb, T_pair, V_pair_raw, S_pair)
   nbas = size(V_pair_raw, 1)
   M      = reshape(V_pair_raw, nbas^2, nbas)
   V_pair = reshape(M * Wv, nbas, nbas)

   A = T_pair + V_pair
   H = Wb * A      * Wb'
   S = Wb * S_pair * Wb'
   return H, S
end

function construct_integral(Wv, Wb, T_pair, V_pair_raw, S_pair)
   return _construct_integral_core(Wv, Wb, T_pair, V_pair_raw, S_pair)
end

function _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw, S_pair)
   nbas = size(V_pair_raw, 1)
   M      = reshape(V_pair_raw, nbas^2, nbas)
   V_pair = reshape(M * Wv, nbas, nbas)

   A = T_pair + V_pair
   H = Wb * A      * Wb'
   S = Wb * S_pair * Wb'
   return H, S, A, M
end

function _construct_integral_rrule_common(Wv, Wb, T_pair, V_pair_raw, S_pair)
   H, S, A, M = _construct_integral_primal(Wv, Wb, T_pair, V_pair_raw, S_pair)

   function pullback_common(ΔΩ)
      ΔH, ΔS = ΔΩ

      # H = Wb * A * Wb'
      ΔWb_H = ΔH * Wb * A'      + ΔH' * Wb * A

      # S = Wb * S_pair * Wb'
      ΔWb_S = ΔS * Wb * S_pair' + ΔS' * Wb * S_pair

      Wb̄ = real.(ΔWb_H + ΔWb_S)
      # Wb̄ = ΔWb_H + ΔWb_S

      # dL/dA = Wb' * ΔH * Wb
      ΔA = real.(Wb' * ΔH * Wb)
      # ΔA = Wb' * ΔH * Wb

      # A = T_pair + V_pair
      ΔV = ΔA

      # V_pair = reshape(M * Wv, nbas, nbas)
      # y = M * Wv, V = reshape(y, nbas, nbas)
      # ⇒ ∂L/∂Wv = M' * vec(ΔV)
      Δy  = vec(ΔV)
      Wv̄ = real.(M' * Δy)
      # Wv̄ = M' * Δy

      return NoTangent(),  # 
               Wv̄,           # Wv
               Wb̄,           # Wb
               NoTangent(),  # T_pair
               NoTangent(),  # V_pair_raw
               NoTangent()   # S_pair
   end

   return (H, S), pullback_common
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
         counts[L+1] += 1
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