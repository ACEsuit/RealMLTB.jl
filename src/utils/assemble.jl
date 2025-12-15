# Implements the differentiable construction of tight-binding Hamiltonian 
# and Overlap matrix blocks from 3-center integrals (with custom rrules), 
# and provides generalized eigenvalue solvers.

function _3center_triplet(G, basis::BasisSet, Wv::AbstractVector, Wb::AbstractMatrix)
   ii, jj = G.ii, G.jj
   Tlist = getfield.(G.edge_data, :t)
   Vlist = getfield.(G.edge_data, :v)
   Slist = getfield.(G.edge_data, :s)
   T = getfield(G.graph_data, :t)
   V = getfield(G.graph_data, :v)
   S = getfield(G.graph_data, :s)
   NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
   TV = eltype(Wv)

   function _init_mat() 
      return SMatrix{NORB, NORB, TV}(zeros(NORB, NORB))
   end
   
   nnz = G.first[end] - 1 + nat
   vH = [ _init_mat() for _ = 1:nnz ]
   vS = [ _init_mat() for _ = 1:nnz ]

   H_AO_onsite, S_AO_onsite = construct_integral(Wv, Wb, T, V, S)
   for i = 1:nat
      Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
      vH[i] = SMatrix{NORB, NORB, TV}(Hij)
      vS[i] = SMatrix{NORB, NORB, TV}(Sij)
   end

   idx = nat
   for (i, j, T_pair, V_pair_raw, S_pair) in zip(ii, jj, Tlist, Vlist, Slist)
      H_AO_offsite, S_AO_offsite = RealMLTB.construct_integral(Wv, Wb, T_pair, V_pair_raw, S_pair)
      Hij, Sij = RealMLTB.siteblock(H_AO_offsite, S_AO_offsite, ao_ranges, i, j)

      idx += 1
      vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
      vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
   end

   return vH, vS
end

using ChainRulesCore: unthunk, NoTangent, AbstractZero, zero_tangent
function ChainRulesCore.rrule(::typeof(_3center_triplet),
                              G,
                              basis::BasisSet,
                              Wv::AbstractVector,
                              Wb::AbstractMatrix)

   ii, jj   = G.ii, G.jj
   Tlist    = getfield.(G.edge_data,  :t)
   Vlist    = getfield.(G.edge_data,  :v)
   Slist    = getfield.(G.edge_data,  :s)
   T        = getfield(G.graph_data,  :t)
   V        = getfield(G.graph_data,  :v)
   S        = getfield(G.graph_data,  :s)

   NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
   TV = eltype(Wv)

   _init_mat() = SMatrix{NORB, NORB, TV}(zeros(TV, NORB, NORB))

   nnz  = G.first[end] - 1 + nat
   vH   = [_init_mat() for _ in 1:nnz]
   vS   = [_init_mat() for _ in 1:nnz]

   # ---------- onsite (match primal): construct_integral(Wv, Wb, T, V, S) ----------
   (H_AO_onsite, S_AO_onsite), pb_onsite =
      ChainRulesCore.rrule(construct_integral, Wv, Wb, T, V, S)

   for i = 1:nat
      Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
      vH[i] = SMatrix{NORB, NORB, TV}(Hij)
      vS[i] = SMatrix{NORB, NORB, TV}(Sij)
   end

   # ---------- offsite (match primal): construct_integral(Wv, Wb, T_pair, V_pair_raw, S_pair) ----------
   nedges  = length(ii)
   pb_offs = Vector{Any}(undef, nedges)

   idx = nat
   for (e, (i, j, T_pair, V_pair_raw, S_pair)) in
         enumerate(zip(ii, jj, Tlist, Vlist, Slist))

      idx += 1
      (H_AO_off, S_AO_off), pb_off =
         ChainRulesCore.rrule(construct_integral, Wv, Wb, T_pair, V_pair_raw, S_pair)
      pb_offs[e] = pb_off

      Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
      vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
      vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
   end

   function pullback(ΔΩ)
      ΔvH_th, ΔvS_th = ΔΩ
      ΔvH = unthunk(ΔvH_th)
      ΔvS = unthunk(ΔvS_th)

      # tangent element type
      TΔ   = eltype(ΔvH[1])
      N_AO = size(H_AO_onsite, 1)

      # ---- scatter onsite adjoints back to AO space ----
      ΔH_AO_onsite = zeros(TΔ, N_AO, N_AO)
      ΔS_AO_onsite = zeros(TΔ, N_AO, N_AO)

      for i in 1:nat
         rows = ao_ranges[i]
         cols = ao_ranges[i]
         ΔHij = ΔvH[i]
         ΔSij = ΔvS[i]
         @inbounds for a in 1:NORB, b in 1:NORB
            ΔH_AO_onsite[rows[a], cols[b]] += ΔHij[a, b]
            ΔS_AO_onsite[rows[a], cols[b]] += ΔSij[a, b]
         end
      end

      # pb returns: (Δf, ΔWv, ΔWb, ΔT, ΔV, ΔS)
      _, Wv̄_ons, Wb̄_ons, _, _, _ =
         pb_onsite((ΔH_AO_onsite, ΔS_AO_onsite))

      Wv̄_total = Wv̄_ons
      Wb̄_total = Wb̄_ons

      # ---- offsite: build ΔH_AO / ΔS_AO per-edge and call each pb ----
      for e in 1:nedges
         i   = ii[e]
         j   = jj[e]
         idx = nat + e

         rows = ao_ranges[i]
         cols = ao_ranges[j]
         ΔHij = ΔvH[idx]
         ΔSij = ΔvS[idx]

         ΔH_AO_e = zeros(TΔ, N_AO, N_AO)
         ΔS_AO_e = zeros(TΔ, N_AO, N_AO)
         @inbounds for a in 1:NORB, b in 1:NORB
            ΔH_AO_e[rows[a], cols[b]] += ΔHij[a, b]
            ΔS_AO_e[rows[a], cols[b]] += ΔSij[a, b]
         end

         _, Wv̄_e, Wb̄_e, _, _, _ = pb_offs[e]((ΔH_AO_e, ΔS_AO_e))

         Wv̄_total .+= Wv̄_e
         Wb̄_total .+= Wb̄_e
      end

      return NoTangent(), NoTangent(), NoTangent(), Wv̄_total, Wb̄_total
   end

   return (vH, vS), pullback
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
