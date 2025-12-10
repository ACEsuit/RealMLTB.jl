#    st.basisset
# G -> integrals  \
#              ps  -> triplet format -> dense format H
#
#
# Questions / TODOs 
#   - better Wb initialization scheme??
export simple_3c_model

struct Simple3Center{NORB, NBAS, NATOM} <: AbstractLuxLayer
   basisset 
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Simple3Center{NORB, NBAS, NATOM}) where {NORB, NBAS, NATOM}
   Wv = glorot_normal(rng, Float64, NBAS)
   Wb = glorot_normal(rng, Float64, NORB * NATOM, NBAS) 

   # S_AO = overlap(l.basisset)
   # vals, vecs = eigen(Symmetric(S_AO))
   # mask = vals .> 1e-6
   # inv_sqrt_vals = zeros(eltype(vals), length(vals))
   # inv_sqrt_vals[mask] .= 1.0 ./ sqrt.(vals[mask])
   # S_inv_sqrt = vecs * Diagonal(inv_sqrt_vals) * vecs'
   # target_rows = NORB * NATOM
   # S_inv_sqrt[1:target_rows, :]
   return (Wv = Wv, Wb = Wb, )
end

function LuxCore.initialstates(rng::AbstractRNG, l::Simple3Center{NORB, NBAS, NATOM}) where {NORB, NBAS, NATOM}
   return (basisset = l.basisset, )
end

function (l::Simple3Center)(G, ps, st)
   basis = st.basisset
   Wv = ps.Wv    # parameters for the effective potential 
   Wb = ps.Wb    # parameters for the basis functions

   vH, vS = _simple3center_triplet(G, basis, Wv, Wb)
   return (G, vH, vS), ps, st
end

function _simple3center_triplet(G, basis::BasisSet, Wv::AbstractVector, Wb::AbstractMatrix)
   ii, jj = G.ii, G.jj
   Tlist = getfield.(G.edge_data, :t)
   Vlist = getfield.(G.edge_data, :v)
   Slist = getfield.(G.edge_data, :s)
   NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
   TV = eltype(Wv)

   function _init_mat() 
      return SMatrix{NORB, NORB, TV}(zeros(NORB, NORB))
   end
   
   nnz = G.first[end] - 1 + nat
   vH = [ _init_mat() for _ = 1:nnz ]
   vS = [ _init_mat() for _ = 1:nnz ]

   H_AO_onsite, S_AO_onsite = construct_integral(basis, Wv, Wb)
   for i = 1:nat
      Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
      vH[i] = SMatrix{NORB, NORB, TV}(Hij)
      vS[i] = SMatrix{NORB, NORB, TV}(Sij)
   end

   idx = nat
   for (i, j, T_pair, V_pair_raw, S_pair) in zip(ii, jj, Tlist, Vlist, Slist)
      idx += 1
      H_AO_offsite, S_AO_offsite = construct_integral(basis, Wv, Wb, T_pair, V_pair_raw, S_pair)
      Hij, Sij = siteblock(H_AO_offsite, S_AO_offsite, ao_ranges, i, j)
      vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
      vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
   end

   return vH, vS
end

using ChainRulesCore: unthunk

function ChainRulesCore.rrule(::typeof(_simple3center_triplet),
                              G,
                              basis::BasisSet,
                              Wv::AbstractVector,
                              Wb::AbstractMatrix)
   ii, jj = G.ii, G.jj
   Tlist = getfield.(G.edge_data, :t)
   Vlist = getfield.(G.edge_data, :v)
   Slist = getfield.(G.edge_data, :s)

   NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
   TV = eltype(Wv)

   function _init_mat()
      return SMatrix{NORB, NORB, TV}(zeros(TV, NORB, NORB))
   end

   nnz  = G.first[end] - 1 + nat
   vH   = [ _init_mat() for _ in 1:nnz ]
   vS   = [ _init_mat() for _ in 1:nnz ]

   # ----- onsite: construct_integral(basis, Wv, Wb) -----
   (H_AO_onsite, S_AO_onsite), pb_onsite =
        ChainRulesCore.rrule(construct_integral, basis, Wv, Wb)

   for i = 1:nat
      Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
      vH[i] = SMatrix{NORB, NORB, TV}(Hij)
      vS[i] = SMatrix{NORB, NORB, TV}(Sij)
   end

   # ----- offsite: construct_integral(basis, Wv, Wb, T_pair, V_pair_raw, S_pair) -----
   nedges = length(ii)
   pb_offs = Vector{Any}(undef, nedges)

   idx = nat
   for (e, (i, j, T_pair, V_pair_raw, S_pair)) in
      enumerate(zip(ii, jj, Tlist, Vlist, Slist))
      idx += 1

      (H_AO_off, S_AO_off), pb_off =
            ChainRulesCore.rrule(construct_integral, basis, Wv, Wb,
                              T_pair, V_pair_raw, S_pair)
      pb_offs[e] = pb_off

      Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
      vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
      vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
   end

   function pullback(ΔΩ)
      ΔvH_th, ΔvS_th = ΔΩ
      ΔvH = unthunk(ΔvH_th)
      ΔvS = unthunk(ΔvS_th)

      TΔ   = eltype(ΔvH[1])
      N_AO = size(H_AO_onsite, 1)

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

      ΔH_AO_off = [zeros(TΔ, N_AO, N_AO) for _ in 1:nedges]
      ΔS_AO_off = [zeros(TΔ, N_AO, N_AO) for _ in 1:nedges]

      for e in 1:nedges
         i   = ii[e]
         j   = jj[e]
         idx = nat + e

         rows = ao_ranges[i]
         cols = ao_ranges[j]
         ΔHij = ΔvH[idx]
         ΔSij = ΔvS[idx]

         @inbounds for a in 1:NORB, b in 1:NORB
            ΔH_AO_off[e][rows[a], cols[b]] += ΔHij[a, b]
            ΔS_AO_off[e][rows[a], cols[b]] += ΔSij[a, b]
         end
      end

      _, _, Wv̄_ons, Wb̄_ons =
            pb_onsite((ΔH_AO_onsite, ΔS_AO_onsite))

      Wv̄_total = Wv̄_ons
      Wb̄_total = Wb̄_ons

      for e in 1:nedges
         _, _, Wv̄_e, Wb̄_e, _, _, _ =
                pb_offs[e]((ΔH_AO_off[e], ΔS_AO_off[e]))

         Wv̄_total .+= Wv̄_e
         Wb̄_total .+= Wb̄_e
      end

      return NoTangent(), NoTangent(), NoTangent(), Wv̄_total, Wb̄_total
   end

   return (vH, vS), pullback
end

# function _simple3center_triplet(G, basis::BasisSet, Wv::AbstractVector, Wb::AbstractMatrix)
#    ii, jj = G.ii, G.jj
#    Rcell = getfield.(G.edge_data, :rij)
#    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
#    TV = eltype(Wv)

#    function _init_mat() 
#       return SMatrix{NORB, NORB, TV}(zeros(NORB, NORB))
#    end
   
#    nnz = G.first[end] - 1 + nat
#    vH = [ _init_mat() for _ = 1:nnz ]
#    vS = [ _init_mat() for _ = 1:nnz ]

#    H_AO_onsite, S_AO_onsite = construct_integral(basis, Wv, Wb)
#    for i = 1:nat
#       Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
#       vH[i] = SMatrix{NORB, NORB, TV}(Hij)
#       vS[i] = SMatrix{NORB, NORB, TV}(Sij)
#    end

#    idx = nat
#    for (i, j, Rij) in zip(ii, jj, Rcell)
#       idx += 1
#       H_AO_offsite, S_AO_offsite = construct_integral(basis, Wv, Wb, Rij)
#       Hij, Sij = siteblock(H_AO_offsite, S_AO_offsite, ao_ranges, i, j)
#       vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
#       vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
#    end

#    return vH, vS
# end

# function ChainRulesCore.rrule(::typeof(_simple3center_triplet),
#                            G,
#                            basis::BasisSet,
#                            Wv::AbstractVector,
#                            Wb::AbstractMatrix)

#    ii, jj = G.ii, G.jj
#    Rcell = getfield.(G.edge_data, :rij)
#    NORB, nat, ao_ranges = atom_ao_ranges(basis, Wb)
#    TV = eltype(Wv)

#    function _init_mat()
#       return SMatrix{NORB, NORB, TV}(zeros(TV, NORB, NORB))
#    end

#    nedges = length(ii)
#    nnz    = nedges + nat

#    vH = [ _init_mat() for _ in 1:nnz ]
#    vS = [ _init_mat() for _ in 1:nnz ]

#    (H_AO_onsite, S_AO_onsite), pb_onsite =
#       ChainRulesCore.rrule(construct_integral, basis, Wv, Wb)

#    for i in 1:nat
#       Hij, Sij = siteblock(H_AO_onsite, S_AO_onsite, ao_ranges, i, i)
#       vH[i] = SMatrix{NORB, NORB, TV}(Hij)
#       vS[i] = SMatrix{NORB, NORB, TV}(Sij)
#    end

#    pb_offs = Vector{Any}(undef, nedges)

#    idx = nat
#    for (e, (i, j, Rij)) in enumerate(zip(ii, jj, Rcell))
#       (H_AO_off, S_AO_off), pb_off =
#          ChainRulesCore.rrule(construct_integral, basis, Wv, Wb, Rij)
#       pb_offs[e] = pb_off

#       idx += 1
#       Hij, Sij = siteblock(H_AO_off, S_AO_off, ao_ranges, i, j)
#       vH[idx] = SMatrix{NORB, NORB, TV}(Hij)
#       vS[idx] = SMatrix{NORB, NORB, TV}(Sij)
#    end

#    function pullback(ΔΩ)
#       ΔvH, ΔvS = ΔΩ  # Vector{SMatrix}

#       N_AO = size(H_AO_onsite, 1)
#       TΔ   = eltype(ΔvH[1])

#       ΔH_AO_onsite = zeros(TΔ, N_AO, N_AO)
#       ΔS_AO_onsite = zeros(TΔ, N_AO, N_AO)

#       for i in 1:nat
#          rows = ao_ranges[i]
#          cols = ao_ranges[i]
#          ΔHij = ΔvH[i]
#          ΔSij = ΔvS[i]

#          @inbounds for a in 1:NORB, b in 1:NORB
#             ΔH_AO_onsite[rows[a], cols[b]] += ΔHij[a, b]
#             ΔS_AO_onsite[rows[a], cols[b]] += ΔSij[a, b]
#          end
#       end

#       ΔH_AO_off = [zeros(TΔ, N_AO, N_AO) for _ in 1:nedges]
#       ΔS_AO_off = [zeros(TΔ, N_AO, N_AO) for _ in 1:nedges]

#       for e in 1:nedges
#          i  = ii[e]
#          j  = jj[e]
#          idx = nat + e
#          rows = ao_ranges[i]
#          cols = ao_ranges[j]
#          ΔHij = ΔvH[idx]
#          ΔSij = ΔvS[idx]

#          @inbounds for a in 1:NORB, b in 1:NORB
#             ΔH_AO_off[e][rows[a], cols[b]] += ΔHij[a, b]
#             ΔS_AO_off[e][rows[a], cols[b]] += ΔSij[a, b]
#          end
#       end

#       _, _, Wv̄_ons, Wb̄_ons =
#          pb_onsite((ΔH_AO_onsite, ΔS_AO_onsite))

#       Wv̄_total = Wv̄_ons
#       Wb̄_total = Wb̄_ons

#       for e in 1:nedges
#          _, _, Wv̄_e, Wb̄_e = pb_offs[e]((ΔH_AO_off[e], ΔS_AO_off[e]))
#          Wv̄_total .+= Wv̄_e
#          Wb̄_total .+= Wb̄_e
#       end

#         return NoTangent(), NoTangent(), NoTangent(), Wv̄_total, Wb̄_total
#    end

#    return (vH, vS), pullback
# end


# utility function to build a model 

function simple_3c_model(bs, NORB)
   
   # generate the Wb parameters 
   # to extract the right basis functions 

   NBAS = bs.nbas 
   NATOM = bs.natoms
    
   model = Chain(
      Simple3Center{NORB, NBAS, NATOM}(bs), 
      AsmTrip2Dense(), 
      WrappedFunction(x -> generate_λ(x[1], x[2]))
   )

   return model 
end