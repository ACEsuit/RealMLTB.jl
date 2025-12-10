# Questions and TODOs: 
#  - klist in state or an input? => arguably a property of the graph??
#  - Rcell needs to be added to the graph 
#  - allow `vS::nothing` to skip assembly of S matrix 
#  - what if norb varies for different species? 


struct AsmTrip2Dense <: AbstractLuxLayer
end

LuxCore.initialparameters(rng::AbstractRNG, l::AsmTrip2Dense) = NamedTuple()
LuxCore.initialstates(rng::AbstractRNG, l::AsmTrip2Dense) = NamedTuple()

function (l::AsmTrip2Dense)(G_vH_vS, ps, st)   # (G, vH, vS)
   G = G_vH_vS[1]
   vH = G_vH_vS[2]
   vS = G_vH_vS[3]
   klist = G.node_data
   return _assemble_dense(G, vH, vS, klist), st 
end

_indexblock(n::Integer, NORB) = (n-1) * NORB .+ (1:NORB)

function _assemble_dense(G,  # this prevents repeated entries 
                         vH::AbstractVector{<:SMatrix{NORB, NORB, TH}}, 
                         vS::AbstractVector{<:SMatrix{NORB, NORB, TH}}, 
                         klist::AbstractVector{<:SVector{DIM, TK}}) where {DIM, NORB, TH, TK}
   
   ii = G.ii; jj = G.jj
   nat = length(G.first) - 1
   # TODO: need to add Rcell to the graph edges 
            
   function _init_mat() 
      A = similar(vH, Complex{TH}, (NORB * nat, NORB * nat))
      fill!(A, zero(TH))
      return A
   end

   # TODO: alternatively we could store this as a 3d tensor, which might 
   #       give us more options when implementing the KA version
   Hout = [ _init_mat() for i = 1:length(klist) ]
   Sout = [ _init_mat() for i = 1:length(klist) ]

   for i = 1:nat
      H_ii, S_ii = vH[i], vS[i]
      Ii = _indexblock(i, NORB)
      @inbounds for a = 1:NORB, b = 1:NORB
         for ik = 1:length(klist) 
            Hout[ik][Ii[a], Ii[b]] += H_ii[a, b]
            Sout[ik][Ii[a], Ii[b]] += S_ii[a, b]
         end
      end
   end
   Rcell = getfield.(G.edge_data, :rij)
   idx = nat
   for (i, j, R) in zip(ii, jj, Rcell)
      Ii, Ij = _indexblock(i, NORB), _indexblock(j, NORB)
      idx += 1
      Hij = vH[idx]
      Sij = vS[idx]
      
      @inbounds for a = 1:NORB, b = 1:NORB
         for (ik, k) in enumerate(klist) 
            phase = exp(im * dot(k, R))

            Hout[ik][Ii[a], Ij[b]] += Hij[a, b] * phase
            Sout[ik][Ii[a], Ij[b]] += Sij[a, b] * phase
         end
      end
   end
   for H in Hout
      H .= (H .+ H') ./ 2
   end
   for S in Sout
      S .= (S .+ S') ./ 2
   end
   return Hout, Sout
end

using ChainRulesCore: unthunk

function ChainRulesCore.rrule(::typeof(_assemble_dense),
                           G,
                           vH::AbstractVector{<:SMatrix{NORB, NORB, TH}},
                           vS::AbstractVector{<:SMatrix{NORB, NORB, TH}},
                           klist::AbstractVector{<:SVector{DIM, TK}}
                           ) where {DIM, NORB, TH, TK}

   Hout, Sout = _assemble_dense(G, vH, vS, klist)

   function pullback(ΔΩ)
      ΔHout_th, ΔSout_th = ΔΩ
      ΔHout = unthunk(ΔHout_th)
      ΔSout = unthunk(ΔSout_th)

      ii = G.ii
      jj = G.jj
      nat = length(G.first) - 1
      Rcell = getfield.(G.edge_data, :rij)

      L = length(vH)
      TΔ = eltype(ΔHout[1])
      TΔ = TΔ <: Real ? Complex{TΔ} : TΔ

      function _init_mat() 
         A = similar(vH, TΔ, (NORB, NORB))
         fill!(A, zero(TH))
         return A
      end

      vH_grad_mat = [_init_mat()  for _ in 1:L]
      vS_grad_mat = [_init_mat()  for _ in 1:L]

      ΔHsym = [ (ΔHk .+ ΔHk') ./ 2 for ΔHk in ΔHout ]
      ΔSsym = [ (ΔSk .+ ΔSk') ./ 2 for ΔSk in ΔSout ]

      for i in 1:nat
         idx = i
         Ii  = _indexblock(i, NORB)

         @inbounds for a in 1:NORB, b in 1:NORB
            for ik in 1:length(klist)
               vH_grad_mat[idx][a, b] += ΔHsym[ik][Ii[a], Ii[b]]
               vS_grad_mat[idx][a, b] += ΔSsym[ik][Ii[a], Ii[b]]
            end
         end
      end

      idx = nat
      for (i, j, R) in zip(ii, jj, Rcell)
         Ii = _indexblock(i, NORB)
         Ij = _indexblock(j, NORB)
         idx += 1

         @inbounds for a in 1:NORB, b in 1:NORB
            for (ik, k) in enumerate(klist)
               phase = exp(im * dot(k, R))
               cphase = conj(phase)

               vH_grad_mat[idx][a, b] += cphase * ΔHsym[ik][Ii[a], Ij[b]]
               vS_grad_mat[idx][a, b] += cphase * ΔSsym[ik][Ii[a], Ij[b]]
            end
         end
      end

      function _init_smat() 
         return SMatrix{NORB, NORB, TΔ}(zeros(NORB, NORB))
      end

      vH̄ = [ _init_smat() for _ = 1:length(vH) ]
      vS̄ = [ _init_smat() for _ = 1:length(vH) ]
      for i in eachindex(vH)
         vH̄[i] = SMatrix{NORB, NORB, TΔ}(vH_grad_mat[i])
         vS̄[i] = SMatrix{NORB, NORB, TΔ}(vS_grad_mat[i])
      end

      return NoTangent(), NoTangent(), vH̄, vS̄, NoTangent()
   end

   return (Hout, Sout), pullback
end
