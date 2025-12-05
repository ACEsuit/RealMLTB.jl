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
   klist = G.graph_data.klist 
   return _assemble_dense(G, vH, vS, klist), st 
end


_indexblock(n::Integer, NORB) = (n-1) * NORB .+ (1:NORB)

function _assemble_dense(G,  # this prevents repeated entries 
                         vH::AbstractVector{SMatrix{NORB, NORB, TH}}, 
                         vS::AbstractVector{SMatrix{NORB, NORB, TH}}, 
                         klist::AbstractVector{SVector{DIM, TK}}) where {DIM, K}
   
   ii = G.ii; jj = G.jj; first = G.first 
   edges = G.edge_data 
   nat = length(G.first) - 1 
   # TODO: need to add Rcell to the graph edges 
            
   function _init_mat() 
      A = similar(vH, TH, (NORB * nat, NORB * nat))
      fill!(A, zero(TH))
      return A
   end

   # TODO: alternatively we could store this as a 3d tensor, which might 
   #       give us more options when implementing the KA version
   Hout = [ _init_mat() for i = 1:length(klist) ]
   Sout = [ _init_mat() for i = 1:length(klist) ]

   for (i, j, H_ij, S_ij, S) in zip(ii, jj, vH, vS, Rcell)
      Ii, Ij = _indexblock(i, NORB), _indexblock(j, NORB)
      @inbounds for a = 1:NORB, b = 1:NORB
         for (ik, k) in enumerate(klist) 
            eikS = exp( im * dot(k, S) )
            Hout[ik][Ii[a], Ij[b]] += H_ij[a, b] * eikS
            Sout[ik][Ii[a], Ij[b]] += S_ij[a, b] * eikS
         end
      end
   end
   for H in Hout
      H .= (H .+ H') ./ 2
   end
   for S in Sout
      S .= (S .+ S') ./ 2
   end
   return H, S
end

