
#    st.basisset
# G -> integrals  \
#              ps  -> triplet format -> dense format H
#
#
# Questions / TODOs 
#   - better Wb initialization scheme??


struct Simple3Center{NORB, NBAS} <: AbstractLuxLayer
   basisset 
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Simple3Center{NORB, NBAS}
                  ) where {NORB, NBAS}
   Wv = glorot_normal(rng, Float64, (NBAS, NORB,))  
   return (Wv = Wv, )
end

LuxCore.initialstates(rng::AbstractRNG, l::Simple3Center{NORB, NBAS}    ) = 
         ( basisset = l.basisset, 
           Wb = l.Wb, )

function (l::Simple3Center)(G, ps, st)
   basis = st.basisset
   Wv = ps.Wv    # parameters for the effective potential 
   Wb = st.Wb    # parameters for the basis functions

   vH, vS = _simple3center_triplet(G, basis, Wv, Wb)

   return (G, vH, vS), ps, st
end

function _simple3center_triplet(G, basis, Wv, Wb)

   vH = [] 
   vS = [] 

   for i in G.ii 
      # ONSITE TERMS 
      #  ... compute H_ii and S_ii blocks 

      # HOPPING TERMS
      for j in G.jj 
         #  ... compute H_ij and S_ij blocks 

      end
   end 

   return vH, vS
end 



# utility function to build a model 

function simple_3c_model(basisset, NORB)
   
   # generate the Wb parameters 
   # to extract the right basis functions 

   NBAS = length(basisset)
    
   model = Chain(
      Simple3Center{NORB, NBAS}(basisset), 
      AsmTrip2Dense(), 
   )

   return model 
end