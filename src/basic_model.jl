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
   return (Wv = Wv, Wb = Wb, )
end

function LuxCore.initialstates(rng::AbstractRNG, l::Simple3Center{NORB, NBAS, NATOM}) where {NORB, NBAS, NATOM}
   return (basisset = l.basisset, )
end

function (l::Simple3Center)(G, ps, st)
   basis = st.basisset
   Wv = ps.Wv    # parameters for the effective potential 
   Wb = ps.Wb    # parameters for the basis functions

   vH, vS = _3center_triplet(G, basis, Wv, Wb)
   return (G, vH, vS), ps, st
end

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
