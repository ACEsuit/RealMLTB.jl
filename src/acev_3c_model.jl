export acev_3c_model

using Lux, Random

struct ACEv3Center{M, B, NORB, NBAS, NATOM} <: AbstractLuxLayer
   ace_model::M
   basisset::B
end

function ACEv3Center(ace_model, bs, NORB)
   NBAS = bs.nbas
   NATOM = bs.natoms
   return ACEv3Center{typeof(ace_model), typeof(bs), NORB, NBAS, NATOM}(ace_model, bs)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ACEv3Center{M, B, NORB, NBAS, NATOM}) where {M, B, NORB, NBAS, NATOM}
   ps_ace = LuxCore.initialparameters(rng, l.ace_model)
   ps_ace = EquivariantTensors.float32(ps_ace)
   Nb = div(NBAS, NATOM)
   Wb = glorot_normal(rng, Float64, NORB, Nb)
   
   return (ace = ps_ace, Wb = Wb)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACEv3Center)
   st_ace = LuxCore.initialstates(rng, l.ace_model)
   st_ace = EquivariantTensors.float32(st_ace)
   return (ace = st_ace, basisset = l.basisset)
end

function (l::ACEv3Center{M, B, NORB, NBAS, NATOM})(G, ps, st) where {M, B, NORB, NBAS, NATOM}
   val_tuple, st_ace = l.ace_model(G, ps.ace, st.ace)
   Wv = assemble_Wv(val_tuple) 
   Wb = kron(Matrix(I, NATOM, NATOM), ps.Wb)
   vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
   new_st = (ace = st_ace, basisset = st.basisset)
   return (G, vH, vS), ps, new_st
end

function acev_3c_model(bs, NORB, rcut; 
   Dtot = 9,
   maxl::Int = 5,
   ORD::Int = 3)

   _, n_channels = lmax_and_nchannels_per_atom(bs)
   ace_backend, _, _ = build_ace_equivariant_model(n_channels, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD) 
   model = Chain(
      ACEv3Center(ace_backend, bs, NORB),
      AsmTrip2Dense(), 
      WrappedFunction(x -> generate_λ(x[1], x[2]))
   )
   return model 
end