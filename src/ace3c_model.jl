export ace_3c_model

using Lux, Random

struct ACE3Center{Mv, Mw, B, NORB, NBAS, NATOM} <: AbstractLuxLayer
   ace_v_model::Mv
   ace_w_model::Mw
   basisset::B
end

function ACE3Center(ace_v_model, ace_w_model, bs, NORB)
   NBAS = bs.nbas
   NATOM = bs.natoms
   return ACE3Center{typeof(ace_v_model), typeof(ace_w_model), typeof(bs), NORB, NBAS, NATOM}(ace_v_model, ace_w_model, bs)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ACE3Center{Mv, Mw, B, NORB, NBAS, NATOM}) where {Mv, Mw, B, NORB, NBAS, NATOM}
   ps_acev = LuxCore.initialparameters(rng, l.ace_v_model)
   ps_acev = EquivariantTensors.float32(ps_acev)

   ps_acew = LuxCore.initialparameters(rng, l.ace_w_model)
   ps_acew = EquivariantTensors.float32(ps_acew)

   return (acev = ps_acev, acew = ps_acew)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACE3Center)
   st_acev = LuxCore.initialstates(rng, l.ace_v_model)
   st_acev = EquivariantTensors.float32(st_acev)

   st_acew = LuxCore.initialstates(rng, l.ace_w_model)
   st_acew = EquivariantTensors.float32(st_acew)
   return (acev = st_acev, acew = st_acew, basisset = l.basisset)
end

function (l::ACE3Center{Mv, Mw, B, NORB, NBAS, NATOM})(G, ps, st) where {Mv, Mw, B, NORB, NBAS, NATOM}
   val_v_tuple, st_acev = l.ace_v_model(G, ps.acev, st.acev)
   val_w_tuple, st_acew = l.ace_w_model(G, ps.acew, st.acew)

   Wv = assemble_Wv(val_v_tuple) 
   Wb = assemble_Wb(NORB, val_w_tuple) 
   vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
   new_st = (acev = st_acev, acew = st_acew, basisset = st.basisset)
   return (G, vH, vS), ps, new_st
end

function ace_3c_model(bs, NORB, rcut; 
   Dtot = 9,
   maxl::Int = 5,
   ORD::Int = 3)

   _, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs)
   acev_backend, _, _ = RealMLTB.build_ace_equivariant_model(n_channels, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD) 
   acew_backend, _, _ = RealMLTB.build_ace_invariant_model(NORB, n_channels, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD) 

   model = Chain(
      RealMLTB.ACE3Center(acev_backend, acew_backend, bs, NORB),
      AsmTrip2Dense(), 
      WrappedFunction(x -> generate_λ(x[1], x[2]))
   )
   return model 
end