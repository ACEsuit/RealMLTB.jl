using Lux, Random, LinearAlgebra, StaticArrays
import LuxCore
import Polynomials4ML as P4ML
import ForwardDiff as FD
using EquivariantTensors

function build_ace_model(n_channels, rcut; 
   Dtot::Int = 9,
   maxl::Int = 5,
   ORD::Int = 3,
   lmax::Int = length(n_channels) - 1,
   rng::AbstractRNG = MersenneTwister(1234))

   # Define Basic Basis Sets (Radial & Angular)
   polys = P4ML.ChebBasis(Dtot + 1)
   Rnl_spec = P4ML.natural_indices(polys)

   ybas = P4ML.real_sphericalharmonics(maxl; T = Float32, static=true)
   Ylm_spec = P4ML.natural_indices(ybas)

   # Construct the Embedding Layer
   ycut = 1 / 2 
   env = y -> (y - ycut)^2 * (y + ycut)^2
    
   rcut_u = ustrip(rcut)

   # Radial basis transformation chain:
   # 1. Transform distance r -> x
   # 2. Evaluate polynomials and apply envelope
   rbasis = Chain( 
      y = EquivariantTensors.NTtransformST((x, st) -> 1 / (1 + norm(x.𝐫 / rcut_u))),
      P = SkipConnection(
         polys,
         WrappedFunction(PY -> env.(PY[2]) .* PY[1])
      ) 
   )

   # Angular basis transformation chain:
   # 1. Extract vector r
   # 2. Evaluate spherical harmonics
   ybasis = Chain( 
      trans = EquivariantTensors.NTtransformST((x, st) -> x.𝐫), 
      basis = ybas 
   )

   # Combine into an Edge Embedding Layer
   embed = EquivariantTensors.EdgeEmbed(BranchLayer(; Rnl = rbasis, Ylm = ybasis))

   # Construct the Sparse ACE Layer
   mb_spec = EquivariantTensors.sparse_nnll_set(; 
      ORD = ORD, 
      minn = 0, 
      maxn = Dtot, 
      maxl = maxl, 
      level = bb -> sum((b.n + b.l) for b in bb; init=0), 
      maxlevel = Dtot
   )

   target_LL = Tuple(collect(0:lmax))
   𝔹basis = EquivariantTensors.sparse_equivariant_tensors(; 
      LL = target_LL, 
      mb_spec = mb_spec, 
      Rnl_spec = Rnl_spec, 
      Ylm_spec = Ylm_spec, 
      basis = real 
   )

   # Create the ACE layer
   acel = EquivariantTensors.SparseACElayer(𝔹basis, n_channels)

   # Assemble Model and Initialize
   model = Lux.Chain(; 
      embed = embed,  
      ace = acel
   )

   # Initialize parameters (ps) and state (st)
   ps, st = LuxCore.setup(rng, model)
   ps = EquivariantTensors.float32(ps)
   st = EquivariantTensors.float32(st)

   return model, ps, st
end