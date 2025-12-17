# Configures the equivariant ACE architecture 
# and implements a custom differentiable assembly function 
# to flatten tensorial outputs into a vector with specific orbital permutations.

using Lux, Random, LinearAlgebra, StaticArrays
import LuxCore
import Polynomials4ML as P4ML
using EquivariantTensors
using UnitfulAtomic

function build_ace_equivariant_model(n_channels, rcut; 
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
    
   rcut_u = UnitfulAtomic.ustrip(rcut)

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

"""
assemble_Wv(val)

General-purpose assembly function. Automatically handles arbitrary angular momenta 
(S, P, D, F, G...) based on the structure of `val`.

Assumptions:
  - val[1] corresponds to L=0 (S), elements are Float64 (or length-1 vectors).
  - val[2] corresponds to L=1 (P), elements are SVector{3}.
  - val[3] corresponds to L=2 (D), elements are SVector{5}.
  - val[k] corresponds to L=k-1, element length is 2(k-1)+1.
"""
function assemble_Wv(val_tuple)
    n_atoms = size(val_tuple[1], 1)

    total_len = 0
    for (L_idx, block) in enumerate(val_tuple)
        L = L_idx - 1
        n_funcs = 2 * L + 1
        n_shells = size(block, 2)
        total_len += n_atoms * n_shells * n_funcs
    end

    Wv = zeros(Float64, total_len)
    cursor = 1

    perm_L1 = SVector{3, Int}(3, 1, 2)
    perm_L2 = SVector{5, Int}(1, 2, 3, 4, 5) 

    @inbounds for i in 1:n_atoms
        for (L_idx, block) in enumerate(val_tuple)
            L = L_idx - 1
            n_shells = size(block, 2)
            
            for j in 1:n_shells
                v = block[i, j]

                if L == 0 
                    Wv[cursor] = v
                    cursor += 1
                    
                elseif L == 1
                    Wv[cursor]     = v[perm_L1[1]]
                    Wv[cursor + 1] = v[perm_L1[2]]
                    Wv[cursor + 2] = v[perm_L1[3]]
                    cursor += 3
                    
                elseif L == 2
                    for k in 1:5
                         Wv[cursor + k - 1] = v[perm_L2[k]]
                    end
                    cursor += 5
                    
                else
                    for k in eachindex(v)
                        Wv[cursor + k - 1] = v[k]
                    end
                    cursor += length(v)
                end
            end
        end
    end

    return Wv
end

using ChainRulesCore
using StaticArrays

function ChainRulesCore.rrule(::typeof(assemble_Wv), val_tuple)
    y = assemble_Wv(val_tuple)

    function assemble_Wv_pullback(Δy)
        ΔWv = unthunk(Δy)
        n_atoms = size(val_tuple[1], 1)
        Δval_tuple = ntuple(length(val_tuple)) do L_idx
            block_primal = val_tuple[L_idx]
            similar(block_primal)
        end

        cursor = 1
        
        @inbounds for i in 1:n_atoms
            for L_idx in 1:length(Δval_tuple)
                L = L_idx - 1
                n_shells = size(Δval_tuple[L_idx], 2)
                
                for j in 1:n_shells
                    if L == 0
                        # --- L=0: ---
                        # Forward: Wv[c] = v
                        # Backward: dv = dWv[c]
                        Δval_tuple[L_idx][i, j] = ΔWv[cursor]
                        cursor += 1
                        
                    elseif L == 1
                        
                        d_raw_1 = ΔWv[cursor]
                        d_raw_2 = ΔWv[cursor + 1]
                        d_raw_3 = ΔWv[cursor + 2]
                        Δval_tuple[L_idx][i, j] = SVector(d_raw_2, d_raw_3, d_raw_1)
                        cursor += 3
                        
                    else 
                        # Forward: Wv[c:...] = v[:]
                        # Backward: dv[:] = dWv[c:...]
                        len = 2 * L + 1
                        
                        val_len = Val(len)
                        Δval_tuple[L_idx][i, j] = SVector{len}(
                            ntuple(k -> ΔWv[cursor + k - 1], val_len)
                        )
                        cursor += len
                    end
                end
            end
        end

        return NoTangent(), Δval_tuple
    end

    return y, assemble_Wv_pullback
end

function build_ace_invariant_model(NORB, n_channels, rcut; 
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
    
   rcut_u = UnitfulAtomic.ustrip(rcut)

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

   NBAS = sum([n_channels[i] * (2i-1) for i = 1:length(n_channels)])
   𝔹basis = EquivariantTensors.sparse_equivariant_tensor(; 
      L = 0, 
      mb_spec = mb_spec, 
      Rnl_spec = Rnl_spec, 
      Ylm_spec = Ylm_spec, 
      basis = real 
   )

   # Create the ACE layer
   acel = EquivariantTensors.SparseACElayer(𝔹basis, (NORB * NBAS,))

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

function assemble_Wb(NORB, val_tuple)
    Wb = val_tuple[1]
    NATOM      = size(Wb, 1)
    n_cols_sub = size(Wb, 2) ÷ NORB

    Wb_final = zeros(eltype(Wb), NATOM*NORB, NATOM*n_cols_sub)

    @inbounds for a in 1:NATOM
        rs = (a-1)*NORB      + 1 : a*NORB
        cs = (a-1)*n_cols_sub + 1 : a*n_cols_sub
        @views Wb_final[rs, cs] .= permutedims(reshape(Wb[a, :], n_cols_sub, NORB), (2,1))
    end
    return Wb_final
end

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(assemble_Wb), NORB::Integer, val_tuple)
    Y = assemble_Wb(NORB, val_tuple)

    Wb         = val_tuple[1]
    NATOM      = size(Wb, 1)
    n_cols_sub = size(Wb, 2) ÷ NORB

    function pullback(Ȳ)
        Ȳ = unthunk(Ȳ)

        Ẇb = zeros(eltype(Wb), size(Wb))  # same shape as Wb

        @inbounds for a in 1:NATOM
            rs = (a-1)*NORB       + 1 : a*NORB
            cs = (a-1)*n_cols_sub + 1 : a*n_cols_sub

            Δblock = @view Ȳ[rs, cs]  # NORB × n_cols_sub

            # forward: block = permutedims(reshape(Wb[a,:], n_cols_sub, NORB), (2,1))
            # reverse: Wb[a,:] gradient = vec(permutedims(Δblock, (2,1)))
            Δrow = vec(permutedims(Δblock, (2,1)))  # length = NORB*n_cols_sub

            @views Ẇb[a, :] .= Δrow
        end

        # tangent for val_tuple: only slot 1 (Wb) has gradient
        Δval_tuple = ntuple(i -> (i == 1 ? Ẇb : ZeroTangent()), length(val_tuple))

        return NoTangent(), NoTangent(), Δval_tuple
    end

    return Y, pullback
end



