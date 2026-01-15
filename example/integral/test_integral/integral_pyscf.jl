# Computes Spherical AO integrals by transforming Julia Cartesian integrals 
# using PySCF normalization and projection matrices.

using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using RealMLTB

const pyscf = pyimport("pyscf")
const np = pyimport("numpy")
const dft = pyimport("pyscf.dft")

"""
   compute_spherical_AO_integrals_pyscf(bs_cart::BasisSet)

Takes a Cartesian BasisSet from Julia, automatically calls PySCF to retrieve 
transformation matrices and normalization factors, and returns Spherical integrals 
(T, V, S) matching PySCF standards.

Arguments:
   bs_cart: GaussianBasis.jl BasisSet object (Must be Cartesian, `lib=:acsint` recommended).

Returns:
   (T_sph, V_sph, S_sph): Three matrices/tensors, all in normalized spherical basis.
"""
const ang2bohr = GaussianBasis.Acsint.ang2bohr
function compute_spherical_AO_integrals_pyscf(bs_cart::BasisSet)
   if bs_cart.basis[1] isa SphericalShell
      bs_cart = BasisSet(bs_cart.name, bs_cart.atoms; spherical=false, lib=:acsint)
   end
   # Reconstruct PySCF Molecule Object
   atom_strs = String[]
   for atom in bs_cart.atoms
      s = "$(atom.Z) $(atom.xyz[1] * ang2bohr) $(atom.xyz[2] * ang2bohr) $(atom.xyz[3] * ang2bohr)"
      push!(atom_strs, s)
   end
   pyscf_atom_str = join(atom_strs, "; ")
   basis_name = bs_cart.name

   # Initialize PySCF molecules (Used to get C matrix and Scaling Factors)
   # mol_cart: Used to compute Cartesian normalization factors
   mol_cart = pyscf.M(atom=pyscf_atom_str, basis=basis_name, cart=true) # unit="Angstrom", 
   # mol_sph:  Used to get the C matrix and serves as the final reference standard
   mol_sph  = pyscf.M(atom=pyscf_atom_str, basis=basis_name, cart=false) # unit="Angstrom", 

   # Compute Auxiliary Data (C and d_c)
   # Get Transformation Matrix C (Cart -> Sph)
   C = mol_sph.cart2sph_coeff()
    
   # Compute Cartesian Normalization/Scaling Factors (d_c)
   # d_c = sqrt( PySCF_Norm_Cart / Julia_Raw_Cart )
   S_julia_raw = overlap(bs_cart)
   S_py_norm   = mol_cart.intor("int1e_ovlp_cart")
    
   # Compute factors (square root of the ratio of diagonals)
   d_c = sqrt.(diag(S_py_norm) ./ diag(S_julia_raw))

   # Compute and Transform Integrals (S, T, V)
   # --- 2-Center Integrals (S, T) ---
   T_julia_raw = kinetic(bs_cart)
    
   # Define 2-Center transformation closure
   # Formula: M_sph = C' * (D * M_raw * D) * C
   function transform_2c(M_raw)
      # 1. Normalize (using broadcasting for speed)
      M_norm = M_raw .* (d_c * d_c')
      # 2. Rotate / Transform to Spherical
      return C' * M_norm * C
   end

   S_sph = transform_2c(S_julia_raw)
   T_sph = transform_2c(T_julia_raw)

   # --- 3-Center Integrals (V) ---
   # Compute raw Cartesian 3-center integrals in Julia
   N_cart = bs_cart.nbas
   V_julia_raw = zeros(N_cart, N_cart, N_cart)
   RealMLTB.EPI_1e3c!(V_julia_raw, bs_cart) 

   # Full Dimensional Transformation (Cart -> Sph)
   # Directly integrated logic verified previously
   N_sph = size(C, 2)
   V_sph = zeros(N_sph, N_sph, N_sph)
    
   # Normalization (Normalize Cartesian)
   # V_norm = V_raw * d[i] * d[j] * d[k]
   # Use temporary array to avoid polluting original data
   V_cart_norm = zeros(N_cart, N_cart, N_cart)
   @inbounds for k in 1:N_cart, j in 1:N_cart, i in 1:N_cart
      V_cart_norm[i,j,k] = V_julia_raw[i,j,k] * d_c[i] * d_c[j] * d_c[k]
   end

   # Tensor Contraction
   # Contract i, j, k dimensions sequentially
   # i -> s
   V_tmp1 = reshape(C' * reshape(V_cart_norm, N_cart, :), N_sph, N_cart, N_cart)
   # j -> s (requires permutation)
   V_p1 = permutedims(V_tmp1, (2, 1, 3)) # (j_c, i_s, k_c)
   V_tmp2 = reshape(C' * reshape(V_p1, N_cart, :), N_sph, N_sph, N_cart) # (j_s, i_s, k_c)
   # k -> s
   V_p2 = permutedims(V_tmp2, (3, 2, 1)) # (k_c, i_s, j_s)
   V_tmp3 = reshape(C' * reshape(V_p2, N_cart, :), N_sph, N_sph, N_sph) # (k_s, i_s, j_s)
    
   # Restore order (i, j, k)
   V_sph = permutedims(V_tmp3, (2, 3, 1))

   return T_sph, V_sph, S_sph
end

