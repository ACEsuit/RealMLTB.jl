# Verifies the accuracy of the full Cartesian-to-Spherical transformation pipeline 
# for 3-center integrals by comparing analytic Julia results 
# against PySCF numerical grid integration.

using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using RealMLTB
using Test

pyscf = pyimport("pyscf")
dft = pyimport("pyscf.dft")
np = pyimport("numpy")

include("../pyscf.jl")


println("=== Spherical Integral Verification (Analytic Transformed vs Grid) ===")
atoms = [Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0])]
mol_cart = pyscf.M(atom="Si 0 0 0", basis="def2-svp", cart=true)
mol_sph  = pyscf.M(atom="Si 0 0 0", basis="def2-svp", cart=false)

# Compute numerical reference integrals using PySCF DFT grids
V_ref_sph = compute_3c_overlap_grid(mol_sph)

# Computing analytic Cartesian integrals (Julia)
# Use `:acsint` to ensure correct Cartesian d-shell handling
bs_cart = BasisSet("def2-svp", atoms; spherical=false, lib=:acsint)
V_julia_cart = zeros(bs_cart.nbas, bs_cart.nbas, bs_cart.nbas)
RealMLTB.EPI_1e3c!(V_julia_cart, bs_cart) 

# Preparing transformation matrices and scaling factors
# Get Transformation Matrix C (Cartesian -> Spherical)
C = mol_sph.cart2sph_coeff()

# Scaling Factors: Cartesian (d_c)
S_gb_cart = overlap(bs_cart)
S_py_cart = mol_cart.intor("int1e_ovlp_cart")
d_c = sqrt.(diag(S_py_cart) ./ diag(S_gb_cart))

# Scaling Factors: Spherical (d_s)
bs_sph = BasisSet("def2-svp", atoms; spherical=true, lib=:libcint)
S_gb_sph = overlap(bs_sph)
S_py_sph = mol_sph.intor("int1e_ovlp_sph")
d_s = sqrt.(diag(S_py_sph) ./ diag(S_gb_sph))

# Step A: Full Tensor Transformation (Cartesian -> Spherical)
# Note: This returns unnormalized integrals (GaussianBasis convention)
V_julia_sph_unnorm = transform_3c_tensor_full(V_julia_cart, C, d_c, d_s)

# Step B: Final Normalization
# Apply spherical scaling factors to match PySCF's normalized grid reference
V_julia_sph_norm = zeros(size(V_julia_sph_unnorm))
N_sph = size(V_julia_sph_unnorm, 1)

for k=1:N_sph, j=1:N_sph, i=1:N_sph
    V_julia_sph_norm[i,j,k] = V_julia_sph_unnorm[i,j,k] * d_s[i] * d_s[j] * d_s[k]
end

diff_sph = norm(V_julia_sph_norm - V_ref_sph)
println("Spherical Integral Error (Analytic Julia vs Grid PySCF): ", diff_sph)
