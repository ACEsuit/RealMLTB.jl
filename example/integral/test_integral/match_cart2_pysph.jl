# Verifies that raw Cartesian integrals (S, T) from GaussianBasis.jl 
# can be accurately transformed to match PySCF's normalized spherical standard.

using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using Test

pyscf = pyimport("pyscf")
np = pyimport("numpy")

include("../pyscf.jl")

atoms = [Molecules.Atom(14, 0.0, SVector{3}([0.0, 0.0, 0.0]))]
mol_cart = pyscf.M(atom="Si 0 0 0", basis="def2-svp", unit="Angstrom", cart=true)
mol_sph  = pyscf.M(atom="Si 0 0 0", basis="def2-svp", unit="Angstrom", cart=false)

#    Use `:acsint` for Cartesian to ensure stability
bs_cart = BasisSet("def2-svp", atoms; spherical=false, lib=:acsint)

# === Step 1: Preparing Data ===
# A. Get Source Data (Raw Cartesian from Julia)
S_gb_cart = overlap(bs_cart)
T_gb_cart = kinetic(bs_cart)

# B. Get Target Data (Normalized Spherical from PySCF)
S_py_sph = mol_sph.intor("int1e_ovlp_sph")
T_py_sph = mol_sph.intor("int1e_kin")

# C. Get Intermediate Data for Transformation
#    - Transformation Matrix C
#    - Normalized Cartesian Integrals (from PySCF) to compute d_c
C = mol_sph.cart2sph_coeff()
S_py_cart = mol_cart.intor("int1e_ovlp_cart")

# D. Compute Cartesian Scaling Factors (d_c)
#    d_c = sqrt( PySCF_Cart_Diag / GB_Cart_Diag )
d_c = compute_scaling_factors(S_py_cart, S_gb_cart)

# === Step 2: Matching Overlap Matrix (S) ==
# Transform: GB Cart -> PySCF Sph
S_match = transform_gb_cart_to_pyscf_sph(S_gb_cart, C, d_c)

diff_S = norm(S_match - S_py_sph)
println("Overlap Matrix (S) Conversion Error: $diff_S")


# Step 3: Matching Kinetic Matrix (T) ===

# Transform: GB Cart -> PySCF Sph
# Note: We use the SAME scaling factors (d_c) and matrix (C) derived from Overlap
T_match = transform_gb_cart_to_pyscf_sph(T_gb_cart, C, d_c)

diff_T = norm(T_match - T_py_sph)
println("Kinetic Matrix (T) Conversion Error: $diff_T")
