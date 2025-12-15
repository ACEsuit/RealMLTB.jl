# Verifies the accuracy of analytic Cartesian 3-center integrals from Julia 
# by comparing them against numerical grid-based integration from PySCF.

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


println("=== Cartesian Integral Verification (Analytic Transformed vs Grid) ===")
atoms = [Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0])]

# Initialize Cartesian BasisSet using `:acsint` backend to avoid libcint issues
bs_cart = BasisSet("def2-svp", atoms; spherical=false, lib=:acsint)
V_julia_cart = zeros(bs_cart.nbas, bs_cart.nbas, bs_cart.nbas)

# Compute analytic 3-center integrals using Julia implementation
RealMLTB.EPI_1e3c!(V_julia_cart, bs_cart) 

# Define PySCF molecules for reference (Cartesian and Spherical)
mol_cart = pyscf.M(atom="Si 0 0 0", basis="def2-svp", unit="Angstrom", cart=true)
mol_sph  = pyscf.M(atom="Si 0 0 0", basis="def2-svp", unit="Angstrom", cart=false)

# Compute numerical reference integrals using PySCF DFT grids
V_ref_cart = compute_3c_overlap_grid(mol_cart)

# Calculate Cartesian Scaling Factors (GaussianBasis -> PySCF)
S_gb_cart = overlap(bs_cart)
S_py_cart = mol_cart.intor("int1e_ovlp_cart")
d_c = sqrt.(diag(S_py_cart) ./ diag(S_gb_cart))

# Normalize the Julia Cartesian integrals for comparison
V_julia_cart_norm = zeros(size(V_julia_cart))
N = size(V_julia_cart, 1)
for k=1:N, j=1:N, i=1:N
    V_julia_cart_norm[i,j,k] = V_julia_cart[i,j,k] * d_c[i] * d_c[j] * d_c[k]
end

diff_cart = norm(V_julia_cart_norm - V_ref_cart)
println("Cartesian Integral Error (Analytic Julia vs Grid PySCF): ", diff_cart)

