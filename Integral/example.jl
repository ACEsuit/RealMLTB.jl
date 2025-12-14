using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using Test

include("utils/integral.jl")
include("utils/pyscf.jl")

# 1. Prepare the single input: a Cartesian BasisSet
atoms = [Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0])]

# Note: Use `:acsint` backend to ensure correct handling of raw Cartesian d-shells
bs_in = BasisSet("def2-svp", atoms; spherical=false, lib=:acsint)

# 2. Compute all spherical integrals in one go
S, T, V = compute_spherical_integrals_from_cart(bs_in)

atoms = [Molecules.Atom(14, 0.0, SVector{3}([0.0, 0.0, 0.0]))]
mol_sph  = pyscf.M(atom="Si 0 0 0", basis="def2-svp", unit="Angstrom", cart=false)
S_py_sph = mol_sph.intor("int1e_ovlp_sph")
T_py_sph = mol_sph.intor("int1e_kin")
V_ref_sph = compute_3c_overlap_grid(mol_sph)

diff_S = norm(S - S_py_sph)
println("Overlap Matrix (S) Conversion Error: $diff_S")

diff_T = norm(T - T_py_sph)
println("Kinetic Matrix (T) Conversion Error: $diff_T")

diff_V = norm(V - V_ref_sph)
println("3cOverlap Tensor (V) (Analytic Julia vs Grid PySCF): $diff_V")

