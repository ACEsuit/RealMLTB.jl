# Verifies the numerical accuracy of the spherical integral transformation pipeline 
# by comparing computed S, T, and V tensors against PySCF reference values.

using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using Test
using AtomsBuilder, NeighbourLists
using Unitful: ustrip, Å
using Unitful

include("../integral.jl")
include("../pyscf.jl")

# Prepare the single input: a Cartesian BasisSet
rcut  = 12.5u"Å"
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :) 
X_ang = [ustrip.(Å, r) for r in X]
mol   = Molecules.Molecule([Molecules.Atom(14, 0.0, X_ang[1]), Molecules.Atom(14, 0.0, X_ang[2])])
bs_in    = BasisSet("def2-SVP", mol.atoms; spherical=true, lib=:acsint)

# Compute all spherical integrals in one go
T, V, S = compute_spherical_AO_integrals(bs_in)

atom_strs = String[]
for atom in bs_in.atoms
    s = "$(atom.Z) $(atom.xyz[1]) $(atom.xyz[2]) $(atom.xyz[3])"
    push!(atom_strs, s)
end
pyscf_atom_str = join(atom_strs, "; ")
mol_sph  = pyscf.M(atom=pyscf_atom_str, basis="def2-svp", unit="Angstrom", cart=false)
S_py_sph = mol_sph.intor("int1e_ovlp_sph")
T_py_sph = mol_sph.intor("int1e_kin")
V_ref_sph = compute_3c_overlap_grid(mol_sph)

diff_S = norm(S - S_py_sph)
println("Overlap Matrix (S) Conversion Error: $diff_S")

diff_T = norm(T - T_py_sph)
println("Kinetic Matrix (T) Conversion Error: $diff_T")

diff_V = norm(V - V_ref_sph)
println("3cOverlap Tensor (V) (Analytic Julia vs Grid PySCF): $diff_V")

