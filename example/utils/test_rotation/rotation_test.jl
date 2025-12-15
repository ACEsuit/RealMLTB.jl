# Verifies that both the ACE model predictions 
# and the computed pair integrals respect rotational symmetry 
# on a Silicon bulk system.

using RealMLTB
using GaussianBasis, Molecules
using AtomsBuilder, NeighbourLists
using Unitful: ustrip, Å
using Unitful
using Lux, Random
using Rotations, WignerD, LinearAlgebra, StaticArrays, Test
using EquivariantTensors
using EquivariantTensors.O3: Ctran

include("../../integral/integral.jl")
include("../rotation.jl")
include("../build_graph.jl")

rcut  = 12.5u"Å"
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :) 
X_ang = [ustrip.(Å, r) for r in X]
mol   = Molecules.Molecule([Molecules.Atom(14, 0.0, X_ang[1]), Molecules.Atom(14, 0.0, X_ang[2])])
bs    = BasisSet("def2-SVP", mol.atoms; spherical=false, lib=:acsint)
lmax, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs)

model, ps, st = RealMLTB.build_ace_model(n_channels, rcut)

klist = []
G = nlist2graph(bs, nlist, sys, klist)
test_equivariance(model, ps, st, G)
test_V_pair_equivariance(model, ps, st, G, bs; tol=1e-5)


