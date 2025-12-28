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
mol   = Molecules.Molecule([
    Molecules.Atom(14, 0.0, X_ang[1]),
    Molecules.Atom(14, 0.0, X_ang[2]),
])

bs    = BasisSet("def2-SVP", mol.atoms; spherical=false, lib=:acsint)
lmax, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs)

klist = []
G = RealMLTB.nlist2graph(bs, nlist, sys, klist);

# ---- Wv / ACE equivariant tensor output ----
modelWv, psWv, stWv = RealMLTB.build_ace_equivariant_model(n_channels, rcut);
test_equivariance(modelWv, psWv, stWv, G)
test_V_pair_equivariance(modelWv, psWv, stWv, G, bs; tol=1e-5)

# ---- Case A ----
nψ = (1, 1, 0, 0, 0)   # NORB = 1*1 + 1*3 = 4
modelWpsiA, psWpsiA, stWpsiA = RealMLTB.build_ace_caseA_Wpsi_model(n_channels, nψ, rcut);
test_Wpsi_caseA(modelWpsiA, psWpsiA, stWpsiA, G, bs, n_channels, nψ; tol=1e-5)
test_Hpsi_caseA(modelWv, psWv, stWv, modelWpsiA, psWpsiA, stWpsiA, G, bs, n_channels, nψ; tol=1e-5)

# ---- Case B ----
NORB = 4
modelWpsiB, psWpsiB, stWpsiB = RealMLTB.build_ace_caseB_Wpsi_model(n_channels, NORB, rcut);
test_Wpsi_caseB(modelWpsiB, psWpsiB, stWpsiB, G, bs, n_channels, NORB; tol=1e-5)
test_Hpsi_caseB(modelWv, psWv, stWv, modelWpsiB, psWpsiB, stWpsiB, G, bs, n_channels, NORB; tol=1e-5)
