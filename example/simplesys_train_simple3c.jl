using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

rcut  = 12.5u"Å"
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :) 
X_ang = [ustrip.(Å, r) for r in X]
mol   = Molecules.Molecule([Molecules.Atom(14, 0.0, X_ang[1]), Molecules.Atom(14, 0.0, X_ang[2])])
bs    = BasisSet("def2-SVP", mol.atoms; spherical=false, lib=:acsint)


NORB  = 2
model = simple_3c_model(bs, NORB)
ps,st = LuxCore.setup(MersenneTwister(1234), model)

kgrid = [6, 7, 8]
keep_bands    = 4
dataset = generate_band_data(sys, kgrid; keep_bands = keep_bands)
klist = getfield.(dataset, 1)
λ_ref = getfield.(dataset, 2)

G     = RealMLTB.nlist2graph(bs, nlist, sys, klist)
ps, x_history = train(G, model, ps, st, keep_bands, λ_ref; iterations = 3000)
error         = RMSE(G, λ_ref, model, ps, st; keep_bands = keep_bands)

compute_and_save_bandstructures(sys, model, ps, st; kgrid = kgrid, keep_bands = keep_bands)
