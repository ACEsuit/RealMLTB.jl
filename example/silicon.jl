using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: Å, ustrip
using LuxCore
using Random

include("generate_data.jl")
include("plot_bands.jl")

rcut  = 12.5u"Å"
NORB  = 2
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :) 
X_ang = [ustrip.(Å, r) for r in X] 
mol   = Molecules.Molecule([Molecules.Atom(14, 0.0, X_ang[1]), Molecules.Atom(14, 0.0, X_ang[2])])
bs    = BasisSet("sto-3g", mol.atoms; spherical=false, lib=:acsint)

kgrid = [3, 4, 5]
dataset = generate_data(sys, kgrid)
klist = getfield.(dataset, 1)
G     = nlist2graph(bs, nlist, sys, klist)

model = simple_3c_model(bs, NORB)
ps,st = LuxCore.setup(MersenneTwister(1234), model)

keep_bands    = 4
ps, x_history = train(G, model, ps, st, keep_bands, dataset; iterations = 1000)
λ_ref         = getfield.(dataset, 2)
error         = RMSE(G, λ_ref, model, ps, st; keep_bands = keep_bands)

λ_pred,  = model(G, ps, st)
data = [(dataset[i][1], λ_pred[i]) for i = 1:length(dataset)]
compute_and_save_bandstructures(sys, model, ps, st; kgrid = kgrid, keep_bands = keep_bands)