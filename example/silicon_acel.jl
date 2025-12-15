using RealMLTB
using GaussianBasis, Molecules
using AtomsBuilder, NeighbourLists
using Unitful: ustrip, Å
using Unitful
using Lux, Random
using Rotations, WignerD, LinearAlgebra, StaticArrays, Test
using EquivariantTensors
using EquivariantTensors.O3: Ctran

include("integral/integral.jl")
include("utils/rotation.jl")
include("utils/ace_model.jl")
include("utils/build_graph.jl")
include("utils/assemble.jl")

rcut  = 12.5u"Å"
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :) 
X_ang = [ustrip.(Å, r) for r in X]
mol   = Molecules.Molecule([Molecules.Atom(14, 0.0, X_ang[1]), Molecules.Atom(14, 0.0, X_ang[2])])
bs    = BasisSet("def2-SVP", mol.atoms; spherical=false, lib=:acsint)
lmax, n_channels = lmax_and_nchannels_per_atom(bs)

model, ps, st = build_ace_model(n_channels, rcut)

NORB  = 2
model = simple_3c_model(bs, NORB)
ps,st = LuxCore.setup(MersenneTwister(1234), model)

kgrid = [3, 4, 5]
keep_bands    = 4
dataset = generate_band_data(sys, kgrid; keep_bands = keep_bands)
klist = getfield.(dataset, 1)
λ_ref = getfield.(dataset, 2)

G = nlist2graph(bs, nlist, sys, klist)

ps, x_history = train(G, model, ps, st, keep_bands, λ_ref; iterations = 3000)
error         = RMSE(G, λ_ref, model, ps, st; keep_bands = keep_bands)

λ_pred,  = model(G, ps, st)
λ_pred = getindex.(λ_pred, Ref(1:keep_bands))

compute_and_save_bandstructures(sys, model, ps, st; kgrid = kgrid, keep_bands = keep_bands)
