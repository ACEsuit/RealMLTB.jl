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

# simple3c
bs_cart    = BasisSet("sto-3g", mol.atoms; spherical=false, lib=:acsint)

NORB  = 2
model_simple3c = simple_3c_model(bs_cart, NORB)
ps_simple3c,st_simple3c = LuxCore.setup(MersenneTwister(1234), model_simple3c)

kgrid = [6, 7, 8]
keep_bands    = 4
dataset = generate_band_data(sys, kgrid; keep_bands = keep_bands)
klist = getfield.(dataset, 1)
λ_ref = getfield.(dataset, 2)

G_cart     = RealMLTB.nlist2graph(bs_cart, nlist, sys, klist)
ps_simple3c, x_history = train(G_cart, model_simple3c, ps_simple3c, st_simple3c, keep_bands, λ_ref; iterations = 3000)
error_simple3c         = RMSE(G_cart, λ_ref, model_simple3c, ps_simple3c, st_simple3c; keep_bands = keep_bands)

compute_and_save_bandstructures(G_cart, sys, model_simple3c, ps_simple3c, st_simple3c;name = "simple3c", kgrid = kgrid, keep_bands = keep_bands)


# ace_v_3c
include("integral/integral.jl")
include("utils/build_graph.jl")

bs_sph    = BasisSet("def2-SVP", mol.atoms; spherical=true, lib=:acsint)
model_acev = acev_3c_model(bs_sph, NORB, rcut; Dtot = 5, maxl = 5, ORD = 2)
ps_acev,st_acev = LuxCore.setup(MersenneTwister(1234), model_acev)

# Note: Using the externally defined `nlist2graph` (PyCall-dependent) 
# instead of the core RealMLTB package version.
G_sph = nlist2graph(bs_sph, nlist, sys, klist)

ps_acev, x_history = train(G_sph, model_acev, ps_acev, st_acev, keep_bands, λ_ref; iterations = 3000)
error_acev         = RMSE(G_sph, λ_ref, model_acev, ps_acev, st_acev; keep_bands = keep_bands)

compute_and_save_bandstructures(G_sph, sys, model_acev, ps_acev, st_acev; name = "acev3c", kgrid = kgrid, keep_bands = keep_bands)


# ace_3c
model_ace = ace_3c_model(bs_sph, NORB, rcut; Dtot = 2, maxl = 2, ORD = 1)
ps_ace,st_ace = LuxCore.setup(MersenneTwister(1234), model_ace)

ps_ace, x_history = train(G_sph, model_ace, ps_ace, st_ace, keep_bands, λ_ref; iterations = 3000)
error_ace         = RMSE(G_sph, λ_ref, model_ace, ps_ace, st_ace; keep_bands = keep_bands)

compute_and_save_bandstructures(G_sph, sys, model_ace, ps_ace, st_ace; name = "ace3c", kgrid = kgrid, keep_bands = keep_bands)
