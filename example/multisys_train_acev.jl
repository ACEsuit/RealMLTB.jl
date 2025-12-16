using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")
include("integral/integral.jl")
include("utils/build_graph.jl")

rcut = 12.5u"Å"
sys  = bulk(:Si)

syslist = [sys; [perturb(sys) for _ in 1:10]]
NORB  = 2
kgrid = [6, 7, 8]
keep_bands    = 4
bs, λ_ref_list, G_list = build_ref_and_graph_lists_sph(syslist, rcut, kgrid; keep_bands= keep_bands)

model  = acev_3c_model(bs, NORB, rcut; Dtot = 5, maxl = 5, ORD = 2)
ps, st = LuxCore.setup(MersenneTwister(1234), model)

ps, x_history = train(G_list, model, ps, st, keep_bands, λ_ref_list; iterations = 1000)
error = RMSE(G_list, λ_ref_list, model, ps, st; keep_bands = keep_bands)

syslist2 = [perturb(sys)]
bs, λ_ref_list2, G_list2 = build_ref_and_graph_lists_sph(syslist2, rcut, kgrid; keep_bands= keep_bands)
G, λ_ref = G_list2[1], λ_ref_list2[1]

error = RMSE(G, λ_ref, model, ps, st; keep_bands = keep_bands)

compute_and_save_bandstructures(perturb(sys), model, ps, st; kgrid = kgrid, keep_bands = keep_bands)
