using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

rcut = 12.5u"Å"
sys  = bulk(:Si)

syslist = [sys; [perturb(sys) for _ in 1:10]]
NORB  = 2
kgrid = [6, 7, 8]
keep_bands    = 4
bs, λ_ref_list, G_list = build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands=keep_bands)

model  = simple_3c_model(bs, NORB)
ps, st = LuxCore.setup(MersenneTwister(1234), model)

ps, x_history = train(G_list, model, ps, st, keep_bands, λ_ref_list; iterations = 1000)
error         = RMSE(G_list, λ_ref_list, model, ps, st; keep_bands = keep_bands)

syslist = [perturb(sys)]
bs, λ_ref_list2, G_list2 = build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands= keep_bands)
sys, G, λ_ref = syslist[1], G_list2[1], λ_ref_list2[1]

λ_pred,  = model(G, ps, st)
λ_pred = getindex.(λ_pred, Ref(1:keep_bands))
norm(λ_pred - λ_ref)

compute_and_save_bandstructures(sys, model, ps, st; kgrid = kgrid, keep_bands = keep_bands)
