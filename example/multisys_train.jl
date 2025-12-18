using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

syslist = [bulk(:Si); [perturb(bulk(:Si)) for _ in 1:10]]
NORB  = 2
kgrid = [6, 7, 8]
keep_bands    = 4
rcut = 12.5u"Å"
bs_cart, λ_ref_list, G_list_cart = build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands=keep_bands)

sys_pert = perturb(bulk(:Si))
#sys_pert = perturb(bulk(:Si, cubic=true))

model_simple3c  = simple_3c_model(bs_cart, NORB)
ps_simple3c, st_simple3c = LuxCore.setup(MersenneTwister(1234), model_simple3c)

ps_simple3c, x_history = train(G_list_cart, model_simple3c, ps_simple3c, st_simple3c, keep_bands, λ_ref_list; iterations = 1000)
error_simple3c         = RMSE(G_list_cart, λ_ref_list, model_simple3c, ps_simple3c, st_simple3c; keep_bands = keep_bands)

_, λ_ref_list_pert, G_list_cart_pert = build_ref_and_graph_lists_cart([sys_pert], rcut, kgrid; keep_bands= keep_bands)
G_cart_pert, λ_ref_pert = G_list_cart_pert[1], λ_ref_list_pert[1]
error_simple3c_pert = RMSE(G_cart_pert, λ_ref_pert, model_simple3c, ps_simple3c, st_simple3c; keep_bands = keep_bands)
compute_and_save_bandstructures(G_cart_pert, sys_pert, model_simple3c, ps_simple3c, st_simple3c; name = "simple3c_perturb", kgrid = kgrid, keep_bands = keep_bands)


# ace model
include("integral/integral.jl")
include("utils/build_graph.jl")

bs_sph, λ_ref_list, G_list_sph = build_ref_and_graph_lists_sph(syslist, rcut, kgrid; keep_bands=keep_bands)
_, _, G_list_sph_pert = build_ref_and_graph_lists_sph([sys_pert], rcut, kgrid; keep_bands= keep_bands)
G_sph_pert = G_list_sph_pert[1]

# acev_3c
ps_acev, st_acev, model_acev = multilevel_train(G_list_sph, λ_ref_list, bs_sph, NORB, rcut; model_type = :acev_3c,
                                target_Dtot = 5, target_maxl = 5, target_ORD = 3,
                                keep_bands = keep_bands,
                                base_iterations = 100)
error_acev = RMSE(G_list, λ_ref_list, model_acev, ps_acev, st_acev; keep_bands = keep_bands)

error_acev_pert = RMSE(G_sph_pert, λ_ref_pert, model_acev, ps_acev, st_acev; keep_bands = keep_bands)
compute_and_save_bandstructures(G_sph_pert, sys_pert, model_acev, ps_acev, st_acev; name = "acev3c_perturb", kgrid = kgrid, keep_bands = keep_bands)


# ace_3c
ps_ace, st_ace, model_ace = multilevel_train(G_list_sph, λ_ref_list, bs_sph, NORB, rcut; model_type = :ace_3c,
                                target_Dtot = 5, target_maxl = 5, target_ORD = 3,
                                keep_bands = keep_bands,
                                base_iterations = 3000)
error_ace = RMSE(G_list, λ_ref_list, model_ace, ps_ace, st_ace; keep_bands = keep_bands)

error_ace_pert = RMSE(G_sph_pert, λ_ref_pert, model_ace, ps_ace, st_ace; keep_bands = keep_bands)
compute_and_save_bandstructures(G_sph_pert, sys_pert, model_ace, ps_ace, st_ace; name = "ace3c_perturb", kgrid = kgrid, keep_bands = keep_bands)
