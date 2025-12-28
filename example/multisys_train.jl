using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

# -----------------------------
# 0) dataset list
# -----------------------------
syslist = [bulk(:Si); [perturb(bulk(:Si)) for _ in 1:10]]
sys_pert = perturb(bulk(:Si))

NORB       = 2
kgrid      = [6, 7, 8]
keep_bands = 4
rcut       = 12.5u"Å"

# -----------------------------
# 1) CART models (simple3c)
# -----------------------------
bs_cart, λ_ref_list_cart, G_list_cart =
    build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands=keep_bands);

model_simple3c = simple_3c_model(bs_cart, NORB);
ps_simple3c, st_simple3c = LuxCore.setup(MersenneTwister(1234), model_simple3c);

# train on cart list
ps_simple3c, x_history_simple3c =
    train(G_list_cart, model_simple3c, ps_simple3c, st_simple3c,
          keep_bands, λ_ref_list_cart; iterations=1000);

error_simple3c =
    RMSE(G_list_cart, λ_ref_list_cart, model_simple3c, ps_simple3c, st_simple3c;
         keep_bands=keep_bands)

# perturb eval + plot (cart)
_, λ_ref_list_cart_pert, G_list_cart_pert =
    build_ref_and_graph_lists_cart([sys_pert], rcut, kgrid; keep_bands=keep_bands);

G_cart_pert, λ_ref_pert = G_list_cart_pert[1], λ_ref_list_cart_pert[1];

error_simple3c_pert =
    RMSE(G_cart_pert, λ_ref_pert, model_simple3c, ps_simple3c, st_simple3c;
         keep_bands=keep_bands)

compute_and_save_bandstructures(G_cart_pert, sys_pert, model_simple3c, ps_simple3c, st_simple3c;
    name="simple3c_perturb", kgrid=kgrid, keep_bands=keep_bands
)

# -----------------------------
# 2) SPH models (ACE)
# -----------------------------
include("integral/integral.jl")
include("utils/build_graph.jl")

bs_sph, λ_ref_list_sph, G_list_sph =
    build_ref_and_graph_lists_sph(syslist, rcut, kgrid; keep_bands=keep_bands);

_, _, G_list_sph_pert =
    build_ref_and_graph_lists_sph([sys_pert], rcut, kgrid; keep_bands=keep_bands);

G_sph_pert = G_list_sph_pert[1];

base_iterations = 1000
# --- (A) acev_3c ---
ps_acev, st_acev, model_acev =
    multilevel_train(G_list_sph, λ_ref_list_sph, bs_sph, NORB, rcut;
        model_type      = :acev_3c,
        target_Dtot     = 5,
        target_maxl     = 5,
        target_ORD      = 3,
        keep_bands      = keep_bands,
        base_iterations = base_iterations,
        seed            = 1234,
        verbose         = true
    );

error_acev =
    RMSE(G_list_sph, λ_ref_list_sph, model_acev, ps_acev, st_acev;
         keep_bands=keep_bands)

error_acev_pert =
    RMSE(G_sph_pert, λ_ref_pert, model_acev, ps_acev, st_acev;
         keep_bands=keep_bands)

compute_and_save_bandstructures(G_sph_pert, sys_pert, model_acev, ps_acev, st_acev;
    name="acev3c_perturb", kgrid=kgrid, keep_bands=keep_bands
)

# --- (B) ace3c_A : Wpsi CaseA (two-sided / co-rotating) ---
nψ = (2, 0, 0, 0, 0)

ps_aceA, st_aceA, model_aceA =
    multilevel_train(G_list_sph, λ_ref_list_sph, bs_sph, NORB, rcut;
        model_type      = :ace3c_A, 
        nψ              = nψ,
        target_Dtot     = 5,
        target_maxl     = 5,
        target_ORD      = 3,
        keep_bands      = keep_bands,
        base_iterations = base_iterations,
        seed            = 1234,
        verbose         = true
    );

error_aceA =
    RMSE(G_list_sph, λ_ref_list_sph, model_aceA, ps_aceA, st_aceA;
         keep_bands=keep_bands)

error_aceA_pert =
    RMSE(G_sph_pert, λ_ref_pert, model_aceA, ps_aceA, st_aceA;
         keep_bands=keep_bands)

compute_and_save_bandstructures(G_sph_pert, sys_pert, model_aceA, ps_aceA, st_aceA;
    name="ace3c_A_perturb", kgrid=kgrid, keep_bands=keep_bands
)


# --- (C) ace3c_B : Wpsi CaseB (right action) ---
ps_aceB, st_aceB, model_aceB =
    multilevel_train(G_list_sph, λ_ref_list_sph, bs_sph, NORB, rcut;
        model_type      = :ace3c_B,   # <<< CaseB
        target_Dtot     = 5,
        target_maxl     = 5,
        target_ORD      = 3,
        keep_bands      = keep_bands,
        base_iterations = base_iterations,
        seed            = 1234,
        verbose         = true
    );

error_aceB =
    RMSE(G_list_sph, λ_ref_list_sph, model_aceB, ps_aceB, st_aceB;
         keep_bands=keep_bands)

error_aceB_pert =
    RMSE(G_sph_pert, λ_ref_pert, model_aceB, ps_aceB, st_aceB;
         keep_bands=keep_bands)

compute_and_save_bandstructures(G_sph_pert, sys_pert, model_aceB, ps_aceB, st_aceB;
    name="ace3c_B_perturb", kgrid=kgrid, keep_bands=keep_bands
)


println("\n==============================")
println("Summary RMSE (train-set)")
println("==============================")
println("simple3c  : ", error_simple3c)
println("acev3c    : ", error_acev)
println("ace3c_B   : ", error_aceB)
println("ace3c_A   : ", error_aceA)

println("\n==============================")
println("Perturb RMSE (1 structure)")
println("==============================")
println("simple3c  : ", error_simple3c_pert)
println("acev3c    : ", error_acev_pert)
println("ace3c_B   : ", error_aceB_pert)
println("ace3c_A   : ", error_aceA_pert)
