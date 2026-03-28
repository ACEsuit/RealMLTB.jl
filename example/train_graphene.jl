using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using NeighbourLists
using EquivariantTensors
using Unitful
using UnitfulAtomic
using StaticArrays
using LinearAlgebra
using Random
using Lux
using Zygote
using MatrixFuns
using Optimisers: destructure

include("data/generate_data.jl")
include("data/graphene_ham.jl")

# ============================================================
# 1. Generate training data
# ============================================================
using JLD2
println("=== Step 1: Generate training data ===")

rcut  = 5.0u"Å"
kgrid = [6, 6, 1]
cache_file = "training_data_cache.jld2"

if isfile(cache_file)
    println("  Found cache, loading from $cache_file ...")
    @load cache_file bs H_ref_list G_list syslist
    println("  Loaded $(length(syslist)) structures from cache.")
else
    println("  No cache found, generating data ...")
    sys0 = build_graphene()
    Random.seed!(0)
    syslist = [sys0; [perturb(sys0; ϵ = 0.02) for _ in 1:3]]
    println("  $(length(syslist)) structures")

    bs, H_ref_list, G_list = build_ref_and_graph_lists_sph_graphene(
    syslist, rcut, kgrid; basis_name = "def2-SVP")

    println("  Saving cache to $cache_file ...")
    @save cache_file bs H_ref_list G_list syslist
    println("  Done.")
end

println("  nbas = $(bs.nbas), natoms = $(bs.natoms)")

# ============================================================
# 2. Define spectral filter for RMSE_H
#    Fermi-Dirac:  f(x) = 1 / (1 + exp((x - μ) / β))
# ============================================================
μ = 0.0
β = 0.1
fermi_filter = x -> 1 / (1 + exp((x - μ) / β))

# ============================================================
# 3. Train
# ============================================================
println("\n=== Step 2: Train ===")

NORB       = 4
ps, st, model = multilevel_train_H(
    G_list, bs, NORB, rcut;
    model_type    = :ace3c_B,
    target_Dtot   = 6,
    target_maxl   = 4,
    target_ORD    = 2,
    H_ref_AO_list = H_ref_list,
    filter        = fermi_filter,
    H_iterations  = 500,
    seed          = 42,
    verbose       = true,
)

println("\n  Training done.")

# ============================================================
# 4. Evaluate on training set
# ============================================================
println("\n=== Step 3: Evaluate ===")

for i in eachindex(G_list)
    err_λ = RMSE(G_list[i], λ_ref_list[i], model, ps, st; keep_bands = keep_bands)
    err_H = RMSE_H(G_list[i], H_ref_list[i], model, ps, st, fermi_filter)
    println("  Structure $i  RMSE_λ = $(round(err_λ, digits=6)) Ha" *
            "   RMSE_H = $(round(err_H, digits=6))")
end

# ============================================================
# 5. Band structure comparison
# ============================================================
println("\n=== Step 4: Band structure ===")

compute_and_save_bandstructures(G_list[1], syslist[1], model, ps, st;
    kgrid      = [6, 6, 1],
    keep_bands = keep_bands,
    name       = "graphene_bands_rmseH.png")
println("  Saved graphene_bands_rmseH.png")

# ============================================================
# 6. Predict on new structure
# ============================================================
println("\n=== Step 5: Predict on new structure ===")

sys_pred = perturb(build_graphene(); ϵ = 0.02)

bs_pred, H_ref_pred_list, G_pred_list = build_ref_and_graph_lists_sph_graphene(
    [sys_pred], rcut, kgrid; basis_name = "def2-SVP")

st_pred   = update_basisset(st, bs_pred)
out, _, _ = model(G_pred_list[1], ps, st_pred)
println("  Predicted $(length(out.λ)) k-points, $(length(out.λ[1])) bands each.")

err_H_pred = RMSE_H(G_pred_list[1], H_ref_pred_list[1], model, ps, st_pred, fermi_filter)
println("  New structure RMSE_H = $(round(err_H_pred, digits=6))")

compute_and_save_bandstructures(G_pred_list[1], sys_pred, model, ps, st_pred;
    kgrid      = [6, 6, 1],
    keep_bands = keep_bands,
    name       = "graphene_pred_rmseH.png")
println("  Saved graphene_pred_rmseH.png")