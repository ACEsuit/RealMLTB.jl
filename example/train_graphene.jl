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
using JLD2

include("data/generate_data.jl")
include("data/graphene_ham.jl")

# ============================================================
# 1. Generate training data
# ============================================================
println("=== Step 1: Generate training data ===")

rcut  = 5.0u"Å"
kgrid = [6, 6, 1]
filter = FermiFilter()
cache_file = "training_data_cache.jld2"

if isfile(cache_file)
    println("  Found cache, loading from $cache_file ...")
    @load cache_file bs λ_ref_list ψ_ref_list filter_list proj_list kw_list G_list syslist
    filter_list = [filter.filter_new for filter in filter_list]
    println("  Loaded $(length(syslist)) structures from cache.")
else
    println("  No cache found, generating data ...")
    sys0 = build_graphene()
    Random.seed!(0)
    syslist = [sys0; [perturb(sys0; ϵ = 0.02) for _ in 1:1]]
    println("  $(length(syslist)) structures")

    bs, λ_ref_list, ψ_ref_list, filter_list, proj_list, kw_list, G_list =
        build_ref_and_graph_lists_sph_graphene(
            syslist, rcut, kgrid, filter;
            basis_name = "def2-SVP",
        )

    println("  Saving cache to $cache_file ...")
    @save cache_file bs λ_ref_list ψ_ref_list filter_list proj_list kw_list G_list syslist
    println("  Done.")
end

println("  nbas = $(bs.nbas), natoms = $(bs.natoms)")

# ============================================================
# 2. Train
# ============================================================
println("\n=== Step 2: Train ===")

NORB = 4

using Random

model = ace3c_B_model(bs, NORB, rcut;
    Dtot = 4,
    maxl = 3,
    ORD  = 1,
    rng  = MersenneTwister(42),
)

import LuxCore
rng = MersenneTwister(42)
ps, st = LuxCore.setup(rng, model)

G = G_list[1]
λ = λ_ref_list[1]
ψ = ψ_ref_list[1]
proj = proj_list[1]
kw = kw_list[1]
f = filter_list[1]

RMSE_H(G, λ, ψ, proj, kw, f, model, ps, st)
Zygote.gradient(p -> RMSE_H(G, λ, ψ, proj, kw, f, model, p, st), ps)



ps, st, model = multilevel_train_H(
    G_list,
    λ_ref_list,
    ψ_ref_list,
    proj_list,
    kw_list,
    bs,
    NORB,
    rcut;
    filter = filter_list,
    model_type      = :ace3c_B,
    target_Dtot     = 6,
    target_maxl     = 4,
    target_ORD      = 2,
    H_iterations    = 1,
    verbose         = true,
)

println("\n  Training done.")

# ============================================================
# 3. Evaluate on training set
# ============================================================
println("\n=== Step 3: Evaluate ===")

for i in eachindex(G_list)
    err_H = RMSE_H(
        G_list[i],
        λ_ref_list[i],
        ψ_ref_list[i],
        proj_list[i],
        kw_list[i],
        filter_list[i],
        model, ps, st,
    )
    println("  Structure $i  RMSE_H = $(round(err_H, digits=6))")
end

# ============================================================
# 4. Band structure comparison
# ============================================================
println("\n=== Step 4: Band structure ===")

compute_and_save_bandstructures(
    G_list[1], syslist[1], model, ps, st;
    kgrid      = [6, 6, 1],
    keep_bands = 4,
    name       = "graphene_bands_rmseH.png",
)
println("  Saved graphene_bands_rmseH.png")

# ============================================================
# 5. Predict on new structure
# ============================================================
println("\n=== Step 5: Predict on new structure ===")

sys_pred = perturb(build_graphene(); ϵ = 0.02)

bs_pred, λ_pred_ref_list, ψ_pred_ref_list, filter_pred_list, proj_pred_list, kw_pred_list, G_pred_list =
    build_ref_and_graph_lists_sph_graphene(
        [sys_pred], rcut, kgrid, filter;
        basis_name = "def2-SVP",
    )

st_pred = update_basisset(st, bs_pred)

out, _ = model(G_pred_list[1], ps, st_pred)
println("  Predicted $(length(out.λ)) k-points, $(length(out.λ[1])) bands each.")

err_H_pred = RMSE_H(
    G_pred_list[1],
    λ_pred_ref_list[1],
    ψ_pred_ref_list[1],
    proj_pred_list[1],
    kw_pred_list[1],
    filter_pred_list[1],
    model, ps, st_pred,
)
println("  New structure RMSE_H = $(round(err_H_pred, digits=6))")

compute_and_save_bandstructures(
    G_pred_list[1], sys_pred, model, ps, st_pred;
    kgrid      = [6, 6, 1],
    keep_bands = 4,
    name       = "graphene_pred_rmseH.png",
)
println("  Saved graphene_pred_rmseH.png")