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
using Optimisers: destructure
using JLD2

include("data/generate_data.jl")
include("data/graphene_data.jl")
#include("data/plot.jl")


# ============================================================
# 1. Generate training data
# ============================================================
println("=== Step 1: Generate training data ===")

rcut  = 20.0u"Å"
kgrid = [6, 6, 1]
temperature = 1e-2
width = 0.05   # half-width of the support
filter = FermiFilter(width)
cache_file = "training_graphene_data_cache.jld2"

if isfile(cache_file)
    println("  Found cache, loading from $cache_file ...")
    @load cache_file bs λ_ref_list ψ_ref_list filter_list proj_list kw_list G_list syslist
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
            Ecut=30,
            cone_scale=3,
            temperature
        );

    println("  Saving cache to $cache_file ...")
    @save cache_file bs λ_ref_list ψ_ref_list filter_list proj_list kw_list G_list syslist
    println("  Done.")
end

println("  nbas = $(bs.nbas), natoms = $(bs.natoms)")

# ============================================================
# 2. Train
# ============================================================
println("\n=== Step 2: Train ===")

NORB = 1
filter_σ = 0.5
cone_w = 2.0

# ----------------------------------------------------------------
# Stage A: train potential V only
# ----------------------------------------------------------------
println("\n--- Stage A: train potential only ---")

ps, st, model = multilevel_train_H(
    G_list,
    λ_ref_list,
    ψ_ref_list,
    proj_list,
    kw_list,
    bs, NORB, rcut;
    filter          = filter_list,
    filter_σ        = filter_σ,
    cone_w          = cone_w,
    model_type      = :acev_3c,
    target_Dtot     = 8,
    target_maxl     = 6,
    target_ORD      = 3,
    H_iterations    = 10,
    verbose         = true,
    train_part      = :potential,
)

# ----------------------------------------------------------------
# Stage B: freeze potential V, train minimal basis Wb only
# ----------------------------------------------------------------
println("\n--- Stage B: freeze potential, train basis only ---")

loss_H_basis(p) = RMSE_H(
    G_list,
    λ_ref_list,
    ψ_ref_list,
    proj_list,
    kw_list,
    filter_list,
    model, p, st;
    filter_σ = filter_σ,
    cone_w   = cone_w,
)

ps, hist_basis, _ = RealMLTB._bfgs_refine(
    loss_H_basis,
    ps;
    iterations = 100,
    train_part = :basis,
)

# ----------------------------------------------------------------
# Stage C: fine-tune all parameters together
# ----------------------------------------------------------------
println("\n--- Stage C: fine-tune all ---")

loss_H_all(p) = RMSE_H(
    G_list,
    λ_ref_list,
    ψ_ref_list,
    proj_list,
    kw_list,
    filter_list,
    model, p, st;
    filter_σ = filter_σ,
    cone_w   = cone_w,
)

ps, hist_all, _ = RealMLTB._bfgs_refine(
    loss_H_all,
    ps;
    iterations = 100,
    train_part = :all,
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
        model, ps, st;
        filter_σ, cone_w,
    )
    println("  Structure $i  RMSE_H = $(round(sqrt(err_H), digits=6))")
end

# ============================================================
# 4. Band structure comparison and orbital plot
# ============================================================
println("\n=== Step 4: Band structure ===")

name_comp = "cone"
save_figure = true
compute_and_save_bandstructures(
    G_list[1], syslist[1], model, ps, st;
    kgrid      = [6, 6, 1],
    keep_bands = 8,
    xc_type    = "pbe",
    psp_type   = "gth",
    sys_type   = "graphene",
    name       = name_comp,
    save_figure
)

plot_and_save_orbitals(
    G_list[1], model, ps, st, bs;
    iatom     = 1,
    orb_index = 1,
    xrange = (-5.5,5.5),
    yrange = (-8.5,2.5),
    orb_name  = L"p_z",
    name      = name_comp,
    save_figure
)
