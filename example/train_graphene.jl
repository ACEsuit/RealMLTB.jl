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
include("data/plot.jl")


# ============================================================
# 1. Generate training data
# ============================================================
println("=== Step 1: Generate training data ===")

rcut  = 20.0u"Å"
kgrid = [6, 6, 1]
temperature = 1e-2
width = 0.08   # half-width of the plateau
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
            cone_scale=5,
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

NORB = 1        # number of orbitals for each atom
filter_σ = 0.6  # smoothing parameter for Fermi filter
cone_w = 1.0    # weight of the Dirac cone
gap_w = 0.5     # weight of the gap
ps, st, model = multilevel_train_H(
    G_list,
    λ_ref_list,
    ψ_ref_list,
    proj_list,
    kw_list,
    bs, NORB, rcut;
    filter = filter_list,
    filter_σ, cone_w, gap_w,
    model_type      = :acev_3c,
    target_Dtot     = 6,
    target_maxl     = 5,
    target_ORD      = 3,
    H_iterations    = 100,
    verbose         = true,
);

#= 
# ----------------------------------------------------------------
# train the band structures
# ----------------------------------------------------------------
ps, st, model = multilevel_train(
    G_list[1], λ_ref_list[1], bs, NORB, rcut;
    model_type      = :acev_3c,
    target_Dtot     = 14,
    target_maxl     = 4,
    target_ORD      = 3,
    keep_bands      = 4,
    adam_iterations = 200,
    base_iterations = 500,
    verbose         = true,
)
=#

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
        filter_σ, cone_w, gap_w,
    )
    println("  Structure $i  RMSE_H = $(round(sqrt(err_H), digits=6))")
end

# ============================================================
# 4. Band structure comparison and orbital plot
# ============================================================
println("\n=== Step 4: Band structure ===")

name_comp = "cone_gap"
save_figure = false
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
    orb_name  = "",
    name      = name_comp,
    save_figure
)
