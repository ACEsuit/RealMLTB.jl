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

include("data/generate_data.jl") 
include("data/plot_bands.jl") 

# ============================================================
# 1. Generate training data
# ============================================================
using JLD2
println("=== Step 1: Generate training data ===")

rcut  = 5.0u"Å"
kgrid = [4, 4, 4]
cache_file = "training_data_cache.jld2"

if isfile(cache_file)
    println("  Found cache, loading from $cache_file ...")
    @load cache_file bs λ_ref_list G_list syslist
    println("  Loaded $(length(syslist)) structures from cache.")
else
    println("  No cache found, generating data ...")
    sys0 = bulk(:Si)
    Random.seed!(0)
    syslist = [sys0; [perturb(sys0; ϵ = 0.02) for _ in 1:3]]
    println("  $(length(syslist)) structures")

    bs, λ_ref_list, G_list = build_ref_and_graph_lists_sph(
        syslist, rcut, kgrid;
        keep_bands = 4,
        basis_name = "def2-SVP",
    )

    println("  Saving cache to $cache_file ...")
    @save cache_file bs λ_ref_list G_list syslist
    println("  Done.")
end

println("  nbas = $(bs.nbas), natoms = $(bs.natoms)")

# ============================================================
# 2. Build model
# ============================================================
println("\n=== Step 2: Build model ===")

NORB = 4   # number of compressed orbitals per atom

# Choose one of:
#   acev_3c_model   — Wv from ACE, Wb plain parameter  (simplest)
ps, st, model = multilevel_train(
    G_list[1], λ_ref_list[1], bs, NORB, rcut;
    model_type       = :acev_3c, 
    target_Dtot      = 6,
    target_maxl      = 4,
    target_ORD       = 2,
    keep_bands       = 4,
    adam_iterations  = 200, 
    base_iterations  = 20, 
    seed             = 42,
    verbose          = true,
);

#   ace3c_B_model   — Wv from ACE, Wb from equivariant ACE (Case B)
ps, st, model = multilevel_train(
    G_list[1], λ_ref_list[1], bs, NORB, rcut;
    model_type       = :ace3c_B,
    target_Dtot      = 6,
    target_maxl      = 4,
    target_ORD       = 2,
    keep_bands       = 4,
    adam_iterations  = 200, 
    base_iterations  = 20, 
    seed             = 42,
    verbose          = true,
);

#   ace3c_A_model   — Wv from ACE, Wb from invariant ACE (Case A)
nψ = (1, 1, 0, 0, 0)
ps, st, model = multilevel_train(
    G_list[1], λ_ref_list[1], bs, NORB, rcut;
    model_type       = :ace3c_A,
    nψ             = nψ,
    target_Dtot      = 6,
    target_maxl      = 4,
    target_ORD       = 2,
    keep_bands       = 4,
    adam_iterations  = 200, 
    base_iterations  = 20, 
    seed             = 42,
    verbose          = true,
);

println("\n  Model trained on structure 1.")

# ============================================================
# 3. Fine-tune on the full dataset
# ============================================================
println("\n=== Step 3: Fine-tune on full dataset ===")

ps_ft, _ = train(
    G_list, model, ps, st, 4, λ_ref_list;
    adam_iterations  = 200, 
    iterations = 20,
);

ps = ps_ft;

# ============================================================
# 4. Evaluate on training set
# ============================================================
println("\n=== Step 4: Evaluate ===")

for i in eachindex(G_list)
    err = RMSE(G_list[i], λ_ref_list[i], model, ps, st; keep_bands = 4)
    println("  Structure $i  RMSE = $(round(err, digits=6)) Hartree")
end

# ============================================================
# 5. Band structure comparison on equilibrium structure
# ============================================================
println("\n=== Step 5: Band structure comparison ===")

i = 2
compute_and_save_bandstructures(G_list[i], syslist[i], model, ps, st;
    kgrid      = [4, 4, 4], 
    keep_bands = 4,
    name       = "si_structure_$(i).png",)
println("  Saved bandstructure_si_$(i).png")

# ============================================================
# 6. Predict on new structure
# ============================================================
sys_pred = perturb(bulk(:Si, cubic=true))#; ϵ = 0.02)

bs_pred, λ_pred, G_pred = build_ref_and_graph_lists_sph(
        [sys_pred], rcut, kgrid;
        keep_bands = 4,
        basis_name = "def2-SVP",
)

λ_pred, _ = model(G_pred, ps, st)
compute_and_save_bandstructures(G_pred, sys_pred, model, ps, st;
    kgrid      = [4, 4, 4], 
    keep_bands = 4,
    name       = "si_pred.png",)