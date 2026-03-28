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
        syslist, rcut, kgrid;
        basis_name = "def2-SVP",
    )

    println("  Saving cache to $cache_file ...")
    @save cache_file bs H_ref_list G_list syslist
    println("  Done.")
end

println("  nbas = $(bs.nbas), natoms = $(bs.natoms)")
