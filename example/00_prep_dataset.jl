# 00_prep_dataset.jl
# Purpose: build reference data/graphs once, then save to disk (may use Python)

using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

using JLD2

include("data/generate_data.jl")
include("integral/integral.jl")
include("utils/build_graph.jl")

# -----------------------------
# 0) dataset list
# -----------------------------
syslist = [bulk(:Si); [perturb(bulk(:Si)) for _ in 1:50]]
#sys_pert = perturb(bulk(:Si))
sys_pert = perturb(bulk(:Si, cubic=true))

NORB       = 2
kgrid      = [6, 7, 8]
keep_bands = 4
rcut       = 12.5u"Å"

println("== PREP: build CART ==")
bs_cart, λ_ref_list_cart, G_list_cart =
    build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands=keep_bands)

_, λ_ref_list_cart_pert, G_list_cart_pert =
    build_ref_and_graph_lists_cart([sys_pert], rcut, kgrid; keep_bands=keep_bands)

println("== PREP: build SPH ==")
bs_sph, λ_ref_list_sph, G_list_sph =
    build_ref_and_graph_lists_sph(syslist, rcut, kgrid; keep_bands=keep_bands)

_, _, G_list_sph_pert =
    build_ref_and_graph_lists_sph([sys_pert], rcut, kgrid; keep_bands=keep_bands)

out = joinpath(@__DIR__, "dataset.jld2")
@save out syslist sys_pert NORB kgrid keep_bands rcut bs_cart λ_ref_list_cart G_list_cart λ_ref_list_cart_pert G_list_cart_pert bs_sph λ_ref_list_sph G_list_sph G_list_sph_pert

println("== PREP DONE: saved to $(out) ==")
