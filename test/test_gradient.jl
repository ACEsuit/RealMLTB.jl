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
using Test
using Zygote
using MatrixFuns
using Optimisers: destructure
using JLD2

include("data/generate_data.jl")
include("data/graphene_ham.jl")

# ============================================================
# 1. Load data
# ============================================================
println("=== Step 1: Load data ===")

rcut       = 5.0u"Å"
kgrid      = [6, 6, 1]
cache_file = "training_data_cache.jld2"

if isfile(cache_file)
    println("  Found cache, loading ...")
    @load cache_file bs H_ref_list G_list syslist
else
    println("  Generating data ...")
    sys0    = build_graphene()
    Random.seed!(0)
    syslist = [sys0; [perturb(sys0; ϵ = 0.02) for _ in 1:3]]
    bs, H_ref_list, G_list = build_ref_and_graph_lists_sph_graphene(
        syslist, rcut, kgrid; basis_name = "def2-SVP")
    @save cache_file bs H_ref_list G_list syslist
end

println("  nbas=$(bs.nbas), natoms=$(bs.natoms), nstructs=$(length(syslist))")

# ============================================================
# 2. Build small model (Dtot=4 for speed)
# ============================================================
println("\n=== Step 2: Build model ===")

NORB = 4
Random.seed!(42)
model = ace3c_B_model(bs, NORB, rcut; Dtot=4, maxl=3, ORD=1)
ps, st = Lux.setup(Random.default_rng(), model)

# Use only first structure for speed
G_test     = G_list[1]
H_ref_test = H_ref_list[1]

# λ_ref: eigenvalues of H_ref (just for RMSE test, format: Vector of vectors)
# We need k-point format — use a simple proxy: repeat H_ref eigenvalues for each k


λ_ref_test = [rand(4) for _ in 1:length(G_test.graph_data.klist)]
μ = 0.0; β = 0.1
fermi_filter = x -> 1 / (1 + exp((x - μ) / β))

println("  Model built, nparams = $(length(destructure(ps)[1]))")

# ============================================================
# Helper: finite-difference gradient check
#
# Checks that AD gradient matches FD gradient for n_check
# randomly chosen parameter directions.
# Returns true if all relative errors < tol.
# ============================================================
function fd_gradient_check(f, x0; n_check=10, ε=1e-4, tol=0.05, label="")
    println("  [$label] Computing AD gradient ...")
    gs = Zygote.gradient(f, x0)[1]
    @assert gs !== nothing "AD gradient is nothing!"

    println("  [$label] Running FD check on $n_check random directions ...")
    rng   = MersenneTwister(1)
    idxs  = randperm(rng, length(x0))[1:n_check]
    all_ok = true

    for i in idxs
        ad = gs[i]
        ep = zeros(length(x0)); ep[i] = 1.0
        fd = (f(x0 .+ ε .* ep) - f(x0 .- ε .* ep)) / (2ε)
        rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-10)
        ok  = rel < tol
        all_ok = all_ok && ok
        status = ok ? "✓" : "✗ FAIL"
        println("    param[$i]  FD=$(round(fd,sigdigits=4))  AD=$(round(ad,sigdigits=4))  rel_err=$(round(rel,sigdigits=3))  $status")
    end

    return all_ok
end

# ============================================================
# 3. Gradient test: RMSE_H
# ============================================================
println("\n=== Step 3: Gradient test — RMSE_H ===")

@testset "RMSE_H gradient correctness" begin
    x0, re = destructure(ps)

    f_H(x) = RMSE_H(G_test, H_ref_test, model, re(x), st, fermi_filter)

    # First check forward pass
    val = f_H(x0)
    println("  RMSE_H forward = $val")
    @test isfinite(val)
    @test val >= 0.0

    # FD gradient check
    ok = fd_gradient_check(f_H, x0;
                           n_check = 10,
                           ε       = 1e-4,
                           tol     = 0.05,
                           label   = "RMSE_H")
    @test ok
    println("  RMSE_H gradient check: $(ok ? "PASSED ✓" : "FAILED ✗")")
end

# ============================================================
# 4. Gradient test: RMSE (eigenvalue loss)
#    Note: needs proper λ_ref format
# ============================================================
println("\n=== Step 4: Gradient test — RMSE (eigenvalue) ===")

@testset "RMSE gradient correctness" begin
    x0, re = destructure(ps)

    keep_bands = 4
    f_λ(x) = RMSE(G_test, λ_ref_test, model, re(x), st; keep_bands=keep_bands)

    # Forward pass
    val = f_λ(x0)
    println("  RMSE_λ forward = $val")
    @test isfinite(val)

    # FD gradient check
    ok = fd_gradient_check(f_λ, x0;
                           n_check = 10,
                           ε       = 1e-4,
                           tol     = 0.05,
                           label   = "RMSE_λ")
    @test ok
    println("  RMSE_λ gradient check: $(ok ? "PASSED ✓" : "FAILED ✗")")
end

# ============================================================
# 5. Sign check: does one gradient step decrease the loss?
# ============================================================
println("\n=== Step 5: Sign check — does gradient step decrease loss? ===")

@testset "Gradient descent sign check" begin
    x0, re = destructure(ps)

    # RMSE_H
    f_H(x) = RMSE_H(G_test, H_ref_test, model, re(x), st, fermi_filter)
    v0  = f_H(x0)
    g   = Zygote.gradient(f_H, x0)[1]
    lr  = 1e-4
    v1  = f_H(x0 .- lr .* g)
    println("  RMSE_H before step: $(round(v0, sigdigits=5))")
    println("  RMSE_H after step:  $(round(v1, sigdigits=5))")
    decreased_H = v1 < v0
    println("  Loss decreased: $(decreased_H ? "YES ✓" : "NO ✗ — gradient might be wrong!")")
    @test decreased_H

    # RMSE_λ
    keep_bands = 4
    f_λ(x) = RMSE(G_test, λ_ref_test, model, re(x), st; keep_bands=keep_bands)
    v0λ = f_λ(x0)
    gλ  = Zygote.gradient(f_λ, x0)[1]
    v1λ = f_λ(x0 .- lr .* gλ)
    println("  RMSE_λ before step: $(round(v0λ, sigdigits=5))")
    println("  RMSE_λ after step:  $(round(v1λ, sigdigits=5))")
    decreased_λ = v1λ < v0λ
    println("  Loss decreased: $(decreased_λ ? "YES ✓" : "NO ✗ — gradient might be wrong!")")
    @test decreased_λ
end

println("\n=== All gradient tests complete ===")