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

# ============================================================
# 1. System setup
# ============================================================
println("=== System setup ===")

rcut = 5.0u"Å"
sys  = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :)
X_ang = [ustrip.(u"Å", r) for r in X]

# Spherical basis for ACE models
bs_sph = BasisSet(
    "def2-SVP",
    [Molecules.Atom(14, 0.0, X_ang[i]) for i in eachindex(X_ang)];
    spherical = true, lib = :acsint,
)

# Cartesian basis for Simple3Center
bs_cart = BasisSet(
    "def2-SVP",
    [Molecules.Atom(14, 0.0, X_ang[i]) for i in eachindex(X_ang)];
    spherical = false, lib = :acsint,
)

lmax, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs_sph)
println("  lmax = $lmax,  n_channels = $n_channels")
println("  nbas (sph) = $(bs_sph.nbas),  nbas (cart) = $(bs_cart.nbas)")

klist = [SVector(0.0, 0.0, 0.0),
         SVector(0.5, 0.0, 0.0)]   # Γ and X points

println("=== Building ETGraph ===")
G = RealMLTB.nlist2graph(bs_sph, nlist, sys, klist)
println("  nnodes = $(EquivariantTensors.nnodes(G))")
println("  nedges = $(EquivariantTensors.nedges(G))")

# ============================================================
# 2. Model definitions
# ============================================================
println("\n=== Building models ===")
Random.seed!(42)

NORB = 4
nψ   = (1, 1, 0, 0, 0)   # 1 s + 1 p compressed orbital per atom → NORB=4

# --- Model A: Wv from ACE, Wb plain parameter ---
model_acev = acev_3c_model(bs_sph, NORB, rcut)
ps_acev, st_acev = Lux.setup(Random.default_rng(), model_acev)
println("  acev model built")

# --- Model B: Wv from ACE, Wb from Wpsi Case-B ---
model_B = ace3c_B_model(bs_sph, NORB, rcut)
ps_B, st_B = Lux.setup(Random.default_rng(), model_B)
println("  Case-B model built")

# --- Model A: Wv from ACE, Wb from Wpsi Case-A ---
model_A = ace3c_A_model(bs_sph, nψ, rcut)
ps_A, st_A = Lux.setup(Random.default_rng(), model_A)
println("  Case-A model built")

# ============================================================
# 3. Forward pass tests
# ============================================================
println("\n=== Forward pass tests ===")

@testset "Forward pass" begin

    @testset "acev model" begin
        λlist, st_new = model_acev(G, ps_acev, st_acev)
        @test λlist isa Vector
        @test length(λlist) == length(klist)
        nk = length(klist)
        for ik in 1:nk
            λ = λlist[ik]
            @test λ isa Vector
            @test length(λ) <= NORB * bs_sph.natoms
            @test length(λ) > 0
            @test all(isfinite, λ)
            println("  acev k=$ik: λ = $(round.(λ, digits=4))")
        end
    end

    @testset "Case-B model" begin
        λlist, st_new = model_B(G, ps_B, st_B)
        @test length(λlist) == length(klist)
        for ik in eachindex(klist)
            @test all(isfinite, λlist[ik])
        end
        println("  Case-B Γ eigenvalues: $(round.(λlist[1], digits=4))")
    end

    @testset "Case-A model" begin
        λlist, st_new = model_A(G, ps_A, st_A)
        @test length(λlist) == length(klist)
        for ik in eachindex(klist)
            @test all(isfinite, λlist[ik])
        end
        println("  Case-A Γ eigenvalues: $(round.(λlist[1], digits=4))")
    end

end

# ============================================================
# 4. Gradient test via Zygote
# ============================================================
println("\n=== Gradient tests ===")

# Define a scalar loss: sum of squared eigenvalues over all k-points
function loss(model, ps, st, G)
    λlist, _ = model(G, ps, st)
    return sum(sum(λ.^2) for λ in λlist)
end

@testset "Gradients (Zygote)" begin

    @testset "acev model gradient" begin
        println("  Computing gradient for acev model...")
        val = loss(model_acev, ps_acev, st_acev, G)
        println("  Loss value = $val")
        @test isfinite(val)
        local gs = nothing
        try
            gs = Zygote.gradient(p -> loss(model_acev, p, st_acev, G), ps_acev)
            println("gradient OK")
        catch e
            println("Error type: ", typeof(e))
            #println("Error message: ", e)
            #Base.showerror(stdout, e)
            #println()
            #Base.show_backtrace(stdout, catch_backtrace())
        end
        @test gs !== nothing
        @test gs[1] !== nothing

        # Check that ACE weights got a gradient
        ace_grad = gs[1].layer_1.ace
        @test ace_grad !== nothing
        Wb_grad = gs[1].layer_1.Wb
        @test Wb_grad !== nothing
        @test all(isfinite, Wb_grad)
        println("  acev: norm(∂L/∂Wb) = $(norm(Wb_grad))")
        println("  acev gradient OK ✓")
    end

    @testset "Case-B model gradient" begin
    println("  Computing gradient for Case-B model...")
    val = loss(model_B, ps_B, st_B, G)
    @test isfinite(val)
    try
        gs = Zygote.gradient(p -> loss(model_B, p, st_B, G), ps_B)
        @test gs !== nothing
        println("  Case-B gradient OK ✓")
    catch e
        println("Case-B ERROR: ", typeof(e))
        println(e)
        rethrow()
    end
end

@testset "Case-A model gradient" begin
    println("  Computing gradient for Case-A model...")
    val = loss(model_A, ps_A, st_A, G)
    @test isfinite(val)
    try
        gs = Zygote.gradient(p -> loss(model_A, p, st_A, G), ps_A)
        @test gs !== nothing
        println("  Case-A gradient OK ✓")
    catch e
        println("Case-A ERROR: ", typeof(e))
        println(e)
        rethrow()
    end
end

end

# ============================================================
# 5. Finite-difference gradient check (acev model, scalar Wb)
# ============================================================
println("\n=== Finite-difference gradient check (acev model) ===")

@testset "Finite-difference check" begin
    function loss_Wb(Wb_flat)
        NORB_loc = size(ps_acev.layer_1.Wb, 1)
        Nb_loc   = size(ps_acev.layer_1.Wb, 2)
        Wb_mat   = reshape(Wb_flat, NORB_loc, Nb_loc)
        ps_new   = (layer_1 = merge(ps_acev.layer_1, (Wb = Wb_mat,)),
                    layer_2 = ps_acev.layer_2,
                    layer_3 = ps_acev.layer_3)
        return loss(model_acev, ps_new, st_acev, G)
    end

    Wb_flat = vec(ps_acev.layer_1.Wb)
    n_check = min(5, length(Wb_flat))
    gs_Wb = Zygote.gradient(loss_Wb, Wb_flat)[1]

    println("  Checking $n_check Wb components:")
    for i in 1:n_check
        e_i = zeros(length(Wb_flat)); e_i[i] = 1.0
        ag_grad = gs_Wb[i]
        
        # scan ε to find stable FD estimate
        println("  i=$i  AD=$ag_grad")
        for ε in [1e-3, 1e-4, 1e-5, 1e-6]
            fd = (loss_Wb(Wb_flat .+ ε .* e_i) - loss_Wb(Wb_flat .- ε .* e_i)) / (2ε)
            rel_err = abs(fd - ag_grad) / (abs(fd) + 1e-10)
            println("    ε=$ε  FD=$(round(fd,digits=4))  rel_err=$(round(rel_err,digits=4))")
        end
    end

    for i in 1:n_check
        e_i = zeros(length(Wb_flat)); e_i[i] = 1.0
        ag_grad = gs_Wb[i]
        ε = 1e-5
        fd = (loss_Wb(Wb_flat .+ ε .* e_i) - loss_Wb(Wb_flat .- ε .* e_i)) / (2ε)
        rel_err = abs(fd - ag_grad) / (abs(fd) + 1e-10)
        ok = rel_err < 0.01
        println("    i=$i  FD=$(round(fd,digits=6))  AD=$(round(ag_grad,digits=6))  rel_err=$(round(rel_err,digits=6))  $(ok ? "✓" : "✗")")
        @test rel_err < 0.01
    end
end

println("\n=== All tests complete ===")